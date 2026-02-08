"""
ADVANCED XRD ANALYSIS ENGINE
========================================================================
Scientific implementation following:
1. Klug & Alexander, X-ray Diffraction Procedures, 1974
2. Williamson & Hall, Acta Metall., 1953, 1, 22-31 (Microstrain)
3. Scherrer, Nachr. Ges. Wiss. Göttingen, 1918, 2, 98 (Crystallite size)
4. Snip, Nucl. Instrum. Methods, 1984, 223, 117 (Background subtraction)
========================================================================
"""

import numpy as np
import pandas as pd
from scipy import signal, optimize, stats
from scipy.ndimage import gaussian_filter1d
from typing import Dict, Tuple, List, Optional, Any
import warnings
import re
from scientific_integration import (
    calculate_phase_fractions,
    map_peaks_to_phases
)
# Replace the phase identification import with:
try:
    from xrd_phase_identifier_nano import identify_phases_universal
    UNIVERSAL_PHASE_ID_AVAILABLE = True
except ImportError:
    UNIVERSAL_PHASE_ID_AVAILABLE = False
    from xrd_phase_identifier import identify_phases
warnings.filterwarnings('ignore')

# ============================================================================
# PHYSICAL CONSTANTS FOR XRD
# ============================================================================
XRAY_WAVELENGTHS = {
    'Cu Kα': 1.5406,      # Å
    'Cu Kα1': 1.54056,    # Å
    'Cu Kα2': 1.54439,    # Å
    'Mo Kα': 0.71073,     # Å
    'Co Kα': 1.78897,     # Å
    'Cr Kα': 2.29100      # Å
}

SCATTERING_FACTORS = {
    'Cu': 29,
    'Mo': 42,
    'Co': 27,
    'Cr': 24
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def safe_trapz(y, x):
    """Safe trapezoidal integration that works with all numpy versions"""
    try:
        # Try numpy's trapz
        return np.trapz(y, x)
    except (AttributeError, TypeError):
        # Fallback to manual implementation
        return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))

# ============================================================================
# XRD DATA EXTRACTION
# ============================================================================
def extract_xrd_data(file, preview_only=False):
    """
    Extract XRD data from various file formats
    
    Parameters:
    -----------
    file : Uploaded file object
    preview_only : If True, returns only preview info
    
    Returns:
    --------
    two_theta, intensity, message
    """
    try:
        filename = file.name.lower()
        
        # Read file content
        content = file.read().decode('utf-8', errors='ignore')
        file.seek(0)  # Reset file pointer
        
        lines = content.split('\n')
        
        # Try different formats
        data_points = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try splitting by various delimiters
            for delimiter in ['\t', ',', ';', ' ']:
                parts = [p.strip() for p in line.split(delimiter) if p.strip()]
                if len(parts) >= 2:
                    try:
                        theta = float(parts[0])
                        intensity = float(parts[1])
                        if -5 <= theta <= 180:  # Valid 2θ range
                            data_points.append((theta, intensity))
                        break
                    except:
                        continue
        
        if not data_points:
            # Try reading with pandas
            try:
                for delimiter in ['\t', ',', ';', ' ']:
                    try:
                        df = pd.read_csv(file, delimiter=delimiter, header=None)
                        file.seek(0)
                        if df.shape[1] >= 2:
                            theta = pd.to_numeric(df.iloc[:, 0], errors='coerce')
                            intensity = pd.to_numeric(df.iloc[:, 1], errors='coerce')
                            valid = theta.notna() & intensity.notna()
                            data_points = list(zip(theta[valid], intensity[valid]))
                            break
                    except:
                        continue
            except:
                pass
        
        if not data_points:
            return None, None, "No valid data found"
        
        # Convert to arrays
        data_points = sorted(data_points, key=lambda x: x[0])
        two_theta = np.array([p[0] for p in data_points])
        intensity = np.array([p[1] for p in data_points])
        
        # Remove duplicates in two_theta
        unique_theta, unique_indices = np.unique(two_theta, return_index=True)
        if len(unique_theta) < len(two_theta):
            two_theta = unique_theta
            intensity = intensity[unique_indices]
        
        # Preview mode
        if preview_only:
            return two_theta, intensity, f"Found {len(two_theta)} data points"
        
        # Validate data
        if len(two_theta) < 50:
            return None, None, f"Insufficient data points: {len(two_theta)} (<50)"
        
        if two_theta.max() - two_theta.min() < 10:
            return None, None, f"Insufficient angular range: {two_theta.max()-two_theta.min():.1f}° (<10°)"
        
        return two_theta, intensity, "Data extracted successfully"
        
    except Exception as e:
        return None, None, f"Extraction error: {str(e)}"

# ============================================================================
# BACKGROUND SUBTRACTION (SNIP ALGORITHM)
# ============================================================================
def snip_background(intensity, iterations=100, reduction_factor=0.8):
    """
    SNIP algorithm for background subtraction
    
    References:
    Ryan et al., Nucl. Instrum. Methods B, 1988, 34, 396-402
    Morháč et al., Nucl. Instrum. Methods A, 1997, 401, 113-132
    """
    # Initial spectrum
    spectrum = np.log(np.log(intensity + 1) + 1)
    background = spectrum.copy()
    
    # SNIP iterations
    for it in range(iterations):
        for i in range(1, len(spectrum) - 1):
            min_val = min(background[i-1], background[i+1])
            if background[i] > min_val:
                background[i] = min_val + (background[i] - min_val) * reduction_factor
    
    # Convert back
    bg_subtracted = np.exp(np.exp(background) - 1) - 1
    
    return intensity - bg_subtracted, bg_subtracted

# ============================================================================
# SCIENTIFIC PEAK DETECTION (PHYSICS-BASED FWHM-VALIDATED)
# ============================================================================
def detect_peaks(two_theta, intensity, min_distance_deg=1.0, min_prominence=0.03):
    """
    SCIENTIFIC peak detection for XRD with physics-based validation
    
    Parameters:
    -----------
    two_theta : Array of 2θ values (degrees)
    intensity : Array of intensity values
    min_distance_deg : Minimum angular distance between peaks (degrees, not points)
    min_prominence : Minimum peak prominence relative to max intensity
    
    Returns:
    --------
    List of validated peak dictionaries
    """
    # 1. Mandatory background subtraction first
    intensity_corr, background = snip_background(intensity)
    
    # 2. Instrumental smoothing (Savitzky-Golay only - XRD standard)
    if len(intensity_corr) > 11:
        intensity_smooth = signal.savgol_filter(
            intensity_corr,
            window_length=11,
            polyorder=3
        )
    else:
        intensity_smooth = intensity_corr
    
    # Normalize for peak detection
    intensity_norm = (intensity_smooth - intensity_smooth.min()) / (intensity_smooth.max() - intensity_smooth.min())
    
    # 3. Convert angular distance to points for find_peaks
    angular_step = np.mean(np.diff(two_theta))
    min_distance_points = int(min_distance_deg / angular_step) if angular_step > 0 else 20
    
    # 4. Physics-based peak detection
    peaks_idx, properties = signal.find_peaks(
        intensity_norm,
        prominence=min_prominence,
        width=(3, None),  # Minimum 3 points width for FWHM
        distance=max(min_distance_points, 5),
        wlen=len(intensity_norm)//10  # Window length for prominence calculation
    )
    
    peaks = []
    
    for i, idx in enumerate(peaks_idx):
        # Validate FWHM physically
        fwhm_points = properties['widths'][i] if 'widths' in properties and i < len(properties['widths']) else 3
        fwhm_deg = fwhm_points * angular_step
        
        # Reject unphysically narrow peaks (instrument noise)
        if fwhm_deg < 0.02:  # < 0.02° is unphysical for laboratory XRD
            continue
            
        # Reject unphysically broad peaks (>5° for nanocrystalline)
        if fwhm_deg > 5.0:
            continue
            
        # Calculate FWHM using interpolation (more accurate)
        peak_intensity = intensity_smooth[idx]
        half_max = peak_intensity / 2
        
        # Find left half-max point
        left_idx = idx
        while left_idx > 0 and intensity_smooth[left_idx] > half_max:
            left_idx -= 1
        
        # Interpolate left side
        if left_idx < idx and left_idx >= 0:
            theta_left = np.interp(half_max, 
                                 [intensity_smooth[left_idx], intensity_smooth[left_idx + 1]],
                                 [two_theta[left_idx], two_theta[left_idx + 1]])
        else:
            theta_left = two_theta[idx]
        
        # Find right half-max point
        right_idx = idx
        while right_idx < len(intensity_smooth) - 1 and intensity_smooth[right_idx] > half_max:
            right_idx += 1
        
        # Interpolate right side
        if right_idx > idx and right_idx < len(intensity_smooth):
            theta_right = np.interp(half_max,
                                  [intensity_smooth[right_idx - 1], intensity_smooth[right_idx]],
                                  [two_theta[right_idx - 1], two_theta[right_idx]])
        else:
            theta_right = two_theta[idx]
        
        # Final FWHM
        fwhm_deg = theta_right - theta_left
        
        # Validate physically reasonable FWHM
        if fwhm_deg < 0.03 or fwhm_deg > 3.0:  # Physical range for nanocrystalline materials
            continue
            
        # Calculate integrated intensity (area under peak)
        peak_start = max(0, int(idx - fwhm_points * 2))
        peak_end = min(len(intensity_smooth), int(idx + fwhm_points * 2))
        
        x_segment = two_theta[peak_start:peak_end]
        y_segment = intensity_smooth[peak_start:peak_end]
        
        if len(x_segment) >= 2:
            peak_area = np.trapz(y_segment, x_segment)
        else:
            peak_area = 0
        
        # Peak asymmetry (for quality assessment)
        left_half = intensity_smooth[idx] - np.min(intensity_smooth[peak_start:idx]) if peak_start < idx else intensity_smooth[idx]
        right_half = intensity_smooth[idx] - np.min(intensity_smooth[idx:peak_end]) if idx < peak_end - 1 else intensity_smooth[idx]
        asymmetry = left_half / right_half if right_half > 0 else 1.0
        
        # Reject highly asymmetric peaks (>2.0 or <0.5)
        if asymmetry > 2.0 or asymmetry < 0.5:
            continue
            
        # Peak signal-to-noise ratio
        local_noise = np.std(background[peak_start:peak_end]) if np.any(background[peak_start:peak_end]) else 1.0
        snr = peak_intensity / local_noise if local_noise > 0 else 1.0
        
        # Minimum SNR requirement
        if snr < 2.0:
            continue
        
        peaks.append({
            'index': int(idx),
            'position': float(two_theta[idx]),
            'intensity': float(peak_intensity),
            'intensity_raw': float(intensity[idx]),  # Keep raw for reference
            'fwhm_deg': float(fwhm_deg),
            'fwhm_rad': float(np.deg2rad(fwhm_deg)),
            'area': float(peak_area),
            'asymmetry': float(asymmetry),
            'snr': float(snr),
            'theta_left': float(theta_left),
            'theta_right': float(theta_right),
            'prominence': float(properties['prominences'][i]) if 'prominences' in properties and i < len(properties['prominences']) else 0.0
        })
    
    # Sort by intensity (descending)
    peaks.sort(key=lambda x: x['intensity'], reverse=True)
    
    return peaks

# ============================================================================
# CRYSTALLOGRAPHIC CALCULATIONS
# ============================================================================
def calculate_d_spacing(theta_deg, wavelength=1.5406):
    """
    Calculate d-spacing from Bragg's law
    
    nλ = 2d sinθ
    d = nλ / (2 sinθ)
    
    Parameters:
    -----------
    theta_deg : Bragg angle in degrees
    wavelength : X-ray wavelength in Å
    
    Returns:
    --------
    d-spacing in Å
    """
    theta_rad = np.deg2rad(theta_deg / 2)  # Convert to θ (not 2θ)
    return wavelength / (2 * np.sin(theta_rad))


# ============================================================================
# CRYSTALLITE SIZE ANALYSIS WITH INSTRUMENTAL BROADENING CORRECTION
# ============================================================================
def scherrer_crystallite_size(fwhm_rad, theta_rad, wavelength=1.5406, K=0.9, 
                            instrument_fwhm_rad=0.0):
    """
    Scherrer equation for crystallite size with instrumental broadening correction
    
    D = Kλ / (β cosθ) where β = √(β_measured² - β_instrument²)
    
    Parameters:
    -----------
    fwhm_rad : Measured FWHM in radians (β_measured)
    theta_rad : Bragg angle in radians (θ)
    wavelength : X-ray wavelength in Å
    K : Shape factor (typically 0.9)
    instrument_fwhm_rad : Instrumental broadening in radians
    
    Returns:
    --------
    Crystallite size in nm
    """
    # Correct for instrumental broadening
    if instrument_fwhm_rad > 0:
        sample_fwhm_rad = np.sqrt(fwhm_rad**2 - instrument_fwhm_rad**2)
        if sample_fwhm_rad <= 0:
            return 0.0
    else:
        sample_fwhm_rad = fwhm_rad
    
    size_angstrom = K * wavelength / (sample_fwhm_rad * np.cos(theta_rad))
    return size_angstrom / 10  # Convert Å to nm

def williamson_hall_analysis(peaks, wavelength=1.5406, instrument_fwhm_rad=0.0):
    """
    SCIENTIFIC Williamson–Hall analysis for nanocrystalline materials
    
    β cosθ = Kλ/D + 4ε sinθ
    
    Only uses isolated peaks with valid FWHM and proper shape
    """
    if not peaks or len(peaks) < 4:
        return None
    
    # Select only clean, isolated peaks for W-H analysis
    valid_peaks = []
    for p in peaks:
        # Physical validation criteria
        fwhm_deg = p.get("fwhm_deg", 0)
        asymmetry = p.get("asymmetry", 1.0)
        snr = p.get("snr", 0)
        
        # Reject peaks that are too narrow or too broad
        if fwhm_deg < 0.05 or fwhm_deg > 2.0:
            continue
            
        # Reject highly asymmetric peaks
        if asymmetry < 0.7 or asymmetry > 1.3:
            continue
            
        # Minimum SNR requirement
        if snr < 3.0:
            continue
            
        valid_peaks.append(p)
    
    # Need at least 4 valid peaks for W-H analysis
    if len(valid_peaks) < 4:
        return None
    
    x_vals, y_vals = [], []
    
    for p in valid_peaks:
        theta_rad = np.deg2rad(p["position"] / 2)
        beta_measured = p["fwhm_rad"]
        
        # Correct for instrumental broadening
        if instrument_fwhm_rad > 0:
            beta_sample = np.sqrt(beta_measured**2 - instrument_fwhm_rad**2)
            if beta_sample <= 0:
                continue
        else:
            beta_sample = beta_measured
        
        # Williamson-Hall plot coordinates
        x_vals.append(4 * np.sin(theta_rad))        # 4 sinθ
        y_vals.append(beta_sample * np.cos(theta_rad))  # β cosθ
    
    # Need at least 4 points for meaningful regression
    if len(x_vals) < 4:
        return None
    
    # Linear regression with error checking
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
        
        # Physical validation of regression results
        if intercept <= 0 or np.isnan(intercept):
            return None
            
        if r_value**2 < 0.7:  # R² < 0.7 indicates poor correlation
            return None
            
        # Constants
        K = 0.9
        size_nm = (K * wavelength) / (intercept * 10)   # Å → nm
        microstrain = slope / 4
        
        # Physical validation of results
        if size_nm < 0.5 or size_nm > 500:  # Unphysical size range
            return None
            
        if microstrain < 0 or microstrain > 0.05:  # Unphysical strain range
            return None
        
        return {
            "crystallite_size": float(size_nm),
            "microstrain": float(microstrain),
            "r_squared": float(r_value ** 2),
            "slope": float(slope),
            "intercept": float(intercept),
            "n_peaks_used": len(x_vals),
            "x_data": [float(x) for x in x_vals],
            "y_data": [float(y) for y in y_vals],
            "valid_peaks": len(valid_peaks)
        }
        
    except Exception as e:
        return None


# ============================================================================
# SCIENTIFIC CRYSTALLINITY INDEX - AREA-BASED METHOD
# ============================================================================
def calculate_crystallinity_index(two_theta, intensity, peaks):
    """
    SCIENTIFIC crystallinity index based on crystalline area / total area
    
    Reference: Klug & Alexander, X-ray Diffraction Procedures, 1974
    
    CI = ∫I_crystalline / ∫I_total
    
    This is the correct, physically meaningful definition.
    """
    if not peaks:
        return 0.0
    
    # First, separate background using SNIP
    intensity_corr, background = snip_background(intensity)
    
    # Total area under the corrected pattern
    total_area = np.trapz(intensity_corr, two_theta)
    
    if total_area <= 0:
        return 0.0
    
    # Crystalline area: integrate only regions above background
    # where intensity is significantly above background (SNR > 2)
    signal_mask = intensity_corr > (background + 2 * np.std(background))
    
    if np.any(signal_mask):
        crystalline_area = np.trapz(intensity_corr[signal_mask], two_theta[signal_mask])
    else:
        # Fallback: integrate peak regions only
        crystalline_area = 0.0
        for p in peaks:
            # Integrate around each peak (3×FWHM)
            fwhm = p.get('fwhm_deg', 0.5)
            peak_start = p['position'] - 1.5 * fwhm
            peak_end = p['position'] + 1.5 * fwhm
            
            mask = (two_theta >= peak_start) & (two_theta <= peak_end)
            if np.any(mask):
                crystalline_area += np.trapz(intensity_corr[mask], two_theta[mask])
    
    # Ensure CI is between 0 and 1
    ci = crystalline_area / total_area
    return min(max(ci, 0.0), 1.0)


# ============================================================================
# LATTICE PARAMETER REFINEMENT
# ============================================================================
def refine_lattice_parameters(peaks, wavelength=1.5406, 
                            crystal_system='cubic', initial_guess=4.0):
    """
    Refine lattice parameters from peak positions
    
    Parameters:
    -----------
    peaks : List of peak dictionaries
    wavelength : X-ray wavelength in Å
    crystal_system : Crystal system
    initial_guess : Initial lattice parameter guess
    
    Returns:
    --------
    Refined lattice parameters
    """
    if len(peaks) < 3:
        return None
    
    # Extract d-spacings
    d_spacings = [calculate_d_spacing(p['position'], wavelength) for p in peaks]
    
    # Simplified lattice parameter calculation
    if crystal_system == 'cubic' and len(d_spacings) >= 3:
        # Use the peak with highest intensity to estimate lattice parameter
        main_peak_d = d_spacings[np.argmax([p['intensity'] for p in peaks])]
        a_estimated = main_peak_d * np.sqrt(3)  # For (111) reflection
        
        # Refine using multiple peaks
        try:
            from scipy.optimize import minimize_scalar
            
            # Define simplified optimization
            def cubic_residual(a_val):
                res = 0
                for d_exp in d_spacings[:5]:  # Use first 5 peaks
                    # Try common cubic reflections
                    for hkl in [(1,0,0), (1,1,0), (1,1,1), (2,0,0), (2,2,0)]:
                        d_calc = a_val / np.sqrt(sum(h**2 for h in hkl))
                        error = abs(d_exp - d_calc) / d_exp
                        if error < 0.05:  # 5% tolerance
                            res += (d_exp - d_calc)**2
                            break
                return res
            
            result = minimize_scalar(cubic_residual, 
                                    bounds=(a_estimated*0.8, a_estimated*1.2),
                                    method='bounded')
            
            if result.success:
                return {'a': float(result.x), 'error': float(result.fun)}
            
        except:
            pass
        
        return {'a': float(a_estimated), 'error': 0.0}
    
    return None

def allowed_hkl(hkl, crystal_system):
    h, k, l = hkl

    if crystal_system in ["cubic_fcc", "fcc"]:
        return (h + k + l) % 2 == 0

    if crystal_system in ["cubic_bcc", "bcc"]:
        return (h % 2 == k % 2 == l % 2)

    return True

# ============================================================================
# ADVANCED XRD ANALYZER CLASS
# ============================================================================
class AdvancedXRDAnalyzer:
    """
    Advanced XRD analysis with scientific rigor
    """
    
    def __init__(self, wavelength=1.5406, background_subtraction=True,
                 smoothing='Savitzky-Golay', instrument_fwhm_deg=0.0):
        """
        Initialize XRD analyzer
        
        Parameters:
        -----------
        wavelength : X-ray wavelength in Å
        background_subtraction : Whether to subtract background
        smoothing : Smoothing method ('Savitzky-Golay', 'None')
        instrument_fwhm_deg : Instrumental broadening in degrees (FWHM)
        """
        self.wavelength = wavelength
        self.background_subtraction = background_subtraction
        self.smoothing = smoothing
        self.scherrer_constant = 0.9
        self.instrument_fwhm_deg = instrument_fwhm_deg
        self.instrument_fwhm_rad = np.deg2rad(instrument_fwhm_deg)
    
    def validate_crystallographic_results(self, xrd_results: Dict) -> Dict:
        """
        Validate XRD results against crystallographic standards.
        Returns validation metrics with confidence intervals.
        """
        validation = {
            'valid': True,
            'warnings': [],
            'confidence_score': 1.0,
            'scientific_checks': []
        }
        
        # Check crystallite size is physically reasonable
        size = xrd_results.get('crystallite_size', {}).get('scherrer', 0)
        if size > 1000:  # > 1 micron - unlikely from XRD
            validation['warnings'].append(f"Crystallite size unusually large: {size:.1f} nm")
            validation['confidence_score'] *= 0.7
        elif size < 0.5:  # < 0.5 nm - unphysical
            validation['warnings'].append(f"Crystallite size unusually small: {size:.1f} nm")
            validation['confidence_score'] *= 0.6
        
        # Check crystallinity index is in valid range
        ci = xrd_results.get('crystallinity_index', 0)
        if ci < 0 or ci > 1:
            validation['warnings'].append(f"Crystallinity index out of range: {ci:.2f}")
            validation['confidence_score'] *= 0.5
        elif ci == 1.00:  # Physically impossible
            validation['warnings'].append(f"Crystallinity index = 1.00 is physically impossible")
            validation['confidence_score'] *= 0.3
        
        # Check peak count is reasonable
        n_peaks = len(xrd_results.get('peaks', []))
        if n_peaks < 3:
            validation['warnings'].append(f"Few peaks detected: {n_peaks}")
            validation['confidence_score'] *= 0.8
        elif n_peaks > 100:  # Too many peaks likely noise
            validation['warnings'].append(f"Excessive peaks detected ({n_peaks}), likely including noise")
            validation['confidence_score'] *= 0.7
        
        # Check for physically impossible FWHM
        peaks = xrd_results.get('peaks', [])
        for peak in peaks:
            fwhm = peak.get('fwhm_deg', 0)
            if fwhm < 0.01 or fwhm > 5.0:  # Unphysical values
                validation['warnings'].append(f"Unphysical FWHM: {fwhm:.4f}° at 2θ={peak.get('position', 0):.2f}°")
                validation['confidence_score'] *= 0.9
        
        # Calculate scientific checks
        validation['scientific_checks'] = [
            {
                'check': 'Peak count adequate',
                'passed': 3 <= n_peaks <= 50,
                'value': n_peaks,
                'expected': '3-50 peaks'
            },
            {
                'check': 'Crystallite size physical',
                'passed': 0.5 < size < 500,
                'value': f"{size:.1f} nm",
                'expected': '0.5-500 nm'
            },
            {
                'check': 'Crystallinity index valid',
                'passed': 0 < ci < 1,
                'value': f"{ci:.3f}",
                'expected': '0.0-1.0 (not 1.00)'
            },
            {
                'check': 'Peak FWHM physical',
                'passed': all(0.03 < p.get('fwhm_deg', 0) < 3.0 for p in peaks),
                'value': f"{np.mean([p.get('fwhm_deg', 0) for p in peaks]):.3f}° avg",
                'expected': '0.03-3.0°'
            }
        ]
        
        # Overall validity
        validation['valid'] = validation['confidence_score'] > 0.5
        
        return validation
    
    def preprocess_pattern(self, two_theta, intensity):
        """
        Preprocess XRD pattern
        
        Parameters:
        -----------
        two_theta : Array of 2θ values
        intensity : Array of intensity values
        
        Returns:
        --------
        Processed two_theta and intensity
        """
        # Ensure two_theta is increasing
        sort_idx = np.argsort(two_theta)
        two_theta = two_theta[sort_idx]
        intensity = intensity[sort_idx]
        
        # Smoothing (only Savitzky-Golay for XRD)
        if self.smoothing == 'Savitzky-Golay':
            if len(intensity) > 11:
                intensity = signal.savgol_filter(intensity, window_length=11, polyorder=3)
        
        # Background subtraction
        if self.background_subtraction:
            intensity, background = snip_background(intensity)
        
        return two_theta, intensity
    
    def analyze_peaks(self, two_theta, intensity, min_prominence=0.03):
        """
        Complete peak analysis with scientific validation
        
        Parameters:
        -----------
        two_theta : Array of 2θ values
        intensity : Array of intensity values
        min_prominence : Minimum peak prominence for detection
        
        Returns:
        --------
        List of analyzed peaks
        """
        # Use scientific peak detection (FWHM-validated)
        peaks = detect_peaks(two_theta, intensity, min_prominence=min_prominence)
        
        # Analyze each peak
        analyzed_peaks = []
        for peak in peaks:
            # Calculate d-spacing
            d_spacing = calculate_d_spacing(peak['position'], self.wavelength)
            
            # Calculate crystallite size (Scherrer with instrumental correction)
            theta_rad = np.deg2rad(peak['position'] / 2)
            crystallite_size = scherrer_crystallite_size(
                peak['fwhm_rad'], theta_rad, self.wavelength, 
                self.scherrer_constant, self.instrument_fwhm_rad
            )
            
            analyzed_peaks.append({
                **peak,
                'd_spacing': float(d_spacing),
                'crystallite_size': float(crystallite_size),
                'theta_bragg': float(peak['position'] / 2)
            })
        
        return analyzed_peaks
    
    def calculate_crystallite_statistics(self, peaks):
        """
        Calculate statistics from multiple peaks
        
        Parameters:
        -----------
        peaks : List of peak dictionaries
        
        Returns:
        --------
        Crystallite size statistics
        """
        if not peaks:
            return {
                'mean_size': 0.0,
                'std_size': 0.0,
                'size_distribution': 'Unknown',
                'n_peaks': 0
            }
        
        sizes = [p['crystallite_size'] for p in peaks if p['crystallite_size'] > 0]
        
        if not sizes:
            return {
                'mean_size': 0.0,
                'std_size': 0.0,
                'size_distribution': 'Unknown',
                'n_peaks': 0
            }
        
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        
        # Classify size distribution scientifically
        cv = std_size / mean_size if mean_size > 0 else 0
        
        if cv < 0.1:
            distribution = 'Narrow (monodisperse)'
        elif cv < 0.25:
            distribution = 'Moderate'
        elif cv < 0.5:
            distribution = 'Broad (polydisperse)'
        else:
            distribution = 'Very broad'
        
        return {
            'mean_size': float(mean_size),
            'std_size': float(std_size),
            'distribution': distribution,
            'cv': float(cv),
            'n_peaks': len(sizes)
        }
    
    def check_ordered_mesopores(self, two_theta, intensity, peaks):
        """
        Check for ordered mesoporous structure (low-angle peaks)
        
        Parameters:
        -----------
        two_theta : Array of 2θ values
        intensity : Array of intensity values
        peaks : List of detected peaks
        
        Returns:
        --------
        Ordered mesopore analysis
        """
        # Check for peaks in low-angle region (0.5-10° 2θ)
        low_angle_peaks = [p for p in peaks if 0.5 <= p['position'] <= 10]
        
        if len(low_angle_peaks) >= 2:
            # Check for regular spacing (characteristic of ordered mesopores)
            positions = [p['position'] for p in low_angle_peaks]
            ratios = [positions[i]/positions[0] for i in range(1, len(positions))]
            
            # Typical ratios for ordered mesopores: √1:√3:√4:√7:√9...
            typical_ratios = [1.0, 1.732, 2.0, 2.646, 3.0]  # √1, √3, √4, √7, √9
            
            match_score = 0
            for r in ratios:
                for tr in typical_ratios:
                    if abs(r - tr) / tr < 0.1:  # 10% tolerance
                        match_score += 1
                        break
            
            ordered = match_score >= 2  # At least 2 peaks match typical pattern
            
            # Calculate d-spacing for main peak
            if low_angle_peaks:
                main_peak = max(low_angle_peaks, key=lambda x: x['intensity'])
                d_spacing = calculate_d_spacing(main_peak['position'], self.wavelength)
                
                # Estimate pore size (simplified - for 2D hexagonal)
                pore_size = d_spacing * 1.05  # Rough approximation
                
                return {
                    'ordered': True,
                    'n_peaks': len(low_angle_peaks),
                    'match_score': match_score,
                    'd_spacing': float(d_spacing),
                    'pore_size_estimate': float(pore_size),
                    'structure': '2D Hexagonal (p6mm)' if match_score >= 2 else 'Possibly Ordered'
                }
        
        return {
            'ordered': False,
            'n_peaks': len(low_angle_peaks),
            'match_score': 0,
            'd_spacing': 0.0,
            'pore_size_estimate': 0.0,
            'structure': 'Disordered'
        }
    
    def complete_analysis(self, two_theta, intensity, elements=None):
        """
        FULL XRD ANALYSIS — SCIENTIFICALLY CORRECTED VERSION
        
        Returns journal-grade analysis with proper error handling
        """
        
        # -----------------------------
        # SCIENTIFICALLY VALID DEFAULTS
        # -----------------------------
        xrd_results = {
            "phases": [],
            "phase_fractions": [],
            "peaks": [],
            "top_peaks": [],
            "crystallinity_index": 0.0,
            "crystallinity_description": "Unknown",
            "crystallite_size": {
                "scherrer": 0.0,
                "williamson_hall": 0.0,
                "distribution": "N/A",
                "confidence": "low"
            },
            "microstrain": 0.0,
            "dislocation_density": 0.0,
            "crystal_system": "Unknown",
            "space_group": "Unknown",
            "lattice_parameters": {},
            "ordered_mesopores": False,
            "wavelength": self.wavelength,
            "instrument_fwhm_deg": self.instrument_fwhm_deg,
            "analysis_notes": []
        }
        
        wh = None
        
        try:
            # -----------------------------
            # PREPROCESS WITH BACKGROUND SUBTRACTION
            # -----------------------------
            two_theta_p, intensity_p = self.preprocess_pattern(two_theta, intensity)
    
            # -----------------------------
            # SCIENTIFIC PEAK DETECTION
            # -----------------------------
            peaks = self.analyze_peaks(two_theta_p, intensity_p)
            
            # Keep only validated peaks (remove noise)
            validated_peaks = [p for p in peaks if p['snr'] > 2.0 and 0.03 < p['fwhm_deg'] < 3.0]
            validated_peaks.sort(key=lambda x: x["intensity"], reverse=True)
    
            xrd_results["peaks"] = validated_peaks
            xrd_results["top_peaks"] = validated_peaks[:10]  # Only top 10 peaks
            xrd_results["n_peaks_total"] = len(validated_peaks)
            
            # -----------------------------
            # SCIENTIFIC CRYSTALLINITY INDEX
            # -----------------------------
            ci = calculate_crystallinity_index(two_theta_p, intensity_p, validated_peaks)
            xrd_results["crystallinity_index"] = ci
            
            # Scientifically accurate description
            if ci < 0.3:
                xrd_results["crystallinity_description"] = "Mostly amorphous"
            elif ci < 0.6:
                xrd_results["crystallinity_description"] = "Partially crystalline"
            elif ci < 0.8:
                xrd_results["crystallinity_description"] = "Crystalline"
            elif ci < 0.95:
                xrd_results["crystallinity_description"] = "Highly crystalline"
            else:
                xrd_results["crystallinity_description"] = "Single crystal-like"
            
            # Warn if CI is suspiciously 1.00
            if ci == 1.00:
                xrd_results["analysis_notes"].append(
                    "Warning: Crystallinity index = 1.00 suggests background subtraction issue"
                )
    
            # -----------------------------
            # CRYSTALLITE SIZE ANALYSIS
            # -----------------------------
            if validated_peaks:
                size_stats = self.calculate_crystallite_statistics(validated_peaks)
                xrd_results["crystallite_size"]["scherrer"] = size_stats["mean_size"]
                xrd_results["crystallite_size"]["distribution"] = size_stats["distribution"]
                xrd_results["crystallite_size"]["cv"] = size_stats["cv"]
                
                # Confidence level based on number of peaks
                if size_stats["n_peaks"] >= 5:
                    xrd_results["crystallite_size"]["confidence"] = "high"
                elif size_stats["n_peaks"] >= 3:
                    xrd_results["crystallite_size"]["confidence"] = "medium"
                else:
                    xrd_results["crystallite_size"]["confidence"] = "low"
    
                # Williamson-Hall analysis (only if enough clean peaks and size > 8 nm)
                if len(validated_peaks) >= 4 and size_stats["mean_size"] > 8.0:
                    wh = williamson_hall_analysis(
                        validated_peaks, 
                        self.wavelength, 
                        self.instrument_fwhm_rad
                    )
                    
                    if wh:
                        xrd_results["williamson_hall"] = wh
                        xrd_results["crystallite_size"]["williamson_hall"] = wh["crystallite_size"]
                        xrd_results["microstrain"] = wh["microstrain"]
                        
                        # Dislocation density (simplified)
                        if wh["crystallite_size"] > 0:
                            xrd_results["dislocation_density"] = (
                                15 * wh["microstrain"] / (wh["crystallite_size"] * 1e-9)
                            )
                        else:
                            xrd_results["dislocation_density"] = 0.0
                else:
                    xrd_results["analysis_notes"].append(
                        "Williamson-Hall analysis requires ≥4 clean peaks and crystallite size > 8 nm"
                    )
    
            # -----------------------------
            # PHASE IDENTIFICATION
            # -----------------------------
            if elements and validated_peaks:
                try:
                    if UNIVERSAL_PHASE_ID_AVAILABLE:
                        # Use universal nanomaterial identification
                        phases = identify_phases_universal(
                            two_theta_p,
                            intensity_p,
                            wavelength=self.wavelength,
                            elements=elements
                        )
                    else:
                        # Fallback to original
                        phases = identify_phases(
                            two_theta_p,
                            intensity_p,
                            wavelength=self.wavelength,
                            elements=elements
                        )
                    
                    xrd_results["phases"] = phases
                    
                    if phases:
                        best = phases[0]
                        xrd_results["crystal_system"] = best["crystal_system"]
                        xrd_results["space_group"] = best["space_group"]
                        xrd_results["lattice_parameters"] = best["lattice"]
                        xrd_results["material_family"] = best.get("material_family", "unknown")
                        
                        # Map peaks to phases
                        try:
                            xrd_results["peaks"] = map_peaks_to_phases(validated_peaks, phases)
                        except:
                            pass
                        
                        # Calculate phase fractions
                        try:
                            xrd_results["phase_fractions"] = calculate_phase_fractions(
                                xrd_results["peaks"], phases
                            )
                        except:
                            pass
                        
                except Exception as phase_error:
                    # Don't fail entire analysis
                    xrd_results["analysis_notes"].append(f"Phase identification issue: {str(phase_error)[:100]}")
            
            # -----------------------------
            # MATERIAL CHARACTERIZATION SUMMARY
            # -----------------------------
            if validated_peaks:
                avg_fwhm = np.mean([p['fwhm_deg'] for p in validated_peaks[:5]])
                avg_size = xrd_results["crystallite_size"]["scherrer"]
                
                if avg_size < 5:
                    xrd_results["material_type"] = "Ultrafine nanocrystalline"
                elif avg_size < 20:
                    xrd_results["material_type"] = "Nanocrystalline"
                elif avg_size < 100:
                    xrd_results["material_type"] = "Sub-micron crystalline"
                else:
                    xrd_results["material_type"] = "Micron-scale crystalline"
                
                # Scientifically accurate statement about crystallinity
                if avg_fwhm > 1.0 and avg_size < 10:
                    xrd_results["crystallinity_statement"] = (
                        "The material exhibits nanocrystalline character with broadened diffraction maxima, "
                        f"indicating coherent scattering domains of ~{avg_size:.1f} nm."
                    )
                else:
                    xrd_results["crystallinity_statement"] = (
                        f"The diffraction pattern shows {xrd_results['crystallinity_description'].lower()} "
                        f"features with an estimated crystallite size of {avg_size:.1f} nm."
                    )
            
            # -----------------------------
            # ORDERED MESOPORES CHECK
            # -----------------------------
            xrd_results["ordered_mesopores"] = self.check_ordered_mesopores(
                two_theta_p, intensity_p, validated_peaks
            )
            
            # -----------------------------
            # VALIDATE RESULTS
            # -----------------------------
            validation = self.validate_crystallographic_results(xrd_results)
            xrd_results["validation"] = validation
            
            # Return successful structure
            return {
                "valid": True,
                "xrd_results": xrd_results,
                "xrd_raw": {
                    "two_theta": two_theta_p.tolist(),
                    "intensity": intensity_p.tolist()
                }
            }
    
        except Exception as e:
            # Return error structure
            return {
                "valid": False,
                "error": str(e),
                "xrd_results": xrd_results
            }
