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
import peakutils
from typing import Dict, Tuple, List, Optional, Any
import warnings
import re
from scientific_integration import (
    calculate_phase_fractions,
    map_peaks_to_phases
)
from xrd_peak_physics import InstrumentProfile, PhysicalPeakValidator

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
# PEAK DETECTION AND ANALYSIS
# ============================================================================
def detect_peaks(two_theta, intensity, min_distance=10, threshold=0.1):
    """
    Liberal candidate peak detection for XRD patterns.

    This function ONLY detects candidate peak indices.
    Physical validation is done later.

    Parameters
    ----------
    two_theta : np.ndarray
        2θ values (degrees)
    intensity : np.ndarray
        Intensity values
    min_distance : int
        Minimum distance between peaks (data points)
    threshold : float
        Relative intensity threshold (0–1)

    Returns
    -------
    list
        List of peak indices (can be empty, never None)
    """

    import numpy as np
    import peakutils

    # -------------------------------
    # Basic safety checks
    # -------------------------------
    if intensity is None or two_theta is None:
        return []

    if len(intensity) < 10 or len(two_theta) < 10:
        return []

    # -------------------------------
    # Normalize intensity safely
    # -------------------------------
    imin = np.min(intensity)
    imax = np.max(intensity)

    if imax - imin <= 0:
        return []

    intensity_norm = (intensity - imin) / (imax - imin)

    # -------------------------------
    # Peak detection (LIBERAL)
    # -------------------------------
    try:
        peak_indices = peakutils.indexes(
            intensity_norm,
            thres=threshold,
            min_dist=min_distance
        )
    except Exception:
        return []

    # -------------------------------
    # ALWAYS return a list
    # -------------------------------
    return list(peak_indices)


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
# CRYSTALLITE SIZE ANALYSIS
# ============================================================================
def scherrer_crystallite_size(fwhm_rad, theta_rad, wavelength=1.5406, K=0.9):
    """
    Scherrer equation for crystallite size
    
    D = Kλ / (β cosθ)
    
    Parameters:
    -----------
    fwhm_rad : FWHM in radians (β)
    theta_rad : Bragg angle in radians (θ)
    wavelength : X-ray wavelength in Å
    K : Shape factor (typically 0.9)
    
    Returns:
    --------
    Crystallite size in nm
    """
    size_angstrom = K * wavelength / (fwhm_rad * np.cos(theta_rad))
    return size_angstrom / 10  # Convert Å to nm

def williamson_hall_analysis(validated_peaks, wavelength=1.5406):
    """
    Williamson–Hall analysis for nanocrystalline materials

    β cosθ = Kλ/D + 4ε sinθ
    """

    import numpy as np
    from scipy import stats
    import streamlit as st

    st.write("🧪 W-H DEBUG → Peaks received:", len(validated_peaks))

    if not validated_peaks or len(validated_peaks) < 3:
        st.warning("🧪 W-H DEBUG → Not enough raw peaks")
        return None

    # ✅ MINIMAL + PHYSICALLY CORRECT FILTER
    valid_peaks = [
        p for p in validated_peaks
        if p.get("fwhm_rad", 0) > 0 and p.get("position", 0) > 0
    ]

    st.write("🧪 W-H DEBUG → Valid peaks after filter:", len(valid_peaks))

    if len(valid_peaks) < 3:
        st.warning("🧪 W-H DEBUG → Insufficient valid peaks for W–H")
        return None

    x_vals, y_vals = [], []

    for p in valid_peaks:
        theta_rad = np.deg2rad(p["position"] / 2)
        beta = p["fwhm_rad"]

        x_vals.append(4 * np.sin(theta_rad))       # 4 sinθ
        y_vals.append(beta * np.cos(theta_rad))    # β cosθ

    # Linear regression
    slope, intercept, r_value, _, _ = stats.linregress(x_vals, y_vals)

    if intercept <= 0:
        st.warning("🧪 W-H DEBUG → Non-physical intercept")
        return None

    # Constants
    K = 0.9
    size_nm = (K * wavelength) / (intercept * 10)   # Å → nm
    microstrain = slope / 4

    return {
        "crystallite_size": float(size_nm),
        "microstrain": float(microstrain),
        "r_squared": float(r_value ** 2),
        "slope": float(slope),
        "intercept": float(intercept),
        "x_data": [float(x) for x in x_vals],
        "y_data": [float(y) for y in y_vals],
    }


# ============================================================================
# CRYSTALLINITY INDEX - FIXED VERSION
# ============================================================================
def calculate_crystallinity_index(two_theta, intensity, validated_peaks):
    """
    Crystallinity index based on crystalline peak area / total area
    (Klug & Alexander, 1974)
    """
    if not validated_peaks:
        return 0.0

    total_area = safe_trapz(intensity, two_theta)
    if total_area <= 0:
        return 0.0

    crystalline_area = 0.0

    for p in validated_peaks:
        left = p["theta_left"]
        right = p["theta_right"]

        mask = (two_theta >= left) & (two_theta <= right)
        if np.any(mask):
            crystalline_area += safe_trapz(intensity[mask], two_theta[mask])

    return min(crystalline_area / total_area, 1.0)


# ============================================================================
# LATTICE PARAMETER REFINEMENT
# ============================================================================
def refine_lattice_parameters(validated_peaks, wavelength=1.5406, 
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
    if len(validated_peaks) < 3:
        return None
    
    # Extract d-spacings
    d_spacings = [calculate_d_spacing(p['position'], wavelength) for p in validated_peaks]
    
    def residual(params, d_experimental, indices):
        """Calculate residual between experimental and calculated d-spacings"""
        if crystal_system == 'cubic':
            a = params[0]
            residual = 0
            for d_exp, idx in zip(d_experimental, indices):
                # For cubic: d = a/√(h²+k²+l²)
                # We need to know (hkl) for each peak - this is simplified
                # In reality, you need to index the pattern first
                d_calc = a / np.sqrt(idx[0]**2 + idx[1]**2 + idx[2]**2)
                residual += (d_exp - d_calc)**2
            return residual
        
        # For now, return a simple estimate
        return 0
    
    # Simplified lattice parameter calculation
    if crystal_system == 'cubic' and len(d_spacings) >= 3:
        # Use the peak with highest intensity to estimate lattice parameter
        # For cubic, d = a/√(h²+k²+l²). Assume (111) for first strong peak
        main_peak_d = d_spacings[np.argmax([p['intensity'] for p in validated_peaks])]
        a_estimated = main_peak_d * np.sqrt(3)  # For (111) reflection
        
        # Refine using multiple peaks
        try:
            from scipy.optimize import least_squares
            
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
            
            result = optimize.minimize_scalar(cubic_residual, 
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
                 smoothing='Savitzky-Golay'):
        """
        Initialize XRD analyzer
        
        Parameters:
        -----------
        wavelength : X-ray wavelength in Å
        background_subtraction : Whether to subtract background
        smoothing : Smoothing method ('Savitzky-Golay', 'Moving Average', 'None')
        """
        self.wavelength = wavelength
        self.background_subtraction = background_subtraction
        self.smoothing = smoothing
        self.scherrer_constant = 0.9
    # xrd_analyzer.py - Add to AdvancedXRDAnalyzer class

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
        
        # Check crystallinity index is in valid range
        ci = xrd_results.get('crystallinity_index', 0)
        if ci < 0 or ci > 1:
            validation['warnings'].append(f"Crystallinity index out of range: {ci:.2f}")
            validation['confidence_score'] *= 0.5
        
        # Check peak count is reasonable
        n_peaks = len(xrd_results.get('validated_peaks', []))
        if n_peaks < 3:
            validation['warnings'].append(f"Few peaks detected: {n_peaks}")
            validation['confidence_score'] *= 0.8
        
        # Check for physically impossible FWHM
        validated_peaks = xrd_results.get('validated_peaks', [])
        for peak in validated_peaks:
            fwhm = peak.get('fwhm_deg', 0)
            if fwhm < 0.01 or fwhm > 5.0:  # Unphysical values
                validation['warnings'].append(f"Unphysical FWHM: {fwhm:.4f}° at 2θ={peak.get('position', 0):.2f}°")
                validation['confidence_score'] *= 0.9
        
        # Calculate scientific checks
        validation['scientific_checks'] = [
            {
                'check': 'Peak count adequate',
                'passed': n_peaks >= 3,
                'value': n_peaks
            },
            {
                'check': 'Crystallite size physical',
                'passed': 0.5 < size < 1000,
                'value': f"{size:.1f} nm"
            },
            {
                'check': 'Crystallinity index valid',
                'passed': 0 <= ci <= 1,
                'value': f"{ci:.3f}"
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
        
        # Smoothing
        if self.smoothing == 'Savitzky-Golay':
            if len(intensity) > 11:
                intensity = signal.savgol_filter(intensity, window_length=11, polyorder=3)
        elif self.smoothing == 'Moving Average':
            window = max(3, len(intensity) // 100)
            if window % 2 == 0:
                window += 1
            intensity = np.convolve(intensity, np.ones(window)/window, mode='same')
        
        # Background subtraction
        if self.background_subtraction:
            intensity, background = snip_background(intensity)
        
        return two_theta, intensity
    
    def analyze_peaks(self, two_theta, intensity, threshold=0.1):
        """
        Complete peak analysis
        
        Parameters:
        -----------
        two_theta : Array of 2θ values
        intensity : Array of intensity values
        threshold : Peak detection threshold
        
        Returns:
        --------
        List of analyzed peaks
        """
        # Detect peaks
        validated_peaks = detect_peaks(two_theta, intensity, threshold=threshold)
        # -----------------------------------
        # Physical peak validation (NEW)
        # -----------------------------------

        # Analyze each peak
        analyzed_peaks = []
        for peak in validated_peaks:
            # Calculate d-spacing
            d_spacing = calculate_d_spacing(peak['position'], self.wavelength)
            
            # Calculate crystallite size (Scherrer)
            theta_rad = np.deg2rad(peak['position'] / 2)
            crystallite_size = scherrer_crystallite_size(
                peak['fwhm_rad'], theta_rad, self.wavelength, self.scherrer_constant
            )
            
            analyzed_peaks.append({
                **peak,
                'd_spacing': float(d_spacing),
                'crystallite_size': float(crystallite_size),
                'theta_bragg': float(peak['position'] / 2)
            })
        
        return analyzed_peaks
    
    def calculate_crystallite_statistics(self, validated_peaks):
        """
        Calculate statistics from multiple peaks
        
        Parameters:
        -----------
        peaks : List of peak dictionaries
        
        Returns:
        --------
        Crystallite size statistics
        """
        if not validated_peaks:
            return {
                'mean_size': 0.0,
                'std_size': 0.0,
                'size_distribution': 'Unknown'
            }
        
        sizes = [p['crystallite_size'] for p in validated_peaks if p['crystallite_size'] > 0]
        
        if not sizes:
            return {
                'mean_size': 0.0,
                'std_size': 0.0,
                'size_distribution': 'Unknown'
            }
        
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        
        # Classify size distribution
        if std_size / mean_size < 0.1:
            distribution = 'Narrow'
        elif std_size / mean_size < 0.3:
            distribution = 'Moderate'
        else:
            distribution = 'Broad'
        
        return {
            'mean_size': float(mean_size),
            'std_size': float(std_size),
            'distribution': distribution
        }
    
    def check_ordered_mesopores(self, two_theta, intensity, validated_peaks):
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
        low_angle_peaks = [p for p in validated_peaks if 0.5 <= p['position'] <= 10]
        
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
    def complete_analysis(self, two_theta, intensity, elements=None, params=None):
        """
        FULL XRD ANALYSIS — DATABASE DRIVEN (COD + OPTIMADE)
        UI-STABLE, JOURNAL-GRADE
        """
        # =====================================================
        # GUARANTEED DEFAULTS (CRITICAL)
        # =====================================================
        intensity_used = intensity.copy()
        background = np.zeros_like(intensity)

        # -----------------------------------
        # Safety: params must exist
        # -----------------------------------
        if params is None:
            params = {}

        # -----------------------------------
        # X-ray wavelength (DEFINE IT!)
        # -----------------------------------
        if isinstance(params.get("wavelength"), str):
            if "Cu" in params["wavelength"]:
                wavelength = 1.5406
            elif "Mo" in params["wavelength"]:
                wavelength = 0.7107
            elif "Co" in params["wavelength"]:
                wavelength = 1.7902
            else:
                wavelength = 1.5406  # safe default
        else:
            wavelength = float(params.get("wavelength", 1.5406))
        instrument = InstrumentProfile(wavelength=wavelength)
        validator = PhysicalPeakValidator(instrument)
        peak_indices = detect_peaks(two_theta, intensity_used) or []


        validated_peaks = []
        
        for idx in peak_indices:
            peak = validator.validate(
                idx,
                two_theta,
                intensity_used,
                background
            )
            if peak:
                validated_peaks.append(peak)

        results["peaks"] = validated_peaks
        # -----------------------------
        # SAFE DEFAULTS (NEVER BREAK UI)
        # -----------------------------
        xrd_results = {
            "phases": [],
            "phase_fractions": [],
            "validated_peaks": [],
            "top_peaks": [],
            "crystallinity_index": 0.0,
            "crystallite_size": {
                "scherrer": 0.0,
                "williamson_hall": 0.0,
                "distribution": "N/A"
            },
            "microstrain": 0.0,
            "dislocation_density": 0.0,
            "crystal_system": "Unknown",
            "space_group": "Unknown",
            "lattice_parameters": {},
            "ordered_mesopores": False,
            "wavelength": self.wavelength  # ADD THIS
        }
        
        # ================================
        # ADD THIS LINE: Initialize wh variable
        # ================================
        wh = None
        
        try:
            # -----------------------------
            # PREPROCESS
            # -----------------------------
            two_theta_p, intensity_p = self.preprocess_pattern(two_theta, intensity)
            
            intensity_bg_subtracted, background = snip_background(intensity)
            intensity_used = intensity_bg_subtracted
            # -----------------------------
            # PEAK DETECTION (ALL PEAKS)
            # -----------------------------
            validated_peaks = self.analyze_peaks(two_theta_p, intensity_p)
            validated_peaks.sort(key=lambda x: x["intensity"], reverse=True)
    
            xrd_results["validated_peaks"] = validated_peaks
            xrd_results["top_peaks"] = validated_peaks[:15]
            xrd_results["n_peaks_total"] = len(validated_peaks)
            
            # -----------------------------
            # CRYSTALLINITY (✅ FIXED)
            # -----------------------------
            xrd_results["crystallinity_index"] = calculate_crystallinity_index(
                two_theta_p, intensity_p, validated_peaks
            )
    
            # -----------------------------
            # CRYSTALLITE SIZE (ALL PEAKS)
            # -----------------------------
            size_stats = self.calculate_crystallite_statistics(validated_peaks)
            xrd_results["crystallite_size"]["scherrer"] = size_stats["mean_size"]
            xrd_results["crystallite_size"]["distribution"] = size_stats["distribution"]
    
            if len(validated_peaks) >= 3:
                wh = williamson_hall_analysis(validated_peaks, self.wavelength)
                if wh:
                    xrd_results["williamson_hall"] = wh  # 🔑 REQUIRED FOR PLOTTING
                    xrd_results["crystallite_size"]["williamson_hall"] = wh["crystallite_size"]
                    xrd_results["microstrain"] = wh["microstrain"]
                    xrd_results["dislocation_density"] = (
                        15 * wh["microstrain"] / (wh["crystallite_size"] * 1e-9)
                    )
    
            # -----------------------------
            # PHASE IDENTIFICATION
            # -----------------------------
            # In complete_analysis method, replace the phase identification section:
            
            if elements:
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
                        xrd_results["validated_peaks"] = map_peaks_to_phases(validated_peaks, phases)
                        
                        # Calculate phase fractions
                        xrd_results["phase_fractions"] = calculate_phase_fractions(
                            xrd_results["validated_peaks"], phases
                        )
                        
                except Exception as phase_error:
                    # Don't fail entire analysis
                    st.warning(f"Phase identification issue: {phase_error}")
                    xrd_results["phases"] = []
            # ===============================
            # FORCE PERSISTENCE (CRITICAL)
            # ===============================
            if "williamson_hall" in xrd_results:
                pass
            elif wh:
                xrd_results["williamson_hall"] = wh
                
            # Return successful structure - FIXED
            return {
                "valid": True,
                "xrd_results": xrd_results,  # This is what app.py expects
                "xrd_raw": {
                    "two_theta": two_theta_p.tolist(),
                    "intensity": intensity_p.tolist()
                }
            }
    
        except Exception as e:
            # Return error structure - FIXED
            return {
                "valid": False,
                "error": str(e),
                "xrd_results": xrd_results  # Return the empty/default structure
            }


    



























































