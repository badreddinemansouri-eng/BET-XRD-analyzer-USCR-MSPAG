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

warnings.filterwarnings('ignore')
def safe_trapz(y, x):
    """Safe trapezoidal integration that works with all numpy versions"""
    try:
        # Try numpy's trapz
        return np.trapz(y, x)
    except (AttributeError, TypeError):
        # Fallback to manual implementation
        return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))
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
    Detect peaks in XRD pattern with scientific validation
    
    Parameters:
    -----------
    two_theta : Array of 2θ values
    intensity : Array of intensity values
    min_distance : Minimum angular distance between peaks (points)
    threshold : Minimum relative intensity threshold
    
    Returns:
    --------
    List of peak dictionaries
    """
    # Normalize intensity
    intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    
    # Find peak indices
    peak_indices = peakutils.indexes(intensity_norm, 
                                     thres=threshold, 
                                     min_dist=min_distance)
    
    peaks = []
    for idx in peak_indices:
        # Peak position
        theta_peak = two_theta[idx]
        intensity_peak = intensity[idx]
        
        # Calculate FWHM using interpolation
        half_max = intensity_peak / 2
        
        # Find left half-max point
        left_idx = idx
        while left_idx > 0 and intensity[left_idx] > half_max:
            left_idx -= 1
        
        # Interpolate left side
        if left_idx < idx and left_idx >= 0:
            theta_left = np.interp(half_max, 
                                 [intensity[left_idx], intensity[left_idx + 1]],
                                 [two_theta[left_idx], two_theta[left_idx + 1]])
        else:
            theta_left = two_theta[idx]
        
        # Find right half-max point
        right_idx = idx
        while right_idx < len(intensity) - 1 and intensity[right_idx] > half_max:
            right_idx += 1
        
        # Interpolate right side
        if right_idx > idx and right_idx < len(intensity):
            theta_right = np.interp(half_max,
                                  [intensity[right_idx - 1], intensity[right_idx]],
                                  [two_theta[right_idx - 1], two_theta[right_idx]])
        else:
            theta_right = two_theta[idx]
        
        # FWHM in degrees
        fwhm_deg = theta_right - theta_left
        
        # Integrated intensity (area under peak)
        peak_start = max(0, idx - int(min_distance/2))
        peak_end = min(len(intensity), idx + int(min_distance/2))
        
        # Use manual trapezoidal integration for peak area
        x_segment = two_theta[peak_start:peak_end]
        y_segment = intensity[peak_start:peak_end]
        if len(x_segment) >= 2:
            peak_area = np.sum(0.5 * (y_segment[1:] + y_segment[:-1]) * 
                             (x_segment[1:] - x_segment[:-1]))
        else:
            peak_area = 0
        
        # Peak asymmetry
        left_half = intensity_peak - intensity[peak_start:idx].min() if peak_start < idx else intensity_peak
        right_half = intensity_peak - intensity[idx:peak_end].min() if idx < peak_end - 1 else intensity_peak
        asymmetry = left_half / right_half if right_half > 0 else 1.0
        
        peaks.append({
            'index': idx,
            'position': float(theta_peak),
            'intensity': float(intensity_peak),
            'fwhm_deg': float(fwhm_deg),
            'fwhm_rad': float(np.deg2rad(fwhm_deg)),
            'area': float(peak_area),
            'asymmetry': float(asymmetry),
            'theta_left': float(theta_left),
            'theta_right': float(theta_right)
        })
    
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

def calculate_miller_indices(d_spacing, a=1.0, b=1.0, c=1.0, 
                           alpha=90, beta=90, gamma=90,
                           crystal_system='cubic'):
    """
    Calculate possible Miller indices for given d-spacing
    
    Parameters:
    -----------
    d_spacing : Experimental d-spacing in Å
    a, b, c : Lattice parameters in Å
    alpha, beta, gamma : Lattice angles in degrees
    crystal_system : Crystal system
    
    Returns:
    --------
    List of possible (hkl) indices
    """
    # Convert angles to radians
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    gamma_rad = np.deg2rad(gamma)
    
    # Reciprocal lattice calculations based on crystal system
    indices = []
    
    if crystal_system.lower() == 'cubic':
        # d = a / √(h² + k² + l²)
        for h in range(-3, 4):
            for k in range(-3, 4):
                for l in range(-3, 4):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    
                    d_calc = a / np.sqrt(h**2 + k**2 + l**2)
                    error = abs(d_calc - d_spacing) / d_spacing
                    
                    if error < 0.02:  # 2% tolerance
                        indices.append({
                            'h': h, 'k': k, 'l': l,
                            'd_calculated': d_calc,
                            'error_percent': error * 100
                        })
    
    elif crystal_system.lower() == 'hexagonal':
        # 1/d² = (4/3a²)(h² + hk + k²) + (l²/c²)
        for h in range(-3, 4):
            for k in range(-3, 4):
                for l in range(-3, 4):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    
                    term1 = (4/3) * (h**2 + h*k + k**2) / (a**2)
                    term2 = l**2 / (c**2)
                    d_calc = 1 / np.sqrt(term1 + term2)
                    
                    error = abs(d_calc - d_spacing) / d_spacing
                    
                    if error < 0.02:
                        indices.append({
                            'h': h, 'k': k, 'l': l,
                            'd_calculated': d_calc,
                            'error_percent': error * 100
                        })
    
    elif crystal_system.lower() == 'tetragonal':
        # 1/d² = (h² + k²)/a² + l²/c²
        for h in range(-3, 4):
            for k in range(-3, 4):
                for l in range(-3, 4):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    
                    d_calc = 1 / np.sqrt((h**2 + k**2)/a**2 + l**2/c**2)
                    error = abs(d_calc - d_spacing) / d_spacing
                    
                    if error < 0.02:
                        indices.append({
                            'h': h, 'k': k, 'l': l,
                            'd_calculated': d_calc,
                            'error_percent': error * 100
                        })
    
    else:
        # Simple calculation for unknown system
        indices.append({
            'h': '?', 'k': '?', 'l': '?',
            'd_calculated': d_spacing,
            'error_percent': 0.0
        })
    
    # Sort by error
    indices.sort(key=lambda x: x['error_percent'])
    
    return indices[:5]  # Return top 5 matches

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

def williamson_hall_analysis(peaks, wavelength=1.5406):
    """
    Williamson-Hall plot for separating size and strain effects
    
    β cosθ = Kλ/D + 4ε sinθ
    
    Plot β cosθ vs 4 sinθ
    
    Parameters:
    -----------
    peaks : List of peak dictionaries
    wavelength : X-ray wavelength in Å
    
    Returns:
    --------
    Dictionary with size, strain, and plot data
    """
    if len(peaks) < 3:
        return None
    
    x_vals = []
    y_vals = []
    
    for peak in peaks:
        theta_deg = peak['position'] / 2  # Convert 2θ to θ
        theta_rad = np.deg2rad(theta_deg)
        fwhm_rad = peak['fwhm_rad']
        
        # Williamson-Hall coordinates
        x = 4 * np.sin(theta_rad)  # 4 sinθ
        y = fwhm_rad * np.cos(theta_rad)  # β cosθ
        
        x_vals.append(x)
        y_vals.append(y)
    
    # Linear regression
    slope, intercept, r_value, _, _ = stats.linregress(x_vals, y_vals)
    
    # Extract parameters
    # intercept = Kλ/D → D = Kλ/intercept
    # slope = 4ε → ε = slope/4
    
    K = 0.9  # Shape factor
    size_nm = (K * wavelength) / (intercept * 10) if intercept > 0 else 0
    microstrain = slope / 4
    
    return {
        'crystallite_size': float(size_nm),
        'microstrain': float(microstrain),
        'r_squared': float(r_value**2),
        'slope': float(slope),
        'intercept': float(intercept),
        'x_data': [float(x) for x in x_vals],
        'y_data': [float(y) for y in y_vals]
    }

# ============================================================================
# CRYSTALLINITY INDEX
# ============================================================================
# In xrd_analyzer.py - Update the calculate_crystallinity_index function:
def calculate_crystallinity_index(two_theta, intensity, peak_indices):
    """
    Calculate crystallinity index
    """
    if len(peak_indices) == 0:
        return 0.0
    
    # Create a baseline (amorphous background)
    amorphous_background = gaussian_filter1d(intensity, sigma=50)
    
    # Calculate areas using safe trapezoidal integration
    total_area = safe_trapz(intensity, two_theta)
    amorphous_area = safe_trapz(amorphous_background, two_theta)
    
    # Crystalline area = total area - amorphous area
    crystalline_area = max(0, total_area - amorphous_area)
    
    if total_area > 0:
        crystallinity = crystalline_area / total_area
    else:
        crystallinity = 0.0
    
    return crystallinity
def estimate_amorphous_background(intensity, iterations=50):
    """Improved background estimation using asymmetric SNIP"""
    # Convert to log scale
    y = np.log(np.log(intensity + 1) + 1)
    
    # Initialize background
    bg = y.copy()
    
    for i in range(iterations):
        # Apply SNIP algorithm
        for j in range(1, len(y) - 1):
            bg[j] = min(bg[j], 0.5 * (bg[j-1] + bg[j+1]))
    
    # Convert back
    bg_corrected = np.exp(np.exp(bg) - 1) - 1
    
    return bg_corrected

# In the AdvancedXRDAnalyzer class, update williamson_hall_analysis:
def williamson_hall_analysis(self, peaks, wavelength=1.5406):
    """
    Williamson-Hall plot for separating size and strain effects
    Using ALL peaks, not just major ones
    
    Reference: 
    Williamson, G.K., & Hall, W.H. (1953). Acta Metallurgica, 1(1), 22-31.
    """
    if len(peaks) < 3:
        return None
    
    x_vals, y_vals = [], []
    errors = []
    
    for peak in peaks:
        try:
            theta_deg = peak['position'] / 2
            theta_rad = np.deg2rad(theta_deg)
            fwhm_rad = peak['fwhm_rad']
            
            # Calculate Williamson-Hall coordinates
            x = 4 * np.sin(theta_rad)  # 4 sinθ
            y = fwhm_rad * np.cos(theta_rad)  # β cosθ
            
            # Calculate error in FWHM (instrumental broadening correction)
            # Instrumental broadening from NIST SRM 640c (Si standard)
            instrumental_broadening = 0.02  # radians, adjust based on instrument
            fwhm_corrected = np.sqrt(fwhm_rad**2 - instrumental_broadening**2)
            
            if fwhm_corrected > 0:
                y_corrected = fwhm_corrected * np.cos(theta_rad)
                x_vals.append(x)
                y_vals.append(y_corrected)
                
                # Error estimation
                error_fwhm = 0.05 * fwhm_rad  # 5% relative error
                error_y = error_fwhm * np.cos(theta_rad)
                errors.append(error_y)
        except:
            continue
    
    if len(x_vals) < 3:
        return None
    
    # Weighted linear regression using errors as weights
    try:
        weights = 1.0 / np.array(errors)
        coeffs = np.polyfit(x_vals, y_vals, 1, w=weights)
        slope, intercept = coeffs
        
        # Calculate R²
        y_pred = slope * np.array(x_vals) + intercept
        ss_res = np.sum((np.array(y_vals) - y_pred)**2)
        ss_tot = np.sum((np.array(y_vals) - np.mean(y_vals))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Extract parameters
        K = 0.9  # Shape factor (Scherrer constant)
        size_nm = (K * wavelength) / (intercept * 10) if intercept > 0 else 0
        microstrain = slope / 4
        
        # Calculate dislocation density (Williamson-Smallman)
        # ρ = 1/D² for edge dislocations
        dislocation_density = 1 / (size_nm * 1e-9)**2 if size_nm > 0 else 0
        
        return {
            'crystallite_size': float(size_nm),
            'microstrain': float(microstrain),
            'r_squared': float(r_squared),
            'slope': float(slope),
            'intercept': float(intercept),
            'x_data': [float(x) for x in x_vals],
            'y_data': [float(y) for y in y_vals],
            'errors': [float(e) for e in errors],
            'dislocation_density': float(dislocation_density),
            'n_peaks_used': len(x_vals),
            'method': 'Williamson-Hall (all peaks, weighted)'
        }
    except:
        return None
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
        main_peak_d = d_spacings[np.argmax([p['intensity'] for p in peaks])]
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
        peaks = detect_peaks(two_theta, intensity, threshold=threshold)
        
        # Analyze each peak
        analyzed_peaks = []
        for peak in peaks:
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
                'size_distribution': 'Unknown'
            }
        
        sizes = [p['crystallite_size'] for p in peaks if p['crystallite_size'] > 0]
        
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
    
    def complete_analysis(self, two_theta, intensity, 
                         crystal_system='Unknown',
                         space_group='',
                         lattice_params=''):
        """
        Complete XRD analysis
        
        Parameters:
        -----------
        two_theta : Array of 2θ values
        intensity : Array of intensity values
        crystal_system : Crystal system
        space_group : Space group
        lattice_params : Lattice parameters string
        
        Returns:
        --------
        Complete analysis dictionary
        """
        try:
            # Preprocess pattern
            two_theta_proc, intensity_proc = self.preprocess_pattern(two_theta, intensity)
            
            # Detect and analyze peaks
            peaks = self.analyze_peaks(two_theta_proc, intensity_proc)
            
            # Sort peaks by intensity (descending) to get most intensive peaks
            peaks.sort(key=lambda x: x['intensity'], reverse=True)
            
            # Store all peaks and top peaks separately
            all_peaks = peaks  # All detected peaks
            top_peaks = peaks[:10]  # Top 10 most intensive peaks for display
            
            # Calculate crystallinity index
            peak_indices = [p['index'] for p in peaks]
            crystallinity_index = calculate_crystallinity_index(
                two_theta_proc, intensity_proc, peak_indices
            )
            
            # Try to assign hkl indices to top peaks if crystal system is known
            if crystal_system != 'Unknown' and lattice_params:
                # Parse lattice parameters
                a, b, c = 1.0, 1.0, 1.0
                # Simple parsing: a=4.0, b=4.0, c=4.0
                for param in lattice_params.split(','):
                    if '=' in param:
                        key, value = param.strip().split('=')
                        key = key.strip()
                        value = float(value.strip())
                        if key == 'a':
                            a = value
                        elif key == 'b':
                            b = value
                        elif key == 'c':
                            c = value
                
                # Assign hkl to top peaks
                for peak in top_peaks:
                    try:
                        hkl_list = calculate_miller_indices(
                            d_spacing=peak['d_spacing'],
                            a=a, b=b, c=c,
                            crystal_system=crystal_system
                        )
                        if hkl_list:
                            # Take the best match (first one)
                            best_hkl = hkl_list[0]
                            peak['hkl'] = f"({abs(best_hkl['h'])}{abs(best_hkl['k'])}{abs(best_hkl['l'])})"
                            peak['hkl_detail'] = best_hkl
                    except Exception as e:
                        peak['hkl'] = ''
            
            # Calculate crystallite statistics
            size_stats = self.calculate_crystallite_statistics(peaks)
            
            # Williamson-Hall analysis (if enough peaks)
            williamson_hall = None
            microstrain = 0.0
            dislocation_density = 0.0
            
            if len(peaks) >= 3:
                williamson_hall = williamson_hall_analysis(peaks, self.wavelength)
                if williamson_hall:
                    microstrain = williamson_hall['microstrain']
                    # Dislocation density: ρ = 15ε/(aD) [simplified]
                    if williamson_hall['crystallite_size'] > 0:
                        dislocation_density = 15 * microstrain / (
                            williamson_hall['crystallite_size'] * 1e-9
                        )
            
            # Check for ordered mesopores
            mesopore_analysis = self.check_ordered_mesopores(
                two_theta_proc, intensity_proc, peaks
            )
            
            # Parse lattice parameters if provided
            lattice_dict = {}
            if lattice_params:
                # Simple parsing: look for a=, b=, c= values
                for match in re.finditer(r'([abc])\s*=\s*([\d\.]+)', lattice_params):
                    lattice_dict[match.group(1)] = float(match.group(2))
            
            # Try to refine lattice parameters
            refined_params = None
            if crystal_system != 'Unknown' and len(peaks) >= 3:
                refined_params = refine_lattice_parameters(
                    peaks, self.wavelength, crystal_system
                )
            
            # Compile results
            results = {
                'valid': True,
                'wavelength': float(self.wavelength),
                'two_theta': two_theta_proc.tolist(),
                'intensity': intensity_proc.tolist(),
                'peaks': all_peaks,  # All peaks
                'top_peaks': top_peaks,  # Top intensive peaks for display
                'n_peaks_total': len(all_peaks),
                'crystallinity_index': float(crystallinity_index),
                'crystallite_size': {
                    'scherrer': float(size_stats['mean_size']),
                    'williamson_hall': williamson_hall['crystallite_size'] if williamson_hall else 0.0,
                    'distribution': size_stats['distribution']
                },
                'microstrain': float(microstrain),
                'dislocation_density': float(dislocation_density),
                'ordered_mesopores': mesopore_analysis['ordered'],
                'mesopore_analysis': mesopore_analysis,
                'williamson_hall': williamson_hall,
                'crystal_system': crystal_system,
                'space_group': space_group,
                'lattice_parameters': lattice_dict,
                'refined_parameters': refined_params,
                'background_subtraction': self.background_subtraction,
                'smoothing': self.smoothing,
                'scherrer_constant': float(self.scherrer_constant),
                'n_points': len(two_theta_proc)
            }
            
            return results
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'wavelength': float(self.wavelength),
                'peaks': [],
                'top_peaks': [],
                'n_peaks_total': 0,
                'crystallinity_index': 0.0,
                'crystallite_size': {'scherrer': 0.0, 'williamson_hall': 0.0, 'distribution': 'Unknown'},
                'microstrain': 0.0,
                'ordered_mesopores': False
            }
    
    


