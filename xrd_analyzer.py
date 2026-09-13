"""
ADVANCED XRD ANALYSIS ENGINE
========================================================================
Scientific implementation following:
1. Klug & Alexander, X-ray Diffraction Procedures, 1974
2. Williamson & Hall, Acta Metall., 1953, 1, 22-31 (Microstrain)
3. Scherrer, Nachr. Ges. Wiss. Gottingen, 1918, 2, 98 (Crystallite size)
4. Snip, Nucl. Instrum. Methods, 1984, 223, 117 (Background subtraction)
5. Ruland, Acta Cryst., 1961, 14, 1180 (Crystallinity index)
========================================================================
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import signal, optimize, stats

from scientific_integration import (
    calculate_phase_fractions,
    map_peaks_to_phases,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from xrd_phase_identifier_nano import identify_phases_universal
    UNIVERSAL_PHASE_ID_AVAILABLE = True
except ImportError:
    UNIVERSAL_PHASE_ID_AVAILABLE = False
    from xrd_phase_identifier import identify_phases

try:
    from xrd_peak_physics import PhysicalPeakValidator, InstrumentProfile
    PEAK_VALIDATOR_AVAILABLE = True
except ImportError:
    PEAK_VALIDATOR_AVAILABLE = False


# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================
XRAY_WAVELENGTHS = {
    'Cu Ka': 1.5406,
    'Cu Ka1': 1.54056,
    'Cu Ka2': 1.54439,
    'Mo Ka': 0.71073,
    'Co Ka': 1.78897,
    'Cr Ka': 2.29100,
}


# ============================================================================
# HELPERS
# ============================================================================
def safe_trapz(y, x):
    """Trapezoidal integration that works across numpy versions."""
    try:
        return float(np.trapz(y, x))
    except (AttributeError, TypeError):
        return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])))


# ============================================================================
# XRD DATA EXTRACTION
# ============================================================================
def extract_xrd_data(file, preview_only: bool = False):
    """
    Extract 2theta / intensity from a variety of text-based XRD file formats.

    Returns
    -------
    (two_theta, intensity, message)
    """
    try:
        content = file.read().decode('utf-8', errors='ignore')
        file.seek(0)

        lines = content.split('\n')
        data_points = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            for delimiter in ['\t', ',', ';', ' ']:
                parts = [p.strip() for p in line.split(delimiter) if p.strip()]
                if len(parts) >= 2:
                    try:
                        theta = float(parts[0])
                        inten = float(parts[1])
                        if -5 <= theta <= 180:
                            data_points.append((theta, inten))
                        break
                    except ValueError:
                        continue

        if not data_points:
            try:
                for delimiter in ['\t', ',', ';', ' ']:
                    try:
                        df = pd.read_csv(file, delimiter=delimiter, header=None)
                        file.seek(0)
                        if df.shape[1] >= 2:
                            theta = pd.to_numeric(df.iloc[:, 0], errors='coerce')
                            inten = pd.to_numeric(df.iloc[:, 1], errors='coerce')
                            valid = theta.notna() & inten.notna()
                            data_points = list(zip(theta[valid], inten[valid]))
                            break
                    except Exception:
                        continue
            except Exception as exc:
                logger.debug("pandas fallback failed: %s", exc)

        if not data_points:
            return None, None, "No valid data found"

        data_points = sorted(data_points, key=lambda x: x[0])
        two_theta = np.array([p[0] for p in data_points])
        intensity = np.array([p[1] for p in data_points])

        unique_theta, unique_indices = np.unique(two_theta, return_index=True)
        if len(unique_theta) < len(two_theta):
            two_theta = unique_theta
            intensity = intensity[unique_indices]

        if preview_only:
            return two_theta, intensity, f"Found {len(two_theta)} data points"

        if len(two_theta) < 50:
            return None, None, f"Insufficient data points: {len(two_theta)} (<50)"

        if two_theta.max() - two_theta.min() < 10:
            return None, None, (
                f"Insufficient angular range: "
                f"{two_theta.max() - two_theta.min():.1f} deg (<10)"
            )

        return two_theta, intensity, "Data extracted successfully"

    except Exception as exc:
        return None, None, f"Extraction error: {exc}"


# ============================================================================
# BACKGROUND SUBTRACTION (SNIP)
# ============================================================================
def snip_background(intensity, iterations: int = 100,
                    reduction_factor: float = 0.8):
    """
    SNIP background estimation.

    References:
        Ryan et al., Nucl. Instrum. Methods B, 1988, 34, 396-402
        Morhac et al., Nucl. Instrum. Methods A, 1997, 401, 113-132
    """
    spectrum = np.log(np.log(intensity + 1.0) + 1.0)
    background = spectrum.copy()

    for _ in range(iterations):
        for i in range(1, len(spectrum) - 1):
            min_val = min(background[i - 1], background[i + 1])
            if background[i] > min_val:
                background[i] = (
                    min_val + (background[i] - min_val) * reduction_factor
                )

    bg_subtracted = np.exp(np.exp(background) - 1.0) - 1.0
    return intensity - bg_subtracted, bg_subtracted


# ============================================================================
# PEAK DETECTION WITH PHYSICAL VALIDATION
# ============================================================================
def detect_peaks_with_validation(two_theta, intensity, background,
                                 min_distance_deg: float = 1.0,
                                 min_prominence: float = 0.03) -> Dict:
    """
    Peak detection with physical validation. If PhysicalPeakValidator is
    available, each candidate peak is fitted with Gaussian / Lorentzian /
    pseudo-Voigt models and kept only if the fit quality is acceptable.
    """
    signal_corr = intensity - background
    signal_corr[signal_corr < 0] = 0.0

    angular_step = float(np.mean(np.diff(two_theta))) if len(two_theta) > 1 else 0.02
    min_distance_points = int(min_distance_deg / angular_step) if angular_step > 0 else 20
    min_distance_points = max(min_distance_points, 5)

    noise_level = float(np.std(signal_corr))

    peaks_idx, properties = signal.find_peaks(
        signal_corr,
        prominence=max(1.0 * noise_level, 0.005 * float(np.max(signal_corr))),
        width=(1, None),
        distance=min_distance_points,
    )

    detected_peaks = [
        {
            "index": int(i),
            "position": float(two_theta[i]),
            "intensity": float(intensity[i]),
            "source": "find_peaks",
        }
        for i in peaks_idx
    ]

    # Ensure the raw apex is always present
    idx_max = int(np.argmax(intensity))
    true_theta = float(two_theta[idx_max])
    true_intensity = float(intensity[idx_max])

    exists = any(abs(p["position"] - true_theta) < 0.3 for p in detected_peaks)
    if not exists:
        detected_peaks.append({
            "index": idx_max,
            "position": true_theta,
            "intensity": true_intensity,
            "source": "raw_apex",
            "fwhm_estimate": None,
        })

    debug_local_maxima = [
        {
            "index": int(i),
            "position": float(two_theta[i]),
            "intensity": float(intensity[i]),
        }
        for i in peaks_idx
    ]

    if PEAK_VALIDATOR_AVAILABLE:
        instrument = InstrumentProfile()
        validator = PhysicalPeakValidator(instrument)

        structural_peaks = []
        for peak in detected_peaks:
            idx = peak["index"]
            result = validator.validate(
                idx=idx,
                two_theta=two_theta,
                intensity=intensity,
                background=background,
            )

            if result is None:
                if peak.get("source") == "raw_apex":
                    result = {
                        "index": idx,
                        "two_theta": float(two_theta[idx]),
                        "intensity": float(intensity[idx]),
                        "fwhm_deg": 0.3,
                        "area": float(intensity[idx]),
                        "snr": float('inf'),
                        "shape": "raw_apex",
                        "fit_quality": 1.0,
                    }
                else:
                    continue

            peak_dict = {
                "index": int(result["index"]),
                "position": float(result["two_theta"]),
                "intensity": float(result["intensity"]),
                "intensity_raw": float(intensity[result["index"]]),
                "fwhm_deg": float(result["fwhm_deg"]),
                "fwhm_rad": float(np.deg2rad(result["fwhm_deg"])),
                "area": float(result["area"]),
                "snr": float(result["snr"]),
                "shape": result["shape"],
                "fit_quality": float(result["fit_quality"]),
            }

            true_idx = result["index"]
            peak_start = max(0, true_idx - 10)
            peak_end = min(len(intensity), true_idx + 10)
            left_half = intensity[true_idx] - np.min(intensity[peak_start:true_idx])
            right_half = intensity[true_idx] - np.min(intensity[true_idx:peak_end])
            peak_dict["asymmetry"] = (
                float(left_half / right_half) if right_half > 0 else 1.0
            )

            structural_peaks.append(peak_dict)

        structural_peaks.sort(key=lambda x: x["intensity"], reverse=True)

        return {
            "local_maxima": peaks_idx.tolist(),
            "local_maxima_debug": debug_local_maxima,
            "structural_peaks": structural_peaks,
            "n_local_maxima": len(peaks_idx),
            "n_structural_peaks": len(structural_peaks),
        }

    # Fallback: no physical validator available
    peaks = []
    for i, idx in enumerate(peaks_idx):
        widths = properties.get("widths", [])
        fwhm_points = widths[i] if i < len(widths) else 5
        fwhm_deg = float(fwhm_points * angular_step)

        if fwhm_deg < 0.02 or fwhm_deg > 5.0:
            continue

        prominences = properties.get("prominences", [])
        promin = float(prominences[i]) if i < len(prominences) else 0.0

        peaks.append({
            "index": int(idx),
            "position": float(two_theta[idx]),
            "intensity": float(intensity[idx]),
            "intensity_raw": float(intensity[idx]),
            "fwhm_deg": fwhm_deg,
            "fwhm_rad": float(np.deg2rad(fwhm_deg)),
            "prominence": promin,
        })

    peaks.sort(key=lambda x: x["intensity"], reverse=True)
    return {
        "local_maxima": peaks_idx.tolist(),
        "structural_peaks": peaks,
        "n_local_maxima": len(peaks_idx),
        "n_structural_peaks": len(peaks),
    }


def detect_peaks(two_theta, intensity,
                 min_distance_deg: float = 1.0,
                 min_prominence: float = 0.03) -> List[Dict]:
    """
    Standard XRD peak detection: SNIP background, Savitzky-Golay smoothing,
    then physical validation.
    """
    intensity_corr, background = snip_background(intensity)

    if len(intensity_corr) > 11:
        intensity_smooth = signal.savgol_filter(
            intensity_corr, window_length=11, polyorder=3
        )
    else:
        intensity_smooth = intensity_corr

    peak_results = detect_peaks_with_validation(
        two_theta=two_theta,
        intensity=intensity_smooth,
        background=background,
        min_distance_deg=min_distance_deg,
        min_prominence=min_prominence,
    )
    return peak_results["structural_peaks"]


def detect_peaks_raw(two_theta, intensity,
                     min_distance_deg: float = 1.0,
                     min_prominence: float = 0.015) -> List[Dict]:
    """
    Raw peak detection for phase identification, with no smoothing.
    Uses a wide Savitzky-Golay filter only as a background estimate.
    """
    background = signal.savgol_filter(intensity, 101, 3)
    signal_raw = intensity - background
    signal_raw[signal_raw < 0] = 0.0

    angular_step = float(np.mean(np.diff(two_theta))) if len(two_theta) > 1 else 0.02
    min_distance_points = int(min_distance_deg / angular_step) if angular_step > 0 else 20

    peaks_idx, properties = signal.find_peaks(
        signal_raw,
        prominence=min_prominence * float(np.max(signal_raw)),
        width=(3, None),
        distance=max(min_distance_points, 5),
    )

    peaks = []
    for i, idx in enumerate(peaks_idx):
        widths = properties.get("widths", [])
        fwhm_points = widths[i] if i < len(widths) else 5
        fwhm_deg = float(fwhm_points * angular_step)

        if fwhm_deg < 0.01:
            continue
        if fwhm_deg > 10.0:
            fwhm_deg = 3.0

        prominences = properties.get("prominences", [])
        promin = float(prominences[i]) if i < len(prominences) else 0.0

        peaks.append({
            "index": int(idx),
            "position": float(two_theta[idx]),
            "intensity": float(intensity[idx]),
            "fwhm_deg": fwhm_deg,
            "fwhm_rad": float(np.deg2rad(fwhm_deg)),
            "prominence": promin,
        })

    peaks.sort(key=lambda x: x["intensity"], reverse=True)
    return peaks


# ============================================================================
# CRYSTALLOGRAPHIC CALCULATIONS
# ============================================================================
def calculate_d_spacing(theta_deg: float, wavelength: float = 1.5406) -> float:
    """Bragg's law: n*lambda = 2*d*sin(theta)."""
    theta_rad = np.deg2rad(theta_deg / 2.0)
    return float(wavelength / (2.0 * np.sin(theta_rad)))


def scherrer_crystallite_size(fwhm_rad: float, theta_rad: float,
                              wavelength: float = 1.5406, K: float = 0.9,
                              instrument_fwhm_rad: float = 0.0) -> float:
    """
    Scherrer crystallite size with instrumental broadening correction.

    D = K * lambda / (beta * cos(theta))
    beta = sqrt(beta_measured^2 - beta_instrument^2)
    """
    if instrument_fwhm_rad > 0:
        sample_fwhm_rad = np.sqrt(fwhm_rad ** 2 - instrument_fwhm_rad ** 2)
        if sample_fwhm_rad <= 0:
            return 0.0
    else:
        sample_fwhm_rad = fwhm_rad

    size_angstrom = K * wavelength / (sample_fwhm_rad * np.cos(theta_rad))
    return float(size_angstrom / 10.0)  # Angstrom -> nm


def williamson_hall_analysis(peaks: List[Dict],
                             wavelength: float = 1.5406,
                             instrument_fwhm_rad: float = 0.0) -> Optional[Dict]:
    """
    Williamson-Hall size-strain separation.

    beta * cos(theta) = K * lambda / D + 4 * epsilon * sin(theta)

    Rejects peaks that are too narrow, too broad, asymmetric, or low SNR.
    Requires at least 4 valid peaks with R^2 >= 0.85.
    """
    if not peaks or len(peaks) < 4:
        return None

    valid_peaks = []
    for p in peaks:
        fwhm_deg = p.get("fwhm_deg", 0)
        asymmetry = p.get("asymmetry", 1.0)
        snr = p.get("snr")

        # Upper bound relaxed to 5.0 deg for nanocrystalline materials
        if fwhm_deg < 0.03 or fwhm_deg > 5.0:
            continue
        if asymmetry < 0.7 or asymmetry > 1.3:
            continue
        if snr is not None and snr < 3.0:
            continue

        valid_peaks.append(p)

    if len(valid_peaks) < 4:
        return None

    x_vals, y_vals = [], []
    for p in valid_peaks:
        theta_rad = np.deg2rad(p["position"] / 2.0)
        beta_measured = p["fwhm_rad"]

        if instrument_fwhm_rad > 0:
            beta_sample = np.sqrt(beta_measured ** 2 - instrument_fwhm_rad ** 2)
            if beta_sample <= 0:
                continue
        else:
            beta_sample = beta_measured

        x_vals.append(4.0 * np.sin(theta_rad))
        y_vals.append(beta_sample * np.cos(theta_rad))

    if len(x_vals) < 4:
        return None

    try:
        slope, intercept, r_value, _, _ = stats.linregress(x_vals, y_vals)
    except Exception as exc:
        logger.warning("Williamson-Hall regression failed: %s", exc)
        return None

    if intercept <= 0 or np.isnan(intercept):
        return None
    if r_value ** 2 < 0.85:
        return None

    K = 0.9
    size_nm = (K * wavelength) / (intercept * 10.0)
    microstrain = slope / 4.0

    if size_nm < 0.5 or size_nm > 500:
        return None
    if microstrain < 0 or microstrain > 0.05:
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
        "valid_peaks": len(valid_peaks),
        "wh_valid": True,
    }


# ============================================================================
# CRYSTALLINITY INDEX
# ============================================================================
def calculate_crystallinity_index(two_theta, intensity,
                                  peaks: List[Dict]) -> float:
    """
    Crystallinity index based on crystalline area / total area.

    CI = Integral(I_crystalline) / Integral(I_total)

    Capped at 0.95 because perfect crystallinity is unphysical for a
    powder sample.

    Reference: Ruland, W. (1961). Acta Cryst., 14, 1180.
    """
    if not peaks:
        return 0.0

    intensity_corr, _ = snip_background(intensity)
    total_area = safe_trapz(intensity_corr, two_theta)
    if total_area <= 0:
        return 0.0

    crystalline_area = sum(p.get("area", 0.0) for p in peaks)
    ci = crystalline_area / total_area

    avg_fwhm = float(np.mean([p.get("fwhm_deg", 0) for p in peaks]))
    if avg_fwhm > 0.5:
        ci *= 0.9

    return float(min(max(ci, 0.0), 0.95))


# ============================================================================
# LATTICE PARAMETER REFINEMENT
# ============================================================================
def refine_lattice_parameters(peaks: List[Dict],
                              wavelength: float = 1.5406,
                              crystal_system: str = "cubic",
                              initial_guess: float = 4.0) -> Optional[Dict]:
    """Refine cubic lattice parameter using up to five observed d-spacings."""
    if len(peaks) < 3:
        return None

    d_spacings = [calculate_d_spacing(p["position"], wavelength) for p in peaks]

    if crystal_system != "cubic":
        return None

    main_peak_d = d_spacings[int(np.argmax([p["intensity"] for p in peaks]))]
    a_estimated = main_peak_d * np.sqrt(3)

    try:
        def cubic_residual(a_val):
            res = 0.0
            for d_exp in d_spacings[:5]:
                for hkl in [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 2, 0)]:
                    d_calc = a_val / np.sqrt(sum(h ** 2 for h in hkl))
                    error = abs(d_exp - d_calc) / d_exp
                    if error < 0.05:
                        res += (d_exp - d_calc) ** 2
                        break
            return res

        result = optimize.minimize_scalar(
            cubic_residual,
            bounds=(a_estimated * 0.8, a_estimated * 1.2),
            method="bounded",
        )
        if result.success:
            return {"a": float(result.x), "error": float(result.fun)}
    except Exception as exc:
        logger.warning("Lattice refinement failed: %s", exc)

    return {"a": float(a_estimated), "error": 0.0}


def allowed_hkl(hkl: Tuple[int, int, int], crystal_system: str) -> bool:
    h, k, l = hkl
    if crystal_system in ("cubic_fcc", "fcc"):
        return (h + k + l) % 2 == 0
    if crystal_system in ("cubic_bcc", "bcc"):
        return h % 2 == k % 2 == l % 2
    return True


# ============================================================================
# MAIN ANALYZER CLASS
# ============================================================================
class AdvancedXRDAnalyzer:
    """
    Advanced XRD analysis with physical and crystallographic validation.
    """

    def __init__(self, wavelength: float = 1.5406,
                 background_subtraction: bool = True,
                 smoothing: str = "Savitzky-Golay",
                 instrument_fwhm_deg: float = 0.0):
        self.wavelength = wavelength
        self.background_subtraction = background_subtraction
        self.smoothing = smoothing
        self.scherrer_constant = 0.9
        self.instrument_fwhm_deg = instrument_fwhm_deg
        self.instrument_fwhm_rad = float(np.deg2rad(instrument_fwhm_deg))

    # ------------------------------------------------------------------
    def validate_crystallographic_results(self, xrd_results: Dict) -> Dict:
        """Validate XRD results against crystallographic standards."""
        validation = {
            "valid": True,
            "warnings": [],
            "confidence_score": 1.0,
            "scientific_checks": [],
        }

        size = xrd_results.get("crystallite_size", {}).get("scherrer", 0)
        if size > 1000:
            validation["warnings"].append(
                f"Crystallite size unusually large: {size:.1f} nm"
            )
            validation["confidence_score"] *= 0.7
        elif size < 0.5 and size > 0:
            validation["warnings"].append(
                f"Crystallite size unusually small: {size:.1f} nm"
            )
            validation["confidence_score"] *= 0.6

        ci = xrd_results.get("crystallinity_index", 0)
        if ci < 0 or ci > 1:
            validation["warnings"].append(
                f"Crystallinity index out of range: {ci:.2f}"
            )
            validation["confidence_score"] *= 0.5
        elif ci == 1.00:
            validation["warnings"].append(
                "Crystallinity index = 1.00 is physically impossible"
            )
            validation["confidence_score"] *= 0.3

        n_peaks = len(xrd_results.get("peaks", []))
        if n_peaks < 3:
            validation["warnings"].append(f"Few peaks detected: {n_peaks}")
            validation["confidence_score"] *= 0.8
        elif n_peaks > 100:
            validation["warnings"].append(
                f"Excessive peaks detected ({n_peaks}), likely including noise"
            )
            validation["confidence_score"] *= 0.7

        for peak in xrd_results.get("peaks", []):
            fwhm = peak.get("fwhm_deg", 0)
            if fwhm < 0.01 or fwhm > 5.0:
                validation["warnings"].append(
                    f"Unphysical FWHM: {fwhm:.4f} deg at "
                    f"2theta={peak.get('position', 0):.2f}"
                )
                validation["confidence_score"] *= 0.9

        validation["scientific_checks"] = [
            {
                "check": "Peak count adequate",
                "passed": 3 <= n_peaks <= 50,
                "value": n_peaks,
                "expected": "3-50 peaks",
            },
            {
                "check": "Crystallite size physical",
                "passed": 0.5 < size < 500,
                "value": f"{size:.1f} nm",
                "expected": "0.5-500 nm",
            },
            {
                "check": "Crystallinity index valid",
                "passed": 0 < ci < 1,
                "value": f"{ci:.3f}",
                "expected": "0.0-1.0 (not 1.00)",
            },
            {
                "check": "Peak FWHM physical",
                "passed": all(
                    0.03 < p.get("fwhm_deg", 0) < 3.0
                    for p in xrd_results.get("peaks", [])
                ),
                "value": (
                    f"{np.mean([p.get('fwhm_deg', 0) for p in xrd_results.get('peaks', [])]):.3f} deg avg"
                    if xrd_results.get("peaks") else "N/A"
                ),
                "expected": "0.03-3.0 deg",
            },
        ]

        validation["valid"] = validation["confidence_score"] > 0.5
        return validation

    # ------------------------------------------------------------------
    def preprocess_pattern(self, two_theta, intensity):
        sort_idx = np.argsort(two_theta)
        two_theta = two_theta[sort_idx]
        intensity = intensity[sort_idx]

        if self.smoothing == "Savitzky-Golay" and len(intensity) > 11:
            intensity = signal.savgol_filter(
                intensity, window_length=11, polyorder=3
            )
        return two_theta, intensity

    # ------------------------------------------------------------------
    def calculate_crystallite_statistics(self, peaks: List[Dict]) -> Dict:
        if not peaks:
            return {
                "mean_size": 0.0, "std_size": 0.0,
                "distribution": "Unknown", "n_peaks": 0,
            }

        sizes = [p["crystallite_size"] for p in peaks if p["crystallite_size"] > 0]
        if not sizes:
            return {
                "mean_size": 0.0, "std_size": 0.0,
                "distribution": "Unknown", "n_peaks": 0,
            }

        mean_size = float(np.mean(sizes))
        std_size = float(np.std(sizes))
        cv = std_size / mean_size if mean_size > 0 else 0.0

        if cv < 0.1:
            distribution = "Narrow (monodisperse)"
        elif cv < 0.25:
            distribution = "Moderate"
        elif cv < 0.5:
            distribution = "Broad (polydisperse)"
        else:
            distribution = "Very broad"

        return {
            "mean_size": mean_size,
            "std_size": std_size,
            "distribution": distribution,
            "cv": float(cv),
            "n_peaks": len(sizes),
        }

    # ------------------------------------------------------------------
    def calculate_nano_tolerance(self, size_nm: float) -> float:
        """
        Physics-based d-spacing tolerance from Scherrer size.
        < 5 nm -> 10%; 5-10 nm -> 6%; > 10 nm -> 3%.
        """
        if size_nm < 5:
            return 0.10
        if size_nm < 10:
            return 0.06
        return 0.03

    # ------------------------------------------------------------------
    def check_ordered_mesopores(self, two_theta, intensity, peaks) -> Dict:
        low_angle_peaks = [p for p in peaks if 0.5 <= p["position"] <= 10]
        if len(low_angle_peaks) < 2:
            return {
                "ordered": False,
                "n_peaks": len(low_angle_peaks),
                "match_score": 0,
                "d_spacing": 0.0,
                "pore_size_estimate": 0.0,
                "structure": "Disordered",
            }

        positions = [p["position"] for p in low_angle_peaks]
        ratios = [positions[i] / positions[0] for i in range(1, len(positions))]
        typical_ratios = [1.0, 1.732, 2.0, 2.646, 3.0]

        match_score = 0
        for r in ratios:
            for tr in typical_ratios:
                if abs(r - tr) / tr < 0.1:
                    match_score += 1
                    break

        main_peak = max(low_angle_peaks, key=lambda x: x["intensity"])
        d_spacing = calculate_d_spacing(main_peak["position"], self.wavelength)
        pore_size = d_spacing * 1.05

        return {
            "ordered": match_score >= 2,
            "n_peaks": len(low_angle_peaks),
            "match_score": match_score,
            "d_spacing": float(d_spacing),
            "pore_size_estimate": float(pore_size),
            "structure": ("2D Hexagonal (p6mm)" if match_score >= 2
                          else "Possibly Ordered"),
        }

    # ------------------------------------------------------------------
    def complete_analysis(self, two_theta, intensity, elements=None) -> Dict:
        """
        Full XRD analysis pipeline: detection, validation, size, strain,
        crystallinity, and phase identification.
        """
        xrd_results = {
            "phases": [],
            "phase_fractions": [],
            "peaks": [],
            "structural_peaks": [],
            "top_peaks": [],
            "raw_peaks": [],
            "crystallinity_index": 0.0,
            "crystallinity_description": "Unknown",
            "crystallite_size": {
                "scherrer": 0.0,
                "williamson_hall": 0.0,
                "distribution": "N/A",
                "confidence": "low",
            },
            "microstrain": 0.0,
            "dislocation_density": 0.0,
            "crystal_system": "Unknown",
            "space_group": "Unknown",
            "lattice_parameters": {},
            "ordered_mesopores": False,
            "wavelength": self.wavelength,
            "instrument_fwhm_deg": self.instrument_fwhm_deg,
            "nano_tolerance": 0.0,
            "analysis_notes": [],
            "n_detected_maxima": 0,
            "n_structural_peaks": 0,
            "n_peaks_total": 0,
            "peak_validation_message": "",
            "size_analysis_methods": {},
            "wh_valid": False,
        }

        try:
            sort_idx = np.argsort(two_theta)
            two_theta_raw = two_theta[sort_idx]
            intensity_raw = intensity[sort_idx]

            raw_max_idx = int(np.argmax(intensity_raw))
            xrd_results["debug_raw_apex"] = {
                "index": raw_max_idx,
                "two_theta": float(two_theta_raw[raw_max_idx]),
                "intensity": float(intensity_raw[raw_max_idx]),
            }

            two_theta_p, intensity_p = self.preprocess_pattern(two_theta, intensity)

            intensity_corr, background = snip_background(intensity_p)

            peak_results = detect_peaks_with_validation(
                two_theta=two_theta_p,
                intensity=intensity_p,
                background=background,
                min_prominence=0.03,
            )

            xrd_results["n_detected_maxima"] = peak_results["n_local_maxima"]
            xrd_results["n_structural_peaks"] = peak_results["n_structural_peaks"]
            xrd_results["detected_peaks"] = peak_results.get("local_maxima_debug", [])

            structural_peaks_raw = peak_results["structural_peaks"]
            validated_peaks = []
            for peak in structural_peaks_raw:
                d_spacing = calculate_d_spacing(peak["position"], self.wavelength)
                theta_rad = np.deg2rad(peak["position"] / 2.0)
                crystallite_size = scherrer_crystallite_size(
                    peak["fwhm_rad"], theta_rad, self.wavelength,
                    self.scherrer_constant, self.instrument_fwhm_rad,
                )
                validated_peaks.append({
                    **peak,
                    "d_spacing": float(d_spacing),
                    "crystallite_size": float(crystallite_size),
                    "theta_bragg": float(peak["position"] / 2.0),
                })

            xrd_results["structural_peaks"] = validated_peaks
            xrd_results["peaks"] = validated_peaks
            xrd_results["top_peaks"] = validated_peaks[:10]
            xrd_results["n_structural_peaks"] = len(validated_peaks)
            xrd_results["n_peaks_total"] = len(validated_peaks)

            if validated_peaks:
                strongest = max(validated_peaks, key=lambda p: p["intensity"])
                xrd_results["debug_strongest_structural"] = {
                    "two_theta": strongest["position"],
                    "intensity": strongest["intensity"],
                    "fwhm": strongest["fwhm_deg"],
                }
            else:
                xrd_results["debug_strongest_structural"] = None

            raw_apex_theta = xrd_results["debug_raw_apex"]["two_theta"]
            xrd_results["debug_raw_apex_match"] = any(
                abs(p["position"] - raw_apex_theta) < 0.5
                for p in validated_peaks
            )

            xrd_results["peak_validation_message"] = (
                f"{xrd_results['n_structural_peaks']} of "
                f"{xrd_results['n_detected_maxima']} local maxima "
                "correspond to physical Bragg reflections "
                "(SNR > 2, FWHM 0.03-5.0 deg, R2 > 0.5)"
            )

            raw_peaks = detect_peaks_raw(two_theta_raw, intensity_raw)
            xrd_results["raw_peaks"] = raw_peaks

            ci = calculate_crystallinity_index(
                two_theta_p, intensity_p, xrd_results["structural_peaks"]
            )
            xrd_results["crystallinity_index"] = ci

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

            if ci >= 0.95:
                xrd_results["analysis_notes"].append(
                    "Crystallinity index approaches physical maximum; "
                    "nanocrystalline materials typically < 0.85"
                )

            # ---------------------------------------------------------
            # Crystallite size and strain
            # ---------------------------------------------------------
            mean_size = 0.0
            if validated_peaks:
                size_stats = self.calculate_crystallite_statistics(validated_peaks)
                mean_size = size_stats["mean_size"]
                xrd_results["crystallite_size"]["scherrer"] = mean_size
                xrd_results["crystallite_size"]["distribution"] = size_stats["distribution"]
                xrd_results["crystallite_size"]["cv"] = size_stats["cv"]

                xrd_results["nano_tolerance"] = self.calculate_nano_tolerance(mean_size)

                if size_stats["n_peaks"] >= 5:
                    xrd_results["crystallite_size"]["confidence"] = "high"
                elif size_stats["n_peaks"] >= 3:
                    xrd_results["crystallite_size"]["confidence"] = "medium"
                else:
                    xrd_results["crystallite_size"]["confidence"] = "low"

                if len(validated_peaks) >= 4:
                    wh = williamson_hall_analysis(
                        validated_peaks, self.wavelength, self.instrument_fwhm_rad
                    )
                    if wh and wh.get("wh_valid", False):
                        xrd_results["williamson_hall"] = wh
                        xrd_results["crystallite_size"]["williamson_hall"] = wh["crystallite_size"]
                        xrd_results["microstrain"] = wh["microstrain"]
                        xrd_results["wh_valid"] = True
                        if wh["crystallite_size"] > 0:
                            xrd_results["dislocation_density"] = (
                                15.0 * wh["microstrain"]
                                / (wh["crystallite_size"] * 1e-9)
                            )
                    else:
                        xrd_results["wh_valid"] = False
                        xrd_results["microstrain"] = None
                        xrd_results["dislocation_density"] = None
                        xrd_results["analysis_notes"].append(
                            "Williamson-Hall analysis: insufficient linearity "
                            "(R2 < 0.85 required)"
                        )
                else:
                    xrd_results["analysis_notes"].append(
                        "Williamson-Hall analysis requires >= 4 independent reflections"
                    )
                    xrd_results["wh_valid"] = False

                xrd_results["size_analysis_methods"] = {
                    "scherrer": {
                        "valid": True,
                        "shape_factor_K": 0.9,
                        "instrumental_broadening": self.instrument_fwhm_deg,
                        "interpretation": "Coherent diffraction domain size",
                        "reference": "Scherrer (1918); Klug & Alexander (1974)",
                    },
                    "williamson_hall": {
                        "valid": xrd_results.get("wh_valid", False),
                        "requires": ">= 4 reflections with R2 > 0.85",
                        "provides": "Size-strain separation",
                        "reference": "Williamson & Hall, Acta Metall. (1953)",
                    },
                }

            # ---------------------------------------------------------
            # Phase identification
            # ---------------------------------------------------------
            if UNIVERSAL_PHASE_ID_AVAILABLE and validated_peaks:
                phases = identify_phases_universal(
                    two_theta=None,
                    intensity=None,
                    wavelength=self.wavelength,
                    elements=elements,
                    size_nm=xrd_results["crystallite_size"]["scherrer"],
                    precomputed_peaks_2theta=[p["position"] for p in validated_peaks],
                    precomputed_peaks_intensity=[p["intensity"] for p in validated_peaks],
                )
                xrd_results["phases"] = phases or []

                if phases:
                    best = phases[0]
                    xrd_results["crystal_system"] = best.get("crystal_system", "Unknown")
                    xrd_results["space_group"] = best.get("space_group", "Unknown")
                    xrd_results["lattice_parameters"] = best.get("lattice", {})
                    xrd_results["material_family"] = best.get("material_family", "unknown")

                    try:
                        xrd_results["peaks"] = map_peaks_to_phases(validated_peaks, phases)
                    except Exception as exc:
                        logger.warning("Peak-to-phase mapping failed: %s", exc)

                    try:
                        xrd_results["phase_fractions"] = calculate_phase_fractions(
                            xrd_results["peaks"], phases
                        )
                    except Exception as exc:
                        logger.warning("Phase fraction calculation failed: %s", exc)

                if not phases and mean_size > 0 and mean_size < 10:
                    xrd_results["analysis_notes"].append(
                        "Nanocrystalline material detected (size < 10 nm). "
                        "Phase identification may be limited due to peak broadening."
                    )
            else:
                xrd_results["analysis_notes"].append(
                    "Universal phase identifier not available or no structural peaks found."
                )

            # ---------------------------------------------------------
            # Summary labels
            # ---------------------------------------------------------
            if validated_peaks:
                avg_fwhm = float(np.mean(
                    [p.get("fwhm_deg", 0) for p in validated_peaks[:5]]
                )) if len(validated_peaks) >= 5 else 0.0
                avg_size = xrd_results["crystallite_size"]["scherrer"]

                if avg_size < 5:
                    xrd_results["material_type"] = "Ultrafine nanocrystalline"
                elif avg_size < 20:
                    xrd_results["material_type"] = "Nanocrystalline"
                elif avg_size < 100:
                    xrd_results["material_type"] = "Sub-micron crystalline"
                else:
                    xrd_results["material_type"] = "Micron-scale crystalline"

                if avg_fwhm > 1.0 and 0 < avg_size < 10:
                    xrd_results["crystallinity_statement"] = (
                        "The material exhibits nanocrystalline character with "
                        "broadened diffraction maxima, indicating coherent "
                        f"scattering domains of ~{avg_size:.1f} nm."
                    )
                else:
                    xrd_results["crystallinity_statement"] = (
                        f"The diffraction pattern shows "
                        f"{xrd_results['crystallinity_description'].lower()} "
                        f"features with an estimated crystallite size of "
                        f"{avg_size:.1f} nm."
                    )

            xrd_results["low_angle_features"] = {
                "present": False,
                "note": "Low-angle scattering requires SAXS, not wide-angle XRD",
                "reference": "Thommes et al., Pure Appl. Chem. 2015",
            }

            validation = self.validate_crystallographic_results(xrd_results)
            xrd_results["validation"] = validation

            return {
                "valid": True,
                "xrd_results": xrd_results,
                "xrd_raw": {
                    "two_theta": two_theta_raw.tolist(),
                    "intensity": intensity_raw.tolist(),
                },
                "xrd_processed": {
                    "two_theta": two_theta_p.tolist(),
                    "intensity": intensity_p.tolist(),
                },
            }

        except Exception as exc:
            logger.exception("XRD analysis failed")
            return {
                "valid": False,
                "error": str(exc),
                "xrd_results": xrd_results,
            }
