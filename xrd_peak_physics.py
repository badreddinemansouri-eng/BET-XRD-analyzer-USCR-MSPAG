import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# -------------------------------
# Instrument definition (universal)
# -------------------------------
class InstrumentProfile:
    def __init__(
        self,
        wavelength=1.5406,        # Cu Kα default
        min_fwhm_deg=0.05,        # physical minimum
        max_fwhm_deg=5.0          # amorphous limit
    ):
        self.wavelength = wavelength
        self.min_fwhm = min_fwhm_deg
        self.max_fwhm = max_fwhm_deg


# -------------------------------
# Peak shape models
# -------------------------------
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def lorentzian(x, a, x0, gamma):
    return a * gamma**2 / ((x - x0)**2 + gamma**2)


def pseudo_voigt(x, a, x0, sigma, eta):
    """Pseudo-Voigt function for nanocrystalline materials"""
    gaussian_part = np.exp(-(x - x0)**2 / (2 * sigma**2))
    lorentzian_part = sigma**2 / ((x - x0)**2 + sigma**2)
    return a * (eta * lorentzian_part + (1 - eta) * gaussian_part)


# -------------------------------
# Physical peak validator (NANOMATERIAL-OPTIMIZED)
# -------------------------------
class PhysicalPeakValidator:
    def __init__(self, instrument: InstrumentProfile):
        self.instrument = instrument

    def validate(self, idx, two_theta, intensity, background):
        """
        Returns a peak dictionary if VALID
        Returns None if NOT physical
        
        NANOMATERIAL-OPTIMIZED CHANGES:
        1. Larger window for broad peaks (10 → 25 points)
        2. Lower SNR threshold (3 → 2)
        3. Lower R² threshold (0.85 → 0.65)
        4. Added pseudo-Voigt shape option
        5. ADDED PHYSICS-BASED NOISE REJECTION
        """
        # ---------------------------
        # Local window around peak (LARGER for nanomaterials)
        # ---------------------------
        window = 25  # INCREASED from 10 for broad nanocrystalline peaks
        left = max(0, idx - window)
        right = min(len(two_theta), idx + window)

        x = two_theta[left:right]
        y = intensity[left:right] - background[left:right]

        if len(x) < 6:
            return None

        peak_height = y.max()
        noise = np.std(y)

        # ---------------------------
        # Rule 1: Signal-to-noise (RELAXED for nanomaterials)
        # ---------------------------
        if noise <= 0 or peak_height / noise < 2:  # CHANGED from 3 to 2
            return None

        # ---------------------------
        # SCIENTIFIC NOISE REJECTION (CRITICAL)
        # A physical Bragg peak must carry significant integrated intensity
        # Reference: Cullity & Stock, Elements of X-ray Diffraction
        # ---------------------------
        peak_area = np.trapz(y[y > 0], x[y > 0])
        total_local_area = np.trapz(np.abs(y), x)

        # Reject noise oscillations (area too small)
        if total_local_area <= 0:
            return None

        # Nanocrystalline-safe threshold
        if peak_area / total_local_area < 0.015:
            return None


        # ---------------------------
        # Estimate FWHM
        # ---------------------------
        half_max = peak_height / 2
        above = np.where(y >= half_max)[0]

        # ANGULAR COHERENCE CHECK
        # Physical diffraction peaks must span multiple contiguous points
        if len(above) < 4:
            return None

        fwhm = x[above[-1]] - x[above[0]]

        if fwhm < self.instrument.min_fwhm:
            return None

        if fwhm > self.instrument.max_fwhm:
            return None

        # ---------------------------
        # Rule 2: Peak shape fit (RELAXED for nanomaterials)
        # ---------------------------
        try:
            popt_g, _ = curve_fit(
                gaussian, x, y,
                p0=[peak_height, two_theta[idx], fwhm],
                maxfev=1000
            )
            fit_g = gaussian(x, *popt_g)
            r2_g = 1 - np.sum((y - fit_g)**2) / np.sum((y - y.mean())**2)
        except:
            r2_g = 0

        try:
            popt_l, _ = curve_fit(
                lorentzian, x, y,
                p0=[peak_height, two_theta[idx], fwhm],
                maxfev=1000
            )
            fit_l = lorentzian(x, *popt_l)
            r2_l = 1 - np.sum((y - fit_l)**2) / np.sum((y - y.mean())**2)
        except:
            r2_l = 0

        # Try pseudo-Voigt for nanocrystalline materials
        try:
            popt_pv, _ = curve_fit(
                pseudo_voigt, x, y,
                p0=[peak_height, two_theta[idx], fwhm/2.3548, 0.5],
                bounds=([0, x[0], 0.01, 0], [peak_height*2, x[-1], fwhm*2, 1]),
                maxfev=2000
            )
            fit_pv = pseudo_voigt(x, *popt_pv)
            r2_pv = 1 - np.sum((y - fit_pv)**2) / np.sum((y - y.mean())**2)
        except:
            r2_pv = 0

        # Determine best fit
        fit_scores = {
            'gaussian': r2_g,
            'lorentzian': r2_l,
            'pseudo_voigt': r2_pv
        }
        
        best_fit = max(fit_scores, key=fit_scores.get)
        best_r2 = fit_scores[best_fit]

        # RELAXED threshold for nanomaterials
        if best_r2 < 0.65:  # CHANGED from 0.85 to 0.65
            return None

        shape = best_fit

        # ---------------------------
        # VALID PEAK
        # ---------------------------
        return {
            "two_theta": float(two_theta[idx]),
            "intensity": float(peak_height),
            "fwhm_deg": float(fwhm),
            "snr": float(peak_height / noise),
            "shape": shape,
            "fit_quality": float(best_r2),
            "area": float(peak_area)  # Added for crystallinity calculation
        }
