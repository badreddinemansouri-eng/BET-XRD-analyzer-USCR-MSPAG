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


# -------------------------------
# Physical peak validator
# -------------------------------
class PhysicalPeakValidator:
    def __init__(self, instrument: InstrumentProfile):
        self.instrument = instrument

    def validate(self, idx, two_theta, intensity, background):
        """
        Returns a peak dictionary if VALID
        Returns None if NOT physical
        """

        # ---------------------------
        # Local window around peak
        # ---------------------------
        window = 10
        left = max(0, idx - window)
        right = min(len(two_theta), idx + window)

        x = two_theta[left:right]
        y = intensity[left:right] - background[left:right]

        if len(x) < 6:
            return None

        peak_height = y.max()
        noise = np.std(y)

        # ---------------------------
        # Rule 1: Signal-to-noise
        # ---------------------------
        if noise <= 0 or peak_height / noise < 3:
            return None

        # ---------------------------
        # Estimate FWHM
        # ---------------------------
        half_max = peak_height / 2
        above = np.where(y >= half_max)[0]

        if len(above) < 2:
            return None

        fwhm = x[above[-1]] - x[above[0]]

        if fwhm < self.instrument.min_fwhm:
            return None

        if fwhm > self.instrument.max_fwhm:
            return None

        # ---------------------------
        # Rule 2: Peak shape fit
        # ---------------------------
        try:
            popt_g, _ = curve_fit(
                gaussian, x, y,
                p0=[peak_height, two_theta[idx], fwhm]
            )
            fit_g = gaussian(x, *popt_g)
            r2_g = 1 - np.sum((y - fit_g)**2) / np.sum((y - y.mean())**2)
        except:
            r2_g = 0

        try:
            popt_l, _ = curve_fit(
                lorentzian, x, y,
                p0=[peak_height, two_theta[idx], fwhm]
            )
            fit_l = lorentzian(x, *popt_l)
            r2_l = 1 - np.sum((y - fit_l)**2) / np.sum((y - y.mean())**2)
        except:
            r2_l = 0

        if max(r2_g, r2_l) < 0.85:
            return None

        shape = "gaussian" if r2_g >= r2_l else "lorentzian"

        # ---------------------------
        # VALID PEAK
        # ---------------------------
        return {
            "two_theta": float(two_theta[idx]),
            "intensity": float(peak_height),
            "fwhm_deg": float(fwhm),
            "snr": float(peak_height / noise),
            "shape": shape
        }
