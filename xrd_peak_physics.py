"""
Physical peak validation and peak-shape fitting for XRD patterns.

This module provides:
    1. InstrumentProfile - defines wavelength and FWHM limits
    2. Gaussian, Lorentzian, and pseudo-Voigt peak shape models
    3. PhysicalPeakValidator - accepts or rejects candidate peaks based on
       SNR, area fraction, FWHM limits, and R-squared of the best fit

References:
    1. Klug & Alexander, X-Ray Diffraction Procedures, 2nd ed. (1974).
    2. Wertheim, G. K. et al., Rev. Sci. Instrum. 45 (1974) 1369.
    3. Ida, T. et al., J. Appl. Cryst. 33 (2000) 1311.
"""

import logging
from typing import Dict, Optional

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instrument profile
# ---------------------------------------------------------------------------
class InstrumentProfile:
    """
    Instrumental parameters used by the validator.

    Parameters
    ----------
    wavelength : float
        X-ray wavelength in Angstroms. Default Cu K-alpha.
    min_fwhm_deg : float
        Physical lower bound on peak FWHM (deg). Below this -> noise.
    max_fwhm_deg : float
        Physical upper bound (deg). Above this -> amorphous.
    """

    def __init__(self,
                 wavelength: float = 1.5406,
                 min_fwhm_deg: float = 0.05,
                 max_fwhm_deg: float = 5.0):
        self.wavelength = wavelength
        self.min_fwhm = min_fwhm_deg
        self.max_fwhm = max_fwhm_deg


# ---------------------------------------------------------------------------
# Peak-shape models
# ---------------------------------------------------------------------------
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def lorentzian(x, a, x0, gamma):
    return a * gamma ** 2 / ((x - x0) ** 2 + gamma ** 2)


def pseudo_voigt(x, a, x0, sigma, eta):
    g = np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    l = sigma ** 2 / ((x - x0) ** 2 + sigma ** 2)
    return a * (eta * l + (1 - eta) * g)


# ---------------------------------------------------------------------------
# Peak validator
# ---------------------------------------------------------------------------
class PhysicalPeakValidator:
    """
    Validates candidate peaks using physical and statistical criteria.

    A peak passes if:
        - SNR > 2
        - Area / local area > 0.01
        - FWHM within [min_fwhm, max_fwhm]
        - Best fit (Gaussian/Lorentzian/pseudo-Voigt) has R-squared >= 0.5

    Returns a dict with all fitted parameters, or None if the candidate
    should be rejected.
    """

    def __init__(self, instrument: InstrumentProfile):
        self.instrument = instrument

    def validate(self, idx: int, two_theta, intensity, background,
                 window: int = 25) -> Optional[Dict]:
        left = max(0, idx - window)
        right = min(len(two_theta), idx + window)

        x = two_theta[left:right]
        y = intensity[left:right] - background[left:right]

        if len(x) < 6:
            return None

        # Recenter on true local maximum within window
        local_max_idx = int(np.argmax(y))
        idx = left + local_max_idx
        peak_height = float(y[local_max_idx])

        noise = float(np.std(y))
        if noise <= 0 or peak_height / noise < 2.0:
            return None

        peak_area = float(np.trapz(y[y > 0], x[y > 0])) if np.any(y > 0) else 0.0
        total_local_area = float(np.trapz(np.abs(y), x))
        if total_local_area <= 0:
            return None
        if peak_area / total_local_area < 0.01:
            return None

        half_max = peak_height / 2.0
        above = np.where(y >= half_max)[0]
        if len(above) < 4:
            return None
        fwhm = float(x[above[-1]] - x[above[0]])

        if fwhm < self.instrument.min_fwhm:
            return None
        if fwhm > self.instrument.max_fwhm:
            return None

        sigma_g = fwhm / 2.3548
        gamma_l = fwhm / 2.0
        sigma_pv = fwhm / 2.2

        r2_g = self._try_fit(
            gaussian, x, y,
            p0=[peak_height, two_theta[idx], sigma_g],
            bounds=([0, x[0], 0.01], [peak_height * 2, x[-1], fwhm * 2]),
            maxfev=2000
        )
        r2_l = self._try_fit(
            lorentzian, x, y,
            p0=[peak_height, two_theta[idx], gamma_l],
            bounds=([0, x[0], 0.01], [peak_height * 2, x[-1], fwhm * 2]),
            maxfev=2000
        )
        r2_pv, popt_pv = self._try_fit_with_popt(
            pseudo_voigt, x, y,
            p0=[peak_height, two_theta[idx], sigma_pv, 0.5],
            bounds=([0, x[0], 0.01, 0], [peak_height * 2, x[-1], fwhm * 2, 1]),
            maxfev=3000
        )

        r2_scores = [r2_g, r2_l, r2_pv]
        shapes = ["gaussian", "lorentzian", "pseudo_voigt"]
        best_idx = int(np.argmax(r2_scores))
        best_r2 = r2_scores[best_idx]
        best_fit = shapes[best_idx]

        if best_r2 < 0.5:
            return None

        # Determine final peak position from the best-fit model
        if best_fit == "pseudo_voigt" and popt_pv is not None:
            peak_pos = float(popt_pv[1])
        else:
            peak_pos = float(two_theta[idx])

        return {
            "two_theta": peak_pos,
            "index": int(idx),
            "intensity": peak_height,
            "fwhm_deg": fwhm,
            "snr": float(peak_height / noise) if noise > 0 else np.inf,
            "shape": best_fit,
            "fit_quality": float(best_r2),
            "area": peak_area,
        }

    # ------------------------------------------------------------------
    # Fit helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _try_fit(func, x, y, p0, bounds, maxfev):
        try:
            popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
            fit = func(x, *popt)
            ss_res = float(np.sum((y - fit) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _try_fit_with_popt(func, x, y, p0, bounds, maxfev):
        try:
            popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
            fit = func(x, *popt)
            ss_res = float(np.sum((y - fit) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            return r2, popt
        except Exception:
            return 0.0, None
