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
    
        window = 25
        left = max(0, idx - window)
        right = min(len(two_theta), idx + window)
    
        x = two_theta[left:right]
        y = intensity[left:right] - background[left:right]
    
        if len(x) < 6:
            return None
    
        # =================================================
        # 🔥 FIX 1 — TRUE PEAK RECENTERING (MANDATORY)
        # =================================================
        local_max_idx = np.argmax(y)
        idx = left + local_max_idx          # 🔥 overwrite idx
        peak_height = y[local_max_idx]
    
        noise = np.std(y)
        if noise <= 0 or peak_height / noise < 2:
            return None
    
        # =================================================
        # 🔥 FIX 2 — AREA TEST (NO SPECIAL PEAKS)
        # =================================================
        peak_area = np.trapz(y[y > 0], x[y > 0])
        total_local_area = np.trapz(np.abs(y), x)
    
        if total_local_area <= 0:
            return None
    
        if peak_area / total_local_area < 0.015:
            return None
    
        # =================================================
        # 🔥 FIX 3 — FWHM AROUND TRUE APEX
        # =================================================
        half_max = peak_height / 2
        above = np.where(y >= half_max)[0]
    
        if len(above) < 4:
            return None
    
        fwhm = x[above[-1]] - x[above[0]]
    
        if fwhm < self.instrument.min_fwhm:
            return None
    
        if fwhm > self.instrument.max_fwhm:
            return None
    
        # =================================================
        # 🔥 FIX 4 — FIT CENTER USES RECENTERED idx
        # =================================================
        try:
            popt_g, _ = curve_fit(
                gaussian, x, y,
                p0=[peak_height, two_theta[idx], fwhm],
                maxfev=2000
            )
            fit_g = gaussian(x, *popt_g)
            r2_g = 1 - np.sum((y - fit_g)**2) / np.sum((y - y.mean())**2)
        except:
            r2_g = 0
    
        try:
            popt_l, _ = curve_fit(
                lorentzian, x, y,
                p0=[peak_height, two_theta[idx], fwhm],
                maxfev=2000
            )
            fit_l = lorentzian(x, *popt_l)
            r2_l = 1 - np.sum((y - fit_l)**2) / np.sum((y - y.mean())**2)
        except:
            r2_l = 0
    
        try:
            popt_pv, _ = curve_fit(
                pseudo_voigt, x, y,
                p0=[peak_height, two_theta[idx], fwhm / 2.3548, 0.5],
                bounds=([0, x[0], 0.01, 0], [peak_height * 2, x[-1], fwhm * 2, 1]),
                maxfev=3000
            )
            fit_pv = pseudo_voigt(x, *popt_pv)
            r2_pv = 1 - np.sum((y - fit_pv)**2) / np.sum((y - y.mean())**2)
        except:
            r2_pv = 0
    
        best_r2 = max(r2_g, r2_l, r2_pv)
        best_fit = ["gaussian", "lorentzian", "pseudo_voigt"][
            np.argmax([r2_g, r2_l, r2_pv])
        ]
    
        if best_r2 < 0.65:
            return None
        # =================================================
        # 🔥 FIX 5 — CONTINUOUS APEX (SUB-GRID TRUE θ)
        # =================================================
        if best_fit == "gaussian":
            peak_pos = popt_g[1]
        elif best_fit == "lorentzian":
            peak_pos = popt_l[1]
        elif best_fit == "pseudo_voigt":
            peak_pos = popt_pv[1]
        else:
            peak_pos = two_theta[idx]

        # =================================================
        # ✅ SAME TERMS — CORRECT VALUES
        # =================================================
        return {
            "two_theta": float(peak_pos),   # TRUE apex
            "index": int(idx),                   # TRUE index
            "intensity": float(peak_height),
            "fwhm_deg": float(fwhm),
            "snr": float(peak_height / noise),
            "shape": best_fit,
            "fit_quality": float(best_r2),
            "area": float(peak_area),
        }

