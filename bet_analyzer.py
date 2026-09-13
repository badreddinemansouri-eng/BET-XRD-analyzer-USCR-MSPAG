"""
IUPAC-COMPLIANT BET ANALYSIS ENGINE
========================================================================
Scientific implementation following:
1. Rouquerol et al., Pure Appl. Chem., 1994, 66, 1739-1758
2. Thommes et al., Pure Appl. Chem., 2015, 87, 1051-1069
3. Harkins & Jura, J. Am. Chem. Soc., 1944, 66, 1366 (t-plot)
4. Barrett, Joyner & Halenda, J. Am. Chem. Soc., 1951, 73, 373 (BJH)
========================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize, integrate
from typing import Dict, Tuple, List, Optional, Any
import warnings
import io

warnings.filterwarnings('ignore')


def custom_trapz(y, x=None, dx=1.0):
    """Trapezoidal integration for compatibility"""
    y = np.asarray(y)
    if x is None:
        return np.sum((y[1:] + y[:-1]) / 2.0) * dx
    else:
        x = np.asarray(x)
        return np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(x))


# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================
PHYSICAL_CONSTANTS = {
    'N2': {
        'cross_section': 0.162e-18,
        'liquid_density': 0.808,
        'molar_volume': 34.7e-6,
        'surface_tension': 8.85e-3,
        'temperature': 77.3
    },
    'Ar': {
        'cross_section': 0.142e-18,
        'liquid_density': 1.40,
        'molar_volume': 28.4e-6,
        'surface_tension': 11.9e-3,
        'temperature': 87.3
    },
    'CO2': {
        'cross_section': 0.187e-18,
        'liquid_density': 1.03,
        'molar_volume': 42.9e-6,
        'surface_tension': 5.9e-3,
        'temperature': 273.15
    }
}

AVOGADRO = 6.02214076e23
GAS_CONSTANT = 8.314462618


# ============================================================================
# ASAP 2420 DATA EXTRACTION
# ============================================================================
def extract_asap2420_data(file, preview_only=False):
    """
    Extract data from Micromeritics ASAP 2420 files

    Returns:
    --------
    p_ads, q_ads, p_des, q_des, psd_data, message
    """
    try:
        filename = file.name.lower()

        if filename.endswith('.csv'):
            df = pd.read_csv(file, header=None)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl', header=None)
        elif filename.endswith('.xls'):
            df = pd.read_excel(file, engine='xlrd', header=None)
        else:
            content = file.read().decode('utf-8', errors='ignore')
            file.seek(0)
            for delimiter in [',', '\t', ';', ' ']:
                try:
                    df = pd.read_csv(io.StringIO(content), delimiter=delimiter, header=None)
                    break
                except Exception:
                    continue

        if preview_only:
            p_vals = []
            for col in range(min(20, df.shape[1])):
                col_data = pd.to_numeric(df.iloc[28:100, col], errors='coerce').dropna()
                if len(col_data) > 10:
                    vals = col_data.values
                    if np.all((vals >= 0) & (vals <= 1.2)):
                        p_vals.extend(vals)

            if p_vals:
                p_arr = np.array(p_vals)
                return {
                    'format': 'ASAP 2420 (detected)',
                    'n_points': len(p_arr),
                    'p_range': (float(p_arr.min()), float(p_arr.max()))
                }
            return {'format': 'Unknown', 'n_points': 0, 'p_range': (0, 0)}

        p_ads, q_ads = [], []
        p_des, q_des = [], []

        for i in range(28, min(200, len(df))):
            try:
                if df.shape[1] > 12:
                    p_val = df.iloc[i, 11]
                    q_val = df.iloc[i, 12]
                    if pd.notna(p_val) and pd.notna(q_val):
                        p_float = float(p_val)
                        q_float = float(q_val)
                        if 0.001 < p_float < 0.999 and q_float > 0:
                            p_ads.append(p_float)
                            q_ads.append(q_float)

                if df.shape[1] > 14:
                    p_val_des = df.iloc[i, 13]
                    q_val_des = df.iloc[i, 14]
                    if pd.notna(p_val_des) and pd.notna(q_val_des):
                        p_des_float = float(p_val_des)
                        q_des_float = float(q_val_des)
                        if 0.001 < p_des_float < 0.999 and q_des_float > 0:
                            p_des.append(p_des_float)
                            q_des.append(q_des_float)
            except Exception:
                continue

        if len(p_ads) >= 10:
            psd_data = extract_psd_data(df)
            return (np.array(p_ads), np.array(q_ads),
                    np.array(p_des) if len(p_des) > 0 else None,
                    np.array(q_des) if len(q_des) > 0 else None,
                    psd_data, "ASAP 2420 format")

        df_numeric = df.apply(pd.to_numeric, errors='coerce')

        for col in range(min(10, df_numeric.shape[1])):
            col_data = df_numeric.iloc[:, col].dropna()
            if len(col_data) > 10:
                sample = col_data.values[:20]
                if np.all((sample >= 0) & (sample <= 1.1)):
                    if col + 1 < df_numeric.shape[1]:
                        q_data = df_numeric.iloc[:, col + 1].dropna()
                        if len(q_data) > 10:
                            p_ads = col_data.values
                            q_ads = q_data.values

                            if col + 3 < df_numeric.shape[1]:
                                p_des_data = df_numeric.iloc[:, col + 2].dropna()
                                q_des_data = df_numeric.iloc[:, col + 3].dropna()
                                if len(p_des_data) > 5:
                                    p_des = p_des_data.values
                                    q_des = q_des_data.values

                            psd_data = extract_psd_data(df)
                            return (p_ads, q_ads,
                                    p_des if len(p_des) > 0 else None,
                                    q_des if len(q_des) > 0 else None,
                                    psd_data, "Auto-detected format")

        df_clean = df.apply(pd.to_numeric, errors='coerce')
        df_clean = df_clean.dropna(axis=1, how='all')

        if df_clean.shape[1] >= 2:
            p_ads = df_clean.iloc[:, 0].dropna().values
            q_ads = df_clean.iloc[:, 1].dropna().values

            if df_clean.shape[1] >= 4:
                p_des = df_clean.iloc[:, 2].dropna().values
                q_des = df_clean.iloc[:, 3].dropna().values

            if len(p_ads) >= 5:
                psd_data = extract_psd_data(df)
                return (p_ads, q_ads,
                        p_des if p_des is not None and len(p_des) > 0 else None,
                        q_des if q_des is not None and len(q_des) > 0 else None,
                        psd_data, "Simple column extraction")

        return None, None, None, None, None, f"Insufficient data: {len(p_ads)} points"

    except Exception as e:
        return None, None, None, None, None, f"Extraction error: {str(e)}"


def extract_psd_data(df):
    """Extract PSD data from ASAP file"""
    try:
        pore_diameters, dv_dlogd = [], []

        for i in range(28, min(100, len(df))):
            try:
                if df.shape[1] > 61:
                    d_val = df.iloc[i, 60]
                    v_val = df.iloc[i, 61]

                    if pd.notna(d_val) and pd.notna(v_val):
                        d_float = float(d_val)
                        v_float = float(v_val)
                        if d_float > 0 and not np.isnan(v_float):
                            pore_diameters.append(d_float)
                            dv_dlogd.append(v_float)
            except Exception:
                continue

        if len(pore_diameters) >= 3:
            return {
                'pore_diameters': np.array(pore_diameters),
                'dv_dlogd': np.array(dv_dlogd)
            }
    except Exception:
        pass

    return None


# ============================================================================
# IUPAC BET ANALYZER CLASS
# ============================================================================
class IUPACBETAnalyzer:
    """
    IUPAC-compliant physisorption analysis

    Implements:
    1. BET surface area with Rouquerol criteria
    2. t-plot analysis for microporosity
    3. BJH pore size distribution
    4. Hysteresis classification
    5. Complete error propagation
    """

    def __init__(self, p_ads: np.ndarray, q_ads: np.ndarray,
                 p_des: Optional[np.ndarray] = None,
                 q_des: Optional[np.ndarray] = None,
                 cross_section: float = 0.162e-18,
                 temperature: float = 77.3):
        self.p_ads = np.asarray(p_ads, dtype=np.float64)
        self.q_ads = np.asarray(q_ads, dtype=np.float64)
        self.p_des = np.asarray(p_des, dtype=np.float64) if p_des is not None else None
        self.q_des = np.asarray(q_des, dtype=np.float64) if q_des is not None else None

        valid_ads = (self.p_ads > 0) & (self.p_ads < 1) & (self.q_ads > 0)
        self.p_ads = self.p_ads[valid_ads]
        self.q_ads = self.q_ads[valid_ads]

        if self.p_des is not None and self.q_des is not None:
            valid_des = (self.p_des > 0) & (self.p_des < 1) & (self.q_des > 0)
            self.p_des = self.p_des[valid_des]
            self.q_des = self.q_des[valid_des]

        ads_sort_idx = np.argsort(self.p_ads)
        self.p_ads = self.p_ads[ads_sort_idx]
        self.q_ads = self.q_ads[ads_sort_idx]

        if self.p_des is not None and self.q_des is not None:
            des_sort_idx = np.argsort(self.p_des)
            self.p_des = self.p_des[des_sort_idx]
            self.q_des = self.q_des[des_sort_idx]

        self.cross_section = cross_section
        self.temperature = temperature
        self.results = {}

    def _validate_data(self):
        if len(self.p_ads) < 10:
            raise ValueError(f"Insufficient adsorption points: {len(self.p_ads)} (<10)")

        if not np.all(np.diff(self.p_ads) > -1e-10):
            raise ValueError("Adsorption pressure must be monotonic increasing")

        if self.p_ads.max() < 0.1:
            raise ValueError(f"Insufficient pressure range: max P/P₀ = {self.p_ads.max():.3f}")

    def _rouquerol_criteria(self) -> Tuple[int, int]:
        """
        Apply Rouquerol criteria for BET range selection.

        The preferred range is centered near 0.25 P/P₀ (IUPAC recommendation).
        Scoring now rewards linearity, point count, and proximity to 0.25.
        """
        n_points = len(self.p_ads)
        valid_ranges = []

        for i in range(n_points - 4):
            for j in range(i + 4, min(i + 15, n_points)):
                p_seg = self.p_ads[i:j]
                q_seg = self.q_ads[i:j]

                with np.errstate(divide='ignore', invalid='ignore'):
                    y_bet = p_seg / (q_seg * (1 - p_seg))

                valid = np.isfinite(y_bet)
                if np.sum(valid) < 4:
                    continue

                p_seg = p_seg[valid]
                y_bet = y_bet[valid]

                if not np.all(np.diff(y_bet) > -1e-8):
                    continue

                q_corrected = q_seg * (1 - p_seg)
                if not np.all(np.diff(q_corrected) > -1e-8):
                    continue

                try:
                    slope, intercept, r_value, _, _ = stats.linregress(p_seg, y_bet)
                except Exception:
                    continue

                if r_value**2 < 0.999 or slope <= 0 or intercept <= 0:
                    continue

                c_constant = slope / intercept + 1
                if c_constant <= 0:
                    continue

                n_pts = j - i

                # Range centered near 0.25 P/P₀ should be preferred
                center_penalty = 1.0 - min(abs(np.mean(p_seg) - 0.25) / 0.25, 1.0)
                score = (
                    r_value**2 * 0.6
                    + (n_pts / 15.0) * 0.2
                    + center_penalty * 0.2
                )

                valid_ranges.append((i, j, score, r_value**2, n_pts))

        if not valid_ranges:
            raise ValueError("No valid BET range found using Rouquerol criteria")

        valid_ranges.sort(key=lambda x: x[2], reverse=True)
        return valid_ranges[0][0], valid_ranges[0][1]

    def bet_surface_area(self) -> Dict[str, Any]:
        """
        Calculate BET surface area with full-covariance error propagation.
        """
        try:
            self._validate_data()

            try:
                i, j = self._rouquerol_criteria()
            except ValueError as e:
                return {
                    'surface_area': 0.0,
                    'surface_area_error': 0.0,
                    'monolayer_capacity': 0.0,
                    'c_constant': 0.0,
                    'bet_regression': {
                        'slope': 0.0,
                        'intercept': 0.0,
                        'r_squared': 0.0,
                        'std_error': 0.0,
                        'p_min': 0.0,
                        'p_max': 0.0,
                        'n_points': 0
                    },
                    'valid': False,
                    'error': str(e)
                }

            p_bet = self.p_ads[i:j]
            q_bet = self.q_ads[i:j]

            with np.errstate(divide='ignore', invalid='ignore'):
                y_bet = p_bet / (q_bet * (1 - p_bet))

            valid = np.isfinite(y_bet)
            p_bet = p_bet[valid]
            y_bet = y_bet[valid]

            slope, intercept, r_value, _, std_err = stats.linregress(p_bet, y_bet)

            q_mono = 1.0 / (slope + intercept)
            c_constant = slope / intercept + 1.0

            # Correct error propagation using full covariance of (slope, intercept).
            # q_mono = 1 / (slope + intercept)
            # Var(q_mono) = (dq/ds, dq/di) · Cov · (dq/ds, dq/di)^T
            try:
                coeffs, cov = np.polyfit(p_bet, y_bet, 1, cov=True)
                var_slope = cov[0, 0]
                var_intercept = cov[1, 1]
                cov_si = cov[0, 1]

                dq_ds = -1.0 / (slope + intercept) ** 2
                dq_di = -1.0 / (slope + intercept) ** 2

                var_q = (
                    (dq_ds ** 2) * var_slope
                    + (dq_di ** 2) * var_intercept
                    + 2 * dq_ds * dq_di * cov_si
                )
                q_mono_error = float(np.sqrt(max(var_q, 0.0)))
            except Exception:
                dq_ds = -1.0 / (slope + intercept) ** 2
                dq_di = -1.0 / (slope + intercept) ** 2
                q_mono_error = float(np.sqrt(
                    (dq_ds * std_err) ** 2 + (dq_di * std_err) ** 2
                ))

            surface_area = q_mono * AVOGADRO * self.cross_section * 1e-4
            surface_area_error = q_mono_error * AVOGADRO * self.cross_section * 1e-4

            return {
                'surface_area': float(surface_area),
                'surface_area_error': float(surface_area_error),
                'monolayer_capacity': float(q_mono),
                'c_constant': float(c_constant),
                'bet_regression': {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_squared': float(r_value**2),
                    'std_error': float(std_err),
                    'p_min': float(p_bet[0]),
                    'p_max': float(p_bet[-1]),
                    'n_points': len(p_bet)
                },
                'valid': True
            }
        except Exception as e:
            return {
                'surface_area': 0.0,
                'surface_area_error': 0.0,
                'monolayer_capacity': 0.0,
                'c_constant': 0.0,
                'bet_regression': {
                    'slope': 0.0,
                    'intercept': 0.0,
                    'r_squared': 0.0,
                    'std_error': 0.0,
                    'p_min': 0.0,
                    'p_max': 0.0,
                    'n_points': 0
                },
                'valid': False,
                'error': str(e)
            }

    def t_plot_analysis(self) -> Dict[str, Any]:
        """t-plot analysis using Harkins-Jura thickness equation."""
        def thickness_nm(p):
            with np.errstate(divide='ignore', invalid='ignore'):
                return (13.99 / (0.034 - np.log10(p + 1e-10))) ** 0.5 * 0.1

        mask = (self.p_ads >= 0.2) & (self.p_ads <= 0.5)
        if np.sum(mask) < 5:
            return {
                'micropore_volume': 0.0,
                'external_surface': 0.0,
                't_plot_r2': 0.0,
                'valid': False
            }

        p_t = self.p_ads[mask]
        q_t = self.q_ads[mask]
        t = thickness_nm(p_t)

        valid = np.isfinite(t) & np.isfinite(q_t)
        t = t[valid]
        q_t = q_t[valid]

        if len(t) < 5:
            return {
                'micropore_volume': 0.0,
                'external_surface': 0.0,
                't_plot_r2': 0.0,
                'valid': False
            }

        slope, intercept, r_value, _, _ = stats.linregress(t, q_t)

        external_surface = slope * 15.47
        micropore_volume = max(0.0, intercept * 0.001548)

        return {
            'micropore_volume': float(micropore_volume),
            'external_surface': float(external_surface),
            't_plot_r2': float(r_value**2),
            't_range': (float(t.min()), float(t.max())),
            'n_points': len(t),
            'valid': True
        }

    def bjh_pore_size_distribution(self) -> Dict[str, Any]:
        """BJH pore-size distribution using the desorption branch."""
        if self.p_des is None or self.q_des is None:
            return {
                'available': False,
                'error': 'No desorption data'
            }

        sort_idx = np.argsort(self.p_des)
        p_des_sorted = self.p_des[sort_idx]
        q_des_sorted = self.q_des[sort_idx]

        gamma = 8.85e-3
        v_molar = 34.7e-6
        t = self.temperature

        with np.errstate(divide='ignore', invalid='ignore'):
            r_kelvin = -2 * gamma * v_molar / (
                GAS_CONSTANT * t * np.log(p_des_sorted + 1e-10)
            )

        pore_diameter = 2 * r_kelvin * 1e9

        valid = (pore_diameter > 0.5) & (pore_diameter < 200) & np.isfinite(pore_diameter)
        if np.sum(valid) < 5:
            return {
                'available': False,
                'error': 'Insufficient valid data for PSD'
            }

        pore_diameter_valid = pore_diameter[valid]
        q_valid = q_des_sorted[valid]

        sort_idx = np.argsort(pore_diameter_valid)
        pore_diameter_sorted = pore_diameter_valid[sort_idx]
        q_sorted = q_valid[sort_idx]

        log_d = np.log10(pore_diameter_sorted)
        dq = np.abs(np.diff(q_sorted))
        dlogd = np.diff(log_d)

        with np.errstate(divide='ignore', invalid='ignore'):
            dv_dlogd = dq * 0.001548 / dlogd

        finite = np.isfinite(dv_dlogd)
        if np.sum(finite) < 3:
            return {
                'available': False,
                'error': 'Invalid PSD calculation'
            }

        pore_diameters_psd = pore_diameter_sorted[:-1][finite]
        dv_dlogd_valid = dv_dlogd[finite]

        if len(pore_diameters_psd) >= 3:
            total_pore_volume = custom_trapz(dv_dlogd_valid,
                                             np.log10(pore_diameters_psd))

            micro_mask = pore_diameters_psd < 2
            meso_mask = (pore_diameters_psd >= 2) & (pore_diameters_psd <= 50)
            macro_mask = pore_diameters_psd > 50

            v_micro = custom_trapz(dv_dlogd_valid[micro_mask],
                                   np.log10(pore_diameters_psd[micro_mask])) if np.any(micro_mask) else 0.0
            v_meso = custom_trapz(dv_dlogd_valid[meso_mask],
                                  np.log10(pore_diameters_psd[meso_mask])) if np.any(meso_mask) else 0.0
            v_macro = custom_trapz(dv_dlogd_valid[macro_mask],
                                   np.log10(pore_diameters_psd[macro_mask])) if np.any(macro_mask) else 0.0

            if total_pore_volume > 0:
                mean_diameter = custom_trapz(
                    pore_diameters_psd * dv_dlogd_valid,
                    np.log10(pore_diameters_psd)
                ) / total_pore_volume
                peak_diameter = pore_diameters_psd[np.argmax(dv_dlogd_valid)]
            else:
                mean_diameter = 0.0
                peak_diameter = 0.0

            return {
                'available': True,
                'pore_diameters': pore_diameters_psd.tolist(),
                'dv_dlogd': dv_dlogd_valid.tolist(),
                'total_pore_volume': float(total_pore_volume),
                'mean_pore_diameter': float(mean_diameter),
                'peak_pore_diameter': float(peak_diameter),
                'micropore_fraction': float(v_micro / total_pore_volume) if total_pore_volume > 0 else 0.0,
                'mesopore_fraction': float(v_meso / total_pore_volume) if total_pore_volume > 0 else 0.0,
                'macropore_fraction': float(v_macro / total_pore_volume) if total_pore_volume > 0 else 0.0
            }

        return {
            'available': False,
            'error': 'Insufficient points for PSD'
        }

    def hysteresis_analysis(self) -> Dict[str, Any]:
        """IUPAC hysteresis classification."""
        if self.p_des is None or self.q_des is None:
            return {
                'type': 'I',
                'iupac_class': 'I',
                'description': 'Reversible (no hysteresis)',
                'loop_area': 0.0,
                'closure_pressure': 0.0,
                'valid': True
            }

        p_ads_sorted = self.p_ads
        q_ads_sorted = self.q_ads

        p_des_sorted = self.p_des
        q_des_sorted = self.q_des

        p_min = max(self.p_ads.min(), self.p_des.min())
        p_max = min(self.p_ads.max(), self.p_des.max())

        if p_max <= p_min:
            return {
                'type': 'Unknown',
                'iupac_class': 'I',
                'description': 'Insufficient overlap',
                'valid': False
            }

        p_common = np.linspace(p_min, p_max, 100)
        q_ads_interp = np.interp(p_common, p_ads_sorted, q_ads_sorted)
        q_des_interp = np.interp(p_common, p_des_sorted, q_des_sorted)

        loop_area = custom_trapz(np.abs(q_des_interp - q_ads_interp), p_common)

        closure_idx = np.argmin(np.abs(q_des_sorted - q_ads_sorted[-1]))
        closure_pressure = p_des_sorted[closure_idx] if closure_idx < len(p_des_sorted) else 0.5

        if loop_area < 5:
            h_type = "H1"
            description = "Uniform mesopores with narrow PSD (e.g., MCM-41, SBA-15)"
            iupac_class = "IV"
        elif closure_pressure > 0.45:
            h_type = "H2"
            description = "Ink-bottle pores or interconnected pore network"
            iupac_class = "IV"
        elif closure_pressure > 0.4:
            h_type = "H3"
            description = "Slit-shaped pores from plate-like particles"
            iupac_class = "II"
        else:
            h_type = "H4"
            description = "Combined micro-mesoporosity (e.g., activated carbons)"
            iupac_class = "I"

        return {
            'type': h_type,
            'iupac_class': iupac_class,
            'description': description,
            'loop_area': float(loop_area),
            'closure_pressure': float(closure_pressure),
            'valid': True
        }

    def total_pore_volume(self) -> float:
        """Total pore volume from the adsorption branch at highest pressure."""
        max_idx = np.argmax(self.p_ads)
        q_max = self.q_ads[max_idx]
        return q_max * 0.001548

    def complete_analysis(self) -> Dict[str, Any]:
        """Full IUPAC-compliant BET analysis."""
        try:
            bet_results = self.bet_surface_area()

            t_plot_results = self.t_plot_analysis()
            hysteresis_results = self.hysteresis_analysis()
            bjh_results = self.bjh_pore_size_distribution()

            total_pore_volume = self.total_pore_volume()

            mean_pore_diameter = 0.0
            if bet_results.get('valid', False) and bet_results['surface_area'] > 0 and total_pore_volume > 0:
                mean_pore_diameter = 4 * total_pore_volume / bet_results['surface_area'] * 1000
            elif total_pore_volume > 0:
                mean_pore_diameter = 2 * (total_pore_volume * 1e9 / np.pi) ** (1/3)

            overall_valid = (bet_results.get('valid', False) or
                             t_plot_results.get('valid', False) or
                             hysteresis_results.get('valid', False) or
                             bjh_results.get('available', False))

            results = {
                'overall_valid': overall_valid,
                'surface_area': bet_results.get('surface_area', 0.0),
                'surface_area_error': bet_results.get('surface_area_error', 0.0),
                'monolayer_capacity': bet_results.get('monolayer_capacity', 0.0),
                'c_constant': bet_results.get('c_constant', 0.0),
                'total_pore_volume': float(total_pore_volume),
                'mean_pore_diameter': float(mean_pore_diameter),
                'micropore_volume': t_plot_results.get('micropore_volume', 0.0),
                'external_surface': t_plot_results.get('external_surface', 0.0),
                'bet_regression': bet_results.get('bet_regression', {}),
                't_plot_analysis': t_plot_results,
                'hysteresis_analysis': hysteresis_results,
                'psd_analysis': bjh_results,
                'cross_section': float(self.cross_section * 1e18),
                'temperature': float(self.temperature),
                'data_points': {
                    'adsorption': len(self.p_ads),
                    'desorption': len(self.p_des) if self.p_des is not None else 0
                }
            }

            if not bet_results.get('valid', False):
                results['bet_error'] = bet_results.get('error', 'BET analysis failed')
                results['valid_without_bet'] = True
                results['bet_valid'] = False
            else:
                results['bet_valid'] = True
                results['valid_without_bet'] = True

            return results

        except Exception as e:
            return {
                'overall_valid': False,
                'error': str(e),
                'surface_area': 0.0,
                'surface_area_error': 0.0,
                'total_pore_volume': 0.0,
                'mean_pore_diameter': 0.0,
                'bet_valid': False,
                'valid_without_bet': False
            }


# ============================================================================
# ADDITIONAL POROSITY CALCULATIONS
# ============================================================================
def calculate_dubinin_astakhov(p_ads, q_ads, temperature=77.3):
    """
    Dubinin-Astakhov method for microporous materials.
    """
    p_absolute = p_ads

    with np.errstate(divide='ignore', invalid='ignore'):
        adsorption_potential = GAS_CONSTANT * temperature * np.log(1 / (p_absolute + 1e-10))

    valid = np.isfinite(adsorption_potential) & np.isfinite(q_ads)
    adsorption_potential = adsorption_potential[valid]
    q_ads_valid = q_ads[valid]

    if len(adsorption_potential) < 10:
        return None

    da_plot = np.log(q_ads_valid)
    x_plot = adsorption_potential**2

    slope, intercept, r_value, _, _ = stats.linregress(x_plot, da_plot)

    micropore_volume_da = np.exp(intercept) * 0.001548

    return {
        'micropore_volume': float(micropore_volume_da),
        'characteristic_energy': float(np.sqrt(-1/slope) if slope < 0 else 0),
        'r_squared': float(r_value**2),
        'method': 'Dubinin-Astakhov (n=2)'
    }
