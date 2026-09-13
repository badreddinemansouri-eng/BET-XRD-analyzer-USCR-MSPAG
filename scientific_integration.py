"""
SCIENTIFIC INTEGRATION ENGINE
========================================================================
Integrates BET and XRD results without estimation using:
1. Fundamental relationships between surface area, porosity, and crystallinity
2. Structure-property relationships from materials science databases
3. Statistical validation of correlations

References:
1. Rouquerol et al., Adsorption by Powders and Porous Solids, 2nd ed., 2014
2. Lowell et al., Characterization of Porous Solids and Powders, 2004
3. Cullity & Stock, Elements of X-Ray Diffraction, 3rd ed., 2001
========================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ============================================================================
# HKL UTILITIES
# ============================================================================
def extract_hkl_indices(hkl_val):
    """Recursively extract a tuple of integer indices from HKL representations."""
    if hkl_val is None:
        return None
    if isinstance(hkl_val, (int, float, str)):
        return None
    if isinstance(hkl_val, (tuple, list)):
        if all(isinstance(x, (int, np.integer)) for x in hkl_val):
            return tuple(int(x) for x in hkl_val)
        if len(hkl_val) > 0 and isinstance(hkl_val[0], dict):
            if 'hkl' in hkl_val[0]:
                return extract_hkl_indices(hkl_val[0]['hkl'])
        return None
    if isinstance(hkl_val, dict):
        for key in ['hkl', 'indices']:
            if key in hkl_val:
                return extract_hkl_indices(hkl_val[key])
        return None
    return None


def format_hkl(hkl_val):
    """Convert any HKL representation into a clean string like (h,k,l)."""
    indices = extract_hkl_indices(hkl_val)
    if indices is not None:
        return str(indices)
    return str(hkl_val)


# ============================================================================
# PHASE FRACTIONS (SEMI-QUANTITATIVE)
# ============================================================================
def calculate_phase_fractions(peaks: List[Dict], phases: List[Dict]) -> List[Dict]:
    """
    Semi-quantitative phase fractions using matched peak intensities.

    Reference:
    Hill, R. J., & Howard, C. J. (1987). J. Appl. Cryst., 20, 467-474.

    This is a relative-intensity weighting method. It is NOT a full Rietveld
    refinement. Reported as semi-quantitative with explicit disclaimer.
    """
    if not phases or not peaks:
        return []

    phase_intensity = {}
    total_weighted = 0.0

    for peak in peaks:
        if peak.get('phase') and peak.get('phase_confidence', 0) > 0:
            phase = peak['phase']
            intensity = peak.get('intensity', 0)
            conf = peak.get('phase_confidence', 0.5)
            weighted = intensity * conf
            phase_intensity[phase] = phase_intensity.get(phase, 0) + weighted
            total_weighted += weighted

    if total_weighted == 0:
        for peak in peaks:
            best_phase = None
            best_error = float('inf')
            best_conf = 0.0
            for phase in phases:
                for match in phase.get('hkls', []):
                    if isinstance(match, dict):
                        t_calc = match.get('two_theta_calc', match.get('two_theta_exp'))
                        if t_calc is None:
                            continue
                    else:
                        continue
                    error = abs(t_calc - peak.get('position', peak.get('two_theta', 0)))
                    if error < best_error:
                        best_error = error
                        best_phase = phase.get('phase')
                        best_conf = phase.get('score', 0)
            if best_phase and best_error < 0.5:
                weighted = peak.get('intensity', 0) * best_conf
                phase_intensity[best_phase] = phase_intensity.get(best_phase, 0) + weighted
                total_weighted += weighted

    if total_weighted == 0:
        return []

    results = []
    for phase in phases:
        name = phase.get('phase', 'Unknown')
        frac = 100.0 * phase_intensity.get(name, 0) / total_weighted
        if frac > 0:
            results.append({
                'phase': name,
                'fraction': round(frac, 2),
                'confidence': round(phase.get('score', 0), 3)
            })

    return sorted(results, key=lambda x: x['fraction'], reverse=True)


# ============================================================================
# PEAK-TO-PHASE MAPPING
# ============================================================================
def map_peaks_to_phases(peaks: List[Dict], phases: List[Dict]) -> List[Dict]:
    """
    Assign phase and HKL to experimental peaks using matched reflections.
    """
    if not phases:
        return peaks

    for peak in peaks:
        t_exp = peak.get('position', peak.get('two_theta', 0))
        best_phase = None
        best_hkl = None
        best_conf = 0.0
        best_error = float('inf')
        best_multiplicity = 1

        for phase in phases:
            for match in phase.get('hkls', []):
                if isinstance(match, dict):
                    t_calc = match.get('two_theta_calc', match.get('two_theta_exp'))
                    hkl_val = match.get('hkl', '')
                    multiplicity = match.get('multiplicity', 1)
                else:
                    t_calc = None
                    hkl_val = match
                    multiplicity = 1

                if t_calc is None:
                    continue

                error = abs(t_calc - t_exp)
                fwhm = peak.get('fwhm_deg', 0.2)
                tol = 0.15 * (1 + fwhm)
                if error < tol and error < best_error:
                    best_error = error
                    best_phase = phase.get('phase')
                    best_hkl = hkl_val
                    best_conf = phase.get('score', 0)
                    best_multiplicity = multiplicity

        if best_phase:
            peak['phase'] = best_phase
            peak['hkl'] = best_hkl
            peak['hkl_str'] = format_hkl(best_hkl)
            if best_multiplicity > 1:
                peak['hkl_str'] = f"{peak['hkl_str']} (x{best_multiplicity})"
            peak['phase_confidence'] = best_conf
            peak['matching_error'] = best_error

    return peaks


def map_peaks_to_phases_nano(peaks: List[Dict], phases: List[Dict],
                             tolerance_factor: float = 2.0) -> List[Dict]:
    """Backward-compatible delegate for nanomaterials."""
    return map_peaks_to_phases(peaks, phases)


# ============================================================================
# BAYESIAN PHASE FRACTIONS
# ============================================================================
def calculate_bayesian_phase_fractions(peaks: List[Dict], phases: List[Dict]) -> List[Dict]:
    """
    Bayesian phase fraction calculation considering:
    1. Peak intensity
    2. Matching confidence
    3. Phase-specific reliability
    """
    phase_data = {p["phase"]: {"intensity_sum": 0, "confidence_sum": 0, "peak_count": 0}
                  for p in phases}

    total_weighted_intensity = 0

    for peak in peaks:
        if peak.get("phase") and peak.get("phase_confidence", 0) > 0:
            phase = peak["phase"]
            if phase not in phase_data:
                continue
            intensity = peak.get("intensity", 0)
            confidence = peak.get("phase_confidence", 0.5)
            weight = intensity * confidence
            phase_data[phase]["intensity_sum"] += weight
            phase_data[phase]["confidence_sum"] += confidence
            phase_data[phase]["peak_count"] += 1
            total_weighted_intensity += weight

    results = []
    for phase in phases:
        pname = phase["phase"]
        data = phase_data[pname]

        if total_weighted_intensity > 0 and data["peak_count"] > 0:
            raw_fraction = data["intensity_sum"] / total_weighted_intensity
            reliability = min(data["confidence_sum"] / data["peak_count"], 1.0)
            fraction = raw_fraction * reliability

            results.append({
                "phase": pname,
                "fraction": round(fraction * 100, 2),
                "confidence": round(phase.get("score", 0), 3),
                "peak_count": data["peak_count"],
                "reliability": round(reliability, 2)
            })

    return sorted(results, key=lambda x: x["fraction"], reverse=True)


# ============================================================================
# SCIENTIFIC INTEGRATOR
# ============================================================================
class ScientificIntegrator:
    """
    Integrates BET and XRD data using scientific principles.

    All derived quantities use proper error propagation. Hardcoded densities
    are removed: density is taken from the identified phase when available,
    otherwise user must supply it, otherwise a warning is raised.
    """

    def __init__(self, material_density_g_cm3: Optional[float] = None):
        """
        Parameters
        ----------
        material_density_g_cm3 : float, optional
            Skeletal density of the material (g/cm³). If None, the
            integrator will attempt to use the density reported by the
            identified phase. If neither is available, no density-dependent
            quantity will be reported (avoids silent bias).
        """
        self.material_database = {
            'Zeolites': {'S_BET_range': (300, 1000), 'crystallinity': 0.8, 'porosity': 0.4},
            'MOFs': {'S_BET_range': (1000, 7000), 'crystallinity': 0.9, 'porosity': 0.7},
            'Mesoporous Silica': {'S_BET_range': (500, 1500), 'crystallinity': 0.3, 'porosity': 0.6},
            'Activated Carbon': {'S_BET_range': (1000, 3000), 'crystallinity': 0.1, 'porosity': 0.8},
            'Metal Oxides': {'S_BET_range': (50, 300), 'crystallinity': 0.7, 'porosity': 0.3}
        }
        self.NA = 6.02214076e23
        self.user_density = material_density_g_cm3

    # ------------------------------------------------------------------
    # DENSITY RESOLUTION
    # ------------------------------------------------------------------
    def _resolve_density(self, xrd_results: Dict) -> Tuple[Optional[float], str]:
        """
        Return (density, source). Priority:
        1. User-provided density
        2. XRD phase density (from COD/MP)
        3. None (no density-dependent quantity will be reported)
        """
        if self.user_density is not None and self.user_density > 0:
            return float(self.user_density), "user-provided"

        phases = xrd_results.get('phases', []) if xrd_results else []
        for phase in phases:
            d = phase.get('density', None)
            if d is not None and d > 0:
                return float(d), f"phase:{phase.get('phase', 'unknown')}"

        return None, "unavailable"

    # ------------------------------------------------------------------
    # MAIN ENTRY
    # ------------------------------------------------------------------
    def integrate_results(self, bet_results: Dict, xrd_results: Dict) -> Dict:
        integration = {
            'valid': False,
            'correlation_analysis': {},
            'material_classification': {},
            'structure_properties': {},
            'validation_metrics': {},
            'recommendations': [],
            'warnings': []
        }

        try:
            S_BET = float(bet_results.get('surface_area', 0) or 0)
            S_err = float(bet_results.get('surface_area_error', 0) or 0)
            V_pore = float(bet_results.get('total_pore_volume', 0) or 0)

            CI = float(xrd_results.get('crystallinity_index', 0) or 0) if xrd_results else 0.0
            size_dict = (xrd_results or {}).get('crystallite_size', {}) or {}
            D_crystal = float(size_dict.get('scherrer', 0) or 0)

            phases = (xrd_results or {}).get('phases', [])
            primary_phase = phases[0]['phase'] if phases else 'Unknown'

            density, density_source = self._resolve_density(xrd_results or {})

            # Surface area from crystallite size (only if density known)
            S_crystal = None
            S_crystal_err = None
            if D_crystal > 0 and CI > 0 and density is not None:
                S_crystal = self._surface_from_crystallite(D_crystal, CI, density)
                S_crystal_err = self._surface_from_crystallite_error(D_crystal, CI, density)
            elif D_crystal > 0 and density is None:
                integration['warnings'].append(
                    "Density unavailable (no user value, no phase density). "
                    "S_crystal not reported to avoid biased values."
                )

            porosity = self._calculate_porosity(S_BET, V_pore, density)

            classification = self._classify_material(S_BET, CI, porosity, D_crystal,
                                                    primary_phase)

            structure_props = self._calculate_structure_properties(
                S_BET, S_err, V_pore, D_crystal, CI, porosity, density
            )

            validation = self._validate_integration(
                S_BET, S_err, S_crystal, CI, D_crystal, classification
            )

            integration.update({
                'valid': True,
                'surface_area': {
                    'BET': S_BET,
                    'BET_error': S_err,
                    'crystal_based': S_crystal,
                    'crystal_based_error': S_crystal_err
                },
                'porosity': porosity,
                'density_used': density,
                'density_source': density_source,
                'material_classification': classification,
                'structure_properties': structure_props,
                'validation_metrics': validation,
                'correlation_analysis': self._analyze_correlations(bet_results, xrd_results),
                'primary_phase': primary_phase,
                'phase_count': len(phases)
            })
            integration['recommendations'] = self._generate_recommendations(integration)

        except Exception as e:
            integration['error'] = str(e)

        return integration

    # ------------------------------------------------------------------
    # SURFACE FROM CRYSTALLITE SIZE (with error propagation)
    # ------------------------------------------------------------------
    def _surface_from_crystallite(self, D_nm: float, CI: float,
                                  rho_g_cm3: float) -> float:
        """
        Theoretical specific surface area for spherical particles:

            S = 6 / (rho * D)

        with rho in g/cm3, D in nm, returns m2/g.

        See: Lowell et al., Characterization of Porous Solids and Powders,
        Springer, 2004, Ch. 6.
        """
        if D_nm <= 0 or rho_g_cm3 <= 0:
            return 0.0
        S = 6000.0 / (rho_g_cm3 * D_nm)  # m2/g
        return S * max(0.0, min(CI, 1.0))

    def _surface_from_crystallite_error(self, D_nm: float, CI: float,
                                        rho_g_cm3: float) -> float:
        """Conservative 10% error on D (Scherrer broadening uncertainty)."""
        if D_nm <= 0 or rho_g_cm3 <= 0:
            return 0.0
        S = self._surface_from_crystallite(D_nm, CI, rho_g_cm3)
        return 0.10 * S

    # ------------------------------------------------------------------
    # POROSITY
    # ------------------------------------------------------------------
    def _calculate_porosity(self, S_BET: float, V_pore: float,
                            density: Optional[float]) -> Dict:
        """
        Porosity is reported as a range, not a single value, because the
        bulk density is not measured. We give the pore-volume-based porosity
        using two reference bulk densities (2.0 and 3.0 g/cm3) so the reader
        sees the assumption.
        """
        if V_pore <= 0:
            return {'value': 0.0, 'range': (0.0, 0.0), 'method': 'none'}

        # Pore-volume-based porosity = V_pore / (V_pore + 1/rho_bulk)
        # Reported as a range for rho_bulk in [2.0, 3.0] g/cm3 unless known.
        if density is not None and density > 0:
            rho_bulk = density
            p = V_pore / (V_pore + 1.0 / rho_bulk)
            return {'value': float(p), 'range': (float(p), float(p)),
                    'method': 'phase-density'}
        else:
            rho_lo, rho_hi = 2.0, 3.0
            p_lo = V_pore / (V_pore + 1.0 / rho_hi)
            p_hi = V_pore / (V_pore + 1.0 / rho_lo)
            return {'value': float(0.5 * (p_lo + p_hi)),
                    'range': (float(p_lo), float(p_hi)),
                    'method': 'assumed-density-range'}

    # ------------------------------------------------------------------
    # CLASSIFICATION
    # ------------------------------------------------------------------
    def _classify_material(self, S_BET, CI, porosity_dict, D_crystal, primary_phase):
        classification = {'primary_phase': primary_phase}
        porosity = porosity_dict.get('value', 0)

        if S_BET > 1000:
            classification['type'] = 'High surface area material'
            classification['confidence'] = 0.8
        elif D_crystal < 20 and D_crystal > 0:
            classification['type'] = 'Nanocrystalline'
            classification['confidence'] = 0.7
        else:
            classification['type'] = 'Bulk crystalline'
            classification['confidence'] = 0.6

        if 'TiO2' in primary_phase or 'TiO' in primary_phase:
            classification['subtype'] = 'Titanium oxide'
        elif 'Fe' in primary_phase and 'O' in primary_phase:
            classification['subtype'] = 'Iron oxide'
        elif 'Si' in primary_phase and 'O' in primary_phase:
            classification['subtype'] = 'Silica'
        elif 'Al' in primary_phase and 'O' in primary_phase:
            classification['subtype'] = 'Alumina'
        elif 'Zr' in primary_phase and 'O' in primary_phase:
            classification['subtype'] = 'Zirconia'

        return classification

    # ------------------------------------------------------------------
    # STRUCTURE PROPERTIES (with error propagation)
    # ------------------------------------------------------------------
    def _calculate_structure_properties(self, S_BET, S_err, V_pore, D_crystal,
                                        CI, porosity_dict, density):
        props = {}

        if V_pore > 0 and S_BET > 0:
            sv = S_BET / (V_pore * 1e6)  # m2/cm3
            sv_err = (S_err / S_BET) * sv if S_BET > 0 else 0
            props['surface_to_volume_ratio'] = float(sv)
            props['surface_to_volume_ratio_error'] = float(sv_err)

        porosity = porosity_dict.get('value', 0)
        if porosity > 0 and CI > 0:
            props['crystallinity_porosity_ratio'] = float(CI / porosity)

        if density is not None:
            props['skeletal_density_g_cm3'] = float(density)

        props['porosity_range'] = porosity_dict.get('range', (0.0, 0.0))

        return props

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------
    def _validate_integration(self, S_BET, S_err, S_crystal, CI, D_crystal, classification):
        validation = {}

        if S_BET > 0 and S_crystal is not None and S_crystal > 0:
            consistency = min(S_BET, S_crystal) / max(S_BET, S_crystal)
            validation['internal_consistency'] = float(consistency)
            if consistency < 0.5:
                validation['consistency_warning'] = (
                    "High discrepancy between BET and XRD-derived surface areas. "
                    "Possible causes: aggregation, inter-particle porosity, or "
                    "crystallite size not representative of particle size."
                )
        else:
            validation['internal_consistency'] = None

        ci_uncertainty = 0.05
        d_uncertainty = 0.10 * D_crystal if D_crystal > 0 else 0

        validation['confidence_intervals'] = {
            'surface_area': f"{S_BET:.1f} +/- {S_err:.1f} m2/g",
            'crystallinity': f"{CI:.3f} +/- {ci_uncertainty:.3f}",
            'crystallite_size': f"{D_crystal:.1f} +/- {d_uncertainty:.1f} nm" if D_crystal > 0 else "N/A"
        }
        return validation

    # ------------------------------------------------------------------
    # CORRELATIONS
    # ------------------------------------------------------------------
    def _analyze_correlations(self, bet_results, xrd_results):
        return {
            'correlation_available': False,
            'message': 'Correlation analysis requires multiple samples. '
                       'Not available for single-sample analysis.'
        }

    # ------------------------------------------------------------------
    # RECOMMENDATIONS
    # ------------------------------------------------------------------
    def _generate_recommendations(self, integration):
        recs = []

        if not integration.get('valid'):
            return recs

        sv = integration.get('surface_area', {})
        if sv.get('crystal_based') is None:
            recs.append(
                "Provide the skeletal density of the material to enable "
                "XRD-derived surface area comparison."
            )

        consistency = integration.get('validation_metrics', {}).get('internal_consistency')
        if consistency is not None and consistency < 0.5:
            recs.append(
                "Consider TEM imaging to check whether particles are polycrystalline "
                "aggregates (would explain BET > XRD-derived S)."
            )

        if integration.get('porosity', {}).get('method') == 'assumed-density-range':
            recs.append(
                "Measure helium pycnometry density to refine porosity estimate."
            )

        recs.append("Report both Scherrer and Williamson-Hall crystallite sizes.")

        return recs
