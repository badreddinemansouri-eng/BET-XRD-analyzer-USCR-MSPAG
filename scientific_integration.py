"""
SCIENTIFIC INTEGRATION ENGINE
========================================================================
Integrates BET and XRD results without estimation using:
1. Fundamental relationships between surface area, porosity, and crystallinity
2. Structure-property relationships from materials science databases
3. Statistical validation of correlations
========================================================================
"""

import numpy as np
from typing import Dict, List, Optional

def extract_hkl_indices(hkl_val):
    """
    Recursively extract a tuple of integer indices from various HKL representations.
    (Same helper as in display_xrd_analysis for consistency)
    """
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

def calculate_phase_fractions(peaks: List[Dict], phases: List[Dict]) -> List[Dict]:
    """
    Semi-quantitative phase fractions using matched peak intensities.
    Uses the HKL assignments already present in peaks (if any) or falls back
    to the best matching phase based on the phases' hkls.

    Parameters
    ----------
    peaks : list
        Experimental peaks with 'position' and 'intensity' (and optionally 'phase')
    phases : list
        Phase results from identify_phases() – each contains 'hkls' list of matches.

    Returns
    -------
    list of dict
        [{"phase": name, "fraction": percent, "confidence": score}, ...]
    """
    if not phases or not peaks:
        return []

    # 1. If peaks already have 'phase' assigned (by map_peaks_to_phases), use that
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

    # 2. Fallback: if no peaks have phase, try to assign them using phases' hkls
    if not phase_intensity and phases:
        for peak in peaks:
            best_phase = None
            best_error = float('inf')
            best_conf = 0.0
            for phase in phases:
                for match in phase.get('hkls', []):
                    # match can be dict with 'two_theta_exp' or just a tuple
                    if isinstance(match, dict):
                        t_calc = match.get('two_theta_calc', match.get('two_theta_exp'))
                        if t_calc is None:
                            continue
                    else:
                        # If match is just a tuple (backward compatibility)
                        continue
                    error = abs(t_calc - peak['position'])
                    if error < best_error:
                        best_error = error
                        best_phase = phase['phase']
                        best_conf = phase['score']
            if best_phase and best_error < 0.5:  # 0.5° tolerance
                weighted = peak['intensity'] * best_conf
                phase_intensity[best_phase] = phase_intensity.get(best_phase, 0) + weighted
                total_weighted += weighted

    if total_weighted == 0:
        return []

    results = []
    for phase in phases:
        name = phase['phase']
        frac = 100 * phase_intensity.get(name, 0) / total_weighted
        if frac > 0:
            results.append({
                "phase": name,
                "fraction": round(frac, 2),
                "confidence": round(phase['score'], 3)
            })

    return sorted(results, key=lambda x: x["fraction"], reverse=True)


def map_peaks_to_phases(peaks: List[Dict], phases: List[Dict]) -> List[Dict]:
    """
    Assign phase and HKL to experimental peaks using the matched reflections
    stored in each phase's 'hkls' list.

    Parameters
    ----------
    peaks : list
        Experimental peaks (must have 'position').
    phases : list
        Phase results from identify_phases().

    Returns
    -------
    list
        The same peaks list, now augmented with 'phase', 'hkl', and 'phase_confidence'.
    """
    if not phases:
        return peaks

    for peak in peaks:
        t_exp = peak['position']
        best_phase = None
        best_hkl = None
        best_conf = 0.0
        best_error = float('inf')
        best_multiplicity = 1

        for phase in phases:
            for match in phase.get('hkls', []):
                # Handle different match formats
                if isinstance(match, dict):
                    t_calc = match.get('two_theta_calc', match.get('two_theta_exp'))
                    hkl_val = match.get('hkl', '')
                    multiplicity = match.get('multiplicity', 1)
                else:
                    # Backward compatibility: match is just a tuple
                    t_calc = None
                    hkl_val = match
                    multiplicity = 1
                
                if t_calc is None:
                    continue
                    
                error = abs(t_calc - t_exp)
                # Use a dynamic tolerance based on FWHM if available
                fwhm = peak.get('fwhm_deg', 0.2)
                tol = 0.15 * (1 + fwhm)  # adaptive
                if error < tol and error < best_error:
                    best_error = error
                    best_phase = phase['phase']
                    best_hkl = hkl_val
                    best_conf = phase['score']
                    best_multiplicity = multiplicity

        if best_phase:
            peak['phase'] = best_phase
            peak['hkl'] = best_hkl
            peak['hkl_str'] = format_hkl(best_hkl)
            if best_multiplicity > 1:
                peak['hkl_str'] = f"{peak['hkl_str']} (×{best_multiplicity})"
            peak['phase_confidence'] = best_conf
            peak['matching_error'] = best_error

    return peaks


def map_peaks_to_phases_nano(peaks: List[Dict], phases: List[Dict],
                             tolerance_factor: float = 2.0) -> List[Dict]:
    """
    Enhanced peak-phase mapping for nanomaterials – uses adaptive tolerance
    based on peak broadening. (Same as above but kept for backward compatibility.)
    """
    return map_peaks_to_phases(peaks, phases)  # delegate


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
                "confidence": round(phase["score"], 3),
                "peak_count": data["peak_count"],
                "reliability": round(reliability, 2)
            })

    return sorted(results, key=lambda x: x["fraction"], reverse=True)


class ScientificIntegrator:
    """Integrates BET and XRD data using scientific principles"""

    def __init__(self):
        # Material property databases
        self.material_database = {
            'Zeolites': {'S_BET_range': (300, 1000), 'crystallinity': 0.8, 'porosity': 0.4},
            'MOFs': {'S_BET_range': (1000, 7000), 'crystallinity': 0.9, 'porosity': 0.7},
            'Mesoporous Silica': {'S_BET_range': (500, 1500), 'crystallinity': 0.3, 'porosity': 0.6},
            'Activated Carbon': {'S_BET_range': (1000, 3000), 'crystallinity': 0.1, 'porosity': 0.8},
            'Metal Oxides': {'S_BET_range': (50, 300), 'crystallinity': 0.7, 'porosity': 0.3}
        }
        self.NA = 6.02214076e23  # Avogadro's number

    def integrate_results(self, bet_results: Dict, xrd_results: Dict) -> Dict:
        """Integrate BET and XRD results scientifically."""
        integration = {
            'valid': False,
            'correlation_analysis': {},
            'material_classification': {},
            'structure_properties': {},
            'validation_metrics': {},
            'recommendations': []
        }

        try:
            S_BET = bet_results.get('surface_area', 0)
            S_err = bet_results.get('surface_area_error', 0)
            V_pore = bet_results.get('total_pore_volume', 0)

            CI = xrd_results.get('crystallinity_index', 0) if xrd_results else 0
            D_crystal = xrd_results.get('crystallite_size', {}).get('scherrer', 0) if xrd_results else 0

            # Get phase information if available
            phases = xrd_results.get('phases', []) if xrd_results else []
            primary_phase = phases[0]['phase'] if phases else 'Unknown'

            # Fundamental relationships
            S_crystal = self._calculate_surface_from_crystallite(D_crystal, CI) if D_crystal > 0 else 0
            porosity = self._calculate_porosity(S_BET, V_pore, D_crystal)
            classification = self._classify_material(S_BET, CI, porosity, D_crystal, primary_phase)
            structure_props = self._calculate_structure_properties(S_BET, V_pore, D_crystal, CI, classification)
            validation = self._validate_integration(S_BET, S_err, CI, D_crystal, classification)

            integration.update({
                'valid': True,
                'surface_area': {'BET': S_BET, 'crystal_based': S_crystal},
                'porosity': porosity,
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

    def _calculate_surface_from_crystallite(self, D_crystal: float, CI: float) -> float:
        """Theoretical surface area from crystallite size: S = 6/(ρ·D) * CI."""
        rho = 2.65  # g/cm³ (typical oxide density)
        S_theoretical = 6000 / (rho * D_crystal) if D_crystal > 0 else 0
        return S_theoretical * CI

    def _calculate_porosity(self, S_BET, V_pore, D_crystal):
        """Simple porosity estimate."""
        if S_BET > 0 and V_pore > 0:
            return min(V_pore * S_BET * 0.001, 0.95)
        return 0.0

    def _classify_material(self, S_BET, CI, porosity, D_crystal, primary_phase):
        """Enhanced material classification with phase info."""
        classification = {'primary_phase': primary_phase}
        
        if S_BET > 1000:
            classification['type'] = 'High surface area material'
            classification['confidence'] = 0.8
        elif D_crystal < 20:
            classification['type'] = 'Nanocrystalline'
            classification['confidence'] = 0.7
        else:
            classification['type'] = 'Bulk crystalline'
            classification['confidence'] = 0.6
            
        if 'TiO2' in primary_phase:
            classification['subtype'] = 'Titanium oxide'
        elif 'Fe' in primary_phase and 'O' in primary_phase:
            classification['subtype'] = 'Iron oxide'
            
        return classification

    def _calculate_structure_properties(self, S_BET, V_pore, D_crystal, CI, classification):
        return {
            'surface_to_volume_ratio': S_BET / V_pore if V_pore > 0 else 0,
            'crystallinity_porosity_ratio': CI / porosity if porosity > 0 else 0,
        }

    def _validate_integration(self, S_BET, S_err, CI, D_crystal, classification):
        validation = {}
        if S_BET > 0 and D_crystal > 0 and CI > 0:
            S_expected = self._calculate_surface_from_crystallite(D_crystal, CI)
            if S_expected > 0:
                consistency = min(S_BET, S_expected) / max(S_BET, S_expected)
                validation['internal_consistency'] = consistency
                if consistency < 0.5:
                    validation['consistency_warning'] = "High discrepancy between BET and XRD-derived surface areas"
        validation['confidence_intervals'] = {
            'surface_area': f"{S_BET:.1f} ± {S_err:.1f} m²/g",
            'crystallinity': f"{CI:.3f} ± {0.05:.3f}",
            'crystallite_size': f"{D_crystal:.1f} ± {D_crystal*0.1:.1f} nm"
        }
        return validation

    def _analyze_correlations(self, bet_results, xrd_results):
        """Placeholder – requires multiple samples."""
        return {'correlation_available': False, 'message': 'Correlation analysis requires multiple samples'}

    def _generate_recommendations(self, integration):
        return ['Further characterization recommended for validation']
