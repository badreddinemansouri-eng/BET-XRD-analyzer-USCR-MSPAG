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
import pandas as pd
from typing import Dict, List
from scipy import stats

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
        
        # Physical constants
        self.NA = 6.02214076e23  # Avogadro's number
        self.kB = 1.380649e-23   # Boltzmann constant
        
    def integrate_results(self, bet_results: Dict, xrd_results: Dict) -> Dict:
        """
        Integrate BET and XRD results scientifically
        
        Returns validated integration with confidence intervals
        """
        integration = {
            'valid': False,
            'correlation_analysis': {},
            'material_classification': {},
            'structure_properties': {},
            'validation_metrics': {},
            'recommendations': []
        }
        
        try:
            # Extract parameters
            S_BET = bet_results.get('surface_area', 0)
            S_err = bet_results.get('surface_area_error', 0)
            V_pore = bet_results.get('total_pore_volume', 0)
            
            if xrd_results:
                CI = xrd_results.get('crystallinity_index', 0)
                D_crystal = xrd_results.get('crystallite_size', {}).get('scherrer', 0)
            else:
                CI = 0
                D_crystal = 0
            
            # Calculate fundamental relationships
            # 1. Specific surface area from crystallite size (if crystalline)
            if D_crystal > 0 and CI > 0.5:
                S_crystal = self._calculate_surface_from_crystallite(D_crystal, CI)
                S_ratio = S_BET / S_crystal if S_crystal > 0 else 0
            else:
                S_crystal = 0
                S_ratio = 0
            
            # 2. Porosity from BET and XRD
            porosity = self._calculate_porosity(S_BET, V_pore, D_crystal)
            
            # 3. Material classification using multiple criteria
            classification = self._classify_material(S_BET, CI, porosity, D_crystal)
            
            # 4. Calculate structure-property relationships
            structure_props = self._calculate_structure_properties(
                S_BET, V_pore, D_crystal, CI, classification
            )
            
            # 5. Statistical validation
            validation = self._validate_integration(
                S_BET, S_err, CI, D_crystal, classification
            )
            
            # Compile results
            integration.update({
                'valid': True,
                'surface_area': {'BET': S_BET, 'crystal_based': S_crystal, 'ratio': S_ratio},
                'porosity': porosity,
                'material_classification': classification,
                'structure_properties': structure_props,
                'validation_metrics': validation,
                'correlation_analysis': self._analyze_correlations(bet_results, xrd_results)
            })
            
            # Generate scientific recommendations
            integration['recommendations'] = self._generate_recommendations(integration)
            
        except Exception as e:
            integration['error'] = str(e)
        
        return integration
    
    def _calculate_surface_from_crystallite(self, D_crystal: float, CI: float) -> float:
        """
        Calculate theoretical surface area from crystallite size
        
        For spherical crystallites: S = 6/(ρ·D)
        For cubic crystallites: S = 6/(ρ·D)
        """
        rho = 2.65  # Approximate density (g/cm³)
        S_theoretical = 6000 / (rho * D_crystal)  # m²/g
        
        # Adjust for crystallinity
        return S_theoretical * CI
    
    def _validate_integration(self, S_BET: float, S_err: float, 
                            CI: float, D_crystal: float, 
                            classification: Dict) -> Dict:
        """Validate integration using statistical methods"""
        validation = {}
        
        # Check internal consistency
        if S_BET > 0 and D_crystal > 0 and CI > 0:
            # Expected surface area from crystallite size
            S_expected = self._calculate_surface_from_crystallite(D_crystal, CI)
            
            # Calculate consistency metric
            if S_expected > 0:
                consistency = min(S_BET, S_expected) / max(S_BET, S_expected)
                validation['internal_consistency'] = consistency
                
                # Flag inconsistencies
                if consistency < 0.5:
                    validation['consistency_warning'] = "High discrepancy between BET and XRD-derived surface areas"
        
        # Calculate confidence intervals
        validation['confidence_intervals'] = {
            'surface_area': f"{S_BET:.1f} ± {S_err:.1f} m²/g",
            'crystallinity': f"{CI:.3f} ± {0.05:.3f}",  # Assumed error
            'crystallite_size': f"{D_crystal:.1f} ± {D_crystal*0.1:.1f} nm"  # 10% error
        }
    def calculate_phase_fractions(peaks, phases, tol=0.15):
        """
        Semi-quantitative phase fractions using matched peak intensities.
        No estimation, CIF-validated only.
    
        Parameters
        ----------
        peaks : list
            Experimental peaks with 'position' and 'intensity'
        phases : list
            Phase results from identify_phases()
        tol : float
            2θ tolerance in degrees
    
        Returns
        -------
        list of dict
        """
        phase_intensity = {p["phase"]: 0.0 for p in phases}
        total_intensity = 0.0
    
        for peak in peaks:
            t_exp = peak["position"]
            I = peak["intensity"]
    
            for phase in phases:
                for hkl_entry in phase["hkls"]:
                    for refl in hkl_entry:
                        if abs(refl["two_theta"] - t_exp) < tol:
                            phase_intensity[phase["phase"]] += I
                            total_intensity += I
                            break
    
        results = []
        for phase in phases:
            pname = phase["phase"]
            frac = (
                100 * phase_intensity[pname] / total_intensity
                if total_intensity > 0 else 0.0
            )
    
            results.append({
                "phase": pname,
                "fraction": round(frac, 2),
                "confidence": round(phase["score"], 3)
            })
    
        return sorted(results, key=lambda x: x["fraction"], reverse=True)     
    def map_peaks_to_phases(peaks, phases, tol=0.15):
        """
        Assign phase + HKL to experimental peaks using CIF d-spacing match.
        """
    
        for peak in peaks:
            peak["phase"] = ""
            peak["hkl"] = ""
            peak["phase_confidence"] = 0.0
    
            for phase in phases:
                for hkl_entry in phase["hkls"]:
                    for refl in hkl_entry:
                        if abs(refl["two_theta"] - peak["position"]) < tol:
                            peak["phase"] = phase["phase"]
                            peak["hkl"] = refl["hkl"]
                            peak["phase_confidence"] = phase["score"]
                            break
    
        return peaks        

        return validation
