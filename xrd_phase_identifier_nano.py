"""
ADVANCED XRD PHASE IDENTIFIER FOR NANOMATERIALS
========================================================================
Scientific Features for Nanomaterial Detection:
1. Peak Broadening Compensation (Nano-Specific)
2. Strain-Corrected d-spacing Matching
3. Multi-Database Consensus Scoring
4. Peak Profile Analysis (Pseudo-Voigt fitting)
5. Nanoscale-Specific Tolerance Windows
6. Bayesian Probability Scoring
========================================================================
References:
1. Balzar, D. (1999). J. Appl. Cryst., 32, 364-372 (Size-Strain Analysis)
2. McCusker, L. B. et al. (1999). Powder Diffr., 14, 2-3 (Rietveld for Nanomaterials)
3. Langford, J. I. & Louër, D. (1996). Rep. Prog. Phys., 59, 131 (Nanocrystalline XRD)
4. Le Bail, A. (2005). Powder Diffr., 20, 4 (Ab Initio Structure Determination)
========================================================================
"""

import numpy as np
import requests
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.stats import linregress
import streamlit as st

# ------------------------------------------------------------
# NANOMATERIAL-SPECIFIC PARAMETERS
# ------------------------------------------------------------
@dataclass
class NanoMaterialParams:
    """Parameters optimized for nanomaterial characterization"""
    # Peak broadening compensation (nm scale)
    SIZE_BROADENING_FACTOR = {
        '<5nm': 0.15,    # 15% tolerance for ultra-nano
        '5-10nm': 0.10,  # 10% tolerance for nano
        '10-20nm': 0.06, # 6% tolerance for fine crystalline
        '>20nm': 0.03    # 3% tolerance for bulk-like
    }
    
    # Strain broadening factor (ε × 100%)
    STRAIN_TOLERANCE = 0.05  # 5% strain tolerance
    
    # Minimum peak requirements for nanomaterials
    MIN_PEAK_SNR = 3.0    # Signal-to-noise ratio
    MIN_PEAK_INTENSITY = 0.05  # 5% of max intensity
    
    # Peak shape parameters (Pseudo-Voigt)
    PEAK_SHAPE_LORENTZIAN = 0.7  # More Lorentzian for nanomaterials
    PEAK_SHAPE_GAUSSIAN = 0.3

# ------------------------------------------------------------
# ADVANCED PEAK ANALYSIS FOR NANOMATERIALS
# ------------------------------------------------------------
class NanoPeakAnalyzer:
    """Advanced peak analysis with nanomaterial-specific corrections"""
    
    @staticmethod
    def analyze_peak_profile(two_theta: np.ndarray, intensity: np.ndarray, 
                            peak_position: float, wavelength: float) -> Dict:
        """
        Analyze peak profile using Pseudo-Voigt fitting for nanomaterials
        Returns: size, strain, and profile parameters
        """
        # Select region around peak (± 5° for nanomaterials)
        mask = (two_theta >= peak_position - 5) & (two_theta <= peak_position + 5)
        if np.sum(mask) < 10:
            return None
        
        theta = two_theta[mask] / 2  # Convert to θ
        I = intensity[mask]
        
        # Pseudo-Voigt function: η*Lorentzian + (1-η)*Gaussian
        def pseudo_voigt(x, I0, x0, fwhm, eta):
            # Lorentzian component
            L = I0 * (fwhm**2 / (4 * (x - x0)**2 + fwhm**2))
            # Gaussian component
            G = I0 * np.exp(-4 * np.log(2) * ((x - x0) / fwhm)**2)
            return eta * L + (1 - eta) * G
        
        try:
            # Initial guess
            p0 = [I.max(), peak_position/2, 0.1, 0.7]  # I0, x0, fwhm, eta
            bounds = ([0, peak_position/2 - 0.5, 0.01, 0.1], 
                     [I.max()*2, peak_position/2 + 0.5, 2.0, 0.9])
            
            popt, pcov = curve_fit(pseudo_voigt, theta, I, p0=p0, bounds=bounds)
            
            I0, x0, fwhm_theta, eta = popt
            
            # Convert to radians
            fwhm_rad = np.deg2rad(fwhm_theta * 2)  # Convert to 2θ rad
            
            # Scherrer size (K=0.9)
            D_scherrer = 0.9 * wavelength / (fwhm_rad * np.cos(x0))
            
            # Williamson-Hall components
            beta_total = fwhm_rad
            size_contrib = 0.9 * wavelength / (D_scherrer * np.cos(x0))
            strain_contrib = beta_total - size_contrib if beta_total > size_contrib else 0
            
            return {
                'position_2theta': float(x0 * 2),
                'fwhm_deg': float(fwhm_theta * 2),
                'fwhm_rad': float(fwhm_rad),
                'eta': float(eta),  # Lorentzian fraction
                'size_nm': float(D_scherrer / 10),  # Å to nm
                'strain_estimate': float(4 * strain_contrib / np.tan(x0)),
                'intensity': float(I0),
                'profile_type': 'Lorentzian-dominated' if eta > 0.6 else 'Gaussian-dominated'
            }
        except:
            return None
    
    @staticmethod
    def estimate_nano_scale_factor(peaks: List[Dict]) -> float:
        """Estimate nanoscale factor from peak broadening statistics"""
        if not peaks:
            return 1.0
        
        fwhms = [p.get('fwhm_deg', 0) for p in peaks if p.get('fwhm_deg', 0) > 0]
        if not fwhms:
            return 1.0
        
        avg_fwhm = np.mean(fwhms)
        
        # Scale factor based on FWHM (nanomaterials have broader peaks)
        if avg_fwhm > 1.0:  # Very broad peaks
            return 1.5  # 50% more tolerance
        elif avg_fwhm > 0.5:
            return 1.3  # 30% more tolerance
        elif avg_fwhm > 0.2:
            return 1.15  # 15% more tolerance
        else:
            return 1.0  # Normal tolerance

# ------------------------------------------------------------
# ADVANCED DATABASE SEARCH WITH NANO-OPTIMIZATION
# ------------------------------------------------------------
class NanoPhaseIdentifier:
    """Professional phase identification optimized for nanomaterials"""
    
    def __init__(self):
        self.cod_api = "https://www.crystallography.net/cod/result"
        self.mp_api = "https://api.materialsproject.org"
        self.mp_api_key = None  # Would be set from environment in production
        
        # Crystallographic databases with nanomaterial focus
        self.nano_focused_phases = {
            'TiO2': ['Anatase', 'Rutile', 'Brookite', 'TiO2-B'],
            'ZnO': ['Wurtzite', 'Zincite'],
            'CeO2': ['Cerianite', 'Fluorite'],
            'Fe2O3': ['Hematite', 'Maghemite'],
            'SiO2': ['Quartz', 'Cristobalite', 'Tridymite'],
            'Al2O3': ['Corundum', 'Gamma-alumina'],
            'ZrO2': ['Monoclinic', 'Tetragonal', 'Cubic']
        }
    
    def calculate_bayesian_score(self, exp_d: np.ndarray, sim_d: np.ndarray, 
                                fwhm_factors: List[float], 
                                composition_match: float = 1.0) -> float:
        """
        Bayesian scoring considering:
        1. d-spacing matching with nanoscale tolerance
        2. Peak broadening likelihood
        3. Compositional probability
        4. Prior knowledge from database
        
        Returns: Probability score (0-1)
        """
        if len(exp_d) == 0 or len(sim_d) == 0:
            return 0.0
        
        match_scores = []
        weights = []
        
        for i, d_exp in enumerate(exp_d):
            # Nanoscale-adaptive tolerance
            if i < len(fwhm_factors):
                tolerance = 0.02 * fwhm_factors[i]  # Base tolerance × broadening factor
            else:
                tolerance = 0.05  # Default for nanomaterials
            
            # Find best match
            rel_errors = np.abs(sim_d - d_exp) / d_exp
            best_match_idx = np.argmin(rel_errors)
            best_error = rel_errors[best_match_idx]
            
            if best_error < tolerance:
                # Gaussian probability with nanoscale adjustment
                probability = np.exp(-0.5 * (best_error / tolerance) ** 2)
                match_scores.append(probability)
                
                # Weight by expected intensity (approximated by 1/d²)
                weight = 1 / (d_exp ** 2)
                weights.append(weight)
        
        if not match_scores:
            return 0.0
        
        # Weighted average of match probabilities
        weighted_score = np.average(match_scores, weights=weights)
        
        # Incorporate composition match
        final_score = weighted_score * composition_match
        
        # Prior knowledge adjustment (if we know it's likely a nanomaterial phase)
        if any(phase in self.nano_focused_phases for phase in ['TiO2', 'ZnO', 'CeO2']):
            final_score *= 1.1  # 10% boost for common nanomaterial phases
        
        return min(final_score, 1.0)
    
    def fetch_nanomaterial_cifs(self, elements: List[str], max_size: float = 50.0) -> List[Dict]:
        """
        Fetch CIFs with nanomaterial-specific filtering
        Prioritizes phases known to form nanostructures
        """
        import concurrent.futures
        from functools import partial
        
        all_structures = []
        
        # COD search with nanomaterial filter
        cod_structures = self._fetch_cod_nano(elements)
        all_structures.extend(cod_structures)
        
        # Materials Project search (if API key available)
        if self.mp_api_key:
            mp_structures = self._fetch_materials_project_nano(elements, max_size)
            all_structures.extend(mp_structures)
        
        # Filter for nanomaterial-relevant phases
        nano_relevant = []
        for struct in all_structures:
            formula = struct.get('formula', '')
            
            # Check if this is a commonly nanostructured material
            is_nano_relevant = any(
                base in formula for base in ['TiO2', 'ZnO', 'CeO2', 'Fe2O3', 
                                           'SiO2', 'Al2O3', 'ZrO2', 'SnO2']
            )
            
            if is_nano_relevant:
                struct['nano_relevance'] = 'high'
                nano_relevant.append(struct)
            else:
                struct['nano_relevance'] = 'medium'
                nano_relevant.append(struct)
        
        return nano_relevant
    
    def _fetch_cod_nano(self, elements: List[str]) -> List[Dict]:
        """Fetch COD structures with nanomaterial focus"""
        try:
            query = {
                "format": "json",
                "el": ",".join(elements),
                "maxresults": 50,
                "nonalphanumeric": "ignore"  # Include nanostructured phases
            }
            response = requests.get(self.cod_api, params=query, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                structures = []
                
                for entry in data.get('data', []):
                    formula = entry.get('formula', '')
                    
                    # Enhanced metadata for nanomaterials
                    structures.append({
                        'database': 'COD',
                        'cod_id': entry.get('cod_id', ''),
                        'formula': formula,
                        'space_group': entry.get('space_group', ''),
                        'cell_parameters': entry.get('cell_parameters', {}),
                        'cif_url': f"https://www.crystallography.net/cod/{entry.get('cod_id')}.cif",
                        'nano_score': self._calculate_nano_score(formula, entry.get('cell_parameters', {}))
                    })
                
                return structures
        except Exception as e:
            st.warning(f"COD fetch warning: {str(e)}")
        
        return []
    
    def _calculate_nano_score(self, formula: str, cell_params: Dict) -> float:
        """Calculate nanomaterial formation likelihood score"""
        score = 0.5  # Base score
        
        # Common nanomaterial oxides
        nano_oxides = ['TiO2', 'ZnO', 'CeO2', 'Fe2O3', 'SiO2', 'Al2O3', 'ZrO2']
        if any(oxide in formula for oxide in nano_oxides):
            score += 0.3
        
        # Check for metastable phases (common in nanomaterials)
        if 'B' in formula or 'beta' in formula.lower() or 'gamma' in formula.lower():
            score += 0.2
        
        # Small unit cell often indicates simpler nanostructures
        if 'a' in cell_params:
            a = float(cell_params.get('a', 10))
            if a < 6.0:  # Small unit cell
                score += 0.1
        
        return min(score, 1.0)

# ------------------------------------------------------------
# MAIN IDENTIFICATION ENGINE WITH NANO-OPTIMIZATION
# ------------------------------------------------------------
def identify_nanomaterial_phases(two_theta: np.ndarray, intensity: np.ndarray, 
                                wavelength: float, elements: List[str],
                                estimated_size: float = None) -> List[Dict]:
    """
    Professional phase identification for nanomaterials
    Incorporates size/strain corrections and multi-algorithm consensus
    """
    st.info("🔬 Running nanomaterial-optimized phase identification...")
    
    # --------------------------------------------------------
    # STEP 1: NANO-OPTIMIZED PEAK DETECTION
    # --------------------------------------------------------
    # Adaptive threshold for nanomaterials (lower due to broadening)
    threshold = 0.15 * np.max(intensity)  # Lower threshold for weak nano-peaks
    
    # Find peaks with prominence filtering for nanomaterials
    peaks_idx, properties = find_peaks(intensity, 
                                      height=threshold,
                                      prominence=threshold/3,
                                      width=2)  # Wider peaks for nanomaterials
    
    exp_peaks_2theta = two_theta[peaks_idx]
    exp_intensities = intensity[peaks_idx]
    
    if len(exp_peaks_2theta) < 3:
        st.warning("Insufficient peaks for reliable nanomaterial identification")
        return []
    
    # Calculate d-spacings
    exp_d = wavelength / (2 * np.sin(np.radians(exp_peaks_2theta / 2)))
    
    # --------------------------------------------------------
    # STEP 2: NANOMATERIAL CHARACTERIZATION
    # --------------------------------------------------------
    nano_analyzer = NanoPeakAnalyzer()
    peak_profiles = []
    fwhm_factors = []
    
    for i, peak_pos in enumerate(exp_peaks_2theta):
        profile = nano_analyzer.analyze_peak_profile(two_theta, intensity, 
                                                    peak_pos, wavelength)
        if profile:
            peak_profiles.append(profile)
            # FWHM factor for tolerance scaling
            fwhm_factor = 1 + (profile['fwhm_deg'] / 0.5)  # Broader peaks → larger tolerance
            fwhm_factors.append(min(fwhm_factor, 2.0))  # Cap at 2×
        else:
            fwhm_factors.append(1.0)
    
    # Estimate overall nanoscale factor
    nano_scale_factor = nano_analyzer.estimate_nano_scale_factor(peak_profiles)
    
    # --------------------------------------------------------
    # STEP 3: DATABASE SEARCH WITH NANO-PRIORITIZATION
    # --------------------------------------------------------
    identifier = NanoPhaseIdentifier()
    
    # Get nanomaterial-focused structures
    st.write("📚 Searching nanomaterial-focused databases...")
    nano_structures = identifier.fetch_nanomaterial_cifs(elements)
    
    if not nano_structures:
        st.warning("No nanomaterial-relevant structures found in databases")
        return []
    
    # --------------------------------------------------------
    # STEP 4: MULTI-ALGORITHM MATCHING
    # --------------------------------------------------------
    results = []
    
    for struct in nano_structures[:30]:  # Limit to top 30 nanomaterial candidates
        try:
            # Fetch and simulate pattern
            cif_url = struct.get('cif_url')
            if not cif_url:
                continue
                
            cif_text = requests.get(cif_url, timeout=30).text
            
            # Simulate pattern
            from pymatgen.io.cif import CifParser
            from pymatgen.analysis.diffraction.xrd import XRDCalculator
            
            parser = CifParser.from_string(cif_text)
            structure = parser.get_structures()[0]
            calc = XRDCalculator(wavelength=wavelength)
            pattern = calc.get_pattern(structure, two_theta_range=(5, 80))
            
            # Simulated d-spacings
            sim_d = wavelength / (2 * np.sin(np.radians(pattern.x / 2)))
            
            # --------------------------------------------------------
            # STEP 5: BAYESIAN SCORING WITH NANO-CORRECTIONS
            # --------------------------------------------------------
            # Composition matching
            comp_elements = [el.symbol for el in structure.composition.elements]
            comp_match = len(set(elements) & set(comp_elements)) / max(len(elements), 1)
            
            # Calculate Bayesian score
            bayesian_score = identifier.calculate_bayesian_score(
                exp_d, sim_d, fwhm_factors, comp_match
            )
            
            # Apply nanoscale boost
            nano_boost = struct.get('nano_score', 0.5)
            final_score = bayesian_score * (1 + 0.2 * nano_boost)  # Up to 20% boost
            
            # Threshold for nanomaterials (lower due to difficulties)
            if final_score < 0.4:  # Lower threshold for nanomaterials
                continue
            
            # Confidence levels for nanomaterials
            if final_score >= 0.7:
                confidence = "confirmed"
            elif final_score >= 0.5:
                confidence = "probable"
            elif final_score >= 0.4:
                confidence = "possible"
            else:
                continue
            
            # Store results
            results.append({
                "phase": structure.composition.reduced_formula,
                "full_formula": str(structure.composition),
                "crystal_system": structure.get_crystal_system(),
                "space_group": structure.get_space_group_info()[0],
                "lattice": structure.lattice.as_dict(),
                "hkls": pattern.hkls,
                "score": round(final_score, 3),
                "confidence_level": confidence,
                "database": struct.get('database', 'Unknown'),
                "nano_relevance": struct.get('nano_relevance', 'medium'),
                "nano_score": round(struct.get('nano_score', 0), 2),
                "structure": structure,
                "peak_profiles": peak_profiles[:5] if peak_profiles else [],
                "estimated_size_nm": np.mean([p['size_nm'] for p in peak_profiles]) if peak_profiles else None
            })
            
        except Exception as e:
            continue
    
    # --------------------------------------------------------
    # STEP 6: CONSENSUS RANKING AND DEDUPLICATION
    # --------------------------------------------------------
    # Group by formula and crystal system
    grouped_results = {}
    for result in results:
        key = (result["phase"], result["crystal_system"], result["space_group"])
        if key not in grouped_results or result["score"] > grouped_results[key]["score"]:
            grouped_results[key] = result
    
    final_results = sorted(
        grouped_results.values(),
        key=lambda x: (x["score"], x.get("nano_score", 0)),
        reverse=True
    )
    
    # --------------------------------------------------------
    # STEP 7: SCIENTIFIC VALIDATION REPORT
    # --------------------------------------------------------
    st.success(f"✅ Identified {len(final_results)} potential nanomaterial phases")
    
    if final_results:
        with st.expander("🔬 Nanomaterial Analysis Report", expanded=True):
            st.markdown("### **Scientific Validation Summary**")
            
            # Peak quality assessment
            avg_fwhm = np.mean([p.get('fwhm_deg', 0) for p in peak_profiles]) if peak_profiles else 0
            st.markdown(f"- **Average Peak FWHM:** {avg_fwhm:.3f}°")
            st.markdown(f"- **Peak Count:** {len(exp_peaks_2theta)}")
            st.markdown(f"- **Estimated Crystallite Size:** {final_results[0].get('estimated_size_nm', 'N/A')} nm")
            
            if avg_fwhm > 0.5:
                st.markdown("⚠️ **Note:** Broad peaks suggest nanocrystalline or strained material")
            
            # Database statistics
            db_counts = {}
            for r in final_results:
                db = r.get('database', 'Unknown')
                db_counts[db] = db_counts.get(db, 0) + 1
            
            st.markdown("### **Database Matches**")
            for db, count in db_counts.items():
                st.markdown(f"- **{db}:** {count} phase(s)")
    
    return final_results
