"""
UNIVERSAL XRD PHASE IDENTIFIER FOR NANOMATERIALS
========================================================================
Professional phase identification system for ANY nanomaterial:
- Metal nanoparticles (Au, Ag, Cu, Pt, Pd, Ni, Fe, Co)
- Metal oxides (all transition metal oxides)
- Metal sulfides, selenides, tellurides
- Perovskites, spinels, layered materials
- MOFs, COFs, and hybrid materials
- 2D materials (graphene, TMDCs, MXenes)
========================================================================
"""

import numpy as np
import requests
import time
from typing import List, Dict, Tuple, Optional
import streamlit as st
from dataclasses import dataclass
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import concurrent.futures
from functools import lru_cache

# ------------------------------------------------------------
# UNIVERSAL NANOMATERIAL PARAMETERS
# ------------------------------------------------------------
@dataclass
class UniversalNanoParams:
    """Universal parameters for all nanomaterial types"""
    
    # Peak broadening tolerance based on expected size
    SIZE_TOLERANCE_MAP = {
        'ultra_nano': (0.1, 20),   # < 5 nm: 0.1-20° tolerance
        'nano': (0.08, 15),        # 5-20 nm: 0.08-15° tolerance  
        'submicron': (0.05, 10),   # 20-100 nm: 0.05-10° tolerance
        'micron': (0.03, 5),       # > 100 nm: 0.03-5° tolerance
    }
    
    # Common nanomaterial structure types with their characteristics
    NANOMATERIAL_FAMILIES = {
        'metal_nanoparticles': ['Au', 'Ag', 'Cu', 'Pt', 'Pd', 'Ni', 'Fe', 'Co'],
        'metal_oxides': ['TiO2', 'ZnO', 'Fe2O3', 'Fe3O4', 'CuO', 'Cu2O', 'NiO', 
                        'Co3O4', 'MnO2', 'Al2O3', 'SiO2', 'ZrO2', 'CeO2'],
        'metal_chalcogenides': ['MoS2', 'WS2', 'MoSe2', 'WSe2', 'CdS', 'CdSe', 
                               'CdTe', 'ZnS', 'ZnSe', 'PbS', 'PbSe'],
        'perovskites': ['MAPbI3', 'CsPbI3', 'BaTiO3', 'SrTiO3', 'LaMnO3'],
        'spinels': ['Fe3O4', 'CoFe2O4', 'MnFe2O4', 'ZnFe2O4'],
        'layered_materials': ['MoS2', 'WS2', 'BN', 'MoSe2', 'WSe2', 'Bi2Se3'],
        'carbon_allotropes': ['C', 'graphene', 'graphite', 'carbon_nanotubes'],
        'mofs_cofs': ['ZIF-8', 'MOF-5', 'UIO-66', 'HKUST-1'],
    }
    
    # Database priority based on material type
    DATABASE_PRIORITY = {
        'metal_nanoparticles': ['COD', 'ICSD', 'MaterialsProject'],
        'metal_oxides': ['COD', 'MaterialsProject', 'AFLOW'],
        'perovskites': ['MaterialsProject', 'AFLOW', 'COD'],
        '2d_materials': ['COD', '2DMatPedia', 'MaterialsProject'],
    }

# ------------------------------------------------------------
# UNIVERSAL PEAK ANALYSIS
# ------------------------------------------------------------
class UniversalPeakAnalyzer:
    """Universal peak analysis for all nanomaterial types"""
    
    @staticmethod
    def detect_peaks_universal(two_theta: np.ndarray, intensity: np.ndarray, 
                               min_snr: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Universal peak detection for nanomaterials with adaptive threshold
        """
        # Calculate noise level
        noise_level = np.std(intensity[:50]) if len(intensity) > 50 else np.std(intensity)
        baseline = np.percentile(intensity, 10)
        
        # Adaptive threshold based on signal-to-noise
        threshold = baseline + min_snr * noise_level
        
        # Find peaks
        peaks_idx, properties = find_peaks(
            intensity, 
            height=threshold,
            prominence=threshold/2,
            width=1,  # Minimum width for nanomaterials
            distance=5  # Minimum distance between peaks
        )
        
        return two_theta[peaks_idx], intensity[peaks_idx]
    
    @staticmethod
    def estimate_material_family(elements: List[str], peak_positions: List[float]) -> str:
        """
        Intelligently estimate material family from elements and peaks
        """
        if not elements:
            return 'unknown'
        
        elements_set = set(elements)
        
        # Check for metals
        common_metals = {'Au', 'Ag', 'Cu', 'Pt', 'Pd', 'Ni', 'Fe', 'Co'}
        if elements_set & common_metals:
            return 'metal_nanoparticles'
        
        # Check for oxides
        if 'O' in elements_set:
            # Check for transition metals
            transition_metals = {'Ti', 'Zn', 'Fe', 'Cu', 'Ni', 'Co', 'Mn', 'Cr', 'V'}
            if elements_set & transition_metals:
                return 'metal_oxides'
            # Check for rare earth oxides
            rare_earths = {'Ce', 'La', 'Nd', 'Pr', 'Sm', 'Eu', 'Gd'}
            if elements_set & rare_earths:
                return 'metal_oxides'
        
        # Check for sulfides/selenides/tellurides
        chalcogens = {'S', 'Se', 'Te'}
        if elements_set & chalcogens:
            return 'metal_chalcogenides'
        
        # Check for perovskites (ABX3)
        if len(elements) >= 3 and 'O' in elements:
            # Simple check for perovskite-like composition
            return 'perovskites'
        
        # Check for carbon materials
        if 'C' in elements_set and len(elements) <= 2:
            return 'carbon_allotropes'
        
        return 'unknown'
    
    @staticmethod
    def calculate_peak_quality_metrics(peaks_2theta: np.ndarray, 
                                      peaks_intensity: np.ndarray,
                                      wavelength: float) -> Dict:
        """
        Calculate quality metrics for peak matching
        """
        if len(peaks_2theta) == 0:
            return {}
        
        # Calculate d-spacings
        d_spacings = wavelength / (2 * np.sin(np.radians(peaks_2theta / 2)))
        
        # Peak intensity statistics
        intensity_ratio = peaks_intensity / np.max(peaks_intensity)
        
        # Peak distribution metrics
        angular_range = peaks_2theta.max() - peaks_2theta.min()
        peak_density = len(peaks_2theta) / angular_range if angular_range > 0 else 0
        
        # Estimate crystallinity from peak sharpness
        # (simplified - in practice would use FWHM)
        avg_intensity = np.mean(intensity_ratio)
        
        return {
            'n_peaks': len(peaks_2theta),
            'angular_range': float(angular_range),
            'peak_density': float(peak_density),
            'avg_intensity_ratio': float(avg_intensity),
            'd_spacing_range': (float(d_spacings.min()), float(d_spacings.max())),
            'quality_score': min(avg_intensity * peak_density * len(peaks_2theta) / 10, 1.0)
        }

# ------------------------------------------------------------
# UNIVERSAL DATABASE SEARCH
# ------------------------------------------------------------
class UniversalDatabaseSearcher:
    """Universal database search for all material types"""
    
    def __init__(self):
        self.databases = {
            'COD': self._search_cod_universal,
            'MaterialsProject': self._search_materials_project,
            'AFLOW': self._search_aflow,
            'ICSD': self._search_icsd,
            'OQMD': self._search_oqmd,
        }
        
        # Cache for database queries
        self.query_cache = {}
    
    @lru_cache(maxsize=100)
    def _search_cod_universal(self, elements: Tuple[str], max_results: int = 30) -> List[Dict]:
        """Universal COD search with robust error handling"""
        try:
            query = {
                "format": "json",
                "el": ",".join(elements),
                "maxresults": max_results
            }
            
            response = requests.get(
                "https://www.crystallography.net/cod/result",
                params=query,
                timeout=20
            )
            
            if response.status_code == 200:
                data = response.json()
                
                structures = []
                # COD returns a list of entries
                if isinstance(data, list):
                    for entry in data[:max_results]:
                        if isinstance(entry, dict) and 'codid' in entry:
                            structures.append({
                                'database': 'COD',
                                'id': str(entry['codid']),
                                'formula': entry.get('formula', ''),
                                'space_group': entry.get('sg', ''),
                                'cif_url': f"https://www.crystallography.net/cod/{entry['codid']}.cif",
                                'confidence': 0.8  # COD is experimental data
                            })
                
                return structures
                
        except Exception as e:
            st.warning(f"COD search: {str(e)[:100]}")
        
        return []
    
    def _search_materials_project(self, elements: Tuple[str], max_results: int = 20) -> List[Dict]:
        """Materials Project search (requires API key)"""
        # This would require an API key
        # For now, return empty - users can add their own key
        return []
    
    def _search_aflow(self, elements: Tuple[str], max_results: int = 20) -> List[Dict]:
        """AFLOW database search for inorganic compounds"""
        try:
            # AFLOW API endpoint for structure search
            formula = "".join(elements)
            url = f"http://aflowlib.duke.edu/search/API/?p={formula}&format=json"
            
            response = requests.get(url, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                structures = []
                
                for entry in data.get('results', [])[:max_results]:
                    structures.append({
                        'database': 'AFLOW',
                        'id': entry.get('auid', ''),
                        'formula': entry.get('compound', ''),
                        'space_group': entry.get('spacegroup', ''),
                        'cif_url': entry.get('cif_url', ''),
                        'confidence': 0.7
                    })
                
                return structures
                
        except:
            pass
        
        return []
    
    def _search_icsd(self, elements: Tuple[str], max_results: int = 20) -> List[Dict]:
        """ICSD search (would require subscription)"""
        # ICSD requires subscription, so just return empty
        return []
    
    def _search_oqmd(self, elements: Tuple[str], max_results: int = 20) -> List[Dict]:
        """Open Quantum Materials Database search"""
        try:
            # OQMD API endpoint
            elements_str = "-".join(elements)
            url = f"https://oqmd.org/api/structures?elements={elements_str}&limit={max_results}"
            
            response = requests.get(url, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                structures = []
                
                for entry in data.get('results', [])[:max_results]:
                    structures.append({
                        'database': 'OQMD',
                        'id': entry.get('id', ''),
                        'formula': entry.get('formula', ''),
                        'space_group': entry.get('spacegroup', ''),
                        'cif_url': f"https://oqmd.org/api/structures/{entry.get('id')}/cif",
                        'confidence': 0.6
                    })
                
                return structures
                
        except:
            pass
        
        return []
    
    def search_all_databases(self, elements: List[str], material_family: str = 'unknown') -> List[Dict]:
        """
        Search all available databases in parallel
        """
        if not elements:
            return []
        
        elements_tuple = tuple(sorted(elements))
        
        # Check cache
        cache_key = (elements_tuple, material_family)
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        st.info(f"🔍 Searching databases for: {', '.join(elements)}")
        
        all_structures = []
        
        # Determine which databases to search based on material family
        if material_family in UniversalNanoParams.DATABASE_PRIORITY:
            db_priority = UniversalNanoParams.DATABASE_PRIORITY[material_family]
        else:
            # Default priority
            db_priority = ['COD', 'MaterialsProject', 'AFLOW', 'OQMD']
        
        # Search databases in parallel for speed
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for db_name in db_priority[:3]:  # Limit to 3 databases for speed
                if db_name in self.databases:
                    futures.append(
                        executor.submit(self.databases[db_name], elements_tuple, 15)
                    )
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    results = future.result(timeout=10)
                    if results:
                        all_structures.extend(results)
                except Exception as e:
                    continue
        
        # Remove duplicates based on formula and space group
        unique_structures = []
        seen = set()
        
        for struct in all_structures:
            key = (struct.get('formula', ''), struct.get('space_group', ''))
            if key not in seen:
                seen.add(key)
                unique_structures.append(struct)
        
        # Cache results
        self.query_cache[cache_key] = unique_structures
        
        st.success(f"✅ Found {len(unique_structures)} unique structures from databases")
        
        return unique_structures

# ------------------------------------------------------------
# UNIVERSAL MATCHING ALGORITHM
# ------------------------------------------------------------
class UniversalPatternMatcher:
    """Universal pattern matching for all material types"""
    
    @staticmethod
    def match_pattern_universal(exp_d: np.ndarray, exp_intensity: np.ndarray,
                               sim_d: np.ndarray, sim_intensity: np.ndarray,
                               material_family: str = 'unknown') -> float:
        """
        Universal pattern matching with family-specific optimizations
        """
        if len(exp_d) == 0 or len(sim_d) == 0:
            return 0.0
        
        # Family-specific matching parameters
        params = UniversalPatternMatcher._get_family_params(material_family)
        
        match_scores = []
        intensity_weights = []
        
        for i, d_exp in enumerate(exp_d):
            # Adaptive tolerance based on material family and peak intensity
            base_tolerance = params['base_tolerance']
            intensity_factor = exp_intensity[i] / np.max(exp_intensity)
            
            # Higher intensity peaks get tighter tolerance
            peak_tolerance = base_tolerance * (1.5 - 0.5 * intensity_factor)
            
            # Find closest simulated peak
            d_errors = np.abs(sim_d - d_exp) / d_exp
            min_error_idx = np.argmin(d_errors)
            min_error = d_errors[min_error_idx]
            
            if min_error < peak_tolerance:
                # Calculate match quality
                match_quality = 1.0 - (min_error / peak_tolerance)
                
                # Intensity correlation
                if len(sim_intensity) > min_error_idx:
                    intensity_match = 1.0 - abs(exp_intensity[i] - sim_intensity[min_error_idx]) / max(exp_intensity[i], sim_intensity[min_error_idx])
                    match_quality *= (0.7 + 0.3 * intensity_match)
                
                match_scores.append(match_quality)
                
                # Weight by experimental intensity
                weight = exp_intensity[i] / np.sum(exp_intensity)
                intensity_weights.append(weight)
        
        if not match_scores:
            return 0.0
        
        # Weighted average match score
        weighted_score = np.average(match_scores, weights=intensity_weights)
        
        # Coverage penalty (how many experimental peaks were matched)
        coverage = len(match_scores) / len(exp_d)
        coverage_penalty = 0.3 + 0.7 * coverage  # 30% penalty for poor coverage
        
        final_score = weighted_score * coverage_penalty
        
        # Family-specific adjustments
        if material_family in ['metal_nanoparticles', 'carbon_allotropes']:
            # These often have fewer peaks
            if len(exp_d) < 5:
                final_score *= 1.2  # Boost for materials with few peaks
        
        return min(final_score, 1.0)
    
    @staticmethod
    def _get_family_params(family: str) -> Dict:
        """Get matching parameters for specific material family"""
        params = {
            'base_tolerance': 0.03,  # Default 3% tolerance
            'intensity_weight': 0.3,
            'coverage_weight': 0.7,
        }
        
        # Family-specific adjustments
        family_adjustments = {
            'metal_nanoparticles': {'base_tolerance': 0.04, 'intensity_weight': 0.4},
            'metal_oxides': {'base_tolerance': 0.025, 'intensity_weight': 0.35},
            'metal_chalcogenides': {'base_tolerance': 0.03, 'intensity_weight': 0.3},
            'perovskites': {'base_tolerance': 0.02, 'intensity_weight': 0.4},
            'spinels': {'base_tolerance': 0.025, 'intensity_weight': 0.35},
            'carbon_allotropes': {'base_tolerance': 0.05, 'intensity_weight': 0.2},
            'unknown': {'base_tolerance': 0.03, 'intensity_weight': 0.3},
        }
        
        if family in family_adjustments:
            params.update(family_adjustments[family])
        
        return params

# ------------------------------------------------------------
# MAIN UNIVERSAL IDENTIFICATION ENGINE
# ------------------------------------------------------------
def identify_phases_universal(two_theta: np.ndarray, intensity: np.ndarray,
                            wavelength: float, elements: List[str]) -> List[Dict]:
    """
    UNIVERSAL phase identification for ANY nanomaterial
    """
    st.info("🔬 Running universal nanomaterial phase identification...")
    
    # --------------------------------------------------------
    # STEP 1: UNIVERSAL PEAK DETECTION
    # --------------------------------------------------------
    peak_analyzer = UniversalPeakAnalyzer()
    
    exp_peaks_2theta, exp_intensities = peak_analyzer.detect_peaks_universal(
        two_theta, intensity
    )
    
    if len(exp_peaks_2theta) < 2:
        st.warning("Insufficient peaks for reliable identification")
        return []
    
    st.success(f"✅ Detected {len(exp_peaks_2theta)} peaks for matching")
    
    # Calculate d-spacings
    exp_d = wavelength / (2 * np.sin(np.radians(exp_peaks_2theta / 2)))
    
    # Normalize intensities for matching
    exp_intensities_norm = exp_intensities / np.max(exp_intensities)
    
    # --------------------------------------------------------
    # STEP 2: ESTIMATE MATERIAL FAMILY
    # --------------------------------------------------------
    material_family = peak_analyzer.estimate_material_family(
        elements, exp_peaks_2theta
    )
    
    st.info(f"📊 Material family estimated: {material_family}")
    
    # Calculate peak quality metrics
    peak_quality = peak_analyzer.calculate_peak_quality_metrics(
        exp_peaks_2theta, exp_intensities, wavelength
    )
    
    # --------------------------------------------------------
    # STEP 3: UNIVERSAL DATABASE SEARCH
    # --------------------------------------------------------
    db_searcher = UniversalDatabaseSearcher()
    
    database_structures = db_searcher.search_all_databases(elements, material_family)
    
    if not database_structures:
        st.warning("No structures found in databases. Try different elements.")
        return []
    
    # --------------------------------------------------------
    # STEP 4: PATTERN SIMULATION AND MATCHING
    # --------------------------------------------------------
    matcher = UniversalPatternMatcher()
    results = []
    
    st.info(f"🧪 Simulating and matching {len(database_structures)} structures...")
    
    progress_bar = st.progress(0)
    
    for i, struct in enumerate(database_structures):
        try:
            # Update progress
            progress = (i + 1) / len(database_structures)
            progress_bar.progress(progress)
            
            # Fetch CIF
            cif_url = struct.get('cif_url')
            if not cif_url:
                continue
            
            # Download CIF
            cif_response = requests.get(cif_url, timeout=15)
            if cif_response.status_code != 200:
                continue
            
            cif_text = cif_response.text
            
            # Simulate XRD pattern
            from pymatgen.io.cif import CifParser
            from pymatgen.analysis.diffraction.xrd import XRDCalculator
            
            parser = CifParser.from_string(cif_text)
            structure = parser.get_structures()[0]
            calc = XRDCalculator(wavelength=wavelength)
            pattern = calc.get_pattern(structure, two_theta_range=(5, 80))
            
            # Simulated data
            sim_d = wavelength / (2 * np.sin(np.radians(pattern.x / 2)))
            sim_intensity = pattern.y / np.max(pattern.y) if len(pattern.y) > 0 else np.zeros_like(pattern.x)
            
            # Match patterns
            match_score = matcher.match_pattern_universal(
                exp_d, exp_intensities_norm,
                sim_d, sim_intensity,
                material_family
            )
            
            # Threshold for nanomaterials (lower due to peak broadening)
            if match_score < 0.35:  # 35% match threshold for nanomaterials
                continue
            
            # Determine confidence level
            if match_score >= 0.7:
                confidence = "confirmed"
            elif match_score >= 0.5:
                confidence = "probable"
            elif match_score >= 0.35:
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
                "score": round(match_score, 3),
                "confidence_level": confidence,
                "database": struct.get('database', 'Unknown'),
                "material_family": material_family,
                "peak_quality": peak_quality,
                "n_peaks_matched": len(exp_d),
                "structure": structure,
            })
            
        except Exception as e:
            # Silently continue on individual structure errors
            continue
    
    progress_bar.empty()
    
    # --------------------------------------------------------
    # STEP 5: RESULTS PROCESSING
    # --------------------------------------------------------
    if not results:
        st.warning("No phases identified with sufficient confidence")
        return []
    
    # Remove duplicates (same formula and space group)
    unique_results = []
    seen = set()
    
    for result in results:
        key = (result["phase"], result["space_group"])
        if key not in seen:
            seen.add(key)
            unique_results.append(result)
    
    # Sort by score
    final_results = sorted(unique_results, key=lambda x: x["score"], reverse=True)
    
    # --------------------------------------------------------
    # STEP 6: SCIENTIFIC REPORT
    # --------------------------------------------------------
    st.success(f"✅ Identified {len(final_results)} potential phases")
    
    with st.expander("📊 Scientific Analysis Report", expanded=False):
        st.markdown(f"### **Material Analysis Summary**")
        st.markdown(f"- **Estimated family**: {material_family}")
        st.markdown(f"- **Peaks detected**: {len(exp_peaks_2theta)}")
        st.markdown(f"- **Peak quality score**: {peak_quality.get('quality_score', 0):.2f}/1.0")
        st.markdown(f"- **Angular range**: {peak_quality.get('angular_range', 0):.1f}°")
        st.markdown(f"- **Databases searched**: {len(set(r['database'] for r in final_results))}")
        
        # Show top matches
        if final_results:
            st.markdown("### **Top Phase Matches**")
            for i, result in enumerate(final_results[:3]):
                st.markdown(f"{i+1}. **{result['phase']}** ({result['crystal_system']}) - "
                          f"Score: {result['score']:.3f} [{result['confidence_level']}]")
    
    return final_results
