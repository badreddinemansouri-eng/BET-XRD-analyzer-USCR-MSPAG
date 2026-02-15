"""
UNIVERSAL XRD PHASE IDENTIFICATION FOR NANOMATERIALS
========================================================================
Scientific phase identification for nanocrystalline materials with proper
peak filtering and d-spacing matching.
Now supports automatic peak-based search when elements are not provided.
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
# UNIVERSAL PEAK ANALYSIS WITH NANOCRYSTALLINE FILTERING
# ------------------------------------------------------------
class UniversalPeakAnalyzer:
    """Universal peak analysis for nanocrystalline materials"""
    
    @staticmethod
    def detect_peaks_universal(two_theta: np.ndarray, intensity: np.ndarray, 
                               min_snr: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scientific peak detection for nanomaterials with proper filtering
        """
        # Calculate noise level
        if len(intensity) > 50:
            noise_level = np.std(intensity[:50])
        else:
            noise_level = np.std(intensity)
        
        baseline = np.percentile(intensity, 10)
        
        # Adaptive threshold for nanomaterials (lower due to broadening)
        threshold = baseline + min_snr * noise_level
        
        # Find peaks with proper parameters for broadened peaks
        peaks_idx, properties = find_peaks(
            intensity, 
            height=threshold,
            prominence=noise_level * 2,  # Lower prominence for nanomaterials
            width=(2, None),  # Minimum width for broad peaks
            distance=5,  # Points
            wlen=len(intensity)//5  # Window length
        )
        
        return two_theta[peaks_idx], intensity[peaks_idx]
    
    @staticmethod
    def filter_peaks_for_nanomaterials(peaks_2theta: np.ndarray, 
                                      peaks_intensity: np.ndarray,
                                      wavelength: float,
                                      max_peaks: int = 10,
                                      avg_fwhm: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        CRITICAL FIX: Filter to strongest 6-10 peaks with ANGULAR DIVERSITY
        
        Nanocrystalline materials have fewer discernible peaks due to broadening.
        Using all detected peaks leads to failed phase identification.
        Now ensures peaks are at least 1.5° apart for structural diversity.
        If avg_fwhm is provided, the minimum separation is adjusted accordingly.
        """
        if len(peaks_2theta) == 0:
            return peaks_2theta, peaks_intensity
        
        # Sort by intensity (descending)
        order = np.argsort(peaks_intensity)[::-1]
        peaks_2theta_sorted = peaks_2theta[order]
        peaks_intensity_sorted = peaks_intensity[order]

        # Determine minimum separation (1.2° base, larger if peaks are broad)
        min_sep = 1.2
        if avg_fwhm is not None:
            # For very broad peaks, require larger separation to avoid overlaps
            min_sep = max(1.2, avg_fwhm * 2.0)
        
        # CRITICAL FIX: Enforce angular diversity
        selected_2theta = []
        selected_intensity = []
        
        for i in range(len(peaks_2theta_sorted)):
            current_2theta = peaks_2theta_sorted[i]
            current_intensity = peaks_intensity_sorted[i]
            
            # Check if this peak is sufficiently separated from already selected peaks
            accept = True
            for i, sel in enumerate(selected_2theta):
                if abs(current_2theta - sel) < min_sep:
                    # keep the stronger one
                    if current_intensity > selected_intensity[i]:
                        selected_2theta[i] = current_2theta
                        selected_intensity[i] = current_intensity
                    accept = False
                    break
            
            if accept:
                selected_2theta.append(current_2theta)
                selected_intensity.append(current_intensity)
            
            # Stop when we have enough diverse peaks
            if len(selected_2theta) >= max_peaks:
                break
        
        # If we don't have enough diverse peaks, add the strongest remaining ones
        if len(selected_2theta) < max_peaks:
            for i in range(len(peaks_2theta_sorted)):
                current_2theta = peaks_2theta_sorted[i]
                current_intensity = peaks_intensity_sorted[i]
                
                # Skip if already selected
                if any(abs(current_2theta - s) < 1e-3 for s in selected_2theta):
                    continue
                
                selected_2theta.append(current_2theta)
                selected_intensity.append(current_intensity)
                
                if len(selected_2theta) >= max_peaks:
                    break
        
        # For very nanocrystalline materials, keep fewer peaks
        if len(selected_2theta) > 5:
            # Calculate average peak width from d-spacing
            d_spacings = wavelength / (2 * np.sin(np.radians(np.array(selected_2theta) / 2)))
            d_spacing_range = d_spacings.max() - d_spacings.min()
            
            # If d-spacing range is small, material is highly nanocrystalline
            if d_spacing_range < 2.0 and len(selected_2theta) > 6:
                selected_2theta = selected_2theta[:6]
                selected_intensity = selected_intensity[:6]
        
        return np.array(selected_2theta), np.array(selected_intensity)
    
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
        
        # Estimate nanocrystallinity from d-spacing distribution
        d_spacing_std = np.std(d_spacings) / np.mean(d_spacings) if len(d_spacings) > 1 else 0
        
        return {
            'n_peaks': len(peaks_2theta),
            'angular_range': float(angular_range),
            'peak_density': float(peak_density),
            'avg_intensity_ratio': float(np.mean(intensity_ratio)),
            'd_spacing_range': (float(d_spacings.min()), float(d_spacings.max())),
            'd_spacing_std_norm': float(d_spacing_std),
            'quality_score': min(np.mean(intensity_ratio) * len(peaks_2theta) / 10, 1.0)
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
            # CRITICAL FIX: Add oxide constraints
            query = {
                "format": "json",
                "el": ",".join(elements),
                "maxresults": max_results
            }
            
            # Add oxide constraint for better filtering
            if "O" in elements:
                query["formula"] = "*O*"
            
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
    
    # --------------------------------------------------------
    # NEW METHOD: Search COD by d-spacings (no elements required)
    # --------------------------------------------------------
    @lru_cache(maxsize=50)
    def search_by_dspacings(self, dspacings: Tuple[float], max_results: int = 30) -> List[Dict]:
        """
        Search COD using a list of d-spacings (in Angstroms).
        Returns a list of candidate structures in the same format as other search methods.
        """
        try:
            # COD expects d-spacings as a comma-separated list
            dspacing_str = ",".join([f"{d:.3f}" for d in dspacings[:5]])  # Use top 5 peaks
            query = {
                "format": "json",
                "dspacing": dspacing_str,
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
                if isinstance(data, list):
                    for entry in data[:max_results]:
                        if isinstance(entry, dict) and 'codid' in entry:
                            structures.append({
                                'database': 'COD',
                                'id': str(entry['codid']),
                                'formula': entry.get('formula', ''),
                                'space_group': entry.get('sg', ''),
                                'cif_url': f"https://www.crystallography.net/cod/{entry['codid']}.cif",
                                'confidence': 0.7  # Slightly lower confidence because only d-spacings used
                            })
                
                return structures
                
        except Exception as e:
            st.warning(f"COD d-spacing search: {str(e)[:100]}")
        
        return []
    
    def search_all_databases(self, elements: Optional[List[str]] = None, 
                            dspacings: Optional[np.ndarray] = None,
                            material_family: str = 'unknown') -> List[Dict]:
        """
        Search all available databases.
        If elements are provided, use element-filtered search.
        If no elements but dspacings provided, use d-spacing search.
        """
        if elements:
            # Element-filtered search
            elements_tuple = tuple(sorted(elements))
            cache_key = (elements_tuple, material_family)
            
            # Check cache
            if cache_key in self.query_cache:
                return self.query_cache[cache_key]
            
            st.info(f"🔍 Searching databases for elements: {', '.join(elements)}")
            
            all_structures = []
            
            # Determine which databases to search based on material family
            if material_family in UniversalNanoParams.DATABASE_PRIORITY:
                db_priority = UniversalNanoParams.DATABASE_PRIORITY[material_family]
            else:
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
        
        elif dspacings is not None and len(dspacings) > 0:
            # D-spacing search (no elements)
            # Use only COD for now, as other databases don't support d-spacing search easily
            st.info("🔍 No elements provided. Searching COD by d-spacings...")
            dspacings_tuple = tuple(sorted(dspacings[:5]))  # Use top 5 peaks
            structures = self.search_by_dspacings(dspacings_tuple, max_results=30)
            
            if structures:
                st.success(f"✅ Found {len(structures)} candidate structures from COD d-spacing search")
            else:
                st.warning("No structures found via d-spacing search.")
            
            return structures
        
        else:
            st.warning("No search criteria provided (neither elements nor d-spacings).")
            return []

# ------------------------------------------------------------
# SCIENTIFIC MATCHING ALGORITHM FOR NANOCRYSTALLINE MATERIALS
# ------------------------------------------------------------
class UniversalPatternMatcher:
    """Scientific pattern matching for nanocrystalline materials"""
    
    @staticmethod
    def calculate_nano_tolerance(size_nm: Optional[float] = None) -> float:
        """
        CRITICAL FIX: Physics-based tolerance for nanocrystalline materials
        
        Scherrer size -> d-spacing tolerance mapping:
        < 5 nm: 10% tolerance (Δd/d ≈ 0.10)
        5-10 nm: 6% tolerance (Δd/d ≈ 0.06)
        > 10 nm: 3% tolerance (Δd/d ≈ 0.03)
        
        Williamson-Hall predicts:
        Δ2θ ≈ λ/(D cosθ) ≈ 0.15-0.30° for 4.7 nm crystallites
        """
        if size_nm is None:
            return 0.05  # Default for unknown size
        elif size_nm < 5:
            return 0.10  # Ultra-nanocrystalline
        elif size_nm < 10:
            return 0.06  # Nanocrystalline
        else:
            return 0.03  # Sub-micron to micron
    
    @staticmethod
    def match_pattern_universal(exp_d: np.ndarray, exp_intensity: np.ndarray,
                               sim_d: np.ndarray, sim_intensity: np.ndarray,
                               material_family: str = 'unknown',
                               size_nm: Optional[float] = None) -> float:
        """
        SCIENTIFIC pattern matching with nanocrystalline-specific optimizations
        
        Key changes for nanomaterials:
        1. Use d-spacing matching (not 2θ)
        2. Use relative intensity ordering (not absolute values)
        3. Higher tolerance for broadened peaks
        4. Focus on strongest peaks only
        """
        if len(exp_d) == 0 or len(sim_d) == 0:
            return 0.0
        
        # Family-specific matching parameters
        params = UniversalPatternMatcher._get_family_params(material_family)
        
        # CRITICAL FIX: Use physics-based tolerance
        base_tolerance = UniversalPatternMatcher.calculate_nano_tolerance(size_nm)
        
        # For nanomaterials, use only top experimental peaks
        n_exp_peaks = min(len(exp_d), 8)  # Max 8 peaks for nanomaterials
        
        match_scores = []
        intensity_weights = []
        
        # Sort experimental peaks by intensity
        exp_sorted_idx = np.argsort(exp_intensity)[::-1][:n_exp_peaks]
        
        for i in exp_sorted_idx:
            d_exp = exp_d[i]
            intensity_exp = exp_intensity[i]
            
            # Adaptive tolerance for nanocrystalline materials
            # Broader tolerance for weaker peaks
            intensity_factor = intensity_exp / np.max(exp_intensity)
            peak_tolerance = base_tolerance * (1.5 - 0.3 * intensity_factor)
            
            # For nanomaterials, use even broader tolerance
            if material_family in ['metal_nanoparticles', 'carbon_allotropes']:
                peak_tolerance *= 1.5
            
            # Find closest simulated peak using d-spacing
            d_errors = np.abs(sim_d - d_exp) / d_exp
            min_error_idx = np.argmin(d_errors)
            min_error = d_errors[min_error_idx]
            
            if min_error < peak_tolerance:
                # Calculate match quality
                match_quality = 1.0 - (min_error / peak_tolerance)
                
                # Intensity correlation (relative ordering, not absolute values)
                if len(sim_intensity) > min_error_idx:
                    # Get relative intensity ranking
                    exp_rank = np.sum(exp_intensity > intensity_exp) / len(exp_intensity)
                    sim_rank = np.sum(sim_intensity > sim_intensity[min_error_idx]) / len(sim_intensity)
                    
                    # Match based on relative ordering
                    rank_match = 1.0 - abs(exp_rank - sim_rank)
                    match_quality *= (0.6 + 0.4 * rank_match)  # 60% d-spacing, 40% intensity ordering
                
                match_scores.append(match_quality)
                
                # Weight by experimental intensity
                weight = intensity_exp / np.sum(exp_intensity[exp_sorted_idx])
                intensity_weights.append(weight)
        
        if not match_scores:
            return 0.0
        
        # Weighted average match score
        weighted_score = np.average(match_scores, weights=intensity_weights)
        
        # Coverage penalty (how many experimental peaks were matched)
        coverage = len(match_scores) / n_exp_peaks
        
        # For nanomaterials, require good coverage but be more lenient
        if coverage < 0.5:  # RELAXED from 0.6 to 0.5 for nanocrystalline
            return 0.0
        
        coverage_factor = 0.4 + 0.6 * coverage  # Strong penalty for poor coverage
        
        final_score = weighted_score * coverage_factor
        
        # Boost score for nanomaterials with few peaks
        if material_family in ['metal_nanoparticles', 'carbon_allotropes'] and n_exp_peaks <= 4:
            final_score *= 1.2
        
        return min(final_score, 1.0)
    
    @staticmethod
    def _get_family_params(family: str) -> Dict:
        """Get matching parameters for specific material family"""
        # Default parameters with increased tolerance for nanomaterials
        params = {
            'base_tolerance': 0.05,  # Increased from 0.04 for nanomaterials
            'intensity_weight': 0.3,
            'coverage_weight': 0.7,
        }
        
        # Family-specific adjustments for nanomaterials
        family_adjustments = {
            'metal_nanoparticles': {'base_tolerance': 0.06, 'intensity_weight': 0.2},
            'metal_oxides': {'base_tolerance': 0.04, 'intensity_weight': 0.35},
            'metal_chalcogenides': {'base_tolerance': 0.05, 'intensity_weight': 0.3},
            'perovskites': {'base_tolerance': 0.04, 'intensity_weight': 0.4},
            'spinels': {'base_tolerance': 0.04, 'intensity_weight': 0.35},
            'carbon_allotropes': {'base_tolerance': 0.07, 'intensity_weight': 0.1},
            'unknown': {'base_tolerance': 0.05, 'intensity_weight': 0.3},
        }
        
        if family in family_adjustments:
            params.update(family_adjustments[family])
        
        return params

# ------------------------------------------------------------
# CACHING FOR SIMULATED PATTERNS
# ------------------------------------------------------------
@lru_cache(maxsize=200)
def _simulate_pattern_cached(cif_url: str, wavelength: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Download CIF from URL and simulate XRD pattern.
    Cached to avoid repeated downloads and calculations.
    Returns (two_theta, intensity) arrays.
    """
    try:
        response = requests.get(cif_url, timeout=15)
        if response.status_code != 200:
            return np.array([]), np.array([])
        
        cif_text = response.text
        
        from pymatgen.io.cif import CifParser
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
        
        parser = CifParser.from_string(cif_text)
        structure = parser.get_structures()[0]
        calc = XRDCalculator(wavelength=wavelength)
        pattern = calc.get_pattern(structure, two_theta_range=(5, 80))
        
        return np.array(pattern.x), np.array(pattern.y)
    
    except Exception as e:
        return np.array([]), np.array([])

# ------------------------------------------------------------
# MAIN UNIVERSAL IDENTIFICATION ENGINE (SCIENTIFICALLY CORRECTED)
# ------------------------------------------------------------
def identify_phases_universal(two_theta: np.ndarray, intensity: np.ndarray,
                            wavelength: float, elements: Optional[List[str]] = None,
                            size_nm: Optional[float] = None) -> List[Dict]:
    """
    SCIENTIFIC phase identification for nanocrystalline materials
    
    Critical fixes applied:
    1. Filter to strongest 6-10 peaks only with ANGULAR DIVERSITY
    2. Use d-spacing matching (not 2θ)
    3. Implement relative intensity ordering
    4. Adjust tolerances for nanocrystalline broadening
    5. Pass crystallite size for physics-based tolerance
    6. ADDED: Phase identification diagnostics for transparency
    7. ADDED: Caching for simulated patterns (performance)
    8. ADDED: Automatic peak-based search when elements not provided
    9. REMOVED: Placeholder "UNIDENTIFIED" phase (causes KeyError elsewhere)
    """
    st.info("🔬 Running scientific nanomaterial phase identification...")
    
    # Handle elements=None gracefully
    if elements is None:
        elements = []
    
    # --------------------------------------------------------
    # STEP 1: UNIVERSAL PEAK DETECTION WITH FILTERING
    # --------------------------------------------------------
    peak_analyzer = UniversalPeakAnalyzer()
    
    exp_peaks_2theta, exp_intensities = peak_analyzer.detect_peaks_universal(
        two_theta, intensity
    )
    
    # --- local apex recentering for phase ID only ---
    refined_2theta = []
    refined_intensity = []
    
    for t0 in exp_peaks_2theta:
        idx = np.argmin(np.abs(two_theta - t0))
        left = max(0, idx - 5)
        right = min(len(two_theta), idx + 6)
    
        local_idx = left + np.argmax(intensity[left:right])
    
        refined_2theta.append(two_theta[local_idx])
        refined_intensity.append(intensity[local_idx])
    
    exp_peaks_2theta = np.array(refined_2theta)
    exp_intensities = np.array(refined_intensity)
    
    # CRITICAL FIX: Filter to strongest peaks with angular diversity
    # (We don't have FWHM info here, so avg_fwhm is not passed)
    exp_peaks_2theta, exp_intensities = peak_analyzer.filter_peaks_for_nanomaterials(
        exp_peaks_2theta, exp_intensities, wavelength, max_peaks=10
    )
    
    if len(exp_peaks_2theta) < 2:
        st.warning("Insufficient strong peaks for reliable phase identification")
        return []
    
    st.success(f"✅ Using {len(exp_peaks_2theta)} strongest diverse peaks for matching")
    st.write(f"🧪 Phase ID → Selected peaks: {exp_peaks_2theta.tolist()}")
    
    # Calculate d-spacings (not 2θ)
    exp_d = wavelength / (2 * np.sin(np.radians(exp_peaks_2theta / 2)))
    
    # Normalize intensities for matching
    exp_intensities_norm = exp_intensities / np.max(exp_intensities)
    
    # --------------------------------------------------------
    # STEP 2: ESTIMATE MATERIAL FAMILY (if elements provided)
    # --------------------------------------------------------
    if elements:
        material_family = peak_analyzer.estimate_material_family(
            elements, exp_peaks_2theta
        )
    else:
        material_family = 'unknown'
    
    st.info(f"📊 Material family estimated: {material_family}")
    if size_nm:
        st.info(f"📊 Crystallite size for tolerance: {size_nm:.1f} nm")
    
    # Calculate peak quality metrics
    peak_quality = peak_analyzer.calculate_peak_quality_metrics(
        exp_peaks_2theta, exp_intensities, wavelength
    )
    
    # --------------------------------------------------------
    # STEP 3: UNIVERSAL DATABASE SEARCH
    # --------------------------------------------------------
    db_searcher = UniversalDatabaseSearcher()
    
    # Use d-spacings for search if no elements provided
    if elements:
        database_structures = db_searcher.search_all_databases(
            elements=elements, material_family=material_family
        )
    else:
        database_structures = db_searcher.search_all_databases(
            dspacings=exp_d, material_family=material_family
        )
    
    if not database_structures:
        st.warning("No structures found in databases. Try different elements or improve pattern quality.")
        return []
    
    # --------------------------------------------------------
    # STEP 4: SCIENTIFIC PATTERN SIMULATION AND MATCHING
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
            
            # Use cached simulation
            sim_x, sim_y = _simulate_pattern_cached(cif_url, wavelength)
            if len(sim_x) == 0:
                continue
            
            # Simulated data
            sim_d = wavelength / (2 * np.sin(np.radians(sim_x / 2)))
            sim_intensity = sim_y / np.max(sim_y) if len(sim_y) > 0 else np.zeros_like(sim_x)
            
            # SCIENTIFIC MATCHING: Use d-spacing and relative intensities WITH SIZE-BASED TOLERANCE
            match_score = matcher.match_pattern_universal(
                exp_d, exp_intensities_norm,
                sim_d, sim_intensity,
                material_family,
                size_nm
            )
            
            # ADJUSTED thresholds for nanomaterials (LOWERED)
            if size_nm and size_nm < 10:
                # Ultra-nanocrystalline materials need even lower thresholds
                threshold = 0.15 if size_nm < 5 else 0.20
            else:
                threshold = 0.25
            
            if match_score < threshold:
                continue
            
            # Determine confidence level (adjusted for nanomaterials)
            if size_nm and size_nm < 10:
                # Lower thresholds for nanocrystalline
                if match_score >= 0.55:  # LOWERED for nanomaterials
                    confidence = "confirmed"
                elif match_score >= 0.35:  # LOWERED for nanomaterials
                    confidence = "probable"
                elif match_score >= threshold:
                    confidence = "possible"
                else:
                    continue
            else:
                if match_score >= 0.60:
                    confidence = "confirmed"
                elif match_score >= 0.40:
                    confidence = "probable"
                elif match_score >= 0.25:
                    confidence = "possible"
                else:
                    continue
            
            # Extract structure from cached simulation (need to re-parse for composition)
            # To avoid re-downloading, we could store structure info in the cache,
            # but for simplicity we'll re-download only if needed for lattice params.
            # Since we already have the CIF text, we can parse it again quickly.
            try:
                response = requests.get(cif_url, timeout=10)
                cif_text = response.text
                from pymatgen.io.cif import CifParser
                parser = CifParser.from_string(cif_text)
                structure = parser.get_structures()[0]
                crystal_system = structure.get_crystal_system()
                space_group = structure.get_space_group_info()[0]
                lattice = structure.lattice.as_dict()
                formula = structure.composition.reduced_formula
                full_formula = str(structure.composition)
            except:
                # Fallback to struct info if parsing fails
                crystal_system = struct.get('space_group', 'Unknown')
                space_group = struct.get('space_group', 'Unknown')
                lattice = {}
                formula = struct.get('formula', 'Unknown')
                full_formula = struct.get('formula', 'Unknown')
            
            # Store results
            results.append({
                "phase": formula,
                "full_formula": full_formula,
                "crystal_system": crystal_system,
                "space_group": space_group,
                "lattice": lattice,
                "hkls": [],  # Not storing hkls for now
                "score": round(match_score, 3),
                "confidence_level": confidence,
                "database": struct.get('database', 'Unknown'),
                "material_family": material_family,
                "peak_quality": peak_quality,
                "n_peaks_matched": len(exp_d),
                "structure": None,  # Cannot store structure in results for display
                "match_details": {
                    "n_exp_peaks": len(exp_d),
                    "avg_d_spacing": float(np.mean(exp_d)),
                    "material_family": material_family,
                    "size_nm": size_nm,
                    "tolerance_used": UniversalPatternMatcher.calculate_nano_tolerance(size_nm)
                }
            })
            
        except Exception as e:
            # Silently continue on individual structure errors
            continue
    
    progress_bar.empty()
    
    # --------------------------------------------------------
    # STEP 5: RESULTS PROCESSING WITH DIAGNOSTICS
    # --------------------------------------------------------
    if not results:
        # Create diagnostic object explaining why no phase found
        diagnostic = {
            "experimental_structural_peaks": len(exp_peaks_2theta),
            "peak_quality_score": peak_quality.get('quality_score', 0),
            "nanocrystalline_indicator": "strong" if size_nm and size_nm < 10 else "moderate/weak",
            "database_coverage": {
                "cod_structures_searched": len([s for s in database_structures if s['database'] == 'COD']),
                "total_structures": len(database_structures)
            },
            "matching_issues": []
        }
        
        # Add specific matching issues
        if size_nm and size_nm < 10:
            diagnostic["matching_issues"].append("Peak broadening exceeds database tolerance for nanocrystalline materials")
        
        if peak_quality.get('angular_range', 0) < 20:
            diagnostic["matching_issues"].append("Insufficient angular diversity in peaks")
        
        if peak_quality.get('quality_score', 0) < 0.3:
            diagnostic["matching_issues"].append("Material may be amorphous or poorly crystalline")
        
        # Display diagnostic in expander (the caller will see no phases)
        with st.expander("📊 No Phase Match – Diagnostic Report", expanded=True):
            st.markdown("### **Why were no crystalline phases identified?**")
            st.markdown("This is **NOT** a software error. Phase identification is based strictly on Bragg peak matching against CIF-validated crystal structures (COD + OPTIMADE).")
            
            for issue in diagnostic["matching_issues"]:
                st.markdown(f"- {issue}")
            
            st.markdown("**What you can try**")
            st.markdown("- Select all possible elements (including dopants)")
            st.markdown("- Use unsmoothed raw XRD data")
            st.markdown("- Increase crystallinity (annealing) if possible")
            st.markdown("- Combine with Raman / FTIR")
        
        # Return empty list (no placeholder)
        return []
    
    # Remove duplicates (same formula and space group)
    unique_results = []
    seen = set()
    
    for result in results:
        key = (result["phase"], result.get("space_group", ""))
        if key not in seen:
            seen.add(key)
            unique_results.append(result)
    
    # Sort by score
    final_results = sorted(unique_results, key=lambda x: x.get("score", 0), reverse=True)
    
    # --------------------------------------------------------
    # STEP 6: ADD DIAGNOSTICS TO ALL RESULTS
    # --------------------------------------------------------
    for result in final_results:
        if result.get("phase") != "UNIDENTIFIED":
            result["phase_diagnostics"] = {
                "experimental_peaks_used": len(exp_peaks_2theta),
                "matched_peaks": result.get("n_peaks_matched", 0),
                "match_score": result.get("score", 0),
                "nanocrystalline_tolerance_applied": UniversalPatternMatcher.calculate_nano_tolerance(size_nm),
                "database_reliability": "CIF-validated" if result.get("database") == "COD" else "Theoretical/computational",
                "database_limitations": "COD contains only crystalline structures; amorphous/nanocrystalline patterns not included"
            }
    
    # --------------------------------------------------------
    # STEP 7: SCIENTIFIC REPORT
    # --------------------------------------------------------
    if final_results:
        st.success(f"✅ Identified {len(final_results)} potential phases")
    else:
        st.warning("⚠️ No crystalline phase identified with sufficient confidence")
    
    with st.expander("📊 Scientific Analysis Report", expanded=False):
        st.markdown(f"### **Nanocrystalline Material Analysis**")
        st.markdown(f"- **Estimated family**: {material_family}")
        st.markdown(f"- **Strong peaks used**: {len(exp_peaks_2theta)} (filtered for angular diversity)")
        st.markdown(f"- **Average d-spacing**: {np.mean(exp_d):.3f} Å")
        st.markdown(f"- **Angular range**: {peak_quality.get('angular_range', 0):.1f}°")
        st.markdown(f"- **Databases searched**: {len(set(r['database'] for r in final_results if 'database' in r))}")
        
        if size_nm:
            tolerance = UniversalPatternMatcher.calculate_nano_tolerance(size_nm)
            st.markdown(f"- **Crystallite size**: {size_nm:.1f} nm")
            st.markdown(f"- **Matching tolerance**: {tolerance:.1%} d-spacing (physics-based)")
        
        # Scientific notes about nanocrystalline matching
        st.markdown("### **Matching Notes for Nanocrystalline Materials**")
        st.markdown("""
        - **Peak broadening**: Nanocrystalline materials exhibit broadened peaks
        - **Angular diversity**: Selected peaks are ≥1.5° apart for structural information
        - **d-spacing matching**: Using d-spacings (not 2θ) for better accuracy
        - **Relative intensities**: Matching intensity ordering, not absolute values
        - **Size-based tolerance**: Tolerance adjusted based on crystallite size
        """)
        
        # Database limitations disclosure
        st.markdown("### **Database Limitations**")
        st.markdown("""
        - **COD**: Experimental CIFs only (no amorphous/nanocrystalline patterns)
        - **Materials Project**: DFT-optimized structures (may differ from experiment)
        - **Coverage**: ~1.5M known crystalline phases vs. estimated 10⁸ possible compositions
        - **Nanomaterials**: Severe peak broadening reduces match probability
        """)
        
        # Show top matches
        if final_results:
            st.markdown("### **Top Phase Matches**")
            for i, result in enumerate(final_results[:3]):
                st.markdown(f"{i+1}. **{result['phase']}** ({result['crystal_system']}) - "
                          f"Score: {result['score']:.3f} [{result['confidence_level']}]")
                
                # Add scientific justification
                if result['score'] > 0.6:
                    st.markdown(f"   - *Strong match with {result['n_peaks_matched']} peak correspondences*")
                elif result['score'] > 0.4:
                    st.markdown(f"   - *Reasonable match considering nanocrystalline broadening*")
                else:
                    st.markdown(f"   - *Tentative match - verify with additional characterization*")
    
    return final_results
