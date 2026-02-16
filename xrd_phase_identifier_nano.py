"""
UNIVERSAL XRD PHASE IDENTIFICATION FOR NANOMATERIALS
========================================================================
Scientific phase identification for nanocrystalline materials with proper
peak filtering and d-spacing matching.
Now supports automatic peak-based search when elements are not provided,
multiple databases (COD, AMCSD, Materials Project), parallel simulation,
and a built‑in fallback database.
========================================================================
"""

import numpy as np
import requests
import time
from typing import List, Dict, Tuple, Optional, Set
import streamlit as st
from dataclasses import dataclass
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import concurrent.futures
from functools import lru_cache
import hashlib
import json
import os

# Attempt to import pymatgen – if not installed, show a helpful message
try:
    from pymatgen.io.cif import CifParser
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    PMG_AVAILABLE = True
except ImportError:
    PMG_AVAILABLE = False
    st.error("pymatgen is required but not installed. Please run: pip install pymatgen")

# ------------------------------------------------------------
# UNIVERSAL NANOMATERIAL PARAMETERS
# ------------------------------------------------------------
@dataclass
class UniversalNanoParams:
    """Universal parameters for all nanomaterial types"""
    
    SIZE_TOLERANCE_MAP = {
        'ultra_nano': (0.1, 20),   # < 5 nm: 0.1-20° tolerance
        'nano': (0.08, 15),        # 5-20 nm: 0.08-15° tolerance  
        'submicron': (0.05, 10),   # 20-100 nm: 0.05-10° tolerance
        'micron': (0.03, 5),       # > 100 nm: 0.03-5° tolerance
    }
    
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
    
    DATABASE_PRIORITY = {
        'metal_nanoparticles': ['COD', 'AMCSD', 'MaterialsProject'],
        'metal_oxides': ['COD', 'AMCSD', 'MaterialsProject'],
        'perovskites': ['MaterialsProject', 'COD', 'AMCSD'],
        '2d_materials': ['COD', 'AMCSD', 'MaterialsProject'],
    }

# ------------------------------------------------------------
# UNIVERSAL PEAK ANALYSIS
# ------------------------------------------------------------
class UniversalPeakAnalyzer:
    """Universal peak analysis for nanocrystalline materials"""
    
    @staticmethod
    def detect_peaks_universal(two_theta: np.ndarray, intensity: np.ndarray, 
                               min_snr: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Scientific peak detection for nanomaterials with proper filtering"""
        if len(intensity) > 50:
            noise_level = np.std(intensity[:50])
        else:
            noise_level = np.std(intensity)
        
        baseline = np.percentile(intensity, 10)
        threshold = baseline + min_snr * noise_level
        
        peaks_idx, properties = find_peaks(
            intensity, 
            height=threshold,
            prominence=noise_level * 2,
            width=(2, None),
            distance=5,
            wlen=len(intensity)//5
        )
        return two_theta[peaks_idx], intensity[peaks_idx]
    
    @staticmethod
    def estimate_fwhm(peaks_2theta: np.ndarray, two_theta: np.ndarray, intensity: np.ndarray) -> float:
        """Estimate average full width at half maximum of peaks (rough approximation)"""
        if len(peaks_2theta) == 0:
            return 1.0
        widths = []
        for p in peaks_2theta:
            idx = np.argmin(np.abs(two_theta - p))
            left = max(0, idx - 10)
            right = min(len(two_theta), idx + 10)
            half_max = intensity[idx] / 2
            # find left crossing
            l_cross = idx
            for i in range(idx, left, -1):
                if intensity[i] <= half_max:
                    l_cross = i
                    break
            # right crossing
            r_cross = idx
            for i in range(idx, right):
                if intensity[i] <= half_max:
                    r_cross = i
                    break
            if r_cross > l_cross:
                widths.append(two_theta[r_cross] - two_theta[l_cross])
        return np.mean(widths) if widths else 1.0
    
    @staticmethod
    def filter_peaks_for_nanomaterials(peaks_2theta: np.ndarray, 
                                      peaks_intensity: np.ndarray,
                                      wavelength: float,
                                      max_peaks: int = 10,
                                      avg_fwhm: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Filter to strongest peaks with angular diversity, using FWHM if available."""
        if len(peaks_2theta) == 0:
            return peaks_2theta, peaks_intensity
        
        order = np.argsort(peaks_intensity)[::-1]
        peaks_2theta_sorted = peaks_2theta[order]
        peaks_intensity_sorted = peaks_intensity[order]

        min_sep = 1.2
        if avg_fwhm is not None:
            min_sep = max(1.2, avg_fwhm * 2.0)
        
        selected_2theta = []
        selected_intensity = []
        
        for i in range(len(peaks_2theta_sorted)):
            current_2theta = peaks_2theta_sorted[i]
            current_intensity = peaks_intensity_sorted[i]
            
            accept = True
            for j, sel in enumerate(selected_2theta):
                if abs(current_2theta - sel) < min_sep:
                    if current_intensity > selected_intensity[j]:
                        selected_2theta[j] = current_2theta
                        selected_intensity[j] = current_intensity
                    accept = False
                    break
            
            if accept:
                selected_2theta.append(current_2theta)
                selected_intensity.append(current_intensity)
            
            if len(selected_2theta) >= max_peaks:
                break
        
        if len(selected_2theta) < max_peaks:
            for i in range(len(peaks_2theta_sorted)):
                current_2theta = peaks_2theta_sorted[i]
                if any(abs(current_2theta - s) < 1e-3 for s in selected_2theta):
                    continue
                selected_2theta.append(current_2theta)
                selected_intensity.append(peaks_intensity_sorted[i])
                if len(selected_2theta) >= max_peaks:
                    break
        
        return np.array(selected_2theta), np.array(selected_intensity)
    
    @staticmethod
    def estimate_material_family(elements: List[str], peak_positions: List[float]) -> str:
        """Intelligently estimate material family from elements and peaks"""
        if not elements:
            return 'unknown'
        
        elements_set = set(elements)
        common_metals = {'Au', 'Ag', 'Cu', 'Pt', 'Pd', 'Ni', 'Fe', 'Co'}
        if elements_set & common_metals:
            return 'metal_nanoparticles'
        
        if 'O' in elements_set:
            transition_metals = {'Ti', 'Zn', 'Fe', 'Cu', 'Ni', 'Co', 'Mn', 'Cr', 'V'}
            rare_earths = {'Ce', 'La', 'Nd', 'Pr', 'Sm', 'Eu', 'Gd'}
            if (elements_set & transition_metals) or (elements_set & rare_earths):
                return 'metal_oxides'
        
        chalcogens = {'S', 'Se', 'Te'}
        if elements_set & chalcogens:
            return 'metal_chalcogenides'
        
        if len(elements) >= 3 and 'O' in elements:
            return 'perovskites'
        
        if 'C' in elements_set and len(elements) <= 2:
            return 'carbon_allotropes'
        
        return 'unknown'

# ------------------------------------------------------------
# MULTI‑DATABASE SEARCHER (COD, AMCSD, Materials Project)
# ------------------------------------------------------------
class UniversalDatabaseSearcher:
    """Searches multiple open crystallographic databases."""
    
    def __init__(self, mp_api_key: Optional[str] = None):
        # Try to read key from environment if not provided
        if mp_api_key is None:
            mp_api_key = os.environ.get("MP_API_KEY", "")
        self.mp_api_key = mp_api_key
        self.session = requests.Session()
        self.query_cache = {}
    
    # ------------------------------------------------------------------
    # COD search (elements)
    # ------------------------------------------------------------------
    @lru_cache(maxsize=100)
    def _search_cod_elements(self, elements: Tuple[str], max_results: int = 30) -> List[Dict]:
        """Search COD by elements."""
        try:
            params = {
                "format": "json",
                "el": ",".join(elements),
                "maxresults": max_results
            }
            if "O" in elements:
                params["formula"] = "*O*"
            
            resp = self.session.get(
                "https://www.crystallography.net/cod/result",
                params=params,
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
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
                                'confidence': 0.8
                            })
                return structures
        except Exception as e:
            st.warning(f"COD element search failed: {str(e)[:100]}")
        return []
    
    # ------------------------------------------------------------------
    # COD search by d‑spacings (robust method: query each peak with range)
    # ------------------------------------------------------------------
    @lru_cache(maxsize=50)
    def _search_cod_dspacings(self, dspacings: Tuple[float], tolerance: float = 0.05) -> List[Dict]:
        """
        Search COD using a list of d-spacings (Å). For each d, we query with a ±tolerance range
        and collect unique COD IDs. This is much more reliable than a comma‑separated list.
        """
        all_structures = []
        seen_ids = set()
        
        for d in dspacings[:5]:  # use top 5 peaks
            lower = d * (1 - tolerance)
            upper = d * (1 + tolerance)
            params = {
                "format": "json",
                "dspacing": f"{lower:.3f}",
                "dspacing2": f"{upper:.3f}",
                "maxresults": 20
            }
            try:
                resp = self.session.get(
                    "https://www.crystallography.net/cod/result",
                    params=params,
                    timeout=10
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        for entry in data:
                            codid = str(entry.get('codid', ''))
                            if codid and codid not in seen_ids:
                                seen_ids.add(codid)
                                all_structures.append({
                                    'database': 'COD',
                                    'id': codid,
                                    'formula': entry.get('formula', ''),
                                    'space_group': entry.get('sg', ''),
                                    'cif_url': f"https://www.crystallography.net/cod/{codid}.cif",
                                    'confidence': 0.7
                                })
            except Exception:
                continue
        
        return all_structures[:30]
    
    # ------------------------------------------------------------------
    # AMCSD search (by elements)
    # ------------------------------------------------------------------
    @lru_cache(maxsize=50)
    def _search_amcsd(self, elements: Tuple[str], max_results: int = 20) -> List[Dict]:
        """
        Search American Mineralogist Crystal Structure Database.
        AMCSD provides a simple text‑based search; we parse the results.
        """
        try:
            # AMCSD search URL (GET with formula)
            formula = "".join(elements)
            url = f"http://rruff.geo.arizona.edu/AMS/result.php?formula={formula}"
            resp = self.session.get(url, timeout=15)
            if resp.status_code != 200:
                return []
            
            # Very crude HTML parsing – look for links to CIF files
            import re
            cif_links = re.findall(r'href="([^"]+\.cif)"', resp.text)
            structures = []
            for link in cif_links[:max_results]:
                full_url = link if link.startswith("http") else f"http://rruff.geo.arizona.edu/AMS/{link}"
                # Try to extract formula from the link text or fallback
                structures.append({
                    'database': 'AMCSD',
                    'id': link.split('/')[-1].replace('.cif', ''),
                    'formula': formula,
                    'space_group': 'Unknown',
                    'cif_url': full_url,
                    'confidence': 0.7
                })
            return structures
        except Exception as e:
            st.warning(f"AMCSD search failed: {str(e)[:100]}")
        return []
    
    # ------------------------------------------------------------------
    # Materials Project search (requires API key)
    # ------------------------------------------------------------------
    @lru_cache(maxsize=50)
    def _search_materials_project(self, elements: Tuple[str], max_results: int = 20) -> List[Dict]:
        if not self.mp_api_key:
            return []
        try:
            headers = {"X-API-KEY": self.mp_api_key}
            # Use MP's new API (v2023)
            elements_str = ",".join(elements)
            url = f"https://api.materialsproject.org/materials/core/?elements={elements_str}&_per_page={max_results}"
            resp = self.session.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                structures = []
                for doc in data.get("data", [])[:max_results]:
                    structures.append({
                        'database': 'MaterialsProject',
                        'id': doc.get("material_id", ""),
                        'formula': doc.get("formula_pretty", ""),
                        'space_group': doc.get("symmetry", {}).get("symbol", ""),
                        'cif_url': f"https://next-gen.materialsproject.org/materials/{doc.get('material_id')}/cif",
                        'confidence': 0.75
                    })
                return structures
        except Exception as e:
            st.warning(f"Materials Project search failed: {str(e)[:100]}")
        return []
    
    # ------------------------------------------------------------------
    # Main search dispatcher
    # ------------------------------------------------------------------
    def search_all_databases(self, 
                            elements: Optional[List[str]] = None, 
                            dspacings: Optional[np.ndarray] = None,
                            material_family: str = 'unknown',
                            progress_callback=None) -> List[Dict]:
        """
        Search all configured databases.
        If elements are provided, use element‑based searches.
        Otherwise, use d‑spacing search (COD only, but multiple peaks).
        """
        all_structures = []
        
        if elements:
            elements_tuple = tuple(sorted(elements))
            st.info(f"🔍 Searching databases for elements: {', '.join(elements)}")
            
            # Determine which databases to query
            if material_family in UniversalNanoParams.DATABASE_PRIORITY:
                db_priority = UniversalNanoParams.DATABASE_PRIORITY[material_family]
            else:
                db_priority = ['COD', 'AMCSD', 'MaterialsProject']
            
            # Query in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_to_db = {}
                if 'COD' in db_priority:
                    future_to_db[executor.submit(self._search_cod_elements, elements_tuple, 20)] = 'COD'
                if 'AMCSD' in db_priority:
                    future_to_db[executor.submit(self._search_amcsd, elements_tuple, 15)] = 'AMCSD'
                if 'MaterialsProject' in db_priority and self.mp_api_key:
                    future_to_db[executor.submit(self._search_materials_project, elements_tuple, 15)] = 'MP'
                
                for future in concurrent.futures.as_completed(future_to_db):
                    db_name = future_to_db[future]
                    try:
                        results = future.result(timeout=15)
                        if results:
                            if progress_callback:
                                progress_callback(f"Found {len(results)} structures from {db_name}")
                            all_structures.extend(results)
                    except Exception as e:
                        if progress_callback:
                            progress_callback(f"Error from {db_name}: {str(e)[:50]}")
        
        elif dspacings is not None and len(dspacings) > 0:
            # D‑spacing search – use COD with range queries
            st.info("🔍 No elements provided. Searching COD by d‑spacings (peak matching)...")
            dspacings_tuple = tuple(sorted(dspacings[:5]))
            all_structures = self._search_cod_dspacings(dspacings_tuple, tolerance=0.05)
            if progress_callback:
                progress_callback(f"COD d‑spacing search returned {len(all_structures)} candidates")
        
        # Remove duplicates (same formula + space group)
        unique = {}
        for s in all_structures:
            key = (s.get('formula', ''), s.get('space_group', ''))
            if key not in unique or s['database'] == 'COD':  # prefer COD if duplicate
                unique[key] = s
        return list(unique.values())

# ------------------------------------------------------------
# SCIENTIFIC MATCHING ALGORITHM
# ------------------------------------------------------------
class UniversalPatternMatcher:
    """Scientific pattern matching for nanocrystalline materials"""
    
    @staticmethod
    def calculate_nano_tolerance(size_nm: Optional[float] = None) -> float:
        if size_nm is None:
            return 0.05
        elif size_nm < 5:
            return 0.10
        elif size_nm < 10:
            return 0.06
        else:
            return 0.03
    
    @staticmethod
    def match_pattern_universal(exp_d: np.ndarray, exp_intensity: np.ndarray,
                               sim_d: np.ndarray, sim_intensity: np.ndarray,
                               material_family: str = 'unknown',
                               size_nm: Optional[float] = None) -> float:
        if len(exp_d) == 0 or len(sim_d) == 0:
            return 0.0
        
        params = UniversalPatternMatcher._get_family_params(material_family)
        base_tolerance = UniversalPatternMatcher.calculate_nano_tolerance(size_nm)
        
        n_exp_peaks = min(len(exp_d), 8)
        match_scores = []
        intensity_weights = []
        
        exp_sorted_idx = np.argsort(exp_intensity)[::-1][:n_exp_peaks]
        
        for i in exp_sorted_idx:
            d_exp = exp_d[i]
            intensity_exp = exp_intensity[i]
            
            intensity_factor = intensity_exp / np.max(exp_intensity)
            peak_tolerance = base_tolerance * (1.5 - 0.3 * intensity_factor)
            
            if material_family in ['metal_nanoparticles', 'carbon_allotropes']:
                peak_tolerance *= 1.5
            
            d_errors = np.abs(sim_d - d_exp) / d_exp
            min_error_idx = np.argmin(d_errors)
            min_error = d_errors[min_error_idx]
            
            if min_error < peak_tolerance:
                match_quality = 1.0 - (min_error / peak_tolerance)
                
                if len(sim_intensity) > min_error_idx:
                    exp_rank = np.sum(exp_intensity > intensity_exp) / len(exp_intensity)
                    sim_rank = np.sum(sim_intensity > sim_intensity[min_error_idx]) / len(sim_intensity)
                    rank_match = 1.0 - abs(exp_rank - sim_rank)
                    match_quality *= (0.6 + 0.4 * rank_match)
                
                match_scores.append(match_quality)
                weight = intensity_exp / np.sum(exp_intensity[exp_sorted_idx])
                intensity_weights.append(weight)
        
        if not match_scores:
            return 0.0
        
        weighted_score = np.average(match_scores, weights=intensity_weights)
        coverage = len(match_scores) / n_exp_peaks
        
        if coverage < 0.5:
            return 0.0
        
        coverage_factor = 0.4 + 0.6 * coverage
        final_score = weighted_score * coverage_factor
        
        if material_family in ['metal_nanoparticles', 'carbon_allotropes'] and n_exp_peaks <= 4:
            final_score *= 1.2
        
        return min(final_score, 1.0)
    
    @staticmethod
    def _get_family_params(family: str) -> Dict:
        params = {'base_tolerance': 0.05, 'intensity_weight': 0.3, 'coverage_weight': 0.7}
        family_adjustments = {
            'metal_nanoparticles': {'base_tolerance': 0.06, 'intensity_weight': 0.2},
            'metal_oxides': {'base_tolerance': 0.04, 'intensity_weight': 0.35},
            'metal_chalcogenides': {'base_tolerance': 0.05, 'intensity_weight': 0.3},
            'perovskites': {'base_tolerance': 0.04, 'intensity_weight': 0.4},
            'spinels': {'base_tolerance': 0.04, 'intensity_weight': 0.35},
            'carbon_allotropes': {'base_tolerance': 0.07, 'intensity_weight': 0.1},
        }
        if family in family_adjustments:
            params.update(family_adjustments[family])
        return params

# ------------------------------------------------------------
# CACHED PATTERN SIMULATION (using pymatgen)
# ------------------------------------------------------------
@lru_cache(maxsize=200)
def _simulate_pattern_cached(cif_url: str, wavelength: float) -> Tuple[np.ndarray, np.ndarray]:
    """Download CIF and simulate XRD pattern. Returns (two_theta, intensity)."""
    if not PMG_AVAILABLE:
        return np.array([]), np.array([])
    try:
        resp = requests.get(cif_url, timeout=15)
        if resp.status_code != 200:
            return np.array([]), np.array([])
        cif_text = resp.text
        
        parser = CifParser.from_string(cif_text)
        structure = parser.get_structures()[0]
        calc = XRDCalculator(wavelength=wavelength)
        pattern = calc.get_pattern(structure, two_theta_range=(5, 80))
        return np.array(pattern.x), np.array(pattern.y)
    except Exception:
        return np.array([]), np.array([])

# ------------------------------------------------------------
# BUILT‑IN FALLBACK DATABASE (common nanomaterials)
# ------------------------------------------------------------
FALLBACK_STRUCTURES = [
    {
        "name": "Gold (Au)",
        "formula": "Au",
        "space_group": "Fm-3m",
        "cif_url": "https://www.crystallography.net/cod/9008463.cif",
        "database": "Fallback"
    },
    {
        "name": "Silver (Ag)",
        "formula": "Ag",
        "space_group": "Fm-3m",
        "cif_url": "https://www.crystallography.net/cod/9008459.cif",
        "database": "Fallback"
    },
    {
        "name": "Titanium dioxide (Anatase)",
        "formula": "TiO2",
        "space_group": "I41/amd",
        "cif_url": "https://www.crystallography.net/cod/9008213.cif",
        "database": "Fallback"
    },
    {
        "name": "Zinc oxide (Zincite)",
        "formula": "ZnO",
        "space_group": "P63mc",
        "cif_url": "https://www.crystallography.net/cod/9008878.cif",
        "database": "Fallback"
    },
    {
        "name": "Silicon (Si)",
        "formula": "Si",
        "space_group": "Fd-3m",
        "cif_url": "https://www.crystallography.net/cod/9011380.cif",
        "database": "Fallback"
    }
]

# ------------------------------------------------------------
# MAIN IDENTIFICATION ENGINE (NOW WITH PRECOMPUTED PEAKS SUPPORT)
# ------------------------------------------------------------
def identify_phases_universal(two_theta: np.ndarray = None, intensity: np.ndarray = None,
                            wavelength: float = 1.5406, elements: Optional[List[str]] = None,
                            size_nm: Optional[float] = None, mp_api_key: Optional[str] = None,
                            # NEW OPTIONAL PARAMETERS:
                            precomputed_peaks_2theta: Optional[np.ndarray] = None,
                            precomputed_peaks_intensity: Optional[np.ndarray] = None) -> List[Dict]:
    """
    SCIENTIFIC phase identification for nanocrystalline materials.
    
    If precomputed_peaks_2theta and precomputed_peaks_intensity are provided,
    they are used directly (skipping internal peak detection). Otherwise,
    peak detection is performed on the raw (two_theta, intensity) data.
    
    This ensures perfect consistency with any prior peak analysis.
    """
    if not PMG_AVAILABLE:
        st.error("pymatgen is required. Please install it and restart.")
        return []
    
    st.info("🔬 Running scientific nanomaterial phase identification...")
    
    # Create a status container for live updates
    status = st.status("Initializing...", expanded=True)
    
    # --------------------------------------------------------
    # STEP 1: PEAK DETECTION (OR USE PRECOMPUTED PEAKS)
    # --------------------------------------------------------
    peak_analyzer = UniversalPeakAnalyzer()
    
    if precomputed_peaks_2theta is not None and precomputed_peaks_intensity is not None:
        # Use the provided structural peaks
        status.update(label="Using pre‑computed structural peaks...", state="running")
        exp_peaks_2theta = np.array(precomputed_peaks_2theta)
        exp_intensities = np.array(precomputed_peaks_intensity)
        status.write(f"✅ Using {len(exp_peaks_2theta)} pre‑computed peaks")
    else:
        # Fall back to detecting peaks from raw data
        status.update(label="Detecting peaks from raw data...", state="running")
        exp_peaks_2theta, exp_intensities = peak_analyzer.detect_peaks_universal(two_theta, intensity)
        
        # Local apex refinement (only if using raw data)
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
        
        # Estimate FWHM for better filtering (if raw data available)
        avg_fwhm = peak_analyzer.estimate_fwhm(exp_peaks_2theta, two_theta, intensity)
        
        exp_peaks_2theta, exp_intensities = peak_analyzer.filter_peaks_for_nanomaterials(
            exp_peaks_2theta, exp_intensities, wavelength, max_peaks=10, avg_fwhm=avg_fwhm
        )
        status.write(f"✅ Detected and filtered {len(exp_peaks_2theta)} peaks")
    
    if len(exp_peaks_2theta) < 2:
        status.update(label="❌ Insufficient peaks found", state="error")
        st.warning("Insufficient strong peaks for reliable phase identification")
        return []
    
    status.write(f"🧪 2θ peaks: {np.round(exp_peaks_2theta, 3).tolist()}")
    
    # Calculate d-spacings (always needed)
    exp_d = wavelength / (2 * np.sin(np.radians(exp_peaks_2theta / 2)))
    status.write(f"🧪 d-spacings (Å): {np.round(exp_d, 3).tolist()}")
    
    exp_intensities_norm = exp_intensities / np.max(exp_intensities)
    
    # --------------------------------------------------------
    # STEP 2: ESTIMATE MATERIAL FAMILY
    # --------------------------------------------------------
    if elements:
        material_family = peak_analyzer.estimate_material_family(elements, exp_peaks_2theta)
        status.write(f"📊 Material family estimated: {material_family}")
    else:
        material_family = 'unknown'
    
    if size_nm:
        status.write(f"📊 Crystallite size: {size_nm:.1f} nm → tolerance: {UniversalPatternMatcher.calculate_nano_tolerance(size_nm):.1%}")
    
    # --------------------------------------------------------
    # STEP 3: DATABASE SEARCH
    # --------------------------------------------------------
    status.update(label="Searching crystallographic databases...", state="running")
    db_searcher = UniversalDatabaseSearcher(mp_api_key=mp_api_key)
    
    def db_progress(msg):
        status.write(f"🔍 {msg}")
    
    if elements:
        database_structures = db_searcher.search_all_databases(
            elements=elements, 
            material_family=material_family,
            progress_callback=db_progress
        )
    else:
        # Use top 5 d‑spacings for search
        dspacings_for_search = exp_d[:5]
        database_structures = db_searcher.search_all_databases(
            dspacings=dspacings_for_search,
            material_family=material_family,
            progress_callback=db_progress
        )
    st.write(f"📊 Retrieved {len(database_structures)} candidates from databases")
    
    # Limit to top 20 candidates for speed
    database_structures = database_structures[:20]
    status.write(f"📚 Retrieved {len(database_structures)} candidate structures")
    
    # If no structures from databases, use fallback
    if not database_structures:
        status.write("⚠️ No structures found in online databases. Using built‑in fallback list.")
        database_structures = FALLBACK_STRUCTURES
    
    # --------------------------------------------------------
    # STEP 4: PARALLEL PATTERN SIMULATION AND MATCHING
    # --------------------------------------------------------
    status.update(label=f"Simulating and matching {len(database_structures)} structures...", state="running")
    
    matcher = UniversalPatternMatcher()
    results = []
    
    # Use ThreadPoolExecutor to simulate patterns in parallel (max 4 workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_struct = {}
        for struct in database_structures:
            cif_url = struct.get('cif_url')
            if cif_url:
                future = executor.submit(_simulate_pattern_cached, cif_url, wavelength)
                future_to_struct[future] = struct
        
        total = len(future_to_struct)
        completed = 0
        for future in concurrent.futures.as_completed(future_to_struct):
            struct = future_to_struct[future]
            completed += 1
            status.write(f"   [{completed}/{total}] Simulating {struct.get('formula', 'unknown')}...")
            try:
                sim_x, sim_y = future.result(timeout=20)
                if len(sim_x) == 0:
                    continue
                
                sim_d = wavelength / (2 * np.sin(np.radians(sim_x / 2)))
                sim_intensity = sim_y / np.max(sim_y) if len(sim_y) > 0 else np.zeros_like(sim_x)
                
                match_score = matcher.match_pattern_universal(
                    exp_d, exp_intensities_norm,
                    sim_d, sim_intensity,
                    material_family, size_nm
                )
                
                # Determine threshold based on input availability
                if not elements:
                    threshold = 0.10
                    if size_nm and size_nm < 5:
                        threshold = 0.08
                else:
                    threshold = 0.15 if size_nm and size_nm < 10 else 0.20
                
                if match_score < threshold:
                    continue
                
                # Determine confidence level
                if not elements:
                    confidence = "probable" if match_score >= 0.30 else "possible"
                else:
                    if size_nm and size_nm < 10:
                        if match_score >= 0.55:
                            confidence = "confirmed"
                        elif match_score >= 0.35:
                            confidence = "probable"
                        else:
                            confidence = "possible"
                    else:
                        if match_score >= 0.60:
                            confidence = "confirmed"
                        elif match_score >= 0.40:
                            confidence = "probable"
                        else:
                            confidence = "possible"
                
                # Extract additional info from CIF if possible (fallback to struct dict)
                try:
                    parser = CifParser.from_string(requests.get(cif_url, timeout=8).text)
                    structure = parser.get_structures()[0]
                    crystal_system = structure.get_crystal_system()
                    space_group = structure.get_space_group_info()[0]
                    lattice = structure.lattice.as_dict()
                    formula = structure.composition.reduced_formula
                    full_formula = str(structure.composition)
                except:
                    crystal_system = struct.get('space_group', 'Unknown')
                    space_group = struct.get('space_group', 'Unknown')
                    lattice = {}
                    formula = struct.get('formula', 'Unknown')
                    full_formula = struct.get('formula', 'Unknown')
                
                results.append({
                    "phase": formula,
                    "full_formula": full_formula,
                    "crystal_system": crystal_system,
                    "space_group": space_group,
                    "lattice": lattice,
                    "hkls": [],
                    "score": round(match_score, 3),
                    "confidence_level": confidence,
                    "database": struct.get('database', 'Unknown'),
                    "material_family": material_family,
                    "n_peaks_matched": len(exp_d),
                    "match_details": {
                        "n_exp_peaks": len(exp_d),
                        "avg_d_spacing": float(np.mean(exp_d)),
                        "size_nm": size_nm,
                        "tolerance_used": UniversalPatternMatcher.calculate_nano_tolerance(size_nm)
                    }
                })
                
                # Early feedback for first good match
                if len(results) <= 3:
                    status.write(f"   → {formula}: score {match_score:.3f} → {confidence}")
                    
            except Exception as e:
                status.write(f"   ⚠️ Error simulating {struct.get('formula', 'unknown')}: {str(e)[:50]}")
    
    # --------------------------------------------------------
    # STEP 5: PROCESS RESULTS
    # --------------------------------------------------------
    if not results:
        status.update(label="❌ No phases matched", state="error")
        with st.expander("📊 Diagnostic Report", expanded=True):
            st.markdown("### Why were no crystalline phases identified?")
            st.markdown("- Peak broadening may exceed database tolerance")
            st.markdown("- Material may be amorphous or poorly crystalline")
            st.markdown("- Try providing element information for better filtering")
            st.markdown("- Consider using raw (unsmoothed) data")
        return []
    
    # Remove duplicates
    unique_results = []
    seen = set()
    for r in results:
        key = (r["phase"], r.get("space_group", ""))
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
    
    final_results = sorted(unique_results, key=lambda x: x["score"], reverse=True)
    
    status.update(label=f"✅ Identified {len(final_results)} potential phases", state="complete")
    
    # Show top matches in expander
    with st.expander("📊 Scientific Analysis Report", expanded=False):
        st.markdown(f"### **Nanocrystalline Material Analysis**")
        st.markdown(f"- **Estimated family**: {material_family}")
        st.markdown(f"- **Strong peaks used**: {len(exp_peaks_2theta)}")
        st.markdown(f"- **Average d-spacing**: {np.mean(exp_d):.3f} Å")
        if size_nm:
            st.markdown(f"- **Crystallite size**: {size_nm:.1f} nm")
        st.markdown("### **Top Phase Matches**")
        for i, res in enumerate(final_results[:5]):
            st.markdown(f"{i+1}. **{res['phase']}** ({res['crystal_system']}) – "
                       f"Score: {res['score']:.3f} [{res['confidence_level']}]")
    
    return final_results

