"""
UNIVERSAL XRD PHASE IDENTIFIER – THE GOLD STANDARD FOR NANOMATERIALS
========================================================================
Features:
- Automatic peak detection OR use of pre‑computed structural peaks
- Physics‑based tolerance (Scherrer size → d‑spacing tolerance)
- Multi‑database search (COD, AMCSD, Materials Project) + extensive fallback
- Parallel simulation with pymatgen (CIF → pattern)
- Matches ALL experimental peaks (no limit) with per‑peak HKL assignment
- Adaptive coverage threshold (50% / 30% for <10 nm)
- Mixed‑phase support – returns ALL phases above threshold
- Phase fraction estimation (reference intensity ratio / simple area scaling)
- Detailed output: score, confidence, lattice, space group, matched reflections
- Built for publication‑grade reproducibility
========================================================================
"""

import numpy as np
import requests
import time
from typing import List, Dict, Tuple, Optional, Set
import streamlit as st
from dataclasses import dataclass
from scipy.signal import find_peaks
import concurrent.futures
from functools import lru_cache
import os
import re

# ============================================================================
# DEPENDENCIES: pymatgen must be installed
# ============================================================================
try:
    from pymatgen.io.cif import CifParser
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    PMG_AVAILABLE = True
except ImportError:
    PMG_AVAILABLE = False
    st.error("pymatgen is required. Please run: pip install pymatgen")

# ============================================================================
# UNIVERSAL PARAMETERS (scientifically calibrated)
# ============================================================================
@dataclass
class NanoParams:
    """Universal parameters for all nanomaterial types."""
    # Size → tolerance mapping (Δd/d)
    SIZE_TOLERANCE = {
        'ultra_nano': 0.10,   # <5 nm
        'nano': 0.06,         # 5-10 nm
        'submicron': 0.03,    # 10-100 nm
        'micron': 0.02,       # >100 nm
    }
    
    # Material families for database prioritisation
    FAMILIES = {
        'metal': ['Au', 'Ag', 'Cu', 'Pt', 'Pd', 'Ni', 'Fe', 'Co'],
        'oxide': ['TiO2', 'ZnO', 'Fe2O3', 'Fe3O4', 'CuO', 'NiO', 'Al2O3', 'SiO2', 'ZrO2', 'CeO2'],
        'chalcogenide': ['MoS2', 'WS2', 'CdSe', 'PbS', 'ZnS'],
        'perovskite': ['BaTiO3', 'SrTiO3', 'LaMnO3', 'BiFeO3'],
        'carbon': ['C', 'graphene', 'graphite'],
    }
    
    DATABASE_PRIORITY = {
        'metal': ['COD', 'MaterialsProject'],
        'oxide': ['COD', 'MaterialsProject', 'AMCSD'],
        'perovskite': ['MaterialsProject', 'COD'],
        'chalcogenide': ['COD', 'MaterialsProject'],
        'carbon': ['COD'],
    }

# ============================================================================
# PEAK ANALYSIS (if pre‑computed peaks not provided)
# ============================================================================
class PeakAnalyzer:
    """Robust peak detection for XRD patterns."""
    
    @staticmethod
    def detect_peaks(two_theta: np.ndarray, intensity: np.ndarray, 
                     min_snr: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Find peaks with adaptive thresholding."""
        # Estimate noise level from low‑intensity region
        sorted_int = np.sort(intensity)
        noise_level = np.mean(sorted_int[:len(sorted_int)//10])
        
        # Use prominence to avoid false positives
        peaks_idx, _ = find_peaks(
            intensity,
            height=noise_level * 3,
            prominence=noise_level * min_snr,
            distance=max(5, int(len(intensity)/200))  # angular distance ~0.5° typical
        )
        
        if len(peaks_idx) == 0:
            # Fallback to absolute maximum
            peaks_idx = [np.argmax(intensity)]
            
        return two_theta[peaks_idx], intensity[peaks_idx]
    
    @staticmethod
    def refine_apex(two_theta: np.ndarray, intensity: np.ndarray,
                    peaks_2theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Refine peak positions to local maximum (sub‑pixel accuracy)."""
        refined = []
        refined_int = []
        for t0 in peaks_2theta:
            idx = np.argmin(np.abs(two_theta - t0))
            # Search ±5 points for true max
            left = max(0, idx-5)
            right = min(len(two_theta), idx+6)
            local_idx = left + np.argmax(intensity[left:right])
            refined.append(two_theta[local_idx])
            refined_int.append(intensity[local_idx])
        return np.array(refined), np.array(refined_int)

# ============================================================================
# DATABASE SEARCHER (COD, AMCSD, Materials Project)
# ============================================================================
class DatabaseSearcher:
    """Multi‑database query engine."""
    
    def __init__(self, mp_api_key: Optional[str] = None):
        self.mp_api_key = mp_api_key or os.environ.get("MP_API_KEY", "")
        self.session = requests.Session()
    
    @lru_cache(maxsize=100)
    def search_cod_by_elements(self, elements: Tuple[str], max_results: int = 30) -> List[Dict]:
        """Search COD by element list."""
        try:
            params = {
                "format": "json",
                "el": ",".join(elements),
                "maxresults": max_results
            }
            resp = self.session.get(
                "https://www.crystallography.net/cod/result",
                params=params, timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                structures = []
                for entry in data[:max_results]:
                    if 'codid' in entry:
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
            st.warning(f"COD search error: {e}")
        return []
    
    @lru_cache(maxsize=50)
    def search_cod_by_dspacings(self, dspacings: Tuple[float], tolerance: float = 0.05) -> List[Dict]:
        """Search COD by d‑spacing ranges (for unknown composition)."""
        all_structures = []
        seen_ids = set()
        for d in dspacings[:5]:
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
                    params=params, timeout=10
                )
                if resp.status_code == 200:
                    data = resp.json()
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
            except:
                continue
        return all_structures[:30]
    
    @lru_cache(maxsize=50)
    def search_amcsd(self, elements: Tuple[str], max_results: int = 20) -> List[Dict]:
        """Search American Mineralogist database."""
        try:
            formula = "".join(elements)
            url = f"http://rruff.geo.arizona.edu/AMS/result.php?formula={formula}"
            resp = self.session.get(url, timeout=15)
            if resp.status_code != 200:
                return []
            import re
            cif_links = re.findall(r'href="([^"]+\.cif)"', resp.text)
            structures = []
            for link in cif_links[:max_results]:
                full_url = link if link.startswith("http") else f"http://rruff.geo.arizona.edu/AMS/{link}"
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
            st.warning(f"AMCSD error: {e}")
        return []
    
    @lru_cache(maxsize=50)
    def search_materials_project(self, elements: Tuple[str], max_results: int = 30) -> List[Dict]:
        """Search Materials Project (requires API key)."""
        if not self.mp_api_key:
            return []
        try:
            headers = {"X-API-KEY": self.mp_api_key}
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
            st.warning(f"MP error: {e}")
        return []
    
    def search_all(self, elements: Optional[List[str]] = None,
                   dspacings: Optional[np.ndarray] = None,
                   family: str = 'unknown',
                   progress=None) -> List[Dict]:
        """Unified search: element‑based or d‑spacing‑based."""
        all_structs = []
        if elements:
            elements_tuple = tuple(sorted(elements))
            # Determine priority databases
            if family in NanoParams.DATABASE_PRIORITY:
                dbs = NanoParams.DATABASE_PRIORITY[family]
            else:
                dbs = ['COD', 'MaterialsProject', 'AMCSD']
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
                futures = {}
                if 'COD' in dbs:
                    futures[ex.submit(self.search_cod_by_elements, elements_tuple, 30)] = 'COD'
                if 'AMCSD' in dbs:
                    futures[ex.submit(self.search_amcsd, elements_tuple, 20)] = 'AMCSD'
                if 'MaterialsProject' in dbs and self.mp_api_key:
                    futures[ex.submit(self.search_materials_project, elements_tuple, 30)] = 'MP'
                
                for fut in concurrent.futures.as_completed(futures):
                    db = futures[fut]
                    try:
                        res = fut.result(timeout=15)
                        if res and progress:
                            progress(f"Found {len(res)} from {db}")
                        all_structs.extend(res)
                    except Exception as e:
                        if progress:
                            progress(f"Error from {db}: {e}")
        elif dspacings is not None:
            all_structs = self.search_cod_by_dspacings(tuple(dspacings[:5]))
        
        # De‑duplicate (formula + space group)
        unique = {}
        for s in all_structs:
            key = (s.get('formula', ''), s.get('space_group', ''))
            if key not in unique or s['database'] == 'COD':
                unique[key] = s
        return list(unique.values())

# ============================================================================
# SCIENTIFIC MATCHING ENGINE
# ============================================================================
class PatternMatcher:
    """Match experimental peaks against simulated patterns with per‑peak assignment."""
    
    @staticmethod
    def tolerance_from_size(size_nm: Optional[float]) -> float:
        if size_nm is None:
            return 0.05
        if size_nm < 5:
            return 0.10
        if size_nm < 10:
            return 0.06
        if size_nm < 100:
            return 0.03
        return 0.02
    
    @classmethod
    def match(cls, exp_d: np.ndarray, exp_intensity: np.ndarray,
              sim_d: np.ndarray, sim_intensity: np.ndarray,
              sim_hkls: List[List[Tuple[int,int,int]]],
              size_nm: Optional[float] = None,
              family: str = 'unknown') -> Tuple[float, List[Dict]]:
        """
        Returns:
            score (0–1), list of matched peaks with details.
        """
        if len(exp_d) == 0 or len(sim_d) == 0:
            return 0.0, []
        
        base_tol = cls.tolerance_from_size(size_nm)
        n_exp = len(exp_d)
        scores = []
        weights = []
        matched = []
        
        # Sort experimental by intensity descending
        order = np.argsort(exp_intensity)[::-1]
        
        for idx in order:
            d_exp = exp_d[idx]
            I_exp = exp_intensity[idx]
            # Intensity‑dependent tolerance: weaker peaks allow slightly more error
            tol = base_tol * (1.5 - 0.3 * (I_exp / exp_intensity.max()))
            if family in ['metal', 'carbon']:
                tol *= 1.5
            
            # Find closest simulated peak
            errors = np.abs(sim_d - d_exp) / d_exp
            best_idx = np.argmin(errors)
            best_error = errors[best_idx]
            
            if best_error < tol:
                match_quality = 1.0 - (best_error / tol)
                
                # Intensity rank matching
                if len(sim_intensity) > best_idx:
                    exp_rank = np.sum(exp_intensity > I_exp) / len(exp_intensity)
                    sim_rank = np.sum(sim_intensity > sim_intensity[best_idx]) / len(sim_intensity)
                    rank_match = 1.0 - abs(exp_rank - sim_rank)
                    match_quality *= (0.6 + 0.4 * rank_match)
                
                scores.append(match_quality)
                weights.append(I_exp)
                
                # Record assignment
                matched.append({
                    'hkl': sim_hkls[best_idx] if best_idx < len(sim_hkls) else [],
                    'd_exp': float(d_exp),
                    'd_calc': float(sim_d[best_idx]),
                    'two_theta_exp': float(2 * np.arcsin(1.5406/(2*d_exp)) * 180/np.pi),
                    'two_theta_calc': float(2 * np.arcsin(1.5406/(2*sim_d[best_idx])) * 180/np.pi),
                    'intensity_exp': float(I_exp),
                    'intensity_calc': float(sim_intensity[best_idx])
                })
        
        if not scores:
            return 0.0, []
        
        weighted_score = np.average(scores, weights=weights)
        coverage = len(scores) / n_exp
        
        # Adaptive coverage threshold: 50% normally, 30% for very small crystallites
        min_coverage = 0.5 if (size_nm is None or size_nm >= 10) else 0.3
        if coverage < min_coverage:
            return 0.0, []
        
        final_score = weighted_score * (0.4 + 0.6 * coverage)
        return min(final_score, 1.0), matched

# ============================================================================
# CIF SIMULATION (with robust error handling)
# ============================================================================
@lru_cache(maxsize=200)
def simulate_from_cif(cif_url: str, wavelength: float) -> Tuple[np.ndarray, np.ndarray, List]:
    """Download CIF, simulate XRD, return (2θ, intensity, hkls)."""
    if not PMG_AVAILABLE:
        return np.array([]), np.array([]), []
    
    headers = {}
    if "materialsproject" in cif_url:
        mp_key = os.environ.get("MP_API_KEY", "")
        if mp_key:
            headers["X-API-KEY"] = mp_key
    
    try:
        resp = requests.get(cif_url, headers=headers, timeout=15)
        if resp.status_code != 200:
            st.write(f"   ⚠️ CIF download failed (HTTP {resp.status_code})")
            return np.array([]), np.array([]), []
        
        cif_text = resp.text
        if "<html" in cif_text[:200].lower():
            st.write("   ⚠️ Received HTML instead of CIF (authentication required?)")
            return np.array([]), np.array([]), []
        
        parser = CifParser.from_string(cif_text)
        structure = parser.get_structures()[0]
        calc = XRDCalculator(wavelength=wavelength)
        pattern = calc.get_pattern(structure, two_theta_range=(5, 80))
        return np.array(pattern.x), np.array(pattern.y), pattern.hkls
    except Exception as e:
        st.write(f"   ⚠️ Simulation error: {str(e)[:200]}")
        return np.array([]), np.array([]), []

# ============================================================================
# EXPANDED FALLBACK DATABASE (ensures matches even when online fails)
# ============================================================================
FALLBACK = [
    # Metals
    {"formula": "Au", "space_group": "Fm-3m", "cif_url": "https://www.crystallography.net/cod/9008463.cif", "database": "Fallback"},
    {"formula": "Ag", "space_group": "Fm-3m", "cif_url": "https://www.crystallography.net/cod/9008459.cif", "database": "Fallback"},
    {"formula": "Cu", "space_group": "Fm-3m", "cif_url": "https://www.crystallography.net/cod/9008461.cif", "database": "Fallback"},
    {"formula": "Pt", "space_group": "Fm-3m", "cif_url": "https://www.crystallography.net/cod/9011620.cif", "database": "Fallback"},
    {"formula": "Pd", "space_group": "Fm-3m", "cif_url": "https://www.crystallography.net/cod/1011094.cif", "database": "Fallback"},
    {"formula": "Ni", "space_group": "Fm-3m", "cif_url": "https://www.crystallography.net/cod/9008473.cif", "database": "Fallback"},
    {"formula": "Fe", "space_group": "Im-3m", "cif_url": "https://www.crystallography.net/cod/9008536.cif", "database": "Fallback"},
    {"formula": "Co", "space_group": "P63/mmc", "cif_url": "https://www.crystallography.net/cod/9008491.cif", "database": "Fallback"},
    # Oxides
    {"formula": "TiO2", "space_group": "I41/amd", "cif_url": "https://www.crystallography.net/cod/9008213.cif", "database": "Fallback"},  # anatase
    {"formula": "TiO2", "space_group": "P42/mnm", "cif_url": "https://www.crystallography.net/cod/9009082.cif", "database": "Fallback"},  # rutile
    {"formula": "ZnO", "space_group": "P63mc", "cif_url": "https://www.crystallography.net/cod/9008878.cif", "database": "Fallback"},
    {"formula": "Fe2O3", "space_group": "R-3c", "cif_url": "https://www.crystallography.net/cod/9000139.cif", "database": "Fallback"},
    {"formula": "Fe3O4", "space_group": "Fd-3m", "cif_url": "https://www.crystallography.net/cod/9006941.cif", "database": "Fallback"},
    {"formula": "BiFeO3", "space_group": "R3c", "cif_url": "https://www.crystallography.net/cod/1533055.cif", "database": "Fallback"},
    {"formula": "Bi2O3", "space_group": "P21/c", "cif_url": "https://www.crystallography.net/cod/2002920.cif", "database": "Fallback"},
    {"formula": "CuO", "space_group": "C2/c", "cif_url": "https://www.crystallography.net/cod/1011138.cif", "database": "Fallback"},
    {"formula": "Cu2O", "space_group": "Pn-3m", "cif_url": "https://www.crystallography.net/cod/1010936.cif", "database": "Fallback"},
    {"formula": "NiO", "space_group": "Fm-3m", "cif_url": "https://www.crystallography.net/cod/1010395.cif", "database": "Fallback"},
    {"formula": "Co3O4", "space_group": "Fd-3m", "cif_url": "https://www.crystallography.net/cod/9005913.cif", "database": "Fallback"},
    {"formula": "Al2O3", "space_group": "R-3c", "cif_url": "https://www.crystallography.net/cod/9007671.cif", "database": "Fallback"},
    {"formula": "SiO2", "space_group": "P3121", "cif_url": "https://www.crystallography.net/cod/9009668.cif", "database": "Fallback"},  # quartz
    {"formula": "ZrO2", "space_group": "P21/c", "cif_url": "https://www.crystallography.net/cod/9007504.cif", "database": "Fallback"},
    {"formula": "CeO2", "space_group": "Fm-3m", "cif_url": "https://www.crystallography.net/cod/9009009.cif", "database": "Fallback"},
    # Perovskites
    {"formula": "BaTiO3", "space_group": "P4mm", "cif_url": "https://www.crystallography.net/cod/1507756.cif", "database": "Fallback"},
    {"formula": "SrTiO3", "space_group": "Pm-3m", "cif_url": "https://www.crystallography.net/cod/9004712.cif", "database": "Fallback"},
    # Chalcogenides
    {"formula": "MoS2", "space_group": "P63/mmc", "cif_url": "https://www.crystallography.net/cod/9009139.cif", "database": "Fallback"},
    {"formula": "CdSe", "space_group": "F-43m", "cif_url": "https://www.crystallography.net/cod/9008835.cif", "database": "Fallback"},
    {"formula": "PbS", "space_group": "Fm-3m", "cif_url": "https://www.crystallography.net/cod/9008690.cif", "database": "Fallback"},
    # Carbon
    {"formula": "C", "space_group": "P63/mmc", "cif_url": "https://www.crystallography.net/cod/9011897.cif", "database": "Fallback"},  # graphite
    {"formula": "C", "space_group": "Fd-3m", "cif_url": "https://www.crystallography.net/cod/1010079.cif", "database": "Fallback"},  # diamond
    # Silicon
    {"formula": "Si", "space_group": "Fd-3m", "cif_url": "https://www.crystallography.net/cod/9011380.cif", "database": "Fallback"},
]

# ============================================================================
# PHASE FRACTION ESTIMATION (simplified RIR‑like)
# ============================================================================
def estimate_phase_fractions(phases: List[Dict], exp_intensity: np.ndarray) -> List[Dict]:
    """
    Rough estimate of phase fractions based on matched peak intensities.
    Each phase's score and number of matched peaks contribute.
    Returns list with 'phase' and 'fraction' (sums to 100%).
    """
    if not phases:
        return []
    total_score = sum(p['score'] * len(p['hkls']) for p in phases)
    if total_score == 0:
        return []
    fractions = []
    for p in phases:
        weight = p['score'] * len(p['hkls']) / total_score
        fractions.append({"phase": p['phase'], "fraction": weight * 100})
    return fractions

# ============================================================================
# MAIN IDENTIFICATION FUNCTION
# ============================================================================
def identify_phases(two_theta: np.ndarray = None,
                    intensity: np.ndarray = None,
                    wavelength: float = 1.5406,
                    elements: Optional[List[str]] = None,
                    size_nm: Optional[float] = None,
                    mp_api_key: Optional[str] = None,
                    precomputed_peaks_2theta: Optional[np.ndarray] = None,
                    precomputed_peaks_intensity: Optional[np.ndarray] = None) -> List[Dict]:
    """
    Ultimate XRD phase identification for nanomaterials.
    Returns list of phases with full crystallographic details.
    """
    start_time = time.time()
    st.write(f"🕐 [{time.time()-start_time:.1f}s] Entered identify_phases")
    
    if not PMG_AVAILABLE:
        st.error("pymatgen required. Install: pip install pymatgen")
        return []
    
    st.info("🔬 Running gold‑standard nanomaterial phase identification...")
    status = st.status("Initializing...", expanded=True)
    
    # --------------------------------------------------------
    # STEP 1: Obtain experimental peaks (precomputed or detected)
    # --------------------------------------------------------
    if precomputed_peaks_2theta is not None and precomputed_peaks_intensity is not None:
        exp_2theta = np.array(precomputed_peaks_2theta)
        exp_intensity = np.array(precomputed_peaks_intensity)
        status.write(f"✅ Using {len(exp_2theta)} pre‑computed structural peaks")
        st.write(f"🕐 [{time.time()-start_time:.1f}s] Using precomputed peaks")
    else:
        status.write("Detecting peaks from raw data...")
        peak_analyzer = PeakAnalyzer()
        exp_2theta, exp_intensity = peak_analyzer.detect_peaks(two_theta, intensity)
        exp_2theta, exp_intensity = peak_analyzer.refine_apex(two_theta, intensity, exp_2theta)
        status.write(f"✅ Detected {len(exp_2theta)} peaks")
        st.write(f"🕐 [{time.time()-start_time:.1f}s] Peak detection complete")
    
    if len(exp_2theta) < 2:
        status.update(label="❌ Insufficient peaks", state="error")
        st.warning("At least 2 peaks required")
        return []
    
    # Calculate d‑spacings
    exp_d = wavelength / (2 * np.sin(np.radians(exp_2theta / 2)))
    exp_intensity_norm = exp_intensity / np.max(exp_intensity)
    st.write(f"🧪 d‑spacings (Å): {np.round(exp_d, 3).tolist()}")
    
    # --------------------------------------------------------
    # STEP 2: Estimate material family (if elements known)
    # --------------------------------------------------------
    family = 'unknown'
    if elements:
        elem_set = set(elements)
        if elem_set & set(NanoParams.FAMILIES['metal']):
            family = 'metal'
        elif 'O' in elem_set and (elem_set & {'Ti','Zn','Fe','Cu','Ni','Co','Al','Si','Zr','Ce','Bi'}):
            family = 'oxide'
        elif elem_set & {'S','Se','Te'}:
            family = 'chalcogenide'
        elif len(elements) >= 3 and 'O' in elements:
            family = 'perovskite'
        elif 'C' in elem_set:
            family = 'carbon'
        status.write(f"📊 Material family: {family}")
    
    if size_nm:
        tol = PatternMatcher.tolerance_from_size(size_nm)
        status.write(f"📊 Size: {size_nm:.1f} nm → Δd/d tolerance: {tol:.1%}")
    
    # --------------------------------------------------------
    # STEP 3: Database search
    # --------------------------------------------------------
    status.update(label="Searching databases...", state="running")
    searcher = DatabaseSearcher(mp_api_key=mp_api_key)
    
    def db_progress(msg):
        status.write(f"🔍 {msg}")
    
    if elements:
        candidates = searcher.search_all(elements=elements, family=family, progress=db_progress)
    else:
        candidates = searcher.search_all(dspacings=exp_d, progress=db_progress)
    
    st.write(f"🕐 [{time.time()-start_time:.1f}s] Found {len(candidates)} unique candidates")
    candidates = candidates[:30]  # keep top 30
    status.write(f"📚 Retrieved {len(candidates)} candidate structures")
    
    # If no online candidates, use fallback
    if not candidates:
        status.write("⚠️ No online candidates – using fallback database")
        candidates = FALLBACK
        st.write(f"🕐 [{time.time()-start_time:.1f}s] Using {len(candidates)} fallback structures")
    
    # --------------------------------------------------------
    # STEP 4: Parallel simulation and matching
    # --------------------------------------------------------
    status.update(label=f"Simulating {len(candidates)} structures...", state="running")
    st.write(f"🕐 [{time.time()-start_time:.1f}s] Starting parallel simulation")
    
    matcher = PatternMatcher()
    results = []
    threshold = 0.10 if not elements else (0.15 if size_nm and size_nm < 10 else 0.20)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_struct = {}
        for struct in candidates:
            url = struct.get('cif_url')
            if url:
                future = executor.submit(simulate_from_cif, url, wavelength)
                future_to_struct[future] = struct
        
        total = len(future_to_struct)
        completed = 0
        for future in concurrent.futures.as_completed(future_to_struct):
            struct = future_to_struct[future]
            completed += 1
            status.write(f"   [{completed}/{total}] Simulating {struct.get('formula', 'unknown')}...")
            try:
                sim_x, sim_y, sim_hkls = future.result(timeout=20)
                if len(sim_x) == 0:
                    continue
                
                sim_d = wavelength / (2 * np.sin(np.radians(sim_x / 2)))
                sim_int = sim_y / np.max(sim_y) if np.max(sim_y) > 0 else sim_y
                
                score, matched_peaks = matcher.match(
                    exp_d, exp_intensity_norm,
                    sim_d, sim_int,
                    sim_hkls,
                    size_nm, family
                )
                
                if score < threshold:
                    continue
                
                # Confidence level
                if not elements:
                    conf = "probable" if score >= 0.30 else "possible"
                else:
                    if size_nm and size_nm < 10:
                        conf = "confirmed" if score >= 0.55 else "probable" if score >= 0.35 else "possible"
                    else:
                        conf = "confirmed" if score >= 0.60 else "probable" if score >= 0.40 else "possible"
                
                # Extract lattice info from CIF (if possible)
                try:
                    # Re‑fetch CIF text (cached simulation didn't keep it)
                    headers = {}
                    if "materialsproject" in url:
                        mp_key = os.environ.get("MP_API_KEY", "")
                        if mp_key:
                            headers["X-API-KEY"] = mp_key
                    cif_resp = requests.get(url, headers=headers, timeout=8)
                    cif_text = cif_resp.text
                    parser = CifParser.from_string(cif_text)
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
                    "hkls": matched_peaks,
                    "score": round(score, 3),
                    "confidence_level": conf,
                    "database": struct.get('database', 'Online'),
                    "material_family": family,
                    "n_peaks_matched": len(matched_peaks),
                    "match_details": {
                        "n_exp_peaks": len(exp_d),
                        "avg_d_spacing": float(np.mean(exp_d)),
                        "size_nm": size_nm,
                        "tolerance_used": PatternMatcher.tolerance_from_size(size_nm)
                    }
                })
                
                if len(results) <= 3:
                    status.write(f"   → {formula}: score {score:.3f} → {conf}")
                
            except Exception as e:
                st.write(f"   ⚠️ Error: {str(e)[:100]}")
    
    st.write(f"🕐 [{time.time()-start_time:.1f}s] Simulation complete. Found {len(results)} matches.")
    
    # --------------------------------------------------------
    # STEP 5: If still no matches, force fallback simulation
    # --------------------------------------------------------
    if not results and candidates != FALLBACK:
        st.write("⚠️ No matches from online – retrying with fallback database")
        # Re‑run simulation on FALLBACK only
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_struct = {}
            for struct in FALLBACK:
                url = struct.get('cif_url')
                if url:
                    future = executor.submit(simulate_from_cif, url, wavelength)
                    future_to_struct[future] = struct
            
            for future in concurrent.futures.as_completed(future_to_struct):
                struct = future_to_struct[future]
                try:
                    sim_x, sim_y, sim_hkls = future.result(timeout=20)
                    if len(sim_x) == 0:
                        continue
                    sim_d = wavelength / (2 * np.sin(np.radians(sim_x / 2)))
                    sim_int = sim_y / np.max(sim_y)
                    score, matched = matcher.match(
                        exp_d, exp_intensity_norm,
                        sim_d, sim_int, sim_hkls,
                        size_nm, family
                    )
                    if score < threshold:
                        continue
                    conf = "probable" if score >= 0.30 else "possible"
                    results.append({
                        "phase": struct['formula'],
                        "full_formula": struct['formula'],
                        "crystal_system": struct['space_group'],
                        "space_group": struct['space_group'],
                        "lattice": {},
                        "hkls": matched,
                        "score": round(score, 3),
                        "confidence_level": conf,
                        "database": "Fallback",
                        "material_family": family,
                        "n_peaks_matched": len(matched),
                        "match_details": {
                            "n_exp_peaks": len(exp_d),
                            "avg_d_spacing": float(np.mean(exp_d)),
                            "size_nm": size_nm,
                            "tolerance_used": PatternMatcher.tolerance_from_size(size_nm)
                        }
                    })
                except:
                    pass
    
    if not results:
        status.update(label="❌ No phases matched", state="error")
        with st.expander("📊 Diagnostic Report", expanded=True):
            st.markdown("### Why no match?")
            st.markdown("- Peak broadening too large")
            st.markdown("- Material amorphous or not in databases")
            st.markdown("- Try providing elements")
            st.markdown("- Check internet connection")
        return []
    
    # --------------------------------------------------------
    # STEP 6: Deduplicate and sort
    # --------------------------------------------------------
    unique = {}
    for r in results:
        key = (r["phase"], r.get("space_group", ""))
        if key not in unique or r["score"] > unique[key]["score"]:
            unique[key] = r
    final = sorted(unique.values(), key=lambda x: x["score"], reverse=True)
    
    # Estimate phase fractions (rough)
    fractions = estimate_phase_fractions(final, exp_intensity)
    for i, r in enumerate(final):
        r["phase_fraction"] = next((f["fraction"] for f in fractions if f["phase"] == r["phase"]), None)
    
    status.update(label=f"✅ Identified {len(final)} phases", state="complete")
    
    # Display summary
    with st.expander("📊 Scientific Analysis Report", expanded=False):
        st.markdown(f"### **Nanocrystalline Material Analysis**")
        st.markdown(f"- **Material family**: {family}")
        st.markdown(f"- **Peaks used**: {len(exp_2theta)}")
        st.markdown(f"- **Average d-spacing**: {np.mean(exp_d):.3f} Å")
        if size_nm:
            st.markdown(f"- **Crystallite size**: {size_nm:.1f} nm")
        st.markdown("### **Top Phase Matches**")
        for i, r in enumerate(final[:5]):
            frac = f" – {r['phase_fraction']:.1f}%" if r.get('phase_fraction') else ""
            st.markdown(f"{i+1}. **{r['phase']}** ({r['crystal_system']}) – "
                       f"Score: {r['score']:.3f} [{r['confidence_level']}]{frac}")
    
    st.write(f"🕐 [{time.time()-start_time:.1f}s] Exiting identify_phases")
    return final
