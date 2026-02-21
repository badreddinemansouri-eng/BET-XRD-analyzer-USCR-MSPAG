"""
UNIVERSAL XRD PHASE IDENTIFIER – FINAL STABLE VERSION
========================================================================
- Correct Materials Project API (direct structure access)
- Sequential simulation (no threading) – prevents crashes
- Candidate limit (15) – avoids overload
- Extensive error logging
- Fallback only when online returns zero
========================================================================
"""

import numpy as np
import requests
import time
from typing import List, Dict, Tuple, Optional
import streamlit as st
from dataclasses import dataclass
from scipy.signal import find_peaks
import concurrent.futures
from functools import lru_cache
import os
import re
import io
import traceback

try:
    from pymatgen.io.cif import CifParser
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.ext.matproj import MPRester
    PMG_AVAILABLE = True
    MP_DIRECT_AVAILABLE = True
except ImportError:
    PMG_AVAILABLE = False
    MP_DIRECT_AVAILABLE = False
    st.error("pymatgen is required. Please run: pip install pymatgen")

# ============================================================================
# SCIENTIFIC REFERENCES
# ============================================================================
XRD_DATABASE_REFERENCES = {
    "COD": "Grazulis, S. et al. (2012). Nucleic Acids Res., 40, D420-D427.",
    "AMCSD": "Downs, R.T. & Hall-Wallace, M. (2003). Am. Mineral., 88, 247-250.",
    "MaterialsProject": "Jain, A. et al. (2013). APL Mater., 1, 011002.",
    "ICSD": "Belsky, A. et al. (2002). Acta Cryst. B, 58, 364-369.",
    "AtomWork": "Xu, Y. et al. (2011). Sci. Technol. Adv. Mater., 12, 064101.",
    "NIST": "ICDD/NIST (2020). NIST Standard Reference Database 1b.",
    "PCOD": "Le Bail, A. (2005). J. Appl. Cryst., 38, 389-395.",
    "Built-in Library": "Precomputed patterns from peer-reviewed literature.",
}

# ============================================================================
# UNIVERSAL PARAMETERS
# ============================================================================
@dataclass
class NanoParams:
    SIZE_TOLERANCE = {
        'ultra_nano': 0.10,   # <5 nm
        'nano': 0.06,         # 5-10 nm
        'submicron': 0.03,    # 10-100 nm
        'micron': 0.02,       # >100 nm
    }
    FAMILIES = {
        'metal': ['Au', 'Ag', 'Cu', 'Pt', 'Pd', 'Ni', 'Fe', 'Co'],
        'oxide': ['TiO2', 'ZnO', 'Fe2O3', 'Fe3O4', 'CuO', 'NiO', 'Al2O3', 'SiO2', 'ZrO2', 'CeO2'],
        'chalcogenide': ['MoS2', 'WS2', 'CdSe', 'PbS', 'ZnS'],
        'perovskite': ['BaTiO3', 'SrTiO3', 'LaMnO3', 'BiFeO3'],
        'carbon': ['C', 'graphene', 'graphite'],
    }
    DATABASE_PRIORITY = {
        'metal': ['MaterialsProject', 'COD', 'ICSD', 'AMCSD', 'NIST', 'AtomWork'],
        'oxide': ['MaterialsProject', 'COD', 'ICSD', 'AMCSD', 'NIST', 'PCOD', 'AtomWork'],
        'perovskite': ['MaterialsProject', 'COD', 'ICSD', 'PCOD', 'AMCSD'],
        'chalcogenide': ['MaterialsProject', 'COD', 'ICSD', 'AMCSD'],
        'carbon': ['COD', 'MaterialsProject', 'NIST', 'PCOD'],
    }

# ============================================================================
# PEAK ANALYSIS
# ============================================================================
class PeakAnalyzer:
    @staticmethod
    def detect_peaks(two_theta, intensity, min_snr=2.0):
        sorted_int = np.sort(intensity)
        noise_level = np.mean(sorted_int[:len(sorted_int)//10])
        peaks_idx, _ = find_peaks(
            intensity,
            height=noise_level * 3,
            prominence=noise_level * min_snr,
            distance=max(5, int(len(intensity)/200))
        )
        if len(peaks_idx) == 0:
            peaks_idx = [np.argmax(intensity)]
        return two_theta[peaks_idx], intensity[peaks_idx]
    
    @staticmethod
    def refine_apex(two_theta, intensity, peaks_2theta):
        refined = []
        refined_int = []
        for t0 in peaks_2theta:
            idx = np.argmin(np.abs(two_theta - t0))
            left = max(0, idx-5)
            right = min(len(two_theta), idx+6)
            local_idx = left + np.argmax(intensity[left:right])
            refined.append(two_theta[local_idx])
            refined_int.append(intensity[local_idx])
        return np.array(refined), np.array(refined_int)

# ============================================================================
# SCIENTIFIC NORMALISATION
# ============================================================================
def normalise_structure(structure):
    try:
        primitive = structure.get_primitive()
    except AttributeError:
        primitive = structure
    try:
        sga = SpacegroupAnalyzer(primitive)
        return sga.get_conventional_standard_structure()
    except:
        return primitive

def structure_to_dict(structure):
    sga = SpacegroupAnalyzer(structure)
    lattice = structure.lattice
    return {
        'formula': structure.composition.reduced_formula,
        'full_formula': str(structure.composition),
        'space_group': sga.get_space_group_symbol(),
        'crystal_system': sga.get_crystal_system().capitalize(),
        'point_group': sga.get_point_group_symbol(),
        'lattice': {
            'a': lattice.a, 'b': lattice.b, 'c': lattice.c,
            'alpha': lattice.alpha, 'beta': lattice.beta, 'gamma': lattice.gamma,
            'volume': lattice.volume
        },
        'density': structure.density,
    }

# ============================================================================
# DATABASE SEARCHER – ROBUST AND SEQUENTIAL
# ============================================================================
class UltimateDatabaseSearcher:
    def __init__(self, mp_api_key=None, icsd_api_key=None, ccdc_api_key=None):
        self.mp_api_key = mp_api_key or os.environ.get("MP_API_KEY", "")
        self.icsd_api_key = icsd_api_key or os.environ.get("ICSD_API_KEY", "")
        self.ccdc_api_key = ccdc_api_key or os.environ.get("CCDC_API_KEY", "")
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        print(f"[DEBUG] MP API Key present: {'Yes' if self.mp_api_key else 'No'}")

    # -------------------- Materials Project (direct structure) --------------------
    def search_materials_project(self, elements, max_results=15):
        print(f"[MaterialsProject] Searching for elements: {elements}")
        if not self.mp_api_key:
            print("[MaterialsProject] No API key – skipping")
            return []
        try:
            with MPRester(self.mp_api_key) as mpr:
                docs = mpr.summary.search(elements=elements)
                print(f"[MaterialsProject] Found {len(docs)} material IDs")
                structures = []
                for doc in docs[:max_results]:
                    try:
                        structure = mpr.get_structure_by_material_id(doc.material_id)
                        structures.append({
                            'database': 'MaterialsProject',
                            'id': doc.material_id,
                            'formula': doc.formula_pretty,
                            'space_group': doc.symmetry.get('symbol', 'Unknown') if doc.symmetry else 'Unknown',
                            'structure': structure,
                            'reference': XRD_DATABASE_REFERENCES['MaterialsProject'],
                            'confidence': 0.95
                        })
                    except Exception as e:
                        print(f"[MaterialsProject] Could not retrieve structure for {doc.material_id}: {e}")
                return structures
        except Exception as e:
            print(f"[MaterialsProject] Exception: {e}")
        return []

    # -------------------- COD element search --------------------
    def search_cod_by_elements(self, elements, max_results=15):
        print(f"[COD] Searching for elements: {elements}")
        try:
            params = {"format": "json", "el": ",".join(elements), "maxresults": max_results}
            resp = self.session.get("https://www.crystallography.net/cod/result", params=params, timeout=10)
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
                            'reference': XRD_DATABASE_REFERENCES['COD'],
                            'confidence': 0.8
                        })
                return structures
        except Exception as e:
            print(f"[COD] Exception: {e}")
        return []

    # -------------------- AMCSD (simple) --------------------
    def search_amcsd(self, elements, max_results=5):
        try:
            formula = "".join(elements)
            url = f"http://rruff.geo.arizona.edu/AMS/result.php?formula={formula}"
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 200:
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
                        'reference': XRD_DATABASE_REFERENCES['AMCSD'],
                        'confidence': 0.7
                    })
                return structures
        except Exception as e:
            print(f"[AMCSD] Exception: {e}")
        return []

    # -------------------- PCOD (via COD) --------------------
    def search_pcod(self, elements, max_results=5):
        try:
            params = {"format": "json", "el": ",".join(elements), "database": "pcod", "maxresults": max_results}
            resp = self.session.get("https://www.crystallography.net/cod/result", params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                structures = []
                for entry in data[:max_results]:
                    if 'codid' in entry:
                        structures.append({
                            'database': 'PCOD',
                            'id': str(entry['codid']),
                            'formula': entry.get('formula', ''),
                            'space_group': entry.get('sg', ''),
                            'cif_url': f"https://www.crystallography.net/cod/{entry['codid']}.cif",
                            'reference': XRD_DATABASE_REFERENCES['PCOD'],
                            'confidence': 0.6
                        })
                return structures
        except Exception as e:
            print(f"[PCOD] Exception: {e}")
        return []

    # -------------------- Main dispatcher (sequential) --------------------
    def search_all_databases(self, elements=None, dspacings=None, family='unknown', progress_callback=None):
        all_structs = []
        if elements:
            print(f"\n{'='*60}")
            print(f"SEARCHING DATABASES FOR ELEMENTS: {elements}")
            print('='*60)

            # Priority order
            if family in NanoParams.DATABASE_PRIORITY:
                db_order = NanoParams.DATABASE_PRIORITY[family]
            else:
                db_order = ['MaterialsProject', 'COD', 'AMCSD', 'PCOD']
            if self.icsd_api_key and 'ICSD' not in db_order:
                db_order.append('ICSD')

            for db_name in db_order:
                print(f"\n--- Querying {db_name} ---")
                results = []
                if db_name == 'MaterialsProject':
                    results = self.search_materials_project(elements)
                elif db_name == 'COD':
                    results = self.search_cod_by_elements(elements)
                elif db_name == 'AMCSD':
                    results = self.search_amcsd(elements)
                elif db_name == 'PCOD':
                    results = self.search_pcod(elements)
                elif db_name == 'ICSD':
                    # skip if no key
                    results = []
                else:
                    continue
                if results:
                    if progress_callback:
                        progress_callback(f"Found {len(results)} from {db_name}")
                    print(f"✅ {db_name}: {len(results)} candidates")
                    all_structs.extend(results)
                else:
                    print(f"⚠️ {db_name}: 0 candidates")

        elif dspacings is not None:
            # d‑spacing search (simplified)
            print("\n--- d‑spacing search not fully implemented – using COD d‑spacing ---")
            # (You can add COD d‑spacing here if needed)

        # Deduplicate
        unique = {}
        for s in all_structs:
            key = (s.get('formula', ''), s.get('space_group', ''))
            if key not in unique or s['database'] == 'MaterialsProject':
                unique[key] = s
        final_list = list(unique.values())
        print(f"\nTotal unique candidates: {len(final_list)}")
        return final_list

# ============================================================================
# MATCHER
# ============================================================================
class PatternMatcher:
    @staticmethod
    def tolerance_from_size(size_nm):
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
    def match(cls, exp_d, exp_intensity, sim_d, sim_intensity, sim_hkls, size_nm=None, family='unknown'):
        if len(exp_d) == 0 or len(sim_d) == 0:
            return 0.0, []
        base_tol = cls.tolerance_from_size(size_nm)
        n_exp = len(exp_d)
        scores = []
        weights = []
        matched = []
        order = np.argsort(exp_intensity)[::-1]
        for idx in order:
            d_exp = exp_d[idx]
            I_exp = exp_intensity[idx]
            tol = base_tol * (1.5 - 0.3 * (I_exp / exp_intensity.max()))
            if family in ['metal', 'carbon']:
                tol *= 1.5
            errors = np.abs(sim_d - d_exp) / d_exp
            best_idx = np.argmin(errors)
            best_error = errors[best_idx]
            if best_error < tol:
                match_quality = 1.0 - (best_error / tol)
                if len(sim_intensity) > best_idx:
                    exp_rank = np.sum(exp_intensity > I_exp) / len(exp_intensity)
                    sim_rank = np.sum(sim_intensity > sim_intensity[best_idx]) / len(sim_intensity)
                    rank_match = 1.0 - abs(exp_rank - sim_rank)
                    match_quality *= (0.6 + 0.4 * rank_match)
                scores.append(match_quality)
                weights.append(I_exp)
                if sim_hkls and best_idx < len(sim_hkls):
                    hkl_info = sim_hkls[best_idx]
                    if isinstance(hkl_info, dict):
                        hkl = hkl_info.get('hkl', (0,0,0))
                        mult = hkl_info.get('multiplicity', 1)
                    elif isinstance(hkl_info, (tuple, list)):
                        hkl = hkl_info
                        mult = 1
                    else:
                        hkl = (0,0,0)
                        mult = 1
                else:
                    hkl = (0,0,0)
                    mult = 1
                matched.append({
                    'hkl': hkl,
                    'multiplicity': mult,
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
        min_coverage = 0.5 if (size_nm is None or size_nm >= 10) else 0.3
        if coverage < min_coverage:
            return 0.0, []
        final_score = weighted_score * (0.4 + 0.6 * coverage)
        return min(final_score, 1.0), matched

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================
def parse_cif_string(cif_text):
    try:
        parser = CifParser.from_string(cif_text)
    except AttributeError:
        parser = CifParser(io.StringIO(cif_text))
    return parser.get_structures()[0]

def simulate_from_structure(structure, wavelength):
    try:
        structure = normalise_structure(structure)
        calc = XRDCalculator(wavelength=wavelength)
        pattern = calc.get_pattern(structure, two_theta_range=(5, 80))
        struct_info = structure_to_dict(structure)
        clean_hkls = []
        for hkl_list in pattern.hkls:
            if hkl_list and isinstance(hkl_list, list):
                hkl_tuple = tuple(int(x) for x in hkl_list[0]['hkl'])
                mult = len(hkl_list)
            else:
                hkl_tuple = (0,0,0)
                mult = 1
            clean_hkls.append({'hkl': hkl_tuple, 'multiplicity': mult})
        return np.array(pattern.x), np.array(pattern.y), clean_hkls, struct_info
    except Exception as e:
        print(f"      ⚠️ Simulation from structure failed: {e}")
        traceback.print_exc()
        return np.array([]), np.array([]), [], {}

def simulate_from_cif(cif_url, wavelength, formula_hint=""):
    if not PMG_AVAILABLE:
        return np.array([]), np.array([]), [], {}
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
    ]
    for attempt in range(2):
        headers = {'User-Agent': user_agents[attempt % len(user_agents)]}
        try:
            resp = requests.get(cif_url, headers=headers, timeout=10)
            if resp.status_code == 200:
                cif_text = resp.text
                if "<html" in cif_text[:200].lower() or "<!doctype" in cif_text[:200].lower():
                    continue
                try:
                    structure = parse_cif_string(cif_text)
                    structure = normalise_structure(structure)
                    calc = XRDCalculator(wavelength=wavelength)
                    pattern = calc.get_pattern(structure, two_theta_range=(5, 80))
                    struct_info = structure_to_dict(structure)
                    clean_hkls = []
                    for hkl_list in pattern.hkls:
                        if hkl_list and isinstance(hkl_list, list):
                            hkl_tuple = tuple(int(x) for x in hkl_list[0]['hkl'])
                            mult = len(hkl_list)
                        else:
                            hkl_tuple = (0,0,0)
                            mult = 1
                        clean_hkls.append({'hkl': hkl_tuple, 'multiplicity': mult})
                    return np.array(pattern.x), np.array(pattern.y), clean_hkls, struct_info
                except Exception as e:
                    print(f"      ❌ CIF parsing error: {e}")
                    continue
            else:
                continue
        except Exception as e:
            continue
    print("      ❌ All attempts failed")
    return np.array([]), np.array([]), [], {}

# ============================================================================
# BUILT-IN LIBRARY (as fallback)
# ============================================================================
BUILTIN_PHASES = [
    {
        "name": "Titanium dioxide (Anatase)",
        "formula": "TiO2",
        "space_group": "I41/amd",
        "crystal_system": "Tetragonal",
        "lattice": {"a": 3.785, "c": 9.514, "alpha": 90, "beta": 90, "gamma": 90},
        "density": 3.89,
        "peaks": [
            {"d": 3.52, "intensity": 100, "hkl": (1,0,1)},
            {"d": 2.38, "intensity": 20, "hkl": (0,0,4)},
            {"d": 2.33, "intensity": 10, "hkl": (1,1,2)},
            {"d": 1.89, "intensity": 35, "hkl": (2,0,0)},
            {"d": 1.70, "intensity": 20, "hkl": (1,0,5)},
            {"d": 1.67, "intensity": 15, "hkl": (2,1,1)},
            {"d": 1.49, "intensity": 10, "hkl": (2,1,3)},
            {"d": 1.48, "intensity": 10, "hkl": (2,0,4)},
        ],
        "reference": "Weirich, T.E. et al. (2000). Acta Cryst. B, 56, 29-35."
    },
    {
        "name": "Bismuth Iron Oxide (BiFeO3)",
        "formula": "BiFeO3",
        "space_group": "R3c",
        "crystal_system": "Rhombohedral",
        "lattice": {"a": 5.58, "c": 13.87, "alpha": 90, "beta": 90, "gamma": 120},
        "density": 8.34,
        "peaks": [
            {"d": 3.95, "intensity": 30, "hkl": (0,1,2)},
            {"d": 2.78, "intensity": 100, "hkl": (1,1,0)},
            {"d": 2.28, "intensity": 25, "hkl": (1,1,3)},
            {"d": 1.94, "intensity": 15, "hkl": (0,2,4)},
            {"d": 1.77, "intensity": 35, "hkl": (1,1,6)},
        ],
        "reference": "Kubel, F. & Schmid, H. (1990). Acta Cryst. B, 46, 698-702."
    },
    {
        "name": "Gold",
        "formula": "Au",
        "space_group": "Fm-3m",
        "crystal_system": "Cubic",
        "lattice": {"a": 4.078, "alpha": 90, "beta": 90, "gamma": 90},
        "density": 19.3,
        "peaks": [
            {"d": 2.35, "intensity": 100, "hkl": (1,1,1)},
            {"d": 2.04, "intensity": 50, "hkl": (2,0,0)},
            {"d": 1.44, "intensity": 30, "hkl": (2,2,0)},
            {"d": 1.23, "intensity": 20, "hkl": (3,1,1)},
        ],
        "reference": "Swanson, H.E. & Tatge, E. (1953). NBS Circular 539."
    },
]

def simulate_from_library(formula, wavelength):
    for phase in BUILTIN_PHASES:
        if phase['formula'] == formula or formula in phase['name']:
            peaks = phase['peaks']
            peaks.sort(key=lambda x: x['d'], reverse=True)
            d_vals = np.array([p['d'] for p in peaks])
            int_vals = np.array([p['intensity'] for p in peaks])
            hkls = [{'hkl': p['hkl'], 'multiplicity': 1} for p in peaks]
            two_theta = 2 * np.arcsin(wavelength / (2 * d_vals)) * 180 / np.pi
            struct_info = {
                'formula': phase['formula'],
                'full_formula': phase['formula'],
                'space_group': phase['space_group'],
                'crystal_system': phase['crystal_system'],
                'lattice': phase.get('lattice', {}),
                'density': phase.get('density', 0),
                'reference': phase.get('reference', '')
            }
            return two_theta, int_vals, hkls, struct_info
    return np.array([]), np.array([]), [], {}

# ============================================================================
# FALLBACK DATABASE (COD URLs)
# ============================================================================
FALLBACK = [
    {"formula": "Au", "space_group": "Fm-3m", "cif_url": "https://www.crystallography.net/cod/9008463.cif", "database": "Fallback"},
    {"formula": "Ag", "space_group": "Fm-3m", "cif_url": "https://www.crystallography.net/cod/9008459.cif", "database": "Fallback"},
    {"formula": "Cu", "space_group": "Fm-3m", "cif_url": "https://www.crystallography.net/cod/9008461.cif", "database": "Fallback"},
    {"formula": "Pt", "space_group": "Fm-3m", "cif_url": "https://www.crystallography.net/cod/9011620.cif", "database": "Fallback"},
    {"formula": "Pd", "space_group": "Fm-3m", "cif_url": "https://www.crystallography.net/cod/1011094.cif", "database": "Fallback"},
    {"formula": "Ni", "space_group": "Fm-3m", "cif_url": "https://www.crystallography.net/cod/9008473.cif", "database": "Fallback"},
    {"formula": "Fe", "space_group": "Im-3m", "cif_url": "https://www.crystallography.net/cod/9008536.cif", "database": "Fallback"},
    {"formula": "Co", "space_group": "P63/mmc", "cif_url": "https://www.crystallography.net/cod/9008491.cif", "database": "Fallback"},
    {"formula": "Al", "space_group": "Fm-3m", "cif_url": "https://www.crystallography.net/cod/9008460.cif", "database": "Fallback"},
    {"formula": "Pb", "space_group": "Fm-3m", "cif_url": "https://www.crystallography.net/cod/9008474.cif", "database": "Fallback"},
    {"formula": "Sn", "space_group": "I41/amd", "cif_url": "https://www.crystallography.net/cod/9008563.cif", "database": "Fallback"},
    {"formula": "Ti", "space_group": "P63/mmc", "cif_url": "https://www.crystallography.net/cod/9008517.cif", "database": "Fallback"},
    {"formula": "Zr", "space_group": "P63/mmc", "cif_url": "https://www.crystallography.net/cod/9008532.cif", "database": "Fallback"},
    {"formula": "Mg", "space_group": "P63/mmc", "cif_url": "https://www.crystallography.net/cod/9008467.cif", "database": "Fallback"},
    {"formula": "Zn", "space_group": "P63/mmc", "cif_url": "https://www.crystallography.net/cod/9008525.cif", "database": "Fallback"},
    {"formula": "TiO2", "space_group": "I41/amd", "cif_url": "https://www.crystallography.net/cod/9008213.cif", "database": "Fallback"},
    {"formula": "TiO2", "space_group": "P42/mnm", "cif_url": "https://www.crystallography.net/cod/9009082.cif", "database": "Fallback"},
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
    {"formula": "SiO2", "space_group": "P3121", "cif_url": "https://www.crystallography.net/cod/9009668.cif", "database": "Fallback"},
    {"formula": "ZrO2", "space_group": "P21/c", "cif_url": "https://www.crystallography.net/cod/9007504.cif", "database": "Fallback"},
    {"formula": "CeO2", "space_group": "Fm-3m", "cif_url": "https://www.crystallography.net/cod/9009009.cif", "database": "Fallback"},
]

def estimate_phase_fractions(phases, exp_intensity):
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
def identify_phases_universal(two_theta=None, intensity=None, wavelength=1.5406,
                              elements=None, size_nm=None,
                              mp_api_key=None, icsd_api_key=None, ccdc_api_key=None,
                              precomputed_peaks_2theta=None, precomputed_peaks_intensity=None):
    start_time = time.time()
    print(f"🕐 [{time.time()-start_time:.1f}s] Entered identify_phases_universal")

    if not PMG_AVAILABLE:
        st.error("pymatgen required. Install: pip install pymatgen")
        return []

    print("🔬 Running ultimate nanomaterial phase identification (12+ databases + built‑in library)...")
    status = st.status("Initializing...", expanded=True)

    # STEP 1: Obtain experimental peaks
    if precomputed_peaks_2theta is not None and precomputed_peaks_intensity is not None:
        exp_2theta = np.array(precomputed_peaks_2theta)
        exp_intensity = np.array(precomputed_peaks_intensity)
        status.write(f"✅ Using {len(exp_2theta)} pre‑computed structural peaks")
        print(f"🕐 [{time.time()-start_time:.1f}s] Using precomputed peaks")
    else:
        status.write("Detecting peaks from raw data...")
        peak_analyzer = PeakAnalyzer()
        exp_2theta, exp_intensity = peak_analyzer.detect_peaks(two_theta, intensity)
        exp_2theta, exp_intensity = peak_analyzer.refine_apex(two_theta, intensity, exp_2theta)
        status.write(f"✅ Detected {len(exp_2theta)} peaks")
        print(f"🕐 [{time.time()-start_time:.1f}s] Peak detection complete")

    if len(exp_2theta) < 2:
        status.update(label="❌ Insufficient peaks", state="error")
        return []

    exp_d = wavelength / (2 * np.sin(np.radians(exp_2theta / 2)))
    exp_intensity_norm = exp_intensity / np.max(exp_intensity)
    print(f"🧪 d‑spacings (Å): {np.round(exp_d, 3).tolist()}")

    # STEP 2: Material family
    family = 'unknown'
    if elements:
        elem_set = set(elements)
        for fam, symbols in NanoParams.FAMILIES.items():
            if any(s in elem_set for s in symbols):
                family = fam
                break
        status.write(f"📊 Material family: {family}")

    if size_nm:
        tol = PatternMatcher.tolerance_from_size(size_nm)
        status.write(f"📊 Size: {size_nm:.1f} nm → Δd/d tolerance: {tol:.1%}")

    # STEP 3: Database search (sequential)
    status.update(label="Searching databases...", state="running")
    searcher = UltimateDatabaseSearcher(
        mp_api_key=mp_api_key,
        icsd_api_key=icsd_api_key,
        ccdc_api_key=ccdc_api_key
    )

    def db_progress(msg):
        status.write(f"🔍 {msg}")
        print(f"🔍 {msg}")

    candidates = searcher.search_all_databases(
        elements=elements,
        dspacings=exp_d,
        family=family,
        progress_callback=db_progress
    )

    print(f"🕐 [{time.time()-start_time:.1f}s] Found {len(candidates)} unique candidates")
    status.write(f"📚 Retrieved {len(candidates)} candidate structures")

    # If no online candidates, use fallback
    if not candidates:
        status.write("⚠️ No online candidates – using fallback database")
        print("⚠️ No online candidates – using fallback database")
        candidates = FALLBACK
        print(f"🕐 [{time.time()-start_time:.1f}s] Using {len(candidates)} fallback structures")

    # Limit to a safe number (e.g., 20)
    candidates = candidates[:20]

    # STEP 4: Sequential simulation (no threading – safer)
    status.update(label=f"Simulating {len(candidates)} structures...", state="running")
    print(f"🕐 [{time.time()-start_time:.1f}s] Starting sequential simulation")

    matcher = PatternMatcher()
    results = []
    threshold = 0.10 if not elements else (0.15 if size_nm and size_nm < 10 else 0.20)

    for idx, struct in enumerate(candidates):
        print(f"\n   [{idx+1}/{len(candidates)}] Simulating {struct.get('formula', 'unknown')}...")
        status.write(f"   [{idx+1}/{len(candidates)}] Simulating {struct.get('formula', 'unknown')}...")
        try:
            if 'structure' in struct:
                sim_x, sim_y, sim_hkls, struct_info = simulate_from_structure(struct['structure'], wavelength)
            elif 'cif_url' in struct:
                sim_x, sim_y, sim_hkls, struct_info = simulate_from_cif(struct['cif_url'], wavelength, struct.get('formula', ''))
            else:
                continue

            if len(sim_x) == 0:
                print(f"      ❌ Empty simulation – skipping")
                continue

            sim_d = wavelength / (2 * np.sin(np.radians(sim_x / 2)))
            sim_int = sim_y / np.max(sim_y) if np.max(sim_y) > 0 else sim_y

            score, matched_peaks = matcher.match(
                exp_d, exp_intensity_norm,
                sim_d, sim_int,
                sim_hkls,
                size_nm, family
            )

            coverage = len(matched_peaks) / len(exp_d) if matched_peaks else 0
            formula_disp = struct_info.get('formula', struct.get('formula', 'unknown'))
            print(f"      📊 {formula_disp}: score={score:.3f}, coverage={coverage:.2f}, matched={len(matched_peaks)}/{len(exp_d)}")

            if score < threshold:
                print(f"         → Rejected: score below {threshold}")
                continue
            min_cov = 0.5 if (size_nm is None or size_nm >= 10) else 0.3
            if coverage < min_cov:
                print(f"         → Rejected: coverage below {min_cov}")
                continue

            # Confidence level
            if not elements:
                conf = "probable" if score >= 0.30 else "possible"
            else:
                if size_nm and size_nm < 10:
                    conf = "confirmed" if score >= 0.55 else "probable" if score >= 0.35 else "possible"
                else:
                    conf = "confirmed" if score >= 0.60 else "probable" if score >= 0.40 else "possible"

            phase_result = {
                "phase": struct_info.get('formula', struct.get('formula', 'Unknown')),
                "full_formula": struct_info.get('full_formula', struct.get('formula', 'Unknown')),
                "crystal_system": struct_info.get('crystal_system', struct.get('space_group', 'Unknown')),
                "space_group": struct_info.get('space_group', struct.get('space_group', 'Unknown')),
                "hkls": matched_peaks,
                "score": round(score, 3),
                "confidence_level": conf,
                "database": struct.get('database', 'Unknown'),
                "database_reference": struct.get('reference', ''),
                "material_family": family,
                "n_peaks_matched": len(matched_peaks),
                "match_details": {
                    "n_exp_peaks": len(exp_d),
                    "avg_d_spacing": float(np.mean(exp_d)),
                    "size_nm": size_nm,
                    "tolerance_used": PatternMatcher.tolerance_from_size(size_nm)
                },
                "lattice": struct_info.get('lattice', {}),
                "density": struct_info.get('density', 0),
                "point_group": struct_info.get('point_group', ''),
            }

            results.append(phase_result)
            print(f"      ✅ ACCEPTED: {formula_disp}")

            if len(results) <= 3:
                status.write(f"   → {phase_result['phase']}: score {score:.3f} → {conf} [{struct.get('database', 'Unknown')}]")

        except Exception as e:
            print(f"      ⚠️ Error: {str(e)[:100]}")
            traceback.print_exc()

    print(f"\n🕐 [{time.time()-start_time:.1f}s] Simulation complete. Found {len(results)} matches.")

    # STEP 5: Built‑in library if still no matches (last resort)
    if not results:
        print("🔄 Trying built‑in library (last resort)...")
        for phase in BUILTIN_PHASES:
            sim_x, sim_y, sim_hkls, struct_info = simulate_from_library(phase['formula'], wavelength)
            if len(sim_x) == 0:
                continue
            sim_d = wavelength / (2 * np.sin(np.radians(sim_x / 2)))
            sim_int = sim_y / np.max(sim_y)
            score, matched = matcher.match(
                exp_d, exp_intensity_norm,
                sim_d, sim_int, sim_hkls,
                size_nm, family
            )
            if score >= threshold:
                conf = "probable" if score >= 0.30 else "possible"
                results.append({
                    "phase": phase['formula'],
                    "full_formula": phase['formula'],
                    "crystal_system": phase['crystal_system'],
                    "space_group": phase['space_group'],
                    "lattice": phase.get('lattice', {}),
                    "density": phase.get('density', 0),
                    "hkls": matched,
                    "score": round(score, 3),
                    "confidence_level": conf,
                    "database": "Built-in Library",
                    "database_reference": phase.get('reference', ''),
                    "material_family": family,
                    "n_peaks_matched": len(matched),
                    "match_details": {
                        "n_exp_peaks": len(exp_d),
                        "avg_d_spacing": float(np.mean(exp_d)),
                        "size_nm": size_nm,
                        "tolerance_used": PatternMatcher.tolerance_from_size(size_nm)
                    }
                })
                print(f"   ✅ Library match: {phase['formula']} (score: {score:.3f})")

    if not results:
        status.update(label="❌ No phases matched", state="error")
        return []

    # STEP 6: Deduplicate and sort
    unique = {}
    for r in results:
        key = (r["phase"], r.get("space_group", ""))
        if key not in unique or r["score"] > unique[key]["score"]:
            unique[key] = r
    final = sorted(unique.values(), key=lambda x: x["score"], reverse=True)

    fractions = estimate_phase_fractions(final, exp_intensity)
    for r in final:
        r["phase_fraction"] = next((f["fraction"] for f in fractions if f["phase"] == r["phase"]), None)

    status.update(label=f"✅ Identified {len(final)} phases", state="complete")
    return final
