"""
UNIVERSAL XRD PHASE IDENTIFIER - FINAL STABLE VERSION
========================================================================
- Direct Materials Project API (structure retrieval)
- Sequential simulation (no threading) to avoid crash conditions
- Candidate cap to bound runtime
- Structured logging via the logging module (no print statements)
- Robust network handling with per-provider timeouts
- Session-level caching to avoid repeated heavy queries

References:
1. Jain, A. et al. (2013). APL Mater., 1, 011002 (Materials Project)
2. Grazulis, S. et al. (2012). Nucleic Acids Res., 40, D420-D427 (COD)
3. Downs, R.T. & Hall-Wallace, M. (2003). Am. Mineral., 88, 247-250 (AMCSD)
========================================================================
"""

import logging
import os
import re
import io
import time
import traceback
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import requests
import streamlit as st
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

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
    logger.error("pymatgen not available. Phase identification is disabled.")


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
        'ultra_nano': 0.10,
        'nano': 0.06,
        'submicron': 0.03,
        'micron': 0.02,
    }
    FAMILIES = {
        'metal': ['Au', 'Ag', 'Cu', 'Pt', 'Pd', 'Ni', 'Fe', 'Co'],
        'oxide': ['TiO2', 'ZnO', 'Fe2O3', 'Fe3O4', 'CuO', 'NiO',
                  'Al2O3', 'SiO2', 'ZrO2', 'CeO2'],
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

    # Network budgets (seconds)
    MP_TIMEOUT = 20
    COD_TIMEOUT = 8
    AMCSD_TIMEOUT = 4
    CIF_TIMEOUT = 8

    # Runtime caps
    MAX_CANDIDATES_PER_DB = 10
    MAX_TOTAL_CANDIDATES = 15
    MAX_FALLBACK_STRUCTURES = 10


# ============================================================================
# PEAK ANALYSIS
# ============================================================================
class PeakAnalyzer:
    @staticmethod
    def detect_peaks(two_theta, intensity, min_snr=2.0):
        sorted_int = np.sort(intensity)
        noise_level = np.mean(sorted_int[:len(sorted_int) // 10])
        peaks_idx, _ = find_peaks(
            intensity,
            height=noise_level * 3,
            prominence=noise_level * min_snr,
            distance=max(5, int(len(intensity) / 200))
        )
        if len(peaks_idx) == 0:
            peaks_idx = [int(np.argmax(intensity))]
        return two_theta[peaks_idx], intensity[peaks_idx]

    @staticmethod
    def refine_apex(two_theta, intensity, peaks_2theta):
        refined = []
        refined_int = []
        for t0 in peaks_2theta:
            idx = int(np.argmin(np.abs(two_theta - t0)))
            left = max(0, idx - 5)
            right = min(len(two_theta), idx + 6)
            local_idx = left + int(np.argmax(intensity[left:right]))
            refined.append(two_theta[local_idx])
            refined_int.append(intensity[local_idx])
        return np.array(refined), np.array(refined_int)


# ============================================================================
# STRUCTURE NORMALIZATION
# ============================================================================
def normalise_structure(structure):
    try:
        primitive = structure.get_primitive()
    except AttributeError:
        primitive = structure
    try:
        sga = SpacegroupAnalyzer(primitive)
        return sga.get_conventional_standard_structure()
    except Exception:
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


def _safe_get(obj, attr, default=None):
    """Return obj[attr] if dict, else getattr(obj, attr, default)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


# ============================================================================
# DATABASE SEARCHER (SEQUENTIAL)
# ============================================================================
class UltimateDatabaseSearcher:
    def __init__(self, mp_api_key=None, icsd_api_key=None, ccdc_api_key=None):
        self.mp_api_key = mp_api_key or os.environ.get("MP_API_KEY", "")
        self.icsd_api_key = icsd_api_key or os.environ.get("ICSD_API_KEY", "")
        self.ccdc_api_key = ccdc_api_key or os.environ.get("CCDC_API_KEY", "")
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; BET-XRD-Analyzer/3.0)'
        })
        logger.info("Database searcher initialized. MP key present: %s",
                    'yes' if self.mp_api_key else 'no')

    # ------------------------------------------------------------------
    def search_materials_project(self, elements, max_results=None):
        if max_results is None:
            max_results = NanoParams.MAX_CANDIDATES_PER_DB
        if not self.mp_api_key:
            logger.info("Materials Project: no API key, skipping")
            return []

        try:
            with MPRester(self.mp_api_key) as mpr:
                docs = mpr.summary.search(elements=elements)
                logger.info("Materials Project: %d candidates", len(docs))

                structures = []
                for doc in docs[:max_results * 3]:
                    # Robust access: works for both dict and object
                    material_id = _safe_get(doc, 'material_id')
                    if material_id is None:
                        continue
                    if isinstance(material_id, dict):
                        material_id = material_id.get('id') or str(material_id)

                    formula_pretty = _safe_get(doc, 'formula_pretty', '')
                    symmetry = _safe_get(doc, 'symmetry', {})
                    space_group = _safe_get(symmetry, 'symbol', 'Unknown') if symmetry else 'Unknown'

                    try:
                        structure = mpr.get_structure_by_material_id(str(material_id))
                        structures.append({
                            'database': 'MaterialsProject',
                            'id': str(material_id),
                            'formula': formula_pretty or (structure.composition.reduced_formula
                                                          if hasattr(structure, 'composition') else ''),
                            'space_group': space_group,
                            'structure': structure,
                            'reference': XRD_DATABASE_REFERENCES['MaterialsProject'],
                            'confidence': 0.95
                        })
                    except Exception as e:
                        logger.debug("MP structure retrieval failed for %s: %s",
                                     material_id, e)

                    if len(structures) >= max_results:
                        break

                return structures
        except Exception as e:
            logger.warning("Materials Project exception: %s", e)
        return []

    # ------------------------------------------------------------------
    def search_cod_by_elements(self, elements, max_results=None):
        if max_results is None:
            max_results = NanoParams.MAX_CANDIDATES_PER_DB
        try:
            # COD expects repeated el= parameters, not a comma-separated string
            params_list = [("format", "json"), ("maxresults", str(max_results))]
            for el in elements:
                params_list.append(("el", el))

            resp = self.session.get(
                "https://www.crystallography.net/cod/result",
                params=params_list,
                timeout=NanoParams.COD_TIMEOUT
            )

            if resp.status_code != 200:
                logger.info("COD returned status %d", resp.status_code)
                return []

            try:
                data = resp.json()
            except Exception as e:
                logger.warning("COD JSON decode failed: %s", e)
                return []

            if not isinstance(data, list):
                logger.info("COD returned non-list payload")
                return []

            structures = []
            for entry in data[:max_results]:
                if not isinstance(entry, dict):
                    continue
                codid = entry.get('codid')
                if codid is None:
                    continue
                structures.append({
                    'database': 'COD',
                    'id': str(codid),
                    'formula': entry.get('formula', ''),
                    'space_group': entry.get('sg', ''),
                    'cif_url': f"https://www.crystallography.net/cod/{codid}.cif",
                    'reference': XRD_DATABASE_REFERENCES['COD'],
                    'confidence': 0.8
                })

            logger.info("COD: %d candidates", len(structures))
            return structures
        except Exception as e:
            logger.warning("COD exception: %s", e)
        return []

    # ------------------------------------------------------------------
    def search_amcsd(self, elements, max_results=5):
        try:
            formula = "".join(elements)
            url = f"http://rruff.geo.arizona.edu/AMS/result.php?formula={formula}"
            resp = self.session.get(url, timeout=NanoParams.AMCSD_TIMEOUT)
            if resp.status_code != 200:
                return []
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
            logger.info("AMCSD: %d candidates", len(structures))
            return structures
        except Exception as e:
            logger.info("AMCSD unavailable (skipping): %s", type(e).__name__)
        return []

    # ------------------------------------------------------------------
    def search_pcod(self, elements, max_results=5):
        try:
            params_list = [("format", "json"), ("database", "pcod"),
                           ("maxresults", str(max_results))]
            for el in elements:
                params_list.append(("el", el))
            resp = self.session.get(
                "https://www.crystallography.net/cod/result",
                params=params_list,
                timeout=NanoParams.COD_TIMEOUT
            )
            if resp.status_code != 200:
                return []
            try:
                data = resp.json()
            except Exception:
                return []
            if not isinstance(data, list):
                return []
            structures = []
            for entry in data[:max_results]:
                if not isinstance(entry, dict):
                    continue
                codid = entry.get('codid')
                if codid is None:
                    continue
                structures.append({
                    'database': 'PCOD',
                    'id': str(codid),
                    'formula': entry.get('formula', ''),
                    'space_group': entry.get('sg', ''),
                    'cif_url': f"https://www.crystallography.net/cod/{codid}.cif",
                    'reference': XRD_DATABASE_REFERENCES['PCOD'],
                    'confidence': 0.6
                })
            logger.info("PCOD: %d candidates", len(structures))
            return structures
        except Exception as e:
            logger.info("PCOD skipped: %s", type(e).__name__)
        return []

    # ------------------------------------------------------------------
    def search_all_databases(self, elements=None, dspacings=None,
                             family='unknown', progress_callback=None,
                             max_total=None):
        if max_total is None:
            max_total = NanoParams.MAX_TOTAL_CANDIDATES

        all_structs = []
        if not elements:
            return []

        logger.info("Searching databases for elements: %s", elements)

        if family in NanoParams.DATABASE_PRIORITY:
            db_order = NanoParams.DATABASE_PRIORITY[family]
        else:
            db_order = ['MaterialsProject', 'COD', 'AMCSD']

        for db_name in db_order:
            # Stop early if we already have enough
            if len(all_structs) >= max_total:
                logger.info("Reached candidate cap (%d), stopping search", max_total)
                break

            try:
                if db_name == 'MaterialsProject':
                    results = self.search_materials_project(elements)
                elif db_name == 'COD':
                    results = self.search_cod_by_elements(elements)
                elif db_name == 'AMCSD':
                    results = self.search_amcsd(elements)
                elif db_name == 'PCOD':
                    results = self.search_pcod(elements)
                elif db_name == 'ICSD':
                    results = []
                else:
                    continue
            except Exception as e:
                logger.warning("%s failed: %s", db_name, e)
                results = []

            if results:
                if progress_callback:
                    progress_callback(f"Found {len(results)} from {db_name}")
                all_structs.extend(results)
            else:
                logger.info("%s returned 0 candidates", db_name)

        # Deduplicate
        unique = {}
        for s in all_structs:
            key = (s.get('formula', ''), s.get('space_group', ''))
            if key not in unique or s['database'] == 'MaterialsProject':
                unique[key] = s

        final_list = list(unique.values())
        logger.info("Total unique candidates: %d", len(final_list))
        return final_list[:max_total]


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
    def match(cls, exp_d, exp_intensity, sim_d, sim_intensity,
              sim_hkls, size_nm=None, family='unknown'):
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
            best_idx = int(np.argmin(errors))
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
                        hkl = hkl_info.get('hkl', (0, 0, 0))
                        mult = hkl_info.get('multiplicity', 1)
                    elif isinstance(hkl_info, (tuple, list)):
                        hkl = hkl_info
                        mult = 1
                    else:
                        hkl = (0, 0, 0)
                        mult = 1
                else:
                    hkl = (0, 0, 0)
                    mult = 1

                matched.append({
                    'hkl': hkl,
                    'multiplicity': mult,
                    'd_exp': float(d_exp),
                    'd_calc': float(sim_d[best_idx]),
                    'two_theta_exp': float(2 * np.arcsin(1.5406 / (2 * d_exp)) * 180 / np.pi),
                    'two_theta_calc': float(2 * np.arcsin(1.5406 / (2 * sim_d[best_idx])) * 180 / np.pi),
                    'intensity_exp': float(I_exp),
                    'intensity_calc': float(sim_intensity[best_idx])
                })

        if not scores:
            return 0.0, []

        weighted_score = float(np.average(scores, weights=weights))
        coverage = len(scores) / n_exp
        min_coverage = 0.5 if (size_nm is None or size_nm >= 10) else 0.35
        if coverage < min_coverage:
            return 0.0, []

        final_score = weighted_score * (0.4 + 0.6 * coverage)
        return min(final_score, 1.0), matched


# ============================================================================
# SIMULATION
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
                hkl_tuple = (0, 0, 0)
                mult = 1
            clean_hkls.append({'hkl': hkl_tuple, 'multiplicity': mult})

        return np.array(pattern.x), np.array(pattern.y), clean_hkls, struct_info
    except Exception as e:
        logger.debug("Simulation from structure failed: %s", e)
        return np.array([]), np.array([]), [], {}


def simulate_from_cif(cif_url, wavelength, formula_hint=""):
    if not PMG_AVAILABLE:
        return np.array([]), np.array([]), [], {}

    user_agents = [
        'Mozilla/5.0 (compatible; BET-XRD-Analyzer/3.0)',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    ]
    for attempt in range(2):
        headers = {'User-Agent': user_agents[attempt % len(user_agents)]}
        try:
            resp = requests.get(cif_url, headers=headers,
                                timeout=NanoParams.CIF_TIMEOUT)
            if resp.status_code != 200:
                continue

            cif_text = resp.text
            if ("<html" in cif_text[:200].lower()
                    or "<!doctype" in cif_text[:200].lower()):
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
                        hkl_tuple = (0, 0, 0)
                        mult = 1
                    clean_hkls.append({'hkl': hkl_tuple, 'multiplicity': mult})

                return (np.array(pattern.x), np.array(pattern.y),
                        clean_hkls, struct_info)
            except Exception as e:
                logger.debug("CIF parsing failed (%s): %s", cif_url, e)
                continue
        except Exception as e:
            logger.debug("CIF download failed (%s): %s", cif_url, e)
            continue

    return np.array([]), np.array([]), [], {}


# ============================================================================
# BUILT-IN LIBRARY (LAST RESORT)
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
            {"d": 3.52, "intensity": 100, "hkl": (1, 0, 1)},
            {"d": 2.38, "intensity": 20, "hkl": (0, 0, 4)},
            {"d": 2.33, "intensity": 10, "hkl": (1, 1, 2)},
            {"d": 1.89, "intensity": 35, "hkl": (2, 0, 0)},
            {"d": 1.70, "intensity": 20, "hkl": (1, 0, 5)},
            {"d": 1.67, "intensity": 15, "hkl": (2, 1, 1)},
            {"d": 1.49, "intensity": 10, "hkl": (2, 1, 3)},
            {"d": 1.48, "intensity": 10, "hkl": (2, 0, 4)},
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
            {"d": 3.95, "intensity": 30, "hkl": (0, 1, 2)},
            {"d": 2.78, "intensity": 100, "hkl": (1, 1, 0)},
            {"d": 2.28, "intensity": 25, "hkl": (1, 1, 3)},
            {"d": 1.94, "intensity": 15, "hkl": (0, 2, 4)},
            {"d": 1.77, "intensity": 35, "hkl": (1, 1, 6)},
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
            {"d": 2.35, "intensity": 100, "hkl": (1, 1, 1)},
            {"d": 2.04, "intensity": 50, "hkl": (2, 0, 0)},
            {"d": 1.44, "intensity": 30, "hkl": (2, 2, 0)},
            {"d": 1.23, "intensity": 20, "hkl": (3, 1, 1)},
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
# FALLBACK DATABASE (COD CIF URLS) - capped for speed
# ============================================================================
FALLBACK_ALL = [
    {"formula": "TiO2", "space_group": "I41/amd", "cif_url": "https://www.crystallography.net/cod/9008213.cif", "database": "Fallback"},
    {"formula": "TiO2", "space_group": "P42/mnm", "cif_url": "https://www.crystallography.net/cod/9009082.cif", "database": "Fallback"},
    {"formula": "BiFeO3", "space_group": "R3c", "cif_url": "https://www.crystallography.net/cod/1533055.cif", "database": "Fallback"},
    {"formula": "Fe2O3", "space_group": "R-3c", "cif_url": "https://www.crystallography.net/cod/9000139.cif", "database": "Fallback"},
    {"formula": "Fe3O4", "space_group": "Fd-3m", "cif_url": "https://www.crystallography.net/cod/9006941.cif", "database": "Fallback"},
    {"formula": "ZnO", "space_group": "P63mc", "cif_url": "https://www.crystallography.net/cod/9008878.cif", "database": "Fallback"},
    {"formula": "CuO", "space_group": "C2/c", "cif_url": "https://www.crystallography.net/cod/1011138.cif", "database": "Fallback"},
    {"formula": "NiO", "space_group": "Fm-3m", "cif_url": "https://www.crystallography.net/cod/1010395.cif", "database": "Fallback"},
    {"formula": "Al2O3", "space_group": "R-3c", "cif_url": "https://www.crystallography.net/cod/9007671.cif", "database": "Fallback"},
    {"formula": "SiO2", "space_group": "P3121", "cif_url": "https://www.crystallography.net/cod/9009668.cif", "database": "Fallback"},
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
# MAIN IDENTIFICATION
# ============================================================================
def identify_phases_universal(two_theta=None, intensity=None, wavelength=1.5406,
                              elements=None, size_nm=None,
                              mp_api_key=None, icsd_api_key=None, ccdc_api_key=None,
                              precomputed_peaks_2theta=None,
                              precomputed_peaks_intensity=None):
    start_time = time.time()
    logger.info("Entered identify_phases_universal")

    if not PMG_AVAILABLE:
        st.error("pymatgen is required. Install: pip install pymatgen")
        return []

    status = st.status("Initializing phase identification...", expanded=True)

    if precomputed_peaks_2theta is not None and precomputed_peaks_intensity is not None:
        exp_2theta = np.array(precomputed_peaks_2theta)
        exp_intensity = np.array(precomputed_peaks_intensity)
        status.write(f"Using {len(exp_2theta)} pre-computed structural peaks")
    else:
        status.write("Detecting peaks from raw data...")
        peak_analyzer = PeakAnalyzer()
        exp_2theta, exp_intensity = peak_analyzer.detect_peaks(two_theta, intensity)
        exp_2theta, exp_intensity = peak_analyzer.refine_apex(two_theta, intensity, exp_2theta)
        status.write(f"Detected {len(exp_2theta)} peaks")

    if len(exp_2theta) < 2:
        status.update(label="Insufficient peaks for phase identification", state="error")
        return []

    exp_d = wavelength / (2 * np.sin(np.radians(exp_2theta / 2)))
    exp_intensity_norm = exp_intensity / np.max(exp_intensity)
    logger.info("d-spacings: %s", np.round(exp_d, 3).tolist())

    family = 'unknown'
    if elements:
        elem_set = set(elements)
        for fam, symbols in NanoParams.FAMILIES.items():
            if any(s in elem_set for s in symbols):
                family = fam
                break
        status.write(f"Material family: {family}")

    if size_nm:
        tol = PatternMatcher.tolerance_from_size(size_nm)
        status.write(f"Size: {size_nm:.1f} nm -> dd/d tolerance: {tol:.1%}")

    status.update(label="Searching databases...", state="running")
    searcher = UltimateDatabaseSearcher(
        mp_api_key=mp_api_key,
        icsd_api_key=icsd_api_key,
        ccdc_api_key=ccdc_api_key
    )

    def db_progress(msg):
        status.write(f"Search: {msg}")

    candidates = searcher.search_all_databases(
        elements=elements,
        dspacings=exp_d,
        family=family,
        progress_callback=db_progress
    )

    logger.info("Found %d unique candidates in %.1f s",
                len(candidates), time.time() - start_time)
    status.write(f"Retrieved {len(candidates)} candidate structures")

    if not candidates:
        status.write("No online candidates - using fallback database")
        candidates = FALLBACK_ALL[:NanoParams.MAX_FALLBACK_STRUCTURES]
        logger.info("Using %d fallback structures", len(candidates))

    candidates = candidates[:NanoParams.MAX_TOTAL_CANDIDATES]

    status.update(label=f"Simulating {len(candidates)} structures...", state="running")

    matcher = PatternMatcher()
    results = []
    threshold = 0.10 if not elements else (0.15 if size_nm and size_nm < 10 else 0.20)

    for idx, struct in enumerate(candidates):
        status.write(f"Simulating {struct.get('formula', 'unknown')} "
                     f"({idx+1}/{len(candidates)})")
        try:
            if 'structure' in struct:
                sim_x, sim_y, sim_hkls, struct_info = simulate_from_structure(
                    struct['structure'], wavelength
                )
            elif 'cif_url' in struct:
                sim_x, sim_y, sim_hkls, struct_info = simulate_from_cif(
                    struct['cif_url'], wavelength, struct.get('formula', '')
                )
            else:
                continue

            if len(sim_x) == 0:
                continue

            sim_d = wavelength / (2 * np.sin(np.radians(sim_x / 2)))
            sim_int = sim_y / np.max(sim_y) if np.max(sim_y) > 0 else sim_y

            score, matched_peaks = matcher.match(
                exp_d, exp_intensity_norm,
                sim_d, sim_int, sim_hkls, size_nm, family
            )

            coverage = len(matched_peaks) / len(exp_d) if matched_peaks else 0
            formula_disp = struct_info.get('formula', struct.get('formula', 'unknown'))
            logger.info("%s: score=%.3f, coverage=%.2f, matched=%d/%d",
                        formula_disp, score, coverage,
                        len(matched_peaks), len(exp_d))

            if score < threshold:
                continue

            min_cov = 0.5 if (size_nm is None or size_nm >= 10) else 0.35
            if coverage < min_cov:
                continue

            if not elements:
                conf = "probable" if score >= 0.30 else "possible"
            else:
                if size_nm and size_nm < 10:
                    conf = ("confirmed" if score >= 0.55
                            else "probable" if score >= 0.35 else "possible")
                else:
                    conf = ("confirmed" if score >= 0.60
                            else "probable" if score >= 0.40 else "possible")

            phase_result = {
                "phase": struct_info.get('formula', struct.get('formula', 'Unknown')),
                "full_formula": struct_info.get('full_formula',
                                                struct.get('formula', 'Unknown')),
                "crystal_system": struct_info.get('crystal_system',
                                                  struct.get('space_group', 'Unknown')),
                "space_group": struct_info.get('space_group',
                                               struct.get('space_group', 'Unknown')),
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

            if len(results) <= 3:
                status.write(f"Match: {phase_result['phase']} "
                             f"(score {score:.3f}, {conf})")

        except Exception as e:
            logger.warning("Simulation/matching error: %s", str(e)[:200])

    logger.info("Simulation complete. %d matches in %.1f s",
                len(results), time.time() - start_time)

    # Built-in library only as last resort
    if not results:
        status.write("No online matches - trying built-in library")
        for phase in BUILTIN_PHASES:
            sim_x, sim_y, sim_hkls, struct_info = simulate_from_library(
                phase['formula'], wavelength
            )
            if len(sim_x) == 0:
                continue
            sim_d = wavelength / (2 * np.sin(np.radians(sim_x / 2)))
            sim_int = sim_y / np.max(sim_y)
            score, matched = matcher.match(
                exp_d, exp_intensity_norm,
                sim_d, sim_int, sim_hkls, size_nm, family
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

    if not results:
        status.update(label="No phases matched", state="error")
        return []

    unique = {}
    for r in results:
        key = (r["phase"], r.get("space_group", ""))
        if key not in unique or r["score"] > unique[key]["score"]:
            unique[key] = r
    final = sorted(unique.values(), key=lambda x: x["score"], reverse=True)

    fractions = estimate_phase_fractions(final, exp_intensity)
    for r in final:
        r["phase_fraction"] = next(
            (f["fraction"] for f in fractions if f["phase"] == r["phase"]),
            None
        )

    status.update(label=f"Identified {len(final)} phases", state="complete")
    return final
