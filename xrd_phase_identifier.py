"""
XRD PHASE IDENTIFIER - MULTI DATABASE (FREE ONLY)

Searches COD and OPTIMADE-compliant providers for candidate phases
matching an experimental peak list. Uses size-dependent d-spacing
tolerance for nanocrystalline materials.

References:
    1. Grazulis, S. et al. (2012). Nucleic Acids Res., 40, D420-D427 (COD).
    2. Andersen, C. W. et al. (2021). OPTIMADE: an open standard.
    3. Scherrer, P. (1918). Nachr. Ges. Wiss. Gottingen, 2, 98.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import requests

from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DATABASE ENDPOINTS (FREE)
# ---------------------------------------------------------------------------
COD_API = "https://www.crystallography.net/cod/result"

OPTIMADE_ENDPOINTS = [
    "https://api.materialsproject.org/optimade/v1/structures",
    "https://optimade.materialscloud.org/v1/structures",
    "https://oqmd.org/optimade/v1/structures",
]


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------
def two_theta_from_d(d, wavelength):
    return np.degrees(2 * np.arcsin(np.clip(wavelength / (2 * d), -1, 1)))


def match_score(exp_peaks_2theta, sim_peaks_2theta, wavelength, tol=0.02):
    """
    d-spacing weighted match score.

    Both experimental and simulated peak lists are converted to d-spacings.
    For each experimental peak, the closest simulated d-spacing is found
    and its relative error computed. Peaks within `tol` contribute.
    """
    if len(exp_peaks_2theta) == 0:
        return 0.0

    exp_d = wavelength / (2 * np.sin(np.radians(exp_peaks_2theta / 2)))
    sim_d = wavelength / (2 * np.sin(np.radians(sim_peaks_2theta / 2)))

    score = 0.0
    for d_exp in exp_d:
        rel_err = np.abs(sim_d - d_exp) / d_exp
        best = float(np.min(rel_err))
        if best < tol:
            score += (1.0 - best)

    return score / len(exp_d)


def match_score_with_tolerance(exp_peaks_2theta, sim_peaks_2theta,
                               wavelength, size_nm=None):
    """
    Size-aware tolerance. Scherrer broadening maps to d-spacing tolerance:
        < 5 nm : 10%   (ultra-nanocrystalline)
        5-10 nm: 6%    (nanocrystalline)
        > 10 nm: 3%    (sub-micron to micron)
    """
    if size_nm is None:
        tol = 0.02
    elif size_nm < 5:
        tol = 0.10
    elif size_nm < 10:
        tol = 0.06
    else:
        tol = 0.03

    return match_score(exp_peaks_2theta, sim_peaks_2theta, wavelength, tol=tol)


# ---------------------------------------------------------------------------
# COD SEARCH
# ---------------------------------------------------------------------------
def fetch_cod_cifs(elements, max_results=30):
    """
    Query COD for structures containing the requested elements.
    If oxygen is present, filters to oxide-like formulas.
    """
    query = {
        "format": "json",
        "el": ",".join(elements),
        "maxresults": max_results,
    }

    if "O" in elements:
        query["formula"] = "*O*"
        query["status"] = "published"

    try:
        r = requests.get(COD_API, params=query, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning("COD fetch error: %s", exc)
        return []


# ---------------------------------------------------------------------------
# OPTIMADE SEARCH
# ---------------------------------------------------------------------------
def fetch_optimade_structures(elements, max_results=30):
    """
    Query multiple OPTIMADE endpoints with a valid filter string.
    """
    structures = []

    filters = " AND ".join([f'elements HAS "{el}"' for el in elements])
    params = {
        "filter": filters,
        "page_limit": max_results,
    }

    for endpoint in OPTIMADE_ENDPOINTS:
        try:
            r = requests.get(endpoint, params=params, timeout=30)
            if r.status_code != 200:
                logger.info("OPTIMADE %s returned status %d", endpoint, r.status_code)
                continue
            data = r.json().get("data", [])
            structures.extend(data)
        except Exception as exc:
            logger.warning("OPTIMADE fetch error from %s: %s", endpoint, exc)
            continue

    return structures


# ---------------------------------------------------------------------------
# PATTERN SIMULATION
# ---------------------------------------------------------------------------
def simulate_pattern_from_cif(cif_text, wavelength):
    parser = CifParser.from_string(cif_text)
    structure = parser.get_structures()[0]
    calc = XRDCalculator(wavelength=wavelength)
    pattern = calc.get_pattern(structure, two_theta_range=(5, 80))
    return pattern, structure


def _cif_from_optimade(attributes):
    """Build a minimal P1 CIF string from OPTIMADE attributes."""
    lattice = attributes["lattice_vectors"]
    sites = attributes["sites"]

    cif_lines = [
        "data_generated",
        "_symmetry_space_group_name_H-M   'P 1'",
        f"_cell_length_a   {np.linalg.norm(lattice[0])}",
        f"_cell_length_b   {np.linalg.norm(lattice[1])}",
        f"_cell_length_c   {np.linalg.norm(lattice[2])}",
        "_cell_angle_alpha 90",
        "_cell_angle_beta  90",
        "_cell_angle_gamma 90",
        "loop_",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]

    for site in sites:
        for species in site["species"]:
            symbol = species["chemical_symbols"][0]
            coords = site["fractional_coordinates"]
            cif_lines.append(f"{symbol} {coords[0]} {coords[1]} {coords[2]}")

    return "\n".join(cif_lines)


# ---------------------------------------------------------------------------
# EXPERIMENTAL PEAK SELECTION
# ---------------------------------------------------------------------------
def _select_experimental_peaks(two_theta, intensity, threshold_fraction=0.30):
    """
    Select peaks above threshold and refine to local maxima within a
    small window. Enforce angular uniqueness to protect the strongest peak.
    """
    if len(two_theta) == 0:
        return np.array([])

    threshold = threshold_fraction * np.max(intensity)
    candidate_idx = np.where(intensity >= threshold)[0]

    refined = []
    for idx in candidate_idx:
        left = max(0, idx - 5)
        right = min(len(intensity), idx + 6)
        local_idx = left + int(np.argmax(intensity[left:right]))
        refined.append(two_theta[local_idx])

    refined = np.unique(refined)

    # Sort by descending intensity
    refined = np.array(sorted(
        refined,
        key=lambda t: intensity[int(np.argmin(np.abs(two_theta - t)))],
        reverse=True
    ))

    unique_peaks = []
    for t in refined:
        if all(abs(t - u) >= 1.0 for u in unique_peaks):
            unique_peaks.append(t)

    return np.array(unique_peaks)


# ---------------------------------------------------------------------------
# MAIN IDENTIFICATION
# ---------------------------------------------------------------------------
def identify_phases(two_theta, intensity, wavelength, elements, size_nm=None):
    """
    Identify crystalline phases using COD and OPTIMADE.

    Parameters
    ----------
    two_theta : np.ndarray
    intensity : np.ndarray
    wavelength : float
    elements : list[str]
    size_nm : float, optional
        Crystallite size for tolerance adjustment.

    Returns
    -------
    list of dict
    """
    import streamlit as st

    if len(two_theta) == 0:
        return []

    exp_peaks = _select_experimental_peaks(two_theta, intensity)

    logger.info("Phase identification: %d experimental peaks, elements=%s, size=%s",
                len(exp_peaks), elements, size_nm)

    results = []

    # ------------------------------------------------------------------
    # 1. COD
    # ------------------------------------------------------------------
    try:
        cod_entries = fetch_cod_cifs(elements, max_results=40)
        logger.info("COD returned %d entries", len(cod_entries))

        for entry in cod_entries:
            try:
                cif_id = entry["codid"]
                cif_url = f"https://www.crystallography.net/cod/{cif_id}.cif"
                cif_text = requests.get(cif_url, timeout=30).text

                pattern, structure = simulate_pattern_from_cif(cif_text, wavelength)

                if size_nm:
                    score = match_score_with_tolerance(
                        exp_peaks, pattern.x, wavelength, size_nm
                    )
                else:
                    score = match_score(exp_peaks, pattern.x, wavelength)

                threshold_score = 0.55 if (size_nm and size_nm < 10) else 0.65
                if score < threshold_score:
                    continue

                confidence = "confirmed" if score >= 0.85 else "probable"

                results.append({
                    "phase": structure.composition.reduced_formula,
                    "crystal_system": structure.get_crystal_system(),
                    "space_group": structure.get_space_group_info()[0],
                    "lattice": structure.lattice.as_dict(),
                    "hkls": pattern.hkls,
                    "score": round(score, 3),
                    "confidence_level": confidence,
                    "database": "COD",
                    "structure": structure,
                })

            except Exception as exc:
                logger.debug("COD entry skipped: %s", exc)
                continue

    except Exception as exc:
        logger.warning("COD search failed: %s", exc)

    # ------------------------------------------------------------------
    # 2. OPTIMADE
    # ------------------------------------------------------------------
    try:
        optimade_structures = fetch_optimade_structures(elements, max_results=40)
        logger.info("OPTIMADE returned %d structures", len(optimade_structures))

        for entry in optimade_structures:
            try:
                attributes = entry["attributes"]
                cif_text = _cif_from_optimade(attributes)

                pattern, structure = simulate_pattern_from_cif(cif_text, wavelength)

                if size_nm:
                    score = match_score_with_tolerance(
                        exp_peaks, pattern.x, wavelength, size_nm
                    )
                else:
                    score = match_score(exp_peaks, pattern.x, wavelength)

                threshold_score = 0.55 if (size_nm and size_nm < 10) else 0.65
                if score < threshold_score:
                    continue

                confidence = "confirmed" if score >= 0.85 else "probable"

                results.append({
                    "phase": structure.composition.reduced_formula,
                    "crystal_system": structure.get_crystal_system(),
                    "space_group": structure.get_space_group_info()[0],
                    "lattice": structure.lattice.as_dict(),
                    "hkls": pattern.hkls,
                    "score": round(score, 3),
                    "confidence_level": confidence,
                    "database": "OPTIMADE",
                    "structure": structure,
                })

            except Exception as exc:
                logger.debug("OPTIMADE entry skipped: %s", exc)
                continue

    except Exception as exc:
        logger.warning("OPTIMADE search failed: %s", exc)

    # ------------------------------------------------------------------
    # Deduplicate and sort
    # ------------------------------------------------------------------
    unique = {}
    for r in results:
        key = (r["phase"], r["space_group"])
        if key not in unique or r["score"] > unique[key]["score"]:
            unique[key] = r

    final_results = sorted(unique.values(), key=lambda x: x["score"], reverse=True)

    logger.info("Phase identification: %d unique phases", len(final_results))
    return final_results
