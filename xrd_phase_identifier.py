# ============================================================
# XRD PHASE IDENTIFIER — MULTI DATABASE (FREE ONLY)
# ============================================================

import numpy as np
import requests
from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# ------------------------------------------------------------
# DATABASE ENDPOINTS (FREE)
# ------------------------------------------------------------
COD_API = "https://www.crystallography.net/cod/result"

OPTIMADE_ENDPOINTS = [
    "https://api.materialsproject.org/optimade/v1/structures",
    "https://optimade.materialscloud.org/v1/structures",
    "https://oqmd.org/optimade/v1/structures",
]

# ------------------------------------------------------------
# CORE UTILITIES
# ------------------------------------------------------------
def two_theta_from_d(d, wavelength):
    return np.degrees(2 * np.arcsin(np.clip(wavelength / (2 * d), -1, 1)))


def match_score(exp_peaks_2theta, sim_peaks_2theta, wavelength, tol=0.02):
    """
    d-spacing weighted score (NO estimation)
    """
    if len(exp_peaks_2theta) == 0:
        return 0.0

    exp_d = wavelength / (2 * np.sin(np.radians(exp_peaks_2theta / 2)))
    sim_d = wavelength / (2 * np.sin(np.radians(sim_peaks_2theta / 2)))

    score = 0.0
    for d_exp in exp_d:
        rel_err = np.abs(sim_d - d_exp) / d_exp
        best = np.min(rel_err)
        if best < tol:
            score += (1 - best)

    return score / len(exp_d)


# ------------------------------------------------------------
# COD FETCH
# ------------------------------------------------------------
def fetch_cod_cifs(elements, max_results=30):
    query = {
        "format": "json",
        "el": ",".join(elements),
        "maxresults": max_results
    }
    r = requests.get(COD_API, params=query, timeout=30)
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------
# OPTIMADE FETCH (MULTI PROVIDER)
# ------------------------------------------------------------
def fetch_optimade_structures(elements, max_results=30):
    structures = []

    formula_filter = ",".join(elements)
    params = {
        "filter": f'elements HAS ALL "{formula_filter}"',
        "page_limit": max_results
    }

    for endpoint in OPTIMADE_ENDPOINTS:
        try:
            r = requests.get(endpoint, params=params, timeout=30)
            if r.status_code != 200:
                continue
            data = r.json().get("data", [])
            structures.extend(data)
        except Exception:
            continue

    return structures


# ------------------------------------------------------------
# SIMULATE XRD
# ------------------------------------------------------------
def simulate_pattern_from_cif(cif_text, wavelength):
    parser = CifParser.from_string(cif_text)
    structure = parser.get_structures()[0]
    calc = XRDCalculator(wavelength=wavelength)
    pattern = calc.get_pattern(structure, two_theta_range=(5, 80))
    return pattern, structure


# ------------------------------------------------------------
# MAIN IDENTIFICATION ENGINE
# ------------------------------------------------------------
def identify_phases(two_theta, intensity, wavelength, elements):
    """
    FULL phase identification using ONLY FREE databases
    (COD + OPTIMADE providers)

    Parameters
    ----------
    two_theta : np.ndarray
    intensity : np.ndarray
    wavelength : float
    elements : list[str]   ← USER SELECTED ELEMENTS

    Returns
    -------
    list of dict
    """
    import streamlit as st
    st.write("🧪 PHASE DEBUG → Elements:", elements)
    st.write("🧪 PHASE DEBUG → Peaks used:", len(peaks))
    # ------------------------------------------------------------
    # EXPERIMENTAL PEAK SELECTION (NO ESTIMATION)
    # ------------------------------------------------------------
    if len(two_theta) == 0:
        return []

    threshold = 0.30 * np.max(intensity)
    exp_peaks = two_theta[intensity >= threshold]

    results = []

    # ============================================================
    # 1️⃣ COD DATABASE
    # ============================================================
    try:
        cod_entries = fetch_cod_cifs(elements, max_results=40)

        for entry in cod_entries:
            try:
                cif_id = entry["codid"]
                cif_url = f"https://www.crystallography.net/cod/{cif_id}.cif"
                cif_text = requests.get(cif_url, timeout=30).text

                pattern, structure = simulate_pattern_from_cif(
                    cif_text, wavelength
                )

                score = match_score(exp_peaks, pattern.x, wavelength)

                if score < 0.65:
                    continue

                confidence = (
                    "confirmed" if score >= 0.85
                    else "probable"
                )

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

            except Exception:
                continue

    except Exception:
        pass

    # ============================================================
    # 2️⃣ OPTIMADE (Materials Project, OQMD, Materials Cloud)
    # ============================================================
    try:
        optimade_structures = fetch_optimade_structures(elements, max_results=40)

        for entry in optimade_structures:
            try:
                attributes = entry["attributes"]
                lattice = attributes["lattice_vectors"]
                sites = attributes["sites"]

                # Build CIF manually
                cif_lines = [
                    "data_generated",
                    "_symmetry_space_group_name_H-M   'P 1'",
                    "_cell_length_a   {}".format(np.linalg.norm(lattice[0])),
                    "_cell_length_b   {}".format(np.linalg.norm(lattice[1])),
                    "_cell_length_c   {}".format(np.linalg.norm(lattice[2])),
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
                    for el in site["species"]:
                        cif_lines.append(
                            f"{el['chemical_symbols'][0]} "
                            f"{site['fractional_coordinates'][0]} "
                            f"{site['fractional_coordinates'][1]} "
                            f"{site['fractional_coordinates'][2]}"
                        )

                cif_text = "\n".join(cif_lines)

                pattern, structure = simulate_pattern_from_cif(
                    cif_text, wavelength
                )

                score = match_score(exp_peaks, pattern.x, wavelength)

                if score < 0.65:
                    continue

                confidence = (
                    "confirmed" if score >= 0.85
                    else "probable"
                )

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

            except Exception:
                continue

    except Exception:
        pass

    # ============================================================
    # FINAL FILTER & SORT
    # ============================================================
    # Remove duplicates (same formula + space group)
    unique = {}
    for r in results:
        key = (r["phase"], r["space_group"])
        if key not in unique or r["score"] > unique[key]["score"]:
            unique[key] = r

    final_results = sorted(
        unique.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return final_results
    


