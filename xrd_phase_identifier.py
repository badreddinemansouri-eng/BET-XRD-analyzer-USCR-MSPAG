import numpy as np
import requests
from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator

COD_API = "https://www.crystallography.net/cod/result"


def fetch_cod_cifs(elements, max_results=10):
    query = {
        "format": "json",
        "el": ",".join(elements),
        "maxresults": max_results
    }
    r = requests.get(COD_API, params=query, timeout=20)
    r.raise_for_status()
    return r.json()


def simulate_pattern(cif_text, wavelength):
    parser = CifParser.from_string(cif_text)
    structure = parser.get_structures()[0]
    calc = XRDCalculator(wavelength=wavelength)
    pattern = calc.get_pattern(structure, two_theta_range=(5, 80))
    return pattern, structure


def match_score(exp_peaks_2theta, sim_peaks_2theta, wavelength, tol=0.02):
    """
    d-spacing weighted match score (nano-safe)
    """
    if len(exp_peaks_2theta) == 0:
        return 0.0

    exp_d = wavelength / (2 * np.sin(np.radians(exp_peaks_2theta / 2)))
    sim_d = wavelength / (2 * np.sin(np.radians(sim_peaks_2theta / 2)))

    score = 0.0

    for d_exp in exp_d:
        diffs = np.abs(sim_d - d_exp) / d_exp
        best = np.min(diffs)
        if best < tol:
            score += (1 - best)

    return score / len(exp_d)


def identify_phases(two_theta, intensity, wavelength):
    """
    Identify crystalline phases using COD + pymatgen.
    Returns CONFIRMED and PROBABLE phases with confidence levels.
    """

    results = []

    # USE PEAK POSITIONS, NOT RAW INTENSITY
    mask = intensity > 0.3 * np.max(intensity)
    exp_peaks = np.array(two_theta)[mask]

    try:
        cif_entries = fetch_cod_cifs(elements_guess)
    except Exception:
        return []

    for entry in cif_entries:
        try:
            cif_id = entry["codid"]
            cif_url = f"https://www.crystallography.net/cod/{cif_id}.cif"
            cif_text = requests.get(cif_url, timeout=20).text

            pattern, structure = simulate_pattern(cif_text, wavelength)

            score = match_score(
                exp_peaks,
                pattern.x,
                wavelength=wavelength
            )

            if score >= 0.85:
                confidence = "confirmed"
            elif score >= 0.65:
                confidence = "probable"
            else:
                continue  # rejected

            results.append({
                "phase": structure.composition.reduced_formula,
                "crystal_system": structure.get_crystal_system(),
                "space_group": structure.get_space_group_info()[0],
                "lattice": structure.lattice.as_dict(),
                "hkls": pattern.hkls,
                "score": round(score, 4),
                "confidence_level": confidence,
                "structure": structure
            })

        except Exception:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results

