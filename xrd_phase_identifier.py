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

def match_score(exp_peaks, sim_peaks, tol=0.15):
    matches = 0
    for t in exp_peaks:
        if np.any(np.abs(sim_peaks - t) < tol):
            matches += 1
    return matches / len(exp_peaks)

def identify_phases(two_theta, intensity, wavelength):
    exp_peaks = two_theta[intensity > 0.3 * np.max(intensity)]
    elements_guess = []  # optional: extract from sample metadata later

    results = []
    cif_entries = fetch_cod_cifs(elements_guess or ["Ti", "O", "Fe", "Bi"])

    for entry in cif_entries:
        cif_id = entry["codid"]
        cif_url = f"https://www.crystallography.net/cod/{cif_id}.cif"
        cif_text = requests.get(cif_url, timeout=20).text

        try:
            pattern, structure = simulate_pattern(cif_text, wavelength)
            score = match_score(exp_peaks, pattern.x)

            if score > 0.85:
                results.append({
                    "phase": structure.composition.reduced_formula,
                    "crystal_system": structure.get_crystal_system(),
                    "space_group": structure.get_space_group_info()[0],
                    "lattice": structure.lattice.as_dict(),
                    "hkls": pattern.hkls,
                    "score": score,
                    "structure": structure
                })
        except:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results