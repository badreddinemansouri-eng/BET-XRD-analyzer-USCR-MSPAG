import numpy as np
from itertools import product

# ==========================================================
# AUTO-INDEXING ENGINE (POWDER XRD)
# ==========================================================

CRYSTAL_SYSTEMS = {
    "cubic": ["a"],
    "tetragonal": ["a", "c"],
    "hexagonal": ["a", "c"],
    "orthorhombic": ["a", "b", "c"]
}

def d_spacing_from_hkl(hkl, lattice, system):
    h, k, l = hkl

    if system == "cubic":
        a = lattice["a"]
        return a / np.sqrt(h*h + k*k + l*l)

    if system == "tetragonal":
        a, c = lattice["a"], lattice["c"]
        return 1 / np.sqrt((h*h + k*k)/(a*a) + (l*l)/(c*c))

    if system == "hexagonal":
        a, c = lattice["a"], lattice["c"]
        return 1 / np.sqrt(
            (4/3)*(h*h + h*k + k*k)/(a*a) + (l*l)/(c*c)
        )

    if system == "orthorhombic":
        a, b, c = lattice["a"], lattice["b"], lattice["c"]
        return 1 / np.sqrt(
            (h*h)/(a*a) + (k*k)/(b*b) + (l*l)/(c*c)
        )

    return None


def generate_hkl(max_index=4):
    hkls = []
    for h, k, l in product(range(0, max_index+1), repeat=3):
        if h == k == l == 0:
            continue
        hkls.append((h, k, l))
    return hkls


def auto_index_peaks(d_exp, system, search_range):
    """
    d_exp : list of experimental d-spacings (strongest peaks)
    """
    best_solution = None
    best_error = np.inf

    hkls = generate_hkl()

    for lattice_params in search_range:
        lattice = lattice_params.copy()
        errors = []

        for d in d_exp:
            d_calc_list = [
                d_spacing_from_hkl(hkl, lattice, system)
                for hkl in hkls
            ]
            d_calc_list = [x for x in d_calc_list if x]

            if not d_calc_list:
                continue

            error = min(abs(d - dc)/d for dc in d_calc_list)
            errors.append(error)

        if len(errors) >= 3:
            mean_error = np.mean(errors)
            if mean_error < best_error:
                best_error = mean_error
                best_solution = {
                    "system": system,
                    "lattice": lattice,
                    "mean_error": mean_error
                }

    return best_solution
