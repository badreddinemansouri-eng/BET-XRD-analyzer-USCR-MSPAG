"""
Auto-indexing engine for powder XRD patterns.

Given a list of experimental d-spacings and a crystal-system hypothesis,
this module searches a lattice-parameter grid and returns the best-scoring
assignment. The error metric is the mean relative d-spacing residual.

References:
    1. Cullity & Stock, Elements of X-Ray Diffraction, 3rd ed. (2001).
    2. International Tables for Crystallography, Vol. A (2006).
"""

import logging
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CRYSTAL_SYSTEMS = {
    "cubic": ["a"],
    "tetragonal": ["a", "c"],
    "hexagonal": ["a", "c"],
    "orthorhombic": ["a", "b", "c"],
}


def d_spacing_from_hkl(hkl: Tuple[int, int, int],
                       lattice: Dict, system: str) -> Optional[float]:
    """Compute d-spacing for a given (h, k, l) and lattice parameters."""
    h, k, l = hkl

    if system == "cubic":
        a = lattice["a"]
        denom = h * h + k * k + l * l
        return a / np.sqrt(denom) if denom > 0 else None

    if system == "tetragonal":
        a, c = lattice["a"], lattice["c"]
        inv_d2 = (h * h + k * k) / (a * a) + (l * l) / (c * c)
        return 1.0 / np.sqrt(inv_d2) if inv_d2 > 0 else None

    if system == "hexagonal":
        a, c = lattice["a"], lattice["c"]
        inv_d2 = (4.0 / 3.0) * (h * h + h * k + k * k) / (a * a) + (l * l) / (c * c)
        return 1.0 / np.sqrt(inv_d2) if inv_d2 > 0 else None

    if system == "orthorhombic":
        a, b, c = lattice["a"], lattice["b"], lattice["c"]
        inv_d2 = (h * h) / (a * a) + (k * k) / (b * b) + (l * l) / (c * c)
        return 1.0 / np.sqrt(inv_d2) if inv_d2 > 0 else None

    return None


def generate_hkl(max_index: int = 4) -> List[Tuple[int, int, int]]:
    """Enumerate all (h, k, l) with indices in [0, max_index], excluding (0,0,0)."""
    hkls = []
    for h, k, l in product(range(0, max_index + 1), repeat=3):
        if h == 0 and k == 0 and l == 0:
            continue
        hkls.append((h, k, l))
    return hkls


def auto_index_peaks(d_exp, system: str,
                     search_range: List[Dict]) -> Optional[Dict]:
    """
    Search the lattice-parameter grid for the best assignment.

    Parameters
    ----------
    d_exp : list of float
        Experimental d-spacings of the strongest observed peaks.
    system : str
        One of 'cubic', 'tetragonal', 'hexagonal', 'orthorhombic'.
    search_range : list of dict
        Each dict contains the lattice parameters for one trial.

    Returns
    -------
    dict or None
        Best candidate with keys: system, lattice, mean_error.
    """
    if not d_exp or not search_range:
        return None

    best_solution = None
    best_error = float("inf")
    hkls = generate_hkl()

    for lattice_params in search_range:
        lattice = lattice_params.copy()
        errors = []

        for d in d_exp:
            d_calc_list = [
                d_spacing_from_hkl(hkl, lattice, system) for hkl in hkls
            ]
            d_calc_list = [x for x in d_calc_list if x is not None and x > 0]
            if not d_calc_list:
                continue
            error = min(abs(d - dc) / d for dc in d_calc_list)
            errors.append(error)

        if len(errors) >= 3:
            mean_error = float(np.mean(errors))
            if mean_error < best_error:
                best_error = mean_error
                best_solution = {
                    "system": system,
                    "lattice": lattice,
                    "mean_error": mean_error,
                }

    return best_solution
