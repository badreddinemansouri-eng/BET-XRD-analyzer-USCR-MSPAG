"""
Automatic indexing driver for powder XRD patterns.
Searches cubic, tetragonal, hexagonal, and orthorhombic lattice
parameter spaces and returns candidate solutions sorted by mean error.

Reference:
    Cullity, B. D., & Stock, S. R. (2001).
    Elements of X-Ray Diffraction, 3rd ed., Prentice Hall.
"""

import logging
from typing import Dict, List

import numpy as np

from xrd_auto_indexer import auto_index_peaks

logger = logging.getLogger(__name__)


def generate_search_space(system: str) -> List[Dict]:
    """Generate a discrete lattice-parameter search space per crystal system."""
    if system == "cubic":
        return [{"a": a} for a in np.linspace(3, 10, 300)]

    if system in ["tetragonal", "hexagonal"]:
        return [
            {"a": a, "c": c}
            for a in np.linspace(3, 10, 60)
            for c in np.linspace(3, 15, 60)
        ]

    if system == "orthorhombic":
        return [
            {"a": a, "b": b, "c": c}
            for a in np.linspace(3, 10, 20)
            for b in np.linspace(3, 10, 20)
            for c in np.linspace(3, 10, 20)
        ]

    logger.warning("Unknown crystal system requested: %s", system)
    return []


def auto_index(d_spacings) -> List[Dict]:
    """
    Try all supported crystal systems and return candidate solutions
    sorted by mean absolute relative error (ascending = better).
    """
    results = []

    for system in ["cubic", "tetragonal", "hexagonal", "orthorhombic"]:
        search = generate_search_space(system)
        if not search:
            continue
        try:
            solution = auto_index_peaks(d_spacings, system, search)
        except Exception as exc:
            logger.warning("Auto-index failed for %s: %s", system, exc)
            continue
        if solution:
            results.append(solution)

    results.sort(key=lambda x: x["mean_error"])
    return results
