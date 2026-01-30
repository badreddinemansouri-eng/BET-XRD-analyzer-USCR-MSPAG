import numpy as np
from xrd_auto_indexer import auto_index_peaks

def generate_search_space(system):
    if system == "cubic":
        return [{"a": a} for a in np.linspace(3, 10, 300)]

    if system in ["tetragonal", "hexagonal"]:
        return [{"a": a, "c": c}
                for a in np.linspace(3, 10, 60)
                for c in np.linspace(3, 15, 60)]

    if system == "orthorhombic":
        return [{"a": a, "b": b, "c": c}
                for a in np.linspace(3, 10, 20)
                for b in np.linspace(3, 10, 20)
                for c in np.linspace(3, 10, 20)]

    return []


def auto_index(d_spacings):
    results = []

    for system in ["cubic", "tetragonal", "hexagonal", "orthorhombic"]:
        search = generate_search_space(system)
        solution = auto_index_peaks(d_spacings, system, search)

        if solution:
            results.append(solution)

    results.sort(key=lambda x: x["mean_error"])
    return results
