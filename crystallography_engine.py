"""
CRYSTALLOGRAPHY ENGINE FOR HKL INDEXING

Provides:
    - Peak indexing for cubic, tetragonal, hexagonal, orthorhombic
    - Extinction rules for common space groups
    - Figures of merit (M20, mean error)

References:
    1. International Tables for Crystallography (2006), Vol. A.
    2. de Wolff, P. M. (1968). J. Appl. Cryst., 1, 108-113.
    3. Taupin, D. (1973). J. Appl. Cryst., 6, 266-273.
"""

import logging
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CrystallographyEngine:
    """
    Accurate hkl indexing engine with extinction rules.
    """

    def __init__(self):
        # Space-group extinction rules (h, k, l -> bool: reflection allowed)
        self.extinction_rules = {
            'Fd-3m': lambda h, k, l: (h + k + l) % 2 == 0,
            'Pm-3m': lambda h, k, l: True,
            'Fm-3m': lambda h, k, l: (
                (h + k) % 2 == 0 and (h + l) % 2 == 0 and (k + l) % 2 == 0
            ),
            'Im-3m': lambda h, k, l: (h + k + l) % 2 == 0,
            'P63/mmc': lambda h, k, l: ((2 * h + k) % 3 == 0) and (l % 2 == 0),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def index_peaks(self, peak_positions: List[float],
                    crystal_system: str,
                    lattice_params: Dict,
                    wavelength: float = 1.5406,
                    space_group: str = '') -> Dict:
        """
        Index a list of 2theta peak positions.

        Returns
        -------
        dict
            {'indexed_peaks': [...], 'figures_of_merit': {...},
             'space_group': str, 'indexing_method': str}
        """
        d_spacings = [self._two_theta_to_d(t, wavelength) for t in peak_positions]

        max_index = 6
        all_indices = self._generate_indices(crystal_system, max_index, space_group)

        indexed_peaks = []
        for i, d_exp in enumerate(d_spacings):
            best_match = None
            best_error = float('inf')

            for hkl in all_indices:
                d_calc = self._calculate_d_spacing(hkl, crystal_system, lattice_params)
                if d_calc is None or d_calc <= 0:
                    continue
                error = abs(d_calc - d_exp) / d_exp
                if error < 0.01 and error < best_error:
                    best_error = error
                    best_match = {
                        'h': hkl[0],
                        'k': hkl[1],
                        'l': hkl[2],
                        'd_calculated': float(d_calc),
                        'd_experimental': float(d_exp),
                        'error_percent': float(error * 100.0),
                        'two_theta': float(peak_positions[i]),
                    }

            if best_match:
                indexed_peaks.append(best_match)

        indexed_peaks.sort(key=lambda x: x['two_theta'])
        fom = self._calculate_figures_of_merit(indexed_peaks)

        return {
            'indexed_peaks': indexed_peaks,
            'figures_of_merit': fom,
            'space_group': space_group,
            'indexing_method': 'Grid search with extinction rules (Taupin 1973)',
        }

    # ------------------------------------------------------------------
    # d-spacing
    # ------------------------------------------------------------------
    def _calculate_d_spacing(self, hkl: Tuple[int, int, int],
                             crystal_system: str,
                             lattice_params: Dict) -> Optional[float]:
        h, k, l = hkl

        if crystal_system == 'Cubic':
            a = lattice_params.get('a')
            if not a:
                return None
            denom = h ** 2 + k ** 2 + l ** 2
            return a / np.sqrt(denom) if denom > 0 else None

        if crystal_system == 'Tetragonal':
            a = lattice_params.get('a')
            c = lattice_params.get('c')
            if not a or not c:
                return None
            inv_d2 = (h ** 2 + k ** 2) / a ** 2 + l ** 2 / c ** 2
            return 1.0 / np.sqrt(inv_d2) if inv_d2 > 0 else None

        if crystal_system == 'Hexagonal':
            a = lattice_params.get('a')
            c = lattice_params.get('c')
            if not a or not c:
                return None
            inv_d2 = (4.0 / 3.0) * (h ** 2 + h * k + k ** 2) / a ** 2 + l ** 2 / c ** 2
            return 1.0 / np.sqrt(inv_d2) if inv_d2 > 0 else None

        if crystal_system == 'Orthorhombic':
            a = lattice_params.get('a')
            b = lattice_params.get('b')
            c = lattice_params.get('c')
            if not a or not b or not c:
                return None
            inv_d2 = h ** 2 / a ** 2 + k ** 2 / b ** 2 + l ** 2 / c ** 2
            return 1.0 / np.sqrt(inv_d2) if inv_d2 > 0 else None

        logger.warning("Crystal system %s not supported in d-spacing routine",
                       crystal_system)
        return None

    # ------------------------------------------------------------------
    # Index generation
    # ------------------------------------------------------------------
    def _generate_indices(self, crystal_system: str, max_index: int,
                          space_group: str) -> List[Tuple[int, int, int]]:
        """
        Generate allowed (h, k, l) triples for a given crystal system and
        space group. Extinction rules are applied if the space group is
        recognized.
        """
        indices = []
        rule = self.extinction_rules.get(space_group)

        for h, k, l in product(range(0, max_index + 1), repeat=3):
            if h == 0 and k == 0 and l == 0:
                continue
            if rule is not None and not rule(h, k, l):
                continue
            indices.append((h, k, l))

        return indices

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _two_theta_to_d(self, two_theta: float, wavelength: float) -> float:
        theta = np.radians(two_theta / 2.0)
        return wavelength / (2.0 * np.sin(theta))

    def _calculate_figures_of_merit(self, indexed_peaks: List[Dict]) -> Dict:
        if not indexed_peaks:
            return {}

        errors = [p['error_percent'] for p in indexed_peaks]
        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors))

        # M20 (de Wolff 1968): M20 = Q20 / (2 * mean_error * N_possible)
        # Simplified: use number of indexed peaks / (mean_error * 100)
        m20 = len(indexed_peaks) / (mean_error * 100.0) if mean_error > 0 else 0.0

        return {
            'mean_error': mean_error,
            'std_error': std_error,
            'M20': float(m20),
            'n_indexed': len(indexed_peaks),
        }
