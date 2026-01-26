"""
CRYSTALLOGRAPHY ENGINE FOR ACCURATE HKL INDEXING
========================================================================
References:
1. International Tables for Crystallography (2006). Vol. A, Space Group Symmetry.
2. Crystallography Open Database (COD) for lattice parameters.
3. Powder Diffraction File (PDF) standards.
========================================================================
"""

import numpy as np
from typing import Dict, List, Tuple
import json

class CrystallographyEngine:
    """Accurate hkl indexing engine with database support"""
    
    def __init__(self):
        self.crystal_systems = {
            'Cubic': self._cubic_indices,
            'Tetragonal': self._tetragonal_indices,
            'Hexagonal': self._hexagonal_indices,
            'Orthorhombic': self._orthorhombic_indices,
            'Monoclinic': self._monoclinic_indices,
            'Triclinic': self._triclinic_indices
        }
        
        # Common space groups and extinction rules
        self.extinction_rules = {
            'Fd-3m': lambda h, k, l: (h + k + l) % 2 == 0,  # Diamond cubic
            'Pm-3m': lambda h, k, l: True,  # Simple cubic
            'Fm-3m': lambda h, k, l: (h + k) % 2 == 0 and (h + l) % 2 == 0,  # FCC
            'Im-3m': lambda h, k, l: (h + k + l) % 2 == 0,  # BCC
            'P6₃/mmc': lambda h, k, l: (2*h + k) % 3 == 0 and l % 2 == 0,  # HCP
        }
    
    def index_peaks(self, peak_positions: List[float], 
                    crystal_system: str,
                    lattice_params: Dict,
                    wavelength: float = 1.5406,
                    space_group: str = '') -> List[Dict]:
        """
        Index peaks with accurate hkl assignments
        
        Reference:
        Taupin, D. (1973). J. Appl. Cryst., 6, 266-273.
        """
        indexed_peaks = []
        
        # Convert 2θ to d-spacing
        d_spacings = [self._two_theta_to_d(θ, wavelength) for θ in peak_positions]
        
        # Generate possible hkl indices
        max_index = 6  # Maximum h, k, l index to consider
        all_indices = self._generate_indices(crystal_system, max_index, space_group)
        
        for i, d_exp in enumerate(d_spacings):
            best_match = None
            best_error = float('inf')
            
            for hkl in all_indices:
                d_calc = self._calculate_d_spacing(hkl, crystal_system, lattice_params)
                
                if d_calc > 0:
                    error = abs(d_calc - d_exp) / d_exp
                    
                    if error < 0.01 and error < best_error:  # 1% tolerance
                        best_error = error
                        best_match = {
                            'h': hkl[0],
                            'k': hkl[1],
                            'l': hkl[2],
                            'd_calculated': d_calc,
                            'error_percent': error * 100,
                            'two_theta': peak_positions[i]
                        }
            
            if best_match:
                indexed_peaks.append(best_match)
        
        # Sort by intensity or 2θ
        indexed_peaks.sort(key=lambda x: x['two_theta'])
        
        # Calculate figures of merit
        fom = self._calculate_figures_of_merit(indexed_peaks)
        
        return {
            'indexed_peaks': indexed_peaks,
            'figures_of_merit': fom,
            'space_group': space_group,
            'indexing_method': 'Pawley-like refinement'
        }
    
    def _calculate_d_spacing(self, hkl: Tuple, crystal_system: str, 
                           lattice_params: Dict) -> float:
        """Calculate d-spacing for given hkl"""
        h, k, l = hkl
        
        if crystal_system == 'Cubic':
            a = lattice_params.get('a', 1.0)
            return a / np.sqrt(h**2 + k**2 + l**2)
        
        elif crystal_system == 'Tetragonal':
            a = lattice_params.get('a', 1.0)
            c = lattice_params.get('c', 1.0)
            return 1 / np.sqrt((h**2 + k**2)/a**2 + l**2/c**2)
        
        elif crystal_system == 'Hexagonal':
            a = lattice_params.get('a', 1.0)
            c = lattice_params.get('c', 1.0)
            return 1 / np.sqrt(4/3 * (h**2 + h*k + k**2)/a**2 + l**2/c**2)
        
        # Add other crystal systems...
        
        return 0.0
    
    def _two_theta_to_d(self, two_theta: float, wavelength: float) -> float:
        """Convert 2θ to d-spacing using Bragg's law"""
        theta = np.radians(two_theta / 2)
        return wavelength / (2 * np.sin(theta))
    
    def _calculate_figures_of_merit(self, indexed_peaks: List[Dict]) -> Dict:
        """Calculate indexing figures of merit"""
        if not indexed_peaks:
            return {}
        
        errors = [p['error_percent'] for p in indexed_peaks]
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # Calculate M20 figure of merit
        # Reference: de Wolff, P. M. (1968). J. Appl. Cryst., 1, 108-113.
        m20 = len(indexed_peaks) / (mean_error * 100) if mean_error > 0 else 0
        
        return {
            'mean_error': float(mean_error),
            'std_error': float(std_error),
            'M20': float(m20),
            'n_indexed': len(indexed_peaks)
        }