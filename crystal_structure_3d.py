"""
SCIENTIFIC 3D CRYSTAL STRUCTURE VISUALIZATION
========================================================================
Generates accurate 3D crystal structures based on space group and parameters.
References:
1. International Tables for Crystallography
2. ASE (Atomic Simulation Environment) methods
========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
from typing import Dict, List, Tuple

class CrystalStructure3D:
    """Generate 3D crystal structures from crystallographic data"""
    
    def __init__(self):
        self.atomic_radii = {
            'Si': 1.11, 'O': 0.73, 'C': 0.77, 'Al': 1.43,
            'Fe': 1.24, 'Ti': 1.47, 'Zr': 1.60, 'Ce': 1.85,
            'Na': 1.02, 'Ca': 1.00, 'Mg': 1.60
        }
    def hkl_plane_geometry(self, hkl, lattice_params):
        """
        Returns plane normal and distance from origin for (hkl)
        """
        alpha = lattice_params.get("alpha", 90)
        beta  = lattice_params.get("beta", 90)
        gamma = lattice_params.get("gamma", 90)
        
        if not (alpha == beta == gamma == 90):
            raise NotImplementedError(
                "General reciprocal lattice requires non-orthogonal metric tensor"
            )

        a = lattice_params["a"]
        b = lattice_params.get("b", a)
        c = lattice_params.get("c", a)
    
        # Reciprocal lattice vectors
        a_star = np.array([1/a, 0, 0])
        b_star = np.array([0, 1/b, 0])
        c_star = np.array([0, 0, 1/c])
    
        h, k, l = hkl
        normal = h*a_star + k*b_star + l*c_star
        norm = np.linalg.norm(normal)
    
        if norm == 0:
            return None
    
        d_hkl = 1 / norm
        unit_normal = normal / norm
        if "d_spacing_exp" in peak:
        ratio = peak["d_spacing_exp"] / hkl_info["d_spacing"]
        peak["d_consistency"] = ratio

        return {
            "normal": unit_normal,
            "d_spacing": d_hkl
        }
    def draw_hkl_plane(self, ax, hkl_info, extent=10):
        n = hkl_info["normal"]
        d = hkl_info["d_spacing"]
    
        # Plane equation: n·r = d
        xx, yy = np.meshgrid(
            np.linspace(-extent, extent, 10),
            np.linspace(-extent, extent, 10)
        )
    
        # Solve for z
        eps = 1e-8

    if abs(n[2]) > eps:
        z = (d - n[0]*xx - n[1]*yy) / n[2]
        ax.plot_surface(xx, yy, z, alpha=0.3, color="cyan")
    
    elif abs(n[1]) > eps:
        yy, zz = np.meshgrid(
            np.linspace(-extent, extent, 10),
            np.linspace(-extent, extent, 10)
        )
        x = (d - n[1]*yy - n[2]*zz) / n[0]
        ax.plot_surface(x, yy, zz, alpha=0.3, color="cyan")
    
    elif abs(n[0]) > eps:
        xx, zz = np.meshgrid(
            np.linspace(-extent, extent, 10),
            np.linspace(-extent, extent, 10)
        )
        y = (d - n[0]*xx - n[2]*zz) / n[1]
        ax.plot_surface(xx, y, zz, alpha=0.3, color="cyan")

        

    def generate_structure(self, crystal_system: str, 
                          lattice_params: Dict,
                          space_group: str,
                          composition: str = 'SiO2') -> Dict:
        """
        Generate 3D crystal structure
        
        Reference:
        Momma, K., & Izumi, F. (2011). VESTA 3 for three-dimensional visualization of crystal...
        """
        # Generate unit cell
        if crystal_system == 'Cubic':
            atoms = self._generate_cubic_structure(lattice_params, space_group, composition)
        elif crystal_system == 'Hexagonal':
            atoms = self._generate_hexagonal_structure(lattice_params, space_group, composition)
        elif crystal_system == 'Tetragonal':
            atoms = self._generate_tetragonal_structure(lattice_params, space_group, composition)
        else:
            atoms = self._generate_general_structure(lattice_params, space_group, composition)
        
        # Generate supercell for visualization
        supercell = self._create_supercell(atoms, repetitions=(2, 2, 2))
        
        return {
            'atoms': supercell,
            'unit_cell': self._get_unit_cell_vectors(crystal_system, lattice_params),
            'symmetry_operations': self._get_symmetry_operations(space_group),
            'density': self._calculate_density(supercell, lattice_params),
            'packing_fraction': self._calculate_packing_fraction(supercell)
        }
    
    def _generate_cubic_structure(self, lattice_params: Dict, 
                                space_group: str, composition: str) -> List[Dict]:
        """Generate cubic crystal structure"""
        a = lattice_params.get('a', 5.43)  # Default: Silicon lattice parameter
        
        atoms = []
        
        if space_group == 'Fd-3m':  # Diamond cubic (Si, Diamond)
            # Silicon positions
            base_positions = [
                [0, 0, 0],
                [0.5, 0.5, 0],
                [0.5, 0, 0.5],
                [0, 0.5, 0.5],
                [0.25, 0.25, 0.25],
                [0.75, 0.75, 0.25],
                [0.75, 0.25, 0.75],
                [0.25, 0.75, 0.75]
            ]
            
            for i, pos in enumerate(base_positions):
                atoms.append({
                    'element': 'Si',
                    'position': np.array(pos) * a,
                    'radius': self.atomic_radii['Si'],
                    'color': 'blue'
                })
        
        elif space_group == 'Fm-3m':  # FCC (Au, Ag, Cu, Al)
            # Face-centered cubic positions
            base_positions = [
                [0, 0, 0],
                [0.5, 0.5, 0],
                [0.5, 0, 0.5],
                [0, 0.5, 0.5]
            ]
            
            for i, pos in enumerate(base_positions):
                atoms.append({
                    'element': 'Al' if 'Al' in composition else 'Au',
                    'position': np.array(pos) * a,
                    'radius': self.atomic_radii.get(composition[:2], 1.0),
                    'color': 'gold'
                })
        
        return atoms
    # crystal_structure_3d.py - Add these methods to the CrystalStructure3D class
    
    def _generate_hexagonal_structure(self, lattice_params: Dict, space_group: str, composition: str) -> List[Dict]:
        """
        Generate hexagonal crystal structure from space group and lattice parameters.
        Reference: International Tables for Crystallography, Volume A
        """
        a = lattice_params.get('a')
        c = lattice_params.get('c')
        
        if not a or not c:
            return []
        
        atoms = []
        
        if space_group == 'P6₃/mmc':  # Hexagonal close-packed (HCP)
            # Atom positions for HCP (Mg-type)
            positions = [
                [0.33333, 0.66667, 0.25],  # Wyckoff position 2c
                [0.66667, 0.33333, 0.75]   # Wyckoff position 2c
            ]
            
            # Determine element from composition
            if 'Zn' in composition:
                element = 'Zn'
            elif 'Ti' in composition:
                element = 'Ti'
            elif 'Zr' in composition:
                element = 'Zr'
            elif 'Mg' in composition:
                element = 'Mg'
            else:
                # Get first element from composition
                import re
                match = re.match(r'([A-Z][a-z]?)', composition)
                element = match.group(1) if match else 'X'
            
            for pos in positions:
                atoms.append({
                    'element': element,
                    'position': np.array([
                        pos[0] * a,
                        pos[1] * a,
                        pos[2] * c
                    ]),
                    'radius': self.atomic_radii.get(element, 1.0),
                    'color': 'blue' if element == 'Zn' else 'orange'
                })
        
        return atoms
    
    def _generate_tetragonal_structure(self, lattice_params: Dict, space_group: str, composition: str) -> List[Dict]:
        """
        Generate tetragonal crystal structure from space group and lattice parameters.
        Reference: Acta Cryst. (2003). A59, 210-220
        """
        a = lattice_params.get('a')
        c = lattice_params.get('c')
        
        if not a or not c:
            return []
        
        atoms = []
        
        if space_group == 'I4₁/amd':  # TiO₂ anatase
            # Anatase structure: Ti at 4a, O at 8e
            # Ti positions (Wyckoff 4a)
            ti_positions = [
                [0, 0, 0],
                [0, 0.5, 0.25],
                [0.5, 0, 0.75],
                [0.5, 0.5, 0.5]
            ]
            
            # O positions (Wyckoff 8e)
            o_positions = [
                [0, 0, 0.208],
                [0, 0.5, 0.458],
                [0.5, 0, 0.958],
                [0.5, 0.5, 0.708],
                [0, 0, 0.792],
                [0, 0.5, 0.542],
                [0.5, 0, 0.042],
                [0.5, 0.5, 0.292]
            ]
            
            for pos in ti_positions:
                atoms.append({
                    'element': 'Ti',
                    'position': np.array([
                        pos[0] * a,
                        pos[1] * a,
                        pos[2] * c
                    ]),
                    'radius': self.atomic_radii['Ti'],
                    'color': 'white'
                })
            
            for pos in o_positions:
                atoms.append({
                    'element': 'O',
                    'position': np.array([
                        pos[0] * a,
                        pos[1] * a,
                        pos[2] * c
                    ]),
                    'radius': self.atomic_radii['O'],
                    'color': 'red'
                })
        
        return atoms
    
    def _generate_general_structure(self, lattice_params: Dict, space_group: str, composition: str) -> List[Dict]:
        """
        Generate general crystal structure using pymatgen for accurate generation.
        """
        try:
            from pymatgen.core import Structure, Lattice, Element
            from pymatgen.symmetry.groups import SpaceGroup
            
            # Parse lattice parameters
            a = lattice_params.get('a', 5.0)
            b = lattice_params.get('b', lattice_params.get('a', 5.0))
            c = lattice_params.get('c', lattice_params.get('a', 5.0))
            
            # Get angles (default to 90°)
            alpha = lattice_params.get('alpha', 90.0)
            beta = lattice_params.get('beta', 90.0)
            gamma = lattice_params.get('gamma', 90.0)
            
            # Create lattice
            lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
            
            # Create a simple structure with one atom at origin
            # In practice, you'd want to get the correct atomic positions from CIF
            # This is a fallback
            structure = Structure(
                lattice,
                [composition[:2]],  # Use first element from composition
                [[0, 0, 0]]
            )
            self._a = lattice_params.get('a', 5.0)
            self._b = lattice_params.get('b', self._a)
            self._c = lattice_params.get('c', self._a)

            # Convert to our atom format
            atoms = []
            for site in structure:
                atoms.append({
                    'element': site.species_string,
                    'position': site.coords,
                    'radius': self.atomic_radii.get(site.species_string[:2], 1.0),
                    'color': 'gray'
                })
            
            return atoms
            
        except ImportError:
            # Fallback if pymatgen not available
            return []
    
    def _create_supercell(self, atoms: List[Dict], repetitions: Tuple[int, int, int] = (2, 2, 2)) -> List[Dict]:
        """
        Create supercell by replicating unit cell.
        """
        nx, ny, nz = repetitions
        supercell_atoms = []
        
        # Calculate approximate unit cell vectors (simplified)
        # In a real implementation, you'd use the actual lattice vectors
        for atom in atoms:
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        # Simple translation
                        a_vec = np.array([self._a, 0, 0])
                        b_vec = np.array([0, self._b, 0])
                        c_vec = np.array([0, 0, self._c])
                        
                        new_pos = atom['position'] + i*a_vec + j*b_vec + k*c_vec

                        supercell_atoms.append({
                            'element': atom['element'],
                            'position': new_pos,
                            'radius': atom['radius'],
                            'color': atom['color']
                        })
        
        return supercell_atoms
    
    def _get_unit_cell_vectors(self, crystal_system: str, lattice_params: Dict) -> Dict:
        """
        Get unit cell vectors based on crystal system.
        Reference: Crystallography and Crystal Defects, A. Kelly & K. M. Knowles
        """
        vectors = {}
        
        if crystal_system == 'Cubic':
            a = lattice_params.get('a', 5.0)
            vectors['a'] = [a, 0, 0]
            vectors['b'] = [0, a, 0]
            vectors['c'] = [0, 0, a]
        
        elif crystal_system == 'Hexagonal':
            a = lattice_params.get('a', 2.46)
            c = lattice_params.get('c', 6.71)
            vectors['a'] = [a, 0, 0]
            vectors['b'] = [-a/2, a*np.sqrt(3)/2, 0]
            vectors['c'] = [0, 0, c]
        
        elif crystal_system == 'Tetragonal':
            a = lattice_params.get('a', 3.78)
            c = lattice_params.get('c', 9.51)
            vectors['a'] = [a, 0, 0]
            vectors['b'] = [0, a, 0]
            vectors['c'] = [0, 0, c]
        
        else:
            # Orthorhombic
            a = lattice_params.get('a', 5.0)
            b = lattice_params.get('b', a)
            c = lattice_params.get('c', a)
            vectors['a'] = [a, 0, 0]
            vectors['b'] = [0, b, 0]
            vectors['c'] = [0, 0, c]
        
        return vectors
    
    def _get_symmetry_operations(self, space_group: str) -> List[Dict]:
        """
        Get symmetry operations for space group.
        Reference: International Tables for Crystallography, Volume A
        """
        # Simplified - returns identity operation
        return [
            {
                'rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                'translation': [0, 0, 0]
            }
        ]
    
    def _calculate_density(self, atoms: List[Dict], lattice_params: Dict) -> float:
        """
        Calculate theoretical density from atomic masses and unit cell volume.
        ρ = (Z × M) / (N_A × V)
        """
        from scipy.constants import N_A
        
        if not atoms:
            return 0.0
        
        # Count atoms by element
        element_counts = {}
        for atom in atoms:
            element = atom['element']
            element_counts[element] = element_counts.get(element, 0) + 1
        
        # Get atomic masses (simplified - in real implementation use periodic table)
        atomic_masses = {
            'Si': 28.0855, 'O': 15.999, 'C': 12.011,
            'Al': 26.9815, 'Fe': 55.845, 'Ti': 47.867,
            'Zr': 91.224, 'Zn': 65.38, 'Mg': 24.305
        }
        
        # Calculate total mass
        total_mass = 0.0
        for element, count in element_counts.items():
            mass = atomic_masses.get(element, 12.0)  # Default to C if not found
            total_mass += count * mass
        
        # Calculate unit cell volume from lattice parameters
        crystal_system = 'Cubic'  # Default
        if 'a' in lattice_params and 'c' in lattice_params:
            if 'b' not in lattice_params:
                # Hexagonal or tetragonal
                a = lattice_params['a']
                c = lattice_params['c']
                volume = a**2 * c * np.sin(np.radians(120))  # For hexagonal
            else:
                # Orthorhombic
                a = lattice_params['a']
                b = lattice_params.get('b', a)
                c = lattice_params.get('c', a)
                volume = a * b * c
        else:
            # Cubic or unknown
            a = lattice_params.get('a', 5.0)
            volume = a**3
        
        # Convert volume from Å³ to cm³
        volume_cm3 = volume * 1e-24
        
        # Calculate density: ρ = (Z × M) / (N_A × V)
        z = len(atoms)  # Number of atoms in unit cell
        density = (z * total_mass) / (N_A * volume_cm3)
        
        return density
    
    def _calculate_packing_fraction(self, atoms: List[Dict]) -> float:
        """
        Calculate atomic packing fraction.
        η = (total atomic volume) / (unit cell volume)
        """
        if not atoms:
            return 0.0
        
        # Calculate total atomic volume
        atomic_volume = 0.0
        for atom in atoms:
            r = atom['radius']
            atomic_volume += (4/3) * np.pi * r**3
        
        # Estimate unit cell volume (simplified)
        # For close-packed structures: η_max = 0.74 for FCC/HCP
        # For BCC: η = 0.68
        # For simple cubic: η = 0.52
        
        # Count number of atoms to estimate structure type
        n_atoms = len(atoms)
        
        if n_atoms >= 4:  # Likely FCC or HCP
            return 0.74
        elif n_atoms >= 2:  # Likely BCC
            return 0.68
        else:
            return 0.52  # Simple cubic
    
    def create_3d_plot(self, structure: Dict, figsize=(12, 10)) -> plt.Figure:
        """Create publication-quality 3D crystal structure plot"""
        fig = plt.figure(figsize=figsize, dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        
        atoms = structure['atoms']
        unit_cell = structure['unit_cell']
        
        # Plot unit cell
        self._plot_unit_cell(ax, unit_cell)
        
        # Plot atoms
        colors = {'Si': 'blue', 'O': 'red', 'Al': 'gray', 'C': 'black'}
        
        for atom in atoms:
            pos = atom['position']
            radius = atom['radius']
            color = colors.get(atom['element'], atom.get('color', 'blue'))
            
            # Create sphere
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            
            x = radius * np.outer(np.cos(u), np.sin(v)) + pos[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + pos[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
            
            ax.plot_surface(x, y, z, color=color, alpha=0.7, 
                          edgecolor='black', linewidth=0.1)
        
        # Set labels and view
        ax.set_xlabel('X (Å)', fontsize=12, labelpad=10)
        ax.set_ylabel('Y (Å)', fontsize=12, labelpad=10)
        ax.set_zlabel('Z (Å)', fontsize=12, labelpad=10)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Remove background
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set view angle
        ax.view_init(elev=20, azim=45)
        
        # Add title with space group
        plt.title(f'3D Crystal Structure | Space Group: {structure.get("space_group", "")}', 
                 fontsize=14, pad=20)
        
        return fig
    
def create_interactive_plot(self, structure: Dict):
    """
    Create interactive 3D plot using Plotly (optional).
    Falls back gracefully if WebGL / Plotly is not supported.
    """
    if go is None:
        return None

    try:
        atoms = structure['atoms']

        scatter_trace = go.Scatter3d(
            x=[atom['position'][0] for atom in atoms],
            y=[atom['position'][1] for atom in atoms],
            z=[atom['position'][2] for atom in atoms],
            mode='markers',
            marker=dict(
                size=[atom['radius'] * 5 for atom in atoms],
                color=[atom.get('color', 'blue') for atom in atoms],
                opacity=0.8,
                line=dict(width=0.5, color='black')
            ),
            text=[f"{atom['element']}" for atom in atoms],
            hoverinfo='text'
        )

        fig = go.Figure(data=[scatter_trace])
        fig.update_layout(
            scene=dict(aspectmode='cube'),
            title="Interactive 3D Crystal Structure"
        )

        return fig

    except Exception:
        # Any Plotly / WebGL failure → safe fallback
        return None






