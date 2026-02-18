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
from typing import Dict, List, Tuple, Optional
import warnings

# Optional Plotly for interactive plots
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None


class CrystalStructure3D:
    """Generate 3D crystal structures from crystallographic data"""

    def __init__(self):
        self.atomic_radii = {
            'Si': 1.11, 'O': 0.73, 'C': 0.77, 'Al': 1.43,
            'Fe': 1.24, 'Ti': 1.47, 'Zr': 1.60, 'Ce': 1.85,
            'Na': 1.02, 'Ca': 1.00, 'Mg': 1.60, 'Zn': 1.42,
            'Au': 1.79, 'Ag': 1.75, 'Cu': 1.45, 'Pt': 1.77,
            'Pd': 1.63, 'Ni': 1.49, 'Co': 1.52, 'Mn': 1.61,
        }
        # Lattice parameters stored after generation
        self._a = self._b = self._c = 5.0  # defaults

    def generate_structure(self, crystal_system: str,
                          lattice_params: Dict,
                          space_group: str,
                          composition: str = 'SiO2') -> Dict:
        """
        Generate 3D crystal structure.

        Parameters
        ----------
        crystal_system : str
            e.g., 'Cubic', 'Hexagonal', 'Tetragonal'
        lattice_params : dict
            e.g., {'a': 5.43, 'c': 6.71} – must contain at least 'a'
        space_group : str
            e.g., 'Fd-3m', 'P6₃/mmc'
        composition : str
            Chemical formula (e.g., 'SiO2', 'TiO2')

        Returns
        -------
        dict with keys: 'atoms', 'unit_cell', 'symmetry_operations', 'density', 'packing_fraction'
        """
        # Store lattice parameters for later use
        self._a = lattice_params.get('a', 5.0)
        self._b = lattice_params.get('b', self._a)
        self._c = lattice_params.get('c', self._a)

        # Generate atoms based on crystal system and space group
        if crystal_system == 'Cubic':
            atoms = self._generate_cubic_structure(lattice_params, space_group, composition)
        elif crystal_system == 'Hexagonal':
            atoms = self._generate_hexagonal_structure(lattice_params, space_group, composition)
        elif crystal_system == 'Tetragonal':
            atoms = self._generate_tetragonal_structure(lattice_params, space_group, composition)
        else:
            atoms = self._generate_general_structure(lattice_params, space_group, composition)

        # Create supercell for better visualization
        supercell = self._create_supercell(atoms, repetitions=(2, 2, 2))

        return {
            'atoms': supercell,
            'unit_cell': self._get_unit_cell_vectors(crystal_system, lattice_params),
            'symmetry_operations': self._get_symmetry_operations(space_group),
            'density': self._calculate_density(supercell, lattice_params),
            'packing_fraction': self._calculate_packing_fraction(supercell),
            'space_group': space_group
        }

    def _generate_cubic_structure(self, lattice_params: Dict,
                                  space_group: str, composition: str) -> List[Dict]:
        """Generate cubic crystal structure."""
        a = lattice_params.get('a', 5.43)
        atoms = []

        if space_group == 'Fd-3m':  # Diamond cubic (Si, C)
            # Diamond structure
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
            element = 'Si' if 'Si' in composition else 'C'
            for pos in base_positions:
                atoms.append({
                    'element': element,
                    'position': np.array(pos) * a,
                    'radius': self.atomic_radii.get(element, 1.0),
                    'color': 'blue' if element == 'Si' else 'gray'
                })

        elif space_group == 'Fm-3m':  # FCC (Au, Ag, Cu, Al)
            base_positions = [
                [0, 0, 0],
                [0.5, 0.5, 0],
                [0.5, 0, 0.5],
                [0, 0.5, 0.5]
            ]
            # Guess element from composition
            import re
            match = re.match(r'([A-Z][a-z]?)', composition)
            element = match.group(1) if match else 'Au'
            color_map = {'Au': 'gold', 'Ag': 'silver', 'Cu': 'orange', 'Al': 'gray'}
            color = color_map.get(element, 'gold')
            for pos in base_positions:
                atoms.append({
                    'element': element,
                    'position': np.array(pos) * a,
                    'radius': self.atomic_radii.get(element, 1.5),
                    'color': color
                })

        return atoms

    def _generate_hexagonal_structure(self, lattice_params: Dict,
                                      space_group: str, composition: str) -> List[Dict]:
        """Generate hexagonal crystal structure."""
        a = lattice_params.get('a')
        c = lattice_params.get('c')
        if not a or not c:
            return []

        atoms = []
        if space_group == 'P6₃/mmc':  # HCP (Mg, Zn, Ti)
            positions = [
                [0.33333, 0.66667, 0.25],
                [0.66667, 0.33333, 0.75]
            ]
            # Guess element
            import re
            match = re.match(r'([A-Z][a-z]?)', composition)
            element = match.group(1) if match else 'Mg'
            color_map = {'Mg': 'blue', 'Zn': 'blue', 'Ti': 'gray', 'Zr': 'gray'}
            color = color_map.get(element, 'blue')
            for pos in positions:
                atoms.append({
                    'element': element,
                    'position': np.array([pos[0]*a, pos[1]*a, pos[2]*c]),
                    'radius': self.atomic_radii.get(element, 1.5),
                    'color': color
                })
        return atoms

    def _generate_tetragonal_structure(self, lattice_params: Dict,
                                       space_group: str, composition: str) -> List[Dict]:
        """Generate tetragonal crystal structure."""
        a = lattice_params.get('a')
        c = lattice_params.get('c')
        if not a or not c:
            return []

        atoms = []
        if space_group == 'I4₁/amd':  # Anatase TiO₂
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
                    'position': np.array([pos[0]*a, pos[1]*a, pos[2]*c]),
                    'radius': self.atomic_radii['Ti'],
                    'color': 'white'
                })
            for pos in o_positions:
                atoms.append({
                    'element': 'O',
                    'position': np.array([pos[0]*a, pos[1]*a, pos[2]*c]),
                    'radius': self.atomic_radii['O'],
                    'color': 'red'
                })
        return atoms

    def _generate_general_structure(self, lattice_params: Dict,
                                    space_group: str, composition: str) -> List[Dict]:
        """
        Fallback for other crystal systems – creates a simple lattice with one atom type.
        """
        a = self._a
        b = self._b
        c = self._c
        # Simple cubic lattice with one atom at origin
        import re
        match = re.match(r'([A-Z][a-z]?)', composition)
        element = match.group(1) if match else 'X'
        color = 'gray'
        atoms = [{
            'element': element,
            'position': np.array([0, 0, 0]),
            'radius': self.atomic_radii.get(element, 1.0),
            'color': color
        }]
        return atoms

    def _create_supercell(self, atoms: List[Dict], repetitions: Tuple[int, int, int] = (2, 2, 2)) -> List[Dict]:
        """Replicate atoms to create a supercell."""
        nx, ny, nz = repetitions
        supercell = []
        for atom in atoms:
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        translation = np.array([i*self._a, j*self._b, k*self._c])
                        new_atom = atom.copy()
                        new_atom['position'] = atom['position'] + translation
                        supercell.append(new_atom)
        return supercell

    def _get_unit_cell_vectors(self, crystal_system: str, lattice_params: Dict) -> Dict:
        """Return unit cell vectors as lists."""
        a = lattice_params.get('a', self._a)
        b = lattice_params.get('b', self._b)
        c = lattice_params.get('c', self._c)

        if crystal_system == 'Hexagonal':
            # Standard hexagonal vectors
            return {
                'a': [a, 0, 0],
                'b': [-a/2, a*np.sqrt(3)/2, 0],
                'c': [0, 0, c]
            }
        else:
            # Orthogonal systems
            return {
                'a': [a, 0, 0],
                'b': [0, b, 0],
                'c': [0, 0, c]
            }

    def _get_symmetry_operations(self, space_group: str) -> List[Dict]:
        """Return a list of symmetry operations (simplified)."""
        # For now, just return identity
        return [{'rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'translation': [0, 0, 0]}]

    def _calculate_density(self, atoms: List[Dict], lattice_params: Dict) -> float:
        """Calculate theoretical density in g/cm³."""
        if not atoms:
            return 0.0
        from scipy.constants import N_A

        # Count atoms by element
        element_counts = {}
        for atom in atoms:
            el = atom['element']
            element_counts[el] = element_counts.get(el, 0) + 1

        # Atomic masses (g/mol) – simplified
        atomic_masses = {
            'Si': 28.0855, 'O': 15.999, 'C': 12.011,
            'Al': 26.9815, 'Fe': 55.845, 'Ti': 47.867,
            'Zr': 91.224, 'Zn': 65.38, 'Mg': 24.305,
            'Au': 196.97, 'Ag': 107.87, 'Cu': 63.546,
            'Pt': 195.08, 'Pd': 106.42, 'Ni': 58.693,
            'Co': 58.933, 'Mn': 54.938
        }
        total_mass = 0.0
        for el, count in element_counts.items():
            mass = atomic_masses.get(el, 12.0)
            total_mass += count * mass

        # Unit cell volume in Å³
        a = lattice_params.get('a', self._a)
        b = lattice_params.get('b', self._b)
        c = lattice_params.get('c', self._c)
        # For non‑orthogonal systems we'd need angles; assume orthogonal for simplicity
        volume = a * b * c  # Å³
        volume_cm3 = volume * 1e-24

        # Number of formula units? Here we use total atom count as Z
        Z = len(atoms)
        density = (Z * total_mass) / (N_A * volume_cm3)
        return density

    def _calculate_packing_fraction(self, atoms: List[Dict]) -> float:
        """Estimate atomic packing fraction."""
        if not atoms:
            return 0.0
        # Very rough estimate based on atom count
        n = len(atoms)
        if n >= 8:
            return 0.74  # FCC/HCP
        elif n >= 4:
            return 0.68  # BCC
        else:
            return 0.52  # simple cubic

    def create_3d_plot(self, structure: Dict, figsize=(12, 10)) -> plt.Figure:
        """Create a publication‑quality 3D matplotlib plot."""
        fig = plt.figure(figsize=figsize, dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        atoms = structure.get('atoms', [])
        unit_cell = structure.get('unit_cell', {})

        # Plot unit cell edges
        self._plot_unit_cell(ax, unit_cell)

        # Plot atoms as spheres
        for atom in atoms:
            pos = atom['position']
            radius = atom['radius']
            color = atom.get('color', 'blue')

            # Sphere coordinates
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = radius * np.outer(np.cos(u), np.sin(v)) + pos[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + pos[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]

            ax.plot_surface(x, y, z, color=color, alpha=0.7,
                            edgecolor='black', linewidth=0.1)

        # Set labels and view
        ax.set_xlabel('X (Å)', fontsize=12, labelpad=10)
        ax.set_ylabel('Y (Å)', fontsize=12, labelpad=10)
        ax.set_zlabel('Z (Å)', fontsize=12, labelpad=10)
        ax.set_box_aspect([1, 1, 1])
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)

        # Remove panes for cleaner look
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        plt.title(f'3D Crystal Structure | Space Group: {structure.get("space_group", "")}',
                  fontsize=14, pad=20)
        return fig

    def _plot_unit_cell(self, ax, unit_cell: Dict):
        """Draw unit cell edges."""
        # Get corners
        a = np.array(unit_cell.get('a', [self._a, 0, 0]))
        b = np.array(unit_cell.get('b', [0, self._b, 0]))
        c = np.array(unit_cell.get('c', [0, 0, self._c]))

        corners = [
            np.array([0, 0, 0]),
            a, b, c,
            a + b, a + c, b + c,
            a + b + c
        ]
        # Edges: pairs of corners
        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (1, 5),
            (2, 4), (2, 6),
            (3, 5), (3, 6),
            (4, 7), (5, 7), (6, 7)
        ]
        for i, j in edges:
            ax.plot([corners[i][0], corners[j][0]],
                    [corners[i][1], corners[j][1]],
                    [corners[i][2], corners[j][2]], 'k-', lw=1.5)

    def create_interactive_plot(self, structure: Dict) -> Optional[go.Figure]:
        """Create an interactive Plotly 3D plot (if Plotly is available)."""
        if not PLOTLY_AVAILABLE:
            return None

        atoms = structure.get('atoms', [])
        if not atoms:
            return None

        x = [a['position'][0] for a in atoms]
        y = [a['position'][1] for a in atoms]
        z = [a['position'][2] for a in atoms]
        sizes = [a['radius'] * 10 for a in atoms]  # scale for visibility
        colors = [a.get('color', 'blue') for a in atoms]
        texts = [f"{a['element']}" for a in atoms]

        scatter = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.8,
                line=dict(width=0.5, color='black')
            ),
            text=texts,
            hoverinfo='text'
        )

        fig = go.Figure(data=[scatter])
        fig.update_layout(
            scene=dict(aspectmode='cube'),
            title="Interactive 3D Crystal Structure"
        )
        return fig
