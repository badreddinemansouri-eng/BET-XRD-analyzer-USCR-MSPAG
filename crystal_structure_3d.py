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
        """Create interactive 3D plot using Plotly"""
        atoms = structure['atoms']
        
        # Create scatter plot for atoms
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
        
        # Create unit cell lines
        unit_cell = structure['unit_cell']
        lines = []
        
        # Add lines for unit cell edges
        vertices = [
            [0, 0, 0],
            unit_cell[0],
            unit_cell[1],
            unit_cell[2],
            unit_cell[0] + unit_cell[1],
            unit_cell[0] + unit_cell[2],
            unit_cell[1] + unit_cell[2],
            unit_cell[0] + unit_cell[1] + unit_cell[2]
        ]
        
        edges = [
            [0, 1], [0, 2], [0, 3],
            [1, 4], [1, 5],
            [2, 4], [2, 6],
            [3, 5], [3, 6],
            [4, 7], [5, 7], [6, 7]
        ]
        
        for edge in edges:
            line = go.Scatter3d(
                x=[vertices[edge[0]][0], vertices[edge[1]][0]],
                y=[vertices[edge[0]][1], vertices[edge[1]][1]],
                z=[vertices[edge[0]][2], vertices[edge[1]][2]],
                mode='lines',
                line=dict(color='black', width=3),
                showlegend=False
            )
            lines.append(line)
        
        fig = go.Figure(data=[scatter_trace] + lines)
        
        fig.update_layout(
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            title=dict(
                text=f"Interactive 3D Crystal Structure<br>Space Group: {structure.get('space_group', 'Unknown')}",
                x=0.5
            ),
            width=800,
            height=600
        )
        
        return fig