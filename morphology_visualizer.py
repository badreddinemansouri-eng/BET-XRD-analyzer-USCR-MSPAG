"""
ADVANCED MORPHOLOGY VISUALIZATION ENGINE
========================================================================
Generates scientific visualizations of material morphology based on
experimental BET and XRD data.

Features:
- 3D pore structure visualization
- 2D cross-sections showing porosity
- Crystal grain visualization
- Hierarchical structure diagrams
- SEM/TEM-style representations
========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 2D MORPHOLOGY VISUALIZATION
# ============================================================================
class MorphologyVisualizer:
    """Creates scientific visualizations of material morphology"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        self.colors = {
            'solid': '#4A90E2',      # Blue for solid material
            'micropores': '#FF6B6B',  # Red for micropores
            'mesopores': '#45B7D1',   # Cyan for mesopores
            'macropores': '#96CEB4',  # Green for macropores
            'crystalline': '#FECA57',  # Yellow for crystalline regions
            'amorphous': '#A8D8EA',    # Light blue for amorphous regions
            'boundary': '#2C3E50'      # Dark for boundaries
        }
    
    def create_pore_structure_2d(self, bet_results: Dict, 
                                 xrd_results: Optional[Dict] = None,
                                 size=1000) -> plt.Figure:
        """
        Create 2D pore structure visualization based on BET data
        
        Parameters:
        -----------
        bet_results : BET analysis results
        xrd_results : XRD analysis results (optional)
        size : Image size in pixels
        
        Returns:
        --------
        matplotlib Figure
        """
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        
        # Extract BET parameters
        S_bet = bet_results.get('surface_area', 0)
        V_total = bet_results.get('total_pore_volume', 0)
        V_micro = bet_results.get('micropore_volume', 0)
        d_mean = bet_results.get('mean_pore_diameter', 0)
        
        # Extract XRD parameters if available
        crystallinity = 0
        if xrd_results:
            crystallinity = xrd_results.get('crystallinity_index', 0)
        
        # Calculate derived parameters
        if V_total > 0:
            porosity = min(V_total / (V_total + 1/2.0), 0.95)  # Assume density ~2 g/cm³
            micro_fraction = V_micro / V_total if V_total > 0 else 0
        else:
            porosity = 0
            micro_fraction = 0
        
        # ====================================================================
        # SUBPLOT 1: Hierarchical pore structure
        # ====================================================================
        ax1 = axes[0, 0]
        self._plot_hierarchical_structure(ax1, S_bet, porosity, micro_fraction, d_mean)
        ax1.set_title('(A) Hierarchical Pore Structure', fontsize=12, pad=10)
        ax1.set_aspect('equal')
        ax1.axis('off')
        
        # ====================================================================
        # SUBPLOT 2: Pore size distribution visualization
        # ====================================================================
        ax2 = axes[0, 1]
        self._plot_pore_size_distribution(ax2, bet_results)
        ax2.set_title('(B) Pore Size Distribution Map', fontsize=12, pad=10)
        ax2.set_aspect('equal')
        ax2.axis('off')
        
        # ====================================================================
        # SUBPLOT 3: 3D-like pore network
        # ====================================================================
        ax3 = axes[0, 2]
        self._plot_3d_pore_network(ax3, bet_results, size=200)
        ax3.set_title('(C) 3D Pore Network Representation', fontsize=12, pad=10)
        ax3.set_aspect('equal')
        ax3.axis('off')
        
        # ====================================================================
        # SUBPLOT 4: Crystal structure integration
        # ====================================================================
        ax4 = axes[1, 0]
        if xrd_results:
            self._plot_crystal_integration(ax4, bet_results, xrd_results)
        else:
            self._plot_material_texture(ax4, S_bet, porosity)
        ax4.set_title('(D) Crystal-Pore Integration' if xrd_results else '(D) Material Texture', 
                     fontsize=12, pad=10)
        ax4.set_aspect('equal')
        ax4.axis('off')
        
        # ====================================================================
        # SUBPLOT 5: Cross-sectional view
        # ====================================================================
        ax5 = axes[1, 1]
        self._plot_cross_section(ax5, bet_results, crystallinity)
        ax5.set_title('(E) Cross-sectional View', fontsize=12, pad=10)
        ax5.set_aspect('equal')
        ax5.axis('off')
        
        # ====================================================================
        # SUBPLOT 6: SEM/TEM-style representation
        # ====================================================================
        ax6 = axes[1, 2]
        self._plot_sem_style(ax6, bet_results, xrd_results)
        ax6.set_title('(F) SEM/TEM-style Representation', fontsize=12, pad=10)
        ax6.set_aspect('equal')
        ax6.axis('off')
        
        # Add overall title
        material_type = self._classify_material(bet_results, xrd_results)
        plt.suptitle(f'Material Morphology Visualization: {material_type}', 
                    fontsize=14, y=0.95)
        
        plt.tight_layout()
        return fig
    
    def _plot_hierarchical_structure(self, ax, S_bet, porosity, micro_fraction, d_mean):
        """Plot hierarchical pore structure diagram"""
        # Create hierarchical representation
        scale_factor = min(S_bet / 500, 1.0)  # Normalize by typical surface area
        
        # Draw main particle
        main_radius = 2.0
        main_circle = Circle((0, 0), main_radius, 
                           facecolor=self.colors['solid'], 
                           edgecolor=self.colors['boundary'],
                           linewidth=2)
        ax.add_patch(main_circle)
        
        # Draw mesopores (medium-sized pores)
        n_mesopores = int(10 * porosity * (1 - micro_fraction))
        for i in range(n_mesopores):
            angle = 2 * np.pi * i / max(n_mesopores, 1)
            radius = 0.8 * main_radius * (1 - micro_fraction)
            x = radius * np.cos(angle) * 0.7
            y = radius * np.sin(angle) * 0.7
            pore_size = 0.15 + 0.1 * np.random.random()
            pore = Circle((x, y), pore_size, 
                         facecolor=self.colors['mesopores'],
                         alpha=0.7)
            ax.add_patch(pore)
            
            # Draw some micropores inside mesopores (hierarchical)
            if micro_fraction > 0.3:
                for j in range(3):
                    micro_x = x + (pore_size * 0.5) * np.cos(2 * np.pi * j / 3)
                    micro_y = y + (pore_size * 0.5) * np.sin(2 * np.pi * j / 3)
                    micro_pore = Circle((micro_x, micro_y), pore_size * 0.3,
                                       facecolor=self.colors['micropores'],
                                       alpha=0.9)
                    ax.add_patch(micro_pore)
        
        # Draw macropores if mean pore diameter is large
        if d_mean > 10:  # nm
            n_macropores = int(3 * porosity)
            for i in range(n_macropores):
                angle = np.pi/2 + np.pi * i / max(n_macropores, 1)
                radius = 1.2 * main_radius
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                pore_size = 0.3 + 0.2 * np.random.random()
                pore = Circle((x, y), pore_size,
                             facecolor=self.colors['macropores'],
                             alpha=0.6)
                ax.add_patch(pore)
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
    
    def _plot_pore_size_distribution(self, ax, bet_results):
        """Create pore size distribution map"""
        # Generate pore size distribution
        if bet_results.get('psd_analysis', {}).get('available', False):
            psd = bet_results['psd_analysis']
            pore_sizes = psd['pore_diameters']
            pore_volumes = psd['dv_dlogd']
            
            # Normalize for visualization
            pore_sizes_norm = np.array(pore_sizes) / max(pore_sizes)
            pore_volumes_norm = np.array(pore_volumes) / max(pore_volumes)
        else:
            # Generate synthetic distribution based on mean pore diameter
            d_mean = bet_results.get('mean_pore_diameter', 5)
            if d_mean < 2:
                # Microporous
                pore_sizes_norm = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
                pore_volumes_norm = np.array([0.8, 0.6, 0.4, 0.2, 0.1])
            elif d_mean < 20:
                # Mesoporous
                pore_sizes_norm = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
                pore_volumes_norm = np.array([0.2, 0.8, 0.6, 0.3, 0.1])
            else:
                # Macroporous
                pore_sizes_norm = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
                pore_volumes_norm = np.array([0.1, 0.3, 0.8, 0.6, 0.4])
        
        # Create pore size map
        n_pores = 50
        patches = []
        colors = []
        
        for i in range(n_pores):
            # Random position
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-2, 2)
            
            # Select pore size based on distribution
            idx = np.random.choice(len(pore_sizes_norm), p=pore_volumes_norm/pore_volumes_norm.sum())
            size = 0.05 + pore_sizes_norm[idx] * 0.3
            
            # Determine pore type
            pore_size_actual = pore_sizes_norm[idx]
            if pore_size_actual < 0.3:
                color = self.colors['micropores']
            elif pore_size_actual < 0.7:
                color = self.colors['mesopores']
            else:
                color = self.colors['macropores']
            
            pore = Circle((x, y), size, facecolor=color, alpha=0.7)
            patches.append(pore)
        
        # Add patches to plot
        collection = PatchCollection(patches, match_original=True)
        ax.add_collection(collection)
        
        # Add background material
        bg_rect = Rectangle((-2.5, -2.5), 5, 5,
                           facecolor=self.colors['solid'],
                           alpha=0.3)
        ax.add_patch(bg_rect)
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
    
    def _plot_3d_pore_network(self, ax, bet_results, size=200):
        """Create 3D-like pore network visualization"""
        # Generate random pore network
        np.random.seed(42)  # For reproducibility
        
        # Create grid
        x = np.linspace(-2, 2, size)
        y = np.linspace(-2, 2, size)
        X, Y = np.meshgrid(x, y)
        
        # Generate base pattern
        Z = np.zeros((size, size))
        
        # Add pores of different sizes
        V_total = bet_results.get('total_pore_volume', 0)
        n_pores = int(50 * min(V_total * 2, 1.0))
        
        for _ in range(n_pores):
            # Random pore center
            cx = np.random.uniform(-1.8, 1.8)
            cy = np.random.uniform(-1.8, 1.8)
            
            # Pore size based on PSD or mean diameter
            d_mean = bet_results.get('mean_pore_diameter', 5)
            pore_radius = 0.1 + 0.3 * np.random.random() * min(d_mean / 20, 1.0)
            
            # Create pore
            distance = np.sqrt((X - cx)**2 + (Y - cy)**2)
            Z += np.exp(-distance**2 / (2 * pore_radius**2))
        
        # Add some connectivity (pore network)
        for _ in range(10):
            x1, y1 = np.random.uniform(-1.8, 1.8, 2)
            x2, y2 = np.random.uniform(-1.8, 1.8, 2)
            
            # Create connecting channel
            t = np.linspace(0, 1, 100)
            channel_x = x1 + (x2 - x1) * t
            channel_y = y1 + (y2 - y1) * t
            
            for xi, yi in zip(channel_x, channel_y):
                dist = np.sqrt((X - xi)**2 + (Y - yi)**2)
                Z += 0.5 * np.exp(-dist**2 / (0.05**2))
        
        # Threshold to create binary pore network
        Z_threshold = gaussian_filter(Z, sigma=2)
        Z_binary = Z_threshold > np.percentile(Z_threshold, 70)
        
        # Create colormap for 3D effect
        from matplotlib.cm import hot
        im = ax.imshow(Z_binary, cmap='viridis', 
                      extent=[-2, 2, -2, 2],
                      alpha=0.8)
        
        # Add contour lines for 3D effect
        ax.contour(X, Y, Z_threshold, levels=5, 
                  colors='black', alpha=0.2, linewidths=0.5)
    
    def _plot_crystal_integration(self, ax, bet_results, xrd_results):
        """Plot integration of crystals and pores"""
        # Extract parameters
        crystallinity = xrd_results.get('crystallinity_index', 0)
        crystal_size = xrd_results.get('crystallite_size', {}).get('scherrer', 5)
        porosity = bet_results.get('total_pore_volume', 0) * 0.5  # Approximate
        
        # Create base material
        bg_rect = Rectangle((-2, -2), 4, 4,
                           facecolor=self.colors['amorphous'],
                           alpha=0.3)
        ax.add_patch(bg_rect)
        
        # Add crystalline regions
        n_crystals = int(20 * crystallinity)
        crystal_patches = []
        
        for i in range(n_crystals):
            # Random position
            x = np.random.uniform(-1.8, 1.8)
            y = np.random.uniform(-1.8, 1.8)
            
            # Crystal size based on XRD
            size = 0.1 + 0.3 * min(crystal_size / 20, 1.0)
            
            # Create crystal shape (hexagon for crystalline look)
            angles = np.linspace(0, 2*np.pi, 7)[:-1]
            crystal_coords = [(x + size * np.cos(angle), 
                              y + size * np.sin(angle)) 
                             for angle in angles]
            
            crystal = Polygon(crystal_coords,
                            facecolor=self.colors['crystalline'],
                            edgecolor=self.colors['boundary'],
                            linewidth=1,
                            alpha=0.8)
            crystal_patches.append(crystal)
            
            # Add some internal structure (atomic planes)
            for j in range(3):
                angle = np.pi * j / 3
                dx = 0.6 * size * np.cos(angle)
                dy = 0.6 * size * np.sin(angle)
                ax.plot([x - dx, x + dx], [y - dy, y + dy],
                       color='white', alpha=0.3, linewidth=0.5)
        
        # Add crystalline regions to plot
        for patch in crystal_patches:
            ax.add_patch(patch)
        
        # Add pores in amorphous regions
        n_pores = int(30 * porosity * (1 - crystallinity))
        for i in range(n_pores):
            # Avoid crystalline regions
            x = np.random.uniform(-1.8, 1.8)
            y = np.random.uniform(-1.8, 1.8)
            
            # Check if in crystalline region
            in_crystal = False
            for crystal in crystal_patches:
                if crystal.contains_point((x, y)):
                    in_crystal = True
                    break
            
            if not in_crystal:
                pore_size = 0.05 + 0.1 * np.random.random()
                pore = Circle((x, y), pore_size,
                             facecolor=self.colors['mesopores'],
                             alpha=0.6)
                ax.add_patch(pore)
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
    
    def _plot_material_texture(self, ax, S_bet, porosity):
        """Plot material texture for non-crystalline materials"""
        # Generate texture based on surface area and porosity
        texture_density = min(S_bet / 1000, 1.0)
        
        # Create base material with texture
        for i in range(100):
            for j in range(100):
                x = -2 + 4 * i / 100
                y = -2 + 4 * j / 100
                
                # Add texture based on surface area
                if np.random.random() < texture_density * 0.1:
                    size = 0.02 + 0.04 * np.random.random()
                    grain = Circle((x, y), size,
                                 facecolor=self.colors['solid'],
                                 alpha=0.5)
                    ax.add_patch(grain)
        
        # Add pores
        n_pores = int(100 * porosity)
        for i in range(n_pores):
            x = np.random.uniform(-1.8, 1.8)
            y = np.random.uniform(-1.8, 1.8)
            
            pore_size = 0.05 + 0.15 * np.random.random()
            pore = Circle((x, y), pore_size,
                         facecolor=self.colors['mesopores'],
                         alpha=0.7)
            ax.add_patch(pore)
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
    
    def _plot_cross_section(self, ax, bet_results, crystallinity):
        """Plot cross-sectional view of material"""
        # Create layered structure
        layers = []
        
        # Determine structure based on BET parameters
        S_bet = bet_results.get('surface_area', 0)
        V_total = bet_results.get('total_pore_volume', 0)
        
        if S_bet > 1000:
            # High surface area - very porous
            n_layers = 8
            pore_density = 0.7
        elif S_bet > 500:
            # Medium surface area
            n_layers = 6
            pore_density = 0.5
        else:
            # Low surface area
            n_layers = 4
            pore_density = 0.3
        
        # Create layers
        layer_height = 4.0 / n_layers
        for layer_idx in range(n_layers):
            y_base = -2 + layer_idx * layer_height
            
            # Create layer rectangle
            layer = Rectangle((-2, y_base), 4, layer_height,
                            facecolor=self.colors['solid'],
                            alpha=0.3,
                            edgecolor=self.colors['boundary'])
            ax.add_patch(layer)
            
            # Add pores to layer
            n_pores_layer = int(20 * pore_density * V_total * 2)
            for pore_idx in range(n_pores_layer):
                x = np.random.uniform(-1.8, 1.8)
                y = y_base + np.random.uniform(0.1, layer_height - 0.1)
                
                # Pore size varies with layer (gradient)
                pore_size = 0.05 + 0.1 * (layer_idx / n_layers)
                
                # Determine pore type
                if pore_size < 0.08:
                    color = self.colors['micropores']
                elif pore_size < 0.12:
                    color = self.colors['mesopores']
                else:
                    color = self.colors['macropores']
                
                pore = Circle((x, y), pore_size,
                             facecolor=color,
                             alpha=0.6)
                ax.add_patch(pore)
        
        # Add crystalline regions if applicable
        if crystallinity > 0.3:
            for layer_idx in range(n_layers):
                y_base = -2 + layer_idx * layer_height
                
                # Add some crystalline patches
                n_crystals = int(3 * crystallinity)
                for _ in range(n_crystals):
                    x = np.random.uniform(-1.5, 1.5)
                    y = y_base + np.random.uniform(0.2, layer_height - 0.2)
                    
                    # Draw crystal symbol (triangle)
                    crystal = Polygon([(x, y), 
                                      (x + 0.2, y + 0.15),
                                      (x - 0.2, y + 0.15)],
                                    facecolor=self.colors['crystalline'],
                                    alpha=0.7)
                    ax.add_patch(crystal)
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
    def _plot_sem_style(self, ax, bet_results, xrd_results=None):
        """
        SEM/TEM-style visualization wrapper.
        Calls the realistic SEM/TEM rendering implementation.
        """
        return self._plot_sem_style_realistic(ax, bet_results, xrd_results)

    def _plot_sem_style_realistic(self, ax, bet_results, xrd_results=None):
        """
        Realistic SEM/TEM visualization based on actual data
        
        References:
        1. Scherzer, O. (1949). J. Appl. Phys., 20, 20-29.
        2. Williams, D. B., & Carter, C. B. (1996). Transmission Electron Microscopy.
        """
        # Extract real data
        S_bet = bet_results.get('surface_area', 0)
        V_total = bet_results.get('total_pore_volume', 0)
        d_mean = bet_results.get('mean_pore_diameter', 0)
        
        if xrd_results:
            crystallinity = xrd_results.get('crystallinity_index', 0)
            crystal_size = xrd_results.get('crystallite_size', {}).get('scherrer', 0)
        else:
            crystallinity = 0
            crystal_size = 0
        
        # Calculate realistic parameters
        porosity = min(V_total * 2, 0.95)  # More realistic porosity estimation
        pore_density = S_bet / (np.pi * d_mean**2) if d_mean > 0 else 100
        
        # Generate realistic image
        size = 500  # Image size
        image = np.zeros((size, size))
        
        # Add crystalline regions if XRD data available
        if crystallinity > 0.3 and crystal_size > 0:
            n_crystals = int((size**2) * crystallinity / (crystal_size**2))
            for _ in range(n_crystals):
                x = np.random.randint(0, size)
                y = np.random.randint(0, size)
                radius = int(crystal_size / 2)  # pixels
                
                # Create crystalline grain with diffraction contrast
                for i in range(-radius, radius):
                    for j in range(-radius, radius):
                        if 0 <= x+i < size and 0 <= y+j < size:
                            dist = np.sqrt(i**2 + j**2)
                            if dist < radius:
                                # Simulate diffraction contrast (crystalline brighter)
                                intensity = 150 + 100 * np.random.random()
                                image[x+i, y+j] = max(image[x+i, y+j], intensity)
        
        # Add pores based on BET data
        if pore_density > 0 and d_mean > 0:
            n_pores = int((size**2) * porosity / (np.pi * (d_mean/2)**2))
            for _ in range(n_pores):
                x = np.random.randint(0, size)
                y = np.random.randint(0, size)
                radius = int(d_mean / 2)  # pixels
                
                # Create pore (dark region)
                for i in range(-radius, radius):
                    for j in range(-radius, radius):
                        if 0 <= x+i < size and 0 <= y+j < size:
                            dist = np.sqrt(i**2 + j**2)
                            if dist < radius:
                                # Pore is darker
                                image[x+i, y+j] = max(0, image[x+i, y+j] - 100)
        
        # Add amorphous background
        background = 100 * np.ones((size, size))
        
        # Apply Gaussian blur to simulate SEM/TEM resolution
        from scipy.ndimage import gaussian_filter
        image = gaussian_filter(image, sigma=1)
        
        # Combine with background
        final_image = np.clip(background + image, 0, 255)
        
        # Display image
        ax.imshow(final_image, cmap='gray', 
                 extent=[0, size/100, 0, size/100])  # Scale to nm
        
        # Add scale bar (100 nm)
        scale_length = 100  # nm
        scale_pixels = scale_length * (size / (size/100))
        
        ax.plot([10, 10 + scale_pixels], [10, 10], 'w-', linewidth=3)
        ax.text(10 + scale_pixels/2, 15, f'{scale_length} nm', 
               color='white', ha='center', fontsize=10)
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, size/100)
        ax.set_ylim(0, size/100)
        
        # Add text with actual parameters
        info_text = f'S_BET: {S_bet:.0f} m²/g\n'
        info_text += f'Porosity: {porosity*100:.1f}%\n'
        if crystal_size > 0:
            info_text += f'Crystal size: {crystal_size:.1f} nm'
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               color='white', fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    def _classify_material(self, bet_results, xrd_results=None):
        """Classify material type for title"""
        S_bet = bet_results.get('surface_area', 0)
        V_total = bet_results.get('total_pore_volume', 0)
        
        if xrd_results:
            crystallinity = xrd_results.get('crystallinity_index', 0)
            
            if S_bet > 1000 and crystallinity > 0.8:
                return "Crystalline Microporous Material"
            elif S_bet > 1000 and crystallinity < 0.3:
                return "Amorphous High-Surface-Area Material"
            elif S_bet > 500 and V_total > 0.8:
                return "Mesoporous Material"
            elif crystallinity > 0.7:
                return "Crystalline Material with Porosity"
            else:
                return "Porous Material"
        else:
            if S_bet > 1000:
                return "High Surface Area Porous Material"
            elif S_bet > 500:
                return "Mesoporous Material"
            else:
                return "Low Surface Area Material"

# ============================================================================
# INTEGRATED MORPHOLOGY ANALYSIS
# ============================================================================
class IntegratedMorphologyAnalyzer:
    """
    Integrated analysis combining visualization and interpretation
    """
    
    def __init__(self):
        self.visualizer = MorphologyVisualizer()
    
    def analyze_morphology(self, bet_results: Dict, xrd_results: Optional[Dict] = None) -> Dict:
        """
        Perform comprehensive morphology analysis
        
        Returns:
        --------
        Dictionary with visualization figure and detailed interpretation
        """
        analysis = {
            'valid': False,
            'visualization': None,
            'interpretation': {},
            'classification': {},
            'structure_properties': {}
        }
        
        try:
            # Generate visualization
            fig = self.visualizer.create_pore_structure_2d(bet_results, xrd_results)
            analysis['visualization'] = fig
            
            # Generate interpretation
            analysis['interpretation'] = self._generate_interpretation(bet_results, xrd_results)
            
            # Generate classification
            analysis['classification'] = self._classify_material_type(bet_results, xrd_results)
            
            # Calculate structure properties
            analysis['structure_properties'] = self._calculate_structure_properties(bet_results, xrd_results)
            
            analysis['valid'] = True
            
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def _generate_interpretation(self, bet_results, xrd_results=None):
        """Generate detailed morphological interpretation"""
        interpretation = {}
        
        # Extract key parameters
        S_bet = bet_results.get('surface_area', 0)
        V_total = bet_results.get('total_pore_volume', 0)
        d_mean = bet_results.get('mean_pore_diameter', 0)
        V_micro = bet_results.get('micropore_volume', 0)
        
        # Porosity interpretation
        porosity = min(V_total / (V_total + 1/2.0), 0.95)
        
        if porosity > 0.7:
            interpretation['porosity_level'] = 'Very High'
            interpretation['porosity_description'] = 'Material is highly porous with extensive void space'
        elif porosity > 0.4:
            interpretation['porosity_level'] = 'High'
            interpretation['porosity_description'] = 'Material shows significant porosity'
        elif porosity > 0.2:
            interpretation['porosity_level'] = 'Moderate'
            interpretation['porosity_description'] = 'Material has moderate porosity'
        else:
            interpretation['porosity_level'] = 'Low'
            interpretation['porosity_description'] = 'Material is relatively dense'
        
        # Surface area interpretation
        if S_bet > 1500:
            interpretation['surface_area_level'] = 'Exceptionally High'
            interpretation['surface_area_description'] = 'Ultra-high surface area suitable for adsorption applications'
        elif S_bet > 800:
            interpretation['surface_area_level'] = 'High'
            interpretation['surface_area_description'] = 'High surface area beneficial for catalytic applications'
        elif S_bet > 300:
            interpretation['surface_area_level'] = 'Moderate'
            interpretation['surface_area_description'] = 'Moderate surface area suitable for various applications'
        else:
            interpretation['surface_area_level'] = 'Low'
            interpretation['surface_area_description'] = 'Low surface area material'
        
        # Pore size interpretation
        if d_mean < 2:
            interpretation['pore_size_type'] = 'Microporous'
            interpretation['pore_size_description'] = 'Dominant micropores provide molecular sieving capability'
        elif d_mean < 50:
            interpretation['pore_size_type'] = 'Mesoporous'
            interpretation['pore_size_description'] = 'Mesopores facilitate mass transport and diffusion'
        else:
            interpretation['pore_size_type'] = 'Macroporous'
            interpretation['pore_size_description'] = 'Macropores provide fast transport pathways'
        
        # Hierarchical structure assessment
        if V_micro > 0 and V_total > V_micro * 2:
            interpretation['hierarchy'] = 'Hierarchical'
            interpretation['hierarchy_description'] = 'Material shows hierarchical porosity with multiple pore sizes'
        else:
            interpretation['hierarchy'] = 'Uniform'
            interpretation['hierarchy_description'] = 'Material has relatively uniform pore structure'
        
        # Add XRD-based interpretation if available
        if xrd_results:
            crystallinity = xrd_results.get('crystallinity_index', 0)
            crystal_size = xrd_results.get('crystallite_size', {}).get('scherrer', 0)
            
            if crystallinity > 0.8:
                interpretation['crystallinity'] = 'Highly Crystalline'
                interpretation['crystal_description'] = f'Well-defined crystalline structure with {crystal_size:.1f} nm crystallites'
            elif crystallinity > 0.5:
                interpretation['crystallinity'] = 'Crystalline'
                interpretation['crystal_description'] = f'Predominantly crystalline with some amorphous regions'
            elif crystallinity > 0.2:
                interpretation['crystallinity'] = 'Semi-crystalline'
                interpretation['crystal_description'] = f'Mixed crystalline and amorphous character'
            else:
                interpretation['crystallinity'] = 'Amorphous'
                interpretation['crystal_description'] = 'Predominantly amorphous structure'
        
        return interpretation
    
    def _classify_material_type(self, bet_results, xrd_results=None):
        """Classify material type based on properties"""
        classification = {}
        
        S_bet = bet_results.get('surface_area', 0)
        V_total = bet_results.get('total_pore_volume', 0)
        d_mean = bet_results.get('mean_pore_diameter', 0)
        
        # Determine primary classification
        if d_mean < 2 and S_bet > 800:
            classification['primary'] = 'Microporous Material'
            classification['examples'] = ['Activated Carbon', 'Zeolites', 'MOFs']
        elif 2 <= d_mean < 50 and V_total > 0.5:
            classification['primary'] = 'Mesoporous Material'
            classification['examples'] = ['Mesoporous Silica', 'Ordered Mesoporous Materials']
        elif d_mean >= 50:
            classification['primary'] = 'Macroporous Material'
            classification['examples'] = ['Porous Ceramics', 'Foams']
        elif S_bet > 1000:
            classification['primary'] = 'High Surface Area Material'
            classification['examples'] = ['Activated Carbon', 'Aerogels']
        else:
            classification['primary'] = 'Porous Solid'
            classification['examples'] = ['Porous Oxides', 'Catalyst Supports']
        
        # Add secondary characteristics
        characteristics = []
        
        if S_bet > 1000:
            characteristics.append('Ultra-high surface area')
        elif S_bet > 500:
            characteristics.append('High surface area')
        
        if V_total > 1.0:
            characteristics.append('High pore volume')
        elif V_total > 0.5:
            characteristics.append('Moderate pore volume')
        
        # Add XRD characteristics if available
        if xrd_results:
            crystallinity = xrd_results.get('crystallinity_index', 0)
            if crystallinity > 0.7:
                characteristics.append('Highly crystalline')
            elif crystallinity > 0.4:
                characteristics.append('Crystalline')
            else:
                characteristics.append('Amorphous')
        
        classification['characteristics'] = characteristics
        
        return classification
    
    def _calculate_structure_properties(self, bet_results, xrd_results=None):
        """Calculate derived structure properties"""
        properties = {}
        
        S_bet = bet_results.get('surface_area', 0)
        V_total = bet_results.get('total_pore_volume', 0)
        d_mean = bet_results.get('mean_pore_diameter', 0)
        
        # Calculate surface-to-volume ratio (m²/cm³)
        if V_total > 0:
            properties['surface_to_volume_ratio'] = S_bet / (V_total * 1e6)
        else:
            properties['surface_to_volume_ratio'] = 0
        
        # Calculate estimated pore wall thickness (nm)
        # Simplified model: assuming cylindrical pores
        if d_mean > 0 and V_total > 0:
            properties['estimated_wall_thickness'] = (4 * V_total * 1e9) / (S_bet * np.pi)
        else:
            properties['estimated_wall_thickness'] = 0
        
        # Calculate porosity percentage
        porosity = min(V_total / (V_total + 1/2.0), 0.95)  # Assume density ~2 g/cm³
        properties['porosity_percentage'] = porosity * 100
        
        # Calculate accessibility factor (0-1)
        # Higher for mesopores, lower for micropores
        if d_mean < 2:
            properties['accessibility_factor'] = 0.3
        elif d_mean < 10:
            properties['accessibility_factor'] = 0.7
        else:
            properties['accessibility_factor'] = 0.9
        
        # Add XRD-based properties if available
        if xrd_results:
            crystallinity = xrd_results.get('crystallinity_index', 0)
            crystal_size = xrd_results.get('crystallite_size', {}).get('scherrer', 0)
            
            properties['crystallinity_index'] = crystallinity
            properties['crystallite_size_nm'] = crystal_size
            
            # Calculate defect density (simplified)
            if crystal_size > 0:
                properties['estimated_defect_density'] = 1 / (crystal_size**2)
            else:
                properties['estimated_defect_density'] = 0
        
        return properties
