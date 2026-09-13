"""
ADVANCED MORPHOLOGY VISUALIZATION ENGINE
========================================================================
Generates scientific visualizations of material morphology based on
experimental BET and XRD data.

All plots are DETERMINISTIC and DATA-DRIVEN. No random generation is used.
Every visual element is derived from measured or computed quantities.

References:
1. Rouquerol et al., Adsorption by Powders and Porous Solids, 2nd ed., 2014
2. Sing et al., Pure Appl. Chem., 1985, 57, 603-619 (IUPAC classification)
3. Thommes et al., Pure Appl. Chem., 2015, 87, 1051-1069
========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.collections import PatchCollection
from typing import Dict, List, Tuple, Optional


# ============================================================================
# MAIN VISUALIZER
# ============================================================================
class MorphologyVisualizer:
    """
    Deterministic, data-driven morphology visualization engine.
    Every plot is derived from measured BET or XRD quantities.
    """

    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        self.colors = {
            'solid': '#4A90E2',
            'micropores': '#FF6B6B',
            'mesopores': '#45B7D1',
            'macropores': '#96CEB4',
            'crystalline': '#FECA57',
            'amorphous': '#A8D8EA',
            'boundary': '#2C3E50'
        }

    # ========================================================================
    # MAIN ENTRY
    # ========================================================================
    def create_pore_structure_2d(self, bet_results: Dict,
                                 xrd_results: Optional[Dict] = None) -> plt.Figure:
        """
        6-panel deterministic morphology figure.
        All panels are derived from real BET/XRD parameters.
        """
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)

        S_bet = float(bet_results.get('surface_area', 0) or 0)
        V_total = float(bet_results.get('total_pore_volume', 0) or 0)
        V_micro = float(bet_results.get('micropore_volume', 0) or 0)
        d_mean = float(bet_results.get('mean_pore_diameter', 0) or 0)
        psd = bet_results.get('psd_analysis', {}) or {}

        crystallinity = 0.0
        crystallite_size = 0.0
        if xrd_results:
            crystallinity = float(xrd_results.get('crystallinity_index', 0) or 0)
            crystallite_size = float(
                (xrd_results.get('crystallite_size', {}) or {}).get('scherrer', 0) or 0
            )

        micro_fraction = (V_micro / V_total) if V_total > 0 else 0.0
        meso_fraction = psd.get('mesopore_fraction', 0.0) if psd.get('available') else 0.0
        macro_fraction = psd.get('macropore_fraction', 0.0) if psd.get('available') else 0.0

        # (A) Pore-size class breakdown from PSD (deterministic bars)
        ax1 = axes[0, 0]
        self._plot_pore_class_bars(ax1, micro_fraction, meso_fraction, macro_fraction, d_mean)
        ax1.set_title('(A) Pore Size Class Distribution', fontsize=12, pad=10)

        # (B) PSD curve from BJH (if available), otherwise histogram proxy
        ax2 = axes[0, 1]
        self._plot_psd_from_bjh(ax2, psd, d_mean)
        ax2.set_title('(B) Pore Size Distribution (BJH)', fontsize=12, pad=10)

        # (C) Surface-area contribution per pore class
        ax3 = axes[0, 2]
        self._plot_surface_contributions(ax3, S_bet, micro_fraction, meso_fraction, macro_fraction)
        ax3.set_title('(C) Surface Area Attribution', fontsize=12, pad=10)

        # (D) Crystallinity vs porosity cross-plot
        ax4 = axes[1, 0]
        self._plot_crystallinity_porosity(ax4, crystallinity, V_total, S_bet)
        ax4.set_title('(D) Crystallinity - Porosity Map', fontsize=12, pad=10)

        # (E) Crystallite size vs BET surface area
        ax5 = axes[1, 1]
        self._plot_size_vs_surface(ax5, crystallite_size, S_bet)
        ax5.set_title('(E) Size - Surface Area Relationship', fontsize=12, pad=10)

        # (F) IUPAC isotherm type hint (from hysteresis analysis if available)
        ax6 = axes[1, 2]
        self._plot_isotherm_class(ax6, bet_results)
        ax6.set_title('(F) IUPAC Isotherm Class', fontsize=12, pad=10)

        material_type = self._classify_material(bet_results, xrd_results)
        plt.suptitle(f'Material Morphology Visualization: {material_type}',
                     fontsize=14, y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    # ========================================================================
    # PANEL (A): Pore size class bars
    # ========================================================================
    def _plot_pore_class_bars(self, ax, micro, meso, macro, d_mean):
        classes = ['Micropore\n(<2 nm)', 'Mesopore\n(2-50 nm)', 'Macropore\n(>50 nm)']
        values = [max(0.0, micro) * 100, max(0.0, meso) * 100, max(0.0, macro) * 100]
        colors = [self.colors['micropores'], self.colors['mesopores'], self.colors['macropores']]

        bars = ax.bar(classes, values, color=colors, edgecolor='black', alpha=0.85)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.02 if max(values) > 0 else 1,
                    f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('Fraction of total pore volume (%)')
        ax.set_ylim(0, max(max(values) * 1.25, 10))
        ax.grid(True, axis='y', alpha=0.3)
        if d_mean > 0:
            ax.text(0.98, 0.95, f'Mean pore\n{d_mean:.2f} nm',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # ========================================================================
    # PANEL (B): PSD from BJH data
    # ========================================================================
    def _plot_psd_from_bjh(self, ax, psd, d_mean):
        if psd and psd.get('available') and psd.get('pore_diameters'):
            diameters = np.array(psd['pore_diameters'])
            dv_dlogd = np.array(psd['dv_dlogd'])
            ax.plot(diameters, dv_dlogd, '-', color=self.colors['mesopores'], linewidth=2)
            ax.fill_between(diameters, 0, dv_dlogd,
                            color=self.colors['mesopores'], alpha=0.3)
            ax.set_xscale('log')
            ax.set_xlabel('Pore diameter (nm)')
            ax.set_ylabel('dV/dlogD (cm3/g)')
            ax.axvspan(0.5, 2, alpha=0.08, color='red')
            ax.axvspan(2, 50, alpha=0.08, color='blue')
            ax.axvspan(50, 200, alpha=0.08, color='green')
            peak = psd.get('peak_pore_diameter', 0)
            if peak > 0:
                ax.axvline(peak, color='red', linestyle='--', linewidth=1, alpha=0.7)
                ax.text(peak * 1.05, max(dv_dlogd) * 0.8 if len(dv_dlogd) else 0,
                        f'Peak: {peak:.1f} nm', fontsize=9)
        else:
            ax.text(0.5, 0.5,
                    'BJH PSD not available\n(desorption branch required)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        ax.grid(True, alpha=0.3)

    # ========================================================================
    # PANEL (C): Surface area attribution
    # ========================================================================
    def _plot_surface_contributions(self, ax, S_bet, micro_f, meso_f, macro_f):
        # Surface area is attributed proportionally to pore volume fraction
        # when only PSD data is available. This is an approximation and is
        # stated explicitly.
        total_f = micro_f + meso_f + macro_f
        if S_bet <= 0 or total_f <= 0:
            ax.text(0.5, 0.5, 'No BET/PSD data available',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        micro_share = micro_f / total_f
        meso_share = meso_f / total_f
        macro_share = macro_f / total_f

        parts = [micro_share * S_bet, meso_share * S_bet, macro_share * S_bet]
        labels = ['Micropores', 'Mesopores', 'Macropores']
        colors = [self.colors['micropores'], self.colors['mesopores'], self.colors['macropores']]

        mask = [p > 0 for p in parts]
        parts_show = [p for p, m in zip(parts, mask) if m]
        labels_show = [l for l, m in zip(labels, mask) if m]
        colors_show = [c for c, m in zip(colors, mask) if m]

        if parts_show:
            ax.pie(parts_show, labels=labels_show, colors=colors_show,
                   autopct='%1.1f%%', startangle=90,
                   textprops={'fontsize': 9})
            ax.text(0.0, -1.35, f'Total SBET: {S_bet:.1f} m2/g\n'
                                '(Attribution by pore volume fraction)',
                    ha='center', va='top', fontsize=8, style='italic')
        else:
            ax.text(0.5, 0.5, 'No pore fractions available',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    # ========================================================================
    # PANEL (D): Crystallinity vs Porosity
    # ========================================================================
    def _plot_crystallinity_porosity(self, ax, crystallinity, V_pore, S_bet):
        # Porosity from pore volume with two assumed bulk densities
        if V_pore <= 0:
            ax.text(0.5, 0.5, 'No pore volume data available',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        rho_lo, rho_hi = 2.0, 3.0
        p_lo = V_pore / (V_pore + 1.0 / rho_hi)
        p_hi = V_pore / (V_pore + 1.0 / rho_lo)

        # Crosshair for current sample
        ax.axvspan(p_lo * 100, p_hi * 100, alpha=0.15, color='blue',
                   label='Porosity range')
        ax.axhline(crystallinity, color='red', linestyle='--', linewidth=1.5,
                   label=f'CI = {crystallinity:.2f}')

        ax.set_xlabel('Porosity (%)')
        ax.set_ylabel('Crystallinity index')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        ax.text(0.5, -0.22,
                'Porosity range reflects uncertainty in skeletal density (2-3 g/cm3)',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=7, style='italic')

    # ========================================================================
    # PANEL (E): Size vs Surface area
    # ========================================================================
    def _plot_size_vs_surface(self, ax, D_nm, S_bet):
        if D_nm <= 0 and S_bet <= 0:
            ax.text(0.5, 0.5, 'No size or surface data',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        # Theoretical curve S = 6000 / (rho * D) for rho in [2, 4] g/cm3
        D_axis = np.logspace(0, 3, 100)
        for rho in [2.0, 3.0, 4.0]:
            S_curve = 6000.0 / (rho * D_axis)
            ax.plot(D_axis, S_curve, '--', alpha=0.5, linewidth=1,
                    label=f'Theory (rho={rho} g/cm3)')

        if D_nm > 0 and S_bet > 0:
            ax.scatter([D_nm], [S_bet], s=120, color='red', zorder=5,
                       edgecolor='black', label='This sample')
            ax.annotate(f'({D_nm:.1f} nm, {S_bet:.1f} m2/g)',
                        xy=(D_nm, S_bet), xytext=(10, 10),
                        textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Crystallite size (nm)')
        ax.set_ylabel('Specific surface area (m2/g)')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='best', fontsize=7)

    # ========================================================================
    # PANEL (F): IUPAC isotherm class
    # ========================================================================
    def _plot_isotherm_class(self, ax, bet_results):
        hyst = bet_results.get('hysteresis_analysis', {}) or {}
        h_type = hyst.get('type', 'Unknown')
        iupac = hyst.get('iupac_class', 'Unknown')

        text = (
            f'IUPAC isotherm class: {iupac}\n'
            f'Hysteresis type: {h_type}\n\n'
            f'{hyst.get("description", "No hysteresis analysis available.")}'
        )
        ax.text(0.5, 0.5, text,
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, wrap=True,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                          edgecolor='black', alpha=0.9))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ========================================================================
    # CLASSIFICATION LABEL
    # ========================================================================
    def _classify_material(self, bet_results, xrd_results=None):
        S_bet = float(bet_results.get('surface_area', 0) or 0)
        V_total = float(bet_results.get('total_pore_volume', 0) or 0)

        if xrd_results:
            crystallinity = float(xrd_results.get('crystallinity_index', 0) or 0)
            if S_bet > 1000 and crystallinity > 0.8:
                return 'Crystalline Microporous Material'
            if S_bet > 1000 and crystallinity < 0.3:
                return 'Amorphous High-Surface-Area Material'
            if S_bet > 500 and V_total > 0.8:
                return 'Mesoporous Material'
            if crystallinity > 0.7:
                return 'Crystalline Material with Porosity'
            return 'Porous Material'

        if S_bet > 1000:
            return 'High Surface Area Porous Material'
        if S_bet > 500:
            return 'Mesoporous Material'
        return 'Low Surface Area Material'


# ============================================================================
# INTEGRATED ANALYZER
# ============================================================================
class IntegratedMorphologyAnalyzer:
    """
    Combines visualization and interpretation into a single workflow.
    """

    def __init__(self):
        self.visualizer = MorphologyVisualizer()

    def analyze_morphology(self, bet_results: Dict,
                           xrd_results: Optional[Dict] = None) -> Dict:
        analysis = {
            'valid': False,
            'visualization': None,
            'interpretation': {},
            'classification': {},
            'structure_properties': {}
        }

        try:
            fig = self.visualizer.create_pore_structure_2d(bet_results, xrd_results)
            analysis['visualization'] = fig
            analysis['interpretation'] = self._generate_interpretation(bet_results, xrd_results)
            analysis['classification'] = self._classify_material_type(bet_results, xrd_results)
            analysis['structure_properties'] = self._calculate_structure_properties(
                bet_results, xrd_results
            )
            analysis['valid'] = True
        except Exception as e:
            analysis['error'] = str(e)

        return analysis

    def _generate_interpretation(self, bet_results, xrd_results=None):
        interpretation = {}

        S_bet = float(bet_results.get('surface_area', 0) or 0)
        V_total = float(bet_results.get('total_pore_volume', 0) or 0)
        d_mean = float(bet_results.get('mean_pore_diameter', 0) or 0)
        V_micro = float(bet_results.get('micropore_volume', 0) or 0)

        if V_total > 0:
            rho_lo, rho_hi = 2.0, 3.0
            p_lo = V_total / (V_total + 1.0 / rho_hi)
            p_hi = V_total / (V_total + 1.0 / rho_lo)
            porosity = 0.5 * (p_lo + p_hi)
        else:
            porosity = 0.0

        if porosity > 0.7:
            interpretation['porosity_level'] = 'Very High'
            interpretation['porosity_description'] = (
                'Material is highly porous with extensive void space'
            )
        elif porosity > 0.4:
            interpretation['porosity_level'] = 'High'
            interpretation['porosity_description'] = (
                'Material shows significant porosity'
            )
        elif porosity > 0.2:
            interpretation['porosity_level'] = 'Moderate'
            interpretation['porosity_description'] = (
                'Material has moderate porosity'
            )
        else:
            interpretation['porosity_level'] = 'Low'
            interpretation['porosity_description'] = (
                'Material is relatively dense'
            )

        if S_bet > 1500:
            interpretation['surface_area_level'] = 'Exceptionally High'
            interpretation['surface_area_description'] = (
                'Ultra-high surface area suitable for adsorption applications'
            )
        elif S_bet > 800:
            interpretation['surface_area_level'] = 'High'
            interpretation['surface_area_description'] = (
                'High surface area beneficial for catalytic applications'
            )
        elif S_bet > 300:
            interpretation['surface_area_level'] = 'Moderate'
            interpretation['surface_area_description'] = (
                'Moderate surface area suitable for various applications'
            )
        else:
            interpretation['surface_area_level'] = 'Low'
            interpretation['surface_area_description'] = (
                'Low surface area material'
            )

        if d_mean < 2:
            interpretation['pore_size_type'] = 'Microporous'
            interpretation['pore_size_description'] = (
                'Dominant micropores provide molecular sieving capability'
            )
        elif d_mean < 50:
            interpretation['pore_size_type'] = 'Mesoporous'
            interpretation['pore_size_description'] = (
                'Mesopores facilitate mass transport and diffusion'
            )
        else:
            interpretation['pore_size_type'] = 'Macroporous'
            interpretation['pore_size_description'] = (
                'Macropores provide fast transport pathways'
            )

        if V_micro > 0 and V_total > V_micro * 2:
            interpretation['hierarchy'] = 'Hierarchical'
            interpretation['hierarchy_description'] = (
                'Material shows hierarchical porosity with multiple pore sizes'
            )
        else:
            interpretation['hierarchy'] = 'Uniform'
            interpretation['hierarchy_description'] = (
                'Material has relatively uniform pore structure'
            )

        if xrd_results:
            crystallinity = float(xrd_results.get('crystallinity_index', 0) or 0)
            crystal_size = float(
                (xrd_results.get('crystallite_size', {}) or {}).get('scherrer', 0) or 0
            )

            if crystallinity > 0.8:
                interpretation['crystallinity'] = 'Highly Crystalline'
                interpretation['crystal_description'] = (
                    f'Well-defined crystalline structure with ~{crystal_size:.1f} nm crystallites'
                )
            elif crystallinity > 0.5:
                interpretation['crystallinity'] = 'Crystalline'
                interpretation['crystal_description'] = (
                    'Predominantly crystalline with some amorphous regions'
                )
            elif crystallinity > 0.2:
                interpretation['crystallinity'] = 'Semi-crystalline'
                interpretation['crystal_description'] = (
                    'Mixed crystalline and amorphous character'
                )
            else:
                interpretation['crystallinity'] = 'Amorphous'
                interpretation['crystal_description'] = (
                    'Predominantly amorphous structure'
                )

        return interpretation

    def _classify_material_type(self, bet_results, xrd_results=None):
        classification = {}

        S_bet = float(bet_results.get('surface_area', 0) or 0)
        V_total = float(bet_results.get('total_pore_volume', 0) or 0)
        d_mean = float(bet_results.get('mean_pore_diameter', 0) or 0)

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

        characteristics = []
        if S_bet > 1000:
            characteristics.append('Ultra-high surface area')
        elif S_bet > 500:
            characteristics.append('High surface area')

        if V_total > 1.0:
            characteristics.append('High pore volume')
        elif V_total > 0.5:
            characteristics.append('Moderate pore volume')

        if xrd_results:
            crystallinity = float(xrd_results.get('crystallinity_index', 0) or 0)
            if crystallinity > 0.7:
                characteristics.append('Highly crystalline')
            elif crystallinity > 0.4:
                characteristics.append('Crystalline')
            else:
                characteristics.append('Amorphous')

        classification['characteristics'] = characteristics
        return classification

    def _calculate_structure_properties(self, bet_results, xrd_results=None):
        properties = {}

        S_bet = float(bet_results.get('surface_area', 0) or 0)
        V_total = float(bet_results.get('total_pore_volume', 0) or 0)
        d_mean = float(bet_results.get('mean_pore_diameter', 0) or 0)

        if V_total > 0:
            properties['surface_to_volume_ratio'] = S_bet / (V_total * 1e6)
        else:
            properties['surface_to_volume_ratio'] = 0.0

        if d_mean > 0 and V_total > 0:
            properties['estimated_wall_thickness'] = (
                (4 * V_total * 1e9) / (S_bet * np.pi)
            ) if S_bet > 0 else 0.0
        else:
            properties['estimated_wall_thickness'] = 0.0

        if V_total > 0:
            rho_lo, rho_hi = 2.0, 3.0
            p_lo = V_total / (V_total + 1.0 / rho_hi)
            p_hi = V_total / (V_total + 1.0 / rho_lo)
            properties['porosity_percentage'] = 0.5 * (p_lo + p_hi) * 100
            properties['porosity_range_percentage'] = (p_lo * 100, p_hi * 100)
        else:
            properties['porosity_percentage'] = 0.0

        if d_mean < 2:
            properties['accessibility_factor'] = 0.3
        elif d_mean < 10:
            properties['accessibility_factor'] = 0.7
        else:
            properties['accessibility_factor'] = 0.9

        if xrd_results:
            crystallinity = float(xrd_results.get('crystallinity_index', 0) or 0)
            crystal_size = float(
                (xrd_results.get('crystallite_size', {}) or {}).get('scherrer', 0) or 0
            )

            properties['crystallinity_index'] = crystallinity
            properties['crystallite_size_nm'] = crystal_size

            if crystal_size > 0:
                properties['estimated_defect_density'] = 1.0 / (crystal_size ** 2)
            else:
                properties['estimated_defect_density'] = 0.0

        return properties
