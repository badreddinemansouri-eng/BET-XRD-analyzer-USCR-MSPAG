"""
SCIENTIFIC PLOTTING ENGINE
========================================================================
Publication-quality plotting for journal submissions.

Features:
- Multi-panel figures with proper labeling (A, B, C, ...)
- HD resolution (600+ DPI)
- Journal-style formatting
- Vector graphics support
- Consistent color schemes
- Per-figure reference annotations for scientific traceability

References:
1. Rouquerol et al., Pure Appl. Chem., 1994, 66, 1739-1758
2. Thommes et al., Pure Appl. Chem., 2015, 87, 1051-1069
3. Klug & Alexander, X-ray Diffraction Procedures, 2nd ed., 1974
4. Williamson & Hall, Acta Metall., 1953, 1, 22-31
5. Barrett, Joyner & Halenda, J. Am. Chem. Soc., 1951, 73, 373
6. Harkins & Jura, J. Am. Chem. Soc., 1944, 66, 1366
========================================================================
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# COLOR SCHEMES FOR JOURNALS
# ============================================================================
COLOR_SCHEMES = {
    'Nature': {
        'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        'secondary': ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
        'background': '#FFFFFF',
        'text': '#000000',
        'grid': '#E0E0E0'
    },
    'Science': {
        'primary': ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600'],
        'secondary': ['#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600'],
        'background': '#FFFFFF',
        'text': '#000000',
        'grid': '#D0D0D0'
    },
    'ACS': {
        'primary': ['#0A6B9C', '#6F9C3B', '#F9A41A', '#D54332', '#854F9E'],
        'secondary': ['#4A90A4', '#8DBF3E', '#F8C53A', '#E27A3F', '#9B6CA6'],
        'background': '#FFFFFF',
        'text': '#000000',
        'grid': '#E5E5E5'
    },
    'RSC': {
        'primary': ['#003A6F', '#9E1B32', '#5D9732', '#D96748', '#7C3A95'],
        'secondary': ['#005CA9', '#C93A4C', '#7ABF3E', '#E88C5D', '#9A5CAC'],
        'background': '#FFFFFF',
        'text': '#000000',
        'grid': '#E8E8E8'
    },
    'Wiley': {
        'primary': ['#0056A6', '#D32F2F', '#388E3C', '#F57C00', '#7B1FA2'],
        'secondary': ['#1976D2', '#F44336', '#4CAF50', '#FF9800', '#9C27B0'],
        'background': '#FFFFFF',
        'text': '#000000',
        'grid': '#F0F0F0'
    }
}


# ============================================================================
# FIGURE CITATIONS (per-figure references)
# ============================================================================
FIGURE_CITATIONS = {
    'bet_isotherm': 'Rouquerol et al., Pure Appl. Chem. 66 (1994) 1739; Thommes et al., Pure Appl. Chem. 87 (2015) 1051',
    'bet_transform': 'Brunauer, Emmett & Teller, J. Am. Chem. Soc. 60 (1938) 309',
    't_plot': 'Harkins & Jura, J. Am. Chem. Soc. 66 (1944) 1366',
    'bjh_psd': 'Barrett, Joyner & Halenda, J. Am. Chem. Soc. 73 (1951) 373',
    'hysteresis': 'Sing et al., Pure Appl. Chem. 57 (1985) 603; IUPAC 2015',
    'xrd_pattern': 'Klug & Alexander, X-ray Diffraction Procedures, 2nd ed. (1974)',
    'williamson_hall': 'Williamson & Hall, Acta Metall. 1 (1953) 22',
    'scherrer_size': 'Scherrer, Nachr. Ges. Wiss. Gottingen 2 (1918) 98',
    'phase_fraction': 'Hill & Howard, J. Appl. Cryst. 20 (1987) 467 (semi-quantitative)',
    'crystallinity': 'Ruland, Acta Cryst. 14 (1961) 1180',
    'summary': 'Composite figure - see individual panel references',
    'morphology': 'IUPAC classification per Sing et al. (1985); Rouquerol et al. (2014)'
}


# ============================================================================
# STYLE
# ============================================================================
def set_publication_style(font_size=10, color_scheme='Nature'):
    plt.rcParams.update({
        'font.size': font_size,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.titlesize': font_size + 2,
        'axes.labelsize': font_size,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.grid.axis': 'both',
        'axes.grid.which': 'major',
        'grid.color': COLOR_SCHEMES[color_scheme]['grid'],
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'legend.fontsize': font_size - 1,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'black',
        'xtick.labelsize': font_size - 1,
        'ytick.labelsize': font_size - 1,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'figure.dpi': 300,
        'figure.figsize': [7.2, 4.8],
        'figure.autolayout': False,
        'savefig.dpi': 600,
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })


# ============================================================================
# PUBLICATION PLOTTER
# ============================================================================
class PublicationPlotter:
    """
    Publication-quality plotting with per-figure scientific citations.
    """

    def __init__(self, color_scheme='Nature', font_size=10):
        self.color_scheme = color_scheme
        self.font_size = font_size
        self.colors = COLOR_SCHEMES[color_scheme]
        set_publication_style(font_size, color_scheme)

    # ------------------------------------------------------------------
    # Helper: attach citation as figure-level text
    # ------------------------------------------------------------------
    def _attach_citation(self, fig, key, y=0.005):
        citation = FIGURE_CITATIONS.get(key, '')
        if citation:
            fig.text(0.5, y, f'Reference: {citation}',
                     ha='center', va='bottom',
                     fontsize=self.font_size - 2, style='italic', color='#444444')

    # ------------------------------------------------------------------
    # BET FIGURE
    # ------------------------------------------------------------------
    def create_bet_figure(self, bet_raw: Dict, bet_results: Dict) -> plt.Figure:
        bet_valid = bet_results.get('bet_valid', True)

        if bet_valid:
            fig = plt.figure(figsize=(12, 10.3))
            gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.32)
        else:
            fig = plt.figure(figsize=(12, 8.3))
            gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32)

        primary_color = self.colors['primary'][0]
        secondary_color = self.colors['primary'][1]

        # (A) Isotherm
        ax1 = fig.add_subplot(gs[0, 0])
        if 'p_ads' in bet_raw and 'q_ads' in bet_raw:
            ax1.plot(bet_raw['p_ads'], bet_raw['q_ads'], 'o-', color=primary_color,
                     markersize=4, linewidth=1.5, label='Adsorption')
        if ('p_des' in bet_raw and 'q_des' in bet_raw and
                bet_raw['p_des'] is not None and len(bet_raw['p_des']) > 0):
            ax1.plot(bet_raw['p_des'], bet_raw['q_des'], 's--', color=secondary_color,
                     markersize=3, linewidth=1.5, label='Desorption')
        ax1.set_xlabel('Relative pressure (P/P0)')
        ax1.set_ylabel('Quantity adsorbed (cm3/g)')
        ax1.set_title('(A) Adsorption-desorption isotherm', pad=8)
        if 'p_des' in bet_raw and bet_raw['p_des'] is not None:
            ax1.legend(loc='best', frameon=True)
        ax1.grid(True, alpha=0.3)
        ax1.text(0.02, 0.02, FIGURE_CITATIONS['bet_isotherm'][:60] + '...',
                 transform=ax1.transAxes, fontsize=self.font_size - 3,
                 style='italic', color='#666666', va='bottom')

        if not bet_valid:
            ax1.text(0.5, 0.95, 'No valid BET range found',
                     transform=ax1.transAxes, fontsize=self.font_size - 1,
                     va='top', ha='center',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

        # (B) BET transform
        if bet_valid:
            ax2 = fig.add_subplot(gs[0, 1])
            if 'bet_regression' in bet_results:
                bet_reg = bet_results['bet_regression']
                p_all = bet_raw['p_ads']
                q_all = bet_raw['q_ads']
                y_bet = p_all / (q_all * (1 - p_all))
                ax2.plot(p_all, y_bet, 'o', color='gray',
                         markersize=3, alpha=0.5, label='All data')
                p_min = bet_reg['p_min']
                p_max = bet_reg['p_max']
                mask = (p_all >= p_min) & (p_all <= p_max)
                if np.any(mask):
                    ax2.plot(p_all[mask], y_bet[mask], 'o', color=primary_color,
                             markersize=5, label='Linear range')
                    slope = bet_reg['slope']
                    intercept = bet_reg['intercept']
                    x_line = np.array([p_min, p_max])
                    ax2.plot(x_line, slope * x_line + intercept, '--',
                             color='red', linewidth=2, label='Linear fit')
                ax2.text(0.05, 0.95, f"y = {slope:.4f}x + {intercept:.4f}",
                         transform=ax2.transAxes, fontsize=self.font_size - 1, va='top')
                ax2.text(0.05, 0.88, f"R2 = {bet_reg['r_squared']:.6f}",
                         transform=ax2.transAxes, fontsize=self.font_size - 1, va='top')
            ax2.set_xlabel('Relative pressure (P/P0)')
            ax2.set_ylabel('p/[n(1-p)] (g/cm3)')
            ax2.set_title('(B) BET transform plot', pad=8)
            ax2.legend(loc='lower right', frameon=True)
            ax2.grid(True, alpha=0.3)
            ax2.text(0.02, 0.02, 'Brunauer-Emmett-Teller (1938)',
                     transform=ax2.transAxes, fontsize=self.font_size - 3,
                     style='italic', color='#666666', va='bottom')
        else:
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.text(0.5, 0.5,
                     'BET transform not available\nNo valid linear range found\n'
                     'in 0.05-0.35 P/P0 region',
                     transform=ax2.transAxes, ha='center', va='center',
                     fontsize=self.font_size)
            ax2.set_title('(B) BET transform plot', pad=8)
            ax2.set_xticks([])
            ax2.set_yticks([])

        # (C) t-plot
        ax3 = fig.add_subplot(gs[0, 2])
        if 't_plot_analysis' in bet_results and bet_results['t_plot_analysis'].get('valid'):
            p_ads = bet_raw['p_ads']
            t = (13.99 / (0.034 - np.log10(p_ads + 1e-10))) ** 0.5 * 0.1
            ax3.plot(t, bet_raw['q_ads'], 'o', color=primary_color,
                     markersize=4, alpha=0.7)
            t_plot = bet_results['t_plot_analysis']
            if t_plot['t_range'][1] > 0:
                mask = (t >= t_plot['t_range'][0]) & (t <= t_plot['t_range'][1])
                if np.sum(mask) >= 3:
                    t_sel = t[mask]
                    q_sel = bet_raw['q_ads'][mask]
                    coeffs = np.polyfit(t_sel, q_sel, 1)
                    t_fit = np.linspace(t_sel.min(), t_sel.max(), 100)
                    ax3.plot(t_fit, np.polyval(coeffs, t_fit), '--',
                             color='red', linewidth=2)
                    results_text = (f"V_micro = {t_plot['micropore_volume']:.3f} cm3/g\n"
                                    f"S_ext = {t_plot['external_surface']:.0f} m2/g")
                    ax3.text(0.05, 0.95, results_text, transform=ax3.transAxes,
                             fontsize=self.font_size - 2, va='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax3.set_xlabel('Statistical thickness t (nm)')
        ax3.set_ylabel('Quantity adsorbed (cm3/g)')
        ax3.set_title('(C) t-plot analysis', pad=8)
        ax3.grid(True, alpha=0.3)
        ax3.text(0.02, 0.02, 'Harkins & Jura (1944)',
                 transform=ax3.transAxes, fontsize=self.font_size - 3,
                 style='italic', color='#666666', va='bottom')

        row_idx = 1 if bet_valid else 0

        # (D) PSD
        ax4 = fig.add_subplot(gs[row_idx, :2])
        if 'psd_analysis' in bet_results and bet_results['psd_analysis'].get('available'):
            psd = bet_results['psd_analysis']
            ax4.plot(psd['pore_diameters'], psd['dv_dlogd'],
                     '-', color=primary_color, linewidth=2)
            ax4.fill_between(psd['pore_diameters'], 0, psd['dv_dlogd'],
                             alpha=0.3, color=primary_color)
            if psd.get('peak_pore_diameter', 0) > 0:
                ax4.axvline(psd['peak_pore_diameter'], color='red',
                            linestyle='--', linewidth=1, alpha=0.7)
                ax4.text(psd['peak_pore_diameter'] * 1.1,
                         max(psd['dv_dlogd']) * 0.8,
                         f"Peak: {psd['peak_pore_diameter']:.1f} nm",
                         fontsize=self.font_size - 1)
            ax4.axvspan(0, 2, alpha=0.08, color='blue', label='Micropores')
            ax4.axvspan(2, 50, alpha=0.08, color='green', label='Mesopores')
            ax4.axvspan(50, 200, alpha=0.08, color='red', label='Macropores')
            if all(k in psd for k in ['micropore_fraction', 'mesopore_fraction',
                                       'macropore_fraction']):
                frac_text = (f"Micro: {psd['micropore_fraction']*100:.1f}%\n"
                             f"Meso: {psd['mesopore_fraction']*100:.1f}%\n"
                             f"Macro: {psd['macropore_fraction']*100:.1f}%")
                ax4.text(0.98, 0.95, frac_text, transform=ax4.transAxes,
                         fontsize=self.font_size - 2, va='top', ha='right',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        ax4.set_xlabel('Pore diameter (nm)')
        ax4.set_ylabel('dV/dlogD (cm3/g)')
        ax4.set_title('(D) Pore size distribution (BJH)', pad=8)
        ax4.set_xscale('log')
        ax4.set_xlim(0.5, 200)
        ax4.legend(loc='upper right', frameon=True)
        ax4.grid(True, alpha=0.3, which='both')
        ax4.text(0.02, 0.02, 'Barrett-Joyner-Halenda (1951)',
                 transform=ax4.transAxes, fontsize=self.font_size - 3,
                 style='italic', color='#666666', va='bottom')

        # (E) Hysteresis
        ax5 = fig.add_subplot(gs[row_idx, 2])
        if ('hysteresis_analysis' in bet_results and
                bet_raw.get('p_des') is not None and
                len(bet_raw['p_des']) > 0):
            ax5.plot(bet_raw['p_ads'], bet_raw['q_ads'], 'o-',
                     color=primary_color, markersize=3, linewidth=1.5,
                     label='Adsorption')
            ax5.plot(bet_raw['p_des'], bet_raw['q_des'], 's--',
                     color=secondary_color, markersize=3, linewidth=1.5,
                     label='Desorption')
            try:
                if len(bet_raw['p_ads']) == len(bet_raw['p_des']):
                    ax5.fill_between(bet_raw['p_ads'], bet_raw['q_ads'],
                                     bet_raw['q_des'], alpha=0.2, color='gray')
            except Exception:
                pass
            hyst = bet_results['hysteresis_analysis']
            if hyst.get('valid'):
                info_text = (f"Type: {hyst['type']}\n"
                             f"IUPAC: {hyst['iupac_class']}\n"
                             f"Area: {hyst['loop_area']:.2f}")
                ax5.text(0.05, 0.95, info_text, transform=ax5.transAxes,
                         fontsize=self.font_size - 2, va='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax5.set_xlabel('Relative pressure (P/P0)')
        ax5.set_ylabel('Quantity adsorbed (cm3/g)')
        ax5.set_title('(E) Hysteresis loop', pad=8)
        if 'p_des' in bet_raw and bet_raw['p_des'] is not None:
            ax5.legend(loc='best', frameon=True)
        ax5.grid(True, alpha=0.3)
        ax5.text(0.02, 0.02, 'IUPAC classification - Sing et al. (1985)',
                 transform=ax5.transAxes, fontsize=self.font_size - 3,
                 style='italic', color='#666666', va='bottom')

        # (F) Summary table
        if bet_valid:
            ax6 = fig.add_subplot(gs[2, :])
            ax6.axis('tight')
            ax6.axis('off')

            summary_data = []
            summary_data.append([
                'Surface Area (SBET)',
                f"{bet_results.get('surface_area', 0):.1f} "
                f"+/- {bet_results.get('surface_area_error', 0):.1f} m2/g"
            ])
            summary_data.append([
                'Total Pore Volume',
                f"{bet_results.get('total_pore_volume', 0):.3f} cm3/g"
            ])
            summary_data.append([
                'Micropore Volume (t-plot)',
                f"{bet_results.get('micropore_volume', 0):.3f} cm3/g"
            ])
            summary_data.append([
                'Mean Pore Diameter',
                f"{bet_results.get('mean_pore_diameter', 0):.2f} nm"
            ])
            summary_data.append([
                'BET C Constant',
                f"{bet_results.get('c_constant', 0):.0f}"
            ])
            hyst_type = (bet_results.get('hysteresis_analysis', {}) or {}).get('type', 'N/A')
            summary_data.append(['Hysteresis Type', hyst_type])

            table = ax6.table(cellText=summary_data,
                              colLabels=['Parameter', 'Value'],
                              colWidths=[0.4, 0.3],
                              cellLoc='left',
                              loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(self.font_size - 1)
            table.scale(1, 1.5)
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor(self.colors['primary'][0])
                    cell.set_text_props(color='white')
                else:
                    cell.set_facecolor('#F5F5F5' if row % 2 == 0 else '#FFFFFF')
            ax6.set_title('(F) BET analysis summary', pad=18)

        if bet_valid:
            plt.suptitle('BET Surface Area and Porosity Analysis',
                         fontsize=self.font_size + 4, y=0.98)
        else:
            plt.suptitle('Porosity Analysis (BET linear range not found)',
                         fontsize=self.font_size + 4, y=0.98)

        self._attach_citation(fig, 'bet_isotherm', y=0.001)
        return fig

    # ------------------------------------------------------------------
    # XRD FIGURE
    # ------------------------------------------------------------------
    def create_xrd_figure(self, xrd_raw: Dict, xrd_results: Dict) -> plt.Figure:
        if "xrd_results" in xrd_results:
            xrd_results = xrd_results["xrd_results"]

        fig = plt.figure(figsize=(12, 10.3))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.28)

        primary_color = self.colors['primary'][0]
        two_theta = np.array(xrd_raw['two_theta'])
        intensity = np.array(xrd_raw['intensity'])

        def extract_hkl_indices(hkl_val):
            if hkl_val is None:
                return None
            if isinstance(hkl_val, (int, float, str)):
                return None
            if isinstance(hkl_val, (tuple, list)):
                if all(isinstance(x, (int, np.integer)) for x in hkl_val):
                    return tuple(int(x) for x in hkl_val)
                if len(hkl_val) > 0 and isinstance(hkl_val[0], dict):
                    return extract_hkl_indices(hkl_val[0])
                return None
            if isinstance(hkl_val, dict):
                for key in ['hkl', 'indices']:
                    if key in hkl_val:
                        return extract_hkl_indices(hkl_val[key])
                return None
            return None

        def format_hkl(hkl_val):
            indices = extract_hkl_indices(hkl_val)
            if indices is not None:
                return str(indices)
            return str(hkl_val)

        structural_peaks = xrd_results.get("structural_peaks", [])
        n_raw = xrd_results.get("n_detected_maxima", 0)
        n_structural = len(structural_peaks)

        display_peaks = sorted(
            sorted(structural_peaks, key=lambda p: p["intensity"], reverse=True)[:10],
            key=lambda p: p["position"]
        )

        # (A) XRD pattern
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(two_theta, intensity, '-', color=primary_color, lw=1.5)
        if display_peaks:
            ax1.scatter(
                [p["position"] for p in display_peaks],
                [p.get("intensity_raw", p["intensity"]) for p in display_peaks],
                color="red", s=35, zorder=5, label="Structural Bragg peaks"
            )
            for p in display_peaks:
                raw_int = p.get("intensity_raw", p["intensity"])
                ax1.annotate(
                    f"{p['position']:.2f} deg",
                    xy=(p["position"], raw_int),
                    xytext=(p["position"], raw_int * 1.1),
                    ha="center", fontsize=self.font_size - 2,
                    arrowprops=dict(arrowstyle="->", lw=0.5)
                )
        ax1.text(0.02, 0.95,
                 f"Structural Bragg peaks: {n_structural}\n"
                 f"Detected local maxima: {n_raw}",
                 transform=ax1.transAxes, fontsize=self.font_size - 1, va='top',
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))
        ci = xrd_results.get("crystallinity_index", 0)
        ax1.text(0.98, 0.95, f"Crystallinity Index: {ci:.2f}",
                 transform=ax1.transAxes, ha="right", va='top',
                 fontsize=self.font_size,
                 bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.85))
        ax1.set_xlabel("2theta (degrees)")
        ax1.set_ylabel("Intensity (a.u.)")
        ax1.set_title("(A) XRD pattern - structural Bragg peaks", pad=8)
        ax1.grid(alpha=0.3)
        ax1.legend(loc='lower right')
        ax1.text(0.02, 0.02, 'Klug & Alexander (1974)',
                 transform=ax1.transAxes, fontsize=self.font_size - 3,
                 style='italic', color='#666666', va='bottom')

        # (B) Williamson-Hall
        ax2 = fig.add_subplot(gs[1, 0])
        wh = xrd_results.get("williamson_hall")
        if isinstance(wh, dict) and len(wh.get("x_data", [])) >= 3:
            ax2.scatter(wh["x_data"], wh["y_data"], s=40,
                        color=primary_color, edgecolor='black')
            xfit = np.linspace(min(wh["x_data"]), max(wh["x_data"]), 100)
            ax2.plot(xfit, wh["slope"] * xfit + wh["intercept"], "--r",
                     linewidth=2)
            ax2.text(0.05, 0.95,
                     f"R2 = {wh.get('r_squared', 0):.3f}\n"
                     f"D = {wh.get('crystallite_size', 0):.1f} nm\n"
                     f"strain = {wh.get('microstrain', 0):.4f}",
                     transform=ax2.transAxes, fontsize=self.font_size - 2, va='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
            ax2.set_xlabel('4 sin(theta)')
            ax2.set_ylabel('beta cos(theta)')
            ax2.set_title("(B) Williamson-Hall plot", pad=8)
            ax2.grid(alpha=0.3)
            ax2.text(0.02, 0.02, 'Williamson & Hall (1953)',
                     transform=ax2.transAxes, fontsize=self.font_size - 3,
                     style='italic', color='#666666', va='bottom')
        else:
            ax2.text(0.5, 0.5,
                     "Williamson-Hall not valid\n"
                     "(requires >=4 independent reflections with R2 > 0.85)",
                     ha="center", va="center", transform=ax2.transAxes)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_title("(B) Williamson-Hall plot", pad=8)

        # (C) Size distribution
        ax3 = fig.add_subplot(gs[1, 1])
        sizes = [p["crystallite_size"] for p in structural_peaks
                 if p.get("crystallite_size", 0) > 0]
        if len(sizes) >= 3:
            ax3.hist(sizes, bins=min(10, len(sizes)), alpha=0.75,
                     color=primary_color, edgecolor='black')
            ax3.axvline(np.mean(sizes), color='red', linestyle='--',
                        linewidth=2, label=f'Mean: {np.mean(sizes):.1f} nm')
            ax3.legend(loc='best')
            ax3.set_xlabel("Crystallite size (nm)")
            ax3.set_ylabel("Frequency")
            ax3.set_title("(C) Scherrer crystallite size distribution", pad=8)
            ax3.grid(alpha=0.3, axis="y")
            ax3.text(0.02, 0.02, 'Scherrer (1918)',
                     transform=ax3.transAxes, fontsize=self.font_size - 3,
                     style='italic', color='#666666', va='bottom')
        else:
            ax3.text(0.5, 0.5, "Insufficient data for size distribution",
                     ha="center", va="center", transform=ax3.transAxes)
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.set_title("(C) Scherrer crystallite size distribution", pad=8)

        # (D) Peak table
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.axis("off")
        if display_peaks:
            table_data = [[
                i + 1,
                f"{p['position']:.2f}",
                f"{p.get('d_spacing', 0):.3f}",
                f"{p['fwhm_deg']:.3f}",
                f"{p.get('crystallite_size', 0):.1f}",
                format_hkl(p.get("hkl", ""))
            ] for i, p in enumerate(display_peaks)]
            table = ax4.table(
                cellText=table_data,
                colLabels=["#", "2theta", "d (A)", "FWHM", "Size (nm)", "HKL"],
                loc="center", cellLoc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(self.font_size - 1)
            table.scale(1, 1.2)
            ax4.set_title("(D) Structural Bragg peaks", pad=8)

        # (E) Summary
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis("off")
        summary = [
            ["Structural Bragg Peaks", str(n_structural)],
            ["Detected Local Maxima", str(n_raw)],
            ["Crystallinity Index", f"{ci:.2f}"],
            ["Scherrer Size",
             f"{(xrd_results.get('crystallite_size', {}) or {}).get('scherrer', 0):.1f} nm"]
        ]
        table = ax5.table(cellText=summary,
                          colLabels=["Parameter", "Value"],
                          loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(self.font_size - 1)
        table.scale(1, 1.2)
        ax5.set_title("(E) XRD analysis summary", pad=8)

        plt.suptitle("X-ray Diffraction Analysis (Nanomaterial-Validated)",
                     fontsize=self.font_size + 4, y=0.98)
        self._attach_citation(fig, 'xrd_pattern', y=0.001)
        return fig

    # ------------------------------------------------------------------
    # PHASE FRACTION FIGURE
    # ------------------------------------------------------------------
    def create_phase_fraction_plot(self, phase_fractions) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 4.3))

        if not phase_fractions:
            ax.text(0.5, 0.5, "No phase fraction data available",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            self._attach_citation(fig, 'phase_fraction')
            return fig

        phases = [p["phase"] for p in phase_fractions]
        fractions = [p["fraction"] for p in phase_fractions]

        bars = ax.bar(phases, fractions, color=self.colors["primary"],
                      edgecolor='black')
        for bar, frac in zip(bars, fractions):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(fractions) * 0.02,
                    f"{frac:.1f}%", ha="center", va="bottom",
                    fontsize=self.font_size)

        ax.set_ylabel("Phase fraction (%)")
        ax.set_title("Phase Composition (Semi-quantitative)")
        ax.set_ylim(0, max(fractions) * 1.2 if max(fractions) > 0 else 1)
        ax.grid(True, axis="y", alpha=0.3)
        ax.text(0.5, -0.20,
                "Semi-quantitative estimate based on peak intensity weighting.\n"
                "Not a full Rietveld refinement.",
                transform=ax.transAxes, ha='center', va='top',
                fontsize=self.font_size - 2, style='italic', color='#666666')
        self._attach_citation(fig, 'phase_fraction')
        return fig

    # ------------------------------------------------------------------
    # SUMMARY FIGURE
    # ------------------------------------------------------------------
    def create_summary_figure(self, results: Dict) -> plt.Figure:
        fig = plt.figure(figsize=(10, 8.3))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

        colors = self.colors['primary']

        # (A) Surface area
        ax1 = fig.add_subplot(gs[0, 0])
        if results.get('bet_results'):
            bet = results['bet_results']
            S_bet = bet.get('surface_area', 0)
            S_err = bet.get('surface_area_error', 0)
            ax1.bar(['SBET'], [S_bet], color=colors[0], alpha=0.75,
                    edgecolor='black', linewidth=1)
            ax1.errorbar(['SBET'], [S_bet], yerr=[S_err],
                         fmt='none', color='black', capsize=6, capthick=1.2)
            ax1.text(0, S_bet + S_err + (S_bet * 0.05),
                     f'{S_bet:.0f} +/- {S_err:.0f} m2/g',
                     ha='center', fontsize=self.font_size)
            ax1.set_ylabel('Surface area (m2/g)')
            ax1.set_title('(A) BET surface area', pad=8)
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_ylim(0, max(S_bet * 1.25, 1))
            ax1.text(0.02, 0.02, 'Rouquerol et al. (1994, 2007)',
                     transform=ax1.transAxes, fontsize=self.font_size - 3,
                     style='italic', color='#666666', va='bottom')
        else:
            ax1.text(0.5, 0.5, 'No BET data', transform=ax1.transAxes,
                     ha='center', va='center', fontsize=self.font_size)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title('(A) BET surface area', pad=8)

        # (B) PSD pie
        ax2 = fig.add_subplot(gs[0, 1])
        if (results.get('bet_results')
                and (results['bet_results'].get('psd_analysis') or {}).get('available')):
            psd = results['bet_results']['psd_analysis']
            fracs = [psd.get('micropore_fraction', 0),
                     psd.get('mesopore_fraction', 0),
                     psd.get('macropore_fraction', 0)]
            labels = ['Micropores', 'Mesopores', 'Macropores']
            pie_colors = [colors[0], colors[1], colors[2]]
            nz = [f > 0 for f in fracs]
            fracs_show = [f for f, m in zip(fracs, nz) if m]
            labels_show = [l for l, m in zip(labels, nz) if m]
            colors_show = [c for c, m in zip(pie_colors, nz) if m]
            if fracs_show:
                wedges, texts, autotexts = ax2.pie(
                    fracs_show, labels=labels_show, colors=colors_show,
                    autopct='%1.1f%%', startangle=90
                )
                for at in autotexts:
                    at.set_color('white')
                    at.set_fontsize(self.font_size - 1)
                ax2.set_title('(B) Pore size distribution', pad=8)
            else:
                ax2.text(0.5, 0.5, 'No porosity data', ha='center', va='center',
                         transform=ax2.transAxes)
                ax2.set_title('(B) Pore size distribution', pad=8)
        else:
            ax2.text(0.5, 0.5, 'No PSD data', ha='center', va='center',
                     transform=ax2.transAxes)
            ax2.set_title('(B) Pore size distribution', pad=8)

        # (C) Crystallinity gauge
        ax3 = fig.add_subplot(gs[1, 0])
        if results.get('xrd_results'):
            xrd_res = results['xrd_results']
            ci = float(xrd_res.get("crystallinity_index", 0.0))
            ci = max(0.0, min(ci, 1.0))
            theta = np.linspace(0, np.pi, 100)
            ax3.plot(theta, np.ones_like(theta), 'k-', linewidth=2)
            fill_theta = np.linspace(0, ci * np.pi, 100)
            ax3.fill_between(fill_theta, 0, 1, color=colors[0], alpha=0.7)
            ax3.plot([0, ci * np.pi], [0, 1], 'r-', linewidth=2)
            ax3.plot(ci * np.pi, 1, 'ro', markersize=8)
            ax3.text(0, -0.2, '0.0', ha='center', fontsize=self.font_size - 1)
            ax3.text(np.pi / 2, -0.2, '0.5', ha='center', fontsize=self.font_size - 1)
            ax3.text(np.pi, -0.2, '1.0', ha='center', fontsize=self.font_size - 1)
            ax3.text(0.5, 0.7, f'{ci:.2f}', transform=ax3.transAxes, ha='center',
                     fontsize=self.font_size + 2,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
            ax3.set_xlim(-0.2, np.pi + 0.2)
            ax3.set_ylim(-0.3, 1.2)
            ax3.set_aspect('equal')
            ax3.axis('off')
            ax3.set_title('(C) Crystallinity index', pad=8)
            ax3.text(0.5, -0.10, 'Ruland (1961)',
                     transform=ax3.transAxes, ha='center',
                     fontsize=self.font_size - 3, style='italic', color='#666666')
        else:
            ax3.text(0.5, 0.5, 'No XRD data', transform=ax3.transAxes,
                     ha='center', va='center', fontsize=self.font_size)
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.set_title('(C) Crystallinity index', pad=8)

        # (D) Classification
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        if results.get('fusion_results'):
            fusion = results['fusion_results']
            if fusion.get('valid', False):
                lines = [
                    f"Classification: {fusion.get('composite_classification', 'Unknown')}",
                    f"Family: {fusion.get('material_family', 'Porous Material')}",
                    f"Feature: {fusion.get('dominant_feature', '')}",
                    f"Confidence: {fusion.get('confidence_score', 0):.2f}"
                ]
                if 'suggested_applications' in fusion:
                    apps = fusion['suggested_applications'][:3]
                    if apps:
                        lines.append("")
                        lines.append("Key Applications:")
                        for app in apps:
                            lines.append(f"- {app}")
                ax4.text(0.5, 0.95, '\n'.join(lines),
                         transform=ax4.transAxes, fontsize=self.font_size,
                         va='top', ha='center',
                         bbox=dict(boxstyle='round', facecolor='wheat',
                                   alpha=0.85, pad=1))
                ax4.set_title('(D) Material classification', pad=8)
            else:
                ax4.text(0.5, 0.5, 'No fusion data', ha='center', va='center',
                         transform=ax4.transAxes)
                ax4.set_title('(D) Material classification', pad=8)
        else:
            ax4.text(0.5, 0.5, 'No fusion data', ha='center', va='center',
                     transform=ax4.transAxes)
            ax4.set_title('(D) Material classification', pad=8)

        plt.suptitle('BET-XRD Morphology Analysis Summary',
                     fontsize=self.font_size + 4, y=0.98)
        self._attach_citation(fig, 'summary', y=0.001)
        return fig

    # ------------------------------------------------------------------
    # MORPHOLOGY FIGURE (delegates to data-driven visualization)
    # ------------------------------------------------------------------
    def create_morphology_figure(self, bet_results: Dict,
                                 xrd_results: Optional[Dict] = None) -> plt.Figure:
        from morphology_visualizer import MorphologyVisualizer
        viz = MorphologyVisualizer()
        fig = viz.create_pore_structure_2d(bet_results, xrd_results)
        self._attach_citation(fig, 'morphology', y=0.001)
        return fig
