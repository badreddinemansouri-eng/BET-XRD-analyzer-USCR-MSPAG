"""
SCIENTIFIC PLOTTING ENGINE
========================================================================
Publication-quality plotting for journal submissions

Features:
- Multi-panel figures with proper labeling (A, B, C, ...)
- HD resolution (600+ DPI)
- Journal-style formatting
- Vector graphics support
- Consistent color schemes
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
# FONT SETTINGS
# ============================================================================
def set_publication_style(font_size=10, color_scheme='Nature'):
    """
    Set publication-style plotting parameters
    
    Parameters:
    -----------
    font_size : Font size in points
    color_scheme : Color scheme name
    """
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
        'figure.figsize': [7.2, 4.8],  # Standard journal width
        'figure.autolayout': False,
        'savefig.dpi': 600,
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

# ============================================================================
# PUBLICATION PLOTTER CLASS
# ============================================================================
class PublicationPlotter:
    """
    Publication-quality plotting for scientific figures
    """
    
    def __init__(self, color_scheme='Nature', font_size=10):
        """
        Initialize plotter
        
        Parameters:
        -----------
        color_scheme : Color scheme name
        font_size : Font size in points
        """
        self.color_scheme = color_scheme
        self.font_size = font_size
        self.colors = COLOR_SCHEMES[color_scheme]
        
        # Set style
        set_publication_style(font_size, color_scheme)
    
    def create_bet_figure(self, bet_raw: Dict, bet_results: Dict) -> plt.Figure:
        """
        Create comprehensive BET analysis figure
        
        Parameters:
        -----------
        bet_raw : Raw BET data dictionary
        bet_results : BET analysis results
        
        Returns:
        --------
        matplotlib Figure object
        """
        # Check if BET was successful
        bet_valid = bet_results.get('bet_valid', True)
        
        # Create figure with subplots
        if bet_valid:
            fig = plt.figure(figsize=(12, 10))
            gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        else:
            # Smaller figure if BET failed
            fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # Colors
        primary_color = self.colors['primary'][0]
        secondary_color = self.colors['primary'][1]
        
        # ====================================================================
        # SUBPLOT A: Adsorption isotherm (always show this)
        # ====================================================================
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Plot adsorption branch
        if 'p_ads' in bet_raw and 'q_ads' in bet_raw:
            p_ads = bet_raw['p_ads']
            q_ads = bet_raw['q_ads']
            ax1.plot(p_ads, q_ads, 'o-', color=primary_color, 
                    markersize=4, linewidth=1.5, label='Adsorption')
        
        # Plot desorption branch if available
        if ('p_des' in bet_raw and 'q_des' in bet_raw and 
            bet_raw['p_des'] is not None and len(bet_raw['p_des']) > 0):
            p_des = bet_raw['p_des']
            q_des = bet_raw['q_des']
            ax1.plot(p_des, q_des, 's--', color=secondary_color, 
                    markersize=3, linewidth=1.5, label='Desorption')
        
        ax1.set_xlabel('Relative Pressure (P/P₀)')
        ax1.set_ylabel('Quantity Adsorbed (cm³/g)')
        ax1.set_title('(A) Adsorption-Desorption Isotherm', pad=10)
        if 'p_des' in bet_raw and bet_raw['p_des'] is not None:
            ax1.legend(loc='best', frameon=True)
        ax1.grid(True, alpha=0.3)
        
        # Add note if BET failed
        if not bet_valid:
            ax1.text(0.5, 0.95, '⚠️ No valid BET range found',
                    transform=ax1.transAxes, fontsize=self.font_size-1,
                    verticalalignment='top', horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # ====================================================================
        # SUBPLOT B: BET plot (only if BET was successful)
        # ====================================================================
        if bet_valid:
            ax2 = fig.add_subplot(gs[0, 1])
            
            # Extract BET range data
            if 'bet_regression' in bet_results:
                bet_reg = bet_results['bet_regression']
                
                # Calculate BET transform for all points
                p_all = bet_raw['p_ads']
                q_all = bet_raw['q_ads']
                y_bet = p_all / (q_all * (1 - p_all))
                
                # Plot all points
                ax2.plot(p_all, y_bet, 'o', color='gray', 
                        markersize=3, alpha=0.5, label='All data')
                
                # Plot linear range
                p_min = bet_reg['p_min']
                p_max = bet_reg['p_max']
                mask = (p_all >= p_min) & (p_all <= p_max)
                
                if np.any(mask):
                    p_lin = p_all[mask]
                    y_lin = y_bet[mask]
                    ax2.plot(p_lin, y_lin, 'o', color=primary_color, 
                            markersize=5, label='Linear range')
                    
                    # Plot regression line
                    slope = bet_reg['slope']
                    intercept = bet_reg['intercept']
                    x_line = np.array([p_min, p_max])
                    y_line = slope * x_line + intercept
                    ax2.plot(x_line, y_line, '--', color='red', 
                            linewidth=2, label='Linear fit')
                
                # Add equation
                equation = f"y = {slope:.4f}x + {intercept:.4f}"
                r2_text = f"R² = {bet_reg['r_squared']:.6f}"
                ax2.text(0.05, 0.95, equation, transform=ax2.transAxes,
                        fontsize=self.font_size-1, verticalalignment='top')
                ax2.text(0.05, 0.88, r2_text, transform=ax2.transAxes,
                        fontsize=self.font_size-1, verticalalignment='top')
            
            ax2.set_xlabel('Relative Pressure (P/P₀)')
            ax2.set_ylabel('p/[n(1-p)] (g/cm³)')
            ax2.set_title('(B) BET Transform Plot', pad=10)
            ax2.legend(loc='best', frameon=True)
            ax2.grid(True, alpha=0.3)
        else:
            # Show message instead of BET plot
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.text(0.5, 0.5, 'BET Analysis Failed\n\nNo valid linear range found\nin 0.05-0.35 P/P₀ region',
                    transform=ax2.transAxes, ha='center', va='center',
                    fontsize=self.font_size)
            ax2.set_title('(B) BET Transform Plot', pad=10)
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        # ====================================================================
        # SUBPLOT C: t-plot (always try to show)
        # ====================================================================
        ax3 = fig.add_subplot(gs[0, 2])
        
        if 't_plot_analysis' in bet_results and bet_results['t_plot_analysis']['valid']:
            # Calculate t values (Harkins-Jura for N₂)
            p_ads = bet_raw['p_ads']
            t = (13.99 / (0.034 - np.log10(p_ads + 1e-10))) ** 0.5 * 0.1
            
            # Plot t-plot
            ax3.plot(t, bet_raw['q_ads'], 'o', color=primary_color, 
                    markersize=4, alpha=0.7)
            
            # Add linear fit for selected range
            t_plot = bet_results['t_plot_analysis']
            if t_plot['t_range'][1] > 0:
                # Select points in t-range
                mask = (t >= t_plot['t_range'][0]) & (t <= t_plot['t_range'][1])
                if np.sum(mask) >= 3:
                    t_selected = t[mask]
                    q_selected = bet_raw['q_ads'][mask]
                    
                    # Fit line
                    coeffs = np.polyfit(t_selected, q_selected, 1)
                    t_fit = np.linspace(t_selected.min(), t_selected.max(), 100)
                    q_fit = np.polyval(coeffs, t_fit)
                    
                    ax3.plot(t_fit, q_fit, '--', color='red', linewidth=2)
                    
                    # Add results
                    results_text = (f"V_micro = {t_plot['micropore_volume']:.3f} cm³/g\n"
                                  f"S_ext = {t_plot['external_surface']:.0f} m²/g")
                    ax3.text(0.05, 0.95, results_text, transform=ax3.transAxes,
                            fontsize=self.font_size-2, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax3.set_xlabel('Statistical Thickness, t (nm)')
        ax3.set_ylabel('Quantity Adsorbed (cm³/g)')
        ax3.set_title('(C) t-Plot Analysis', pad=10)
        ax3.grid(True, alpha=0.3)
        
        # ====================================================================
        # SUBPLOT D: Pore size distribution (if available)
        # ====================================================================
        row_idx = 1 if bet_valid else 0
        ax4 = fig.add_subplot(gs[row_idx, :2])
        
        if ('psd_analysis' in bet_results and 
            bet_results['psd_analysis']['available']):
            psd = bet_results['psd_analysis']
            
            # Plot PSD
            ax4.plot(psd['pore_diameters'], psd['dv_dlogd'], 
                    '-', color=primary_color, linewidth=2)
            
            ax4.fill_between(psd['pore_diameters'], 0, psd['dv_dlogd'],
                           alpha=0.3, color=primary_color)
            
            # Mark peak diameter
            if psd['peak_pore_diameter'] > 0:
                ax4.axvline(psd['peak_pore_diameter'], color='red', 
                          linestyle='--', linewidth=1, alpha=0.7)
                ax4.text(psd['peak_pore_diameter'] * 1.1, 
                        max(psd['dv_dlogd']) * 0.8,
                        f"Peak: {psd['peak_pore_diameter']:.1f} nm",
                        fontsize=self.font_size-1)
            
            # Add pore type regions
            ax4.axvspan(0, 2, alpha=0.1, color='blue', label='Micropores')
            ax4.axvspan(2, 50, alpha=0.1, color='green', label='Mesopores')
            ax4.axvspan(50, 200, alpha=0.1, color='red', label='Macropores')
            
            # Add pore fractions if available
            if all(k in psd for k in ['micropore_fraction', 'mesopore_fraction', 'macropore_fraction']):
                frac_text = (f"Micro: {psd['micropore_fraction']*100:.1f}%\n"
                           f"Meso: {psd['mesopore_fraction']*100:.1f}%\n"
                           f"Macro: {psd['macropore_fraction']*100:.1f}%")
                ax4.text(0.95, 0.95, frac_text, transform=ax4.transAxes,
                        fontsize=self.font_size-2, verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax4.set_xlabel('Pore Diameter (nm)')
        ax4.set_ylabel('dV/dlogD (cm³/g)')
        ax4.set_title('(D) Pore Size Distribution (BJH)', pad=10)
        ax4.set_xscale('log')
        ax4.set_xlim(0.5, 200)
        ax4.legend(loc='upper right', frameon=True)
        ax4.grid(True, alpha=0.3, which='both')
        
        # ====================================================================
        # SUBPLOT E: Hysteresis analysis (if available)
        # ====================================================================
        ax5 = fig.add_subplot(gs[row_idx, 2])
        
        if ('hysteresis_analysis' in bet_results and 
            bet_raw.get('p_des') is not None and 
            len(bet_raw['p_des']) > 0):
            
            # Plot hysteresis loop
            ax5.plot(bet_raw['p_ads'], bet_raw['q_ads'], 'o-', 
                    color=primary_color, markersize=3, linewidth=1.5, 
                    label='Adsorption')
            ax5.plot(bet_raw['p_des'], bet_raw['q_des'], 's--', 
                    color=secondary_color, markersize=3, linewidth=1.5, 
                    label='Desorption')
            
            # Fill hysteresis loop if we have matching points
            try:
                if len(bet_raw['p_ads']) == len(bet_raw['p_des']):
                    ax5.fill_between(bet_raw['p_ads'], bet_raw['q_ads'], bet_raw['q_des'],
                                   alpha=0.2, color='gray')
            except:
                pass
            
            # Add hysteresis info
            hyst = bet_results['hysteresis_analysis']
            if hyst['valid']:
                info_text = (f"Type: {hyst['type']}\n"
                           f"IUPAC: {hyst['iupac_class']}\n"
                           f"Area: {hyst['loop_area']:.2f}")
                ax5.text(0.05, 0.95, info_text, transform=ax5.transAxes,
                        fontsize=self.font_size-2, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax5.set_xlabel('Relative Pressure (P/P₀)')
        ax5.set_ylabel('Quantity Adsorbed (cm³/g)')
        ax5.set_title('(E) Hysteresis Loop Analysis', pad=10)
        if 'p_des' in bet_raw and bet_raw['p_des'] is not None:
            ax5.legend(loc='best', frameon=True)
        ax5.grid(True, alpha=0.3)
        
        # ====================================================================
        # SUBPLOT F: Summary table (only if BET was successful)
        # ====================================================================
        if bet_valid:
            ax6 = fig.add_subplot(gs[2, :])
            ax6.axis('tight')
            ax6.axis('off')
            
            # Create summary table
            summary_data = []
            
            # Surface area
            S_bet = bet_results.get('surface_area', 0)
            S_err = bet_results.get('surface_area_error', 0)
            summary_data.append(['Surface Area (Sᴮᴱᵀ)', f'{S_bet:.1f} ± {S_err:.1f} m²/g'])
            
            # Pore volume
            V_total = bet_results.get('total_pore_volume', 0)
            summary_data.append(['Total Pore Volume', f'{V_total:.3f} cm³/g'])
            
            # Micropore volume
            V_micro = bet_results.get('micropore_volume', 0)
            summary_data.append(['Micropore Volume (t-plot)', f'{V_micro:.3f} cm³/g'])
            
            # Mean pore diameter
            d_mean = bet_results.get('mean_pore_diameter', 0)
            summary_data.append(['Mean Pore Diameter', f'{d_mean:.2f} nm'])
            
            # C constant
            C = bet_results.get('c_constant', 0)
            summary_data.append(['BET C Constant', f'{C:.0f}'])
            
            # Hysteresis
            hyst_type = bet_results.get('hysteresis_analysis', {}).get('type', 'N/A')
            summary_data.append(['Hysteresis Type', hyst_type])
            
            # Create table
            table = ax6.table(cellText=summary_data,
                             colLabels=['Parameter', 'Value'],
                             colWidths=[0.4, 0.3],
                             cellLoc='left',
                             loc='center')
            
            # Style table
            table.auto_set_font_size(False)
            table.set_fontsize(self.font_size - 1)
            table.scale(1, 1.5)
            
            # Style header
            for (row, col), cell in table.get_celld().items():
                if row == 0:  # Header row
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor(self.colors['primary'][0])
                    cell.set_text_props(color='white')
                else:
                    if row % 2 == 0:
                        cell.set_facecolor('#F5F5F5')
                    else:
                        cell.set_facecolor('#FFFFFF')
            
            ax6.set_title('(F) BET Analysis Summary', pad=20)
        
        # ====================================================================
        # FINAL TOUCHES
        # ====================================================================
        if bet_valid:
            plt.suptitle('BET Surface Area and Porosity Analysis', 
                        fontsize=self.font_size + 4, y=0.98)
        else:
            plt.suptitle('Porosity Analysis (BET Failed - Non-linear in 0.05-0.35 P/P₀)', 
                        fontsize=self.font_size + 4, y=0.98)
        
        return fig
    
    def create_xrd_figure(self, xrd_raw: Dict, xrd_results: Dict) -> plt.Figure:
        """
        Create scientifically consistent XRD analysis figure
        (Nanomaterial-safe, reviewer-grade)
        """
    
        # ===============================
        # SAFE UNWRAP
        # ===============================
        if "xrd_results" in xrd_results:
            xrd_results = xrd_results["xrd_results"]
    
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
    
        primary_color = self.colors['primary'][0]
    
        two_theta = np.array(xrd_raw['two_theta'])
        intensity = np.array(xrd_raw['intensity'])
    
        # ============================================================
        # STRICT PEAK SEMANTICS
        # ============================================================
        raw_peaks = xrd_results.get("raw_peaks", [])
        structural_peaks = xrd_results.get("structural_peaks", [])
    
        n_raw = len(raw_peaks)
        n_structural = len(structural_peaks)
    
        # UI subset (display only)
        display_peaks = sorted(
            structural_peaks,
            key=lambda p: p["intensity"],
            reverse=True
        )[:8]
    
        # ============================================================
        # (A) XRD PATTERN
        # ============================================================
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(two_theta, intensity, '-', color=primary_color, lw=1.5)
    
        if display_peaks:
            ax1.scatter(
                [p["position"] for p in display_peaks],
                [p["intensity"] for p in display_peaks],
                color="red", s=35, zorder=5, label="Structural Bragg peaks"
            )
    
            for p in display_peaks:
                ax1.annotate(
                    f"{p['position']:.2f}°",
                    xy=(p["position"], p["intensity"]),
                    xytext=(p["position"], p["intensity"] * 1.1),
                    ha="center", fontsize=self.font_size - 2,
                    arrowprops=dict(arrowstyle="->", lw=0.5)
                )
    
        # Annotation — explicit and honest
        ax1.text(
            0.02, 0.95,
            f"Structural Bragg peaks: {n_structural}\n"
            f"Detected local maxima: {n_raw}",
            transform=ax1.transAxes,
            fontsize=self.font_size - 1,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        )
    
        ci = xrd_results.get("crystallinity_index", 0)
        ax1.text(
            0.98, 0.95, f"Crystallinity Index: {ci:.2f}",
            transform=ax1.transAxes, ha="right",
            fontsize=self.font_size,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8)
        )
    
        ax1.set_xlabel("2θ (degrees)")
        ax1.set_ylabel("Intensity (a.u.)")
        ax1.set_title("(A) XRD Pattern – Structural Peaks Only")
        ax1.grid(alpha=0.3)
        ax1.legend()
    
        # ============================================================
        # (B) WILLIAMSON–HALL
        # ============================================================
        ax2 = fig.add_subplot(gs[1, 0])
        wh = xrd_results.get("williamson_hall")
    
        if isinstance(wh, dict) and len(wh.get("x_data", [])) >= 3:
            ax2.scatter(wh["x_data"], wh["y_data"], s=40)
            xfit = np.linspace(min(wh["x_data"]), max(wh["x_data"]), 100)
            ax2.plot(xfit, wh["slope"] * xfit + wh["intercept"], "--r")
            ax2.set_title("(B) Williamson–Hall Plot")
            ax2.grid(alpha=0.3)
        else:
            ax2.text(
                0.5, 0.5,
                "Williamson–Hall not valid\n(insufficient independent peaks)",
                ha="center", va="center"
            )
            ax2.set_xticks([])
            ax2.set_yticks([])
    
        # ============================================================
        # (C) SIZE DISTRIBUTION (STRUCTURAL ONLY)
        # ============================================================
        ax3 = fig.add_subplot(gs[1, 1])
        sizes = [p["crystallite_size"] for p in structural_peaks if p.get("crystallite_size", 0) > 0]
    
        if len(sizes) >= 3:
            ax3.hist(sizes, bins=min(10, len(sizes)), alpha=0.7)
            ax3.set_title("(C) Crystallite Size Distribution")
            ax3.set_xlabel("Size (nm)")
            ax3.set_ylabel("Frequency")
            ax3.grid(alpha=0.3, axis="y")
        else:
            ax3.text(0.5, 0.5, "Insufficient data for size distribution",
                     ha="center", va="center")
            ax3.set_xticks([])
            ax3.set_yticks([])
    
        # ============================================================
        # (D) PEAK TABLE (STRUCTURAL)
        # ============================================================
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.axis("off")
    
        if display_peaks:
            table_data = [[
                i + 1,
                f"{p['position']:.2f}",
                f"{p.get('d_spacing', 0):.3f}",
                f"{p['fwhm_deg']:.3f}",
                f"{p.get('crystallite_size', 0):.1f}",
                p.get("hkl", "")
            ] for i, p in enumerate(display_peaks)]
    
            table = ax4.table(
                cellText=table_data,
                colLabels=["#", "2θ (°)", "d (Å)", "FWHM (°)", "Size (nm)", "HKL"],
                loc="center", cellLoc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(self.font_size - 1)
            table.scale(1, 1.2)
            ax4.set_title("(D) Structural Bragg Peaks")
    
        # ============================================================
        # (E) SUMMARY (NO MESOPORES!)
        # ============================================================
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis("off")
    
        summary = [
            ["Structural Bragg Peaks", str(n_structural)],
            ["Detected Local Maxima", str(n_raw)],
            ["Crystallinity Index", f"{ci:.2f}"],
            ["Scherrer Size", f"{xrd_results.get('crystallite_size', {}).get('scherrer', 0):.1f} nm"]
        ]
    
        table = ax5.table(
            cellText=summary,
            colLabels=["Parameter", "Value"],
            loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(self.font_size - 1)
        table.scale(1, 1.2)
        ax5.set_title("(E) XRD Analysis Summary")
    
        plt.suptitle("X-ray Diffraction Analysis (Nanomaterial-Validated)", fontsize=self.font_size + 4)
        return fig

    def create_phase_fraction_plot(self, phase_fractions):
        """
        Bar chart of phase fractions (CIF-validated only)
        """
        fig, ax = plt.subplots(figsize=(6, 4))
    
        if not phase_fractions:
            ax.text(0.5, 0.5, "No phase fraction data",
                    ha="center", va="center")
            return fig
    
        phases = [p["phase"] for p in phase_fractions]
        fractions = [p["fraction"] for p in phase_fractions]
    
        bars = ax.bar(phases, fractions, color=self.colors["primary"])
    
        for bar, frac in zip(bars, fractions):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f"{frac:.1f}%",
                    ha="center", va="bottom",
                    fontsize=self.font_size - 1)
    
        ax.set_ylabel("Phase fraction (%)")
        ax.set_title("Phase Composition (Semi-quantitative)")
        ax.set_ylim(0, max(fractions) * 1.2)
        ax.grid(True, axis="y", alpha=0.3)
    
        return fig

    def create_summary_figure(self, results: Dict) -> plt.Figure:
        """
        Create summary figure combining key results
        
        Parameters:
        -----------
        results : Complete analysis results dictionary
        
        Returns:
        --------
        matplotlib Figure object
        """
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
        
        # Colors
        colors = self.colors['primary']
        
        # ====================================================================
        # SUBPLOT A: BET Surface Area
        # ====================================================================
        ax1 = fig.add_subplot(gs[0, 0])
        
        if results.get('bet_results'):
            bet = results['bet_results']
            
            # Create bar for surface area with error
            S_bet = bet['surface_area']
            S_err = bet['surface_area_error']
            
            bars = ax1.bar(['Sᴮᴱᵀ'], [S_bet], 
                          color=colors[0], alpha=0.7,
                          edgecolor='black', linewidth=1)
            
            # Add error bars
            ax1.errorbar(['Sᴮᴱᵀ'], [S_bet], yerr=[S_err],
                        fmt='none', color='black', capsize=5, capthick=1)
            
            # Add value on top of bar
            ax1.text(0, S_bet + S_err + (S_bet * 0.05), 
                    f'{S_bet:.0f} ± {S_err:.0f} m²/g',
                    ha='center', fontsize=self.font_size)
            
            ax1.set_ylabel('Surface Area (m²/g)')
            ax1.set_title('(A) BET Surface Area', pad=10)
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_ylim(0, S_bet * 1.2)
        
        else:
            ax1.text(0.5, 0.5, 'No BET data',
                    transform=ax1.transAxes, ha='center', va='center',
                    fontsize=self.font_size)
            ax1.set_title('(A) BET Surface Area', pad=10)
            ax1.set_xticks([])
            ax1.set_yticks([])
        
        # ====================================================================
        # SUBPLOT B: Porosity Pie Chart
        # ====================================================================
        ax2 = fig.add_subplot(gs[0, 1])
        
        if results.get('bet_results') and 'psd_analysis' in results['bet_results']:
            psd = results['bet_results']['psd_analysis']
            
            if psd['available']:
                # Create pie chart of pore fractions
                fractions = [
                    psd['micropore_fraction'],
                    psd['mesopore_fraction'],
                    psd['macropore_fraction']
                ]
                
                labels = ['Micropores', 'Mesopores', 'Macropores']
                pie_colors = [colors[0], colors[1], colors[2]]
                
                # Only show non-zero fractions
                non_zero = [f > 0 for f in fractions]
                fractions_show = [f for f, nz in zip(fractions, non_zero) if nz]
                labels_show = [l for l, nz in zip(labels, non_zero) if nz]
                colors_show = [c for c, nz in zip(pie_colors, non_zero) if nz]
                
                if fractions_show:
                    wedges, texts, autotexts = ax2.pie(fractions_show,
                                                      labels=labels_show,
                                                      colors=colors_show,
                                                      autopct='%1.1f%%',
                                                      startangle=90)
                    
                    # Style percentage text
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontsize(self.font_size - 1)
                    
                    ax2.set_title('(B) Pore Size Distribution', pad=10)
                else:
                    ax2.text(0.5, 0.5, 'No porosity data',
                            transform=ax2.transAxes, ha='center', va='center',
                            fontsize=self.font_size)
                    ax2.set_title('(B) Pore Size Distribution', pad=10)
            else:
                ax2.text(0.5, 0.5, 'No PSD data',
                        transform=ax2.transAxes, ha='center', va='center',
                        fontsize=self.font_size)
                ax2.set_title('(B) Pore Size Distribution', pad=10)
        else:
            ax2.text(0.5, 0.5, 'No BET data',
                    transform=ax2.transAxes, ha='center', va='center',
                    fontsize=self.font_size)
            ax2.set_title('(B) Pore Size Distribution', pad=10)
        
        # ====================================================================
        # SUBPLOT C: Crystallinity
        # ====================================================================
        ax3 = fig.add_subplot(gs[1, 0])
        
        if results.get('xrd_results'):
            xrd_res = results['xrd_results']
            # SAFE extraction of crystallinity
            crystallinity = float(xrd_res.get("crystallinity_index", 0.0))
            crystallinity = max(0.0, min(crystallinity, 1.0))  # clamp to [0,1]
                        
            # Create gauge chart for crystallinity
            xrd_res.get('crystallinity_index', 0.0)
            
            # Draw gauge
            theta = np.linspace(0, np.pi, 100)
            r = 1
            
            # Background
            ax3.plot(theta, np.ones_like(theta) * r, 'k-', linewidth=2)
            
            # Fill based on crystallinity
            fill_theta = np.linspace(0, crystallinity * np.pi, 100)
            ax3.fill_between(fill_theta, 0, r, color=colors[0], alpha=0.7)
            
            # Add needle
            needle_theta = crystallinity * np.pi
            ax3.plot([0, needle_theta], [0, r], 'r-', linewidth=2)
            ax3.plot(needle_theta, r, 'ro', markersize=8)
            
            # Add labels
            ax3.text(0, -0.2, '0.0', ha='center', fontsize=self.font_size-1)
            ax3.text(np.pi/2, -0.2, '0.5', ha='center', fontsize=self.font_size-1)
            ax3.text(np.pi, -0.2, '1.0', ha='center', fontsize=self.font_size-1)
            
            # Add value
            ax3.text(0.5, 0.7, f'{crystallinity:.2f}',
                    transform=ax3.transAxes, ha='center', fontsize=self.font_size+2,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax3.set_xlim(-0.2, np.pi + 0.2)
            ax3.set_ylim(-0.3, 1.2)
            ax3.set_aspect('equal')
            ax3.axis('off')
            ax3.set_title('(C) Crystallinity Index', pad=10)
        
        else:
            ax3.text(0.5, 0.5, 'No XRD data',
                    transform=ax3.transAxes, ha='center', va='center',
                    fontsize=self.font_size)
            ax3.set_title('(C) Crystallinity Index', pad=10)
        
        # ====================================================================
        # SUBPLOT D: Material Classification
        # ====================================================================
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('tight')
        ax4.axis('off')
        
        if results.get('fusion_results'):
            fusion = results['fusion_results']
            
            if fusion['valid']:
                # Create classification display
                classification = fusion.get('composite_classification', 'Unknown')
                material_family = fusion.get('material_family', 'Porous Material')
                dominant_feature = fusion.get('dominant_feature', '')
                confidence = fusion.get('confidence_score', 0)
                
                info_text = [
                    f"Classification: {classification}",
                    f"Family: {material_family}",
                    f"Feature: {dominant_feature}",
                    f"Confidence: {confidence:.2f}"
                ]
                
                # Add key applications if available
                if 'suggested_applications' in fusion:
                    apps = fusion['suggested_applications'][:3]  # First 3
                    if apps:
                        info_text.append("\nKey Applications:")
                        for app in apps:
                            info_text.append(f"• {app}")
                
                # Create text box
                ax4.text(0.5, 0.95, '\n'.join(info_text),
                        transform=ax4.transAxes,
                        fontsize=self.font_size,
                        verticalalignment='top',
                        horizontalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8,
                                 pad=1))
                
                ax4.set_title('(D) Material Classification', pad=10)
            
            else:
                ax4.text(0.5, 0.5, 'No fusion data',
                        transform=ax4.transAxes, ha='center', va='center',
                        fontsize=self.font_size)
                ax4.set_title('(D) Material Classification', pad=10)
        else:
            ax4.text(0.5, 0.5, 'No fusion data',
                    transform=ax4.transAxes, ha='center', va='center',
                    fontsize=self.font_size)
            ax4.set_title('(D) Material Classification', pad=10)
        
        # ====================================================================
        # FINAL TOUCHES
        # ====================================================================
        plt.suptitle('BET-XRD Morphology Analysis Summary', 
                    fontsize=self.font_size + 4, y=0.98)
        

        return fig





















