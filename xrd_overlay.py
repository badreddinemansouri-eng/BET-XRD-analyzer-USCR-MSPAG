"""
XRD MULTI-PATTERN OVERLAY AND WATERFALL PLOTTING
========================================================================
Publication-quality rendering of multiple XRD patterns on a single
figure, for use in journal articles.

Two display modes:
    - Overlay   : all patterns share one intensity axis, stacked with
                  shared y-scale (classic comparison figure).
    - Waterfall : each pattern is offset vertically for clarity.

Both modes support:
    - Intensity normalization (max = 1) per pattern
    - Optional peak markers with HKL labels
    - Two x-axes: 2theta (bottom) and d-spacing (top)
    - Journal color schemes from scientific_plots.COLOR_SCHEMES
    - 600 DPI or vector export

References:
    Cullity, B. D., & Stock, S. R. (2001).
    Elements of X-Ray Diffraction, 3rd ed., Prentice Hall.
========================================================================
"""

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

logger = logging.getLogger(__name__)


DEFAULT_JOURNAL_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#003f5c', '#bc5090', '#ff6361', '#58508d',
]


def two_theta_to_d(two_theta_deg, wavelength=1.5406):
    """Convert 2theta (deg) to d-spacing (Angstrom)."""
    two_theta_rad = np.radians(np.asarray(two_theta_deg, dtype=float))
    sin_theta = np.sin(two_theta_rad / 2.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        d = wavelength / (2.0 * sin_theta)
    d[~np.isfinite(d)] = np.nan
    return d


def d_to_two_theta(d_angstrom, wavelength=1.5406):
    """Convert d-spacing (Angstrom) to 2theta (deg)."""
    d = np.asarray(d_angstrom, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        arg = np.clip(wavelength / (2.0 * d), -1.0, 1.0)
        two_theta = 2.0 * np.degrees(np.arcsin(arg))
    return two_theta


def _normalize_intensity(intensity):
    """Normalize intensity to max = 1. Handles negatives and zeros."""
    arr = np.asarray(intensity, dtype=float)
    arr = np.clip(arr, 0.0, None)
    m = np.max(arr) if arr.size else 0.0
    if m <= 0:
        return arr
    return arr / m


class XROverlayPlotter:
    """
    Render multiple XRD patterns on a single publication-quality figure.

    Parameters
    ----------
    color_scheme : str
        Name of a palette. Ignored if `colors` is provided.
    font_size : int
        Base font size in points.
    colors : list of str, optional
        Explicit colors. If provided, overrides `color_scheme`.
    """

    def __init__(self, color_scheme: str = 'Nature',
                 font_size: int = 10,
                 colors: Optional[Sequence[str]] = None):
        self.color_scheme = color_scheme
        self.font_size = font_size

        if colors is not None:
            self.colors = list(colors)
        else:
            try:
                from scientific_plots import COLOR_SCHEMES
                scheme = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES.get('Nature'))
                self.colors = list(scheme['primary']) + list(scheme['secondary'])
            except Exception:
                self.colors = list(DEFAULT_JOURNAL_COLORS)

        self._set_style()

    def _set_style(self):
        plt.rcParams.update({
            'font.size': self.font_size,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'axes.titlesize': self.font_size + 2,
            'axes.labelsize': self.font_size,
            'axes.linewidth': 0.9,
            'xtick.labelsize': self.font_size - 1,
            'ytick.labelsize': self.font_size - 1,
            'legend.fontsize': self.font_size - 1,
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'legend.edgecolor': 'black',
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'figure.dpi': 300,
            'savefig.dpi': 600,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plot(self,
             patterns: List[Dict],
             mode: str = "waterfall",
             normalize: bool = True,
             offset_step: float = 1.15,
             wavelength: float = 1.5406,
             xlim: Optional[tuple] = None,
             ylim: Optional[tuple] = None,
             show_legend: bool = True,
             show_peaks: bool = False,
             title: str = "",
             xlabel: str = "2\u03b8 (degrees)",
             ylabel: str = "Normalized intensity (a.u.)",
             figsize: tuple = (10.0, 6.5),
             show_d_spacing_axis: bool = False) -> plt.Figure:
        """
        Parameters
        ----------
        patterns : list of dict
            Each dict must contain:
                - 'two_theta' : array-like, 2theta in degrees
                - 'intensity' : array-like
            Optional keys:
                - 'label'   : str, appears in legend
                - 'color'   : str, overrides palette
                - 'peaks'   : list of {'position': float, 'hkl': str}
        mode : 'overlay' or 'waterfall'
        normalize : bool
            Normalize each pattern to max = 1 before display.
        offset_step : float
            Vertical spacing between patterns in waterfall mode.
        wavelength : float
            Used for the top d-spacing axis.
        """
        if not patterns:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No patterns provided",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return fig

        mode = mode.lower().strip()
        if mode not in ("overlay", "waterfall"):
            mode = "waterfall"

        fig, ax = plt.subplots(figsize=figsize)

        n = len(patterns)
        all_two_theta = []
        all_intensity_display = []

        for i, pat in enumerate(patterns):
            tt = np.asarray(pat.get('two_theta', []), dtype=float)
            ii = np.asarray(pat.get('intensity', []), dtype=float)
            if tt.size == 0 or ii.size == 0 or tt.size != ii.size:
                continue

            # Sort by 2theta
            order = np.argsort(tt)
            tt = tt[order]
            ii = ii[order]

            if normalize:
                ii = _normalize_intensity(ii)

            color = pat.get('color', self.colors[i % len(self.colors)])
            label = pat.get('label', f"Pattern {i + 1}")

            if mode == "waterfall":
                offset = (n - 1 - i) * offset_step
                y_display = ii + offset
                ax.plot(tt, y_display, '-', color=color, linewidth=1.4, label=label)
                ax.fill_between(tt, offset, y_display, color=color, alpha=0.10)
            else:
                y_display = ii
                ax.plot(tt, y_display, '-', color=color, linewidth=1.4, label=label)

            all_two_theta.append(tt)
            all_intensity_display.append(y_display)

            if show_peaks and pat.get('peaks'):
                for pk in pat['peaks']:
                    pos = pk.get('position')
                    hkl = pk.get('hkl', '')
                    if pos is None:
                        continue
                    idx = int(np.argmin(np.abs(tt - pos)))
                    y_pk = y_display[idx]
                    ax.plot([pos], [y_pk], 'v', color=color,
                            markersize=6, markeredgecolor='black',
                            markeredgewidth=0.4, zorder=5)
                    if hkl:
                        ax.text(pos, y_pk + 0.03 * (n if mode == "waterfall" else 1),
                                hkl, ha='center', va='bottom',
                                fontsize=self.font_size - 3,
                                color='black', rotation=45)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, pad=10)

        ax.grid(True, which='major', alpha=0.35, linewidth=0.5)
        ax.grid(True, which='minor', alpha=0.15, linewidth=0.3)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        if xlim is not None:
            ax.set_xlim(xlim)
        elif all_two_theta:
            xmin = min(arr.min() for arr in all_two_theta)
            xmax = max(arr.max() for arr in all_two_theta)
            ax.set_xlim(xmin, xmax)

        if ylim is not None:
            ax.set_ylim(ylim)

        if show_legend and n > 1:
            ax.legend(loc='upper right', frameon=True, ncol=1)

        # Optional top axis showing d-spacing
        if show_d_spacing_axis:
            self._attach_d_spacing_axis(ax, wavelength)

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    def plot_overlay(self, patterns, **kwargs):
        """Shortcut for overlay mode."""
        kwargs['mode'] = "overlay"
        return self.plot(patterns, **kwargs)

    def plot_waterfall(self, patterns, **kwargs):
        """Shortcut for waterfall mode."""
        kwargs['mode'] = "waterfall"
        return self.plot(patterns, **kwargs)

    def combine_multiple(self,
                         pattern_sets: Dict[str, List[Dict]],
                         wavelength: float = 1.5406,
                         figsize: tuple = (12.0, 8.0),
                         mode: str = "waterfall") -> plt.Figure:
        """
        Create a multi-panel figure, one panel per set of patterns.

        Parameters
        ----------
        pattern_sets : dict
            Mapping {panel_title: [pattern_dict, ...]}
        """
        n = max(len(pattern_sets), 1)
        ncols = 1 if n == 1 else 2
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                 squeeze=False)

        for idx, (panel_title, patterns) in enumerate(pattern_sets.items()):
            r = idx // ncols
            c = idx % ncols
            ax = axes[r][c]

            temp_fig = self.plot(patterns, mode=mode, wavelength=wavelength,
                                 show_legend=True, title=panel_title)
            temp_ax = temp_fig.axes[0]

            # Transfer lines to target axes
            for line in temp_ax.get_lines():
                x = line.get_xdata()
                y = line.get_ydata()
                ax.plot(x, y, color=line.get_color(),
                        linewidth=line.get_linewidth(),
                        label=line.get_label())
            for coll in temp_ax.collections:
                try:
                    paths = coll.get_paths()
                    for path in paths:
                        verts = path.vertices
                        ax.fill(verts[:, 0], verts[:, 1],
                                color=coll.get_facecolor()[0],
                                alpha=0.10)
                except Exception:
                    pass

            ax.set_xlabel(temp_ax.get_xlabel())
            ax.set_ylabel(temp_ax.get_ylabel())
            ax.set_title(temp_ax.get_title())
            ax.grid(True, alpha=0.35)
            ax.legend(loc='upper right')

            plt.close(temp_fig)

        # Turn off unused axes
        for idx in range(len(pattern_sets), nrows * ncols):
            r = idx // ncols
            c = idx % ncols
            axes[r][c].axis('off')

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    def _attach_d_spacing_axis(self, ax, wavelength):
        """Add a top axis showing d-spacing for the visible 2theta range."""
        xmin, xmax = ax.get_xlim()
        d_at_xmin = two_theta_to_d(np.array([xmax]), wavelength)[0]
        d_at_xmax = two_theta_to_d(np.array([xmin]), wavelength)[0]

        if not (np.isfinite(d_at_xmin) and np.isfinite(d_at_xmax)):
            return

        top = ax.twiny()
        top.set_xlim(ax.get_xlim())

        # Choose a small set of nice ticks in the visible d range
        d_min = min(d_at_xmin, d_at_xmax)
        d_max = max(d_at_xmin, d_at_xmax)

        if d_min <= 0 or d_max <= d_min:
            return

        candidates = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0,
                      5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0]
        d_ticks = [d for d in candidates if d_min <= d <= d_max]
        if not d_ticks:
            return

        tt_ticks = d_to_two_theta(np.array(d_ticks), wavelength)
        top.set_xticks(tt_ticks)
        top.set_xticklabels([f"{d:.2f}" for d in d_ticks])
        top.set_xlabel("d-spacing (\u00c5)", fontsize=self.font_size - 1, labelpad=6)
        top.tick_params(axis='x', labelsize=self.font_size - 2)

    # ------------------------------------------------------------------
    @staticmethod
    def build_pattern_record(label: str,
                             two_theta,
                             intensity,
                             color: Optional[str] = None,
                             peaks: Optional[list] = None) -> Dict:
        """Helper to build a pattern dict."""
        rec = {
            'label': label,
            'two_theta': np.asarray(two_theta, dtype=float),
            'intensity': np.asarray(intensity, dtype=float),
        }
        if color:
            rec['color'] = color
        if peaks:
            rec['peaks'] = peaks
        return rec
