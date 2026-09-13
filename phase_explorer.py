"""
PHASE EXPLORER - PER-PHASE ELEMENT SEARCH AND REFERENCE COMPARISON
========================================================================
Manual phase identification workflow.

For each phase the user picks the constituent elements. The module:
    1. Searches OPTIMADE providers for structures whose element set
       matches the user selection exactly.
    2. Simulates the reference powder XRD pattern from the CIF.
    3. Computes a peak-position match fraction against the experimental
       pattern.
    4. Draws all reference patterns as stick patterns below the
       experimental curve, HighScore / Match! style.

The comparison is qualitative. Quantitative phase fractions require a
full Rietveld refinement, which is out of scope.

References:
    Bragg, W. L. (1913). Proc. R. Soc. A, 89, 248-277.
    Andersen, C. W. et al. (2021). OPTIMADE, an open standard.
    Grazulis, S. et al. (2012). Nucleic Acids Res., 40, D420-D427 (COD).
========================================================================
"""

import io
import re
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)


OPTIMADE_PROVIDERS = [
    ("MaterialsCloud", "https://optimade.materialscloud.org/v1/structures"),
    ("OQMD", "https://oqmd.org/optimade/v1/structures"),
    ("NOMAD", "https://nomad-lab.eu/prod/rae/optimade/v1/structures"),
    ("MP-OPTIMADE", "https://optimade.materialsproject.org/v1/structures"),
]

PHASE_COLORS = [
    '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
    '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f',
]


# ============================================================================
# FORMULA PARSING
# ============================================================================
def parse_formula_elements(formula: str) -> List[str]:
    """Return unique element symbols from a chemical formula string."""
    if not formula:
        return []
    symbols = re.findall(r'([A-Z][a-z]?)', formula)
    seen = set()
    out = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ============================================================================
# OPTIMADE SEARCH WITH EXACT ELEMENT SET
# ============================================================================
def _build_cif_from_optimade(attrs: dict) -> Optional[str]:
    """Build a P1 CIF string from OPTIMADE structure attributes."""
    try:
        lattice_vectors = attrs.get("lattice_vectors")
        species = attrs.get("species_at_sites")
        positions = attrs.get("cartesian_site_positions")

        if not lattice_vectors or not species or not positions:
            return None

        a_vec = np.array(lattice_vectors[0], dtype=float)
        b_vec = np.array(lattice_vectors[1], dtype=float)
        c_vec = np.array(lattice_vectors[2], dtype=float)

        a = float(np.linalg.norm(a_vec))
        b = float(np.linalg.norm(b_vec))
        c = float(np.linalg.norm(c_vec))
        if a <= 0 or b <= 0 or c <= 0:
            return None

        alpha = float(np.degrees(np.arccos(np.clip(
            np.dot(b_vec, c_vec) / (b * c), -1.0, 1.0))))
        beta = float(np.degrees(np.arccos(np.clip(
            np.dot(a_vec, c_vec) / (a * c), -1.0, 1.0))))
        gamma = float(np.degrees(np.arccos(np.clip(
            np.dot(a_vec, b_vec) / (a * b), -1.0, 1.0))))

        M = np.column_stack([a_vec, b_vec, c_vec])
        try:
            M_inv = np.linalg.inv(M)
        except Exception:
            return None

        lines = [
            "data_optimade",
            "_symmetry_space_group_name_H-M 'P 1'",
            f"_cell_length_a {a}",
            f"_cell_length_b {b}",
            f"_cell_length_c {c}",
            f"_cell_angle_alpha {alpha}",
            f"_cell_angle_beta {beta}",
            f"_cell_angle_gamma {gamma}",
            "loop_",
            "_atom_site_type_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
        ]
        for sym, pos in zip(species, positions):
            frac = M_inv @ np.array(pos, dtype=float)
            lines.append(f"{sym} {frac[0]:.8f} {frac[1]:.8f} {frac[2]:.8f}")

        return "\n".join(lines)
    except Exception as e:
        logger.debug(f"CIF build failed: {e}")
        return None


def search_phases_by_exact_elements(elements: List[str],
                                    max_per_provider: int = 20,
                                    timeout: int = 20) -> List[Dict]:
    """
    Query OPTIMADE providers for structures whose element set matches
    `elements` exactly. Returns candidate dicts, sorted by formula length.
    """
    if not elements:
        return []

    target_set = set(elements)
    filter_str = " AND ".join([f'elements HAS "{el}"' for el in elements])

    results = []
    seen_formulas = set()

    for provider_name, url in OPTIMADE_PROVIDERS:
        try:
            r = requests.get(url, params={
                "filter": filter_str,
                "page_limit": str(max_per_provider),
            }, timeout=timeout)

            if r.status_code != 200:
                logger.info(f"OPTIMADE {provider_name}: HTTP {r.status_code}")
                continue

            try:
                payload = r.json()
            except Exception:
                continue

            for entry in payload.get("data", []) or []:
                attrs = entry.get("attributes", {}) or {}
                formula = attrs.get("chemical_formula_reduced", "")
                if not formula:
                    continue

                if set(parse_formula_elements(formula)) != target_set:
                    continue
                if formula in seen_formulas:
                    continue
                seen_formulas.add(formula)

                cif_text = _build_cif_from_optimade(attrs)
                if not cif_text:
                    continue

                results.append({
                    'formula': formula,
                    'provider': provider_name,
                    'entry_id': entry.get('id', 'unknown'),
                    'cif_text': cif_text,
                    'space_group': attrs.get('space_group_symmetry', 'Unknown'),
                })
        except Exception as e:
            logger.info(f"OPTIMADE {provider_name} unavailable: {e}")

    results.sort(key=lambda x: (len(x['formula']), x['formula']))
    return results


# ============================================================================
# CIF SIMULATION
# ============================================================================
def simulate_phase_pattern(cif_text: str,
                           wavelength: float = 1.5406,
                           two_theta_range: Tuple[float, float] = (5.0, 80.0)
                           ) -> Optional[Dict]:
    """Simulate a powder XRD pattern from a CIF string."""
    try:
        from pymatgen.io.cif import CifParser
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        try:
            parser = CifParser.from_string(cif_text)
        except AttributeError:
            parser = CifParser(io.StringIO(cif_text))

        structure = parser.get_structures()[0]

        try:
            sga = SpacegroupAnalyzer(structure.get_primitive())
            structure = sga.get_conventional_standard_structure()
        except Exception:
            pass

        calc = XRDCalculator(wavelength=wavelength)
        pattern = calc.get_pattern(structure, two_theta_range=two_theta_range)

        hkls = []
        for hkl_list in pattern.hkls:
            if hkl_list and isinstance(hkl_list, list):
                hkl_tuple = tuple(int(x) for x in hkl_list[0]['hkl'])
                mult = len(hkl_list)
            else:
                hkl_tuple = (0, 0, 0)
                mult = 1
            hkls.append({'hkl': hkl_tuple, 'multiplicity': mult})

        sga2 = SpacegroupAnalyzer(structure)
        return {
            'two_theta': np.array(pattern.x),
            'intensity': np.array(pattern.y),
            'hkls': hkls,
            'formula': structure.composition.reduced_formula,
            'space_group': sga2.get_space_group_symbol(),
            'crystal_system': sga2.get_crystal_system().capitalize(),
            'lattice': {
                'a': float(structure.lattice.a),
                'b': float(structure.lattice.b),
                'c': float(structure.lattice.c),
                'alpha': float(structure.lattice.alpha),
                'beta': float(structure.lattice.beta),
                'gamma': float(structure.lattice.gamma),
            },
            'density': float(structure.density),
        }
    except Exception as e:
        logger.warning(f"CIF simulation failed: {e}")
        return None


# ============================================================================
# SCORING (PEAK POSITION MATCH FRACTION)
# ============================================================================
def _detect_experimental_peaks(two_theta, intensity, height_frac=0.05):
    """Detect experimental peaks with scipy.find_peaks."""
    from scipy.signal import find_peaks
    try:
        tt = np.asarray(two_theta, dtype=float)
        ii = np.asarray(intensity, dtype=float)
        if len(tt) < 10:
            return np.array([])
        height = float(np.max(ii)) * height_frac
        idx, _ = find_peaks(ii, height=height, distance=5)
        return tt[idx]
    except Exception:
        return np.array([])


def score_phase_against_experimental(phase_pattern: Dict,
                                     exp_two_theta,
                                     exp_intensity,
                                     tol_deg: float = 0.20) -> Dict:
    """
    Peak position match fraction.

    For each experimental peak, count as matched if at least one reference
    peak lies within tol_deg. Return matched / total experimental peaks
    expressed as a percentage. This is a peak-position indicator only.
    """
    if phase_pattern is None:
        return {'score': 0.0, 'matched': 0, 'total_exp_peaks': 0,
                'coverage': 0.0, 'matched_tt': []}

    exp_peak_tt = _detect_experimental_peaks(exp_two_theta, exp_intensity)
    if len(exp_peak_tt) == 0:
        return {'score': 0.0, 'matched': 0, 'total_exp_peaks': 0,
                'coverage': 0.0, 'matched_tt': []}

    sim_tt = np.asarray(phase_pattern['two_theta'], dtype=float)
    if len(sim_tt) == 0:
        return {'score': 0.0, 'matched': 0, 'total_exp_peaks': len(exp_peak_tt),
                'coverage': 0.0, 'matched_tt': []}

    matched = 0
    matched_positions = []
    for et in exp_peak_tt:
        if float(np.min(np.abs(sim_tt - et))) <= tol_deg:
            matched += 1
            matched_positions.append(float(et))

    coverage = matched / len(exp_peak_tt)
    return {
        'score': round(coverage * 100.0, 1),
        'matched': matched,
        'total_exp_peaks': int(len(exp_peak_tt)),
        'coverage': round(coverage, 3),
        'matched_tt': matched_positions,
    }


# ============================================================================
# MULTI-PHASE COMPARISON PLOT
# ============================================================================
def plot_multiphase_comparison(exp_two_theta, exp_intensity,
                               phase_patterns: List[Dict],
                               wavelength: float = 1.5406,
                               font_size: int = 10,
                               title: str = ""):
    """
    HighScore-style multi-phase comparison.

    Layout:
        - Top panel: experimental pattern, normalised to 1.
        - One row per phase below, each row fixed height.
        - Each phase drawn as vertical sticks coloured by a fixed palette.
        - Strongest 3 reflections labelled with HKL.
        - Per-row label (phase name) and score.

    The figure states explicitly that the score is a peak-position
    match fraction and NOT a quantitative phase fraction.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    n = len(phase_patterns)
    if n == 0:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.text(0.5, 0.5, "No phases to compare",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    row_height_in = 0.55
    base_height_in = 4.0
    fig_height = base_height_in + row_height_in * n
    fig_width = 11.0

    height_ratios = [3.2] + [1.0] * n
    fig, axes = plt.subplots(
        n + 1, 1,
        figsize=(fig_width, fig_height),
        gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.05},
        sharex=True,
    )

    if n + 1 == 1:
        axes = [axes]

    exp_tt = np.asarray(exp_two_theta, dtype=float)
    exp_ii = np.asarray(exp_intensity, dtype=float)
    exp_ii_norm = exp_ii / float(np.max(exp_ii)) if float(np.max(exp_ii)) > 0 else exp_ii

    ax0 = axes[0]
    ax0.plot(exp_tt, exp_ii_norm, '-', color='black', linewidth=0.9)
    ax0.set_ylabel('Normalised intensity', fontsize=font_size)
    ax0.set_ylim(-0.05, 1.20)
    ax0.grid(True, axis='y', alpha=0.20)
    ax0.set_yticks([0.0, 0.5, 1.0])
    ax0.set_title(title or 'Experimental pattern and reference phases',
                  fontsize=font_size + 2, pad=8)
    ax0.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    for idx, pp in enumerate(phase_patterns):
        ax = axes[idx + 1]
        color = pp.get('color', PHASE_COLORS[idx % len(PHASE_COLORS)])
        label = pp.get('label', f'Phase {idx + 1}')
        pattern = pp.get('pattern')
        score = pp.get('score') or {}

        ax.axhline(0.0, color=color, linewidth=0.8, alpha=0.5)

        if pattern is not None:
            sim_tt = np.asarray(pattern['two_theta'], dtype=float)
            sim_ii = np.asarray(pattern['intensity'], dtype=float)
            if len(sim_ii) > 0 and float(np.max(sim_ii)) > 0:
                sim_ii_norm = sim_ii / float(np.max(sim_ii))
            else:
                sim_ii_norm = sim_ii
            sim_hkls = pattern.get('hkls', [])

            for i_v, t in enumerate(sim_tt):
                if i_v >= len(sim_ii_norm):
                    break
                h = float(sim_ii_norm[i_v])
                if h < 0.02:
                    continue
                ax.plot([t, t], [0, h], '-', color=color, linewidth=1.3)

            try:
                order = np.argsort(sim_ii_norm)[::-1][:3]
                for oi in order:
                    if oi >= len(sim_hkls):
                        continue
                    hkl = sim_hkls[oi].get('hkl')
                    if not hkl or all(x == 0 for x in hkl):
                        continue
                    hkl_str = "(" + "".join(str(int(x)) for x in hkl) + ")"
                    ax.text(sim_tt[oi], float(sim_ii_norm[oi]) + 0.06,
                            hkl_str, ha='center', va='bottom',
                            fontsize=font_size - 4, color=color)
            except Exception:
                pass

        ax.set_ylim(-0.05, 1.25)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(False)

        ax.text(0.004, 0.72, label,
                transform=ax.transAxes, ha='left', va='top',
                fontsize=font_size - 1, color=color, weight='bold')

        if score:
            txt = (f"match: {score.get('score', 0):.1f}%  "
                   f"({score.get('matched', 0)}/{score.get('total_exp_peaks', 0)} exp. peaks)")
            ax.text(0.996, 0.72, txt,
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=font_size - 2, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor='white', edgecolor=color, alpha=0.85))

    bottom_ax = axes[-1]
    bottom_ax.set_xlabel(r'2$\theta$ (degrees)', fontsize=font_size)
    if len(exp_tt) > 0:
        bottom_ax.set_xlim(float(np.min(exp_tt)), float(np.max(exp_tt)))
    bottom_ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    fig.text(0.5, 0.012,
             'Reference patterns simulated from CIF (Bragg, 1913). '
             'Match value = fraction of experimental peaks whose position '
             'is reproduced by the reference. NOT a quantitative phase fraction.',
             ha='center', va='bottom',
             fontsize=font_size - 3, style='italic', color='#555555')

    plt.subplots_adjust(top=0.94, bottom=0.09, left=0.06, right=0.985)
    return fig
