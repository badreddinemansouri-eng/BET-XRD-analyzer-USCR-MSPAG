"""
RIETVELD QUANTITATIVE PHASE ANALYSIS (QPA) MODULE
========================================================================
Scientifically correct phase weight fractions from powder XRD.

Implements the Hill & Howard (1987) ZMV formula using the GSAS-II
Scriptable API as the Rietveld engine.

    w_alpha = S_alpha * (ZMV)_alpha / sum_j S_j * (ZMV)_j

where:
    S_alpha  = refined Rietveld scale factor for phase alpha
    Z        = formula units per unit cell
    M        = formula unit mass (g/mol)
    V        = unit cell volume (Angstrom^3)

VALIDITY CONDITIONS (stated on every output figure):
    1. All phases in the sample are crystalline.
    2. All phases have been identified and included in the refinement.
    3. A structural model (CIF) exists for each phase.
    4. The refinement has converged to a good fit (Rwp low, chi2 ~ 1).

If any condition is violated, the fractions are still normalized but
will overestimate the crystalline phases relative to the true sample.

References:
    Hill, R. J., & Howard, C. J. (1987). J. Appl. Cryst., 20, 467-474.
    Rietveld, H. M. (1969). J. Appl. Cryst., 2, 65-71.
    Toby, B. H., & Von Dreele, R. B. (2013). J. Appl. Cryst., 46, 544-549.
    Bish, D. L., & Howard, S. A. (1988). J. Appl. Cryst., 21, 86-91.
========================================================================
"""

import os
import io
import json
import logging
import tempfile
from typing import List, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import GSAS-II Scriptable
# ---------------------------------------------------------------------------
GSASII_AVAILABLE = False
try:
    import GSASIIscriptable as G2sc
    GSASII_AVAILABLE = True
    logger.info("GSAS-II Scriptable is available.")
except ImportError:
    logger.warning(
        "GSAS-II Scriptable is not available. "
        "Quantitative phase analysis requires GSAS-II. "
        "Install with: pip install GSAS-II"
    )


# ============================================================================
# ZMV CALCULATION FROM CIF
# ============================================================================
def compute_ZMV_from_cif(cif_path: str) -> Optional[Dict]:
    """
    Compute Z, M, V, and ZMV from a CIF file using pymatgen.

    Returns
    -------
    dict with keys: Z, M, V, ZMV, formula, formula_weight, density
    or None on failure.
    """
    try:
        from pymatgen.io.cif import CifParser

        try:
            parser = CifParser(cif_path)
        except Exception:
            with open(cif_path, 'r') as f:
                content = f.read()
            try:
                parser = CifParser.from_string(content)
            except AttributeError:
                parser = CifParser(io.StringIO(content))

        structures = parser.get_structures()
        if not structures:
            logger.warning(f"No structures found in CIF: {cif_path}")
            return None

        structure = structures[0]

        # Z = number of formula units per unit cell
        # pymatgen: structure.composition is the full cell contents
        reduced = structure.composition.reduced_composition
        full = structure.composition

        # Z = (number of atoms in cell) / (number of atoms in reduced formula)
        n_atoms_cell = len(structure)
        n_atoms_reduced = int(sum(reduced.values()))
        Z = n_atoms_cell // n_atoms_reduced if n_atoms_reduced > 0 else 1

        # M = mass of reduced formula (g/mol)
        M = reduced.weight

        # V = unit cell volume (Angstrom^3)
        V = structure.lattice.volume

        # ZMV = product
        ZMV = Z * M * V

        # Density check
        density = structure.density

        return {
            'Z': int(Z),
            'M': float(M),
            'V': float(V),
            'ZMV': float(ZMV),
            'formula': reduced.reduced_formula,
            'formula_weight': float(M),
            'density': float(density),
        }
    except Exception as e:
        logger.error(f"ZMV computation failed for {cif_path}: {e}")
        return None


# ============================================================================
# GSAS-II RIETVELD REFINEMENT
# ============================================================================
def run_rietveld_refinement(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    phases: List[Dict],
    wavelength: float = 1.5406,
    max_cycles: int = 10,
    refine_background: bool = True,
    refine_cell: bool = True,
    instrument_params: Optional[Dict] = None,
) -> Dict:
    """
    Run a Rietveld refinement using GSAS-II Scriptable.

    Parameters
    ----------
    two_theta : array
        Experimental 2theta values in degrees.
    intensity : array
        Experimental intensity values.
    phases : list of dict
        Each dict must contain:
            - 'name'     : str, phase name
            - 'cif_path' : str, path to a CIF file
            - 'color'    : optional str, hex color
    wavelength : float
        X-ray wavelength in Angstrom.
    max_cycles : int
        Maximum number of refinement cycles.
    refine_background : bool
        Whether to refine the background polynomial.
    refine_cell : bool
        Whether to refine unit cell parameters.
    instrument_params : dict, optional
        Additional instrument parameters (U, V, W, etc.).

    Returns
    -------
    dict with:
        - 'success'         : bool
        - 'Rwp'             : weighted profile R-factor (%)
        - 'Rp'              : profile R-factor (%)
        - 'chi2'            : goodness of fit
        - 'phase_fractions' : list of {name, weight_fraction, S, ZMV, ...}
        - 'calculated'      : calculated pattern
        - 'difference'      : observed - calculated
        - 'error'           : error message if failed
    """
    if not GSASII_AVAILABLE:
        return {
            'success': False,
            'error': 'GSAS-II is not installed. '
                     'Install with: pip install GSAS-II',
        }

    if not phases:
        return {'success': False, 'error': 'No phases provided.'}

    # Validate all CIF files and compute ZMV
    zmv_info = []
    for ph in phases:
        cif_path = ph.get('cif_path')
        if not cif_path or not os.path.exists(cif_path):
            return {
                'success': False,
                'error': f"CIF file not found for phase {ph.get('name')}: {cif_path}",
            }
        info = compute_ZMV_from_cif(cif_path)
        if info is None:
            return {
                'success': False,
                'error': f"Could not compute ZMV for phase {ph.get('name')}",
            }
        zmv_info.append(info)

    work_dir = tempfile.mkdtemp(prefix="rietveld_")
    gpx_path = os.path.join(work_dir, "refinement.gpx")
    data_path = os.path.join(work_dir, "experimental.dat")

    # Write experimental data
    try:
        with open(data_path, 'w') as f:
            for t, i in zip(two_theta, intensity):
                f.write(f"{t:.6f}  {i:.4f}\n")
    except Exception as e:
        return {'success': False, 'error': f"Could not write data file: {e}"}

    try:
        # ------------------------------------------------------------------
        # 1. Create GSAS-II project
        # ------------------------------------------------------------------
        gpx = G2sc.G2Project(newgpx=gpx_path)

        # ------------------------------------------------------------------
        # 2. Add histogram (powder data)
        # ------------------------------------------------------------------
        # GSAS-II expects "PXC" for constant-wavelength X-ray powder data
        hist = gpx.add_powder_histogram(
            data_path,
            instr_file=None,
            fmthint="PXC",
            phase_list=[],
        )

        # Set wavelength
        try:
            hist.data['Instrument Parameters'][0]['Lam'] = wavelength
            hist.data['Instrument Parameters'][0]['Lam1'] = wavelength
        except Exception:
            pass

        # ------------------------------------------------------------------
        # 3. Add phases
        # ------------------------------------------------------------------
        for ph, info in zip(phases, zmv_info):
            cif_path = ph['cif_path']
            try:
                g2phase = gpx.add_phase(
                    cif_path,
                    phasename=ph.get('name', 'unknown'),
                    histograms=[hist],
                    fmthint="CIF",
                )
            except Exception as e:
                logger.error(f"Failed to add phase {ph.get('name')}: {e}")
                return {
                    'success': False,
                    'error': f"Could not add phase {ph.get('name')}: {e}",
                }

        # ------------------------------------------------------------------
        # 4. Set initial refinement flags
        # ------------------------------------------------------------------
        gpx.data['Controls']['data']['maxCycles'] = max_cycles

        # Refine scale factors for all phases
        for p in gpx.phases():
            p.set_refinements({'Scale': True})

        # Refine unit cell parameters
        if refine_cell:
            for p in gpx.phases():
                p.set_refinements({'Cell': True})

        # Refine background
        if refine_background:
            hist.set_refinements({'Background': True})

        # Refine histogram scale
        hist.set_refinements({'Scale': True})

        # ------------------------------------------------------------------
        # 5. Run refinement
        # ------------------------------------------------------------------
        gpx.refine()

        # ------------------------------------------------------------------
        # 6. Extract results
        # ------------------------------------------------------------------
        # R-factors
        rwp = None
        rp = None
        chi2 = None
        try:
            rwp = float(hist.residuals.get('Rwp', 0))
            rp = float(hist.residuals.get('Rp', 0))
            chi2 = float(hist.residuals.get('chi2', 0))
        except Exception:
            try:
                stats = hist.get_wR()
                rwp = float(stats)
            except Exception:
                pass

        # Scale factors and phase fractions
        phase_results = []
        scale_factors = []
        for i, (p, ph, info) in enumerate(zip(gpx.phases(), phases, zmv_info)):
            try:
                # Get refined scale factor
                s = None
                try:
                    s = float(p.HAPvalue('Scale', targethistlist=[0]))
                except Exception:
                    try:
                        s = float(p.HAPvalue('Scale'))
                    except Exception:
                        pass

                # GSAS-II may also expose the phase fraction directly
                pf = None
                try:
                    pf = float(p.HAPvalue('PhaseFraction', targethistlist=[0]))
                except Exception:
                    pass

                if s is not None:
                    scale_factors.append(s)
                else:
                    scale_factors.append(0.0)

                phase_results.append({
                    'name': ph.get('name', f'Phase {i+1}'),
                    'scale_factor_S': s,
                    'ZMV': info['ZMV'],
                    'Z': info['Z'],
                    'M': info['M'],
                    'V': info['V'],
                    'formula': info['formula'],
                    'density': info['density'],
                    'gsas_phase_fraction': pf,
                })
            except Exception as e:
                logger.warning(f"Could not extract scale factor for {ph.get('name')}: {e}")
                phase_results.append({
                    'name': ph.get('name', f'Phase {i+1}'),
                    'scale_factor_S': None,
                    'ZMV': info['ZMV'],
                    'Z': info['Z'],
                    'M': info['M'],
                    'V': info['V'],
                    'formula': info['formula'],
                    'density': info['density'],
                    'gsas_phase_fraction': None,
                })

        # ------------------------------------------------------------------
        # 7. Apply Hill & Howard ZMV formula
        # ------------------------------------------------------------------
        valid_pairs = []
        for pr in phase_results:
            s = pr['scale_factor_S']
            zmv = pr['ZMV']
            if s is not None and s > 0 and zmv > 0:
                valid_pairs.append((pr, s * zmv))

        if valid_pairs:
            denominator = sum(v for _, v in valid_pairs)
            for pr, sv in valid_pairs:
                pr['weight_fraction'] = round(100.0 * sv / denominator, 2)
                pr['weight_fraction_formula'] = (
                    f"S={pr['scale_factor_S']:.6e}, "
                    f"ZMV={pr['ZMV']:.2f}"
                )
        else:
            for pr in phase_results:
                pr['weight_fraction'] = None
                pr['weight_fraction_formula'] = ''

        # ------------------------------------------------------------------
        # 8. Compute calculated pattern and difference
        # ------------------------------------------------------------------
        calculated = None
        difference = None
        try:
            # GSAS-II can compute the calculated pattern
            calc = hist.calc_std_uncertainty()
            if hasattr(hist, 'compute_graphics'):
                graphics = hist.compute_graphics()
                calculated = np.array(graphics.get('calc', []))
                observed = np.array(graphics.get('obs', []))
                if len(calculated) > 0 and len(observed) > 0:
                    difference = observed - calculated
        except Exception as e:
            logger.info(f"Could not extract calculated pattern: {e}")

        # ------------------------------------------------------------------
        # 9. Copy the GPX file for download
        # ------------------------------------------------------------------
        gpx_bytes = None
        try:
            with open(gpx_path, 'rb') as f:
                gpx_bytes = f.read()
        except Exception:
            pass

        return {
            'success': True,
            'Rwp': rwp,
            'Rp': rp,
            'chi2': chi2,
            'phase_fractions': phase_results,
            'calculated': calculated,
            'difference': difference,
            'gpx_bytes': gpx_bytes,
            'wavelength': wavelength,
            'n_cycles': max_cycles,
        }

    except Exception as e:
        logger.exception("Rietveld refinement failed")
        return {
            'success': False,
            'error': f"Refinement failed: {e}",
        }


# ============================================================================
# FORMATTING FOR DISPLAY
# ============================================================================
def format_qpa_table(phase_fractions: List[Dict]) -> List[Dict]:
    """
    Format the phase fraction results for a pandas DataFrame.
    """
    rows = []
    for pf in phase_fractions:
        wf = pf.get('weight_fraction')
        rows.append({
            'Phase': pf.get('name', '?'),
            'Formula': pf.get('formula', '?'),
            'Z': pf.get('Z', '?'),
            'M (g/mol)': round(pf.get('M', 0), 3),
            'V (A^3)': round(pf.get('V', 0), 2),
            'ZMV': round(pf.get('ZMV', 0), 2),
            'Scale S': f"{pf.get('scale_factor_S'):.4e}"
                       if pf.get('scale_factor_S') is not None else 'N/A',
            'Weight %': f"{wf:.2f}"
                        if wf is not None else 'N/A',
        })
    return rows


def qpa_validity_statement() -> str:
    """Return the standard validity statement for the QPA results."""
    return (
        "Hill & Howard (1987) ZMV method. Weight fractions are normalised "
        "to 100% over the crystalline phases included in the refinement. "
        "Valid only if all phases are crystalline and identified. "
        "If amorphous or unidentified phases are present, the reported "
        "fractions overestimate the crystalline phases."
    )
