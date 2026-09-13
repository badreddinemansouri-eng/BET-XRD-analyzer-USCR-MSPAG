"""
MAIN APPLICATION - BET-XRD MORPHOLOGY ANALYZER
========================================================================
Journal Submission Version - Scientific Precision Required
========================================================================
References:
1. Rouquerol et al., Pure Appl. Chem., 1994, 66, 1739-1758 (BET)
2. Thommes et al., Pure Appl. Chem., 2015, 87, 1051-1069 (Physisorption)
3. Klug & Alexander, X-ray Diffraction Procedures, 1974 (XRD)
4. Williamson & Hall, Acta Metall., 1953, 1, 22-31 (Microstrain)
========================================================================
"""

import os
import streamlit as st

os.environ["MP_API_KEY"] = st.secrets.get("MP_API_KEY", "")

import numpy as np
import pandas as pd
import sys
import warnings
import io
import json
import re
import time
import traceback
import functools
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

try:
    from scientific_integration import ScientificIntegrator
except ImportError:
    class ScientificIntegrator:
        def __init__(self):
            pass

        def integrate_results(self, bet_results, xrd_results):
            return {
                'valid': False,
                'error': 'ScientificIntegrator import failed',
                'correlation_analysis': {},
                'material_classification': {},
                'structure_properties': {},
                'validation_metrics': {},
                'recommendations': []
            }

from bet_analyzer import IUPACBETAnalyzer, extract_asap2420_data
from xrd_analyzer import AdvancedXRDAnalyzer, extract_xrd_data
from morphology_fusion import MorphologyFusionEngine
from scientific_plots import PublicationPlotter
from morphology_visualizer import MorphologyVisualizer
from xrd_phase_identifier_nano import identify_phases_universal
from xrd_overlay import XROverlayPlotter, two_theta_to_d
from export_utils import (
    panel_export_buttons,
    save_all_to_zip,
    fig_to_bytes,
    fig_to_download,
    available_export_formats,
    close_fig_safely,
)


def memory_safe_plot(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            import matplotlib.pyplot as plt
            plt.close('all')
            import gc
            gc.collect()
            return result
        except Exception as e:
            import matplotlib.pyplot as plt
            plt.close('all')
            raise e
    return wrapper


st.set_page_config(
    page_title="BET-XRD Morphology Analyzer | Journal Edition",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

if 'scientific_data' not in st.session_state:
    st.session_state.scientific_data = {
        'bet_raw': None,
        'xrd_raw': None,
        'xrd_patterns': [],
        'bet_results': None,
        'xrd_results': None,
        'fusion_results': None,
        'analysis_valid': False
    }


def _has_bet_data(results: Dict) -> bool:
    if not isinstance(results, dict):
        return False
    if results.get('bet_raw') is None:
        return False
    bet_res = results.get('bet_results')
    return isinstance(bet_res, dict)


def _has_xrd_data(results: Dict) -> bool:
    if not isinstance(results, dict):
        return False
    if results.get('xrd_raw') is None:
        return False
    xrd_res = results.get('xrd_results')
    return isinstance(xrd_res, dict)


def _has_multiple_xrd(results: Dict) -> bool:
    pats = results.get('xrd_patterns', [])
    return isinstance(pats, list) and len(pats) > 1


def create_sidebar():
    with st.sidebar:
        st.title("⚗️ Scientific Controls")

        st.markdown("---")
        st.subheader("BET Analysis Parameters")

        bet_params = {
            'gas': st.selectbox(
                "Adsorbate Gas",
                ["N2 (77 K)", "Ar (87 K)", "CO2 (273 K)"],
                index=0
            ),
            'cross_section': st.number_input(
                "Cross-section (nm2)",
                value=0.162, min_value=0.1, max_value=0.3, step=0.001,
                help="N2: 0.162 nm2, Ar: 0.142 nm2, CO2: 0.187 nm2"
            ),
            'bet_range_min': st.number_input(
                "BET min P/P0", value=0.05, min_value=0.01, max_value=0.1,
                step=0.01, help="IUPAC recommends 0.05-0.35"
            ),
            'bet_range_max': st.number_input(
                "BET max P/P0", value=0.35, min_value=0.2, max_value=0.5,
                step=0.01
            )
        }

        st.markdown("---")
        st.subheader("XRD Analysis Parameters")

        xrd_params = {
            'wavelength': st.selectbox(
                "X-ray Wavelength",
                ["Cu Ka (0.15406 nm)", "Mo Ka (0.07107 nm)", "Co Ka (0.17902 nm)"],
                index=0
            ),
            'background_subtraction': st.checkbox(
                "Background Subtraction", value=True,
                help="Remove amorphous background using SNIP algorithm"
            ),
            'peak_threshold': st.slider(
                "Peak Detection Threshold", min_value=0.01, max_value=0.5,
                value=0.1, step=0.01
            ),
            'smoothing': st.selectbox(
                "Smoothing Method",
                ["Savitzky-Golay", "Moving Average", "None"],
                index=0
            )
        }

        st.markdown("---")
        st.subheader("Crystal Structure")

        crystal_params = {
            'system': st.selectbox(
                "Crystal System",
                ["Unknown", "Cubic", "Tetragonal", "Hexagonal",
                 "Rhombohedral", "Orthorhombic", "Monoclinic", "Triclinic"],
                index=0
            ),
            'space_group': st.text_input(
                "Space Group", value="",
                help="e.g., Fd-3m, P63/mmc, Im-3m"
            ),
            'lattice_params': st.text_input(
                "Lattice Parameters (Angstrom)", value="",
                help="e.g., a=4.05 (Cubic), a=4.05 c=6.7 (Hexagonal)"
            ),
            'composition': st.text_input(
                "Material Composition", value="SiO2",
                help="e.g., SiO2, TiO2, Al2O3, ZrO2, CeO2"
            )
        }

        with st.expander("Advanced Crystallography", expanded=False):
            crystal_params['enable_3d'] = st.checkbox(
                "Generate 3D Crystal Structure", value=True)
            crystal_params['show_interactive'] = st.checkbox(
                "Interactive 3D View", value=False)
            crystal_params['supercell_size'] = st.slider(
                "Supercell Size", min_value=1, max_value=4, value=2)

        st.markdown("---")
        st.subheader("Export Settings")

        export_params = {
            'figure_format': st.selectbox(
                "Figure Format",
                ["PNG (600 DPI)", "PDF (Vector)", "SVG (Vector)", "TIFF (1200 DPI)"],
                index=0
            ),
            'color_scheme': st.selectbox(
                "Color Scheme",
                ["Nature", "Science", "ACS", "RSC", "Wiley"],
                index=0
            ),
            'font_size': st.slider(
                "Font Size (pt)", min_value=8, max_value=14, value=10, step=1)
        }

        from pymatgen.core import Element
        st.sidebar.subheader("Expected Elements in Sample")
        ALL_ELEMENTS = [el.symbol for el in Element]
        selected_elements = st.sidebar.multiselect(
            "Select elements present in your sample",
            options=ALL_ELEMENTS,
            help="Used for phase identification (COD + OPTIMADE)"
        )
        if not selected_elements:
            st.sidebar.warning("No elements selected - phase identification disabled")
        st.session_state["xrd_elements"] = selected_elements

        st.markdown("---")
        st.subheader("Scientific References")
        with st.expander("View References"):
            st.markdown(
                "**BET Analysis:**\n"
                "1. Rouquerol, J.; Llewellyn, P.; Rouquerol, F. Stud. Surf. Sci. Catal. 2007, 160, 49-56.\n"
                "2. Thommes, M. et al. Pure Appl. Chem. 2015, 87, 1051-1069.\n\n"
                "**XRD Analysis:**\n"
                "1. Klug, H. P.; Alexander, L. E. X-ray Diffraction Procedures, 2nd ed.; Wiley: 1974.\n"
                "2. Williamson, G. K.; Hall, W. H. Acta Metall. 1953, 1, 22-31.\n\n"
                "**Porosity Analysis:**\n"
                "1. Barrett, E. P.; Joyner, L. G.; Halenda, P. P. J. Am. Chem. Soc. 1951, 73, 373-380.\n"
                "2. Harkins, W. D.; Jura, G. J. Am. Chem. Soc. 1944, 66, 1366-1373."
            )

        return {
            'bet': bet_params,
            'xrd': xrd_params,
            'crystal': crystal_params,
            'export': export_params
        }


def file_upload_section():
    st.header("Experimental Data Upload")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Physisorption Data")
        bet_file = st.file_uploader(
            "Upload BET Isotherm File",
            type=["xls", "xlsx", "csv", "txt", "dat"],
            help="Supported: Micromeritics ASAP 2420, Quantachrome, BELSORP, custom 2-4 columns."
        )

        if bet_file:
            st.success(f"BET file: {bet_file.name}")
            with st.expander("Preview and Validate", expanded=False):
                try:
                    preview_data = extract_asap2420_data(bet_file, preview_only=True)
                    if preview_data:
                        st.write(f"Format detected: {preview_data['format']}")
                        st.write(f"Points found: {preview_data['n_points']}")
                        st.write(
                            f"Pressure range: {preview_data['p_range'][0]:.3f} - "
                            f"{preview_data['p_range'][1]:.3f} P/P0"
                        )
                        if preview_data.get('adsorption'):
                            st.write("Adsorption data preview:")
                            st.dataframe(pd.DataFrame({
                                'P/P0': preview_data['adsorption']['p'][:10],
                                'Q (mmol/g)': preview_data['adsorption']['q'][:10]
                            }))
                    bet_file.seek(0)
                    if bet_file.name.endswith('.xls'):
                        df_preview = pd.read_excel(bet_file, engine='xlrd', nrows=20)
                    elif bet_file.name.endswith('.xlsx'):
                        df_preview = pd.read_excel(bet_file, engine='openpyxl', nrows=20)
                    else:
                        df_preview = pd.read_csv(bet_file, nrows=20)
                    st.write("Raw file structure:")
                    st.dataframe(df_preview)
                except Exception as e:
                    st.error(f"Preview error: {str(e)}")

    with col2:
        st.subheader("XRD Data (Multiple Files Allowed)")
        xrd_files = st.file_uploader(
            "Upload one or more XRD Pattern Files",
            type=["csv", "txt", "xy", "dat", "xrdml"],
            accept_multiple_files=True,
            help="Upload multiple files to compare. The first file drives the main analysis. "
                 "All files appear in the Overlay tab."
        )

        if xrd_files:
            st.success(f"{len(xrd_files)} XRD file(s) uploaded.")
            for f in xrd_files:
                st.caption(f"- {f.name}")

            with st.expander("Preview & Validate Each File", expanded=False):
                for f in xrd_files:
                    try:
                        f.seek(0)
                        theta, intensity, msg = extract_xrd_data(f, preview_only=True)
                        if theta is not None:
                            st.write(f"**{f.name}** - {msg} "
                                     f"({len(theta)} points, "
                                     f"{theta[0]:.2f}-{theta[-1]:.2f} deg)")
                        else:
                            st.warning(f"**{f.name}** - no valid data: {msg}")
                    except Exception as e:
                        st.error(f"{f.name}: {str(e)}")

    return bet_file, xrd_files


def validate_input_data(bet_file, xrd_files, params):
    validation_results = {
        'bet_valid': False,
        'xrd_valid': False,
        'warnings': [],
        'recommendations': []
    }

    if bet_file:
        try:
            bet_file.seek(0)
            filename = bet_file.name.lower()
            valid_formats = ['.xls', '.xlsx', '.csv', '.txt', '.dat']
            if not any(filename.endswith(fmt) for fmt in valid_formats):
                validation_results['warnings'].append(
                    f"BET file format {filename} may not be optimal")

            if filename.endswith('.xls'):
                df = pd.read_excel(bet_file, engine='xlrd', nrows=100)
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(bet_file, engine='openpyxl', nrows=100)
            else:
                try:
                    df = pd.read_csv(bet_file, nrows=100)
                except Exception:
                    content = bet_file.read().decode('utf-8')
                    bet_file.seek(0)
                    for delimiter in ['\t', ';', ',', ' ']:
                        try:
                            df = pd.read_csv(io.StringIO(content),
                                             delimiter=delimiter, nrows=100)
                            break
                        except Exception:
                            continue

            n_rows = len(df)
            if n_rows < 10:
                validation_results['warnings'].append(
                    f"BET file has only {n_rows} rows (min 10 recommended)")
            else:
                validation_results['bet_valid'] = True

            numeric_cols = df.apply(pd.to_numeric, errors='coerce').notna().sum()
            if numeric_cols.sum() < 2:
                validation_results['warnings'].append(
                    "BET file may not contain numeric pressure/quantity columns")
            bet_file.seek(0)
        except Exception as e:
            validation_results['warnings'].append(
                f"BET file reading error: {str(e)[:100]}")

    if xrd_files:
        files_to_check = xrd_files if isinstance(xrd_files, list) else [xrd_files]
        valid_count = 0
        for xrd_file in files_to_check:
            try:
                xrd_file.seek(0)
                filename = xrd_file.name.lower()
                valid_formats = ['.csv', '.txt', '.xy', '.dat', '.xrdml']
                if not any(filename.endswith(fmt) for fmt in valid_formats):
                    validation_results['warnings'].append(
                        f"XRD file format {filename} may not be optimal")

                two_theta, intensity, msg = extract_xrd_data(xrd_file, preview_only=True)
                if two_theta is not None:
                    n_points = len(two_theta)
                    theta_range = two_theta.max() - two_theta.min()
                    if n_points < 100:
                        validation_results['warnings'].append(
                            f"{xrd_file.name}: only {n_points} points")
                    if theta_range < 10:
                        validation_results['warnings'].append(
                            f"{xrd_file.name}: angular range {theta_range:.1f} deg")
                    if n_points >= 50 and theta_range >= 5:
                        valid_count += 1
                xrd_file.seek(0)
            except Exception as e:
                validation_results['warnings'].append(
                    f"XRD file reading error: {str(e)[:100]}")

        if valid_count > 0:
            validation_results['xrd_valid'] = True

    return validation_results


def perform_analysis_validation(results):
    validation = {
        'bet_checks': [],
        'xrd_checks': [],
        'consistency_checks': [],
        'all_passed': True
    }

    if results.get('bet_results'):
        bet = results['bet_results']
        r2 = (bet.get('bet_regression') or {}).get('r_squared', 0)
        if r2 > 0.999:
            validation['bet_checks'].append({
                'check': 'BET Linearity', 'status': 'OK',
                'value': f"R2 = {r2:.6f}"})
        elif r2 > 0.995:
            validation['bet_checks'].append({
                'check': 'BET Linearity', 'status': 'WARN',
                'value': f"R2 = {r2:.6f} (moderate)"})
            validation['all_passed'] = False
        else:
            validation['bet_checks'].append({
                'check': 'BET Linearity', 'status': 'FAIL',
                'value': f"R2 = {r2:.6f} (poor)"})
            validation['all_passed'] = False

        if bet.get('c_constant', 0) > 0:
            validation['bet_checks'].append({
                'check': 'BET C Constant', 'status': 'OK',
                'value': f"C = {bet['c_constant']:.0f}"})
        else:
            validation['bet_checks'].append({
                'check': 'BET C Constant', 'status': 'FAIL',
                'value': 'Negative or zero C constant'})
            validation['all_passed'] = False

        S_bet = bet.get('surface_area', 0)
        if 0 < S_bet < 10000:
            validation['bet_checks'].append({
                'check': 'Surface Area Range', 'status': 'OK',
                'value': f"{S_bet:.1f} m2/g"})
        else:
            validation['bet_checks'].append({
                'check': 'Surface Area Range', 'status': 'FAIL',
                'value': 'Invalid surface area'})
            validation['all_passed'] = False

    if results.get('xrd_results'):
        xrd = results['xrd_results']
        n_peaks = len(xrd.get('structural_peaks', []))
        if n_peaks >= 3:
            validation['xrd_checks'].append({
                'check': 'Number of Peaks', 'status': 'OK',
                'value': f"{n_peaks} peaks detected"})
        else:
            validation['xrd_checks'].append({
                'check': 'Number of Peaks', 'status': 'WARN',
                'value': f"Only {n_peaks} peaks detected"})
            validation['all_passed'] = False

        ci = xrd.get('crystallinity_index', 0)
        if 0 <= ci <= 1:
            validation['xrd_checks'].append({
                'check': 'Crystallinity Index', 'status': 'OK',
                'value': f"{ci:.3f}"})
        else:
            validation['xrd_checks'].append({
                'check': 'Crystallinity Index', 'status': 'FAIL',
                'value': f"{ci:.3f} (outside 0-1)"})
            validation['all_passed'] = False

    return validation


def _get_wavelength(params):
    wavelength_str = params['xrd']['wavelength']
    if "Cu" in wavelength_str:
        return 1.5406
    elif "Mo" in wavelength_str:
        return 0.7107
    else:
        return 1.7902


def _analyze_single_xrd_file(xrd_file, params, elements):
    """Analyze one XRD file and return (raw_dict, results_dict) or (None, None)."""
    try:
        xrd_file.seek(0)
        two_theta, intensity, msg = extract_xrd_data(xrd_file)
        if two_theta is None or len(two_theta) < 50:
            return None, None, msg

        wavelength = _get_wavelength(params)
        analyzer = AdvancedXRDAnalyzer(
            wavelength=wavelength,
            background_subtraction=params['xrd']['background_subtraction'],
            smoothing=params['xrd']['smoothing'],
        )
        xrd_out = analyzer.complete_analysis(
            two_theta=two_theta,
            intensity=intensity,
            elements=elements
        )

        if not isinstance(xrd_out, dict):
            return None, None, "Invalid analysis output"

        if xrd_out.get("valid", False):
            xrd_results = xrd_out.get("xrd_results", {})
            if 'structural_peaks' not in xrd_results:
                xrd_results['structural_peaks'] = []
            if 'phases' not in xrd_results:
                xrd_results['phases'] = []
            if 'phase_fractions' not in xrd_results:
                xrd_results['phase_fractions'] = []
            if 'crystallite_size' not in xrd_results:
                xrd_results['crystallite_size'] = {
                    'scherrer': 0.0, 'williamson_hall': 0.0,
                    'distribution': 'Unknown'
                }
            if 'wavelength' not in xrd_results:
                xrd_results['wavelength'] = wavelength

            xrd_raw = xrd_out.get("xrd_raw", {
                "two_theta": two_theta.tolist(),
                "intensity": intensity.tolist()
            })
            return xrd_raw, xrd_results, "ok"

        partial = xrd_out.get("xrd_results", {})
        if isinstance(partial, dict) and partial:
            if 'structural_peaks' not in partial:
                partial['structural_peaks'] = []
            xrd_raw = xrd_out.get("xrd_raw", {
                "two_theta": two_theta.tolist(),
                "intensity": intensity.tolist()
            })
            return xrd_raw, partial, xrd_out.get("error", "partial")
    except Exception as e:
        return None, None, str(e)

    return None, None, "no result"


def execute_scientific_analysis(bet_file, xrd_files, params):
    analysis_results = {}

    st.subheader("Input Data Validation")
    validation_results = validate_input_data(bet_file, xrd_files, params)

    col1, col2 = st.columns(2)
    with col1:
        if bet_file:
            if validation_results['bet_valid']:
                st.success("BET data: Valid format and sufficient points")
            else:
                st.warning("BET data: Validation warnings")
    with col2:
        if xrd_files:
            if validation_results['xrd_valid']:
                st.success("XRD data: Valid format and sufficient points")
            else:
                st.warning("XRD data: Validation warnings")

    if validation_results['warnings']:
        with st.expander("Data Quality Warnings", expanded=True):
            for warning in validation_results['warnings']:
                st.warning(f"- {warning}")

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Extracting experimental data...")
        progress_bar.progress(10)

        if bet_file:
            bet_file.seek(0)
            p_ads, q_ads, p_des, q_des, psd_data, extraction_msg = extract_asap2420_data(bet_file)
            if p_ads is not None and len(p_ads) >= 5:
                analysis_results['bet_raw'] = {
                    'p_ads': p_ads, 'q_ads': q_ads,
                    'p_des': p_des, 'q_des': q_des,
                    'psd': psd_data,
                    'extraction_info': extraction_msg
                }
                st.success(f"BET data extracted: {len(p_ads)} adsorption points")
                if p_des is not None:
                    st.success(f"Desorption data: {len(p_des)} points")
            else:
                st.error(f"BET extraction failed: {extraction_msg}")

        # ============================================================
        # XRD: handle multiple files
        # ============================================================
        xrd_patterns = []
        if xrd_files:
            files_list = xrd_files if isinstance(xrd_files, list) else [xrd_files]
            st.info(f"Analyzing {len(files_list)} XRD file(s)...")
            for idx, xf in enumerate(files_list):
                label = xf.name
                with st.spinner(f"Analyzing XRD file {idx+1}/{len(files_list)}: {label}"):
                    raw, res, msg = _analyze_single_xrd_file(
                        xf, params, st.session_state.get("xrd_elements", [])
                    )
                if raw is not None and res is not None:
                    xrd_patterns.append({
                        'filename': label,
                        'xrd_raw': raw,
                        'xrd_results': res,
                        'message': msg,
                    })

            if xrd_patterns:
                st.success(f"XRD: {len(xrd_patterns)} file(s) analyzed successfully")
                # Primary XRD = first file
                analysis_results['xrd_raw'] = xrd_patterns[0]['xrd_raw']
                analysis_results['xrd_results'] = xrd_patterns[0]['xrd_results']
                analysis_results['xrd_patterns'] = xrd_patterns
            else:
                st.error("No XRD file could be analyzed successfully")

        # ============================================================
        # BET analysis
        # ============================================================
        if 'bet_raw' in analysis_results:
            status_text.text("Performing IUPAC-compliant BET analysis...")
            progress_bar.progress(30)
            try:
                gas_type = params['bet']['gas']
                if "N2" in gas_type:
                    cross_section = 0.162e-18
                    temperature = 77.3
                elif "Ar" in gas_type:
                    cross_section = 0.142e-18
                    temperature = 87.3
                else:
                    cross_section = 0.187e-18
                    temperature = 273.15

                bet_analyzer = IUPACBETAnalyzer(
                    p_ads=analysis_results['bet_raw']['p_ads'],
                    q_ads=analysis_results['bet_raw']['q_ads'],
                    p_des=analysis_results['bet_raw']['p_des'],
                    q_des=analysis_results['bet_raw']['q_des'],
                    cross_section=cross_section,
                    temperature=temperature
                )
                analysis_results['bet_results'] = bet_analyzer.complete_analysis()
                bet_res = analysis_results['bet_results']

                if bet_res.get('bet_valid', False):
                    st.success("BET analysis completed successfully")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("SBET",
                                  f"{bet_res['surface_area']:.1f} "
                                  f"+/- {bet_res['surface_area_error']:.1f} m2/g")
                    with col2:
                        st.metric("Vp", f"{bet_res['total_pore_volume']:.3f} cm3/g")
                    with col3:
                        st.metric("C", f"{bet_res['c_constant']:.0f}")
                    with col4:
                        st.metric("R2", f"{bet_res['bet_regression']['r_squared']:.4f}")
                else:
                    st.warning(f"BET surface area calculation failed: "
                               f"{bet_res.get('bet_error', 'Unknown error')}")
            except Exception as e:
                st.error(f"BET analysis error: {str(e)}")
                analysis_results['bet_results'] = {
                    'overall_valid': False, 'bet_valid': False,
                    'error': str(e), 'surface_area': 0.0,
                    'surface_area_error': 0.0, 'total_pore_volume': 0.0,
                    'mean_pore_diameter': 0.0
                }

        # ============================================================
        # Fusion + Integration
        # ============================================================
        bet_valid = analysis_results.get('bet_results', {}).get('overall_valid', False)
        xrd_valid = analysis_results.get('xrd_results', {}).get('valid', False)

        if bet_valid or xrd_valid:
            status_text.text("Fusing BET-XRD morphology data...")
            progress_bar.progress(80)
            try:
                fusion_engine = MorphologyFusionEngine()
                analysis_results['fusion_results'] = fusion_engine.fuse(
                    bet_results=analysis_results.get('bet_results', {}),
                    xrd_results=analysis_results.get('xrd_results', {})
                )
                if analysis_results['fusion_results'].get('valid', False):
                    st.success("Morphology fusion completed")
                    fusion = analysis_results['fusion_results']
                    st.info(f"Material Classification: {fusion.get('composite_classification', 'Unknown')}")
                    st.info(f"Dominant Feature: {fusion.get('dominant_feature', '')}")
                else:
                    st.warning("Morphology fusion completed with warnings")
            except Exception as e:
                st.error(f"Morphology fusion error: {str(e)}")

        if bet_valid or xrd_valid:
            status_text.text("Performing scientific integration...")
            progress_bar.progress(85)
            try:
                integrator = ScientificIntegrator()
                integration_results = integrator.integrate_results(
                    bet_results=analysis_results.get('bet_results', {}),
                    xrd_results=analysis_results.get('xrd_results', {})
                )
                if integration_results.get('valid', False):
                    analysis_results['integration'] = integration_results
                    st.success("Scientific integration completed")
                else:
                    st.warning("Scientific integration completed with warnings")
            except Exception as e:
                st.error(f"Scientific integration error: {str(e)}")

        status_text.text("Preparing scientific outputs...")
        progress_bar.progress(95)

        has_bet_data = 'bet_results' in analysis_results
        has_xrd_data = 'xrd_results' in analysis_results

        analysis_results['analysis_valid'] = (has_bet_data or has_xrd_data)
        analysis_results['parameters'] = params
        analysis_results['timestamp'] = pd.Timestamp.now().isoformat()
        analysis_results['validation'] = {
            'input_validation': validation_results,
            'analysis_validation': perform_analysis_validation(analysis_results)
        }

        progress_bar.progress(100)

        if analysis_results['analysis_valid']:
            st.success("Analysis complete!")
            st.balloons()
        else:
            st.warning("Some analyses failed, but available results are shown below")

        return analysis_results

    except Exception as e:
        st.error(f"Analysis pipeline failed: {str(e)}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
        return None


def main():
    st.title("BET-XRD Morphology Analyzer")
    st.markdown(
        "**Scientific Edition for Journal Publication**\n\n"
        "IUPAC-compliant physisorption analysis and advanced XRD characterization. "
        "All calculations follow established scientific literature with proper error analysis."
    )

    scientific_params = create_sidebar()
    bet_file, xrd_files = file_upload_section()

    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        analyze_button = st.button(
            "EXECUTE SCIENTIFIC ANALYSIS",
            type="primary",
            use_container_width=True,
            disabled=not (bet_file or xrd_files)
        )

    if analyze_button:
        with st.spinner("Initializing scientific analysis pipeline..."):
            results = execute_scientific_analysis(bet_file, xrd_files, scientific_params)
            if results:
                st.session_state.scientific_data = results

    if st.session_state.scientific_data.get('analysis_valid', False):
        display_scientific_results(st.session_state.scientific_data, scientific_params)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"Python {sys.version.split()[0]}")
        st.caption(f"NumPy {np.__version__}")
    with col2:
        st.caption("**Scientific Software v3.0**")
        st.caption("For Journal Publication")
    with col3:
        st.caption("(c) 2024 Materials Science Laboratory")
        st.caption("IUPAC Standards Compliant")


def display_scientific_results(results, scientific_params):
    st.header("Scientific Results")

    has_bet = _has_bet_data(results)
    has_xrd = _has_xrd_data(results)
    multi_xrd = _has_multiple_xrd(results)

    show_crystal_tab = (
        has_xrd and
        scientific_params['crystal']['system'] != 'Unknown' and
        bool(scientific_params['crystal']['lattice_params']) and
        scientific_params['crystal'].get('enable_3d', False)
    )

    all_tabs = ["Overview"]
    if has_bet:
        all_tabs.append("BET Analysis")
    if has_xrd:
        all_tabs.append("XRD Analysis")
        all_tabs.append("3D XRD Visualization")
        if multi_xrd:
            all_tabs.append("XRD Overlay")
    if show_crystal_tab:
        all_tabs.append("Crystal Structure")
    if has_bet and has_xrd:
        all_tabs.append("Morphology")
    if has_bet or has_xrd:
        all_tabs.append("Validation")
        all_tabs.append("Methods")
        all_tabs.append("Export")

    tabs = st.tabs(all_tabs)

    from scientific_plots import PublicationPlotter
    plotter = PublicationPlotter(
        color_scheme=scientific_params['export']['color_scheme'],
        font_size=scientific_params['export']['font_size']
    )

    tab_index = 0

    with tabs[tab_index]:
        display_overview(results, plotter)
    tab_index += 1

    if has_bet:
        with tabs[tab_index]:
            display_bet_analysis(results, plotter)
        tab_index += 1

    if has_xrd:
        with tabs[tab_index]:
            display_xrd_analysis(results, plotter)
        tab_index += 1

        with tabs[tab_index]:
            display_3d_xrd_visualization(results, scientific_params)
        tab_index += 1

        if multi_xrd:
            with tabs[tab_index]:
                display_xrd_overlay(results, scientific_params)
            tab_index += 1

    if show_crystal_tab:
        with tabs[tab_index]:
            try:
                from crystal_structure_3d import CrystalStructure3D
                crystal_3d = CrystalStructure3D()
                lattice_params = {}
                lattice_str = scientific_params['crystal']['lattice_params']
                for match in re.finditer(r'([abc])\s*=\s*([\d\.]+)', lattice_str):
                    lattice_params[match.group(1)] = float(match.group(2))

                structure = crystal_3d.generate_structure(
                    crystal_system=scientific_params['crystal']['system'],
                    lattice_params=lattice_params,
                    space_group=scientific_params['crystal']['space_group'],
                    composition=scientific_params['crystal'].get('composition', 'SiO2')
                )

                if not isinstance(structure, dict):
                    st.warning("Crystal structure generation returned invalid data.")
                else:
                    fig = crystal_3d.create_3d_plot(structure)
                    st.pyplot(fig)
                    panel_export_buttons(fig, "crystal_structure", "cs_tab")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Crystal System", scientific_params['crystal']['system'])
                    with col2:
                        st.metric("Space Group",
                                  scientific_params['crystal']['space_group'] or "Not specified")
                    with col3:
                        if 'density' in structure:
                            st.metric("Density", f"{structure['density']:.2f} g/cm3")
            except Exception as e:
                st.error(f"Could not generate 3D structure: {str(e)}")
        tab_index += 1

    if has_bet and has_xrd:
        with tabs[tab_index]:
            display_morphology(results)
        tab_index += 1

    if has_bet or has_xrd:
        with tabs[tab_index]:
            display_validation(results)
        tab_index += 1

        with tabs[tab_index]:
            display_methods(results, scientific_params)
        tab_index += 1

        with tabs[tab_index]:
            display_export(results, scientific_params)


@memory_safe_plot
def display_overview(results, plotter):
    st.subheader("Comprehensive Analysis Dashboard")

    try:
        fig = plotter.create_summary_figure(results)
        if fig is not None:
            st.pyplot(fig)
            panel_export_buttons(fig, "overview_summary", "ov_tab")
    except Exception as e:
        st.warning(f"Could not generate summary figure: {str(e)}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if _has_bet_data(results):
            bet = results['bet_results']
            st.metric("Surface Area (SBET)",
                      f"{bet.get('surface_area', 0):.1f} "
                      f"+/- {bet.get('surface_area_error', 0):.1f} m2/g")
    with col2:
        if _has_xrd_data(results):
            xrd = results['xrd_results']
            st.metric("Crystallinity Index",
                      f"{xrd.get('crystallinity_index', 0):.2f}")
    with col3:
        if _has_bet_data(results):
            bet = results['bet_results']
            st.metric("Total Pore Volume",
                      f"{bet.get('total_pore_volume', 0):.3f} cm3/g")
    with col4:
        classification = None
        if results.get('integration'):
            mc = results['integration'].get('material_classification')
            if isinstance(mc, dict):
                classification = mc.get('type') or mc.get('primary_phase')
        if classification:
            st.metric("Material Type", classification)
        elif results.get('fusion_results'):
            fusion = results['fusion_results']
            if isinstance(fusion, dict):
                st.metric("Material Type",
                          fusion.get('composite_classification', 'Unknown'))


@memory_safe_plot
def display_bet_analysis(results, plotter):
    st.subheader("IUPAC-Compliant BET Analysis")

    if not _has_bet_data(results):
        st.info("No BET data available.")
        return

    bet_res = results['bet_results']
    bet_raw = results['bet_raw']

    st.markdown("### Full BET Figure (all panels)")
    try:
        fig_full = plotter.create_bet_figure(bet_raw, bet_res)
        if fig_full is not None:
            st.pyplot(fig_full)
            panel_export_buttons(fig_full, "bet_full_6panel", "bet_full")
    except Exception as e:
        st.warning(f"Could not generate BET full figure: {str(e)}")

    st.markdown("---")
    st.markdown("### Individual Panels (for journals & slides)")
    st.caption("Each panel is exported independently below the plot.")

    # Panel A
    with st.expander("Panel A - Adsorption/Desorption Isotherm", expanded=False):
        try:
            f = plotter.plot_isotherm_only(bet_raw, bet_res)
            st.pyplot(f)
            panel_export_buttons(f, "bet_A_isotherm", "bet_A")
        except Exception as e:
            st.warning(str(e))

    # Panel B
    with st.expander("Panel B - BET Transform Plot", expanded=False):
        try:
            f = plotter.plot_bet_transform_only(bet_raw, bet_res)
            st.pyplot(f)
            panel_export_buttons(f, "bet_B_transform", "bet_B")
        except Exception as e:
            st.warning(str(e))

    # Panel C
    with st.expander("Panel C - t-Plot Analysis", expanded=False):
        try:
            f = plotter.plot_tplot_only(bet_raw, bet_res)
            st.pyplot(f)
            panel_export_buttons(f, "bet_C_tplot", "bet_C")
        except Exception as e:
            st.warning(str(e))

    # Panel D
    with st.expander("Panel D - BJH Pore Size Distribution", expanded=False):
        try:
            f = plotter.plot_psd_only(bet_res)
            st.pyplot(f)
            panel_export_buttons(f, "bet_D_psd", "bet_D")
        except Exception as e:
            st.warning(str(e))

    # Panel E
    with st.expander("Panel E - Hysteresis Loop", expanded=False):
        try:
            f = plotter.plot_hysteresis_only(bet_raw, bet_res)
            st.pyplot(f)
            panel_export_buttons(f, "bet_E_hysteresis", "bet_E")
        except Exception as e:
            st.warning(str(e))

    # Panel F
    with st.expander("Panel F - BET Summary Table", expanded=False):
        try:
            f = plotter.plot_bet_summary_table_only(bet_res)
            st.pyplot(f)
            panel_export_buttons(f, "bet_F_summary_table", "bet_F")
        except Exception as e:
            st.warning(str(e))

    st.markdown("---")
    with st.expander("BET Regression Details", expanded=False):
        reg = bet_res.get('bet_regression') or {}
        if reg:
            st.write(f"Linear Range: {reg.get('p_min', 0):.3f} - {reg.get('p_max', 0):.3f} P/P0")
            st.write(f"Equation: p/[n(1-p)] = {reg.get('slope', 0):.4f}p + {reg.get('intercept', 0):.4f}")
            st.write(f"R2: {reg.get('r_squared', 0):.6f}")
            st.write(f"Points used: {reg.get('n_points', 0)}")

    with st.expander("Porosity Analysis", expanded=False):
        st.write(f"Total Pore Volume: {bet_res.get('total_pore_volume', 0):.4f} cm3/g")
        st.write(f"Micropore Volume (t-plot): {bet_res.get('micropore_volume', 0):.4f} cm3/g")
        st.write(f"External Surface Area: {bet_res.get('external_surface', 0):.1f} m2/g")
        st.write(f"Mean Pore Diameter: {bet_res.get('mean_pore_diameter', 0):.2f} nm")


@memory_safe_plot
def display_xrd_analysis(results, plotter):
    st.subheader("Advanced XRD Analysis")

    if not _has_xrd_data(results):
        st.info("No XRD data available.")
        return

    xrd_res = results.get("xrd_results", {})
    xrd_raw = results.get("xrd_raw", {})
    patterns = results.get('xrd_patterns', [])

    if len(patterns) > 1:
        st.info(f"Main analysis uses the first file: "
                f"**{patterns[0].get('filename', 'unknown')}**. "
                f"See the XRD Overlay tab to compare all files.")

    structural_peaks = xrd_res.get("structural_peaks", [])
    n_detected = xrd_res.get("n_detected_maxima", 0)
    n_structural = len(structural_peaks)

    st.markdown("### Peak Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Detected local maxima", n_detected)
    with col2:
        st.metric("Structural Bragg peaks", n_structural)

    st.markdown("### Full XRD Figure (all panels)")
    if xrd_raw and xrd_res:
        try:
            fig_full = plotter.create_xrd_figure(xrd_raw, xrd_res)
            if fig_full is not None:
                st.pyplot(fig_full)
                panel_export_buttons(fig_full, "xrd_full_5panel", "xrd_full")
        except Exception as e:
            st.warning(f"Could not generate XRD full figure: {str(e)}")

    st.markdown("---")
    st.markdown("### Individual Panels (for journals & slides)")

    # Panel A
    with st.expander("Panel A - XRD Pattern with Peak Labels", expanded=True):
        try:
            f = plotter.plot_xrd_pattern_only(xrd_raw, xrd_res)
            st.pyplot(f)
            panel_export_buttons(f, "xrd_A_pattern", "xrd_A")
        except Exception as e:
            st.warning(str(e))

    # Panel B
    with st.expander("Panel B - Williamson-Hall Plot", expanded=False):
        try:
            f = plotter.plot_williamson_hall_only(xrd_res)
            st.pyplot(f)
            panel_export_buttons(f, "xrd_B_wh", "xrd_B")
        except Exception as e:
            st.warning(str(e))

    # Panel C
    with st.expander("Panel C - Crystallite Size Distribution", expanded=False):
        try:
            f = plotter.plot_size_distribution_only(xrd_res)
            st.pyplot(f)
            panel_export_buttons(f, "xrd_C_size_dist", "xrd_C")
        except Exception as e:
            st.warning(str(e))

    # Panel D
    with st.expander("Panel D - Structural Bragg Peaks Table", expanded=False):
        try:
            f = plotter.plot_xrd_peak_table_only(xrd_res)
            st.pyplot(f)
            panel_export_buttons(f, "xrd_D_peak_table", "xrd_D")
        except Exception as e:
            st.warning(str(e))

    # Panel E
    with st.expander("Panel E - XRD Summary Table", expanded=False):
        try:
            f = plotter.plot_xrd_summary_table_only(xrd_res)
            st.pyplot(f)
            panel_export_buttons(f, "xrd_E_summary_table", "xrd_E")
        except Exception as e:
            st.warning(str(e))


@memory_safe_plot
def display_xrd_overlay(results, scientific_params):
    st.subheader("XRD Overlay and Waterfall Comparison")
    st.caption("Compare all uploaded XRD patterns on one figure. "
               "Two display modes are available.")

    patterns = results.get('xrd_patterns', [])
    if not isinstance(patterns, list) or len(patterns) < 2:
        st.info("Upload at least 2 XRD files to use this tab.")
        return

    colA, colB, colC, colD = st.columns(4)
    with colA:
        mode = st.selectbox("Display mode", ["waterfall", "overlay"], index=0)
    with colB:
        normalize = st.checkbox("Normalize each pattern", value=True)
    with colC:
        show_peaks = st.checkbox("Show peak markers", value=False)
    with colD:
        offset_step = st.slider("Waterfall offset", 0.5, 3.0, 1.15, 0.05)

    wavelength = _get_wavelength(scientific_params)

    pattern_records = []
    for pat in patterns:
        raw = pat.get('xrd_raw', {}) or {}
        res = pat.get('xrd_results', {}) or {}
        peaks = []
        if show_peaks:
            for pk in (res.get('structural_peaks', []) or [])[:20]:
                peaks.append({
                    'position': pk.get('position', 0),
                    'hkl': _fmt_hkl_short(pk.get('hkl', '')),
                })
        pattern_records.append({
            'label': pat.get('filename', 'unknown'),
            'two_theta': raw.get('two_theta', []),
            'intensity': raw.get('intensity', []),
            'peaks': peaks,
        })

    try:
        overlay = XROverlayPlotter(
            color_scheme=scientific_params['export']['color_scheme'],
            font_size=scientific_params['export']['font_size']
        )
        fig = overlay.plot(
            patterns=pattern_records,
            mode=mode,
            normalize=normalize,
            offset_step=offset_step,
            wavelength=wavelength,
            show_peaks=show_peaks,
            title="XRD patterns comparison",
            xlabel="2theta (degrees)",
            ylabel="Normalized intensity (a.u.)",
        )
        st.pyplot(fig)
        panel_export_buttons(fig, f"xrd_overlay_{mode}", f"overlay_{mode}")
    except Exception as e:
        st.error(f"Overlay plot failed: {str(e)}")
        st.code(traceback.format_exc())

    # Per-pattern quick panel
    st.markdown("---")
    st.markdown("### Individual patterns (one figure per file)")
    for idx, rec in enumerate(pattern_records):
        with st.expander(f"{idx+1}. {rec['label']}", expanded=False):
            try:
                fig_single = overlay.plot(
                    patterns=[rec],
                    mode="overlay",
                    normalize=normalize,
                    wavelength=wavelength,
                    show_peaks=show_peaks,
                    show_legend=False,
                    title=rec['label'],
                )
                st.pyplot(fig_single)
                panel_export_buttons(fig_single,
                                     f"xrd_single_{idx+1}",
                                     f"xrd_single_{idx+1}")
            except Exception as e:
                st.warning(str(e))


def _fmt_hkl_short(hkl_val):
    if hkl_val is None:
        return ''
    if isinstance(hkl_val, (tuple, list)):
        try:
            return "(" + ",".join(str(int(x)) for x in hkl_val) + ")"
        except Exception:
            return str(hkl_val)
    if isinstance(hkl_val, dict):
        for k in ('hkl', 'indices'):
            if k in hkl_val:
                return _fmt_hkl_short(hkl_val[k])
    return str(hkl_val)


@memory_safe_plot
def display_3d_xrd_visualization(results, scientific_params):
    st.subheader("3D XRD Pattern and Crystal Structure Visualization")

    if not _has_xrd_data(results):
        st.info("No XRD data available.")
        return

    xrd_res = results['xrd_results']
    structural_peaks = xrd_res.get("structural_peaks", [])
    phases = xrd_res.get("phases", [])

    if not structural_peaks:
        st.warning("No peaks detected in XRD data")
        return

    def extract_hkl_indices(hkl_val):
        if hkl_val is None:
            return None
        if isinstance(hkl_val, (int, float, str)):
            return None
        if isinstance(hkl_val, (tuple, list)):
            if all(isinstance(x, (int, np.integer)) for x in hkl_val):
                return tuple(int(x) for x in hkl_val)
            if len(hkl_val) > 0 and isinstance(hkl_val[0], dict):
                if 'hkl' in hkl_val[0]:
                    return extract_hkl_indices(hkl_val[0]['hkl'])
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

    selected_phases = st.session_state.get("selected_phases_xrd", [])

    if selected_phases and phases:
        peak_to_phase = {}
        for phase in phases:
            if phase.get("phase") in selected_phases:
                for match in phase.get('hkls', []):
                    if not isinstance(match, dict):
                        continue
                    t_exp = match.get('two_theta_exp')
                    if t_exp is not None:
                        peak_to_phase[t_exp] = phase["phase"]
        filtered_peaks = [p for p in structural_peaks if p.get("position") in peak_to_phase]
        if not filtered_peaks:
            filtered_peaks = structural_peaks
    else:
        filtered_peaks = structural_peaks

    viz_tabs = st.tabs(["3D XRD Pattern", "Crystal Structure", "Combined View"])

    with viz_tabs[0]:
        try:
            import plotly.graph_objects as go
            positions = [p.get('position', 0) for p in filtered_peaks]
            intensities = [p.get('intensity', 0) for p in filtered_peaks]
            max_intensity = max(intensities) if intensities else 1
            norm_intensities = [i / max_intensity for i in intensities]

            fig = go.Figure()
            colors = {'TiO2': 'red', 'TiO': 'blue', 'Fe2O3': 'orange',
                      'ZnO': 'green', 'SiO2': 'purple', 'Al2O3': 'brown',
                      'default': 'gray'}

            for pos, intensity, norm_int in zip(positions, intensities, norm_intensities):
                wavelength = xrd_res.get("wavelength", 1.5406)
                theta = np.radians(pos / 2)
                q = (4 * np.pi / wavelength) * np.sin(theta)
                color = colors.get('default')
                fig.add_trace(go.Scatter3d(
                    x=[q, q], y=[0, norm_int], z=[0, 0],
                    mode='lines', line=dict(color=color, width=3),
                    showlegend=False, hoverinfo='none'
                ))
                fig.add_trace(go.Scatter3d(
                    x=[q], y=[norm_int], z=[0], mode='markers',
                    marker=dict(size=8, color=color, opacity=0.9,
                                line=dict(width=1, color='black')),
                    text=f"2theta: {pos:.2f} deg<br>Intensity: {intensity:.1f}",
                    hoverinfo='text', showlegend=False
                ))

            fig.update_layout(
                scene=dict(
                    xaxis_title='Q (A-1)',
                    yaxis_title='Normalized Intensity',
                    zaxis_title='', bgcolor='white'
                ),
                title='3D XRD Pattern with HKL Labels',
                width=800, height=600, showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not create 3D visualization: {str(e)}")

    with viz_tabs[1]:
        st.subheader("3D Crystal Structure")
        if not phases:
            st.info("No phases identified. Cannot display crystal structure.")
        elif not selected_phases:
            st.info("Select at least one phase above.")
        else:
            try:
                from crystal_structure_3d import CrystalStructure3D
                crystal_3d = CrystalStructure3D()
                for phase_name in selected_phases:
                    phase_data = next((p for p in phases if p.get("phase") == phase_name), None)
                    if not phase_data:
                        continue
                    st.markdown(f"### {phase_name}")
                    structure = crystal_3d.from_phase_data(phase_data)
                    if not isinstance(structure, dict):
                        st.warning(f"Could not generate a valid structure for {phase_name}.")
                        continue
                    if structure.get('error'):
                        st.warning(f"Could not generate exact structure for {phase_name}.")
                    fig = crystal_3d.create_3d_plot(structure, figsize=(10, 8))
                    st.pyplot(fig)
                    panel_export_buttons(fig, f"crystal_{phase_name}", f"crys_{phase_name}")
                    st.markdown("---")
            except Exception as e:
                st.error(f"Error generating crystal structure: {str(e)}")

    with viz_tabs[2]:
        st.subheader("Combined XRD and Crystal Structure")
        if not phases or not selected_phases:
            st.info("Select phases above to see combined view.")
        else:
            try:
                from crystal_structure_3d import CrystalStructure3D
                crystal_3d = CrystalStructure3D()
                for phase_name in selected_phases:
                    phase_data = next((p for p in phases if p.get("phase") == phase_name), None)
                    if not phase_data:
                        continue
                    st.markdown(f"### {phase_name}")
                    col_left, col_right = st.columns(2)
                    with col_left:
                        st.markdown("**XRD Pattern Contributions**")
                        phase_peaks = [p for p in structural_peaks if p.get("phase") == phase_name]
                        if phase_peaks:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(6, 4))
                            if results.get('xrd_raw'):
                                ax.plot(results['xrd_raw']['two_theta'],
                                        results['xrd_raw']['intensity'],
                                        'k-', linewidth=1, alpha=0.3)
                            positions = [p.get('position', 0) for p in phase_peaks]
                            ints = [p.get('intensity', 0) for p in phase_peaks]
                            ax.vlines(positions, 0, ints, colors='red', linewidth=2, alpha=0.7)
                            ax.plot(positions, ints, 'ro', markersize=4)
                            ax.set_xlabel('2theta (degrees)')
                            ax.set_ylabel('Intensity')
                            ax.set_title(f'{phase_name}')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            panel_export_buttons(fig, f"combined_xrd_{phase_name}",
                                                 f"comb_xrd_{phase_name}")
                    with col_right:
                        st.markdown("**Crystal Structure**")
                        structure = crystal_3d.from_phase_data(phase_data)
                        if isinstance(structure, dict) and not structure.get('error'):
                            fig = crystal_3d.create_3d_plot(structure, figsize=(6, 5))
                            st.pyplot(fig)
                            panel_export_buttons(fig, f"combined_crystal_{phase_name}",
                                                 f"comb_crys_{phase_name}")
                        else:
                            st.warning("Could not generate crystal structure")
                    st.markdown("---")
            except Exception as e:
                st.error(f"Error in combined view: {str(e)}")


@memory_safe_plot
def display_morphology(results):
    st.subheader("Morphology Visualization")
    st.info("Morphology figures are data-driven and deterministic, "
            "based on measured BET and XRD quantities.")

    try:
        visualizer = MorphologyVisualizer()
        fig = visualizer.create_pore_structure_2d(
            results.get('bet_results', {}) or {},
            results.get('xrd_results')
        )
        if fig is not None:
            st.pyplot(fig)
            panel_export_buttons(fig, "morphology_overview", "morph")
    except Exception as e:
        st.warning(f"Could not generate morphology figure: {str(e)}")


def display_methods(results, scientific_params):
    st.subheader("Scientific Methods and Calculations")

    with st.expander("A. BET Surface Area Analysis", expanded=True):
        st.markdown(
            "### BET Theory Fundamentals\n\n"
            "**BET Equation:**\n\n"
            "    p / (n(1-p)) = 1/(n_m * C) + (C-1)/(n_m * C) * p\n\n"
            "### IUPAC-Compliant Implementation\n\n"
            "**1. Rouquerol criteria for linear range selection.**\n\n"
            "**2. Surface area:**\n\n"
            "    S_BET = n_m * N_A * sigma * 10^-20\n\n"
            "**3. Error propagation:**\n\n"
            "    DeltaS_BET = S_BET * sqrt((Delta_n_m / n_m)^2 + (Delta_sigma / sigma)^2)\n\n"
            "**References:**\n"
            "1. Brunauer, S., Emmett, P. H., and Teller, E. (1938). J. Am. Chem. Soc., 60, 309-319.\n"
            "2. Rouquerol, J., Llewellyn, P., and Rouquerol, F. (2007). Stud. Surf. Sci. Catal., 160, 49-56.\n"
            "3. Thommes, M., et al. (2015). Pure Appl. Chem., 87, 1051-1069."
        )

    with st.expander("C. XRD Analysis Procedures", expanded=False):
        st.markdown(
            "### X-ray Diffraction Fundamentals\n\n"
            "**1. Bragg's Law:**\n\n"
            "    n * lambda = 2 * d * sin(theta)\n\n"
            "**2. Scherrer Equation:**\n\n"
            "    D = K * lambda / (beta * cos(theta))\n\n"
            "**3. Williamson-Hall Analysis:**\n\n"
            "    beta * cos(theta) = K * lambda / D + 4 * epsilon * sin(theta)\n\n"
            "**4. Crystallinity Index (Ruland):**\n\n"
            "    CI = A_crystalline / (A_crystalline + A_amorphous)\n\n"
            "**References:**\n"
            "1. Klug, H. P., and Alexander, L. E. (1974). X-ray Diffraction Procedures.\n"
            "2. Williamson, G. K., and Hall, W. H. (1953). Acta Metall., 1, 22-31.\n"
            "3. Ruland, W. (1961). Acta Cryst., 14, 1180."
        )


def display_validation(results):
    st.subheader("Scientific Validation")

    if 'validation' not in results:
        st.info("No validation data available.")
        return

    validation = results['validation']

    with st.expander("Detailed Validation Results", expanded=True):
        if validation.get('analysis_validation', {}).get('bet_checks'):
            st.subheader("BET Validation Checks")
            for check in validation['analysis_validation']['bet_checks']:
                if check['status'] == 'OK':
                    st.success(f"{check['check']}: {check['value']}")
                elif check['status'] == 'WARN':
                    st.warning(f"{check['check']}: {check['value']}")
                else:
                    st.error(f"{check['check']}: {check['value']}")

        if validation.get('analysis_validation', {}).get('xrd_checks'):
            st.subheader("XRD Validation Checks")
            for check in validation['analysis_validation']['xrd_checks']:
                if check['status'] == 'OK':
                    st.success(f"{check['check']}: {check['value']}")
                elif check['status'] == 'WARN':
                    st.warning(f"{check['check']}: {check['value']}")
                else:
                    st.error(f"{check['check']}: {check['value']}")


@memory_safe_plot
def display_export(results, scientific_params):
    st.subheader("Export Scientific Data")

    col1, col2 = st.columns(2)

    with col1:
        def build_export_data(results, scientific_params):
            return {
                "metadata": {
                    "app": "BET-XRD Analyzer",
                    "version": "1.0",
                    "timestamp": results.get("timestamp"),
                },
                "xrd": {
                    "phases": results.get("phases", []),
                    "phase_fractions": results.get("phase_fractions", []),
                    "crystallinity_index": results.get("crystallinity_index"),
                    "crystallite_size": results.get("crystallite_size"),
                    "microstrain": results.get("microstrain"),
                    "dislocation_density": results.get("dislocation_density"),
                    "crystal_system": results.get("crystal_system"),
                    "space_group": results.get("space_group"),
                    "lattice_parameters": results.get("lattice_parameters", {}),
                    "structural_peaks": (results.get("xrd_results") or {}).get("structural_peaks", []),
                    "n_detected_maxima": (results.get("xrd_results") or {}).get("n_detected_maxima", 0),
                    "parameters": scientific_params,
                }
            }

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                return super().default(obj)

        try:
            export_data = build_export_data(results, scientific_params)
            export_data = json.loads(json.dumps(export_data, cls=NumpyEncoder))
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="Download Complete Analysis (JSON)",
                data=json_str,
                file_name="scientific_analysis.json",
                mime="application/json",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Could not build JSON export: {str(e)}")

    with col2:
        try:
            report_text = generate_scientific_report(results)
            st.download_button(
                label="Download Scientific Report (TXT)",
                data=report_text,
                file_name="scientific_report.txt",
                mime="text/plain",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Could not build report: {str(e)}")

    # ============================================================
    # ZIP Export - All figures at once
    # ============================================================
    st.markdown("---")
    st.subheader("Download All Figures (ZIP)")
    st.caption("Package every available figure into a single ZIP archive.")

    plotter = PublicationPlotter(
        color_scheme=scientific_params['export']['color_scheme'],
        font_size=scientific_params['export']['font_size']
    )

    figures = {}

    if _has_bet_data(results):
        bet_raw = results['bet_raw']
        bet_res = results['bet_results']
        try:
            figures['bet_A_isotherm'] = plotter.plot_isotherm_only(bet_raw, bet_res)
            figures['bet_B_transform'] = plotter.plot_bet_transform_only(bet_raw, bet_res)
            figures['bet_C_tplot'] = plotter.plot_tplot_only(bet_raw, bet_res)
            figures['bet_D_psd'] = plotter.plot_psd_only(bet_res)
            figures['bet_E_hysteresis'] = plotter.plot_hysteresis_only(bet_raw, bet_res)
            figures['bet_F_summary_table'] = plotter.plot_bet_summary_table_only(bet_res)
            figures['bet_full_6panel'] = plotter.create_bet_figure(bet_raw, bet_res)
        except Exception as e:
            st.warning(f"Some BET figures could not be generated: {str(e)}")

    if _has_xrd_data(results):
        xrd_raw = results['xrd_raw']
        xrd_res = results['xrd_results']
        try:
            figures['xrd_A_pattern'] = plotter.plot_xrd_pattern_only(xrd_raw, xrd_res)
            figures['xrd_B_wh'] = plotter.plot_williamson_hall_only(xrd_res)
            figures['xrd_C_size_dist'] = plotter.plot_size_distribution_only(xrd_res)
            figures['xrd_D_peak_table'] = plotter.plot_xrd_peak_table_only(xrd_res)
            figures['xrd_E_summary_table'] = plotter.plot_xrd_summary_table_only(xrd_res)
            figures['xrd_full_5panel'] = plotter.create_xrd_figure(xrd_raw, xrd_res)
        except Exception as e:
            st.warning(f"Some XRD figures could not be generated: {str(e)}")

    if _has_multiple_xrd(results):
        try:
            overlay = XROverlayPlotter(
                color_scheme=scientific_params['export']['color_scheme'],
                font_size=scientific_params['export']['font_size']
            )
            wavelength = _get_wavelength(scientific_params)
            pattern_records = []
            for pat in results['xrd_patterns']:
                raw = pat.get('xrd_raw', {}) or {}
                pattern_records.append({
                    'label': pat.get('filename', 'unknown'),
                    'two_theta': raw.get('two_theta', []),
                    'intensity': raw.get('intensity', []),
                })
            figures['xrd_overlay_waterfall'] = overlay.plot(
                pattern_records, mode='waterfall', wavelength=wavelength)
            figures['xrd_overlay_overlay'] = overlay.plot(
                pattern_records, mode='overlay', wavelength=wavelength)
        except Exception as e:
            st.warning(f"Could not generate overlay figures: {str(e)}")

    if figures:
        colA, colB = st.columns(2)
        with colA:
            save_all_to_zip(figures, "all_figures_png.zip", fmt="png", dpi=600,
                            key="zip_png")
        with colB:
            save_all_to_zip(figures, "all_figures_pdf.zip", fmt="pdf", dpi=None,
                            key="zip_pdf")

        st.caption(f"Figures included in the ZIP: {', '.join(sorted(figures.keys()))}")

        for k, f in figures.items():
            close_fig_safely(f)
    else:
        st.info("No figures available yet. Run an analysis first.")


def generate_scientific_report(results):
    report = []
    report.append("=" * 70)
    report.append("SCIENTIFIC ANALYSIS REPORT - BET-XRD MORPHOLOGY ANALYSIS")
    report.append("=" * 70)
    report.append(f"Generated: {results.get('timestamp', 'N/A')}")
    report.append("Software Version: 3.0 (IUPAC Compliant)")
    report.append("")

    if _has_bet_data(results):
        bet = results['bet_results']
        report.append("BET SURFACE AREA ANALYSIS")
        report.append("-" * 40)
        report.append(f"Surface Area (SBET): "
                      f"{bet.get('surface_area', 0):.1f} +/- "
                      f"{bet.get('surface_area_error', 0):.1f} m2/g")
        report.append(f"Total Pore Volume: {bet.get('total_pore_volume', 0):.4f} cm3/g")
        report.append(f"Micropore Volume: {bet.get('micropore_volume', 0):.4f} cm3/g")
        report.append(f"Mean Pore Diameter: {bet.get('mean_pore_diameter', 0):.2f} nm")
        report.append(f"BET C Constant: {bet.get('c_constant', 0):.0f}")
        if bet.get('bet_regression'):
            report.append(f"BET R2: {bet['bet_regression'].get('r_squared', 0):.4f}")
        report.append("")

    if _has_xrd_data(results):
        xrd = results['xrd_results']
        report.append("XRD ANALYSIS")
        report.append("-" * 40)
        report.append(f"Crystallinity Index: {xrd.get('crystallinity_index', 0):.2f}")
        if xrd.get('crystallite_size'):
            report.append(f"Crystallite Size (Scherrer): "
                          f"{xrd['crystallite_size'].get('scherrer', 0):.1f} nm")
        ms = xrd.get('microstrain', None)
        report.append(f"Microstrain: {ms:.4f}" if ms is not None
                      else "Microstrain: not determined")
        report.append(f"Structural Bragg Peaks: {len(xrd.get('structural_peaks', []))}")
        report.append(f"Detected Local Maxima: {xrd.get('n_detected_maxima', 0)}")
        report.append("")

    patterns = results.get('xrd_patterns', [])
    if isinstance(patterns, list) and len(patterns) > 1:
        report.append(f"XRD FILES ANALYZED: {len(patterns)}")
        for i, pat in enumerate(patterns):
            report.append(f"  {i+1}. {pat.get('filename', 'unknown')}")
        report.append("")

    if results.get('fusion_results'):
        fusion = results['fusion_results']
        if isinstance(fusion, dict) and fusion.get('valid', False):
            report.append("INTEGRATED MORPHOLOGY ANALYSIS")
            report.append("-" * 40)
            if 'composite_classification' in fusion:
                report.append(f"Classification: {fusion['composite_classification']}")
            if 'material_family' in fusion:
                report.append(f"Family: {fusion['material_family']}")
            if 'dominant_feature' in fusion:
                report.append(f"Dominant Feature: {fusion['dominant_feature']}")
            report.append("")

    report.append("METHODS")
    report.append("-" * 40)
    report.append("BET Analysis: IUPAC Rouquerol criteria")
    report.append("XRD Analysis: Scherrer, Williamson-Hall methods")
    report.append("Porosity: t-plot, BJH methods")
    report.append("")

    report.append("REFERENCES")
    report.append("-" * 40)
    report.append("1. Rouquerol et al., Pure Appl. Chem., 1994, 66, 1739")
    report.append("2. Thommes et al., Pure Appl. Chem., 2015, 87, 1051")
    report.append("3. Klug & Alexander, X-ray Diffraction Procedures, 1974")

    return "\n".join(report)


if __name__ == "__main__":
    main()
