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

import streamlit as st
import numpy as np
import pandas as pd
import sys

import warnings
import io     # ADD THIS
import traceback
def safe_import(name):
    try:
        __import__(name)
        st.success(f"✅ Imported {name}")
        return sys.modules[name]
    except Exception as e:
        st.error(f"❌ Failed to import {name}")
        st.code("".join(traceback.format_exception(e)), language="python")
        st.stop()


# ===============================
# 🔍 STRICT IMPORT DIAGNOSTICS
# ===============================

st.write("🔎 Running import diagnostics...")

xrd_analyzer = safe_import("xrd_analyzer")
scientific_plots = safe_import("scientific_plots")
scientific_integration = safe_import("scientific_integration")
morphology_visualizer = safe_import("morphology_visualizer")
morphology_fusion = safe_import("morphology_fusion")
crystallography_engine = safe_import("crystallography_engine")
xrd_auto_solve = safe_import("xrd_auto_solve")
xrd_auto_indexer = safe_import("xrd_auto_indexer")
xrd_phase_identifier = safe_import("xrd_phase_identifier")

st.success("🎯 All core modules imported successfully")
warnings.filterwarnings('ignore')
import json  # <-- ADD THIS LINE
# Add to imports at the top of the file

# ADD THESE IMPORTS - FIX FOR SCIENTIFICINTEGRATOR ERROR
try:
    from scientific_integration import ScientificIntegrator
except ImportError:
    # Create a dummy class if import fails
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

# Import scientific engines
from bet_analyzer import IUPACBETAnalyzer, extract_asap2420_data
from xrd_analyzer import AdvancedXRDAnalyzer, extract_xrd_data
from morphology_fusion import MorphologyFusionEngine
from scientific_plots import PublicationPlotter
# At the top of app.py with other imports
from morphology_visualizer import MorphologyVisualizer

import functools

def memory_safe_plot(func):
    """Decorator to ensure figures are closed after display"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Call the function
            result = func(*args, **kwargs)
            
            # Clean up matplotlib
            import matplotlib.pyplot as plt
            plt.close('all')
            
            # Force garbage collection
            import gc
            gc.collect()
            
            return result
        except Exception as e:
            # Clean up even on error
            import matplotlib.pyplot as plt
            plt.close('all')
            raise e
    return wrapper

# Use it like this:
# ============================================================================
# SCIENTIFIC CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="BET–XRD Morphology Analyzer | Journal Edition",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'scientific_data' not in st.session_state:
    st.session_state.scientific_data = {
        'bet_raw': None,
        'xrd_raw': None,
        'bet_results': None,
        'xrd_results': None,
        'fusion_results': None,
        'analysis_valid': False
    }

# ============================================================================
# SIDEBAR - SCIENTIFIC CONTROLS
# ============================================================================
def create_sidebar():
    """Scientific controls sidebar"""
    with st.sidebar:
        st.title("⚗️ Scientific Controls")
        
        st.markdown("---")
        st.subheader("BET Analysis Parameters")
        
        bet_params = {
            'gas': st.selectbox(
                "Adsorbate Gas",
                ["N₂ (77 K)", "Ar (87 K)", "CO₂ (273 K)"],
                index=0
            ),
            'cross_section': st.number_input(
                "Cross-section (nm²)",
                value=0.162,
                min_value=0.1,
                max_value=0.3,
                step=0.001,
                help="N₂: 0.162 nm², Ar: 0.142 nm², CO₂: 0.187 nm²"
            ),
            'bet_range_min': st.number_input(
                "BET min P/P₀",
                value=0.05,
                min_value=0.01,
                max_value=0.1,
                step=0.01,
                help="IUPAC recommends 0.05-0.35"
            ),
            'bet_range_max': st.number_input(
                "BET max P/P₀",
                value=0.35,
                min_value=0.2,
                max_value=0.5,
                step=0.01
            )
        }
        
        st.markdown("---")
        st.subheader("XRD Analysis Parameters")
        
        xrd_params = {
            'wavelength': st.selectbox(
                "X-ray Wavelength",
                ["Cu Kα (0.15406 nm)", "Mo Kα (0.07107 nm)", "Co Kα (0.17902 nm)"],
                index=0
            ),
            'background_subtraction': st.checkbox(
                "Background Subtraction",
                value=True,
                help="Remove amorphous background using SNIP algorithm"
            ),
            'peak_threshold': st.slider(
                "Peak Detection Threshold",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01
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
                "Space Group",
                value="",
                help="e.g., Fd-3m, P6₃/mmc, Im-3m"
            ),
            'lattice_params': st.text_input(
                "Lattice Parameters (Å)",
                value="",
                help="e.g., a=4.05 (Cubic), a=4.05 c=6.7 (Hexagonal)"
            ),
            'composition': st.text_input(  # NEW
                "Material Composition",
                value="SiO2",
                help="e.g., SiO2, TiO2, Al2O3, ZrO2, CeO2"
            )
        }
        
        # NEW: Advanced crystallography options
        with st.expander("Advanced Crystallography", expanded=False):
            crystal_params['enable_3d'] = st.checkbox(
                "Generate 3D Crystal Structure",
                value=True,
                help="Create 3D visualization of crystal structure"
            )
            
            crystal_params['show_interactive'] = st.checkbox(
                "Interactive 3D View",
                value=False,
                help="Show interactive 3D visualization (requires Plotly)"
            )
            
            crystal_params['supercell_size'] = st.slider(
                "Supercell Size",
                min_value=1,
                max_value=4,
                value=2,
                help="Number of unit cells in each direction"
            )
        
        
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
                "Font Size (pt)",
                min_value=8,
                max_value=14,
                value=10,
                step=1
            )
        }
        # ============================================================
        # XRD PHASE SEARCH SETTINGS
        # ============================================================
        from pymatgen.core import Element
        
        st.sidebar.subheader("🧪 Expected Elements in Sample")
        
        ALL_ELEMENTS = [el.symbol for el in Element]
        selected_elements = st.sidebar.multiselect(
            "Select elements present in your sample",
            options=ALL_ELEMENTS,
            help="Used for phase identification (COD + OPTIMADE)"
        )
        
        # Fallback safety
        if not selected_elements:
            st.sidebar.warning("No elements selected → phase identification disabled")

        st.session_state["xrd_elements"] = selected_elements

        st.markdown("---")
        st.subheader("Scientific References")
        
        with st.expander("View References"):
            st.markdown("""
            **BET Analysis:**
            1. Rouquerol, J.; Llewellyn, P.; Rouquerol, F. *Stud. Surf. Sci. Catal.* **2007**, 160, 49-56.
            2. Thommes, M. et al. *Pure Appl. Chem.* **2015**, 87, 1051-1069.
            
            **XRD Analysis:**
            1. Klug, H. P.; Alexander, L. E. *X-ray Diffraction Procedures*, 2nd ed.; Wiley: 1974.
            2. Williamson, G. K.; Hall, W. H. *Acta Metall.* **1953**, 1, 22-31.
            
            **Porosity Analysis:**
            1. Barrett, E. P.; Joyner, L. G.; Halenda, P. P. *J. Am. Chem. Soc.* **1951**, 73, 373-380.
            2. Harkins, W. D.; Jura, G. *J. Am. Chem. Soc.* **1944**, 66, 1366-1373.
            """)
        
        return {
            'bet': bet_params,
            'xrd': xrd_params,
            'crystal': crystal_params,
            'export': export_params
        }

# ============================================================================
# FILE UPLOAD SECTION
# ============================================================================
def file_upload_section():
    """Professional file upload with validation"""
    st.header("📁 Experimental Data Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Physisorption Data")
        
        bet_file = st.file_uploader(
            "Upload BET Isotherm File",
            type=["xls", "xlsx", "csv", "txt", "dat"],
            help="""**Supported formats:**
            - Micromeritics ASAP 2420 (.xls, .xlsx)
            - Quantachrome (.csv)
            - BELSORP (.txt)
            - Custom 2-4 column format"""
        )
        
        if bet_file:
            st.success(f"✅ BET file: {bet_file.name}")
            
            # Preview and validate
            with st.expander("🔍 Preview & Validate", expanded=False):
                try:
                    preview_data = extract_asap2420_data(bet_file, preview_only=True)
                    if preview_data:
                        st.write(f"**Format detected:** {preview_data['format']}")
                        st.write(f"**Points found:** {preview_data['n_points']}")
                        st.write(f"**Pressure range:** {preview_data['p_range'][0]:.3f} - {preview_data['p_range'][1]:.3f} P/P₀")
                        
                        if preview_data['adsorption']:
                            st.write("**Adsorption data preview:**")
                            st.dataframe(pd.DataFrame({
                                'P/P₀': preview_data['adsorption']['p'][:10],
                                'Q (mmol/g)': preview_data['adsorption']['q'][:10]
                            }))
                    
                    # Show raw data structure
                    bet_file.seek(0)
                    if bet_file.name.endswith('.xls'):
                        df_preview = pd.read_excel(bet_file, engine='xlrd', nrows=20)
                    elif bet_file.name.endswith('.xlsx'):
                        df_preview = pd.read_excel(bet_file, engine='openpyxl', nrows=20)
                    else:
                        df_preview = pd.read_csv(bet_file, nrows=20)
                    
                    st.write("**Raw file structure:**")
                    st.dataframe(df_preview)
                    
                except Exception as e:
                    st.error(f"Preview error: {str(e)}")
    
    with col2:
        st.subheader("XRD Data")
        
        xrd_file = st.file_uploader(
            "Upload XRD Pattern File",
            type=["csv", "txt", "xy", "dat", "xrdml"],
            help="""**Required format:**
            - Two columns: 2θ (degrees) and Intensity
            - Tab, comma, or space separated
            - Minimum 100 data points
            - 2θ range typically 5-80°"""
        )
        
        if xrd_file:
            st.success(f"✅ XRD file: {xrd_file.name}")
            
            # Preview and validate
            with st.expander("🔍 Preview & Validate", expanded=False):
                try:
                    theta, intensity, msg = extract_xrd_data(xrd_file, preview_only=True)
                    if theta is not None:
                        st.write(f"**Validation:** {msg}")
                        st.write(f"**Data points:** {len(theta)}")
                        st.write(f"**2θ range:** {theta[0]:.2f} - {theta[-1]:.2f}°")
                        st.write(f"**Intensity range:** {intensity.min():.2f} - {intensity.max():.2f}")
                        
                        # Quick plot preview
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(theta, intensity, 'k-', linewidth=0.5)
                        ax.set_xlabel('2θ (degrees)')
                        ax.set_ylabel('Intensity')
                        ax.set_title('XRD Pattern Preview')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close(fig)
                    
                except Exception as e:
                    st.error(f"Preview error: {str(e)}")
    
    return bet_file, xrd_file
# ============================================================================
# SCIENTIFIC VALIDATION FUNCTIONS
# ============================================================================
def validate_input_data(bet_file, xrd_file, params):
    """Validate input data for scientific analysis"""
    validation_results = {
        'bet_valid': False,
        'xrd_valid': False,
        'warnings': [],
        'recommendations': []
    }
    
    # Validate BET data
    if bet_file:
        try:
            bet_file.seek(0)
            
            # Check file format
            filename = bet_file.name.lower()
            valid_formats = ['.xls', '.xlsx', '.csv', '.txt', '.dat']
            if not any(filename.endswith(fmt) for fmt in valid_formats):
                validation_results['warnings'].append(f"BET file format {filename} may not be optimal")
            
            # Try to read and check content
            if filename.endswith('.xls'):
                df = pd.read_excel(bet_file, engine='xlrd', nrows=100)
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(bet_file, engine='openpyxl', nrows=100)
            else:
                try:
                    df = pd.read_csv(bet_file, nrows=100)
                except:
                    # Try with different delimiters
                    content = bet_file.read().decode('utf-8')
                    bet_file.seek(0)
                    for delimiter in ['\t', ';', ',', ' ']:
                        try:
                            df = pd.read_csv(io.StringIO(content), 
                                           delimiter=delimiter, 
                                           nrows=100)
                            break
                        except:
                            continue
            
            # Check minimum data requirements
            n_rows = len(df)
            if n_rows < 10:
                validation_results['warnings'].append(f"BET file has only {n_rows} rows (minimum 10 recommended)")
            else:
                validation_results['bet_valid'] = True
            
            # Check for pressure column
            numeric_cols = df.apply(pd.to_numeric, errors='coerce').notna().sum()
            if numeric_cols.sum() < 2:
                validation_results['warnings'].append("BET file may not contain numeric pressure/quantity columns")
            
            bet_file.seek(0)
            
        except Exception as e:
            validation_results['warnings'].append(f"BET file reading error: {str(e)[:100]}")
    
    # Validate XRD data
    if xrd_file:
        try:
            xrd_file.seek(0)
            
            # Check file format
            filename = xrd_file.name.lower()
            valid_formats = ['.csv', '.txt', '.xy', '.dat', '.xrdml']
            if not any(filename.endswith(fmt) for fmt in valid_formats):
                validation_results['warnings'].append(f"XRD file format {filename} may not be optimal")
            
            # Try to extract data
            two_theta, intensity, msg = extract_xrd_data(xrd_file, preview_only=True)
            
            if two_theta is not None:
                n_points = len(two_theta)
                theta_range = two_theta.max() - two_theta.min()
                
                if n_points < 100:
                    validation_results['warnings'].append(f"XRD file has only {n_points} points (minimum 100 recommended)")
                
                if theta_range < 10:
                    validation_results['warnings'].append(f"XRD angular range is only {theta_range:.1f}° (minimum 10° recommended)")
                
                if n_points >= 50 and theta_range >= 5:
                    validation_results['xrd_valid'] = True
                else:
                    validation_results['recommendations'].append("Consider using XRD data with wider angular range (5-80° 2θ)")
            
            xrd_file.seek(0)
            
        except Exception as e:
            validation_results['warnings'].append(f"XRD file reading error: {str(e)[:100]}")
    
    return validation_results

def perform_analysis_validation(results):
    """Perform scientific validation of analysis results"""
    validation = {
        'bet_checks': [],
        'xrd_checks': [],
        'consistency_checks': [],
        'all_passed': True
    }
    
    # BET Validation
    if results.get('bet_results'):
        bet = results['bet_results']
        
        # Check BET linearity
        if bet.get('bet_regression', {}).get('r_squared', 0) > 0.999:
            validation['bet_checks'].append({'check': 'BET Linearity', 'status': '✅', 'value': f"R² = {bet['bet_regression']['r_squared']:.6f}"})
        elif bet.get('bet_regression', {}).get('r_squared', 0) > 0.995:
            validation['bet_checks'].append({'check': 'BET Linearity', 'status': '⚠️', 'value': f"R² = {bet['bet_regression']['r_squared']:.6f} (moderate)"})
            validation['all_passed'] = False
        else:
            validation['bet_checks'].append({'check': 'BET Linearity', 'status': '❌', 'value': f"R² = {bet['bet_regression']['r_squared']:.6f} (poor)"})
            validation['all_passed'] = False
        
        # Check C constant
        if bet.get('c_constant', 0) > 0:
            validation['bet_checks'].append({'check': 'BET C Constant', 'status': '✅', 'value': f"C = {bet['c_constant']:.0f}"})
        else:
            validation['bet_checks'].append({'check': 'BET C Constant', 'status': '❌', 'value': 'Negative or zero C constant'})
            validation['all_passed'] = False
        
        # Check surface area range
        S_bet = bet.get('surface_area', 0)
        if 0 < S_bet < 10000:
            validation['bet_checks'].append({'check': 'Surface Area Range', 'status': '✅', 'value': f"{S_bet:.1f} m²/g"})
        elif S_bet >= 10000:
            validation['bet_checks'].append({'check': 'Surface Area Range', 'status': '⚠️', 'value': f"{S_bet:.1f} m²/g (unusually high)"})
        else:
            validation['bet_checks'].append({'check': 'Surface Area Range', 'status': '❌', 'value': 'Invalid surface area'})
            validation['all_passed'] = False
    
    # XRD Validation
    if results.get('xrd_results'):
        xrd = results['xrd_results']
        
        # Check number of peaks
        n_peaks = len(xrd.get('peaks', []))
        if n_peaks >= 3:
            validation['xrd_checks'].append({'check': 'Number of Peaks', 'status': '✅', 'value': f"{n_peaks} peaks detected"})
        else:
            validation['xrd_checks'].append({'check': 'Number of Peaks', 'status': '⚠️', 'value': f"Only {n_peaks} peaks detected"})
            validation['all_passed'] = False
        
        # Check crystallinity index
        ci = xrd.get('crystallinity_index', 0)
        if 0 <= ci <= 1:
            validation['xrd_checks'].append({'check': 'Crystallinity Index', 'status': '✅', 'value': f"{ci:.3f}"})
        else:
            validation['xrd_checks'].append({'check': 'Crystallinity Index', 'status': '❌', 'value': f"{ci:.3f} (outside 0-1 range)"})
            validation['all_passed'] = False
        
        # Check crystallite size
        size = xrd.get('crystallite_size', {}).get('scherrer', 0)
        if 0 < size < 1000:
            validation['xrd_checks'].append({'check': 'Crystallite Size', 'status': '✅', 'value': f"{size:.1f} nm"})
        elif size >= 1000:
            validation['xrd_checks'].append({'check': 'Crystallite Size', 'status': '⚠️', 'value': f"{size:.1f} nm (unusually large)"})
            validation['all_passed'] = False
    
    # Consistency Checks (if both BET and XRD available)
    if results.get('bet_results') and results.get('xrd_results'):
        bet = results['bet_results']
        xrd = results['xrd_results']
        
        S_bet = bet.get('surface_area', 0)
        D_xrd = xrd.get('crystallite_size', {}).get('scherrer', 0)
        CI = xrd.get('crystallinity_index', 0)
        
        # Theoretical surface area from crystallite size
        if D_xrd > 0 and CI > 0.3:
            # For spherical particles: S = 6/(ρ·D)
            rho = 2.65  # g/cm³ (typical for oxides)
            S_theoretical = 6000 / (rho * D_xrd) * CI  # m²/g
            
            if S_theoretical > 0:
                ratio = S_bet / S_theoretical
                if 0.1 < ratio < 10:
                    validation['consistency_checks'].append({
                        'check': 'BET-XRD Consistency',
                        'status': '✅',
                        'value': f"Ratio S_BET/S_theoretical = {ratio:.2f}"
                    })
                else:
                    validation['consistency_checks'].append({
                        'check': 'BET-XRD Consistency',
                        'status': '⚠️',
                        'value': f"Ratio S_BET/S_theoretical = {ratio:.2f} (outside 0.1-10 range)"
                    })
                    validation['all_passed'] = False
    
    return validation
# ============================================================================
# SCIENTIFIC ANALYSIS EXECUTION
# ============================================================================
def execute_scientific_analysis(bet_file, xrd_file, params):
    """Execute complete scientific analysis pipeline"""
    
    analysis_results = {}
        
    # ============================================================================
    # INPUT VALIDATION - ADD THIS SECTION AT THE START
    # ============================================================================
    st.subheader("🔬 Input Data Validation")
    
    validation_results = validate_input_data(bet_file, xrd_file, params)
    
    # Display validation results
    col1, col2 = st.columns(2)
    
    with col1:
        if bet_file:
            if validation_results['bet_valid']:
                st.success("✅ BET data: Valid format and sufficient points")
            else:
                st.warning("⚠️ BET data: Validation warnings")
    
    with col2:
        if xrd_file:
            if validation_results['xrd_valid']:
                st.success("✅ XRD data: Valid format and sufficient points")
            else:
                st.warning("⚠️ XRD data: Validation warnings")
    
    # Show warnings
    if validation_results['warnings']:
        with st.expander("🔄 Data Quality Warnings", expanded=True):
            for warning in validation_results['warnings']:
                st.warning(f"• {warning}")
    
    # Show recommendations
    if validation_results['recommendations']:
        with st.expander("📋 Data Quality Recommendations", expanded=False):
            for rec in validation_results['recommendations']:
                st.info(f"• {rec}")
    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # ====================================================================
        # STEP 1: DATA EXTRACTION
        # ====================================================================
        status_text.text("📥 Extracting experimental data...")
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
                st.success(f"✅ BET data extracted: {len(p_ads)} adsorption points")
                
                if p_des is not None:
                    st.success(f"✅ Desorption data: {len(p_des)} points")
            else:
                st.error(f"❌ BET extraction failed: {extraction_msg}")
                # Don't return None, continue with XRD if available
        
        if xrd_file:
            xrd_file.seek(0)
            two_theta, intensity, extraction_msg = extract_xrd_data(xrd_file)
            
            if two_theta is not None and len(two_theta) >= 50:
                analysis_results['xrd_raw'] = {
                    'two_theta': two_theta,
                    'intensity': intensity,
                    'extraction_info': extraction_msg
                }
                st.success(f"✅ XRD data extracted: {len(two_theta)} points")
            else:
                st.error(f"❌ XRD extraction failed: {extraction_msg}")
                # Don't return None, continue with BET if available
        
        # ====================================================================
        # STEP 2: BET ANALYSIS
        # ====================================================================
        if 'bet_raw' in analysis_results:
            status_text.text("📊 Performing IUPAC-compliant BET analysis...")
            progress_bar.progress(30)
            
            try:
                # Get gas properties
                gas_type = params['bet']['gas']
                if "N₂" in gas_type:
                    cross_section = 0.162e-18  # m²
                    temperature = 77.3  # K
                elif "Ar" in gas_type:
                    cross_section = 0.142e-18  # m²
                    temperature = 87.3  # K
                else:  # CO₂
                    cross_section = 0.187e-18  # m²
                    temperature = 273.15  # K
                
                bet_analyzer = IUPACBETAnalyzer(
                    p_ads=analysis_results['bet_raw']['p_ads'],
                    q_ads=analysis_results['bet_raw']['q_ads'],
                    p_des=analysis_results['bet_raw']['p_des'],
                    q_des=analysis_results['bet_raw']['q_des'],
                    cross_section=cross_section,
                    temperature=temperature
                )
                
                analysis_results['bet_results'] = bet_analyzer.complete_analysis()
                
                # Display results (even if BET failed)
                bet_res = analysis_results['bet_results']
                
                if bet_res.get('bet_valid', False):
                    st.success("✅ BET analysis completed successfully")
                    
                    # Display key results
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Sᴮᴱᵀ", f"{bet_res['surface_area']:.1f} ± {bet_res['surface_area_error']:.1f} m²/g")
                    with col2:
                        st.metric("Vₚ", f"{bet_res['total_pore_volume']:.3f} cm³/g")
                    with col3:
                        st.metric("C", f"{bet_res['c_constant']:.0f}")
                    with col4:
                        st.metric("R²", f"{bet_res['bet_regression']['r_squared']:.4f}")
                else:
                    # BET failed but other analyses may still work
                    st.warning(f"⚠️ BET surface area calculation failed: {bet_res.get('bet_error', 'Unknown error')}")
                    
                    # Still show available results
                    if bet_res.get('total_pore_volume', 0) > 0:
                        st.info(f"✅ Total pore volume: {bet_res['total_pore_volume']:.3f} cm³/g")
                    if bet_res.get('mean_pore_diameter', 0) > 0:
                        st.info(f"✅ Mean pore diameter: {bet_res['mean_pore_diameter']:.2f} nm")
                    if bet_res.get('micropore_volume', 0) > 0:
                        st.info(f"✅ Micropore volume: {bet_res['micropore_volume']:.3f} cm³/g")
                    
                    # Show detailed error message
                    with st.expander("🔍 BET Analysis Details", expanded=False):
                        st.write(f"**Issue:** {bet_res.get('bet_error', 'No valid BET range found')}")
                        st.write("""
                        **Possible reasons:**
                        1. Non-porous or low-surface-area material
                        2. Microporous material (BET not applicable below 0.05 P/P₀)
                        3. Measurement issues (insufficient data points in BET range)
                        4. Adsorbate-adsorbent interactions causing non-linearity
                        
                        **Recommendations:**
                        - Check adsorption isotherm shape
                        - Consider alternative methods (t-plot, Dubinin-Radushkevich)
                        - Verify experimental conditions
                        """)
                    
            except Exception as e:
                st.error(f"❌ BET analysis error: {str(e)}")
                analysis_results['bet_results'] = {
                    'overall_valid': False,
                    'bet_valid': False,
                    'error': str(e),
                    'surface_area': 0.0,
                    'surface_area_error': 0.0,
                    'total_pore_volume': 0.0,
                    'mean_pore_diameter': 0.0
                }
        
        # ====================================================================
        # STEP 3: XRD ANALYSIS
        # ====================================================================
        # ====================================================================
        # STEP 3: XRD ANALYSIS - ENHANCED WITH CRYSTALLOGRAPHY ENGINE
        # ====================================================================
        if 'xrd_raw' in analysis_results:
            status_text.text("📈 Performing advanced XRD analysis...")
            progress_bar.progress(60)
            
            try:
                # Get wavelength
                wavelength_str = params['xrd']['wavelength']
                if "Cu" in wavelength_str:
                    wavelength = 0.15406
                elif "Mo" in wavelength_str:
                    wavelength = 0.07107
                else:  # Co
                    wavelength = 0.17902
                
                xrd_analyzer = AdvancedXRDAnalyzer(
                    wavelength=wavelength,
                    background_subtraction=params['xrd']['background_subtraction'],
                    smoothing=params['xrd']['smoothing']
                )
                
                xrd_results = xrd_analyzer.complete_analysis(
                    two_theta=analysis_results['xrd_raw']['two_theta'],
                    intensity=analysis_results['xrd_raw']['intensity'],
                    elements=st.session_state.get("xrd_elements", [])
                )
                xrd_out = xrd_analyzer.complete_analysis(
                    two_theta=analysis_results['xrd_raw']['two_theta'],
                    intensity=analysis_results['xrd_raw']['intensity'],
                    elements=st.session_state.get("xrd_elements", [])
                )
                
                # 🔧 GLOBAL NORMALIZATION FIX (ADD THIS)
                if xrd_out.get("valid") and "xrd_results" in xrd_out:
                    xrd_results = xrd_out["xrd_results"]
                else:
                    xrd_results = {
                        "peaks": [],
                        "crystallinity_index": 0.0,
                        "crystallite_size": {},
                        "phases": []
                    }
                # 🔍 DEBUG — PUT THIS HERE

                # ===============================
                # NORMALIZE PHASE IDENTIFICATION
                # ===============================
                if xrd_results.get("primary_phase"):
                    primary = xrd_results["primary_phase"]
                
                    xrd_results["crystal_system"] = primary.get("crystal_system", "Unknown")
                    xrd_results["space_group"] = primary.get("space_group", "")
                    xrd_results["lattice_parameters"] = primary.get("lattice", {})
                
                    # Overwrite HKL into peaks (UI expects this)
                    for p in xrd_results.get("peaks", []):
                        if "hkl" not in p:
                            p["hkl"] = ""
                
                else:
                    xrd_results.setdefault("crystal_system", "Unknown")
                    xrd_results.setdefault("space_group", "")
                    xrd_results.setdefault("lattice_parameters", {})
                    # SAFETY MIRROR (UI STABILITY)
                xrd_results["xrd_results"] = xrd_results

                # After performing XRD analysis, add:
                if 'xrd_results' in analysis_results:
                    # Try to find missing hkl indices
                    analysis_results['xrd_results'] = find_missing_hkl_indices(
                        analysis_results['xrd_results'],
                        params['crystal']
                    )
                # ENHANCE WITH CRYSTALLOGRAPHY ENGINE FOR BETTER HKL INDEXING
                if (xrd_results.get("peaks") and 
                    params['crystal']['system'] != 'Unknown' and 
                    params['crystal']['lattice_params']):
                    
                    try:
                        # Initialize crystallography engine
                        from crystallography_engine import CrystallographyEngine
                        ce = CrystallographyEngine()
                        
                        # Parse lattice parameters
                        lattice_dict = {}
                        import re
                        lattice_str = params['crystal']['lattice_params']
                        for match in re.finditer(r'([abc])\s*=\s*([\d\.]+)', lattice_str):
                            lattice_dict[match.group(1)] = float(match.group(2))
                        
                        # Index peaks using crystallography engine
                        peak_positions = [p['position'] for p in xrd_results['peaks']]
                        indexing_result = ce.index_peaks(
                            peak_positions=peak_positions,
                            crystal_system=params['crystal']['system'],
                            lattice_params=lattice_dict,
                            wavelength=wavelength,
                            space_group=params['crystal']['space_group']
                        )
                        
                        # Update peaks with better hkl indexing
                        indexed_peaks = indexing_result.get('indexed_peaks', [])
                        for i, peak in enumerate(xrd_results['peaks']):
                            if i < len(indexed_peaks):
                                hkl_info = indexed_peaks[i]
                                peak['hkl'] = f"({hkl_info['h']}{hkl_info['k']}{hkl_info['l']})"
                                peak['hkl_detail'] = hkl_info
                                peak['indexing_error'] = hkl_info['error_percent']
                        
                        # Add indexing results to XRD results
                        xrd_results['indexing'] = indexing_result
                        xrd_results['indexing_method'] = 'Pawley-like refinement'
                        
                    except Exception as e:
                        st.warning(f"Crystallography engine: {str(e)}")
                
                analysis_results['xrd_results'] = xrd_results
                
                if xrd_results.get("peaks"):
                    st.success("✅ XRD analysis completed successfully")
                    
                    # Display key results
                    xrd_res = xrd_results
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Crystallinity", f"{xrd_results["xrd_results"]["crystallinity_index"]:.2f}")
                    with col2:
                        size = size = xrd_res["xrd_results"]["crystallite_size"]['scherrer']
                        st.metric("Size", f"{size:.1f} nm" if size else "N/A")
                    with col3:
                        st.metric("Peaks", f"{len(xrd_res['peaks'])}")
                    with col4:
                        # Show indexing quality if available
                        if 'indexing' in xrd_res:
                            fom = xrd_res['indexing']['figures_of_merit']
                            st.metric("M₂₀", f"{fom.get('M20', 0):.1f}")
                        else:
                            st.metric("Ordered", "Yes" if xrd_res['ordered_mesopores'] else "No")
                
                else:
                    st.warning(f"⚠️ XRD analysis completed with warnings: {xrd_results.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ XRD analysis error: {str(e)}")
                analysis_results['xrd_results'] = {
                    'valid': False,
                    'error': str(e),
                    'wavelength': wavelength,
                    'peaks': [],
                    'crystallinity_index': 0.0,
                    'crystallite_size': {'scherrer': 0.0, 'williamson_hall': 0.0, 'distribution': 'Unknown'},
                    'microstrain': 0.0,
                    'ordered_mesopores': False
                }
        
        # ====================================================================
        # STEP 4: MORPHOLOGY FUSION (if we have any valid results)
        # ====================================================================
        bet_valid = analysis_results.get('bet_results', {}).get('overall_valid', False)
        xrd_valid = analysis_results.get('xrd_results', {}).get('valid', False)
        
        if bet_valid or xrd_valid:
            status_text.text("🧬 Fusing BET-XRD morphology data...")
            progress_bar.progress(80)
            
            try:
                fusion_engine = MorphologyFusionEngine()
                analysis_results['fusion_results'] = fusion_engine.fuse(
                    bet_results=analysis_results.get('bet_results', {}),
                    xrd_results=analysis_results.get('xrd_results', {})
                )
                
                if analysis_results['fusion_results']['valid']:
                    st.success("✅ Morphology fusion completed")
                    
                    # Display fusion classification
                    fusion = analysis_results['fusion_results']
                    st.info(f"**Material Classification:** {fusion['composite_classification']}")
                    st.info(f"**Dominant Feature:** {fusion['dominant_feature']}")
                
                else:
                    st.warning("⚠️ Morphology fusion completed with warnings")
                    
            except Exception as e:
                st.error(f"❌ Morphology fusion error: {str(e)}")
                # Continue anyway as fusion is optional
                # ====================================================================
        # STEP 4: SCIENTIFIC INTEGRATION (NEW)
        # ====================================================================
        bet_valid = analysis_results.get('bet_results', {}).get('overall_valid', False)
        xrd_valid = analysis_results.get('xrd_results', {}).get('valid', False)
        
        if bet_valid or xrd_valid:
            status_text.text("🔗 Performing scientific integration...")
            progress_bar.progress(85)
            
            try:
                # Initialize scientific integrator
                integrator = ScientificIntegrator()
                
                # Perform scientific integration
                integration_results = integrator.integrate_results(
                    bet_results=analysis_results.get('bet_results', {}),
                    xrd_results=analysis_results.get('xrd_results', {})
                )
                
                if integration_results['valid']:
                    analysis_results['integration'] = integration_results
                    st.success("✅ Scientific integration completed")
                    
                    # Display integration summary
                    with st.expander("🔗 Integration Summary", expanded=False):
                        if 'material_classification' in integration_results:
                            classification = integration_results['material_classification']
                            st.write(f"**Classification:** {classification.get('primary', 'Unknown')}")
                        
                        if 'validation_metrics' in integration_results:
                            validation = integration_results['validation_metrics']
                            if 'internal_consistency' in validation:
                                st.write(f"**Internal Consistency:** {validation['internal_consistency']:.2f}")
                
                else:
                    st.warning("⚠️ Scientific integration completed with warnings")
                    
            except Exception as e:
                st.error(f"❌ Scientific integration error: {str(e)}")
        # ====================================================================
        # STEP 5: FINALIZATION - ADD VALIDATION METRICS
        # ====================================================================
        status_text.text("🎨 Preparing scientific outputs...")
        progress_bar.progress(95)
        
        # Determine if we have any valid results
        has_bet_data = 'bet_results' in analysis_results
        has_xrd_data = 'xrd_results' in analysis_results
        has_fusion_data = 'fusion_results' in analysis_results
        
        analysis_results['analysis_valid'] = (has_bet_data or has_xrd_data)
        analysis_results['parameters'] = params
        analysis_results['timestamp'] = pd.Timestamp.now().isoformat()
        
        # ============================================================================
        # ADD SCIENTIFIC VALIDATION METRICS - NEW SECTION
        # ============================================================================
        analysis_results['validation'] = {
            'input_validation': validation_results,
            'analysis_validation': perform_analysis_validation(analysis_results)
        }
        
        progress_bar.progress(100)
        
        if analysis_results['analysis_valid']:
            # Display validation summary
            if 'analysis_validation' in analysis_results['validation']:
                av = analysis_results['validation']['analysis_validation']
                if av.get('all_passed', False):
                    st.success("✅ All scientific validation checks passed")
                else:
                    st.warning("⚠️ Some validation checks have warnings")
            
            status_text.text("✅ Analysis complete!")
            st.balloons()
        else:
            status_text.text("⚠️ Partial analysis complete (some analyses failed)")
            st.warning("Some analyses failed, but available results are shown below")
        
        return analysis_results
        
    except Exception as e:
        st.error(f"❌ Analysis pipeline failed: {str(e)}")
        import traceback
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
        return None

       
# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application interface"""
    
    # Title and introduction
    st.title("🔬 BET–XRD Morphology Analyzer")
    st.markdown("""
    **Scientific Edition for Journal Publication**
    
    This application performs IUPAC-compliant physisorption analysis and advanced XRD characterization
    for comprehensive porous materials morphology determination. All calculations follow established
    scientific literature and include proper error analysis.
    """)
    
    # Get scientific parameters
    scientific_params = create_sidebar()
    
    # File upload
    bet_file, xrd_file = file_upload_section()
    
    # Analysis button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        analyze_button = st.button(
            "🚀 EXECUTE SCIENTIFIC ANALYSIS",
            type="primary",
            use_container_width=True,
            disabled=not (bet_file or xrd_file)
        )
    
    # Execute analysis
    if analyze_button:
        with st.spinner("Initializing scientific analysis pipeline..."):
            results = execute_scientific_analysis(bet_file, xrd_file, scientific_params)
            
            if results:
                st.session_state.scientific_data = results
    
    # Display results if available
    if st.session_state.scientific_data.get('analysis_valid', False):
        display_scientific_results(st.session_state.scientific_data, scientific_params)
    
    # Footer
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption(f"Python {sys.version.split()[0]}")
        st.caption("NumPy {np.__version__}")
    
    with col2:
        st.caption("**Scientific Software v3.0**")
        st.caption("For Journal Publication")
    
    with col3:
        st.caption("© 2024 Materials Science Laboratory")
        st.caption("IUPAC Standards Compliant")

# ============================================================================
# RESULTS DISPLAY
# ============================================================================
def display_scientific_results(results, scientific_params):
    """Display comprehensive scientific results"""
    
    st.header("📊 Scientific Results")
    
    # ============================================================================
    # FIXED TAB STRUCTURE - Handle variable number of tabs
    # ============================================================================
    
    # Determine which tabs to show
    show_crystal_tab = (
        scientific_params['crystal']['system'] != 'Unknown' and 
        scientific_params['crystal']['lattice_params'] and
        scientific_params['crystal'].get('enable_3d', False)
    )
    
    # Define all possible tabs
    all_tabs = [
        "📈 Overview", 
        "🔬 BET Analysis", 
        "📉 XRD Analysis",
        "🧬 3D XRD Visualization"  # NEW TAB
    ]
    
    # Add crystal structure tab if enabled
    if show_crystal_tab:
        all_tabs.append("🏛️ Crystal Structure")
    
    # Add remaining tabs
    all_tabs.extend([
        "📊 Morphology", 
        "🔍 Validation",
        "📚 Methods",
        "📤 Export"
    ])
    
    # Create tabs
    tabs = st.tabs(all_tabs)
    
    # Import plotter
    from scientific_plots import PublicationPlotter
    plotter = PublicationPlotter(
        color_scheme=scientific_params['export']['color_scheme'],
        font_size=scientific_params['export']['font_size']
    )
    
    # ============================================================================
    # SIMPLIFIED TAB HANDLING - No unpacking errors
    # ============================================================================
    
    # Create a mapping of tab indices to functions
    tab_index = 0
    
    # Tab 1: Overview
    with tabs[tab_index]:
        display_overview(results, plotter)
    tab_index += 1
    
    # Tab 2: BET Analysis
    with tabs[tab_index]:
        display_bet_analysis(results, plotter)
    tab_index += 1
    
    # Tab 3: XRD Analysis
    with tabs[tab_index]:
        display_xrd_analysis(results, plotter)
    tab_index += 1
    
    # Tab 4: 3D XRD Visualization (NEW)
    with tabs[tab_index]:
        display_3d_xrd_visualization(results, scientific_params)
    tab_index += 1
    
    # Tab 5: Crystal Structure (optional)
    if show_crystal_tab:
        with tabs[tab_index]:
            # Import and display crystal structure
            try:
                from crystal_structure_3d import CrystalStructure3D
                crystal_3d = CrystalStructure3D()
                
                # Parse lattice parameters
                lattice_params = {}
                import re
                lattice_str = scientific_params['crystal']['lattice_params']
                for match in re.finditer(r'([abc])\s*=\s*([\d\.]+)', lattice_str):
                    lattice_params[match.group(1)] = float(match.group(2))
                
                # Generate and display structure
                structure = crystal_3d.generate_structure(
                    crystal_system=scientific_params['crystal']['system'],
                    lattice_params=lattice_params,
                    space_group=scientific_params['crystal']['space_group'],
                    composition=scientific_params['crystal'].get('composition', 'SiO2')
                )
                
                # Create visualization
                fig = crystal_3d.create_3d_plot(structure)
                st.pyplot(fig)
                
                # Add information
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Crystal System", scientific_params['crystal']['system'])
                with col2:
                    st.metric("Space Group", scientific_params['crystal']['space_group'] or "Not specified")
                with col3:
                    if 'density' in structure:
                        st.metric("Density", f"{structure['density']:.2f} g/cm³")
                
            except Exception as e:
                st.error(f"Could not generate 3D structure: {str(e)}")
        tab_index += 1
    
    # Tab 6: Morphology
    with tabs[tab_index]:
        # Only display morphology if we have BET results
        if results.get('bet_results'):
            display_morphology(results)
        else:
            st.warning("BET analysis data is required to visualize material morphology")
    tab_index += 1
    
    # Tab 7: Validation
    with tabs[tab_index]:
        display_validation(results)
    tab_index += 1
    
    # Tab 8: Methods
    with tabs[tab_index]:
        display_methods(results, scientific_params)
    tab_index += 1
    
    # Tab 9: Export
    with tabs[tab_index]:
        display_export(results, scientific_params)
# ============================================================================
# DISPLAY FUNCTIONS (To be implemented in detail)
# ============================================================================
@memory_safe_plot
def display_overview(results, plotter):
    """Display overview dashboard"""
    st.subheader("Comprehensive Analysis Dashboard")
    
    # Create summary figure
    fig = plotter.create_summary_figure(results)
    st.pyplot(fig)
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if results.get('bet_results'):
            bet = results['bet_results']
            st.metric("**Surface Area (Sᴮᴱᵀ)**", 
                     f"{bet['surface_area']:.1f} ± {bet['surface_area_error']:.1f} m²/g",
                     help="BET surface area with 95% confidence interval")
    
    with col2:
        if results.get('xrd_results'):
            xrd = results['xrd_results']
            st.metric("**Crystallinity Index**",
                     f"{xrd['crystallinity_index']:.2f}",
                     help="Crystalline to amorphous ratio")
    
    with col3:
        if results.get('bet_results'):
            bet = results['bet_results']
            st.metric("**Total Pore Volume**",
                     f"{bet['total_pore_volume']:.3f} cm³/g",
                     help="Total pore volume at P/P₀ ≈ 0.99")
    
    with col4:
        # Show integration result if available
        if results.get('integration'):
            integration = results['integration']
            if 'material_classification' in integration:
                classification = integration['material_classification']['primary']
                st.metric("**Material Type**", classification)
        elif results.get('fusion_results'):
            fusion = results['fusion_results']
            st.metric("**Material Type**", 
                     fusion.get('composite_classification', 'Unknown'))
    
    # NEW: Show scientific integration validation
    if results.get('integration'):
        integration = results['integration']
        
        with st.expander("🔬 Scientific Integration Validation", expanded=False):
            if 'validation_metrics' in integration:
                validation = integration['validation_metrics']
                
                col_val1, col_val2 = st.columns(2)
                
                with col_val1:
                    if 'internal_consistency' in validation:
                        consistency = validation['internal_consistency']
                        if consistency > 0.8:
                            st.success(f"✅ High consistency: {consistency:.2f}")
                        elif consistency > 0.5:
                            st.warning(f"⚠️ Moderate consistency: {consistency:.2f}")
                        else:
                            st.error(f"❌ Low consistency: {consistency:.2f}")
                
                with col_val2:
                    if 'confidence_intervals' in validation:
                        ci = validation['confidence_intervals']
                        st.write("**Confidence Intervals:**")
                        for key, value in ci.items():
                            st.write(f"• {key}: {value}")
@memory_safe_plot
def display_bet_analysis(results, plotter):
    """Display detailed BET analysis"""
    st.subheader("IUPAC-Compliant BET Analysis")
    
    if results.get('bet_results') and results.get('bet_raw'):
        bet_res = results['bet_results']
        bet_raw = results['bet_raw']
        
        # Multi-panel BET figure
        fig = plotter.create_bet_figure(bet_raw, bet_res)
        st.pyplot(fig)
        
        # Detailed results in expanders
        with st.expander("📋 BET Regression Details", expanded=False):
            reg = bet_res['bet_regression']
            st.write(f"**Linear Range:** {reg['p_min']:.3f} - {reg['p_max']:.3f} P/P₀")
            st.write(f"**Equation:** p/[n(1-p)] = {reg['slope']:.4f}p + {reg['intercept']:.4f}")
            st.write(f"**R²:** {reg['r_squared']:.6f}")
            st.write(f"**Standard Error:** {reg['std_error']:.6f}")
            st.write(f"**Points used:** {reg['n_points']}")
        
        with st.expander("📊 Porosity Analysis", expanded=False):
            st.write(f"**Total Pore Volume:** {bet_res['total_pore_volume']:.4f} cm³/g")
            st.write(f"**Micropore Volume (t-plot):** {bet_res['micropore_volume']:.4f} cm³/g")
            st.write(f"**External Surface Area:** {bet_res['external_surface']:.1f} m²/g")
            st.write(f"**Mean Pore Diameter:** {bet_res['mean_pore_diameter']:.2f} nm")
            
            if 'psd_analysis' in bet_res:
                psd = bet_res['psd_analysis']
                st.write("**Pore Size Distribution (BJH):**")
                st.write(f"  - Micropores (<2 nm): {psd['micropore_fraction']:.1%}")
                st.write(f"  - Mesopores (2-50 nm): {psd['mesopore_fraction']:.1%}")
                st.write(f"  - Macropores (>50 nm): {psd['macropore_fraction']:.1%}")
        
        with st.expander("📈 Hysteresis Analysis", expanded=False):
            hyst = bet_res['hysteresis_analysis']
            st.write(f"**Type:** {hyst['type']} ({hyst['iupac_class']})")
            st.write(f"**Description:** {hyst['description']}")
            st.write(f"**Loop Area:** {hyst['loop_area']:.2f}")
            st.write(f"**Closure Pressure:** {hyst['closure_pressure']:.3f} P/P₀")
@memory_safe_plot
def display_xrd_analysis(results, plotter):
    """Display detailed XRD analysis (STABLE VERSION)"""

    st.subheader("Advanced XRD Analysis")

    # ============================================================
    # SAFE EXTRACTION
    # ============================================================
    xrd_res = results.get("xrd_results", {})
    xrd_raw = results.get("xrd_raw", {})
    # ==========================================================
    # (2) WHY NO PHASE DETECTED — DIAGNOSTIC PANEL
    # ==========================================================
    if not xrd_res.get("phases"):
        with st.expander("❓ Why were no crystalline phases identified?", expanded=True):
    
            st.markdown("""
    **This is NOT a software error.**  
    Phase identification is based strictly on **experimental peak matching**
    against **CIF-validated structures** (COD + OPTIMADE).
    
    Possible scientific reasons:
    """)
    
            reasons = []
    
            # 1. Peak broadening (nano / amorphous)
            if xrd_res.get("crystallite_size", {}).get("scherrer", 0) < 10:
                reasons.append("• Peaks are strongly broadened → nanocrystalline or partially amorphous material")
    
            # 2. Too few peaks
            if len(xrd_res.get("peaks", [])) < 5:
                reasons.append("• Too few resolved diffraction peaks for reliable indexing")
    
            # 3. Element constraints
            elements = st.session_state.get("xrd_elements", [])
            if not elements:
                reasons.append("• No elements were selected in the sidebar → database search disabled")
    
            # 4. Database limitation (truthful)
            reasons.append("• No CIF structure in free databases matches the experimental peak positions within tolerance")
    
            for r in reasons:
                st.markdown(r)
    
            st.markdown("""
    ### What you can try
    - Select **all possible elements** present (dopants included)
    - Use **raw, unsmoothed XRD data**
    - Increase crystallinity (annealing) if experimentally possible
    - Combine with **Raman / FTIR** for phase confirmation
    """)
    # ============================================================
    # PHASE IDENTIFICATION TABLE
    # ============================================================
    phases = xrd_res.get("phases", [])

    if phases:
        st.subheader("🔬 Identified Phases (CIF-Validated)")

        phase_df = pd.DataFrame([
            {
                "Phase": p["phase"],
                "Crystal system": p["crystal_system"],
                "Space group": p["space_group"],
                "Confidence level": p.get("confidence_level", ""),
                "Score": round(p["score"], 3),
            }
            for p in phases
        ])

        st.dataframe(phase_df, use_container_width=True)
    else:
        st.info("No crystalline phase identified with sufficient confidence.")

    # ============================================================
    # PHASE FRACTIONS
    # ============================================================
    fractions = xrd_res.get("phase_fractions", [])

    if fractions:
        st.subheader("📊 Phase Fractions (Intensity-Weighted)")

        frac_df = pd.DataFrame(fractions)
        st.dataframe(frac_df, use_container_width=True)

        fig, ax = plt.subplots()
        ax.pie(
            frac_df["fraction"],
            labels=[
                f'{row["phase"]} ({row["fraction"]:.1f}%)'
                for _, row in frac_df.iterrows()
            ],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.axis("equal")
        st.pyplot(fig)

    # ============================================================
    # XRD PATTERN + PEAK INDEXING
    # ============================================================
    if xrd_raw and xrd_res:
        fig = plotter.create_xrd_figure(xrd_raw, xrd_res)
        st.pyplot(fig)

    # ============================================================
    # PEAK TABLE (ALL PEAKS, NOT ONLY MAJOR)
    # ============================================================
    peaks = xrd_res.get("peaks", [])

    if peaks:
        st.subheader("📌 Peak Analysis with HKL & Phase Assignment")

        n_total = len(peaks)
        n_show = st.slider(
            "Number of peaks to display",
            1,
            n_total,
            min(20, n_total),
        )

        peaks_sorted = sorted(peaks, key=lambda x: x["intensity"], reverse=True)

        table = []
        for i, p in enumerate(peaks_sorted[:n_show]):
            table.append({
                "Rank": i + 1,
                "2θ (°)": round(p["position"], 3),
                "d (Å)": round(p.get("d_spacing", 0), 4),
                "Intensity": round(p["intensity"], 1),
                "FWHM (°)": round(p["fwhm_deg"], 4),
                "Size (nm)": round(p.get("crystallite_size", 0), 2),
                "Phase": p.get("phase", ""),
                "HKL": p.get("hkl", ""),
                "Confidence": round(p.get("phase_confidence", 0), 3),
            })

        st.dataframe(pd.DataFrame(table), use_container_width=True)

    # ============================================================
    # CRYSTALLITE SIZE ANALYSIS
    # ============================================================
    with st.expander("🔬 Crystallite Size Analysis", expanded=False):
        size = xrd_res.get("crystallite_size", {})

        st.write(f"**Scherrer:** {size.get('scherrer', 0):.2f} nm")

        wh = size.get("williamson_hall", 0)
        if wh > 0:
            st.write(f"**Williamson–Hall:** {wh:.2f} nm")
        else:
            st.write("**Williamson–Hall:** Not available")

        st.write(f"**Distribution:** {size.get('distribution', 'N/A')}")

        microstrain = xrd_res.get("microstrain", 0)
        if microstrain > 0:
            st.write(f"**Microstrain:** {microstrain:.3e}")

        rho = xrd_res.get("dislocation_density", 0)
        if rho > 0:
            st.write(f"**Dislocation density:** {rho:.3e} m⁻²")

    # ============================================================
    # CSV EXPORT (ALL PEAKS)
    # ============================================================
    if peaks:
        all_peaks_df = pd.DataFrame([
            {
                "2theta_deg": p["position"],
                "d_spacing_A": p.get("d_spacing", 0),
                "intensity": p["intensity"],
                "fwhm_deg": p["fwhm_deg"],
                "crystallite_size_nm": p.get("crystallite_size", 0),
                "phase": p.get("phase", ""),
                "hkl": p.get("hkl", ""),
                "confidence": p.get("phase_confidence", 0),
            }
            for p in peaks
        ])

        st.download_button(
            "📥 Download Full XRD Peak Data (CSV)",
            all_peaks_df.to_csv(index=False),
            file_name="xrd_peak_analysis.csv",
            mime="text/csv",
        )
@memory_safe_plot            
def display_3d_xrd_visualization(results, scientific_params):
    """Display 3D XRD visualization with hkl indices"""
    st.subheader("3D XRD Pattern Visualization")
    
    if not results.get('xrd_results'):
        st.warning("XRD data is required for 3D visualization")
        return
    
    xrd_res = results['xrd_results']
    if not xrd_res.get('peaks'):
        st.warning("No peaks detected in XRD data")
        return
    
    # Create a simple 3D XRD visualization
    try:
        import plotly.graph_objects as go
        import numpy as np
        
        # Get peaks data
        peaks = xrd_res['peaks']
        positions = [p['position'] for p in peaks]
        intensities = [p['intensity'] for p in peaks]
        
        # Normalize intensities for visualization
        max_intensity = max(intensities)
        norm_intensities = [i/max_intensity for i in intensities]
        
        # Get hkl indices
        hkl_labels = []
        for peak in peaks:
            hkl = peak.get('hkl', '')
            if not hkl and 'hkl_detail' in peak:
                hkl_detail = peak['hkl_detail']
                if isinstance(hkl_detail, dict):
                    h = hkl_detail.get('h', '?')
                    k = hkl_detail.get('k', '?')
                    l = hkl_detail.get('l', '?')
                    hkl = f"({h}{k}{l})"
            hkl_labels.append(hkl)
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add peak lines (vertical lines)
        for pos, intensity, hkl in zip(positions, norm_intensities, hkl_labels):
            # Line from base to peak
            fig.add_trace(go.Scatter3d(
                x=[pos, pos],
                y=[0, intensity],
                z=[0, 0],
                mode='lines',
                line=dict(color='blue', width=3),
                showlegend=False
            ))
            
            # Peak marker
            fig.add_trace(go.Scatter3d(
                x=[pos],
                y=[intensity],
                z=[0],
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',
                    opacity=0.8
                ),
                text=f"2θ: {pos:.2f}°<br>Intensity: {intensity:.2f}<br>hkl: {hkl}",
                hoverinfo='text',
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title='2θ (degrees)',
                yaxis_title='Normalized Intensity',
                zaxis_title='',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.5)
                )
            ),
            title='3D XRD Pattern Visualization',
            width=800,
            height=600
        )
        
        # Display plot
 
        st.plotly_chart(fig, use_container_width=True)

        
        # Add crystal structure if available
        crystal_system = scientific_params['crystal']['system']
        lattice_params = scientific_params['crystal']['lattice_params']
        
        if crystal_system != 'Unknown' and lattice_params:
            st.subheader("Crystal Structure Information")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Crystal System:** {crystal_system}")
                st.info(f"**Lattice Parameters:** {lattice_params}")
            
            with col2:
                if 'indexing' in xrd_res:
                    indexing = xrd_res['indexing']
                    if 'figures_of_merit' in indexing:
                        fom = indexing['figures_of_merit']
                        if 'M20' in fom:
                            st.info(f"**M₂₀ Figure of Merit:** {fom['M20']:.1f}")
            
            # Generate simple unit cell visualization
            if crystal_system == 'Cubic':
                st.info("**Unit Cell:** Simple cubic structure shown")
                # You can add more detailed visualization here
            
    except Exception as e:
        st.error(f"Could not create 3D visualization: {str(e)}")
        
        # Fallback: Create a simple 2D plot with hkl annotations
        st.subheader("2D XRD Pattern with HKL Indices")
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot XRD pattern
        if results.get('xrd_raw'):
            ax.plot(results['xrd_raw']['two_theta'], 
                   results['xrd_raw']['intensity'], 
                   'k-', linewidth=1)
        
        # Mark peaks with hkl
        for peak in peaks:
            pos = peak['position']
            intensity = peak['intensity']
            
            # Get hkl
            hkl = peak.get('hkl', '')
            if not hkl and 'hkl_detail' in peak:
                hkl_detail = peak['hkl_detail']
                if isinstance(hkl_detail, dict):
                    h = hkl_detail.get('h', '?')
                    k = hkl_detail.get('k', '?')
                    l = hkl_detail.get('l', '?')
                    hkl = f"({h}{k}{l})"
            
            ax.plot([pos], [intensity], 'ro', markersize=5)
            ax.text(pos, intensity * 1.05, hkl, 
                   ha='center', va='bottom', fontsize=8,
                   rotation=45)
        
        ax.set_xlabel('2θ (degrees)')
        ax.set_ylabel('Intensity')
        ax.set_title('XRD Pattern with HKL Indices')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)            
@memory_safe_plot            
def display_morphology(results):
    """Display morphology visualization and interpretation"""

    st.subheader("Realistic SEM/TEM Representation")

    st.info(
        "ℹ️ SEM/TEM-style morphology figures are generated offline "
        "to ensure scientific accuracy and platform stability."
    )
    
    # --- SEM/TEM REQUEST FORM ---
    st.markdown("### 📩 Request SEM/TEM Morphology Figures")
    
    email = st.text_input(
        "Your email address (for delivery)",
        placeholder="name@institution.edu"
    )
    
    request_sem_tem = st.button("📤 Submit SEM/TEM Generation Request")
    
    if request_sem_tem:
        if not email:
            st.error("Please provide a valid email address.")
        else:
            # Store request locally (SAFE)
            import json, time, os
    
            def make_json_safe(obj):
                if isinstance(obj, dict):
                    return {k: make_json_safe(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_safe(v) for v in obj]
                elif hasattr(obj, "item"):  # numpy scalar
                    return obj.item()
                elif hasattr(obj, "tolist"):  # numpy array
                    return obj.tolist()
                else:
                    return obj
            
            
            request_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "email": email,
                "bet_results": make_json_safe(results["bet_results"]),
                "xrd_results": make_json_safe(results.get("xrd_results")),
                "fusion_results": make_json_safe(results.get("fusion_results"))
            }
            # --- SAVE REQUEST TO DISK ---
            # === SAVE REQUEST TO DISK ===
            REQUEST_DIR = "sem_tem_requests"
            os.makedirs(REQUEST_DIR, exist_ok=True)
    
            filename = f"sem_tem_request_{int(time.time())}.json"
            file_path = os.path.join(REQUEST_DIR, filename)
    
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(request_data, f, indent=2)
    
            # === USER FEEDBACK (CRITICAL) ===
            st.success("✅ SEM/TEM request submitted successfully.")
    
            # === DOWNLOAD FOR USER (CRITICAL FOR STREAMLIT CLOUD) ===
            with open(file_path, "rb") as f:
                st.download_button(
                    label="⬇ Download your SEM/TEM request file",
                    data=f,
                    file_name=filename,
                    mime="application/json"
                )




# ADD THIS NEW FUNCTION outside display_morphology
def generate_morphology_report(morphology, bet, xrd):
    """Generate morphology report text"""
    report = []
    
    report.append("=" * 70)
    report.append("MATERIAL MORPHOLOGY ANALYSIS REPORT")
    report.append("=" * 70)
    
    # Material classification
    classification = morphology['classification']
    report.append(f"\nMATERIAL CLASSIFICATION")
    report.append("-" * 40)
    report.append(f"Primary Type: {classification['primary']}")
    
    if 'examples' in classification:
        report.append(f"Typical Examples: {', '.join(classification['examples'])}")
    
    if 'characteristics' in classification:
        report.append("Characteristics:")
        for char in classification['characteristics']:
            report.append(f"  • {char}")
    
    # Structure properties
    properties = morphology['structure_properties']
    report.append(f"\nSTRUCTURE PROPERTIES")
    report.append("-" * 40)
    
    for key, value in properties.items():
        if key == 'porosity_percentage':
            report.append(f"Porosity: {value:.1f}%")
        elif key == 'surface_to_volume_ratio':
            report.append(f"Surface-to-Volume Ratio: {value:.0f} m²/cm³")
        elif key == 'accessibility_factor':
            report.append(f"Accessibility Factor: {value:.2f}/1.0")
        elif key == 'estimated_wall_thickness':
            report.append(f"Estimated Wall Thickness: {value:.1f} nm")
        elif key == 'crystallinity_index':
            report.append(f"Crystallinity Index: {value:.2f}")
        elif key == 'crystallite_size_nm':
            report.append(f"Crystallite Size: {value:.1f} nm")
    
    # Experimental parameters
    report.append(f"\nEXPERIMENTAL PARAMETERS")
    report.append("-" * 40)
    report.append(f"BET Surface Area: {bet.get('surface_area', 0):.1f} m²/g")
    report.append(f"Total Pore Volume: {bet.get('total_pore_volume', 0):.3f} cm³/g")
    report.append(f"Mean Pore Diameter: {bet.get('mean_pore_diameter', 0):.1f} nm")
    
    if xrd:
        report.append(f"Crystallinity Index: {xrd.get('crystallinity_index', 0):.2f}")
        report.append(f"Crystallite Size: {xrd.get('crystallite_size', {}).get('scherrer', 0):.1f} nm")
    
    # Interpretation
    interpretation = morphology['interpretation']
    report.append(f"\nMORPHOLOGICAL INTERPRETATION")
    report.append("-" * 40)
    
    for key, value in interpretation.items():
        if key.endswith('_description'):
            continue
        if key in ['porosity_level', 'surface_area_level', 'pore_size_type', 'crystallinity']:
            report.append(f"{key.replace('_', ' ').title()}: {value}")
    
    return "\n".join(report)

def display_methods(results, scientific_params):
    """Display detailed methods section with complete references"""
    st.subheader("📚 Scientific Methods & Calculations")
    
    # Table of Contents
    toc = st.expander("📖 Table of Contents", expanded=True)
    with toc:
        st.markdown("""
        **A. BET Surface Area Analysis**  
        **B. Porosity Analysis Methods**  
        **C. XRD Analysis Procedures**  
        **D. Crystallite Size Determination**  
        **E. Morphology Integration**  
        **F. Statistical Analysis**  
        **G. Error Propagation**  
        **H. References**
        """)
    
    # BET Methods with full references
    with st.expander("A. BET Surface Area Analysis", expanded=True):
        st.markdown("""
        ### BET Theory Fundamentals
        
        The Brunauer-Emmett-Teller (BET) theory extends Langmuir theory to multilayer adsorption:
        
        **BET Equation:**
        ```
        p/(n(1-p)) = 1/(n_m·C) + (C-1)/(n_m·C)·p
        ```
        
        where:
        - **p** = relative pressure (P/P₀)
        - **n** = amount adsorbed (mmol/g)
        - **nₘ** = monolayer capacity (mmol/g)
        - **C** = BET constant related to adsorption enthalpy
        
        ### IUPAC-Compliant Implementation
        
        **1. Rouquerol Criteria for Linear Range Selection:**
        ```
        Conditions: 
        1. Q(1-p) must increase with p
        2. C > 0 for physical meaningfulness
        3. R² > 0.999 for validity
        ```
        
        **2. Surface Area Calculation:**
        ```
        S_BET = n_m · N_A · σ · 10⁻²⁰
        ```
        - **N_A** = 6.02214076×10²³ mol⁻¹ (CODATA 2018)
        - **σ** = 0.162 nm² (N₂ cross-sectional area at 77K)
        
        **3. Error Propagation:**
        ```
        ΔS_BET = S_BET · √((Δn_m/n_m)² + (Δσ/σ)²)
        ```
        
        **References:**
        1. Brunauer, S., Emmett, P. H., & Teller, E. (1938). J. Am. Chem. Soc., 60, 309-319.
        2. Rouquerol, J., Llewellyn, P., & Rouquerol, F. (2007). Stud. Surf. Sci. Catal., 160, 49-56.
        3. Thommes, M., et al. (2015). Pure Appl. Chem., 87, 1051-1069.
        
        ### Applied Parameters
        """)
        
        if results.get('bet_results'):
            bet = results['bet_results']
            st.write(f"- **Cross-section:** {bet['cross_section']:.3f} nm²")
            st.write(f"- **Temperature:** {bet['temperature']:.1f} K")
            st.write(f"- **BET Range:** {bet['bet_regression']['p_min']:.3f}-{bet['bet_regression']['p_max']:.3f} P/P₀")
            st.write(f"- **Number of points:** {bet['bet_regression']['n_points']}")
    
    # XRD Methods with full crystallography
    with st.expander("C. XRD Analysis Procedures", expanded=False):
        st.markdown("""
        ### X-ray Diffraction Fundamentals
        
        **1. Bragg's Law:**
        ```
        nλ = 2d sinθ
        ```
        
        **2. Scherrer Equation (Crystallite Size):**
        ```
        D = K·λ/(β·cosθ)
        ```
        - **K** = 0.9 (Scherrer constant for spherical crystals)
        - **β** = FWHM in radians (corrected for instrumental broadening)
        
        **3. Williamson-Hall Analysis (Size-Strain Separation):**
        ```
        β·cosθ = K·λ/D + 4ε·sinθ
        ```
        - **ε** = microstrain
        
        **4. Crystallinity Index (Ruland Method):**
        ```
        CI = A_crystalline/(A_crystalline + A_amorphous)
        ```
        
        **5. hkl Indexing (Pawley Method):**
        - Systematic absences applied based on space group
        - Figures of merit: M₂₀ > 10 indicates reliable indexing
        
        **References:**
        1. Klug, H. P., & Alexander, L. E. (1974). X-ray Diffraction Procedures.
        2. Williamson, G. K., & Hall, W. H. (1953). Acta Metall., 1, 22-31.
        3. Ruland, W. (1961). Acta Cryst., 14, 1180.
        4. Pawley, G. S. (1981). J. Appl. Cryst., 14, 357-361.
        """)
def display_validation(results):
    """Display scientific validation results"""
    st.subheader("🔬 Scientific Validation")
    
    if 'validation' not in results:
        st.info("No validation data available. Run analysis to see validation results.")
        return
    
    validation = results['validation']
    
    # Show detailed validation results
    with st.expander("📋 Detailed Validation Results", expanded=True):
        # BET Validation
        if validation.get('analysis_validation', {}).get('bet_checks'):
            st.subheader("BET Validation Checks")
            for check in validation['analysis_validation']['bet_checks']:
                if check['status'] == '✅':
                    st.success(f"{check['check']}: {check['value']}")
                elif check['status'] == '⚠️':
                    st.warning(f"{check['check']}: {check['value']}")
                else:
                    st.error(f"{check['check']}: {check['value']}")
        
        # XRD Validation
        if validation.get('analysis_validation', {}).get('xrd_checks'):
            st.subheader("XRD Validation Checks")
            for check in validation['analysis_validation']['xrd_checks']:
                if check['status'] == '✅':
                    st.success(f"{check['check']}: {check['value']}")
                elif check['status'] == '⚠️':
                    st.warning(f"{check['check']}: {check['value']}")
                else:
                    st.error(f"{check['check']}: {check['value']}")
    
    # Show summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bet_passed = sum(1 for check in validation.get('analysis_validation', {}).get('bet_checks', []) 
                        if check['status'] == '✅')
        bet_total = len(validation.get('analysis_validation', {}).get('bet_checks', []))
        st.metric("BET Checks", f"{bet_passed}/{bet_total}")
    
    with col2:
        xrd_passed = sum(1 for check in validation.get('analysis_validation', {}).get('xrd_checks', []) 
                        if check['status'] == '✅')
        xrd_total = len(validation.get('analysis_validation', {}).get('xrd_checks', []))
        st.metric("XRD Checks", f"{xrd_passed}/{xrd_total}")
    
    with col3:
        if validation.get('analysis_validation', {}).get('all_passed', True):
            st.success("✅ All Checks Passed")
        else:
            st.warning("⚠️ Some Checks Failed")
    
    # Show recommendations for fixing warnings
    if not validation.get('analysis_validation', {}).get('all_passed', True):
        with st.expander("🔧 How to Fix Warnings", expanded=True):
            st.markdown("""
            **Common issues and solutions:**
            
            1. **BET Linearity (R² < 0.999):**
               - Check if material is microporous (BET may not be applicable)
               - Verify pressure range is 0.05-0.35 P/P₀
               - Consider using t-plot or DR methods instead
            
            2. **Negative C constant:**
               - Material may not follow BET theory
               - Common for highly microporous materials
               - Consider alternative surface area methods
            
            3. **Few XRD peaks:**
               - Material may be amorphous or nanocrystalline
               - Check XRD measurement parameters
               - Consider longer scan times or higher resolution
            
            4. **Inconsistent BET/XRD results:**
               - Different sample preparations may cause discrepancies
               - Ensure samples are from same batch
               - Consider surface roughness effects
            """)
def display_crystal_structure(results, scientific_params):
    """Display 3D crystal structure visualization"""
    st.subheader("🏛️ 3D Crystal Structure")
    
    # Get crystal parameters
    crystal_system = scientific_params['crystal']['system']
    space_group = scientific_params['crystal']['space_group']
    lattice_str = scientific_params['crystal']['lattice_params']
    composition = scientific_params['crystal'].get('composition', 'SiO2')
    
    if crystal_system == 'Unknown' or not lattice_str:
        st.info("""
        **Crystal structure visualization requires:**
        1. Crystal system (select from sidebar)
        2. Lattice parameters (e.g., a=4.05, c=6.7)
        
        Please provide these parameters in the sidebar under "Crystal Structure".
        """)
        return
    
    # Parse lattice parameters
    lattice_params = {}
    import re
    for match in re.finditer(r'([abc])\s*=\s*([\d\.]+)', lattice_str):
        lattice_params[match.group(1)] = float(match.group(2))
    
    if not lattice_params:
        st.warning("Could not parse lattice parameters. Please use format: a=4.05, b=4.05, c=6.7")
        return
    
    try:
        # Initialize 3D crystal structure generator
        try:
            from crystal_structure_3d import CrystalStructure3D
            structure_3d = CrystalStructure3D()
        except ImportError:
            st.error("Could not import CrystalStructure3D. Make sure crystal_structure_3d.py is in the same directory.")
            return
        
        # Generate crystal structure
        structure = structure_3d.generate_structure(
            crystal_system=crystal_system,
            lattice_params=lattice_params,
            space_group=space_group,
            composition=composition
        )
        
        # Add space group to structure
        structure['space_group'] = space_group
        
        # Display static 3D plot
        st.subheader("Static 3D Visualization")
        fig_static = structure_3d.create_3d_plot(structure, figsize=(10, 8))
        st.pyplot(fig_static)
        
        # Display crystal information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Crystal System", crystal_system)
        with col2:
            st.metric("Space Group", space_group if space_group else "Not specified")
        with col3:
            if 'density' in structure:
                st.metric("Density", f"{structure['density']:.2f} g/cm³")
        
        # Download options
        st.subheader("📥 Download Structure")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Save static figure
            buf = io.BytesIO()
            fig_static.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            st.download_button(
                label="📷 Download 3D Structure (PNG)",
                data=buf,
                file_name="crystal_structure_3d.png",
                mime="image/png",
                use_container_width=True
            )
        
        with col2:
            # Save structure data
            structure_data = {
                'crystal_system': crystal_system,
                'space_group': space_group,
                'lattice_parameters': lattice_params,
                'composition': composition,
                'generation_method': 'CrystalStructure3D',
                'references': [
                    'International Tables for Crystallography (2006)',
                    'Momma, K., & Izumi, F. (2011). VESTA 3'
                ]
            }
            
            json_data = json.dumps(structure_data, indent=2)
            st.download_button(
                label="📄 Download Structure Data (JSON)",
                data=json_data,
                file_name="crystal_structure_data.json",
                mime="application/json",
                use_container_width=True
            )
        
    except Exception as e:
        st.error(f"Error generating 3D crystal structure: {str(e)}")                
@memory_safe_plot
def display_export(results, scientific_params):
    """Export functionality"""
    st.subheader("📤 Export Scientific Data")
    
    # Export formats
    col1, col2 = st.columns(2)
    
    with col1:
        # Export data - be careful with None values
        def build_export_data(results, scientific_params):
            return {
                "metadata": {
                    "app": "BET-XRD Analyzer",
                    "version": "1.0",
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
                    "peaks": results.get("peaks", []),
                    "top_peaks": results.get("top_peaks", []),
                },
                "parameters": scientific_params,
            }

        import numpy as np
        import json
        # Use a custom JSON encoder to handle numpy arrays and other non-serializable objects
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
      
        
        # 1️⃣ Sanitize export data (remove NumPy objects)
        # 0️⃣ Build CLEAN export data (THIS WAS MISSING)
        export_data = build_export_data(results, scientific_params)
        
        # 1️⃣ Sanitize export data (remove NumPy objects)
        export_data = json.loads(json.dumps(export_data, cls=NumpyEncoder))

        
        # 2️⃣ Create JSON string (THIS WAS MISSING)
        json_str = json.dumps(export_data, indent=2)
        
        # 3️⃣ Download button
        st.download_button(
            label="📄 Download Complete Analysis (JSON)",
            data=json_str,
            file_name="scientific_analysis.json",
            mime="application/json",
            width="stretch"   # updated API
        )
    
    with col2:
        # Export report
        report_text = generate_scientific_report(results)
        
        st.download_button(
            label="📋 Download Scientific Report (TXT)",
            data=report_text,
            file_name="scientific_report.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    # Figure export
    st.subheader("Figure Export")
    
    from scientific_plots import PublicationPlotter
    plotter = PublicationPlotter(
        color_scheme=scientific_params['export']['color_scheme'],
        font_size=scientific_params['export']['font_size']
    )
    
    # BET Figure (if BET data exists)
    if results.get('bet_results') and results.get('bet_raw'):
        try:
            fig = plotter.create_bet_figure(results['bet_raw'], results['bet_results'])
            
            # Save to buffer
            import io
            buf = io.BytesIO()
            
            format_map = {
                'PNG (600 DPI)': ('png', 600),
                'PDF (Vector)': ('pdf', None),
                'SVG (Vector)': ('svg', None),
                'TIFF (1200 DPI)': ('tiff', 1200)
            }
            
            export_format = scientific_params['export']['figure_format']
            fmt, dpi = format_map.get(export_format, ('png', 600))
            
            fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight')
            buf.seek(0)
            
            st.download_button(
                label=f"🖼️ Download BET Figure ({export_format})",
                data=buf,
                file_name=f"bet_analysis.{fmt}",
                mime=f"image/{fmt}" if fmt != 'pdf' else 'application/pdf',
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Could not generate BET figure: {str(e)}")
    
    # XRD Figure (if XRD data exists)
    if results.get('xrd_results') and results.get('xrd_raw'):
        try:
            fig = plotter.create_xrd_figure(results['xrd_raw'], results['xrd_results'])
            
            # Save to buffer
            import io
            buf = io.BytesIO()
            
            format_map = {
                'PNG (600 DPI)': ('png', 600),
                'PDF (Vector)': ('pdf', None),
                'SVG (Vector)': ('svg', None),
                'TIFF (1200 DPI)': ('tiff', 1200)
            }
            
            export_format = scientific_params['export']['figure_format']
            fmt, dpi = format_map.get(export_format, ('png', 600))
            
            fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight')
            buf.seek(0)
            
            st.download_button(
                label=f"🖼️ Download XRD Figure ({export_format})",
                data=buf,
                file_name=f"xrd_analysis.{fmt}",
                mime=f"image/{fmt}" if fmt != 'pdf' else 'application/pdf',
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Could not generate XRD figure: {str(e)}")

def generate_scientific_report(results):
    """Generate comprehensive scientific report"""
    report = []
    
    report.append("=" * 70)
    report.append("SCIENTIFIC ANALYSIS REPORT - BET-XRD MORPHOLOGY ANALYSIS")
    report.append("=" * 70)
    report.append(f"Generated: {results.get('timestamp', 'N/A')}")
    report.append(f"Software Version: 3.0 (IUPAC Compliant)")
    report.append("")
    
    # BET Results
    if results.get('bet_results'):
        bet = results['bet_results']
        report.append("BET SURFACE AREA ANALYSIS")
        report.append("-" * 40)
        report.append(f"Surface Area (Sᴮᴱᵀ): {bet.get('surface_area', 0):.1f} ± {bet.get('surface_area_error', 0):.1f} m²/g")
        report.append(f"Total Pore Volume: {bet.get('total_pore_volume', 0):.4f} cm³/g")
        report.append(f"Micropore Volume (t-plot): {bet.get('micropore_volume', 0):.4f} cm³/g")
        report.append(f"Mean Pore Diameter: {bet.get('mean_pore_diameter', 0):.2f} nm")
        report.append(f"BET C Constant: {bet.get('c_constant', 0):.0f}")
        
        if bet.get('bet_regression'):
            report.append(f"BET R²: {bet['bet_regression'].get('r_squared', 0):.4f}")
        
        if bet.get('hysteresis_analysis'):
            report.append(f"Hysteresis Type: {bet['hysteresis_analysis'].get('type', 'N/A')}")
        report.append("")
    
    # XRD Results
    if results.get('xrd_results'):
        xrd = results['xrd_results']
        report.append("XRD ANALYSIS")
        report.append("-" * 40)
        report.append(f"Crystallinity Index: {xrd.get('crystallinity_index', 0):.2f}")
        
        if xrd.get('crystallite_size'):
            report.append(f"Crystallite Size (Scherrer): {xrd['crystallite_size'].get('scherrer', 0):.1f} nm")
        
        report.append(f"Microstrain: {xrd.get('microstrain', 0):.4f}")
        report.append(f"Ordered Mesopores: {'Yes' if xrd.get('ordered_mesopores') else 'No'}")
        
        if xrd.get('peaks'):
            report.append(f"Peaks Detected: {len(xrd['peaks'])}")
        report.append("")
    
    # Morphology/Fusion Results
    if results.get('fusion_results'):
        fusion = results['fusion_results']
        if fusion.get('valid', False):
            report.append("INTEGRATED MORPHOLOGY ANALYSIS")
            report.append("-" * 40)
            if 'composite_classification' in fusion:
                report.append(f"Classification: {fusion['composite_classification']}")
            if 'material_family' in fusion:
                report.append(f"Family: {fusion['material_family']}")
            if 'dominant_feature' in fusion:
                report.append(f"Dominant Feature: {fusion['dominant_feature']}")
            report.append("")
    
    # Methods
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

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()













































































