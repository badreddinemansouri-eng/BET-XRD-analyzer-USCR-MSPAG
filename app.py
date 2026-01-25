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


warnings.filterwarnings('ignore')
import json  # <-- ADD THIS LINE
# Import scientific engines
from bet_analyzer import IUPACBETAnalyzer, extract_asap2420_data
from xrd_analyzer import AdvancedXRDAnalyzer, extract_xrd_data
from morphology_fusion import MorphologyFusionEngine
from scientific_plots import PublicationPlotter
# At the top of app.py with other imports
from morphology_visualizer import IntegratedMorphologyAnalyzer

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
            )
        }
        
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
# SCIENTIFIC ANALYSIS EXECUTION
# ============================================================================
def execute_scientific_analysis(bet_file, xrd_file, params):
    """Execute complete scientific analysis pipeline"""
    
    analysis_results = {}
    
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
                
                analysis_results['xrd_results'] = xrd_analyzer.complete_analysis(
                    two_theta=analysis_results['xrd_raw']['two_theta'],
                    intensity=analysis_results['xrd_raw']['intensity'],
                    crystal_system=params['crystal']['system'],
                    space_group=params['crystal']['space_group'],
                    lattice_params=params['crystal']['lattice_params']
                )
                
                if analysis_results['xrd_results']['valid']:
                    st.success("✅ XRD analysis completed successfully")
                    
                    # Display key results
                    xrd_res = analysis_results['xrd_results']
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Crystallinity", f"{xrd_res['crystallinity_index']:.2f}")
                    with col2:
                        size = xrd_res['crystallite_size']['scherrer']
                        st.metric("Size", f"{size:.1f} nm" if size else "N/A")
                    with col3:
                        st.metric("Peaks", f"{len(xrd_res['peaks'])}")
                    with col4:
                        st.metric("Ordered", "Yes" if xrd_res['ordered_mesopores'] else "No")
                
                else:
                    st.warning(f"⚠️ XRD analysis completed with warnings: {analysis_results['xrd_results'].get('error', 'Unknown error')}")
                    
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
        # STEP 5: FINALIZATION
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
        
        progress_bar.progress(100)
        
        if analysis_results['analysis_valid']:
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
def display_scientific_results(results, params):
    """Display comprehensive scientific results"""
    
    st.header("📊 Scientific Results")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview", 
        "🔬 BET Analysis", 
        "📉 XRD Analysis", 
        "🧬 Morphology", 
        "📚 Methods",
        "📤 Export"
    ])
    
    # Import plotter here to avoid circular imports
    from scientific_plots import PublicationPlotter
    
    plotter = PublicationPlotter(
        color_scheme=params['export']['color_scheme'],
        font_size=params['export']['font_size']
    )
    
    with tab1:
        display_overview(results, plotter)
    
    with tab2:
        display_bet_analysis(results, plotter)
    
    with tab3:
        display_xrd_analysis(results, plotter)
    
    with tab4:
        # Only display morphology if we have BET results
        if results.get('bet_results'):
            display_morphology(results)
        else:
            st.warning("BET analysis data is required to visualize material morphology")
            st.info("""
            **Why morphology visualization requires BET data:**
            
            The morphology visualization generates material structure representations based on:
            1. **Surface area** - determines pore density and texture
            2. **Pore volume** - controls porosity level
            3. **Pore size distribution** - determines pore sizes in visualization
            4. **Crystallinity** (from XRD) - adds crystalline regions if available
            
            Please upload BET data to enable morphology visualization.
            """)
    
    with tab5:
        display_methods(results, params)
    
    with tab6:
        display_export(results, params)

# ============================================================================
# DISPLAY FUNCTIONS (To be implemented in detail)
# ============================================================================
def display_overview(results, plotter):
    """Display overview dashboard"""
    st.subheader("Comprehensive Analysis Dashboard")
    
    # Create summary figure
    fig = plotter.create_summary_figure(results)
    st.pyplot(fig)
    
    # Key metrics
    if results.get('bet_results'):
        bet = results['bet_results']
        st.metric("**Surface Area (Sᴮᴱᵀ)**", 
                 f"{bet['surface_area']:.1f} ± {bet['surface_area_error']:.1f} m²/g",
                 help="BET surface area with 95% confidence interval")
    
    if results.get('xrd_results'):
        xrd = results['xrd_results']
        st.metric("**Crystallinity Index**",
                 f"{xrd['crystallinity_index']:.2f}",
                 help="Crystalline to amorphous ratio")

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

def display_xrd_analysis(results, plotter):
    """Display detailed XRD analysis"""
    st.subheader("Advanced XRD Analysis")
    
    if results.get('xrd_results') and results.get('xrd_raw'):
        xrd_res = results['xrd_results']
        xrd_raw = results['xrd_raw']
        
        # XRD figure with peak indexing
        fig = plotter.create_xrd_figure(xrd_raw, xrd_res)
        st.pyplot(fig)
        
        # Peak table - show all peaks if user wants
        if xrd_res.get('peaks'):
            st.subheader("Peak Analysis")
            
            # Let user choose how many peaks to show
            n_peaks_total = xrd_res.get('n_peaks_total', len(xrd_res['peaks']))
            n_to_show = st.slider(
                "Number of peaks to display",
                min_value=1,
                max_value=min(20, n_peaks_total),
                value=min(10, n_peaks_total),
                help="Show the most intensive peaks"
            )
            
            # Sort peaks by intensity for display
            all_peaks = sorted(xrd_res['peaks'], key=lambda x: x['intensity'], reverse=True)
            
            peaks_df = pd.DataFrame([
                {
                    'Rank': i+1,
                    '2θ (°)': peak['position'],
                    'd-spacing (Å)': peak['d_spacing'],
                    'Intensity': peak['intensity'],
                    'FWHM (°)': peak['fwhm_deg'],
                    'hkl': peak.get('hkl', peak.get('hkl_detail', {}).get('hkl', '')),
                    'Size (nm)': peak.get('crystallite_size', 0)
                }
                for i, peak in enumerate(all_peaks[:n_to_show])
            ])
            
            st.dataframe(peaks_df.style.format({
                'Rank': '{:.0f}',
                '2θ (°)': '{:.3f}',
                'd-spacing (Å)': '{:.3f}',
                'Intensity': '{:.0f}',
                'FWHM (°)': '{:.3f}',
                'Size (nm)': '{:.1f}'
            }))
        
        # Crystallite size analysis
        with st.expander("🔬 Crystallite Size Analysis", expanded=False):
            size = xrd_res['crystallite_size']
            st.write(f"**Scherrer Method:** {size['scherrer']:.1f} nm")
            st.write(f"**Williamson-Hall Method:** {size['williamson_hall']:.1f} nm" if size['williamson_hall'] else "N/A")
            st.write(f"**Size Distribution:** {size['distribution']}" if size['distribution'] else "")
            
            if xrd_res['microstrain']:
                st.write(f"**Microstrain:** {xrd_res['microstrain']:.4f}")
                st.write(f"**Dislocation Density:** {xrd_res['dislocation_density']:.2e} m⁻²")
        
        # Download full peak data
        if xrd_res.get('peaks'):
            st.subheader("Download Peak Data")
            
            # Create CSV with all peaks
            all_peaks_data = []
            for i, peak in enumerate(xrd_res['peaks']):
                all_peaks_data.append({
                    'peak_number': i+1,
                    'two_theta_deg': peak['position'],
                    'd_spacing_angstrom': peak.get('d_spacing', 0),
                    'intensity': peak['intensity'],
                    'fwhm_deg': peak['fwhm_deg'],
                    'fwhm_rad': peak['fwhm_rad'],
                    'peak_area': peak.get('area', 0),
                    'asymmetry': peak.get('asymmetry', 1.0),
                    'crystallite_size_nm': peak.get('crystallite_size', 0),
                    'hkl': peak.get('hkl', ''),
                    'hkl_h': peak.get('hkl_detail', {}).get('h', ''),
                    'hkl_k': peak.get('hkl_detail', {}).get('k', ''),
                    'hkl_l': peak.get('hkl_detail', {}).get('l', '')
                })
            
            peaks_df_full = pd.DataFrame(all_peaks_data)
            csv = peaks_df_full.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Full Peak Data (CSV)",
                data=csv,
                file_name="xrd_peak_analysis.csv",
                mime="text/csv"
            )
def display_morphology(results):
    """Display morphology visualization and interpretation"""
    st.subheader("Integrated Morphology Analysis")
    
    # Check if we have BET results (even if failed)
    if results.get('bet_results'):
        bet = results['bet_results']
        xrd = results.get('xrd_results')
        
        # We can still do morphology analysis even without fusion results
        try:
            # Initialize morphology analyzer
            from morphology_visualizer import IntegratedMorphologyAnalyzer
            analyzer = IntegratedMorphologyAnalyzer()
            
            # Perform morphology analysis
            morphology = analyzer.analyze_morphology(bet, xrd)
            
            if morphology['valid']:
                # Display visualization
                st.subheader("Morphology Visualization")
                st.pyplot(morphology['visualization'])
                
                # Display interpretation in expanders
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("📊 Material Classification", expanded=True):
                        classification = morphology['classification']
                        st.write(f"**Primary Type:** {classification['primary']}")
                        st.write("**Typical Examples:**")
                        for example in classification.get('examples', []):
                            st.write(f"- {example}")
                        
                        if 'characteristics' in classification:
                            st.write("**Key Characteristics:**")
                            for char in classification['characteristics']:
                                st.write(f"- {char}")
                
                with col2:
                    with st.expander("🔍 Structure Properties", expanded=True):
                        properties = morphology['structure_properties']
                        
                        # Create metrics
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("Porosity", f"{properties.get('porosity_percentage', 0):.1f}%")
                        with col_b:
                            st.metric("S/V Ratio", f"{properties.get('surface_to_volume_ratio', 0):.0f} m²/cm³")
                        with col_c:
                            st.metric("Accessibility", f"{properties.get('accessibility_factor', 0):.1f}/1.0")
                        
                        if 'crystallinity_index' in properties:
                            st.metric("Crystallinity", f"{properties['crystallinity_index']:.2f}")
                        if 'estimated_wall_thickness' in properties:
                            st.metric("Wall Thickness", f"{properties['estimated_wall_thickness']:.1f} nm")
                
                # Detailed interpretation
                st.subheader("Scientific Interpretation")
                
                interpretation = morphology['interpretation']
                
                # Create interpretation cards
                cols = st.columns(3)
                
                with cols[0]:
                    st.info(f"**Porosity:** {interpretation.get('porosity_level', 'N/A')}")
                    st.caption(interpretation.get('porosity_description', ''))
                
                with cols[1]:
                    st.info(f"**Surface Area:** {interpretation.get('surface_area_level', 'N/A')}")
                    st.caption(interpretation.get('surface_area_description', ''))
                
                with cols[2]:
                    st.info(f"**Pore Size:** {interpretation.get('pore_size_type', 'N/A')}")
                    st.caption(interpretation.get('pore_size_description', ''))
                
                # Additional information
                with st.expander("🧪 Detailed Morphological Analysis", expanded=False):
                    if 'hierarchy' in interpretation:
                        st.write(f"**Structural Hierarchy:** {interpretation['hierarchy']}")
                        st.write(interpretation.get('hierarchy_description', ''))
                    
                    if 'crystallinity' in interpretation:
                        st.write(f"**Crystallinity:** {interpretation['crystallinity']}")
                        st.write(interpretation.get('crystal_description', ''))
                    
                    # Add structure-property relationships if available from fusion results
                    if results.get('fusion_results') and 'structure_property_relationships' in results['fusion_results']:
                        st.write("**Structure-Property Relationships:**")
                        for relationship in results['fusion_results']['structure_property_relationships']:
                            st.write(f"• {relationship}")
                
                # Applications and recommendations
                with st.expander("🚀 Applications & Recommendations", expanded=False):
                    # Check if we have fusion results
                    if results.get('fusion_results'):
                        fusion = results['fusion_results']
                        if 'suggested_applications' in fusion:
                            st.write("**Suggested Applications:**")
                            for app in fusion['suggested_applications']:
                                st.write(f"• {app}")
                        
                        if 'recommended_techniques' in fusion:
                            st.write("\n**Recommended Further Characterization:**")
                            for tech in fusion['recommended_techniques']:
                                st.write(f"• {tech}")
                    else:
                        # Provide general recommendations based on morphology
                        if interpretation.get('porosity_level') == 'Very High':
                            st.write("**Suggested Applications:**")
                            st.write("• Gas adsorption and storage")
                            st.write("• Catalyst support materials")
                            st.write("• Environmental remediation")
                        
                        if bet.get('surface_area', 0) > 1000:
                            st.write("**Recommended Further Characterization:**")
                            st.write("• CO₂ adsorption for micropore analysis")
                            st.write("• High-pressure gas adsorption")
                
                # Download visualization
                st.subheader("Download Visualization")
                
                # Save figure to buffer
                import io
                buf = io.BytesIO()
                morphology['visualization'].savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📥 Download Morphology Figure (PNG)",
                        data=buf,
                        file_name="morphology_visualization.png",
                        mime="image/png"
                    )
                
                with col2:
                    # Generate morphology report
                    report_text = generate_morphology_report(morphology, bet, xrd)
                    st.download_button(
                        label="📄 Download Morphology Report (TXT)",
                        data=report_text,
                        file_name="morphology_report.txt",
                        mime="text/plain"
                    )
            else:
                st.error("Morphology visualization failed")
                if 'error' in morphology:
                    st.error(f"Error: {morphology['error']}")
                
        except Exception as e:
            st.error(f"Error in morphology analysis: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.warning("BET analysis data required for morphology visualization")

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

def display_methods(results, params):
    """Display detailed methods section"""
    st.subheader("📚 Methods & Calculations")
    
    # BET Methods
    with st.expander("BET Surface Area Analysis", expanded=True):
        st.markdown("""
        **BET Equation (Multilayer Adsorption Theory):**
        ```
        p/(n(1-p)) = 1/(n_m·C) + (C-1)/(n_m·C)·p
        ```
        where:
        - p = relative pressure (P/P₀)
        - n = amount adsorbed (mmol/g)
        - n_m = monolayer capacity
        - C = BET constant
        
        **IUPAC Compliance:**
        - Linear range selected using Rouquerol criteria
        - C > 0 for physical meaningfulness
        - R² > 0.999 for validity
        - Pressure range: 0.05-0.35 P/P₀ (N₂ at 77K)
        
        **Surface Area Calculation:**
        ```
        S_BET = n_m · N_A · σ · 10⁻²⁰
        ```
        where:
        - N_A = 6.022×10²³ mol⁻¹ (Avogadro's number)
        - σ = 0.162 nm² (N₂ cross-sectional area)
        """)
        
        if results.get('bet_results'):
            bet = results['bet_results']
            st.write(f"**Applied Parameters:**")
            st.write(f"- Cross-section: {bet['cross_section']:.3f} nm²")
            st.write(f"- Temperature: {bet['temperature']:.1f} K")
            st.write(f"- BET Range: {bet['bet_regression']['p_min']:.3f}-{bet['bet_regression']['p_max']:.3f} P/P₀")
    
    # Porosity Methods
    with st.expander("Porosity Analysis", expanded=False):
        st.markdown("""
        **t-Plot Method (Harkins-Jura):**
        ```
        t = [13.99/(0.034 - log(p))]^(1/2) × 0.1
        ```
        where t is statistical thickness in nm.
        
        **Pore Size Distribution (BJH):**
        - Kelvin equation for pore radius
        - Cylindrical pore model
        - Desorption branch used for hysteresis analysis
        """)
    
    # XRD Methods
    with st.expander("XRD Analysis", expanded=False):
        st.markdown("""
        **Scherrer Equation:**
        ```
        D = K·λ/(β·cosθ)
        ```
        where:
        - D = crystallite size (nm)
        - K = shape factor (0.9)
        - λ = X-ray wavelength (nm)
        - β = FWHM in radians
        - θ = Bragg angle
        
        **Williamson-Hall Plot:**
        ```
        β·cosθ = K·λ/D + 4ε·sinθ
        ```
        where ε is microstrain.
        
        **Crystallinity Index:**
        ```
        CI = A_crystalline/(A_crystalline + A_amorphous)
        ```
        """)
        
        if results.get('xrd_results'):
            xrd = results['xrd_results']
            st.write(f"**Applied Parameters:**")
            st.write(f"- Wavelength: {xrd['wavelength']:.5f} nm")
            st.write(f"- Scherrer constant: {xrd['scherrer_constant']}")
            st.write(f"- Background subtraction: {xrd['background_subtraction']}")

def display_export(results, params):
    """Export functionality"""
    st.subheader("📤 Export Scientific Data")
    
    # Export formats
    col1, col2 = st.columns(2)
    
    with col1:
        # Export data - be careful with None values
        export_data = {
            'metadata': {
                'analysis_name': 'BET_XRD_Analysis',
                'timestamp': results.get('timestamp', 'N/A'),
                'software_version': '3.0',
                'references': [
                    'Rouquerol et al., Pure Appl. Chem., 1994, 66, 1739',
                    'Thommes et al., Pure Appl. Chem., 2015, 87, 1051'
                ]
            },
            'parameters': results.get('parameters', {}),
            'bet_results': results.get('bet_results'),
            'xrd_results': results.get('xrd_results'),
            'fusion_results': results.get('fusion_results')
        }
        
        # Use a custom JSON encoder to handle numpy arrays and other non-serializable objects
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                return super(NumpyEncoder, self).default(obj)
        
        json_str = json.dumps(export_data, indent=2, cls=NumpyEncoder)
        
        st.download_button(
            label="📄 Download Complete Analysis (JSON)",
            data=json_str,
            file_name="scientific_analysis.json",
            mime="application/json",
            use_container_width=True
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
        color_scheme=params['export']['color_scheme'],
        font_size=params['export']['font_size']
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
            
            export_format = params['export']['figure_format']
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
            
            export_format = params['export']['figure_format']
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
    
    # Morphology/Fusion Results - CHECK IF EXISTS
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
        report.append(f"Crystallinity Index: {xrd['crystallinity_index']:.2f}")
        report.append(f"Crystallite Size (Scherrer): {xrd['crystallite_size']['scherrer']:.1f} nm")
        report.append(f"Microstrain: {xrd['microstrain']:.4f}")
        report.append(f"Ordered Mesopores: {'Yes' if xrd['ordered_mesopores'] else 'No'}")
        report.append(f"Peaks Detected: {len(xrd['peaks'])}")
        report.append("")
    
    # Morphology
    if results.get('fusion_results'):
        fusion = results['fusion_results']
        report.append("INTEGRATED MORPHOLOGY ANALYSIS")
        report.append("-" * 40)
        report.append(f"Classification: {fusion['composite_classification']}")
        report.append(f"Dominant Feature: {fusion['dominant_feature']}")
        report.append(f"Material Family: {fusion['material_family']}")
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








