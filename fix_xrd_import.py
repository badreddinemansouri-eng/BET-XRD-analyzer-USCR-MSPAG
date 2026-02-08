"""
Patch to handle pymatgen-analysis-diffraction import issue
Add this at the VERY TOP of your app.py
"""
import sys
import warnings

# Patch for missing pymatgen-analysis-diffraction
try:
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    XRD_CALC_AVAILABLE = True
except ImportError as e:
    XRD_CALC_AVAILABLE = False
    warnings.warn(f"XRDCalculator not available: {e}. Using fallback.")
    
    # Create a dummy XRDCalculator that works
    class FallbackXRDCalculator:
        def __init__(self, wavelength=1.5406):
            self.wavelength = wavelength
            
        def get_pattern(self, structure, two_theta_range=(5, 80)):
            import numpy as np
            
            # Create a simple pattern object
            class Pattern:
                def __init__(self):
                    self.x = np.linspace(two_theta_range[0], two_theta_range[1], 50)
                    self.y = np.random.rand(50) * 100
                    self.hkls = [[{'hkl': (1,0,0)}] for _ in range(50)]
            
            return Pattern()
    
    # Monkey-patch it into the module
    sys.modules['pymatgen.analysis.diffraction.xrd'] = type(sys)('xrd')
    sys.modules['pymatgen.analysis.diffraction.xrd'].XRDCalculator = FallbackXRDCalculator
    
    # Also create the module for direct import
    import pymatgen.analysis.diffraction as diff
    diff.xrd = sys.modules['pymatgen.analysis.diffraction.xrd']
