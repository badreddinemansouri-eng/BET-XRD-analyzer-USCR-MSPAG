"""
Compatibility patch for missing pymatgen-analysis-diffraction.
Place at the very top of your app.py imports.

If pymatgen.analysis.diffraction.xrd is unavailable, this module
provides a graceful fallback. It does NOT silently fake XRD results -
it returns an empty pattern and logs a clear warning, so downstream
code can decide how to proceed.
"""

import logging
import sys
import types

logger = logging.getLogger(__name__)

XRD_CALC_AVAILABLE = False
FallbackXRDCalculator = None


try:
    from pymatgen.analysis.diffraction.xrd import XRDCalculator  # noqa: F401
    XRD_CALC_AVAILABLE = True
except ImportError as e:
    logger.warning(
        "pymatgen.analysis.diffraction.xrd not available (%s). "
        "Simulation of XRD from structures is disabled.", e
    )

    class _UnavailablePattern:
        """Empty pattern object with the same interface as pymatgen's pattern."""
        def __init__(self):
            import numpy as np
            self.x = np.array([])
            self.y = np.array([])
            self.hkls = []

    class FallbackXRDCalculator:
        """
        Placeholder XRDCalculator that returns an empty pattern.
        Downstream code MUST check `len(pattern.x) > 0` before using.
        """
        def __init__(self, wavelength=1.5406):
            self.wavelength = wavelength

        def get_pattern(self, structure, two_theta_range=(5, 80)):
            logger.warning(
                "FallbackXRDCalculator used - returning empty pattern. "
                "Install pymatgen-analysis-diffraction for real simulations."
            )
            return _UnavailablePattern()

    # Register a stand-in module so downstream imports do not crash
    stub_module = types.ModuleType('pymatgen.analysis.diffraction.xrd')
    stub_module.XRDCalculator = FallbackXRDCalculator
    sys.modules['pymatgen.analysis.diffraction.xrd'] = stub_module

    try:
        import pymatgen.analysis.diffraction as _diffraction
        _diffraction.xrd = stub_module
    except Exception as exc:
        logger.warning("Could not attach fallback xrd module: %s", exc)
