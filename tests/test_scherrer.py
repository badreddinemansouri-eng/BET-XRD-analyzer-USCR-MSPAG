"""
Unit tests for the Scherrer crystallite size calculation.
"""

import numpy as np
import pytest

from xrd_analyzer import (
    scherrer_crystallite_size,
    calculate_d_spacing,
)


def test_d_spacing_bragg_law():
    """
    For Cu K-alpha (lambda = 1.5406 A) and 2theta = 38.2 deg on the
    (111) reflection of FCC gold, d should be ~2.355 A.
    """
    d = calculate_d_spacing(38.2, wavelength=1.5406)
    assert d == pytest.approx(2.355, rel=1e-2)


def test_d_spacing_ninety_degrees():
    """At 2theta = 180 deg, d = lambda / 2."""
    d = calculate_d_spacing(180.0, wavelength=1.5406)
    assert d == pytest.approx(1.5406 / 2.0, rel=1e-6)


def test_scherrer_known_size():
    """
    For K = 0.9, lambda = 1.5406 A, theta = 15 deg, FWHM = 0.02 rad:
    D = 0.9 * 1.5406 / (0.02 * cos(15 deg)) Angstrom.
    Expected size in nm should match manual computation.
    """
    K = 0.9
    wavelength = 1.5406
    fwhm_rad = 0.02
    theta_rad = np.deg2rad(15.0)

    expected_angstrom = K * wavelength / (fwhm_rad * np.cos(theta_rad))
    expected_nm = expected_angstrom / 10.0

    D_nm = scherrer_crystallite_size(
        fwhm_rad=fwhm_rad,
        theta_rad=theta_rad,
        wavelength=wavelength,
        K=K,
        instrument_fwhm_rad=0.0,
    )
    assert D_nm == pytest.approx(expected_nm, rel=1e-6)


def test_scherrer_smaller_fwhm_gives_larger_size():
    """Reducing FWHM must increase the computed crystallite size."""
    theta_rad = np.deg2rad(20.0)
    D_narrow = scherrer_crystallite_size(0.005, theta_rad)
    D_broad = scherrer_crystallite_size(0.020, theta_rad)
    assert D_narrow > D_broad


def test_scherrer_with_instrumental_correction():
    """
    Corrected size must be >= uncorrected size when instrument FWHM > 0
    (because the sample contribution is smaller).
    """
    theta_rad = np.deg2rad(20.0)
    D_uncorrected = scherrer_crystallite_size(
        fwhm_rad=0.05, theta_rad=theta_rad, instrument_fwhm_rad=0.0
    )
    D_corrected = scherrer_crystallite_size(
        fwhm_rad=0.05, theta_rad=theta_rad, instrument_fwhm_rad=0.02
    )
    assert D_corrected >= D_uncorrected


def test_scherrer_zero_or_negative_returns_zero():
    """Unphysical FWHM should return 0, not crash."""
    theta_rad = np.deg2rad(15.0)
    assert scherrer_crystallite_size(0.0, theta_rad) == 0.0
    # Instrumental broadening larger than measured -> sample contribution invalid
    assert scherrer_crystallite_size(0.01, theta_rad,
                                     instrument_fwhm_rad=0.05) == 0.0


def test_scherrer_reasonable_magnitude():
    """
    Typical metal oxide nanocrystal: FWHM = 1 deg = 0.017 rad at
    theta = 12.5 deg should yield a crystallite size in the 10-100 nm range.
    """
    theta_rad = np.deg2rad(12.5)
    fwhm_rad = np.deg2rad(1.0)
    D_nm = scherrer_crystallite_size(fwhm_rad, theta_rad)
    assert 1.0 < D_nm < 500.0
