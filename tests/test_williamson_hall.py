"""
Unit tests for the Williamson-Hall size-strain separation.
"""

import numpy as np
import pytest

from xrd_analyzer import williamson_hall_analysis


def _make_synthetic_peaks(D_nm, epsilon, K=0.9, wavelength=1.5406,
                          theta_degrees=None):
    """
    Generate synthetic peak dictionaries that follow the Williamson-Hall
    relation exactly:

        beta * cos(theta) = K * lambda / D + 4 * epsilon * sin(theta)

    Parameters
    ----------
    D_nm : float
        True crystallite size in nm.
    epsilon : float
        True microstrain.
    theta_degrees : list of float, optional
        Bragg angles in degrees (theta, not 2theta).
    """
    if theta_degrees is None:
        theta_degrees = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0]

    peaks = []
    D_angstrom = D_nm * 10.0

    for th_deg in theta_degrees:
        theta_rad = np.deg2rad(th_deg)
        beta_rad = (K * wavelength) / (D_angstrom * np.cos(theta_rad)) \
                   + 4.0 * epsilon * np.tan(theta_rad)

        peaks.append({
            "position": float(th_deg * 2.0),   # 2theta
            "fwhm_rad": float(beta_rad),
            "fwhm_deg": float(np.rad2deg(beta_rad)),
            "intensity": 100.0,
            "asymmetry": 1.0,
            "snr": 50.0,
            "area": 1.0,
        })
    return peaks


def test_williamson_hall_recovers_known_size():
    """Synthetic data with D = 20 nm, strain = 0 should be recovered."""
    peaks = _make_synthetic_peaks(D_nm=20.0, epsilon=0.0)
    result = williamson_hall_analysis(peaks)

    assert result is not None
    assert result["wh_valid"] is True
    assert result["crystallite_size"] == pytest.approx(20.0, rel=2e-1)
    assert result["microstrain"] == pytest.approx(0.0, abs=1e-3)


def test_williamson_hall_recovers_known_strain():
    """Synthetic data with known strain should be recovered approximately."""
    peaks = _make_synthetic_peaks(D_nm=30.0, epsilon=0.002)
    result = williamson_hall_analysis(peaks)

    assert result is not None
    assert result["wh_valid"] is True
    assert result["microstrain"] == pytest.approx(0.002, abs=5e-4)


def test_williamson_hall_requires_four_peaks():
    """With fewer than 4 peaks the function must return None."""
    peaks = _make_synthetic_peaks(D_nm=20.0, epsilon=0.0,
                                  theta_degrees=[10.0, 15.0, 20.0])
    assert williamson_hall_analysis(peaks) is None


def test_williamson_hall_rejects_asymmetric_peaks():
    """Highly asymmetric peaks must be filtered out."""
    peaks = _make_synthetic_peaks(D_nm=20.0, epsilon=0.0)
    for p in peaks:
        p["asymmetry"] = 2.5   # far from unity
    assert williamson_hall_analysis(peaks) is None


def test_williamson_hall_rejects_low_snr():
    """Peaks with SNR below the threshold must be filtered out."""
    peaks = _make_synthetic_peaks(D_nm=20.0, epsilon=0.0)
    for p in peaks:
        p["snr"] = 1.0
    assert williamson_hall_analysis(peaks) is None


def test_williamson_hall_returns_linear_fit_metadata():
    """Valid result should contain slope, intercept, R^2, and data arrays."""
    peaks = _make_synthetic_peaks(D_nm=25.0, epsilon=0.001)
    result = williamson_hall_analysis(peaks)

    assert result is not None
    for key in ("slope", "intercept", "r_squared",
                "x_data", "y_data", "n_peaks_used"):
        assert key in result
    assert result["r_squared"] >= 0.85
    assert len(result["x_data"]) == len(result["y_data"])
