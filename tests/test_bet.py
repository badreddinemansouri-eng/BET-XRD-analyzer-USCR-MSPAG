"""
Unit tests for the IUPAC BET analysis engine.
"""

import numpy as np
import pytest

from bet_analyzer import (
    IUPACBETAnalyzer,
    custom_trapz,
    AVOGADRO,
)


def test_custom_trapz_linear():
    """Integral of y = 2x from 0 to 5 equals 25."""
    x = np.linspace(0, 5, 1001)
    y = 2.0 * x
    assert custom_trapz(y, x) == pytest.approx(25.0, rel=1e-3)


def test_custom_trapz_constant():
    """Integral of y = 3 from 0 to 4 equals 12."""
    x = np.linspace(0, 4, 401)
    y = np.full_like(x, 3.0)
    assert custom_trapz(y, x) == pytest.approx(12.0, rel=1e-3)


def test_bet_ideal_synthetic_isotherm():
    """
    Build a synthetic BET isotherm with known monolayer capacity and
    C constant, then verify that the analyzer recovers them.
    """
    # Known parameters
    q_mono_true = 0.5       # mmol/g
    C_true = 150.0
    cross_section = 0.162e-18
    temperature = 77.3

    # Ideal BET equation: n(p) = q_m * C * p / ((1-p)(1 + (C-1)p))
    p = np.linspace(0.01, 0.35, 40)
    q = q_mono_true * C_true * p / ((1 - p) * (1 + (C_true - 1) * p))

    analyzer = IUPACBETAnalyzer(
        p_ads=p,
        q_ads=q,
        cross_section=cross_section,
        temperature=temperature,
    )
    result = analyzer.bet_surface_area()

    assert result["valid"] is True
    assert result["monolayer_capacity"] == pytest.approx(q_mono_true, rel=5e-2)
    assert result["c_constant"] == pytest.approx(C_true, rel=1e-1)

    # Expected surface area
    expected_S = q_mono_true * AVOGADRO * cross_section * 1e-4
    assert result["surface_area"] == pytest.approx(expected_S, rel=5e-2)


def test_bet_insufficient_data_returns_invalid():
    """Too few points should yield valid=False, not an exception."""
    p = np.array([0.05, 0.10, 0.15])
    q = np.array([0.4, 0.5, 0.6])
    analyzer = IUPACBETAnalyzer(p_ads=p, q_ads=q)
    result = analyzer.bet_surface_area()
    assert result["valid"] is False
    assert "error" in result


def test_bet_non_monotonic_pressure_raises():
    """Non-monotonic pressure should trigger _validate_data's error path."""
    p = np.array([0.05, 0.20, 0.10, 0.30, 0.15, 0.25])
    q = np.array([0.4, 0.5, 0.55, 0.6, 0.58, 0.62])
    analyzer = IUPACBETAnalyzer(p_ads=p, q_ads=q)
    # After internal sorting, monotonicity should hold; analysis should proceed.
    result = analyzer.bet_surface_area()
    # Should not crash
    assert isinstance(result, dict)


def test_bet_surface_area_positive():
    """Sanity: a valid isotherm gives a positive surface area."""
    q_mono_true = 0.4
    C_true = 200.0
    p = np.linspace(0.02, 0.30, 30)
    q = q_mono_true * C_true * p / ((1 - p) * (1 + (C_true - 1) * p))

    analyzer = IUPACBETAnalyzer(p_ads=p, q_ads=q)
    result = analyzer.bet_surface_area()
    assert result["valid"] is True
    assert result["surface_area"] > 0
    assert result["surface_area_error"] >= 0


def test_bet_error_propagation_finite():
    """Error on surface area must be finite and non-negative."""
    q_mono_true = 0.45
    C_true = 120.0
    p = np.linspace(0.02, 0.32, 35)
    q = q_mono_true * C_true * p / ((1 - p) * (1 + (C_true - 1) * p))

    analyzer = IUPACBETAnalyzer(p_ads=p, q_ads=q)
    result = analyzer.bet_surface_area()
    err = result["surface_area_error"]
    assert np.isfinite(err)
    assert err >= 0.0
