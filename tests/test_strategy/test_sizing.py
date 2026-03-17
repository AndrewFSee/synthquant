"""Tests for position sizing strategies."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.strategy.sizing import CVaROptimalSizer, KellyCriterion, RiskParitySizer


# ── Kelly Criterion ────────────────────────────────────────────────────────────

def test_kelly_full_positive_expected_returns() -> None:
    """Full Kelly is positive for a return series with positive mean."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.01, 252)  # positive drift
    kelly = KellyCriterion()
    fk = kelly.full_kelly(returns)
    assert fk > 0, f"Expected positive Kelly fraction, got {fk}"


def test_kelly_fractional_less_than_full() -> None:
    """Fractional Kelly < Full Kelly for fraction < 1."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.01, 252)
    kelly = KellyCriterion()
    fk = kelly.full_kelly(returns)
    half_k = kelly.fractional_kelly(returns, fraction=0.5)
    assert half_k < fk


def test_kelly_fractional_equals_full_at_fraction_1() -> None:
    """Fractional Kelly with fraction=1.0 equals full Kelly."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, 500)
    kelly = KellyCriterion()
    fk = kelly.full_kelly(returns)
    frac_1 = kelly.fractional_kelly(returns, fraction=1.0)
    np.testing.assert_allclose(frac_1, fk)


def test_kelly_zero_variance_raises() -> None:
    """KellyCriterion raises ValueError for constant returns."""
    kelly = KellyCriterion()
    with pytest.raises(ValueError, match="Zero variance"):
        kelly.full_kelly(np.ones(100) * 0.001)


def test_kelly_invalid_fraction_raises() -> None:
    """fractional_kelly raises for fraction outside (0, 1]."""
    rng = np.random.default_rng(1)
    returns = rng.normal(0.001, 0.01, 100)
    kelly = KellyCriterion()
    with pytest.raises(ValueError, match="fraction"):
        kelly.fractional_kelly(returns, fraction=0.0)
    with pytest.raises(ValueError, match="fraction"):
        kelly.fractional_kelly(returns, fraction=1.5)


# ── Risk Parity Sizer ──────────────────────────────────────────────────────────

def test_risk_parity_weights_sum_to_one() -> None:
    """RiskParitySizer weights sum to 1 (approximately)."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, (252, 4))
    rps = RiskParitySizer()
    weights = rps.compute_weights(returns, target_vol=0.10)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)


def test_risk_parity_weights_non_negative() -> None:
    """All risk-parity weights are non-negative."""
    rng = np.random.default_rng(5)
    returns = rng.normal(0, 0.01, (500, 3))
    rps = RiskParitySizer()
    weights = rps.compute_weights(returns)
    assert np.all(weights >= 0)


def test_risk_parity_equal_vol_gives_equal_weights() -> None:
    """With equal vol and zero correlation, all weights should be equal."""
    rng = np.random.default_rng(7)
    # All assets have the same vol ~1%
    returns = np.column_stack([rng.normal(0, 0.01, 500) for _ in range(3)])
    rps = RiskParitySizer()
    weights = rps.compute_weights(returns)
    # Weights should be roughly equal (within 5%)
    np.testing.assert_allclose(weights, np.ones(3) / 3, atol=0.05)


# ── CVaR Optimal Sizer ────────────────────────────────────────────────────────

def test_cvar_sizer_returns_valid_size() -> None:
    """CVaROptimalSizer returns a size in [0, 1]."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.015, 1000)
    sizer = CVaROptimalSizer()
    size = sizer.compute_size(returns, target_cvar=-0.02)
    assert 0.0 <= size <= 1.0


def test_cvar_sizer_tighter_constraint_gives_smaller_size() -> None:
    """Tighter CVaR constraint (less negative) yields a smaller position."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.015, 1000)
    sizer = CVaROptimalSizer()
    size_loose = sizer.compute_size(returns, target_cvar=-0.05)
    size_tight = sizer.compute_size(returns, target_cvar=-0.01)
    assert size_tight <= size_loose
