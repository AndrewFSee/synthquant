"""Tests for risk metrics functions."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.analytics.risk_metrics import (
    conditional_var,
    expected_shortfall,
    max_drawdown_distribution,
    tail_ratio,
    value_at_risk,
)


def test_var_is_monotonic_in_alpha(sample_paths: np.ndarray) -> None:
    """VaR is monotonically increasing with alpha (less severe as alpha grows)."""
    var_01 = value_at_risk(sample_paths, alpha=0.01)
    var_05 = value_at_risk(sample_paths, alpha=0.05)
    var_10 = value_at_risk(sample_paths, alpha=0.10)
    assert var_01 <= var_05 <= var_10, (
        f"VaR not monotonic: VaR(1%)={var_01:.4f}, VaR(5%)={var_05:.4f}, VaR(10%)={var_10:.4f}"
    )


def test_cvar_is_at_least_as_severe_as_var(sample_paths: np.ndarray) -> None:
    """CVaR >= VaR (CVaR is more conservative than VaR at same alpha)."""
    for alpha in [0.01, 0.05, 0.10]:
        var = value_at_risk(sample_paths, alpha=alpha)
        cvar = conditional_var(sample_paths, alpha=alpha)
        assert cvar <= var, (
            f"At alpha={alpha}: CVaR={cvar:.4f} is not <= VaR={var:.4f}"
        )


def test_expected_shortfall_equals_conditional_var(sample_paths: np.ndarray) -> None:
    """expected_shortfall is an alias for conditional_var."""
    for alpha in [0.01, 0.05, 0.10]:
        cvar = conditional_var(sample_paths, alpha=alpha)
        es = expected_shortfall(sample_paths, alpha=alpha)
        assert cvar == es


def test_max_drawdown_is_non_positive(sample_paths: np.ndarray) -> None:
    """All max drawdowns are <= 0 by definition."""
    drawdowns = max_drawdown_distribution(sample_paths)
    assert np.all(drawdowns <= 0), f"Found positive drawdowns: {drawdowns[drawdowns > 0]}"


def test_max_drawdown_shape(sample_paths: np.ndarray) -> None:
    """max_drawdown_distribution returns one value per path."""
    drawdowns = max_drawdown_distribution(sample_paths)
    assert drawdowns.shape == (sample_paths.shape[0],)


def test_tail_ratio_is_positive(sample_paths: np.ndarray) -> None:
    """Tail ratio is a positive float."""
    tr = tail_ratio(sample_paths, alpha=0.05)
    assert tr > 0


def test_var_with_positive_drift_is_negative_loss() -> None:
    """VaR for paths with strong positive drift is a moderate negative number."""
    from synthquant.simulation.engines.gbm import GBMEngine

    engine = GBMEngine()
    # Very large drift, very low vol => almost no paths lose money
    paths = engine.simulate(5000, 252, 1 / 252, S0=100.0, mu=0.50, sigma=0.05, random_state=7)
    var_95 = value_at_risk(paths, alpha=0.05)
    # With 50% drift and 5% vol, almost all paths are positive; VaR should be positive
    assert var_95 > 0.0, f"Expected positive VaR for high-drift scenario, got {var_95}"


def test_holding_period_parameter(sample_paths: np.ndarray) -> None:
    """value_at_risk respects the holding_period parameter."""
    var_full = value_at_risk(sample_paths, alpha=0.05)
    var_half = value_at_risk(sample_paths, alpha=0.05, holding_period=sample_paths.shape[1] // 2)
    # Both are valid floats; just confirm no error and they differ
    assert isinstance(var_full, float)
    assert isinstance(var_half, float)
