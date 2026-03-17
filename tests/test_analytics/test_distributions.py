"""Tests for empirical distribution estimation."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.analytics.distributions import EmpiricalDistribution
from synthquant.simulation.engines.gbm import GBMEngine


@pytest.fixture()
def fitted_dist() -> EmpiricalDistribution:
    """EmpiricalDistribution fitted to 1000 GBM paths."""
    engine = GBMEngine()
    paths = engine.simulate(1000, 252, 1 / 252, random_state=42)
    dist = EmpiricalDistribution()
    dist.fit(paths)
    return dist


# ── Fitting ───────────────────────────────────────────────────────────────────

def test_fit_returns_self() -> None:
    """fit() returns the distribution instance (method chaining)."""
    engine = GBMEngine()
    paths = engine.simulate(100, 10, 1 / 252, random_state=0)
    dist = EmpiricalDistribution()
    result = dist.fit(paths)
    assert result is dist


def test_raises_before_fitting() -> None:
    """pdf() raises RuntimeError if distribution has not been fitted."""
    dist = EmpiricalDistribution()
    with pytest.raises(RuntimeError, match="fitted"):
        dist.pdf(np.array([0.0]))


def test_fit_with_horizon_parameter() -> None:
    """fit() with explicit horizon extracts the right column."""
    engine = GBMEngine()
    paths = engine.simulate(200, 20, 1 / 252, random_state=1)
    dist = EmpiricalDistribution()
    dist.fit(paths, horizon=10)  # mid-horizon, not terminal
    # Should not raise
    _ = dist.pdf(np.array([0.0]))


# ── PDF ───────────────────────────────────────────────────────────────────────

def test_pdf_non_negative(fitted_dist: EmpiricalDistribution) -> None:
    """PDF values are non-negative."""
    x = np.linspace(-0.5, 0.5, 50)
    pdf_vals = fitted_dist.pdf(x)
    assert np.all(pdf_vals >= 0)


def test_pdf_peaks_near_distribution_center(fitted_dist: EmpiricalDistribution) -> None:
    """PDF has higher density near the mean than in the tails."""
    x_center = np.array([0.05])  # near the typical annual log return
    x_tail = np.array([-2.0])
    assert fitted_dist.pdf(x_center)[0] > fitted_dist.pdf(x_tail)[0]


# ── CDF ───────────────────────────────────────────────────────────────────────

def test_cdf_is_monotonically_non_decreasing(fitted_dist: EmpiricalDistribution) -> None:
    """CDF values are non-decreasing."""
    x = np.linspace(-0.5, 0.5, 100)
    cdf_vals = fitted_dist.cdf(x)
    assert np.all(np.diff(cdf_vals) >= 0)


def test_cdf_at_minus_infinity_near_zero(fitted_dist: EmpiricalDistribution) -> None:
    """CDF at a very low value is approximately 0."""
    cdf_low = fitted_dist.cdf(np.array([-10.0]))
    assert cdf_low[0] < 0.01


def test_cdf_at_plus_infinity_near_one(fitted_dist: EmpiricalDistribution) -> None:
    """CDF at a very high value is approximately 1."""
    cdf_high = fitted_dist.cdf(np.array([10.0]))
    assert cdf_high[0] > 0.99


# ── Quantile ──────────────────────────────────────────────────────────────────

def test_quantile_median_is_reasonable(fitted_dist: EmpiricalDistribution) -> None:
    """Median log return from 252-step GBM with positive drift is positive."""
    median = fitted_dist.quantile(np.array([0.50]))[0]
    assert median > 0  # mu=0.07 > 0


def test_quantile_monotonically_increasing(fitted_dist: EmpiricalDistribution) -> None:
    """Quantiles are non-decreasing."""
    q = np.array([0.1, 0.25, 0.50, 0.75, 0.90])
    vals = fitted_dist.quantile(q)
    assert np.all(np.diff(vals) >= 0)


def test_quantile_p0_less_than_p100(fitted_dist: EmpiricalDistribution) -> None:
    """0th percentile < 100th percentile."""
    low = fitted_dist.quantile(np.array([0.0]))[0]
    high = fitted_dist.quantile(np.array([1.0]))[0]
    assert low < high


# ── Confidence Interval ───────────────────────────────────────────────────────

def test_confidence_interval_lower_less_than_upper(fitted_dist: EmpiricalDistribution) -> None:
    """Lower CI bound is less than upper CI bound."""
    lower, upper = fitted_dist.confidence_interval(alpha=0.05)
    assert lower < upper


def test_confidence_interval_95_wider_than_90(fitted_dist: EmpiricalDistribution) -> None:
    """95% CI is wider than 90% CI."""
    l95, u95 = fitted_dist.confidence_interval(alpha=0.05)
    l90, u90 = fitted_dist.confidence_interval(alpha=0.10)
    assert (u95 - l95) > (u90 - l90)
