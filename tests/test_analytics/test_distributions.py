"""Tests for EmpiricalDistribution."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.analytics.distributions import EmpiricalDistribution


@pytest.fixture()
def fitted_dist(sample_paths: np.ndarray) -> EmpiricalDistribution:
    """EmpiricalDistribution fitted to sample GBM paths."""
    dist = EmpiricalDistribution()
    dist.fit(sample_paths)
    return dist


def test_fit_returns_self(sample_paths: np.ndarray) -> None:
    """fit() returns self for method chaining."""
    dist = EmpiricalDistribution()
    result = dist.fit(sample_paths)
    assert result is dist


def test_pdf_before_fit_raises() -> None:
    """pdf() raises RuntimeError before fitting."""
    dist = EmpiricalDistribution()
    with pytest.raises(RuntimeError, match="fitted"):
        dist.pdf(np.array([0.0]))


def test_cdf_before_fit_raises() -> None:
    """cdf() raises RuntimeError before fitting."""
    dist = EmpiricalDistribution()
    with pytest.raises(RuntimeError, match="fitted"):
        dist.cdf(np.array([0.0]))


def test_quantile_before_fit_raises() -> None:
    """quantile() raises RuntimeError before fitting."""
    dist = EmpiricalDistribution()
    with pytest.raises(RuntimeError, match="fitted"):
        dist.quantile(np.array([0.5]))


def test_pdf_is_non_negative(fitted_dist: EmpiricalDistribution) -> None:
    """PDF values are non-negative everywhere."""
    x = np.linspace(-1.0, 1.0, 50)
    pdf_vals = fitted_dist.pdf(x)
    assert np.all(pdf_vals >= 0), "PDF has negative values"


def test_cdf_is_monotone(fitted_dist: EmpiricalDistribution) -> None:
    """CDF is monotonically non-decreasing."""
    x = np.linspace(-1.0, 1.0, 100)
    cdf_vals = fitted_dist.cdf(x)
    assert np.all(np.diff(cdf_vals) >= 0)


def test_cdf_bounds(fitted_dist: EmpiricalDistribution) -> None:
    """CDF values are in [0, 1]."""
    x = np.linspace(-2.0, 2.0, 50)
    cdf_vals = fitted_dist.cdf(x)
    assert np.all(cdf_vals >= 0) and np.all(cdf_vals <= 1)


def test_quantile_monotone(fitted_dist: EmpiricalDistribution) -> None:
    """Quantile is non-decreasing."""
    q = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    vals = fitted_dist.quantile(q)
    assert np.all(np.diff(vals) >= 0)


def test_confidence_interval_ordering(fitted_dist: EmpiricalDistribution) -> None:
    """Lower bound < upper bound for confidence interval."""
    lo, hi = fitted_dist.confidence_interval(alpha=0.05)
    assert lo < hi


def test_confidence_interval_width_increases_with_alpha(
    fitted_dist: EmpiricalDistribution,
) -> None:
    """Wider alpha means narrower interval (less tail)."""
    lo_wide, hi_wide = fitted_dist.confidence_interval(alpha=0.10)
    lo_narrow, hi_narrow = fitted_dist.confidence_interval(alpha=0.02)
    assert (hi_narrow - lo_narrow) >= (hi_wide - lo_wide)


def test_horizon_parameter(sample_paths: np.ndarray) -> None:
    """fit() respects the horizon column index."""
    dist_half = EmpiricalDistribution()
    mid = sample_paths.shape[1] // 2
    dist_half.fit(sample_paths, horizon=mid)
    lo, hi = dist_half.confidence_interval(alpha=0.05)
    assert lo < hi


def test_bandwidth_silverman(sample_paths: np.ndarray) -> None:
    """EmpiricalDistribution works with 'silverman' bandwidth."""
    dist = EmpiricalDistribution(bw_method="silverman")
    dist.fit(sample_paths)
    x = np.array([0.0])
    val = dist.pdf(x)
    assert val[0] >= 0
