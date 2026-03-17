"""Tests for probabilistic forecast scoring metrics."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.analytics.scoring import ForecastScorer


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def perfect_ensemble(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble whose members are centred on the observation."""
    n_obs, n_members = 100, 50
    observations = rng.normal(0.0, 0.01, n_obs)
    forecasts = observations[:, np.newaxis] + rng.normal(0, 1e-6, (n_obs, n_members))
    return forecasts, observations


@pytest.fixture()
def noisy_ensemble(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble with significant noise relative to observations."""
    n_obs, n_members = 100, 50
    observations = rng.normal(0.0, 0.01, n_obs)
    forecasts = rng.normal(0.0, 0.05, (n_obs, n_members))
    return forecasts, observations


# ── CRPS ──────────────────────────────────────────────────────────────────────

def test_crps_returns_float(perfect_ensemble: tuple) -> None:
    """crps() returns a scalar float."""
    forecasts, observations = perfect_ensemble
    score = ForecastScorer.crps(forecasts, observations)
    assert isinstance(score, float)


def test_crps_perfect_forecast_near_zero(perfect_ensemble: tuple) -> None:
    """CRPS of a near-perfect ensemble is close to zero."""
    forecasts, observations = perfect_ensemble
    score = ForecastScorer.crps(forecasts, observations)
    assert score < 0.001


def test_crps_noisy_forecast_larger_than_perfect(
    perfect_ensemble: tuple, noisy_ensemble: tuple
) -> None:
    """CRPS of noisy ensemble exceeds that of perfect ensemble."""
    score_perfect = ForecastScorer.crps(*perfect_ensemble)
    score_noisy = ForecastScorer.crps(*noisy_ensemble)
    assert score_noisy > score_perfect


def test_crps_non_negative() -> None:
    """CRPS is always non-negative."""
    rng = np.random.default_rng(0)
    forecasts = rng.normal(0, 0.01, (50, 20))
    observations = rng.normal(0, 0.01, 50)
    assert ForecastScorer.crps(forecasts, observations) >= 0.0


# ── PIT Histogram ─────────────────────────────────────────────────────────────

def test_pit_histogram_returns_correct_shapes() -> None:
    """pit_histogram() returns counts and edges of the correct shape."""
    rng = np.random.default_rng(1)
    n_bins = 10
    forecasts = rng.normal(0, 1, (200, 50))
    observations = rng.normal(0, 1, 200)
    counts, edges = ForecastScorer.pit_histogram(forecasts, observations, n_bins=n_bins)
    assert len(counts) == n_bins
    assert len(edges) == n_bins + 1


def test_pit_histogram_counts_sum_to_n_obs() -> None:
    """Total PIT histogram counts equal number of observations."""
    rng = np.random.default_rng(2)
    n_obs = 150
    forecasts = rng.normal(0, 1, (n_obs, 30))
    observations = rng.normal(0, 1, n_obs)
    counts, _ = ForecastScorer.pit_histogram(forecasts, observations)
    assert counts.sum() == n_obs


def test_pit_histogram_edges_span_unit_interval() -> None:
    """Bin edges span [0, 1]."""
    rng = np.random.default_rng(3)
    forecasts = rng.normal(0, 1, (100, 20))
    observations = rng.normal(0, 1, 100)
    _, edges = ForecastScorer.pit_histogram(forecasts, observations)
    assert edges[0] == pytest.approx(0.0)
    assert edges[-1] == pytest.approx(1.0)


# ── Brier Score ───────────────────────────────────────────────────────────────

def test_brier_score_perfect_forecast_is_zero() -> None:
    """Brier score is 0 for a perfect binary forecast."""
    prob = np.array([1.0, 0.0, 1.0, 0.0])
    outcomes = np.array([1.0, 0.0, 1.0, 0.0])
    assert ForecastScorer.brier_score(prob, outcomes) == pytest.approx(0.0)


def test_brier_score_worst_forecast_is_one() -> None:
    """Brier score is 1 for the worst binary forecast (fully wrong)."""
    prob = np.array([0.0, 1.0, 0.0, 1.0])
    outcomes = np.array([1.0, 0.0, 1.0, 0.0])
    assert ForecastScorer.brier_score(prob, outcomes) == pytest.approx(1.0)


def test_brier_score_is_non_negative() -> None:
    """Brier score is always non-negative."""
    rng = np.random.default_rng(4)
    prob = rng.uniform(0, 1, 100)
    outcomes = rng.integers(0, 2, 100).astype(float)
    assert ForecastScorer.brier_score(prob, outcomes) >= 0.0


def test_brier_score_constant_half_is_point_25() -> None:
    """Uninformative constant 0.5 forecast has Brier score 0.25."""
    prob = np.full(100, 0.5)
    outcomes = np.array([1.0, 0.0] * 50)
    assert ForecastScorer.brier_score(prob, outcomes) == pytest.approx(0.25)


# ── Coverage Test ─────────────────────────────────────────────────────────────

def test_coverage_test_returns_correct_keys() -> None:
    """coverage_test() returns dict with required keys."""
    rng = np.random.default_rng(5)
    forecasts = rng.normal(0, 1, (100, 50))
    observations = rng.normal(0, 1, 100)
    result = ForecastScorer.coverage_test(forecasts, observations, alpha=0.05)
    assert "nominal_coverage" in result
    assert "empirical_coverage" in result
    assert "coverage_error" in result


def test_coverage_test_nominal_value() -> None:
    """nominal_coverage equals 1 - alpha."""
    rng = np.random.default_rng(6)
    forecasts = rng.normal(0, 1, (100, 50))
    observations = rng.normal(0, 1, 100)
    result = ForecastScorer.coverage_test(forecasts, observations, alpha=0.10)
    assert result["nominal_coverage"] == pytest.approx(0.90)


def test_coverage_test_error_is_difference() -> None:
    """coverage_error = empirical_coverage - nominal_coverage."""
    rng = np.random.default_rng(7)
    forecasts = rng.normal(0, 1, (100, 50))
    observations = rng.normal(0, 1, 100)
    result = ForecastScorer.coverage_test(forecasts, observations)
    assert result["coverage_error"] == pytest.approx(
        result["empirical_coverage"] - result["nominal_coverage"], abs=1e-9
    )


def test_perfect_coverage_wide_intervals() -> None:
    """Very wide prediction intervals should achieve ~100% coverage."""
    rng = np.random.default_rng(8)
    n_obs = 200
    observations = rng.normal(0, 0.01, n_obs)
    forecasts = rng.normal(0, 10.0, (n_obs, 100))  # very wide ensemble
    result = ForecastScorer.coverage_test(forecasts, observations, alpha=0.05)
    assert result["empirical_coverage"] > 0.90


# ── KS Test ───────────────────────────────────────────────────────────────────

def test_ks_test_returns_correct_keys() -> None:
    """ks_test() returns dict with 'statistic' and 'p_value'."""
    rng = np.random.default_rng(9)
    sim = rng.normal(0, 1, 500)
    actual = rng.normal(0, 1, 500)
    result = ForecastScorer.ks_test(sim, actual)
    assert "statistic" in result
    assert "p_value" in result


def test_ks_test_same_distribution_high_p_value() -> None:
    """KS test on samples from identical distributions has high p-value."""
    rng = np.random.default_rng(10)
    sim = rng.normal(0, 1, 1000)
    actual = rng.normal(0, 1, 1000)
    result = ForecastScorer.ks_test(sim, actual)
    assert result["p_value"] > 0.05


def test_ks_test_different_distributions_low_p_value() -> None:
    """KS test on samples from very different distributions has low p-value."""
    rng = np.random.default_rng(11)
    sim = rng.normal(0, 1, 500)
    actual = rng.normal(10, 1, 500)  # completely different mean
    result = ForecastScorer.ks_test(sim, actual)
    assert result["p_value"] < 0.05


def test_ks_statistic_in_range() -> None:
    """KS statistic is in [0, 1]."""
    rng = np.random.default_rng(12)
    result = ForecastScorer.ks_test(rng.normal(0, 1, 200), rng.normal(0, 1, 200))
    assert 0.0 <= result["statistic"] <= 1.0
