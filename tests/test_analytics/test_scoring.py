"""Tests for probabilistic forecast scoring metrics."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.analytics.scoring import ForecastScorer


@pytest.fixture()
def ensemble_forecasts() -> np.ndarray:
    """10 observations x 200 ensemble members, centred on 0.01."""
    rng = np.random.default_rng(42)
    return rng.normal(0.01, 0.02, (10, 200))


@pytest.fixture()
def observations(ensemble_forecasts: np.ndarray) -> np.ndarray:
    """Realisations close to ensemble mean."""
    rng = np.random.default_rng(99)
    return rng.normal(0.01, 0.005, len(ensemble_forecasts))


class TestCRPS:
    def test_crps_returns_float(
        self, ensemble_forecasts: np.ndarray, observations: np.ndarray
    ) -> None:
        score = ForecastScorer.crps(ensemble_forecasts, observations)
        assert isinstance(score, float)

    def test_crps_non_negative(
        self, ensemble_forecasts: np.ndarray, observations: np.ndarray
    ) -> None:
        score = ForecastScorer.crps(ensemble_forecasts, observations)
        assert score >= 0.0

    def test_crps_perfect_forecast_near_zero(self) -> None:
        """CRPS is near zero for a degenerate (perfect) ensemble."""
        y = np.array([1.0, 2.0, 3.0])
        # Ensemble is a point mass exactly at the observation
        F = np.column_stack([y] * 500)
        score = ForecastScorer.crps(F, y)
        assert score < 0.01

    def test_crps_better_than_random(
        self, ensemble_forecasts: np.ndarray, observations: np.ndarray
    ) -> None:
        """Skill score: calibrated ensemble beats noise ensemble."""
        rng = np.random.default_rng(7)
        bad_forecasts = rng.normal(0.05, 0.10, ensemble_forecasts.shape)
        good = ForecastScorer.crps(ensemble_forecasts, observations)
        bad = ForecastScorer.crps(bad_forecasts, observations)
        assert good < bad


class TestPITHistogram:
    def test_pit_histogram_shape(
        self, ensemble_forecasts: np.ndarray, observations: np.ndarray
    ) -> None:
        counts, edges = ForecastScorer.pit_histogram(ensemble_forecasts, observations, n_bins=10)
        assert len(counts) == 10
        assert len(edges) == 11

    def test_pit_histogram_counts_sum_to_n_obs(
        self, ensemble_forecasts: np.ndarray, observations: np.ndarray
    ) -> None:
        counts, _ = ForecastScorer.pit_histogram(ensemble_forecasts, observations, n_bins=5)
        assert counts.sum() == len(observations)


class TestBrierScore:
    def test_brier_perfect_score_zero(self) -> None:
        """Perfect probability forecast gives Brier score 0."""
        p = np.array([1.0, 0.0, 1.0, 0.0])
        o = np.array([1.0, 0.0, 1.0, 0.0])
        assert ForecastScorer.brier_score(p, o) == pytest.approx(0.0)

    def test_brier_worst_score(self) -> None:
        """Worst possible forecast (confident wrong) gives score 1.0."""
        p = np.array([1.0, 0.0])
        o = np.array([0.0, 1.0])
        assert ForecastScorer.brier_score(p, o) == pytest.approx(1.0)

    def test_brier_range(
        self, ensemble_forecasts: np.ndarray, observations: np.ndarray
    ) -> None:
        """Brier score is in [0, 1]."""
        prob = np.clip(np.mean(ensemble_forecasts > 0.0, axis=1), 0, 1)
        outcomes = (observations > 0.0).astype(float)
        score = ForecastScorer.brier_score(prob, outcomes)
        assert 0.0 <= score <= 1.0


class TestCoverageTest:
    def test_coverage_test_keys(
        self, ensemble_forecasts: np.ndarray, observations: np.ndarray
    ) -> None:
        result = ForecastScorer.coverage_test(ensemble_forecasts, observations, alpha=0.05)
        assert {"nominal_coverage", "empirical_coverage", "coverage_error"} == set(result.keys())

    def test_nominal_coverage_correct(
        self, ensemble_forecasts: np.ndarray, observations: np.ndarray
    ) -> None:
        result = ForecastScorer.coverage_test(ensemble_forecasts, observations, alpha=0.05)
        assert result["nominal_coverage"] == pytest.approx(0.95)

    def test_empirical_coverage_in_range(
        self, ensemble_forecasts: np.ndarray, observations: np.ndarray
    ) -> None:
        result = ForecastScorer.coverage_test(ensemble_forecasts, observations)
        assert 0.0 <= result["empirical_coverage"] <= 1.0


class TestKSTest:
    def test_ks_test_keys(self, sample_returns: np.ndarray) -> None:
        rng = np.random.default_rng(0)
        actual = rng.normal(0, 0.01, 252)
        result = ForecastScorer.ks_test(sample_returns, actual)
        assert {"statistic", "p_value"} == set(result.keys())

    def test_ks_identical_distributions_high_pvalue(self, sample_returns: np.ndarray) -> None:
        """KS test on the same distribution gives high p-value."""
        result = ForecastScorer.ks_test(sample_returns, sample_returns)
        assert result["p_value"] == pytest.approx(1.0) or result["statistic"] == pytest.approx(0.0)

    def test_ks_very_different_distributions_low_pvalue(self) -> None:
        """KS test on clearly different distributions gives low p-value."""
        rng = np.random.default_rng(5)
        a = rng.normal(0, 0.01, 1000)
        b = rng.normal(0.10, 0.01, 1000)
        result = ForecastScorer.ks_test(a, b)
        assert result["p_value"] < 0.01
