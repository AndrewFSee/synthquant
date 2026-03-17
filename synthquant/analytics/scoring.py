"""Probabilistic forecast scoring metrics."""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

__all__ = ["ForecastScorer"]


class ForecastScorer:
    """Evaluates the quality of probabilistic forecasts against realised outcomes.

    All methods are static and can be called on the class directly.
    """

    @staticmethod
    def crps(forecasts: np.ndarray, observations: np.ndarray) -> float:
        """Continuous Ranked Probability Score (CRPS).

        Measures the accuracy of probabilistic forecasts. Lower is better.
        Uses the ensemble representation of CRPS.

        Args:
            forecasts: Array of shape (n_obs, n_members) with forecast ensemble members.
            observations: Array of shape (n_obs,) with realised values.

        Returns:
            Mean CRPS over all observations.
        """
        F = np.asarray(forecasts, dtype=float)
        y = np.asarray(observations, dtype=float)
        n_members = F.shape[1]

        # CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
        mae = np.mean(np.abs(F - y[:, np.newaxis]), axis=1)
        spread = np.mean(
            np.abs(F[:, :, np.newaxis] - F[:, np.newaxis, :]), axis=(1, 2)
        ) / 2
        crps_vals = mae - spread
        result = float(np.mean(crps_vals))
        logger.debug(f"CRPS = {result:.6f}")
        return result

    @staticmethod
    def pit_histogram(
        forecasts: np.ndarray,
        observations: np.ndarray,
        n_bins: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Probability Integral Transform (PIT) histogram.

        A uniform PIT histogram indicates a calibrated forecast.

        Args:
            forecasts: Array of shape (n_obs, n_members).
            observations: Array of shape (n_obs,).
            n_bins: Number of histogram bins.

        Returns:
            Tuple of (counts, bin_edges).
        """
        F = np.asarray(forecasts, dtype=float)
        y = np.asarray(observations, dtype=float)
        pit = np.mean(F <= y[:, np.newaxis], axis=1)
        counts, edges = np.histogram(pit, bins=n_bins, range=(0, 1))
        logger.debug(f"PIT histogram computed: {n_bins} bins")
        return counts, edges

    @staticmethod
    def brier_score(
        prob_forecasts: np.ndarray,
        binary_outcomes: np.ndarray,
    ) -> float:
        """Brier Score for binary probability forecasts.

        Args:
            prob_forecasts: Array of shape (n,) with forecast probabilities in [0,1].
            binary_outcomes: Array of shape (n,) with 0/1 outcomes.

        Returns:
            Mean Brier score (lower is better).
        """
        p = np.asarray(prob_forecasts, dtype=float)
        o = np.asarray(binary_outcomes, dtype=float)
        score = float(np.mean((p - o) ** 2))
        logger.debug(f"Brier score = {score:.6f}")
        return score

    @staticmethod
    def coverage_test(
        forecasts: np.ndarray,
        observations: np.ndarray,
        alpha: float = 0.05,
    ) -> dict[str, float]:
        """Test whether forecast intervals achieve the nominal coverage.

        Args:
            forecasts: Array of shape (n_obs, n_members).
            observations: Array of shape (n_obs,).
            alpha: Nominal tail probability (e.g., 0.05 for 95% intervals).

        Returns:
            Dict with 'nominal_coverage', 'empirical_coverage', 'coverage_error'.
        """
        F = np.asarray(forecasts, dtype=float)
        y = np.asarray(observations, dtype=float)
        lower = np.quantile(F, alpha / 2, axis=1)
        upper = np.quantile(F, 1 - alpha / 2, axis=1)
        covered = np.mean((y >= lower) & (y <= upper))
        result = {
            "nominal_coverage": 1 - alpha,
            "empirical_coverage": float(covered),
            "coverage_error": float(covered - (1 - alpha)),
        }
        logger.debug(f"Coverage test: empirical={covered:.4f}, nominal={1-alpha:.4f}")
        return result

    @staticmethod
    def ks_test(
        simulated_returns: np.ndarray,
        actual_returns: np.ndarray,
    ) -> dict[str, float]:
        """Kolmogorov-Smirnov test comparing simulated and actual return distributions.

        Args:
            simulated_returns: 1-D array of simulated returns.
            actual_returns: 1-D array of actual returns.

        Returns:
            Dict with 'statistic' and 'p_value'.
        """
        ks_stat, p_value = stats.ks_2samp(
            np.asarray(simulated_returns, dtype=float),
            np.asarray(actual_returns, dtype=float),
        )
        result = {"statistic": float(ks_stat), "p_value": float(p_value)}
        logger.debug(f"KS test: stat={ks_stat:.4f}, p={p_value:.4f}")
        return result
