"""Empirical distribution estimation from simulated paths."""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

__all__ = ["EmpiricalDistribution"]


class EmpiricalDistribution:
    """Empirical distribution of terminal returns from MC paths.

    Uses Gaussian KDE for density estimation and linear interpolation
    for the CDF.

    Args:
        bw_method: Bandwidth method passed to scipy.stats.gaussian_kde.
            Can be 'scott', 'silverman', or a scalar.
    """

    def __init__(self, bw_method: str | float = "scott") -> None:
        self.bw_method = bw_method
        self._samples: np.ndarray | None = None
        self._kde: stats.gaussian_kde | None = None

    def fit(self, paths: np.ndarray, horizon: int | None = None) -> "EmpiricalDistribution":
        """Fit the distribution to path terminal (or horizon) values.

        Args:
            paths: Array of shape (n_paths, n_steps+1).
            horizon: Column index to extract. If None, uses the last column.

        Returns:
            self (fitted distribution).
        """
        col = horizon if horizon is not None else paths.shape[1] - 1
        S0 = paths[:, 0]
        terminal = paths[:, col]
        self._samples = np.log(terminal / S0)
        self._kde = stats.gaussian_kde(self._samples, bw_method=self.bw_method)
        logger.info(
            f"EmpiricalDistribution fitted to {len(self._samples)} samples"
        )
        return self

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the KDE probability density function.

        Args:
            x: Points at which to evaluate the PDF.

        Returns:
            PDF values at x.

        Raises:
            RuntimeError: If distribution has not been fitted.
        """
        self._check_fitted()
        return self._kde.evaluate(np.asarray(x))  # type: ignore[union-attr]

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the empirical CDF via sorted sample interpolation.

        Args:
            x: Points at which to evaluate the CDF.

        Returns:
            CDF values in [0, 1].
        """
        self._check_fitted()
        sorted_samples = np.sort(self._samples)  # type: ignore[arg-type]
        n = len(sorted_samples)
        empirical_probs = np.arange(1, n + 1) / n
        return np.interp(np.asarray(x), sorted_samples, empirical_probs)

    def quantile(self, q: np.ndarray) -> np.ndarray:
        """Compute quantiles of the empirical distribution.

        Args:
            q: Probability levels in [0, 1].

        Returns:
            Quantile values.
        """
        self._check_fitted()
        return np.quantile(self._samples, np.asarray(q))  # type: ignore[arg-type]

    def confidence_interval(self, alpha: float = 0.05) -> tuple[float, float]:
        """Compute a (1-alpha) central confidence interval.

        Args:
            alpha: Total tail probability (split equally between tails).

        Returns:
            Tuple of (lower, upper) quantile values.
        """
        self._check_fitted()
        lower = float(np.quantile(self._samples, alpha / 2))   # type: ignore[arg-type]
        upper = float(np.quantile(self._samples, 1 - alpha / 2))  # type: ignore[arg-type]
        return lower, upper

    def _check_fitted(self) -> None:
        if self._samples is None:
            raise RuntimeError("Distribution has not been fitted. Call fit() first.")
