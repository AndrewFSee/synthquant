"""Markov-Switching GARCH regime detector."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["MarkovSwitchingGARCH"]


class MarkovSwitchingGARCH:
    """Markov-Switching model with regime-conditional volatility.

    Uses statsmodels MarkovRegression to identify regimes with
    distinct mean and variance parameters.

    Args:
        k_regimes: Number of regimes.
        switching_variance: Allow variance to switch between regimes.
        switching_mean: Allow mean to switch between regimes.
    """

    def __init__(
        self,
        k_regimes: int = 2,
        switching_variance: bool = True,
        switching_mean: bool = True,
    ) -> None:
        self.k_regimes = k_regimes
        self.switching_variance = switching_variance
        self.switching_mean = switching_mean
        self._result: Any = None

    def fit(self, returns: np.ndarray) -> MarkovSwitchingGARCH:
        """Fit the Markov-Switching model.

        Args:
            returns: 1-D array of log returns.

        Returns:
            self (fitted detector).

        Raises:
            ImportError: If statsmodels is not installed.
        """
        try:
            from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        except ImportError as e:
            raise ImportError(
                "statsmodels is required. Install with: pip install synthquant"
            ) from e

        endog = np.asarray(returns, dtype=float)
        model = MarkovRegression(
            endog,
            k_regimes=self.k_regimes,
            switching_variance=self.switching_variance,
            switching_trend=self.switching_mean,
        )
        self._result = model.fit(disp=False)
        logger.info(
            f"MarkovSwitchingGARCH fitted: {self.k_regimes} regimes, "
            f"AIC={self._result.aic:.4f}"
        )
        return self

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """Predict the most likely regime at each point in time.

        Args:
            returns: 1-D array of log returns (must match training data length).

        Returns:
            Integer array of regime labels.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_fitted()
        proba = self.predict_proba(returns)
        labels = np.argmax(proba, axis=1)
        logger.debug(f"MarkovSwitching predicted {len(labels)} regime labels")
        return labels

    def predict_proba(self, returns: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Return smoothed regime probabilities.

        Args:
            returns: 1-D array of log returns (ignored; smoothed probs from training used).

        Returns:
            Array of shape (n, k_regimes) with smoothed probabilities.
        """
        self._check_fitted()
        return np.asarray(self._result.smoothed_marginal_probabilities)

    def get_regime_params(self) -> dict[int, dict[str, float]]:
        """Return fitted mean and variance for each regime.

        Returns:
            Dict mapping regime index to {'mu': float, 'sigma': float}.
        """
        self._check_fitted()
        params: dict[int, dict[str, float]] = {}
        for i in range(self.k_regimes):
            mu = float(self._result.params.get(f"const[{i}]", 0.0))
            var = float(self._result.params.get(f"sigma2[{i}]", 1.0))
            params[i] = {"mu": mu, "sigma": float(np.sqrt(var))}
        return params

    def _check_fitted(self) -> None:
        if self._result is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
