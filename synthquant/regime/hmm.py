"""Hidden Markov Model regime detector."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["HMMRegimeDetector"]


class HMMRegimeDetector:
    """Gaussian HMM-based regime detector wrapping hmmlearn.

    Fits a Gaussian HMM to a univariate return series and identifies
    distinct market regimes (e.g., bull/bear, low/high volatility).

    Args:
        n_components: Number of hidden states (regimes).
        covariance_type: HMM covariance type ('full', 'diag', 'spherical', 'tied').
        n_iter: Maximum EM iterations.
        tol: EM convergence tolerance.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 2,
        covariance_type: str = "full",
        n_iter: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self._model: Any = None

    def fit(self, returns: np.ndarray) -> HMMRegimeDetector:
        """Fit the HMM to a return series.

        Args:
            returns: 1-D array of log returns.

        Returns:
            self (fitted detector).

        Raises:
            ImportError: If hmmlearn is not installed.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as e:
            raise ImportError(
                "hmmlearn is required. Install with: pip install synthquant"
            ) from e

        X = np.asarray(returns, dtype=float).reshape(-1, 1)
        self._model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        self._model.fit(X)
        logger.info(
            f"HMMRegimeDetector fitted: {self.n_components} states, "
            f"converged={self._model.monitor_.converged}"
        )
        return self

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """Predict the most likely regime sequence.

        Args:
            returns: 1-D array of log returns.

        Returns:
            Integer array of regime labels, shape (n,).

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_fitted()
        X = np.asarray(returns, dtype=float).reshape(-1, 1)
        labels = self._model.predict(X)
        logger.debug(f"Predicted {len(labels)} regime labels")
        return labels

    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        """Compute posterior state probabilities.

        Args:
            returns: 1-D array of log returns.

        Returns:
            Array of shape (n, n_components) where each row sums to 1.
        """
        self._check_fitted()
        X = np.asarray(returns, dtype=float).reshape(-1, 1)
        _, posteriors = self._model.score_samples(X)
        return posteriors

    def get_regime_params(self) -> dict[int, dict[str, float]]:
        """Return fitted parameters for each regime.

        Returns:
            Dict mapping regime index to {'mu': float, 'sigma': float}.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_fitted()
        params: dict[int, dict[str, float]] = {}
        for i in range(self.n_components):
            mu = float(self._model.means_[i, 0])
            cov = self._model.covars_[i]
            # covars_ shape varies by covariance_type; use flat[0] as the variance
            sigma = float(np.sqrt(abs(float(cov.flat[0]))))
            params[i] = {"mu": mu, "sigma": sigma}
        return params

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
