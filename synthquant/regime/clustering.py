"""Gaussian Mixture Model clustering-based regime detector."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ClusteringRegimeDetector"]


class ClusteringRegimeDetector:
    """Gaussian Mixture Model regime detector using scikit-learn.

    Unlike HMM, this approach does not model temporal transitions;
    it clusters feature vectors independently at each time step.

    Args:
        n_components: Number of mixture components (regimes).
        covariance_type: GMM covariance type ('full', 'diag', 'tied', 'spherical').
        n_init: Number of initialisations.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 2,
        covariance_type: str = "full",
        n_init: int = 5,
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.random_state = random_state
        self._model: Any = None

    def fit(self, features: np.ndarray) -> "ClusteringRegimeDetector":
        """Fit the GMM to a feature matrix.

        Args:
            features: Array of shape (n, d). Each row is one observation.
                      For a univariate return series, reshape to (-1, 1).

        Returns:
            self (fitted detector).
        """
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required. Install with: pip install synthquant"
            ) from e

        X = np.asarray(features, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        self._model.fit(X)
        logger.info(
            f"ClusteringRegimeDetector fitted: {self.n_components} components, "
            f"converged={self._model.converged_}"
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict regime labels.

        Args:
            features: Array of shape (n, d) or (n,).

        Returns:
            Integer array of regime labels, shape (n,).
        """
        self._check_fitted()
        X = np.asarray(features, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        labels = self._model.predict(X)
        logger.debug(f"ClusteringRegimeDetector predicted {len(labels)} labels")
        return labels

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict regime membership probabilities.

        Args:
            features: Array of shape (n, d) or (n,).

        Returns:
            Array of shape (n, n_components).
        """
        self._check_fitted()
        X = np.asarray(features, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self._model.predict_proba(X)

    def get_regime_params(self) -> dict[int, dict[str, Any]]:
        """Return fitted GMM parameters for each component.

        Returns:
            Dict mapping component index to {'means': array, 'covariances': array}.
        """
        self._check_fitted()
        params: dict[int, dict[str, Any]] = {}
        for i in range(self.n_components):
            params[i] = {
                "means": self._model.means_[i].tolist(),
                "covariances": self._model.covariances_[i].tolist(),
                "weight": float(self._model.weights_[i]),
            }
        return params

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
