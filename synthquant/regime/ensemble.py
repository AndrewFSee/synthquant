"""Ensemble regime detector combining multiple base detectors."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["EnsembleRegimeDetector"]


class EnsembleRegimeDetector:
    """Combines multiple regime detectors via voting or probability averaging.

    All base detectors must be pre-fitted before passing to this class,
    OR fitting will be delegated to each detector's own ``fit()`` method.

    Args:
        detectors: List of fitted (or unfitted) regime detector instances.
            Each must implement ``predict(x)`` and ``predict_proba(x)``.
        method: Combination method: 'majority_vote' or 'proba_average'.
        n_regimes: Expected number of regimes. Required for 'proba_average'.
    """

    def __init__(
        self,
        detectors: list[Any],
        method: str = "majority_vote",
        n_regimes: int = 2,
    ) -> None:
        if method not in ("majority_vote", "proba_average"):
            raise ValueError("method must be 'majority_vote' or 'proba_average'")
        self.detectors = detectors
        self.method = method
        self.n_regimes = n_regimes

    def fit(self, features: np.ndarray) -> "EnsembleRegimeDetector":
        """Fit all base detectors.

        Args:
            features: Feature array passed directly to each detector's fit().

        Returns:
            self
        """
        for det in self.detectors:
            det.fit(features)
        logger.info(f"EnsembleRegimeDetector fitted {len(self.detectors)} detectors")
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict regime labels using the configured combination method.

        Args:
            features: Feature array passed to each detector.

        Returns:
            Integer array of regime labels, shape (n,).
        """
        if self.method == "majority_vote":
            all_labels = np.stack([d.predict(features) for d in self.detectors], axis=1)
            labels = np.apply_along_axis(
                lambda row: np.bincount(row, minlength=self.n_regimes).argmax(),
                axis=1,
                arr=all_labels,
            )
        else:
            avg_proba = self.predict_proba(features)
            labels = np.argmax(avg_proba, axis=1)
        logger.debug(f"Ensemble predicted {len(labels)} labels via {self.method}")
        return labels

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Average posterior probabilities across all detectors.

        Args:
            features: Feature array passed to each detector.

        Returns:
            Array of shape (n, n_regimes) with averaged probabilities.
        """
        probas = [d.predict_proba(features) for d in self.detectors]
        avg = np.mean(np.stack(probas, axis=0), axis=0)
        return avg
