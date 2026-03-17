"""Tests for EnsembleRegimeDetector."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.regime.clustering import ClusteringRegimeDetector
from synthquant.regime.ensemble import EnsembleRegimeDetector
from synthquant.regime.hmm import HMMRegimeDetector


@pytest.fixture()
def two_detectors() -> list[HMMRegimeDetector | ClusteringRegimeDetector]:
    return [
        HMMRegimeDetector(n_components=2, random_state=0),
        ClusteringRegimeDetector(n_components=2, random_state=0),
    ]


@pytest.fixture()
def fitted_ensemble(
    two_detectors: list[HMMRegimeDetector | ClusteringRegimeDetector],
    sample_returns: np.ndarray,
) -> EnsembleRegimeDetector:
    ens = EnsembleRegimeDetector(two_detectors, method="majority_vote", n_regimes=2)
    ens.fit(sample_returns)
    return ens


def test_invalid_method_raises() -> None:
    with pytest.raises(ValueError, match="method"):
        EnsembleRegimeDetector([], method="bagging")


def test_fit_returns_self(
    two_detectors: list[HMMRegimeDetector | ClusteringRegimeDetector],
    sample_returns: np.ndarray,
) -> None:
    ens = EnsembleRegimeDetector(two_detectors, method="majority_vote", n_regimes=2)
    assert ens.fit(sample_returns) is ens


def test_majority_vote_length(
    fitted_ensemble: EnsembleRegimeDetector, sample_returns: np.ndarray
) -> None:
    labels = fitted_ensemble.predict(sample_returns)
    assert len(labels) == len(sample_returns)


def test_majority_vote_valid_labels(
    fitted_ensemble: EnsembleRegimeDetector, sample_returns: np.ndarray
) -> None:
    labels = fitted_ensemble.predict(sample_returns)
    assert set(labels).issubset({0, 1})


def test_proba_average_method(
    two_detectors: list[HMMRegimeDetector | ClusteringRegimeDetector],
    sample_returns: np.ndarray,
) -> None:
    ens = EnsembleRegimeDetector(two_detectors, method="proba_average", n_regimes=2)
    ens.fit(sample_returns)
    proba = ens.predict_proba(sample_returns)
    assert proba.shape == (len(sample_returns), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_predict_proba_majority_vote(
    fitted_ensemble: EnsembleRegimeDetector, sample_returns: np.ndarray
) -> None:
    proba = fitted_ensemble.predict_proba(sample_returns)
    assert proba.shape == (len(sample_returns), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
