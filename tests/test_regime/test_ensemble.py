"""Tests for ensemble regime detector."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.regime.clustering import ClusteringRegimeDetector
from synthquant.regime.ensemble import EnsembleRegimeDetector
from synthquant.regime.hmm import HMMRegimeDetector


@pytest.fixture()
def returns_1d() -> np.ndarray:
    rng = np.random.default_rng(42)
    bull = rng.normal(0.001, 0.008, 150)
    bear = rng.normal(-0.001, 0.020, 100)
    return np.concatenate([bull, bear])


@pytest.fixture()
def ensemble(returns_1d: np.ndarray) -> EnsembleRegimeDetector:
    """Fitted ensemble of two detectors."""
    det1 = ClusteringRegimeDetector(n_components=2, random_state=0)
    det2 = ClusteringRegimeDetector(n_components=2, random_state=1)
    ens = EnsembleRegimeDetector([det1, det2], method="majority_vote", n_regimes=2)
    ens.fit(returns_1d)
    return ens


# ── Construction ──────────────────────────────────────────────────────────────

def test_invalid_method_raises() -> None:
    """Creating an ensemble with unknown method raises ValueError."""
    with pytest.raises(ValueError, match="method"):
        EnsembleRegimeDetector([], method="unknown_method")


# ── Fitting ───────────────────────────────────────────────────────────────────

def test_fit_returns_self(returns_1d: np.ndarray) -> None:
    """fit() returns the ensemble instance (method chaining)."""
    det1 = ClusteringRegimeDetector(n_components=2, random_state=0)
    det2 = ClusteringRegimeDetector(n_components=2, random_state=1)
    ens = EnsembleRegimeDetector([det1, det2], n_regimes=2)
    result = ens.fit(returns_1d)
    assert result is ens


# ── Predict (Majority Vote) ───────────────────────────────────────────────────

def test_majority_vote_length_matches_input(
    ensemble: EnsembleRegimeDetector, returns_1d: np.ndarray
) -> None:
    """predict() length matches input length."""
    labels = ensemble.predict(returns_1d)
    assert len(labels) == len(returns_1d)


def test_majority_vote_valid_labels(
    ensemble: EnsembleRegimeDetector, returns_1d: np.ndarray
) -> None:
    """All predicted labels are in [0, n_regimes)."""
    labels = ensemble.predict(returns_1d)
    assert np.all(labels >= 0) and np.all(labels < 2)


# ── Predict (Probability Average) ────────────────────────────────────────────

def test_proba_average_predict_length(returns_1d: np.ndarray) -> None:
    """proba_average predict() length matches input."""
    det1 = ClusteringRegimeDetector(n_components=2, random_state=0)
    det2 = ClusteringRegimeDetector(n_components=2, random_state=1)
    ens = EnsembleRegimeDetector([det1, det2], method="proba_average", n_regimes=2)
    ens.fit(returns_1d)
    labels = ens.predict(returns_1d)
    assert len(labels) == len(returns_1d)


# ── Predict Proba ─────────────────────────────────────────────────────────────

def test_predict_proba_shape(
    ensemble: EnsembleRegimeDetector, returns_1d: np.ndarray
) -> None:
    """predict_proba() returns (n_obs, n_regimes) array."""
    proba = ensemble.predict_proba(returns_1d)
    assert proba.shape == (len(returns_1d), 2)


def test_predict_proba_sums_to_one(
    ensemble: EnsembleRegimeDetector, returns_1d: np.ndarray
) -> None:
    """Averaged probabilities sum to 1 for each observation."""
    proba = ensemble.predict_proba(returns_1d)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_proba_in_unit_interval(
    ensemble: EnsembleRegimeDetector, returns_1d: np.ndarray
) -> None:
    """All averaged probabilities are in [0, 1]."""
    proba = ensemble.predict_proba(returns_1d)
    assert np.all(proba >= -1e-8) and np.all(proba <= 1 + 1e-8)


# ── HMM + GMM ensemble ────────────────────────────────────────────────────────

def test_hmm_gmm_ensemble(returns_1d: np.ndarray) -> None:
    """Ensemble with HMM + GMM detectors works end-to-end."""
    hmm = HMMRegimeDetector(n_components=2, random_state=0)
    gmm = ClusteringRegimeDetector(n_components=2, random_state=0)
    ens = EnsembleRegimeDetector([hmm, gmm], method="proba_average", n_regimes=2)
    ens.fit(returns_1d)
    labels = ens.predict(returns_1d)
    assert len(labels) == len(returns_1d)
