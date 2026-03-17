"""Tests for GMM clustering-based regime detector."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.regime.clustering import ClusteringRegimeDetector


@pytest.fixture()
def returns_1d() -> np.ndarray:
    rng = np.random.default_rng(42)
    bull = rng.normal(0.001, 0.008, 150)
    bear = rng.normal(-0.001, 0.020, 100)
    return np.concatenate([bull, bear])


@pytest.fixture()
def features_2d(returns_1d: np.ndarray) -> np.ndarray:
    """2-D feature matrix: (returns, |returns|)."""
    return np.column_stack([returns_1d, np.abs(returns_1d)])


# ── Fitting ───────────────────────────────────────────────────────────────────

def test_fit_returns_self(returns_1d: np.ndarray) -> None:
    """fit() returns the detector instance (method chaining)."""
    det = ClusteringRegimeDetector(n_components=2, random_state=0)
    result = det.fit(returns_1d)
    assert result is det


def test_raises_before_fitting() -> None:
    """predict() raises RuntimeError if model has not been fitted."""
    det = ClusteringRegimeDetector()
    with pytest.raises(RuntimeError, match="fitted"):
        det.predict(np.array([0.0, 0.001, -0.001]))


def test_fit_1d_input(returns_1d: np.ndarray) -> None:
    """fit() accepts 1-D arrays (auto-reshapes to (n, 1))."""
    det = ClusteringRegimeDetector(n_components=2, random_state=0)
    det.fit(returns_1d)  # should not raise
    labels = det.predict(returns_1d)
    assert labels.shape == (len(returns_1d),)


def test_fit_2d_input(features_2d: np.ndarray) -> None:
    """fit() accepts 2-D feature arrays."""
    det = ClusteringRegimeDetector(n_components=2, random_state=0)
    det.fit(features_2d)
    labels = det.predict(features_2d)
    assert labels.shape == (len(features_2d),)


# ── Predict ───────────────────────────────────────────────────────────────────

def test_predict_length_matches_input(returns_1d: np.ndarray) -> None:
    """predict() output length matches input length."""
    det = ClusteringRegimeDetector(n_components=2, random_state=0)
    det.fit(returns_1d)
    labels = det.predict(returns_1d)
    assert len(labels) == len(returns_1d)


def test_predict_valid_label_range(returns_1d: np.ndarray) -> None:
    """All predicted labels are in [0, n_components)."""
    n_components = 3
    det = ClusteringRegimeDetector(n_components=n_components, random_state=0)
    det.fit(returns_1d)
    labels = det.predict(returns_1d)
    assert np.all(labels >= 0) and np.all(labels < n_components)


# ── Predict Proba ─────────────────────────────────────────────────────────────

def test_predict_proba_shape(returns_1d: np.ndarray) -> None:
    """predict_proba() returns (n_obs, n_components) array."""
    n_components = 2
    det = ClusteringRegimeDetector(n_components=n_components, random_state=0)
    det.fit(returns_1d)
    proba = det.predict_proba(returns_1d)
    assert proba.shape == (len(returns_1d), n_components)


def test_predict_proba_sums_to_one(returns_1d: np.ndarray) -> None:
    """Probabilities sum to 1 for each observation."""
    det = ClusteringRegimeDetector(n_components=2, random_state=0)
    det.fit(returns_1d)
    proba = det.predict_proba(returns_1d)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_proba_in_unit_interval(returns_1d: np.ndarray) -> None:
    """All probabilities are in [0, 1]."""
    det = ClusteringRegimeDetector(n_components=2, random_state=0)
    det.fit(returns_1d)
    proba = det.predict_proba(returns_1d)
    assert np.all(proba >= 0) and np.all(proba <= 1)


# ── Regime Params ─────────────────────────────────────────────────────────────

def test_get_regime_params_returns_dict(returns_1d: np.ndarray) -> None:
    """get_regime_params() returns a dict with keys for each component."""
    n_components = 2
    det = ClusteringRegimeDetector(n_components=n_components, random_state=0)
    det.fit(returns_1d)
    params = det.get_regime_params()
    assert len(params) == n_components
    for i in range(n_components):
        assert "means" in params[i]
        assert "covariances" in params[i]
        assert "weight" in params[i]


def test_get_regime_params_weights_sum_to_one(returns_1d: np.ndarray) -> None:
    """Component weights from get_regime_params() sum to 1."""
    det = ClusteringRegimeDetector(n_components=2, random_state=0)
    det.fit(returns_1d)
    params = det.get_regime_params()
    total_weight = sum(p["weight"] for p in params.values())
    assert total_weight == pytest.approx(1.0, abs=1e-6)
