"""Tests for GMM clustering-based regime detector."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.regime.clustering import ClusteringRegimeDetector


@pytest.fixture()
def fitted_gmm(sample_returns: np.ndarray) -> ClusteringRegimeDetector:
    detector = ClusteringRegimeDetector(n_components=2, random_state=0)
    detector.fit(sample_returns)
    return detector


def test_fit_returns_self(sample_returns: np.ndarray) -> None:
    det = ClusteringRegimeDetector(n_components=2, random_state=0)
    assert det.fit(sample_returns) is det


def test_predict_correct_length(
    fitted_gmm: ClusteringRegimeDetector, sample_returns: np.ndarray
) -> None:
    labels = fitted_gmm.predict(sample_returns)
    assert len(labels) == len(sample_returns)


def test_predict_valid_labels(
    fitted_gmm: ClusteringRegimeDetector, sample_returns: np.ndarray
) -> None:
    labels = fitted_gmm.predict(sample_returns)
    assert set(labels).issubset({0, 1})


def test_predict_proba_shape(
    fitted_gmm: ClusteringRegimeDetector, sample_returns: np.ndarray
) -> None:
    proba = fitted_gmm.predict_proba(sample_returns)
    assert proba.shape == (len(sample_returns), 2)


def test_predict_proba_sums_to_one(
    fitted_gmm: ClusteringRegimeDetector, sample_returns: np.ndarray
) -> None:
    proba = fitted_gmm.predict_proba(sample_returns)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_before_fit_raises() -> None:
    det = ClusteringRegimeDetector(n_components=2)
    with pytest.raises(RuntimeError, match="fitted"):
        det.predict(np.array([0.01, -0.02]))


def test_get_regime_params_structure(fitted_gmm: ClusteringRegimeDetector) -> None:
    params = fitted_gmm.get_regime_params()
    assert len(params) == 2
    for _i, p in params.items():
        assert "means" in p and "covariances" in p and "weight" in p


def test_three_component_gmm() -> None:
    """GMM works with n_components=3 on clearly separable data."""
    rng = np.random.default_rng(5)
    data = np.concatenate([
        rng.normal(-0.02, 0.005, 200),
        rng.normal(0.00, 0.005, 200),
        rng.normal(0.02, 0.005, 200),
    ])
    det = ClusteringRegimeDetector(n_components=3, random_state=0)
    det.fit(data)
    labels = det.predict(data)
    assert set(labels).issubset({0, 1, 2})


def test_2d_feature_input(sample_returns: np.ndarray) -> None:
    """ClusteringRegimeDetector accepts 2-D feature matrix."""
    features = np.column_stack([sample_returns, np.abs(sample_returns)])
    det = ClusteringRegimeDetector(n_components=2, random_state=0)
    det.fit(features)
    labels = det.predict(features)
    assert len(labels) == len(sample_returns)
