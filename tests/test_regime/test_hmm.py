"""Tests for HMM regime detector."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.regime.hmm import HMMRegimeDetector


@pytest.fixture()
def fitted_detector(sample_returns: np.ndarray) -> HMMRegimeDetector:
    """Return a fitted HMMRegimeDetector."""
    detector = HMMRegimeDetector(n_components=2, random_state=42)
    detector.fit(sample_returns)
    return detector


def test_fit_returns_self(sample_returns: np.ndarray) -> None:
    """fit() returns the detector itself for method chaining."""
    detector = HMMRegimeDetector(n_components=2, random_state=42)
    result = detector.fit(sample_returns)
    assert result is detector


def test_predict_returns_correct_length(
    fitted_detector: HMMRegimeDetector, sample_returns: np.ndarray
) -> None:
    """predict() returns an array of the same length as input."""
    labels = fitted_detector.predict(sample_returns)
    assert len(labels) == len(sample_returns)


def test_predict_labels_are_valid(
    fitted_detector: HMMRegimeDetector, sample_returns: np.ndarray
) -> None:
    """predict() returns labels in {0, ..., n_components-1}."""
    labels = fitted_detector.predict(sample_returns)
    assert set(labels).issubset({0, 1})


def test_predict_proba_sums_to_one(
    fitted_detector: HMMRegimeDetector, sample_returns: np.ndarray
) -> None:
    """predict_proba() returns probability rows that sum to 1."""
    proba = fitted_detector.predict_proba(sample_returns)
    assert proba.shape == (len(sample_returns), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_proba_values_in_0_1(
    fitted_detector: HMMRegimeDetector, sample_returns: np.ndarray
) -> None:
    """All probabilities are in [0, 1]."""
    proba = fitted_detector.predict_proba(sample_returns)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


def test_get_regime_params_structure(fitted_detector: HMMRegimeDetector) -> None:
    """get_regime_params() returns a dict with 'mu' and 'sigma' per regime."""
    params = fitted_detector.get_regime_params()
    assert len(params) == 2
    for regime_id, p in params.items():
        assert "mu" in p
        assert "sigma" in p
        assert isinstance(p["mu"], float)
        assert p["sigma"] > 0.0


def test_predict_before_fit_raises() -> None:
    """predict() raises RuntimeError if model not fitted."""
    detector = HMMRegimeDetector(n_components=2)
    with pytest.raises(RuntimeError, match="fitted"):
        detector.predict(np.array([0.01, -0.02, 0.005]))


def test_different_n_components() -> None:
    """Detector works with n_components=3 on clearly separated synthetic data."""
    rng = np.random.default_rng(7)
    # Generate clearly separated 3-regime data
    seg1 = rng.normal(-0.02, 0.005, 300)
    seg2 = rng.normal(0.00, 0.005, 300)
    seg3 = rng.normal(0.02, 0.005, 300)
    returns_large = np.concatenate([seg1, seg2, seg3])
    detector = HMMRegimeDetector(n_components=3, covariance_type="diag", random_state=0)
    detector.fit(returns_large)
    labels = detector.predict(returns_large)
    assert set(labels).issubset({0, 1, 2})
    params = detector.get_regime_params()
    assert len(params) == 3
