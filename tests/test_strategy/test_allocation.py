"""Tests for portfolio allocation optimizers."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.strategy.allocation import MeanCVaROptimizer, RiskParityAllocator


@pytest.fixture()
def returns_matrix() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(0.0004, 0.012, (500, 4))


@pytest.fixture()
def covariance_matrix() -> np.ndarray:
    rng = np.random.default_rng(42)
    A = rng.normal(0, 1, (4, 4))
    return A.T @ A / 100 + np.eye(4) * 0.001


# ── MeanCVaROptimizer ─────────────────────────────────────────────────────────

def test_cvar_optimizer_weights_sum_to_one(returns_matrix: np.ndarray) -> None:
    """Optimal weights sum to 1."""
    opt = MeanCVaROptimizer()
    weights = opt.optimize(returns_matrix)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)


def test_cvar_optimizer_weights_non_negative(returns_matrix: np.ndarray) -> None:
    """All optimal weights are non-negative (long-only)."""
    opt = MeanCVaROptimizer()
    weights = opt.optimize(returns_matrix)
    assert np.all(weights >= -1e-8)


def test_cvar_optimizer_returns_correct_shape(returns_matrix: np.ndarray) -> None:
    """Optimize returns array of shape (n_assets,)."""
    opt = MeanCVaROptimizer()
    weights = opt.optimize(returns_matrix)
    assert weights.shape == (4,)


def test_cvar_optimizer_with_target_return(returns_matrix: np.ndarray) -> None:
    """optimize() works with a target_return constraint."""
    opt = MeanCVaROptimizer()
    weights = opt.optimize(returns_matrix, target_return=0.05)
    assert weights.shape == (4,)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-4)


def test_cvar_optimizer_alpha_affects_result() -> None:
    """Different alpha values produce different weights."""
    rng = np.random.default_rng(0)
    R = rng.normal(0.0005, 0.01, (300, 3))
    opt_strict = MeanCVaROptimizer(alpha=0.01)
    opt_relaxed = MeanCVaROptimizer(alpha=0.10)
    w_strict = opt_strict.optimize(R)
    w_relaxed = opt_relaxed.optimize(R)
    # They should be different (not identical)
    assert not np.allclose(w_strict, w_relaxed, atol=1e-4)


# ── RiskParityAllocator ───────────────────────────────────────────────────────

def test_risk_parity_weights_sum_to_one(covariance_matrix: np.ndarray) -> None:
    """ERC weights sum to 1."""
    alloc = RiskParityAllocator()
    weights = alloc.allocate(covariance_matrix)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)


def test_risk_parity_weights_non_negative(covariance_matrix: np.ndarray) -> None:
    """All ERC weights are non-negative."""
    alloc = RiskParityAllocator()
    weights = alloc.allocate(covariance_matrix)
    assert np.all(weights >= 0)


def test_risk_parity_inverse_variance_diagonal() -> None:
    """With diagonal covariance, allocator converges to inverse-variance weights."""
    vols = np.array([0.10, 0.20, 0.40])
    cov = np.diag(vols**2)
    alloc = RiskParityAllocator()
    weights = alloc.allocate(cov)
    # The iterative w = w / (Sigma @ w) algorithm converges to minimum-variance
    # (inverse-variance weights) for uncorrelated assets
    expected = (1 / vols**2) / np.sum(1 / vols**2)
    np.testing.assert_allclose(weights, expected, rtol=1e-4)


def test_risk_parity_returns_correct_shape(covariance_matrix: np.ndarray) -> None:
    """allocate() returns array of shape (n_assets,)."""
    alloc = RiskParityAllocator()
    weights = alloc.allocate(covariance_matrix)
    assert weights.shape == (4,)


def test_risk_parity_equal_vols_equal_weights() -> None:
    """With equal, uncorrelated assets, ERC gives equal weights."""
    n = 3
    cov = np.eye(n) * 0.01  # identical, uncorrelated
    alloc = RiskParityAllocator()
    weights = alloc.allocate(cov)
    np.testing.assert_allclose(weights, np.ones(n) / n, atol=1e-4)
