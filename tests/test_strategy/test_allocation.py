"""Tests for portfolio allocation optimizers."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.strategy.allocation import MeanCVaROptimizer, RiskParityAllocator


@pytest.fixture()
def returns_matrix() -> np.ndarray:
    """500 scenarios x 4 assets."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0, 0.01, (500, 4))


@pytest.fixture()
def cov_matrix() -> np.ndarray:
    """4x4 positive-definite covariance matrix."""
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (4, 4))
    return A @ A.T / 16 * 0.0001  # Keep it small (~daily vol)


class TestMeanCVaROptimizer:
    def test_weights_sum_to_one(self, returns_matrix: np.ndarray) -> None:
        opt = MeanCVaROptimizer(alpha=0.05)
        weights = opt.optimize(returns_matrix)
        assert abs(weights.sum() - 1.0) < 1e-4

    def test_weights_non_negative(self, returns_matrix: np.ndarray) -> None:
        opt = MeanCVaROptimizer(alpha=0.05)
        weights = opt.optimize(returns_matrix)
        assert np.all(weights >= -1e-8)

    def test_n_assets_correct(self, returns_matrix: np.ndarray) -> None:
        opt = MeanCVaROptimizer()
        weights = opt.optimize(returns_matrix)
        assert len(weights) == returns_matrix.shape[1]

    def test_with_target_return_constraint(self, returns_matrix: np.ndarray) -> None:
        """Optimiser runs without error when target_return is set."""
        opt = MeanCVaROptimizer()
        weights = opt.optimize(returns_matrix, target_return=0.01)
        assert abs(weights.sum() - 1.0) < 1e-4

    def test_two_assets(self) -> None:
        rng = np.random.default_rng(1)
        R = rng.normal(0, 0.01, (200, 2))
        opt = MeanCVaROptimizer()
        w = opt.optimize(R)
        assert len(w) == 2
        assert abs(w.sum() - 1.0) < 1e-4


class TestRiskParityAllocator:
    def test_weights_sum_to_one(self, cov_matrix: np.ndarray) -> None:
        alloc = RiskParityAllocator()
        w = alloc.allocate(cov_matrix)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self, cov_matrix: np.ndarray) -> None:
        alloc = RiskParityAllocator()
        w = alloc.allocate(cov_matrix)
        assert np.all(w >= 0)

    def test_equal_cov_gives_equal_weights(self) -> None:
        """Diagonal covariance with equal variances -> equal weights."""
        sigma = np.eye(3) * 0.0001
        alloc = RiskParityAllocator()
        w = alloc.allocate(sigma)
        np.testing.assert_allclose(w, np.ones(3) / 3, atol=1e-4)

    def test_high_vol_asset_gets_lower_weight(self) -> None:
        """Higher-volatility asset should get a lower weight under risk parity."""
        sigma = np.diag([0.0001, 0.0004])  # second asset has 2x vol
        alloc = RiskParityAllocator()
        w = alloc.allocate(sigma)
        assert w[0] > w[1], f"High-vol asset got higher weight: {w}"

    def test_n_assets_correct(self, cov_matrix: np.ndarray) -> None:
        alloc = RiskParityAllocator()
        w = alloc.allocate(cov_matrix)
        assert len(w) == cov_matrix.shape[0]
