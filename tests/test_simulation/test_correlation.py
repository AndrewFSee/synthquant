"""Tests for CopulaSimulator."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from synthquant.simulation.correlation import CopulaSimulator


@pytest.fixture()
def corr_2x2() -> np.ndarray:
    return np.array([[1.0, 0.6], [0.6, 1.0]])


@pytest.fixture()
def corr_3x3() -> np.ndarray:
    return np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.4],
        [0.3, 0.4, 1.0],
    ])


class TestCopulaSimulatorShape:
    def test_gaussian_shape(self, corr_2x2: np.ndarray) -> None:
        sim = CopulaSimulator(random_state=0)
        U = sim.simulate(1000, 2, corr_2x2, copula_type="gaussian")
        assert U.shape == (1000, 2)

    def test_student_t_shape(self, corr_2x2: np.ndarray) -> None:
        sim = CopulaSimulator(random_state=0)
        U = sim.simulate(500, 2, corr_2x2, copula_type="student_t", df=5.0)
        assert U.shape == (500, 2)

    def test_clayton_shape(self) -> None:
        sim = CopulaSimulator(random_state=0)
        corr = np.eye(3)
        U = sim.simulate(500, 3, corr, copula_type="clayton", theta=1.5)
        assert U.shape == (500, 3)

    def test_unknown_copula_raises(self, corr_2x2: np.ndarray) -> None:
        sim = CopulaSimulator()
        with pytest.raises(ValueError, match="copula_type"):
            sim.simulate(100, 2, corr_2x2, copula_type="frank")


class TestUniformMarginals:
    @pytest.mark.parametrize("copula_type", ["gaussian", "student_t", "clayton"])
    def test_uniform_marginals(self, corr_2x2: np.ndarray, copula_type: str) -> None:
        """Each marginal should look uniform on [0, 1]."""
        sim = CopulaSimulator(random_state=42)
        kwargs = {"df": 5.0} if copula_type == "student_t" else {}
        if copula_type == "clayton":
            kwargs = {"theta": 1.0}  # type: ignore[assignment]
        U = sim.simulate(2000, 2, corr_2x2, copula_type=copula_type, **kwargs)
        assert np.all(U >= 0) and np.all(U <= 1)
        for col in range(U.shape[1]):
            _, p = stats.kstest(U[:, col], "uniform")
            assert p > 0.001, f"Marginal {col} not uniform for {copula_type}: p={p:.4f}"


class TestGaussianCorrelation:
    def test_positive_correlation_preserved(self, corr_2x2: np.ndarray) -> None:
        """Gaussian copula with rho=0.6 produces positively correlated samples."""
        sim = CopulaSimulator(random_state=0)
        U = sim.simulate(5000, 2, corr_2x2, copula_type="gaussian")
        empirical_corr = np.corrcoef(U[:, 0], U[:, 1])[0, 1]
        assert empirical_corr > 0.3, f"Expected positive correlation, got {empirical_corr:.3f}"


class TestGenerateCorrelatedReturns:
    def test_shape(self, corr_2x2: np.ndarray) -> None:
        sim = CopulaSimulator(random_state=0)
        marginals = [
            lambda u: stats.norm.ppf(u, loc=0.0, scale=0.01),
            lambda u: stats.norm.ppf(u, loc=0.0, scale=0.02),
        ]
        returns = sim.generate_correlated_returns(corr_2x2, marginals, n_paths=500)
        assert returns.shape == (500, 2)

    def test_reproducibility(self, corr_2x2: np.ndarray) -> None:
        marginals = [lambda u: stats.norm.ppf(u)] * 2
        r1 = CopulaSimulator(random_state=7).generate_correlated_returns(
            corr_2x2, marginals, n_paths=100
        )
        r2 = CopulaSimulator(random_state=7).generate_correlated_returns(
            corr_2x2, marginals, n_paths=100
        )
        np.testing.assert_array_almost_equal(r1, r2)
