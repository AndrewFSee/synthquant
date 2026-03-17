"""Copula-based multi-asset correlated simulation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

__all__ = ["CopulaSimulator"]


class CopulaSimulator:
    """Simulates correlated multi-asset returns using copulas.

    Supported copula types:
      - 'gaussian': Gaussian copula (linear correlation)
      - 'student_t': Student-t copula (tail dependence)
      - 'clayton': Clayton copula (lower tail dependence)

    Args:
        random_state: Seed or Generator for reproducibility.
    """

    def __init__(self, random_state: int | np.random.Generator | None = None) -> None:
        self._rng = np.random.default_rng(random_state)

    def simulate(
        self,
        n_paths: int,
        n_assets: int,
        correlation_matrix: np.ndarray,
        copula_type: str = "gaussian",
        df: float = 5.0,
        theta: float = 1.0,
    ) -> np.ndarray:
        """Simulate correlated uniform samples on [0,1]^n_assets.

        Args:
            n_paths: Number of scenarios.
            n_assets: Number of assets.
            correlation_matrix: Correlation matrix of shape (n_assets, n_assets).
            copula_type: One of 'gaussian', 'student_t', 'clayton'.
            df: Degrees of freedom for Student-t copula.
            theta: Dependence parameter for Clayton copula.

        Returns:
            Array of shape (n_paths, n_assets) with uniform marginals on [0,1].

        Raises:
            ValueError: If copula_type is not recognized.
        """
        corr = np.asarray(correlation_matrix, dtype=float)
        if copula_type == "gaussian":
            U = self._gaussian_copula(n_paths, corr)
        elif copula_type == "student_t":
            U = self._student_t_copula(n_paths, corr, df)
        elif copula_type == "clayton":
            U = self._clayton_copula(n_paths, n_assets, theta)
        else:
            raise ValueError(
                f"Unknown copula_type '{copula_type}'. "
                "Choose from: 'gaussian', 'student_t', 'clayton'"
            )
        logger.debug(
            f"CopulaSimulator: {n_paths} paths x {n_assets} assets, type={copula_type}"
        )
        return U

    def generate_correlated_returns(
        self,
        correlation_matrix: np.ndarray,
        marginals: list[Callable[[np.ndarray], np.ndarray]],
        n_paths: int,
        copula_type: str = "gaussian",
        **copula_kwargs: Any,
    ) -> np.ndarray:
        """Generate correlated returns by applying marginal inverse CDFs.

        Args:
            correlation_matrix: Correlation matrix of shape (n_assets, n_assets).
            marginals: List of inverse CDF callables, one per asset.
                Each callable maps uniform samples in [0,1] to returns.
            n_paths: Number of scenarios.
            copula_type: Copula family to use.
            **copula_kwargs: Additional kwargs passed to simulate().

        Returns:
            Array of shape (n_paths, n_assets) with correlated returns.
        """
        n_assets = len(marginals)
        U = self.simulate(n_paths, n_assets, correlation_matrix, copula_type, **copula_kwargs)
        returns = np.column_stack([marginals[i](U[:, i]) for i in range(n_assets)])
        return returns

    def _gaussian_copula(self, n_paths: int, corr: np.ndarray) -> np.ndarray:
        """Sample from Gaussian copula."""
        L = np.linalg.cholesky(corr)
        Z = self._rng.standard_normal((n_paths, corr.shape[0]))
        X = Z @ L.T
        return stats.norm.cdf(X)

    def _student_t_copula(self, n_paths: int, corr: np.ndarray, df: float) -> np.ndarray:
        """Sample from Student-t copula."""
        L = np.linalg.cholesky(corr)
        Z = self._rng.standard_normal((n_paths, corr.shape[0]))
        X = Z @ L.T
        chi2 = self._rng.chisquare(df, size=(n_paths, 1))
        T = X / np.sqrt(chi2 / df)
        return stats.t.cdf(T, df=df)

    def _clayton_copula(self, n_paths: int, n_assets: int, theta: float) -> np.ndarray:
        """Sample from Clayton copula via Marshall-Olkin algorithm."""
        # Gamma frailty variable
        V = self._rng.gamma(shape=1.0 / theta, scale=1.0, size=n_paths)
        E = self._rng.exponential(scale=1.0, size=(n_paths, n_assets))
        U = (1 + E / V[:, np.newaxis]) ** (-1.0 / theta)
        return np.clip(U, 1e-10, 1 - 1e-10)
