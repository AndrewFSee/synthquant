"""Heston stochastic volatility model."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["HestonModel"]


class HestonModel:
    """Heston (1993) stochastic volatility model.

    dS = mu*S dt + sqrt(V)*S dW_S
    dV = kappa*(theta - V) dt + sigma_v*sqrt(V) dW_V
    Corr(dW_S, dW_V) = rho

    Args:
        v0: Initial variance.
        kappa: Mean-reversion speed of variance.
        theta: Long-run mean variance.
        sigma_v: Volatility of variance (vol-of-vol).
        rho: Correlation between asset and variance Brownian motions.
        mu: Asset drift (annualised).
        r: Risk-free rate (for option pricing).
        q: Dividend yield.
    """

    def __init__(
        self,
        v0: float = 0.04,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma_v: float = 0.3,
        rho: float = -0.7,
        mu: float = 0.05,
        r: float = 0.05,
        q: float = 0.0,
    ) -> None:
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.mu = mu
        self.r = r
        self.q = q

    def simulate(
        self,
        S0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        random_state: int | np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate Heston price and variance paths via Euler-Maruyama.

        Args:
            S0: Initial asset price.
            T: Time horizon in years.
            n_paths: Number of Monte Carlo paths.
            n_steps: Number of time steps.
            random_state: Random seed or Generator.

        Returns:
            Tuple of (price_paths, var_paths) each with shape (n_paths, n_steps+1).
        """
        rng = np.random.default_rng(random_state)
        dt = T / n_steps

        S = np.empty((n_paths, n_steps + 1))
        V = np.empty((n_paths, n_steps + 1))
        S[:, 0] = S0
        V[:, 0] = self.v0

        # Cholesky decomposition for correlated Brownians
        cov = np.array([[1.0, self.rho], [self.rho, 1.0]])
        L = np.linalg.cholesky(cov)

        sqrt_dt = np.sqrt(dt)
        for t in range(n_steps):
            Z = rng.standard_normal((n_paths, 2))
            corr = Z @ L.T  # (n_paths, 2)
            Z_S, Z_V = corr[:, 0], corr[:, 1]

            v_pos = np.maximum(V[:, t], 0.0)
            sqrt_v = np.sqrt(v_pos)

            S[:, t + 1] = S[:, t] * np.exp(
                (self.mu - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * Z_S
            )
            V[:, t + 1] = (
                v_pos
                + self.kappa * (self.theta - v_pos) * dt
                + self.sigma_v * sqrt_v * sqrt_dt * Z_V
            )
            V[:, t + 1] = np.maximum(V[:, t + 1], 0.0)

        logger.debug(f"Heston simulated {n_paths} paths x {n_steps} steps")
        return S, V
