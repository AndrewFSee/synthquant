"""Rough Bergomi simulation engine."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from synthquant.simulation.engines.base import SimulationEngine

logger = logging.getLogger(__name__)

__all__ = ["RoughBergomiEngine"]


class RoughBergomiEngine(SimulationEngine):
    """Rough Bergomi stochastic volatility simulation engine.

    Implements the hybrid scheme for fractional Brownian motion
    to simulate the Rough Bergomi model of Bayer, Friz, Gatheral (2016).

    V(t) = xi * exp(eta * W^H(t) - 0.5 * eta^2 * t^{2H})
    """

    def simulate(  # type: ignore[override]
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        S0: float = 100.0,
        H: float = 0.1,
        xi: float = 0.04,
        eta: float = 1.9,
        rho: float = -0.9,
        mu: float = 0.0,
        random_state: int | np.random.Generator | None = None,
        **params: Any,
    ) -> np.ndarray:
        """Simulate Rough Bergomi asset price paths.

        Args:
            n_paths: Number of simulation paths.
            n_steps: Number of time steps.
            dt: Time step size in years.
            S0: Initial asset price.
            H: Hurst exponent (0 < H < 0.5).
            xi: Initial forward variance level.
            eta: Vol-of-vol parameter.
            rho: Correlation between price and vol Brownians.
            mu: Asset drift (annualised).
            random_state: Random seed or Generator.
            **params: Ignored.

        Returns:
            Array of shape (n_paths, n_steps+1) with simulated prices.
        """
        if not 0 < H < 0.5:
            raise ValueError(f"H must be in (0, 0.5), got {H}")

        rng = np.random.default_rng(random_state)
        alpha = H - 0.5

        # Fractional kernel weights
        kernel = np.zeros(n_steps)
        kernel[0] = dt**alpha / (1 + alpha) if alpha > -1 else 1.0
        for j in range(1, n_steps):
            kernel[j] = (
                ((j + 1) ** (alpha + 1) - j ** (alpha + 1)) * dt**alpha / (alpha + 1)
            )

        # Correlated Brownian increments
        cov = np.array([[1.0, rho], [rho, 1.0]])
        L = np.linalg.cholesky(cov)
        sqrt_dt = np.sqrt(dt)

        Z = rng.standard_normal((n_paths, n_steps, 2))
        W = Z @ L.T  # (n_paths, n_steps, 2)
        dW_V = W[:, :, 0] * sqrt_dt
        dW_S = W[:, :, 1] * sqrt_dt

        # Approximate fBm via convolution
        fBm = np.zeros((n_paths, n_steps + 1))
        for k in range(1, n_steps + 1):
            w = kernel[:k][::-1]
            fBm[:, k] = np.dot(dW_V[:, :k], w)

        # Variance and price paths
        S = np.empty((n_paths, n_steps + 1))
        S[:, 0] = S0

        for k in range(n_steps):
            t = k * dt
            v = xi * np.exp(eta * fBm[:, k] - 0.5 * eta**2 * t ** (2 * H))
            v_pos = np.maximum(v, 0.0)
            S[:, k + 1] = S[:, k] * np.exp(
                (mu - 0.5 * v_pos) * dt + np.sqrt(v_pos) * dW_S[:, k]
            )

        logger.debug(f"RoughBergomiEngine simulated {n_paths} paths x {n_steps} steps")
        return S
