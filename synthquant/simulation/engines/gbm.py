"""Geometric Brownian Motion simulation engine."""

from __future__ import annotations

import logging

import numpy as np

from synthquant.simulation.engines.base import SimulationEngine

logger = logging.getLogger(__name__)

__all__ = ["GBMEngine"]


class GBMEngine(SimulationEngine):
    """Geometric Brownian Motion simulation engine.

    Simulates paths using the exact GBM solution:
        S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    where Z ~ N(0,1).
    """

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        S0: float = 100.0,
        mu: float = 0.05,
        sigma: float = 0.2,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        """Simulate GBM paths.

        Args:
            n_paths: Number of simulation paths.
            n_steps: Number of time steps.
            dt: Time step size in years.
            S0: Initial price.
            mu: Drift (annualized).
            sigma: Volatility (annualized).
            random_state: Random seed or Generator for reproducibility.

        Returns:
            Array of shape (n_paths, n_steps+1) with simulated prices.
        """
        rng = np.random.default_rng(random_state)

        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)

        Z = rng.standard_normal((n_paths, n_steps))
        log_returns = drift + diffusion * Z
        log_paths = np.cumsum(log_returns, axis=1)

        paths = S0 * np.exp(np.concatenate([np.zeros((n_paths, 1)), log_paths], axis=1))
        logger.debug(f"GBM simulated {n_paths} paths x {n_steps} steps")
        return paths
