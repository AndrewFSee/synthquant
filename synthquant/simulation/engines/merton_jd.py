"""Merton Jump-Diffusion simulation engine."""

from __future__ import annotations

import logging

import numpy as np

from synthquant.simulation.engines.base import SimulationEngine

logger = logging.getLogger(__name__)

__all__ = ["MertonJDEngine"]


class MertonJDEngine(SimulationEngine):
    """Merton Jump-Diffusion simulation engine.

    Simulates GBM paths augmented by compound Poisson jumps:
        log S(t+dt) - log S(t) = (mu - lambda_j*kappa_j - sigma^2/2)*dt
                                 + sigma*sqrt(dt)*Z
                                 + sum_{i=1}^{N(dt)} Y_i

    where:
      - N(dt) ~ Poisson(lambda_j * dt)
      - Y_i ~ Normal(mu_j, sigma_j^2)
      - kappa_j = exp(mu_j + sigma_j^2/2) - 1
    """

    def simulate(  # type: ignore[override]
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        S0: float = 100.0,
        mu: float = 0.05,
        sigma: float = 0.15,
        lambda_j: float = 1.0,
        mu_j: float = -0.05,
        sigma_j: float = 0.10,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        """Simulate Merton JD paths.

        Args:
            n_paths: Number of simulation paths.
            n_steps: Number of time steps.
            dt: Time step size in years.
            S0: Initial price.
            mu: Continuous drift (annualised).
            sigma: Continuous volatility (annualised).
            lambda_j: Jump intensity (jumps per year).
            mu_j: Mean log jump size.
            sigma_j: Std dev of log jump size.
            random_state: Random seed or Generator.

        Returns:
            Array of shape (n_paths, n_steps+1).
        """
        rng = np.random.default_rng(random_state)
        kappa_j = np.exp(mu_j + 0.5 * sigma_j**2) - 1
        drift = (mu - lambda_j * kappa_j - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)

        Z = rng.standard_normal((n_paths, n_steps))
        n_jumps = rng.poisson(lambda_j * dt, size=(n_paths, n_steps))

        # Vectorised jump contribution
        jump_mean = n_jumps * mu_j
        jump_std = np.sqrt(n_jumps * sigma_j**2)
        log_jump = np.where(
            n_jumps > 0,
            jump_mean + jump_std * rng.standard_normal((n_paths, n_steps)),
            0.0,
        )

        log_returns = drift + diffusion * Z + log_jump
        log_paths = np.cumsum(log_returns, axis=1)
        paths = S0 * np.exp(np.concatenate([np.zeros((n_paths, 1)), log_paths], axis=1))
        logger.debug(f"MertonJD simulated {n_paths} paths x {n_steps} steps")
        return paths
