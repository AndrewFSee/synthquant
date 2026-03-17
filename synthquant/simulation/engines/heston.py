"""Heston stochastic volatility simulation engine (QE scheme)."""

from __future__ import annotations

import logging

import numpy as np

from synthquant.simulation.engines.base import SimulationEngine

logger = logging.getLogger(__name__)

__all__ = ["HestonEngine"]


class HestonEngine(SimulationEngine):
    """Heston stochastic volatility simulation engine.

    Uses the Quadratic-Exponential (QE) discretisation scheme of
    Andersen (2008) for the variance process, which avoids the
    negative-variance artefact of naive Euler discretisation.
    """

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        S0: float = 100.0,
        v0: float = 0.04,
        mu: float = 0.05,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma_v: float = 0.3,
        rho: float = -0.7,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        """Simulate Heston price paths via QE scheme.

        Args:
            n_paths: Number of simulation paths.
            n_steps: Number of time steps.
            dt: Time step size in years.
            S0: Initial asset price.
            v0: Initial variance.
            mu: Asset drift (annualised).
            kappa: Mean-reversion speed of variance.
            theta: Long-run mean variance.
            sigma_v: Volatility of variance.
            rho: Correlation between asset and variance Brownians.
            random_state: Random seed or Generator.

        Returns:
            Array of shape (n_paths, n_steps+1) with simulated prices.
        """
        rng = np.random.default_rng(random_state)

        S = np.empty((n_paths, n_steps + 1))
        V = np.empty((n_paths, n_steps + 1))
        S[:, 0] = S0
        V[:, 0] = v0

        # Cholesky for correlated normals
        cov = np.array([[1.0, rho], [rho, 1.0]])
        L = np.linalg.cholesky(cov)
        sqrt_dt = np.sqrt(dt)

        # QE constants
        exp_kappa = np.exp(-kappa * dt)
        c2 = sigma_v**2 * exp_kappa * (1 - exp_kappa) / kappa
        c1 = theta * (1 - exp_kappa) ** 2

        for t in range(n_steps):
            # Conditional mean and variance of V(t+dt) | V(t)
            m = theta + (V[:, t] - theta) * exp_kappa
            s2 = V[:, t] * c2 + c1
            psi = s2 / (m**2)

            # QE sampling of V(t+dt)
            V_next = np.empty(n_paths)
            # Low dispersion regime: moment-matched normal
            mask_low = psi <= 1.5
            if np.any(mask_low):
                b2 = 2 / psi[mask_low] - 1 + np.sqrt(2 / psi[mask_low]) * np.sqrt(
                    2 / psi[mask_low] - 1
                )
                a = m[mask_low] / (1 + b2)
                Z_v = rng.standard_normal(np.sum(mask_low))
                V_next[mask_low] = a * (np.sqrt(b2) + Z_v) ** 2

            # High dispersion regime: exponential
            mask_high = ~mask_low
            if np.any(mask_high):
                p = (psi[mask_high] - 1) / (psi[mask_high] + 1)
                beta = 2 / (m[mask_high] * (1 + psi[mask_high]))
                U = rng.random(np.sum(mask_high))
                V_next[mask_high] = np.where(
                    U <= p,
                    0.0,
                    np.log((1 - p) / (1 - U)) / beta,
                )

            V[:, t + 1] = np.maximum(V_next, 0.0)

            # Asset price step (log-Euler)
            Z2 = rng.standard_normal((n_paths, 2))
            corr = Z2 @ L.T
            v_pos = np.maximum(V[:, t], 0.0)
            S[:, t + 1] = S[:, t] * np.exp(
                (mu - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * corr[:, 0]
            )

        logger.debug(f"HestonEngine QE simulated {n_paths} paths x {n_steps} steps")
        return S
