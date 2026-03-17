"""Regime-Switching simulation engine."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from synthquant.simulation.engines.base import SimulationEngine

logger = logging.getLogger(__name__)

__all__ = ["RegimeSwitchingEngine"]


class RegimeSwitchingEngine(SimulationEngine):
    """Regime-Switching Monte Carlo engine.

    Simulates paths where the drift and volatility switch according to
    a discrete-time Markov chain. Each regime has its own GBM parameters.

    Args:
        regime_params: List of dicts, one per regime.
            Each dict must contain 'mu' (drift) and 'sigma' (volatility).
        transition_matrix: Row-stochastic matrix of shape (n_regimes, n_regimes).
            transition_matrix[i, j] = P(regime j | regime i).
        initial_regime: Starting regime index. If None, sampled from stationary dist.
    """

    def __init__(
        self,
        regime_params: list[dict[str, float]],
        transition_matrix: np.ndarray,
        initial_regime: int | None = None,
    ) -> None:
        self.regime_params = regime_params
        self.transition_matrix = np.asarray(transition_matrix)
        self.initial_regime = initial_regime
        self._n_regimes = len(regime_params)

    def simulate(  # type: ignore[override]
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        S0: float = 100.0,
        random_state: int | np.random.Generator | None = None,
        **params: Any,
    ) -> np.ndarray:
        """Simulate regime-switching GBM paths.

        Args:
            n_paths: Number of simulation paths.
            n_steps: Number of time steps.
            dt: Time step size in years.
            S0: Initial price.
            random_state: Random seed or Generator.
            **params: Ignored (regime params set at construction).

        Returns:
            Array of shape (n_paths, n_steps+1) with simulated prices.
        """
        rng = np.random.default_rng(random_state)

        # Determine initial regime
        if self.initial_regime is not None:
            regimes = np.full(n_paths, self.initial_regime, dtype=int)
        else:
            # Use stationary distribution
            eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
            stat_idx = np.argmin(np.abs(eigenvalues - 1.0))
            stat_dist = np.real(eigenvectors[:, stat_idx])
            stat_dist = stat_dist / stat_dist.sum()
            regimes = rng.choice(self._n_regimes, size=n_paths, p=stat_dist)

        S = np.empty((n_paths, n_steps + 1))
        S[:, 0] = S0

        for t in range(n_steps):
            # Transition each path to a new regime
            new_regimes = np.empty(n_paths, dtype=int)
            for r in range(self._n_regimes):
                mask = regimes == r
                if np.any(mask):
                    new_regimes[mask] = rng.choice(
                        self._n_regimes,
                        size=np.sum(mask),
                        p=self.transition_matrix[r],
                    )
            regimes = new_regimes

            # Simulate one step for each path using its current regime params
            mu_arr = np.array([self.regime_params[r]["mu"] for r in regimes])
            sigma_arr = np.array([self.regime_params[r]["sigma"] for r in regimes])

            Z = rng.standard_normal(n_paths)
            log_ret = (mu_arr - 0.5 * sigma_arr**2) * dt + sigma_arr * np.sqrt(dt) * Z
            S[:, t + 1] = S[:, t] * np.exp(log_ret)

        logger.debug(
            f"RegimeSwitching simulated {n_paths} paths x {n_steps} steps, "
            f"{self._n_regimes} regimes"
        )
        return S
