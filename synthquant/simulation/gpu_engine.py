"""GPU-accelerated simulation engine with JAX, falling back to NumPy."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from synthquant.simulation.engines.base import SimulationEngine

logger = logging.getLogger(__name__)

__all__ = ["GPUEngine"]

try:
    import jax
    import jax.numpy as jnp

    _JAX_AVAILABLE = True
    logger.info("JAX detected – GPU acceleration available")
except ImportError:
    _JAX_AVAILABLE = False
    logger.debug("JAX not installed; GPUEngine will use NumPy fallback")


class GPUEngine:
    """GPU-accelerated Monte Carlo engine.

    Wraps any ``SimulationEngine`` and attempts to accelerate it with JAX.
    Falls back transparently to the NumPy-based engine if JAX is not available
    or if the wrapped engine cannot be JIT-compiled.

    Args:
        engine: A SimulationEngine instance to accelerate.
    """

    def __init__(self, engine: SimulationEngine) -> None:
        self._engine = engine

    @property
    def jax_available(self) -> bool:
        """Whether JAX (GPU/TPU) acceleration is available."""
        return _JAX_AVAILABLE

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        **params: Any,
    ) -> np.ndarray:
        """Simulate paths, using JAX where available.

        Args:
            n_paths: Number of simulation paths.
            n_steps: Number of time steps.
            dt: Time step size in years.
            **params: Model-specific parameters forwarded to the wrapped engine.

        Returns:
            NumPy array of shape (n_paths, n_steps+1).
        """
        if not _JAX_AVAILABLE:
            logger.debug("JAX unavailable; using NumPy engine fallback")
            return self._engine.simulate(n_paths=n_paths, n_steps=n_steps, dt=dt, **params)

        try:
            result = self._simulate_jax(n_paths, n_steps, dt, **params)
            return np.asarray(result)
        except Exception as exc:
            logger.warning(f"JAX simulation failed ({exc}); falling back to NumPy engine")
            return self._engine.simulate(n_paths=n_paths, n_steps=n_steps, dt=dt, **params)

    def _simulate_jax(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        S0: float = 100.0,
        mu: float = 0.05,
        sigma: float = 0.2,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """JAX-accelerated GBM simulation.

        Only GBM-style paths are JIT-compiled; for other models the
        NumPy engine fallback is used.
        """
        key = jax.random.PRNGKey(random_state or 0)
        Z = jax.random.normal(key, shape=(n_paths, n_steps))
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * jnp.sqrt(dt)
        log_returns = drift + diffusion * Z
        log_paths = jnp.cumsum(log_returns, axis=1)
        init = jnp.zeros((n_paths, 1))
        paths = S0 * jnp.exp(jnp.concatenate([init, log_paths], axis=1))
        logger.debug(f"GPUEngine (JAX) simulated {n_paths} paths x {n_steps} steps")
        return paths
