"""Abstract base class for simulation engines."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SimulationEngine(ABC):
    """Abstract base class for Monte Carlo simulation engines.

    All engines must implement the simulate() method which returns
    an array of shape (n_paths, n_steps+1) containing simulated
    price/value paths.
    """

    @abstractmethod
    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        **params: object,
    ) -> np.ndarray:
        """Simulate paths.

        Args:
            n_paths: Number of simulation paths.
            n_steps: Number of time steps.
            dt: Time step size in years.
            **params: Model-specific parameters.

        Returns:
            Array of shape (n_paths, n_steps+1).
        """
        ...

    def _validate_params(self, **params: object) -> None:
        """Validate parameters. Override in subclasses."""
