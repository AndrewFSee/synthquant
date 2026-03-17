"""Rough Bergomi stochastic volatility model."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["RoughBergomi"]


class RoughBergomi:
    """Rough Bergomi model (Bayer, Friz, Gatheral 2016).

    The variance process is driven by fractional Brownian motion with
    Hurst exponent H < 0.5, capturing the roughness of empirical vol surfaces.

    V(t) = xi * exp(eta * W^H(t) - 0.5 * eta^2 * t^(2H))

    where W^H is a fractional Brownian motion with Hurst exponent H.

    Args:
        H: Hurst exponent (0 < H < 0.5 for rough volatility).
        xi: Initial forward variance (level parameter).
        eta: Vol-of-vol scaling parameter.
        rho: Correlation between asset and variance Brownian motions.
        mu: Asset drift (annualised).
    """

    def __init__(
        self,
        H: float = 0.1,
        xi: float = 0.04,
        eta: float = 1.9,
        rho: float = -0.9,
        mu: float = 0.0,
    ) -> None:
        if not 0 < H < 0.5:
            raise ValueError(f"Hurst exponent H must be in (0, 0.5), got {H}")
        self.H = H
        self.xi = xi
        self.eta = eta
        self.rho = rho
        self.mu = mu

    def simulate(
        self,
        S0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        random_state: int | np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate Rough Bergomi price and variance paths.

        Uses a hybrid scheme: exact fractional kernel on [0, dt] and
        approximation for subsequent steps.

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
        alpha = self.H - 0.5  # memory parameter

        # Build fractional kernel weights (Gamma approximation)
        kernel = np.zeros(n_steps)
        kernel[0] = dt**alpha / (1 + alpha) if alpha > -1 else 1.0
        for j in range(1, n_steps):
            kernel[j] = ((j + 1) ** (alpha + 1) - j ** (alpha + 1)) * dt**alpha / (alpha + 1)

        # Correlated Brownian increments
        cov = np.array([[1.0, self.rho], [self.rho, 1.0]])
        L = np.linalg.cholesky(cov)
        sqrt_dt = np.sqrt(dt)

        Z = rng.standard_normal((n_paths, n_steps, 2))
        W = Z @ L.T  # (n_paths, n_steps, 2): [W_V increments, W_S increments]
        dW_V = W[:, :, 0] * sqrt_dt  # (n_paths, n_steps)
        dW_S = W[:, :, 1] * sqrt_dt  # (n_paths, n_steps)

        # Fractional Brownian motion via convolution with kernel
        # W^H(t_k) ≈ sum_{j=0}^{k-1} kernel[k-1-j] * dW_V[j]
        fBm = np.zeros((n_paths, n_steps + 1))
        for k in range(1, n_steps + 1):
            w = kernel[: k][::-1]  # flip kernel
            fBm[:, k] = np.sum(dW_V[:, :k] * w[np.newaxis, :], axis=1)

        # Variance process: V(t) = xi * exp(eta * W^H(t) - 0.5 * eta^2 * t^(2H))
        V = np.empty((n_paths, n_steps + 1))
        for k in range(n_steps + 1):
            t = k * dt
            V[:, k] = self.xi * np.exp(
                self.eta * fBm[:, k] - 0.5 * self.eta**2 * (t ** (2 * self.H))
            )

        # Asset price: log-Euler scheme
        S = np.empty((n_paths, n_steps + 1))
        S[:, 0] = S0
        for k in range(n_steps):
            v_pos = np.maximum(V[:, k], 0.0)
            S[:, k + 1] = S[:, k] * np.exp(
                (self.mu - 0.5 * v_pos) * dt + np.sqrt(v_pos) * dW_S[:, k]
            )

        logger.debug(f"RoughBergomi simulated {n_paths} paths x {n_steps} steps, H={self.H}")
        return S, V
