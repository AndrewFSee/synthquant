"""Jump-diffusion models: Merton (1976) and Kou (2002)."""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

__all__ = ["MertonJumpDiffusion", "KouModel"]


class MertonJumpDiffusion:
    """Merton (1976) jump-diffusion model.

    dS/S = (mu - lambda_j * kappa_j) dt + sigma dW + J dN

    where:
      - N(t) ~ Poisson(lambda_j * t) is the jump counting process
      - log(1+J) ~ Normal(mu_j, sigma_j^2)
      - kappa_j = exp(mu_j + 0.5*sigma_j^2) - 1

    Args:
        mu: Continuous drift (annualised).
        sigma: Continuous volatility (annualised).
        lambda_j: Jump intensity (expected jumps per year).
        mu_j: Mean log jump size.
        sigma_j: Std dev of log jump size.
    """

    def __init__(
        self,
        mu: float = 0.05,
        sigma: float = 0.15,
        lambda_j: float = 1.0,
        mu_j: float = -0.05,
        sigma_j: float = 0.10,
    ) -> None:
        self.mu = mu
        self.sigma = sigma
        self.lambda_j = lambda_j
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self._kappa_j = np.exp(mu_j + 0.5 * sigma_j**2) - 1

    def simulate(
        self,
        S0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        """Simulate price paths.

        Args:
            S0: Initial asset price.
            T: Time horizon in years.
            n_paths: Number of Monte Carlo paths.
            n_steps: Number of time steps.
            random_state: Random seed or Generator.

        Returns:
            Array of shape (n_paths, n_steps+1) with simulated prices.
        """
        rng = np.random.default_rng(random_state)
        dt = T / n_steps
        drift = (self.mu - self.lambda_j * self._kappa_j - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)

        Z = rng.standard_normal((n_paths, n_steps))
        # Poisson jumps
        n_jumps = rng.poisson(self.lambda_j * dt, size=(n_paths, n_steps))
        log_jump = np.where(
            n_jumps > 0,
            n_jumps * self.mu_j
            + np.sqrt(n_jumps * self.sigma_j**2) * rng.standard_normal((n_paths, n_steps)),
            0.0,
        )

        log_returns = drift + diffusion * Z + log_jump
        log_paths = np.cumsum(log_returns, axis=1)
        paths = S0 * np.exp(
            np.concatenate([np.zeros((n_paths, 1)), log_paths], axis=1)
        )
        logger.debug(f"MertonJD simulated {n_paths} paths x {n_steps} steps")
        return paths

    def european_call_price(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        n_terms: int = 50,
    ) -> float:
        """Closed-form European call price via Merton's series expansion.

        Args:
            S0: Current asset price.
            K: Strike price.
            T: Time to maturity in years.
            r: Risk-free rate.
            n_terms: Number of Poisson series terms.

        Returns:
            European call option price.
        """
        from math import exp, factorial

        price = 0.0
        lambda_prime = self.lambda_j * (1 + self._kappa_j)
        for n in range(n_terms):
            sigma_n = np.sqrt(self.sigma**2 + n * self.sigma_j**2 / T)
            r_n = r - self.lambda_j * self._kappa_j + n * (self.mu_j + 0.5 * self.sigma_j**2) / T
            d1 = (np.log(S0 / K) + (r_n + 0.5 * sigma_n**2) * T) / (sigma_n * np.sqrt(T))
            d2 = d1 - sigma_n * np.sqrt(T)
            bs_call = S0 * norm.cdf(d1) - K * exp(-r_n * T) * norm.cdf(d2)
            weight = exp(-lambda_prime * T) * (lambda_prime * T) ** n / factorial(n)
            price += weight * bs_call
        return price


class KouModel:
    """Kou (2002) double-exponential jump-diffusion model.

    Jump sizes follow a double-exponential (asymmetric Laplace) distribution:
      - Positive jumps with rate eta1 (probability p_up)
      - Negative jumps with rate eta2 (probability 1-p_up)

    Args:
        mu: Continuous drift (annualised).
        sigma: Continuous volatility (annualised).
        lambda_j: Jump intensity.
        p_up: Probability of an upward jump.
        eta1: Rate of upward exponential (>1 for finite mean).
        eta2: Rate of downward exponential (>0).
    """

    def __init__(
        self,
        mu: float = 0.05,
        sigma: float = 0.15,
        lambda_j: float = 1.0,
        p_up: float = 0.4,
        eta1: float = 10.0,
        eta2: float = 5.0,
    ) -> None:
        self.mu = mu
        self.sigma = sigma
        self.lambda_j = lambda_j
        self.p_up = p_up
        self.eta1 = eta1
        self.eta2 = eta2
        # E[e^J - 1] for compensation
        self._kappa_j = p_up * eta1 / (eta1 - 1) + (1 - p_up) * eta2 / (eta2 + 1) - 1

    def simulate(
        self,
        S0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        """Simulate price paths.

        Args:
            S0: Initial asset price.
            T: Time horizon in years.
            n_paths: Number of Monte Carlo paths.
            n_steps: Number of time steps.
            random_state: Random seed or Generator.

        Returns:
            Array of shape (n_paths, n_steps+1) with simulated prices.
        """
        rng = np.random.default_rng(random_state)
        dt = T / n_steps
        drift = (self.mu - self.lambda_j * self._kappa_j - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)

        Z = rng.standard_normal((n_paths, n_steps))
        n_jumps = rng.poisson(self.lambda_j * dt, size=(n_paths, n_steps))

        # Double-exponential jump sizes
        log_jump = np.zeros((n_paths, n_steps))
        for i in range(n_paths):
            for t in range(n_steps):
                nj = n_jumps[i, t]
                if nj > 0:
                    up_mask = rng.random(nj) < self.p_up
                    jumps = np.where(
                        up_mask,
                        rng.exponential(1 / self.eta1, nj),
                        -rng.exponential(1 / self.eta2, nj),
                    )
                    log_jump[i, t] = np.sum(jumps)

        log_returns = drift + diffusion * Z + log_jump
        log_paths = np.cumsum(log_returns, axis=1)
        paths = S0 * np.exp(
            np.concatenate([np.zeros((n_paths, 1)), log_paths], axis=1)
        )
        logger.debug(f"KouModel simulated {n_paths} paths x {n_steps} steps")
        return paths
