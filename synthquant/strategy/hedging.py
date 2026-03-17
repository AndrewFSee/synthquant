"""Delta and gamma hedging utilities."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["DeltaHedger", "GammaHedger"]


class DeltaHedger:
    """Computes delta hedge ratios from Monte Carlo paths via finite differences.

    The delta is estimated as the finite-difference sensitivity of the option
    price to a small bump in the initial spot price.
    """

    def compute_hedge_ratio(
        self,
        paths: np.ndarray,
        K: float,
        T: float,
        r: float,
        bump_size: float = 0.01,
        option_type: str = "call",
    ) -> float:
        """Compute the option delta.

        Args:
            paths: Array of shape (n_paths, n_steps+1) with price paths.
            K: Strike price.
            T: Time to maturity in years.
            r: Risk-free rate.
            bump_size: Relative bump size for finite differences.
            option_type: 'call' or 'put'.

        Returns:
            Delta (dV/dS).
        """
        terminal = paths[:, -1]
        S0 = float(paths[0, 0])

        up_terminal = terminal * (1 + bump_size)
        down_terminal = terminal * (1 - bump_size)

        if option_type == "call":
            payoff_up = np.maximum(up_terminal - K, 0.0)
            payoff_down = np.maximum(down_terminal - K, 0.0)
        else:
            payoff_up = np.maximum(K - up_terminal, 0.0)
            payoff_down = np.maximum(K - down_terminal, 0.0)

        price_up = float(np.exp(-r * T) * np.mean(payoff_up))
        price_down = float(np.exp(-r * T) * np.mean(payoff_down))

        delta = (price_up - price_down) / (2 * bump_size * S0)
        logger.debug(f"DeltaHedger: delta={delta:.6f}, K={K}, T={T}")
        return delta


class GammaHedger(DeltaHedger):
    """Computes delta and gamma hedge ratios.

    Extends DeltaHedger with a second-order finite difference for gamma.
    """

    def compute_gamma(
        self,
        paths: np.ndarray,
        K: float,
        T: float,
        r: float,
        bump_size: float = 0.01,
        option_type: str = "call",
    ) -> float:
        """Compute the option gamma.

        Args:
            paths: Array of shape (n_paths, n_steps+1) with price paths.
            K: Strike price.
            T: Time to maturity in years.
            r: Risk-free rate.
            bump_size: Relative bump size for finite differences.
            option_type: 'call' or 'put'.

        Returns:
            Gamma (d²V/dS²).
        """
        terminal = paths[:, -1]
        S0 = float(paths[0, 0])

        up_terminal = terminal * (1 + bump_size)
        down_terminal = terminal * (1 - bump_size)

        if option_type == "call":
            payoff_base = np.maximum(terminal - K, 0.0)
            payoff_up = np.maximum(up_terminal - K, 0.0)
            payoff_down = np.maximum(down_terminal - K, 0.0)
        else:
            payoff_base = np.maximum(K - terminal, 0.0)
            payoff_up = np.maximum(K - up_terminal, 0.0)
            payoff_down = np.maximum(K - down_terminal, 0.0)

        disc = np.exp(-r * T)
        price_base = float(disc * np.mean(payoff_base))
        price_up = float(disc * np.mean(payoff_up))
        price_down = float(disc * np.mean(payoff_down))

        gamma = (price_up - 2 * price_base + price_down) / (bump_size * S0) ** 2
        logger.debug(f"GammaHedger: gamma={gamma:.8f}, K={K}, T={T}")
        return gamma
