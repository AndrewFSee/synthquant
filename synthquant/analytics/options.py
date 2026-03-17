"""Option pricing and implied volatility from Monte Carlo paths."""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats
from scipy.optimize import brentq

logger = logging.getLogger(__name__)

__all__ = ["MCOptionPricer", "ImpliedVolSurface"]


class MCOptionPricer:
    """Option pricing via Monte Carlo simulation paths.

    All pricing methods expect price paths as input, not return series.
    """

    def price_european(
        self,
        paths: np.ndarray,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
    ) -> float:
        """Price a European option.

        Args:
            paths: Array of shape (n_paths, n_steps+1) with price paths.
            K: Strike price.
            T: Time to maturity in years.
            r: Risk-free rate.
            option_type: 'call' or 'put'.

        Returns:
            Option price (discounted expected payoff).

        Raises:
            ValueError: If option_type is not 'call' or 'put'.
        """
        terminal = paths[:, -1]
        if option_type == "call":
            payoffs = np.maximum(terminal - K, 0.0)
        elif option_type == "put":
            payoffs = np.maximum(K - terminal, 0.0)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        price = float(np.exp(-r * T) * np.mean(payoffs))
        logger.debug(f"European {option_type} price: K={K}, T={T}, price={price:.4f}")
        return price

    def price_asian(
        self,
        paths: np.ndarray,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
    ) -> float:
        """Price an Asian (arithmetic average) option.

        Args:
            paths: Array of shape (n_paths, n_steps+1).
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            option_type: 'call' or 'put'.

        Returns:
            Asian option price.
        """
        avg_price = np.mean(paths, axis=1)
        if option_type == "call":
            payoffs = np.maximum(avg_price - K, 0.0)
        elif option_type == "put":
            payoffs = np.maximum(K - avg_price, 0.0)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        price = float(np.exp(-r * T) * np.mean(payoffs))
        logger.debug(f"Asian {option_type} price: K={K}, T={T}, price={price:.4f}")
        return price

    def price_barrier(
        self,
        paths: np.ndarray,
        K: float,
        B: float,
        T: float,
        r: float,
        barrier_type: str = "knock_out",
        option_type: str = "call",
    ) -> float:
        """Price a barrier option.

        Args:
            paths: Array of shape (n_paths, n_steps+1).
            K: Strike price.
            B: Barrier level.
            T: Time to maturity.
            r: Risk-free rate.
            barrier_type: 'knock_out' (option dies if barrier hit) or
                'knock_in' (option activates only if barrier hit).
            option_type: 'call' or 'put'.

        Returns:
            Barrier option price.
        """
        terminal = paths[:, -1]
        S0 = paths[:, 0]

        if B > float(np.mean(S0)):  # Up barrier
            barrier_hit = np.any(paths >= B, axis=1)
        else:  # Down barrier
            barrier_hit = np.any(paths <= B, axis=1)

        if barrier_type == "knock_out":
            active = ~barrier_hit
        elif barrier_type == "knock_in":
            active = barrier_hit
        else:
            raise ValueError(f"barrier_type must be 'knock_out' or 'knock_in'")

        if option_type == "call":
            payoffs = np.where(active, np.maximum(terminal - K, 0.0), 0.0)
        else:
            payoffs = np.where(active, np.maximum(K - terminal, 0.0), 0.0)

        price = float(np.exp(-r * T) * np.mean(payoffs))
        logger.debug(f"Barrier {barrier_type} {option_type}: K={K}, B={B}, price={price:.4f}")
        return price

    def compute_greeks(
        self,
        paths: np.ndarray,
        K: float,
        T: float,
        r: float,
        bump_size: float = 0.01,
        option_type: str = "call",
    ) -> dict[str, float]:
        """Compute option Greeks via finite differences.

        Args:
            paths: Base price paths, shape (n_paths, n_steps+1).
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            bump_size: Relative price bump for finite difference.
            option_type: 'call' or 'put'.

        Returns:
            Dict with 'delta', 'gamma', 'vega' (approximate).
        """
        S0 = float(paths[0, 0])
        base_price = self.price_european(paths, K, T, r, option_type)

        # Delta: bump S0 up and down
        up_paths = paths * (1 + bump_size)
        down_paths = paths * (1 - bump_size)
        price_up = self.price_european(up_paths, K, T, r, option_type)
        price_down = self.price_european(down_paths, K, T, r, option_type)

        delta = (price_up - price_down) / (2 * bump_size * S0)
        gamma = (price_up - 2 * base_price + price_down) / (bump_size * S0) ** 2

        # Approximate vega via vol perturbation of log-returns
        log_ret = np.diff(np.log(paths), axis=1)
        vol = float(np.std(log_ret) * np.sqrt(252))
        sigma_up = vol * (1 + bump_size)
        sigma_down = vol * (1 - bump_size)
        scale_up = sigma_up / vol if vol > 0 else 1.0
        scale_down = sigma_down / vol if vol > 0 else 1.0
        price_vol_up = self.price_european(
            paths[:, 0:1] * np.exp(np.cumsum(log_ret * scale_up, axis=1)),
            K, T, r, option_type,
        ) if vol > 0 else base_price
        price_vol_down = self.price_european(
            paths[:, 0:1] * np.exp(np.cumsum(log_ret * scale_down, axis=1)),
            K, T, r, option_type,
        ) if vol > 0 else base_price
        vega = (price_vol_up - price_vol_down) / (2 * bump_size * vol) if vol > 0 else 0.0

        return {"delta": delta, "gamma": gamma, "vega": vega}


def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if sigma <= 0 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2))


class ImpliedVolSurface:
    """Compute implied volatility surface from option prices.

    Uses Brent's method to invert the Black-Scholes formula.
    """

    def compute(
        self,
        S: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        option_prices: np.ndarray,
        r: float = 0.05,
        option_type: str = "call",
    ) -> np.ndarray:
        """Compute the implied vol surface.

        Args:
            S: Current spot price.
            strikes: Array of shape (n_strikes,) with strike prices.
            maturities: Array of shape (n_maturities,) with maturities in years.
            option_prices: Array of shape (n_maturities, n_strikes) with market prices.
            r: Risk-free rate.
            option_type: 'call' (put-call parity used for puts).

        Returns:
            Array of shape (n_maturities, n_strikes) with implied vols.
        """
        n_mat = len(maturities)
        n_strikes = len(strikes)
        iv_surface = np.full((n_mat, n_strikes), np.nan)

        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                mkt_price = option_prices[i, j]
                intrinsic = max(S - K * np.exp(-r * T), 0.0)
                if mkt_price <= intrinsic:
                    continue
                try:
                    iv = brentq(
                        lambda sigma: _bs_call(S, K, T, r, sigma) - mkt_price,
                        1e-6,
                        10.0,
                        xtol=1e-6,
                    )
                    iv_surface[i, j] = iv
                except ValueError:
                    pass

        logger.info(
            f"ImpliedVolSurface computed: {n_mat} maturities x {n_strikes} strikes"
        )
        return iv_surface
