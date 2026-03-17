"""Tests for Monte Carlo option pricing and implied volatility surface."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.analytics.options import ImpliedVolSurface, MCOptionPricer
from synthquant.simulation.engines.gbm import GBMEngine


@pytest.fixture()
def atm_paths() -> np.ndarray:
    """5000-path GBM with S0=100, K=100 at-the-money setup."""
    engine = GBMEngine()
    return engine.simulate(
        n_paths=5_000, n_steps=252, dt=1 / 252,
        S0=100.0, mu=0.05, sigma=0.20, random_state=42,
    )


class TestMCOptionPricer:
    def test_european_call_positive(self, atm_paths: np.ndarray) -> None:
        """European call price is positive."""
        pricer = MCOptionPricer()
        price = pricer.price_european(atm_paths, K=100.0, T=1.0, r=0.05, option_type="call")
        assert price > 0

    def test_european_put_positive(self, atm_paths: np.ndarray) -> None:
        """European put price is positive."""
        pricer = MCOptionPricer()
        price = pricer.price_european(atm_paths, K=100.0, T=1.0, r=0.05, option_type="put")
        assert price > 0

    def test_european_invalid_type_raises(self, atm_paths: np.ndarray) -> None:
        """price_european raises ValueError for unknown option type."""
        pricer = MCOptionPricer()
        with pytest.raises(ValueError, match="option_type"):
            pricer.price_european(atm_paths, K=100.0, T=1.0, r=0.05, option_type="straddle")

    def test_put_call_parity_approximate(self, atm_paths: np.ndarray) -> None:
        """Put-call parity holds approximately: C - P ≈ S0 - K*exp(-rT)."""
        pricer = MCOptionPricer()
        K, T, r = 100.0, 1.0, 0.05
        call = pricer.price_european(atm_paths, K=K, T=T, r=r, option_type="call")
        put = pricer.price_european(atm_paths, K=K, T=T, r=r, option_type="put")
        S0 = float(atm_paths[0, 0])
        parity = S0 - K * np.exp(-r * T)
        assert abs((call - put) - parity) < 1.0, (
            f"Put-call parity violation: C-P={call-put:.3f}, S0-K*e^-rT={parity:.3f}"
        )

    def test_deep_itm_call_approximately_intrinsic(self, atm_paths: np.ndarray) -> None:
        """Deep ITM call price ≈ S0 - K*exp(-rT) (large positive payoff)."""
        pricer = MCOptionPricer()
        price = pricer.price_european(atm_paths, K=50.0, T=1.0, r=0.05, option_type="call")
        S0 = 100.0
        K = 50.0
        r = 0.05
        intrinsic = S0 - K * np.exp(-r)
        assert price > intrinsic * 0.8, (
            f"Deep ITM call {price:.2f} far below intrinsic {intrinsic:.2f}"
        )

    def test_deep_otm_call_near_zero(self, atm_paths: np.ndarray) -> None:
        """Deep OTM call price is near zero."""
        pricer = MCOptionPricer()
        price = pricer.price_european(atm_paths, K=500.0, T=1.0, r=0.05, option_type="call")
        assert price < 1.0, f"Deep OTM call {price:.4f} should be near zero"

    def test_asian_call_cheaper_than_european(self, atm_paths: np.ndarray) -> None:
        """Asian call <= European call (averaging reduces variance)."""
        pricer = MCOptionPricer()
        K, T, r = 100.0, 1.0, 0.05
        euro = pricer.price_european(atm_paths, K=K, T=T, r=r)
        asian = pricer.price_asian(atm_paths, K=K, T=T, r=r)
        assert asian <= euro + 0.5, (
            f"Asian call {asian:.3f} should not exceed European call {euro:.3f}"
        )

    def test_asian_invalid_type_raises(self, atm_paths: np.ndarray) -> None:
        """price_asian raises ValueError for unknown option type."""
        pricer = MCOptionPricer()
        with pytest.raises(ValueError, match="option_type"):
            pricer.price_asian(atm_paths, K=100.0, T=1.0, r=0.05, option_type="binary")

    def test_barrier_knock_out_cheaper_than_vanilla(self, atm_paths: np.ndarray) -> None:
        """Knock-out call <= vanilla call (barrier kills some paths)."""
        pricer = MCOptionPricer()
        K, T, r = 100.0, 1.0, 0.05
        vanilla = pricer.price_european(atm_paths, K=K, T=T, r=r)
        # Barrier just above S0 - many paths will knock out
        barrier = pricer.price_barrier(atm_paths, K=K, B=110.0, T=T, r=r, barrier_type="knock_out")
        assert barrier <= vanilla + 0.1

    def test_barrier_invalid_type_raises(self, atm_paths: np.ndarray) -> None:
        """price_barrier raises ValueError for unknown barrier type."""
        pricer = MCOptionPricer()
        with pytest.raises(ValueError, match="barrier_type"):
            pricer.price_barrier(atm_paths, K=100.0, B=110.0, T=1.0, r=0.05,
                                 barrier_type="knock_sideways")

    def test_greeks_delta_call_in_range(self, atm_paths: np.ndarray) -> None:
        """Delta for an ATM call is roughly in (0, 1)."""
        pricer = MCOptionPricer()
        greeks = pricer.compute_greeks(atm_paths, K=100.0, T=1.0, r=0.05, option_type="call")
        assert 0.0 < greeks["delta"] < 1.0, f"Delta out of range: {greeks['delta']:.4f}"

    def test_greeks_gamma_positive(self, atm_paths: np.ndarray) -> None:
        """Gamma for a call is non-negative."""
        pricer = MCOptionPricer()
        greeks = pricer.compute_greeks(atm_paths, K=100.0, T=1.0, r=0.05)
        assert greeks["gamma"] >= 0.0


class TestImpliedVolSurface:
    def test_iv_surface_shape(self, atm_paths: np.ndarray) -> None:
        """compute() returns array of shape (n_maturities, n_strikes)."""
        pricer = MCOptionPricer()
        S = 100.0
        strikes = np.array([90.0, 100.0, 110.0])
        maturities = np.array([0.5, 1.0])
        r = 0.05

        engine = GBMEngine()
        prices_mat = np.zeros((len(maturities), len(strikes)))
        for i, T in enumerate(maturities):
            n_steps = max(int(T * 252), 1)
            paths = engine.simulate(
                n_paths=5_000, n_steps=n_steps, dt=T / n_steps,
                S0=S, mu=0.0, sigma=0.20, random_state=0,
            )
            for j, K in enumerate(strikes):
                prices_mat[i, j] = pricer.price_european(paths, K=K, T=T, r=r)

        surf = ImpliedVolSurface()
        iv = surf.compute(S, strikes, maturities, prices_mat, r=r)
        assert iv.shape == (len(maturities), len(strikes))

    def test_iv_values_positive_where_valid(self, atm_paths: np.ndarray) -> None:
        """Non-NaN IV values are positive."""
        pricer = MCOptionPricer()
        S = 100.0
        strikes = np.array([95.0, 100.0, 105.0])
        maturities = np.array([1.0])
        r = 0.05

        engine = GBMEngine()
        paths = engine.simulate(
            n_paths=10_000, n_steps=252, dt=1 / 252,
            S0=S, mu=0.0, sigma=0.20, random_state=1,
        )
        prices_mat = np.array([[
            pricer.price_european(paths, K=K, T=1.0, r=r) for K in strikes
        ]])

        surf = ImpliedVolSurface()
        iv = surf.compute(S, strikes, maturities, prices_mat, r=r)
        valid = iv[~np.isnan(iv)]
        assert len(valid) > 0
        assert np.all(valid > 0)
