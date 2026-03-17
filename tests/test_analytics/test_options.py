"""Tests for Monte Carlo option pricing."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.analytics.options import MCOptionPricer
from synthquant.simulation.engines.gbm import GBMEngine


@pytest.fixture()
def gbm_paths() -> np.ndarray:
    """5000 GBM paths of 252 steps, S0=100."""
    engine = GBMEngine()
    return engine.simulate(
        n_paths=5000, n_steps=252, dt=1 / 252,
        S0=100.0, mu=0.05, sigma=0.20, random_state=42,
    )


@pytest.fixture()
def pricer() -> MCOptionPricer:
    return MCOptionPricer()


# ── European Options ──────────────────────────────────────────────────────────

def test_european_call_non_negative(pricer: MCOptionPricer, gbm_paths: np.ndarray) -> None:
    """European call price is non-negative."""
    price = pricer.price_european(gbm_paths, K=100.0, T=1.0, r=0.02, option_type="call")
    assert price >= 0.0


def test_european_put_non_negative(pricer: MCOptionPricer, gbm_paths: np.ndarray) -> None:
    """European put price is non-negative."""
    price = pricer.price_european(gbm_paths, K=100.0, T=1.0, r=0.02, option_type="put")
    assert price >= 0.0


def test_european_call_put_parity(pricer: MCOptionPricer, gbm_paths: np.ndarray) -> None:
    """Put-call parity: C - P ≈ S0*exp(mu*T) - K*exp(-r*T) for risk-neutral paths."""
    K = 100.0
    T = 1.0
    r = 0.0  # zero rate simplifies comparison
    call = pricer.price_european(gbm_paths, K=K, T=T, r=r, option_type="call")
    put = pricer.price_european(gbm_paths, K=K, T=T, r=r, option_type="put")
    # call - put ≈ E[S_T] - K (for r=0)
    expected_diff = np.mean(gbm_paths[:, -1]) - K
    np.testing.assert_allclose(call - put, expected_diff, rtol=0.02)


def test_european_call_increases_with_spot_price(pricer: MCOptionPricer) -> None:
    """Higher initial price leads to a more expensive call (fixed strike)."""
    engine = GBMEngine()
    paths_low = engine.simulate(3000, 252, 1 / 252, S0=90.0, random_state=0)
    paths_high = engine.simulate(3000, 252, 1 / 252, S0=110.0, random_state=0)
    call_low = pricer.price_european(paths_low, K=100.0, T=1.0, r=0.0)
    call_high = pricer.price_european(paths_high, K=100.0, T=1.0, r=0.0)
    assert call_high > call_low


def test_european_deep_itm_call_near_intrinsic(pricer: MCOptionPricer) -> None:
    """Deep-in-the-money call ≈ S0 - K * exp(-r*T)."""
    engine = GBMEngine()
    S0 = 100.0
    K = 10.0  # very low strike
    T = 1.0
    r = 0.05
    paths = engine.simulate(5000, 252, 1 / 252, S0=S0, mu=0.05, sigma=0.2, random_state=7)
    call = pricer.price_european(paths, K=K, T=T, r=r)
    intrinsic = np.exp(-r * T) * np.mean(paths[:, -1]) - K * np.exp(-r * T)
    assert abs(call - intrinsic) / intrinsic < 0.05


def test_european_invalid_option_type_raises(pricer: MCOptionPricer, gbm_paths: np.ndarray) -> None:
    """Passing unknown option_type raises ValueError."""
    with pytest.raises(ValueError, match="option_type"):
        pricer.price_european(gbm_paths, K=100.0, T=1.0, r=0.0, option_type="straddle")


# ── Asian Options ─────────────────────────────────────────────────────────────

def test_asian_call_non_negative(pricer: MCOptionPricer, gbm_paths: np.ndarray) -> None:
    """Asian call price is non-negative."""
    price = pricer.price_asian(gbm_paths, K=100.0, T=1.0, r=0.02)
    assert price >= 0.0


def test_asian_put_non_negative(pricer: MCOptionPricer, gbm_paths: np.ndarray) -> None:
    """Asian put price is non-negative."""
    price = pricer.price_asian(gbm_paths, K=100.0, T=1.0, r=0.02, option_type="put")
    assert price >= 0.0


def test_asian_call_cheaper_than_european(pricer: MCOptionPricer, gbm_paths: np.ndarray) -> None:
    """Asian call is cheaper than European call (averaging reduces variance)."""
    asian = pricer.price_asian(gbm_paths, K=100.0, T=1.0, r=0.0)
    european = pricer.price_european(gbm_paths, K=100.0, T=1.0, r=0.0)
    assert asian <= european + 1.0  # allow small MC noise


def test_asian_invalid_option_type_raises(pricer: MCOptionPricer, gbm_paths: np.ndarray) -> None:
    """Passing unknown option_type raises ValueError."""
    with pytest.raises(ValueError, match="option_type"):
        pricer.price_asian(gbm_paths, K=100.0, T=1.0, r=0.0, option_type="exotic")


# ── Barrier Options ───────────────────────────────────────────────────────────

def test_barrier_knockout_non_negative(pricer: MCOptionPricer, gbm_paths: np.ndarray) -> None:
    """Barrier knock-out option price is non-negative."""
    price = pricer.price_barrier(
        gbm_paths, K=100.0, B=120.0, T=1.0, r=0.0, barrier_type="knock_out"
    )
    assert price >= 0.0


def test_barrier_knockin_non_negative(pricer: MCOptionPricer, gbm_paths: np.ndarray) -> None:
    """Barrier knock-in option price is non-negative."""
    price = pricer.price_barrier(
        gbm_paths, K=100.0, B=120.0, T=1.0, r=0.0, barrier_type="knock_in"
    )
    assert price >= 0.0


def test_barrier_knockout_cheaper_than_vanilla(
    pricer: MCOptionPricer, gbm_paths: np.ndarray
) -> None:
    """Knock-out call is cheaper than vanilla European call."""
    vanilla = pricer.price_european(gbm_paths, K=100.0, T=1.0, r=0.0)
    knockout = pricer.price_barrier(
        gbm_paths, K=100.0, B=130.0, T=1.0, r=0.0, barrier_type="knock_out"
    )
    assert knockout <= vanilla + 0.5  # small MC tolerance
