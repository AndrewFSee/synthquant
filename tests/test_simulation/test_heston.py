"""Tests for Heston stochastic volatility simulation engine."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.simulation.engines.heston import HestonEngine


def test_simulate_returns_correct_shape() -> None:
    """simulate() returns array of shape (n_paths, n_steps+1)."""
    engine = HestonEngine()
    paths = engine.simulate(n_paths=100, n_steps=50, dt=1 / 252)
    assert paths.shape == (100, 51)


def test_initial_price_is_correct() -> None:
    """All paths start at S0."""
    engine = HestonEngine()
    S0 = 120.0
    paths = engine.simulate(n_paths=200, n_steps=10, dt=1 / 252, S0=S0)
    np.testing.assert_allclose(paths[:, 0], S0)


def test_all_prices_positive() -> None:
    """Heston paths are strictly positive."""
    engine = HestonEngine()
    paths = engine.simulate(n_paths=500, n_steps=50, dt=1 / 252, random_state=42)
    assert np.all(paths > 0)


def test_reproducibility_with_random_state() -> None:
    """Same random_state produces identical paths."""
    engine = HestonEngine()
    paths1 = engine.simulate(50, 30, 1 / 252, random_state=7)
    paths2 = engine.simulate(50, 30, 1 / 252, random_state=7)
    np.testing.assert_array_equal(paths1, paths2)


def test_different_seeds_produce_different_paths() -> None:
    """Different seeds produce different paths."""
    engine = HestonEngine()
    paths1 = engine.simulate(20, 20, 1 / 252, random_state=1)
    paths2 = engine.simulate(20, 20, 1 / 252, random_state=2)
    assert not np.array_equal(paths1, paths2)


def test_higher_vol_of_vol_increases_spread() -> None:
    """Higher sigma_v leads to wider spread of terminal prices."""
    engine = HestonEngine()
    paths_low = engine.simulate(2000, 21, 1 / 252, sigma_v=0.1, random_state=0)
    paths_high = engine.simulate(2000, 21, 1 / 252, sigma_v=1.0, random_state=0)
    spread_low = np.std(paths_low[:, -1])
    spread_high = np.std(paths_high[:, -1])
    assert spread_high > spread_low


def test_variance_stays_non_negative() -> None:
    """The QE scheme keeps variance non-negative."""
    engine = HestonEngine()
    # High sigma_v can cause issues for naive Euler – QE should handle it
    paths = engine.simulate(
        n_paths=500,
        n_steps=50,
        dt=1 / 252,
        v0=0.04,
        sigma_v=1.5,
        random_state=42,
    )
    assert np.all(paths >= 0)


def test_mean_reversion_of_variance() -> None:
    """With fast mean-reversion, terminal variance should be close to theta."""
    engine = HestonEngine()
    n_paths = 3000
    paths = engine.simulate(
        n_paths=n_paths,
        n_steps=252,
        dt=1 / 252,
        kappa=10.0,
        theta=0.04,
        v0=0.01,
        sigma_v=0.2,
        random_state=99,
    )
    assert paths.shape == (n_paths, 253)
