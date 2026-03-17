"""Tests for Regime-Switching simulation engine."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.simulation.engines.regime_switching import RegimeSwitchingEngine


@pytest.fixture()
def two_regime_engine() -> RegimeSwitchingEngine:
    """A simple 2-regime engine: bull and bear."""
    regime_params = [
        {"mu": 0.10, "sigma": 0.15},  # regime 0: bull
        {"mu": -0.20, "sigma": 0.40},  # regime 1: bear
    ]
    transition_matrix = np.array([[0.95, 0.05], [0.10, 0.90]])
    return RegimeSwitchingEngine(
        regime_params=regime_params,
        transition_matrix=transition_matrix,
        initial_regime=0,
    )


def test_simulate_returns_correct_shape(two_regime_engine: RegimeSwitchingEngine) -> None:
    """simulate() returns array of shape (n_paths, n_steps+1)."""
    paths = two_regime_engine.simulate(n_paths=100, n_steps=50, dt=1 / 252)
    assert paths.shape == (100, 51)


def test_initial_price_is_correct(two_regime_engine: RegimeSwitchingEngine) -> None:
    """All paths start at S0."""
    S0 = 80.0
    paths = two_regime_engine.simulate(n_paths=200, n_steps=10, dt=1 / 252, S0=S0)
    np.testing.assert_allclose(paths[:, 0], S0)


def test_all_prices_positive(two_regime_engine: RegimeSwitchingEngine) -> None:
    """All regime-switching paths are strictly positive."""
    paths = two_regime_engine.simulate(n_paths=500, n_steps=50, dt=1 / 252, random_state=42)
    assert np.all(paths > 0)


def test_reproducibility_with_random_state(two_regime_engine: RegimeSwitchingEngine) -> None:
    """Same random_state produces identical paths."""
    paths1 = two_regime_engine.simulate(50, 30, 1 / 252, random_state=55)
    paths2 = two_regime_engine.simulate(50, 30, 1 / 252, random_state=55)
    np.testing.assert_array_equal(paths1, paths2)


def test_different_seeds_produce_different_paths(two_regime_engine: RegimeSwitchingEngine) -> None:
    """Different seeds produce different paths."""
    paths1 = two_regime_engine.simulate(20, 20, 1 / 252, random_state=3)
    paths2 = two_regime_engine.simulate(20, 20, 1 / 252, random_state=4)
    assert not np.array_equal(paths1, paths2)


def test_bear_regime_lowers_mean_returns() -> None:
    """Starting in bear regime (negative drift) lowers mean terminal price."""
    regime_params = [
        {"mu": 0.20, "sigma": 0.15},   # regime 0: strong bull
        {"mu": -0.40, "sigma": 0.40},  # regime 1: strong bear
    ]
    tm = np.array([[0.99, 0.01], [0.01, 0.99]])  # very sticky regimes
    n_paths = 2000
    bull_engine = RegimeSwitchingEngine(regime_params, tm, initial_regime=0)
    bear_engine = RegimeSwitchingEngine(regime_params, tm, initial_regime=1)

    bull_paths = bull_engine.simulate(n_paths, 252, 1 / 252, random_state=0)
    bear_paths = bear_engine.simulate(n_paths, 252, 1 / 252, random_state=0)

    assert np.mean(bull_paths[:, -1]) > np.mean(bear_paths[:, -1])


def test_stationary_distribution_initial_regime() -> None:
    """Without initial_regime, engine samples from stationary distribution."""
    regime_params = [
        {"mu": 0.05, "sigma": 0.15},
        {"mu": 0.05, "sigma": 0.15},
    ]
    tm = np.array([[0.90, 0.10], [0.10, 0.90]])
    engine = RegimeSwitchingEngine(regime_params, tm, initial_regime=None)
    paths = engine.simulate(n_paths=100, n_steps=10, dt=1 / 252, random_state=42)
    assert paths.shape == (100, 11)
