"""Tests for GBM simulation engine."""

from __future__ import annotations

import numpy as np

from synthquant.simulation.engines.gbm import GBMEngine


def test_simulate_returns_correct_shape() -> None:
    """simulate() returns array of shape (n_paths, n_steps+1)."""
    engine = GBMEngine()
    paths = engine.simulate(n_paths=100, n_steps=50, dt=1 / 252)
    assert paths.shape == (100, 51)


def test_initial_price_is_correct() -> None:
    """All paths start at S0."""
    engine = GBMEngine()
    S0 = 150.0
    paths = engine.simulate(n_paths=500, n_steps=10, dt=1 / 252, S0=S0)
    np.testing.assert_allclose(paths[:, 0], S0)


def test_all_paths_start_at_s0() -> None:
    """Every path's first value equals S0 (no exceptions)."""
    engine = GBMEngine()
    paths = engine.simulate(n_paths=200, n_steps=5, dt=0.01, S0=42.0)
    assert np.all(paths[:, 0] == 42.0)


def test_all_prices_positive() -> None:
    """GBM paths are strictly positive."""
    engine = GBMEngine()
    paths = engine.simulate(n_paths=1000, n_steps=252, dt=1 / 252, sigma=0.5)
    assert np.all(paths > 0)


def test_gbm_mean_convergence() -> None:
    """Terminal price mean converges to S0 * exp(mu * T) with large n_paths."""
    engine = GBMEngine()
    mu = 0.08
    T = 1.0
    S0 = 100.0
    n_steps = 252
    dt = T / n_steps
    # Use large n_paths for statistical accuracy
    paths = engine.simulate(
        n_paths=50_000, n_steps=n_steps, dt=dt, S0=S0, mu=mu, sigma=0.20,
        random_state=0
    )
    expected_mean = S0 * np.exp(mu * T)
    actual_mean = float(np.mean(paths[:, -1]))
    # Within 1% relative error for 50k paths
    assert abs(actual_mean - expected_mean) / expected_mean < 0.01, (
        f"Mean {actual_mean:.4f} not close to expected {expected_mean:.4f}"
    )


def test_gbm_log_return_variance() -> None:
    """Variance of log returns matches sigma^2 * dt * n_steps."""
    engine = GBMEngine()
    mu = 0.05
    sigma = 0.20
    dt = 1 / 252
    n_steps = 252
    paths = engine.simulate(
        n_paths=20_000, n_steps=n_steps, dt=dt, S0=100.0, mu=mu, sigma=sigma,
        random_state=1,
    )
    log_returns = np.log(paths[:, -1] / paths[:, 0])
    expected_var = sigma**2 * dt * n_steps
    actual_var = float(np.var(log_returns))
    # Within 5% relative error
    assert abs(actual_var - expected_var) / expected_var < 0.05, (
        f"Variance {actual_var:.6f} not close to expected {expected_var:.6f}"
    )


def test_reproducibility_with_random_state() -> None:
    """Same random_state produces identical paths."""
    engine = GBMEngine()
    paths1 = engine.simulate(100, 50, 1 / 252, random_state=99)
    paths2 = engine.simulate(100, 50, 1 / 252, random_state=99)
    np.testing.assert_array_equal(paths1, paths2)


def test_different_seeds_produce_different_paths() -> None:
    """Different seeds produce different paths."""
    engine = GBMEngine()
    paths1 = engine.simulate(10, 10, 1 / 252, random_state=1)
    paths2 = engine.simulate(10, 10, 1 / 252, random_state=2)
    assert not np.array_equal(paths1, paths2)
