"""Tests for Merton Jump-Diffusion simulation engine."""

from __future__ import annotations

import numpy as np

from synthquant.simulation.engines.merton_jd import MertonJDEngine


def test_simulate_returns_correct_shape() -> None:
    """simulate() returns array of shape (n_paths, n_steps+1)."""
    engine = MertonJDEngine()
    paths = engine.simulate(n_paths=100, n_steps=50, dt=1 / 252)
    assert paths.shape == (100, 51)


def test_initial_price_is_correct() -> None:
    """All paths start at S0."""
    engine = MertonJDEngine()
    S0 = 75.0
    paths = engine.simulate(n_paths=300, n_steps=10, dt=1 / 252, S0=S0)
    np.testing.assert_allclose(paths[:, 0], S0)


def test_all_prices_positive() -> None:
    """Merton JD paths are strictly positive."""
    engine = MertonJDEngine()
    paths = engine.simulate(n_paths=500, n_steps=100, dt=1 / 252, random_state=42)
    assert np.all(paths > 0)


def test_reproducibility_with_random_state() -> None:
    """Same random_state produces identical paths."""
    engine = MertonJDEngine()
    paths1 = engine.simulate(50, 30, 1 / 252, random_state=13)
    paths2 = engine.simulate(50, 30, 1 / 252, random_state=13)
    np.testing.assert_array_equal(paths1, paths2)


def test_different_seeds_produce_different_paths() -> None:
    """Different seeds produce different paths."""
    engine = MertonJDEngine()
    paths1 = engine.simulate(20, 20, 1 / 252, random_state=1)
    paths2 = engine.simulate(20, 20, 1 / 252, random_state=2)
    assert not np.array_equal(paths1, paths2)


def test_higher_jump_intensity_increases_dispersion() -> None:
    """Higher jump intensity leads to greater dispersion of terminal prices."""
    engine = MertonJDEngine()
    paths_no_jump = engine.simulate(
        2000, 21, 1 / 252, lambda_j=0.0, random_state=0
    )
    paths_many_jumps = engine.simulate(
        2000, 21, 1 / 252, lambda_j=10.0, random_state=0
    )
    std_no_jump = np.std(np.log(paths_no_jump[:, -1] / paths_no_jump[:, 0]))
    std_many_jumps = np.std(np.log(paths_many_jumps[:, -1] / paths_many_jumps[:, 0]))
    assert std_many_jumps > std_no_jump


def test_zero_jump_intensity_resembles_gbm() -> None:
    """With lambda_j=0, Merton JD reduces to GBM (same log return distribution)."""
    from synthquant.simulation.engines.gbm import GBMEngine

    n_paths = 5000
    n_steps = 21
    dt = 1 / 252
    mu = 0.05
    sigma = 0.20
    seed = 42

    gbm_engine = GBMEngine()
    merton_engine = MertonJDEngine()

    # Use the same seed so random normals match
    gbm_paths = gbm_engine.simulate(n_paths, n_steps, dt, mu=mu, sigma=sigma, random_state=seed)
    merton_paths = merton_engine.simulate(
        n_paths, n_steps, dt, mu=mu, sigma=sigma, lambda_j=0.0, random_state=seed
    )

    gbm_log_ret = np.log(gbm_paths[:, -1] / gbm_paths[:, 0])
    merton_log_ret = np.log(merton_paths[:, -1] / merton_paths[:, 0])

    # Means and stds should be close (within 5%)
    np.testing.assert_allclose(np.mean(gbm_log_ret), np.mean(merton_log_ret), rtol=0.05)
    np.testing.assert_allclose(np.std(gbm_log_ret), np.std(merton_log_ret), rtol=0.05)


def test_negative_mean_jump_lowers_mean_terminal_price() -> None:
    """Negative mean jump size pulls terminal prices lower on average."""
    engine = MertonJDEngine()
    n_paths = 3000
    S0 = 100.0
    base = engine.simulate(n_paths, 21, 1 / 252, S0=S0, lambda_j=0.0, random_state=1)
    with_jumps = engine.simulate(
        n_paths, 21, 1 / 252, S0=S0, lambda_j=5.0, mu_j=-0.10, random_state=1
    )
    assert np.mean(with_jumps[:, -1]) < np.mean(base[:, -1])
