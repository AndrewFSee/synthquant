"""Tests for Heston, MertonJD, and RegimeSwitching simulation engines."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.simulation.engines.heston import HestonEngine
from synthquant.simulation.engines.merton_jd import MertonJDEngine
from synthquant.simulation.engines.regime_switching import RegimeSwitchingEngine

# ── HestonEngine ──────────────────────────────────────────────────────────────

class TestHestonEngine:
    def test_shape(self) -> None:
        engine = HestonEngine()
        paths = engine.simulate(n_paths=100, n_steps=50, dt=1 / 252)
        assert paths.shape == (100, 51)

    def test_initial_price(self) -> None:
        engine = HestonEngine()
        S0 = 150.0
        paths = engine.simulate(n_paths=200, n_steps=10, dt=1 / 252, S0=S0)
        np.testing.assert_allclose(paths[:, 0], S0)

    def test_all_prices_positive(self) -> None:
        engine = HestonEngine()
        paths = engine.simulate(
            n_paths=500, n_steps=100, dt=1 / 252,
            v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7,
            random_state=42,
        )
        assert np.all(paths > 0)

    def test_reproducibility(self) -> None:
        engine = HestonEngine()
        p1 = engine.simulate(100, 50, 1 / 252, random_state=5)
        p2 = engine.simulate(100, 50, 1 / 252, random_state=5)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self) -> None:
        engine = HestonEngine()
        p1 = engine.simulate(20, 20, 1 / 252, random_state=1)
        p2 = engine.simulate(20, 20, 1 / 252, random_state=2)
        assert not np.array_equal(p1, p2)

    def test_higher_vol_of_vol_increases_terminal_spread(self) -> None:
        """Higher sigma_v should produce wider spread in terminal prices."""
        rng_seed = 0
        engine = HestonEngine()
        low_vov = engine.simulate(
            3000, 252, 1 / 252, v0=0.04, sigma_v=0.1, random_state=rng_seed
        )
        high_vov = engine.simulate(
            3000, 252, 1 / 252, v0=0.04, sigma_v=1.0, random_state=rng_seed
        )
        assert np.std(high_vov[:, -1]) > np.std(low_vov[:, -1])


# ── MertonJDEngine ────────────────────────────────────────────────────────────

class TestMertonJDEngine:
    def test_shape(self) -> None:
        engine = MertonJDEngine()
        paths = engine.simulate(n_paths=100, n_steps=50, dt=1 / 252)
        assert paths.shape == (100, 51)

    def test_initial_price(self) -> None:
        engine = MertonJDEngine()
        S0 = 80.0
        paths = engine.simulate(200, 10, 1 / 252, S0=S0, random_state=0)
        np.testing.assert_allclose(paths[:, 0], S0)

    def test_all_prices_positive(self) -> None:
        engine = MertonJDEngine()
        paths = engine.simulate(500, 252, 1 / 252, random_state=1)
        assert np.all(paths > 0)

    def test_reproducibility(self) -> None:
        engine = MertonJDEngine()
        p1 = engine.simulate(100, 50, 1 / 252, random_state=7)
        p2 = engine.simulate(100, 50, 1 / 252, random_state=7)
        np.testing.assert_array_equal(p1, p2)

    def test_zero_jump_intensity_matches_gbm_distribution(self) -> None:
        """With lambda_j=0, Merton JD reduces to GBM (same distributional shape)."""
        engine = MertonJDEngine()
        paths = engine.simulate(
            2000, 252, 1 / 252, mu=0.05, sigma=0.20,
            lambda_j=0.0, mu_j=0.0, sigma_j=0.0, random_state=42,
        )
        # No jumps: log-returns should be approx Normal
        log_ret = np.log(paths[:, -1] / paths[:, 0])
        # Mean and std should be close to GBM theoretical values
        assert abs(np.mean(log_ret) - (0.05 - 0.5 * 0.20**2)) < 0.05

    def test_negative_jumps_depress_prices(self) -> None:
        """Frequent large negative jumps should give lower terminal prices on average."""
        engine = MertonJDEngine()
        base = engine.simulate(3000, 252, 1 / 252, mu=0.05, lambda_j=0.0, random_state=0)
        jumpy = engine.simulate(
            3000, 252, 1 / 252, mu=0.05, lambda_j=10.0, mu_j=-0.10, sigma_j=0.05,
            random_state=0,
        )
        assert np.mean(jumpy[:, -1]) < np.mean(base[:, -1])


# ── RegimeSwitchingEngine ──────────────────────────────────────────────────────

@pytest.fixture()
def two_regime_engine() -> RegimeSwitchingEngine:
    """Simple 2-regime GBM engine: bull (high mu) and bear (high vol)."""
    params = [
        {"mu": 0.15, "sigma": 0.10},   # bull
        {"mu": -0.05, "sigma": 0.40},  # bear
    ]
    transition = np.array([[0.95, 0.05], [0.10, 0.90]])
    return RegimeSwitchingEngine(params, transition, initial_regime=0)


class TestRegimeSwitchingEngine:
    def test_shape(self, two_regime_engine: RegimeSwitchingEngine) -> None:
        paths = two_regime_engine.simulate(100, 50, 1 / 252)
        assert paths.shape == (100, 51)

    def test_initial_price(self, two_regime_engine: RegimeSwitchingEngine) -> None:
        S0 = 120.0
        paths = two_regime_engine.simulate(200, 10, 1 / 252, S0=S0, random_state=0)
        np.testing.assert_allclose(paths[:, 0], S0)

    def test_all_prices_positive(self, two_regime_engine: RegimeSwitchingEngine) -> None:
        paths = two_regime_engine.simulate(300, 100, 1 / 252, random_state=3)
        assert np.all(paths > 0)

    def test_reproducibility(self, two_regime_engine: RegimeSwitchingEngine) -> None:
        p1 = two_regime_engine.simulate(50, 20, 1 / 252, random_state=42)
        p2 = two_regime_engine.simulate(50, 20, 1 / 252, random_state=42)
        np.testing.assert_array_equal(p1, p2)

    def test_stationary_initial_regime(self) -> None:
        """Engine with initial_regime=None samples from stationary distribution."""
        params = [{"mu": 0.10, "sigma": 0.15}, {"mu": -0.05, "sigma": 0.30}]
        transition = np.array([[0.90, 0.10], [0.20, 0.80]])
        engine = RegimeSwitchingEngine(params, transition, initial_regime=None)
        paths = engine.simulate(500, 50, 1 / 252, random_state=0)
        assert paths.shape == (500, 51)
        assert np.all(paths > 0)
