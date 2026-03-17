"""Tests for HestonModel stochastic volatility model."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.models.stochastic_vol import HestonModel


@pytest.fixture()
def heston() -> HestonModel:
    return HestonModel(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, mu=0.05)


class TestHestonModel:
    def test_simulate_price_shape(self, heston: HestonModel) -> None:
        S, V = heston.simulate(S0=100.0, T=1.0, n_paths=100, n_steps=50, random_state=0)
        assert S.shape == (100, 51)

    def test_simulate_var_shape(self, heston: HestonModel) -> None:
        S, V = heston.simulate(S0=100.0, T=1.0, n_paths=100, n_steps=50, random_state=0)
        assert V.shape == (100, 51)

    def test_initial_price(self, heston: HestonModel) -> None:
        S0 = 120.0
        S, _ = heston.simulate(S0=S0, T=1.0, n_paths=200, n_steps=20, random_state=1)
        np.testing.assert_allclose(S[:, 0], S0)

    def test_initial_variance(self, heston: HestonModel) -> None:
        v0 = 0.09
        model = HestonModel(v0=v0)
        _, V = model.simulate(S0=100.0, T=1.0, n_paths=100, n_steps=20, random_state=2)
        np.testing.assert_allclose(V[:, 0], v0)

    def test_all_prices_positive(self, heston: HestonModel) -> None:
        S, _ = heston.simulate(S0=100.0, T=1.0, n_paths=300, n_steps=50, random_state=3)
        assert np.all(S > 0)

    def test_variance_non_negative(self, heston: HestonModel) -> None:
        """Full truncation keeps variance non-negative."""
        _, V = heston.simulate(S0=100.0, T=1.0, n_paths=500, n_steps=252, random_state=4)
        assert np.all(V >= 0)

    def test_reproducibility(self, heston: HestonModel) -> None:
        S1, _ = heston.simulate(100.0, 1.0, 100, 50, random_state=10)
        S2, _ = heston.simulate(100.0, 1.0, 100, 50, random_state=10)
        np.testing.assert_array_equal(S1, S2)

    def test_mean_reversion_of_variance(self) -> None:
        """With strong kappa, variance mean-reverts to theta."""
        theta = 0.04
        model = HestonModel(v0=0.16, kappa=10.0, theta=theta, sigma_v=0.1)
        _, V = model.simulate(S0=100.0, T=5.0, n_paths=2000, n_steps=1260, random_state=0)
        terminal_var = np.mean(V[:, -1])
        assert abs(terminal_var - theta) < 0.01, (
            f"Terminal variance {terminal_var:.4f} far from theta={theta}"
        )
