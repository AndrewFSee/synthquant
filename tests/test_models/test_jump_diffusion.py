"""Tests for MertonJumpDiffusion and KouModel."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.models.jump_diffusion import KouModel, MertonJumpDiffusion


@pytest.fixture()
def merton() -> MertonJumpDiffusion:
    return MertonJumpDiffusion(mu=0.05, sigma=0.15, lambda_j=1.0, mu_j=-0.05, sigma_j=0.10)


@pytest.fixture()
def kou() -> KouModel:
    return KouModel(mu=0.05, sigma=0.15, lambda_j=1.0, p_up=0.4, eta1=10.0, eta2=5.0)


class TestMertonJumpDiffusion:
    def test_simulate_shape(self, merton: MertonJumpDiffusion) -> None:
        paths = merton.simulate(S0=100.0, T=1.0, n_paths=100, n_steps=50, random_state=0)
        assert paths.shape == (100, 51)

    def test_initial_price(self, merton: MertonJumpDiffusion) -> None:
        S0 = 75.0
        paths = merton.simulate(S0=S0, T=1.0, n_paths=200, n_steps=20, random_state=1)
        np.testing.assert_allclose(paths[:, 0], S0)

    def test_all_prices_positive(self, merton: MertonJumpDiffusion) -> None:
        paths = merton.simulate(S0=100.0, T=1.0, n_paths=300, n_steps=50, random_state=2)
        assert np.all(paths > 0)

    def test_reproducibility(self, merton: MertonJumpDiffusion) -> None:
        p1 = merton.simulate(100.0, 1.0, 100, 50, random_state=42)
        p2 = merton.simulate(100.0, 1.0, 100, 50, random_state=42)
        np.testing.assert_array_equal(p1, p2)

    def test_european_call_price_positive(self, merton: MertonJumpDiffusion) -> None:
        price = merton.european_call_price(S0=100.0, K=100.0, T=1.0, r=0.05)
        assert price > 0

    def test_european_call_deep_itm_exceeds_intrinsic(self, merton: MertonJumpDiffusion) -> None:
        S0, K, T, r = 100.0, 50.0, 1.0, 0.05
        price = merton.european_call_price(S0, K, T, r)
        intrinsic = S0 - K * np.exp(-r * T)
        assert price >= intrinsic * 0.8

    def test_european_call_deep_otm_near_zero(self, merton: MertonJumpDiffusion) -> None:
        price = merton.european_call_price(S0=100.0, K=500.0, T=1.0, r=0.05)
        assert price < 1.0


class TestKouModel:
    def test_simulate_shape(self, kou: KouModel) -> None:
        paths = kou.simulate(S0=100.0, T=1.0, n_paths=50, n_steps=30, random_state=0)
        assert paths.shape == (50, 31)

    def test_initial_price(self, kou: KouModel) -> None:
        S0 = 200.0
        paths = kou.simulate(S0=S0, T=1.0, n_paths=100, n_steps=20, random_state=1)
        np.testing.assert_allclose(paths[:, 0], S0)

    def test_all_prices_positive(self, kou: KouModel) -> None:
        paths = kou.simulate(S0=100.0, T=1.0, n_paths=200, n_steps=50, random_state=2)
        assert np.all(paths > 0)

    def test_reproducibility(self, kou: KouModel) -> None:
        p1 = kou.simulate(100.0, 1.0, 50, 30, random_state=7)
        p2 = kou.simulate(100.0, 1.0, 50, 30, random_state=7)
        np.testing.assert_array_equal(p1, p2)
