"""Shared pytest fixtures for all SynthQuant tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthquant.simulation.engines.gbm import GBMEngine


@pytest.fixture()
def sample_returns() -> np.ndarray:
    """252 normally-distributed daily log returns (annualised ~7% drift, 20% vol)."""
    rng = np.random.default_rng(42)
    return rng.normal(0.07 / 252, 0.20 / np.sqrt(252), 252)


@pytest.fixture()
def sample_ohlcv() -> pd.DataFrame:
    """Synthetic 252-row OHLCV DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(42)
    n = 252
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    spread = rng.uniform(0.002, 0.006, n)
    return pd.DataFrame(
        {
            "open": close * (1 - spread / 2),
            "high": close * (1 + spread),
            "low": close * (1 - spread),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )


@pytest.fixture()
def sample_paths() -> np.ndarray:
    """1000 GBM paths of 252 steps using default parameters."""
    engine = GBMEngine()
    return engine.simulate(
        n_paths=1_000,
        n_steps=252,
        dt=1 / 252,
        S0=100.0,
        mu=0.07,
        sigma=0.20,
        random_state=42,
    )


@pytest.fixture()
def sample_regime_labels() -> np.ndarray:
    """252 alternating regime labels (0 and 1) for testing."""
    rng = np.random.default_rng(42)
    labels = np.zeros(252, dtype=int)
    regime = 0
    for t in range(252):
        if regime == 0 and rng.random() < 0.05:
            regime = 1
        elif regime == 1 and rng.random() < 0.10:
            regime = 0
        labels[t] = regime
    return labels
