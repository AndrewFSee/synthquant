"""Tests for regime-aware trading signal generation."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.strategy.signals import RegimeSignalGenerator


@pytest.fixture()
def regimes() -> np.ndarray:
    rng = np.random.default_rng(42)
    labels = np.zeros(200, dtype=int)
    regime = 0
    for t in range(200):
        if regime == 0 and rng.random() < 0.05:
            regime = 1
        elif regime == 1 and rng.random() < 0.10:
            regime = 0
        labels[t] = regime
    return labels


# ── Default Logic (no map, no forecasts) ──────────────────────────────────────

def test_default_regime_0_maps_to_long(regimes: np.ndarray) -> None:
    """Default: regime 0 generates long (+1) signals."""
    gen = RegimeSignalGenerator()
    result = gen.generate(regimes)
    assert np.all(result["signal"][regimes == 0] == 1)


def test_default_non_zero_regime_maps_to_flat(regimes: np.ndarray) -> None:
    """Default: non-zero regimes generate flat (0) signals."""
    gen = RegimeSignalGenerator()
    result = gen.generate(regimes)
    assert np.all(result["signal"][regimes != 0] == 0)


def test_output_has_required_keys(regimes: np.ndarray) -> None:
    """generate() returns dict with 'signal', 'entry', 'exit' keys."""
    gen = RegimeSignalGenerator()
    result = gen.generate(regimes)
    assert "signal" in result
    assert "entry" in result
    assert "exit" in result


def test_signal_values_in_valid_set(regimes: np.ndarray) -> None:
    """signal values are always in {-1, 0, +1}."""
    gen = RegimeSignalGenerator()
    result = gen.generate(regimes)
    assert set(np.unique(result["signal"])).issubset({-1, 0, 1})


def test_output_lengths_match_input(regimes: np.ndarray) -> None:
    """All output arrays have the same length as the input."""
    gen = RegimeSignalGenerator()
    result = gen.generate(regimes)
    n = len(regimes)
    assert len(result["signal"]) == n
    assert len(result["entry"]) == n
    assert len(result["exit"]) == n


# ── Regime Signal Map ─────────────────────────────────────────────────────────

def test_regime_signal_map_respected(regimes: np.ndarray) -> None:
    """Regime signal map overrides default logic."""
    signal_map = {0: 1, 1: -1}
    gen = RegimeSignalGenerator(regime_signal_map=signal_map)
    result = gen.generate(regimes)
    assert np.all(result["signal"][regimes == 0] == 1)
    assert np.all(result["signal"][regimes == 1] == -1)


def test_regime_signal_map_missing_key_defaults_to_flat() -> None:
    """Regime not in map generates 0 signal."""
    regimes = np.array([0, 1, 2, 0, 2])
    gen = RegimeSignalGenerator(regime_signal_map={0: 1})  # 1 and 2 not mapped
    result = gen.generate(regimes)
    assert result["signal"][2] == 0  # regime 2 → flat


# ── Forecast-Based Signals ────────────────────────────────────────────────────

def test_forecast_positive_generates_long() -> None:
    """Positive forecast generates long (+1) signal."""
    rng = np.random.default_rng(1)
    regimes = np.zeros(100, dtype=int)
    forecasts = np.abs(rng.normal(0.01, 0.001, 100))  # all positive
    gen = RegimeSignalGenerator()
    result = gen.generate(regimes, forecasts=forecasts)
    assert np.all(result["signal"] == 1)


def test_forecast_negative_generates_short() -> None:
    """Negative forecast generates short (-1) signal."""
    rng = np.random.default_rng(2)
    regimes = np.zeros(100, dtype=int)
    forecasts = -np.abs(rng.normal(0.01, 0.001, 100))  # all negative
    gen = RegimeSignalGenerator()
    result = gen.generate(regimes, forecasts=forecasts)
    assert np.all(result["signal"] == -1)


def test_forecast_below_threshold_generates_flat() -> None:
    """Forecast below entry threshold generates flat (0) signal."""
    regimes = np.zeros(10, dtype=int)
    forecasts = np.full(10, 0.001)  # small positive value
    gen = RegimeSignalGenerator()
    result = gen.generate(regimes, forecasts=forecasts, thresholds={"entry": 0.01})
    assert np.all(result["signal"] == 0)


# ── Entry / Exit Logic ────────────────────────────────────────────────────────

def test_entry_signal_only_on_new_position(regimes: np.ndarray) -> None:
    """entry is True only when transitioning from flat (0) to non-zero."""
    gen = RegimeSignalGenerator()
    result = gen.generate(regimes)
    signal = result["signal"]
    entry = result["entry"]
    # Entry at t means signal[t] != 0 and signal[t-1] == 0
    for t in range(1, len(signal)):
        if entry[t]:
            assert signal[t] != 0
            assert signal[t - 1] == 0


def test_exit_signal_only_on_closing_position(regimes: np.ndarray) -> None:
    """exit is True only when transitioning from non-zero to flat (0)."""
    gen = RegimeSignalGenerator()
    result = gen.generate(regimes)
    signal = result["signal"]
    exits = result["exit"]
    for t in range(1, len(signal)):
        if exits[t]:
            assert signal[t] == 0
            assert signal[t - 1] != 0
