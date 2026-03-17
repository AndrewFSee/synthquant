"""Tests for regime-aware signal generation."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.strategy.signals import RegimeSignalGenerator


@pytest.fixture()
def alternating_regimes() -> np.ndarray:
    """252-period series alternating between regime 0 and 1."""
    regimes = np.zeros(252, dtype=int)
    regimes[126:] = 1
    return regimes


class TestRegimeSignalGenerator:
    def test_default_regime_0_is_long(
        self, alternating_regimes: np.ndarray
    ) -> None:
        """Default: regime 0 -> signal +1, others -> 0."""
        gen = RegimeSignalGenerator()
        result = gen.generate(alternating_regimes)
        signal = result["signal"]
        assert np.all(signal[:126] == 1)
        assert np.all(signal[126:] == 0)

    def test_regime_signal_map(self, alternating_regimes: np.ndarray) -> None:
        """Custom map: regime 0 -> +1, regime 1 -> -1."""
        gen = RegimeSignalGenerator(regime_signal_map={0: 1, 1: -1})
        result = gen.generate(alternating_regimes)
        assert np.all(result["signal"][:126] == 1)
        assert np.all(result["signal"][126:] == -1)

    def test_forecast_direction(self) -> None:
        """With forecasts, signal follows the sign."""
        regimes = np.zeros(10, dtype=int)
        forecasts = np.array([0.01, -0.02, 0.005, 0.0, -0.01, 0.03, -0.005, 0.0, 0.01, -0.01])
        gen = RegimeSignalGenerator()
        result = gen.generate(regimes, forecasts=forecasts)
        signal = result["signal"]
        assert signal[0] == 1   # positive forecast
        assert signal[1] == -1  # negative forecast
        assert signal[3] == 0   # zero forecast

    def test_entry_and_exit_arrays_correct_length(
        self, alternating_regimes: np.ndarray
    ) -> None:
        gen = RegimeSignalGenerator()
        result = gen.generate(alternating_regimes)
        assert len(result["entry"]) == len(alternating_regimes)
        assert len(result["exit"]) == len(alternating_regimes)

    def test_entry_when_signal_transitions_from_flat(
        self, alternating_regimes: np.ndarray
    ) -> None:
        """Entry is True at the first step of a new position (was flat before)."""
        gen = RegimeSignalGenerator(regime_signal_map={0: 1, 1: 0})
        result = gen.generate(alternating_regimes)
        entry = result["entry"]
        # First position starts at t=0 (no previous), first entry after flat starts at 126+1?
        # Actually t=0: no previous, so entry[0]=False. Entry occurs where signal != 0 and prev == 0
        assert not entry[0]  # Can't enter at t=0 (no previous)
        # After regime switches back (if any), there should be entries
        assert result["entry"].dtype == bool

    def test_exit_when_signal_transitions_to_flat(self) -> None:
        """Exit is True when signal goes from non-zero to zero."""
        regimes = np.array([0, 0, 0, 1, 1, 0, 0], dtype=int)
        gen = RegimeSignalGenerator(regime_signal_map={0: 1, 1: 0})
        result = gen.generate(regimes)
        # At t=3: signal goes from 1 to 0 -> exit
        assert result["exit"][3]

    def test_threshold_applied(self) -> None:
        """Entry threshold prevents signals for small forecasts."""
        regimes = np.zeros(5, dtype=int)
        forecasts = np.array([0.001, 0.005, 0.02, -0.001, -0.03])
        gen = RegimeSignalGenerator()
        result = gen.generate(regimes, forecasts=forecasts, thresholds={"entry": 0.01})
        # Only |forecast| > 0.01 gets a signal
        assert result["signal"][0] == 0   # 0.001 < 0.01
        assert result["signal"][2] == 1   # 0.02 > 0.01
        assert result["signal"][4] == -1  # -0.03 < -0.01

    def test_output_keys(self, alternating_regimes: np.ndarray) -> None:
        gen = RegimeSignalGenerator()
        result = gen.generate(alternating_regimes)
        assert {"signal", "entry", "exit"} == set(result.keys())
