"""Regime-aware trading signal generation."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["RegimeSignalGenerator"]


class RegimeSignalGenerator:
    """Generates entry/exit trading signals conditioned on detected regime.

    Args:
        regime_signal_map: Optional pre-defined mapping from regime index
            to signal value (+1 long, -1 short, 0 flat).
    """

    def __init__(
        self,
        regime_signal_map: dict[int, int] | None = None,
    ) -> None:
        self.regime_signal_map = regime_signal_map or {}

    def generate(
        self,
        regimes: np.ndarray,
        forecasts: np.ndarray | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate trading signals from regime labels and optional forecasts.

        Signal logic:
          - If regime_signal_map is provided, map regimes to signals directly.
          - Otherwise, use forecast direction and threshold (if provided).
          - Default fallback: regime 0 = long (+1), others = flat (0).

        Args:
            regimes: 1-D integer array of regime labels, shape (n,).
            forecasts: Optional 1-D array of forward-return forecasts, shape (n,).
            thresholds: Optional dict with keys 'entry' and 'exit' (float).

        Returns:
            Dict with keys:
              - 'signal': array of {-1, 0, +1}
              - 'entry': boolean array (True where new long/short entry)
              - 'exit': boolean array (True where position should be closed)
        """
        n = len(regimes)
        thresholds = thresholds or {}
        entry_thresh = thresholds.get("entry", 0.0)

        signal = np.zeros(n, dtype=int)

        if self.regime_signal_map:
            for i, r in enumerate(regimes):
                signal[i] = self.regime_signal_map.get(int(r), 0)
        elif forecasts is not None:
            fc = np.asarray(forecasts, dtype=float)
            signal = np.where(fc > entry_thresh, 1, np.where(fc < -entry_thresh, -1, 0))  # type: ignore[assignment]
        else:
            signal = np.where(regimes == 0, 1, 0)  # type: ignore[assignment]

        entry = np.zeros(n, dtype=bool)
        exit_sig = np.zeros(n, dtype=bool)
        for i in range(1, n):
            entry[i] = signal[i] != 0 and signal[i - 1] == 0
            exit_sig[i] = signal[i] == 0 and signal[i - 1] != 0

        logger.debug(
            f"RegimeSignalGenerator: {np.sum(entry)} entries, "
            f"{np.sum(exit_sig)} exits over {n} periods"
        )
        return {"signal": signal, "entry": entry, "exit": exit_sig}
