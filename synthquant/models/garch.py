"""GARCH volatility model wrapping the arch library."""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["GARCHModel"]

VolModel = Literal["GARCH", "EGARCH", "GJRGARCH"]

# Maps internal names to arch library vol names and o (GJR) parameter
_VOL_MAP: dict[str, dict[str, Any]] = {
    "GARCH": {"vol": "GARCH", "o": 0},
    "EGARCH": {"vol": "EGARCH", "o": 0},
    "GJRGARCH": {"vol": "GARCH", "o": 1},  # GJR-GARCH uses o=1 in arch
}


class GARCHModel:
    """GARCH family volatility model.

    Wraps the ``arch`` library to provide a consistent interface for
    fitting, forecasting, and simulating GARCH-type models.

    Args:
        vol_model: Volatility specification: 'GARCH', 'EGARCH', or 'GJRGARCH'.
        p: GARCH lag order.
        q: ARCH lag order.
        dist: Innovation distribution ('normal', 't', 'skewt').
    """

    def __init__(
        self,
        vol_model: VolModel = "GARCH",
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
    ) -> None:
        if vol_model not in _VOL_MAP:
            raise ValueError(
                f"Unknown vol_model '{vol_model}'. Choose from: {list(_VOL_MAP.keys())}"
            )
        self.vol_model = vol_model
        self.p = p
        self.q = q
        self.dist = dist
        self._result: Any = None

    def fit(self, returns: np.ndarray | pd.Series, scale: float = 100.0) -> GARCHModel:
        """Fit the GARCH model to a return series.

        Args:
            returns: 1-D array or Series of log returns.
            scale: Scaling factor (arch library works better with scaled returns).

        Returns:
            self (fitted model).

        Raises:
            ImportError: If the arch library is not installed.
        """
        try:
            from arch import arch_model
        except ImportError as e:
            raise ImportError(
                "arch library is required. Install with: pip install synthquant"
            ) from e

        r = np.asarray(returns, dtype=float) * scale
        spec = _VOL_MAP[self.vol_model]
        model = arch_model(
            r,
            vol=str(spec["vol"]),  # type: ignore[arg-type]
            p=self.p,
            o=int(spec["o"]),
            q=self.q,
            dist=self.dist,  # type: ignore[arg-type]
        )
        self._result = model.fit(disp="off")
        self._scale = scale
        logger.info(
            f"{self.vol_model}({self.p},{self.q}) fitted, "
            f"AIC={self._result.aic:.4f}, BIC={self._result.bic:.4f}"
        )
        return self

    def forecast(self, horizon: int = 21) -> np.ndarray:
        """Forecast conditional volatility over a horizon.

        Args:
            horizon: Number of periods to forecast.

        Returns:
            Array of annualised volatility forecasts, shape (horizon,).

        Raises:
            RuntimeError: If model has not been fitted.
        """
        self._check_fitted()
        # EGARCH doesn't support analytic multi-step forecasts; use simulation
        if self.vol_model == "EGARCH":
            fc = self._result.forecast(
                horizon=horizon, method="simulation", simulations=500, reindex=False
            )
        else:
            fc = self._result.forecast(horizon=horizon, reindex=False)
        # variance forecasts, un-scale and convert to annualised vol
        var_forecasts = fc.variance.values[-1] / (self._scale**2)
        vol_forecasts = np.sqrt(var_forecasts * 252)
        logger.debug(f"Forecasted {horizon} periods of conditional volatility")
        return vol_forecasts

    def simulate(
        self,
        n_paths: int,
        horizon: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Simulate return paths from the fitted model.

        Args:
            n_paths: Number of simulation paths.
            horizon: Number of time steps per path.
            random_state: Random seed.

        Returns:
            Array of shape (n_paths, horizon) with simulated log returns.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        self._check_fitted()
        paths = np.empty((n_paths, horizon), dtype=float)
        for i in range(n_paths):
            sim = self._result.model.simulate(
                self._result.params,
                nobs=horizon,
            )
            paths[i] = sim["data"].values / self._scale
        logger.debug(f"Simulated {n_paths} GARCH paths of length {horizon}")
        return paths

    def _check_fitted(self) -> None:
        if self._result is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
