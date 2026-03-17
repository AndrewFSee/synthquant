"""Feature engineering for financial time series."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["FeatureEngine"]


class FeatureEngine:
    """Computes technical and statistical features from OHLCV data.

    All methods accept pandas Series or DataFrames and return
    pandas Series or DataFrames with matching indices.
    """

    def rolling_returns(self, close: pd.Series, window: int = 1) -> pd.Series:
        """Compute rolling log returns.

        Args:
            close: Closing price series.
            window: Return window in periods.

        Returns:
            Log return series of the same length (NaN for first `window` rows).
        """
        returns = np.log(close / close.shift(window))
        logger.debug(f"Computed {window}-period log returns, shape={returns.shape}")
        return returns

    def realized_volatility(
        self,
        ohlcv: pd.DataFrame,
        window: int = 21,
        estimator: str = "close_to_close",
        annualize: bool = True,
        trading_days: int = 252,
    ) -> pd.Series:
        """Compute realized volatility using various estimators.

        Args:
            ohlcv: DataFrame with 'open', 'high', 'low', 'close' columns.
            window: Rolling window in periods.
            estimator: One of 'close_to_close', 'parkinson', 'garman_klass', 'yang_zhang'.
            annualize: Whether to annualize the result.
            trading_days: Annualization factor.

        Returns:
            Rolling realized volatility series.

        Raises:
            ValueError: If estimator is not recognized.
        """
        scale = np.sqrt(trading_days) if annualize else 1.0

        if estimator == "close_to_close":
            log_ret = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
            rv = log_ret.rolling(window).std() * scale
        elif estimator == "parkinson":
            rv = self._parkinson(ohlcv, window, scale)
        elif estimator == "garman_klass":
            rv = self._garman_klass(ohlcv, window, scale)
        elif estimator == "yang_zhang":
            rv = self._yang_zhang(ohlcv, window, scale)
        else:
            raise ValueError(
                f"Unknown estimator '{estimator}'. "
                "Choose from: 'close_to_close', 'parkinson', 'garman_klass', 'yang_zhang'"
            )

        logger.debug(f"Computed {estimator} realized vol, window={window}")
        return rv.rename("realized_vol")

    def _parkinson(self, df: pd.DataFrame, window: int, scale: float) -> pd.Series:
        """Parkinson high-low volatility estimator."""
        hl = np.log(df["high"] / df["low"]) ** 2
        rv = np.sqrt(hl.rolling(window).mean() / (4 * np.log(2))) * scale
        return rv

    def _garman_klass(self, df: pd.DataFrame, window: int, scale: float) -> pd.Series:
        """Garman-Klass volatility estimator."""
        hl2 = 0.5 * np.log(df["high"] / df["low"]) ** 2
        co2 = (2 * np.log(2) - 1) * np.log(df["close"] / df["open"]) ** 2
        rv = np.sqrt((hl2 - co2).rolling(window).mean()) * scale
        return rv

    def _yang_zhang(self, df: pd.DataFrame, window: int, scale: float) -> pd.Series:
        """Yang-Zhang volatility estimator (overnight + open-to-close + Rogers-Satchell)."""
        log_oc = np.log(df["open"] / df["close"].shift(1))
        log_co = np.log(df["close"] / df["open"])
        log_ho = np.log(df["high"] / df["open"])
        log_lo = np.log(df["low"] / df["open"])

        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        var_overnight = log_oc.rolling(window).var()
        var_open_close = log_co.rolling(window).var()
        rs = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(window).mean()

        rv = np.sqrt(var_overnight + k * var_open_close + (1 - k) * rs) * scale
        return rv

    def rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index.

        Args:
            close: Closing price series.
            period: RSI period.

        Returns:
            RSI series in [0, 100].
        """
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        logger.debug(f"Computed RSI({period})")
        return rsi.rename(f"rsi_{period}")

    def macd(
        self,
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """Compute MACD, signal line, and histogram.

        Args:
            close: Closing price series.
            fast: Fast EMA period.
            slow: Slow EMA period.
            signal: Signal EMA period.

        Returns:
            DataFrame with columns: 'macd', 'signal', 'histogram'.
        """
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        logger.debug(f"Computed MACD({fast},{slow},{signal})")
        return pd.DataFrame(
            {"macd": macd_line, "signal": signal_line, "histogram": histogram}
        )

    def bollinger_bands(
        self,
        close: pd.Series,
        window: int = 20,
        n_std: float = 2.0,
    ) -> pd.DataFrame:
        """Compute Bollinger Bands.

        Args:
            close: Closing price series.
            window: Rolling window for moving average.
            n_std: Number of standard deviations for bands.

        Returns:
            DataFrame with columns: 'middle', 'upper', 'lower', 'pct_b', 'bandwidth'.
        """
        middle = close.rolling(window).mean()
        std = close.rolling(window).std()
        upper = middle + n_std * std
        lower = middle - n_std * std
        bandwidth = (upper - lower) / middle
        pct_b = (close - lower) / (upper - lower)
        logger.debug(f"Computed Bollinger Bands(window={window}, n_std={n_std})")
        return pd.DataFrame(
            {
                "middle": middle,
                "upper": upper,
                "lower": lower,
                "pct_b": pct_b,
                "bandwidth": bandwidth,
            }
        )

    def rolling_skew(self, close: pd.Series, window: int = 60) -> pd.Series:
        """Compute rolling skewness of log returns.

        Args:
            close: Closing price series.
            window: Rolling window.

        Returns:
            Rolling skewness series.
        """
        log_ret = np.log(close / close.shift(1))
        skew = log_ret.rolling(window).skew()
        logger.debug(f"Computed rolling skew, window={window}")
        return skew.rename(f"skew_{window}")

    def rolling_kurtosis(self, close: pd.Series, window: int = 60) -> pd.Series:
        """Compute rolling excess kurtosis of log returns.

        Args:
            close: Closing price series.
            window: Rolling window.

        Returns:
            Rolling excess kurtosis series.
        """
        log_ret = np.log(close / close.shift(1))
        kurt = log_ret.rolling(window).kurt()
        logger.debug(f"Computed rolling kurtosis, window={window}")
        return kurt.rename(f"kurt_{window}")
