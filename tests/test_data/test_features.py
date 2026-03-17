"""Tests for financial feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthquant.data.features import FeatureEngine


@pytest.fixture()
def engine() -> FeatureEngine:
    return FeatureEngine()


@pytest.fixture()
def ohlcv() -> pd.DataFrame:
    """Synthetic 300-row OHLCV DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(42)
    n = 300
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


# ── Rolling Returns ───────────────────────────────────────────────────────────

def test_rolling_returns_length(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """rolling_returns() output has same length as input."""
    result = engine.rolling_returns(ohlcv["close"])
    assert len(result) == len(ohlcv)


def test_rolling_returns_first_value_nan(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """First value of rolling_returns(window=1) is NaN."""
    result = engine.rolling_returns(ohlcv["close"], window=1)
    assert np.isnan(result.iloc[0])


def test_rolling_returns_window_5(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """rolling_returns(window=5) has first 5 values as NaN."""
    result = engine.rolling_returns(ohlcv["close"], window=5)
    assert np.all(np.isnan(result.iloc[:5]))


# ── Realized Volatility ───────────────────────────────────────────────────────

def test_realized_vol_close_to_close(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """close_to_close realized vol returns a Series of correct length."""
    rv = engine.realized_volatility(ohlcv, window=21, estimator="close_to_close")
    assert isinstance(rv, pd.Series)
    assert len(rv) == len(ohlcv)


def test_realized_vol_non_negative(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """Realized volatility values are non-negative where not NaN."""
    rv = engine.realized_volatility(ohlcv, window=21, estimator="close_to_close")
    valid = rv.dropna()
    assert np.all(valid >= 0)


def test_realized_vol_parkinson(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """Parkinson estimator returns valid non-negative values."""
    rv = engine.realized_volatility(ohlcv, window=21, estimator="parkinson")
    valid = rv.dropna()
    assert np.all(valid >= 0)


def test_realized_vol_garman_klass(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """Garman-Klass estimator returns a Series without errors."""
    rv = engine.realized_volatility(ohlcv, window=21, estimator="garman_klass")
    assert isinstance(rv, pd.Series)


def test_realized_vol_yang_zhang(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """Yang-Zhang estimator returns a Series without errors."""
    rv = engine.realized_volatility(ohlcv, window=30, estimator="yang_zhang")
    assert isinstance(rv, pd.Series)


def test_realized_vol_unknown_estimator_raises(
    engine: FeatureEngine, ohlcv: pd.DataFrame
) -> None:
    """Unknown estimator raises ValueError."""
    with pytest.raises(ValueError, match="estimator"):
        engine.realized_volatility(ohlcv, estimator="unknown")


def test_realized_vol_annualized_larger_than_non_annualized(
    engine: FeatureEngine, ohlcv: pd.DataFrame
) -> None:
    """Annualized vol is larger than non-annualized vol."""
    rv_ann = engine.realized_volatility(ohlcv, window=21, annualize=True)
    rv_raw = engine.realized_volatility(ohlcv, window=21, annualize=False)
    valid_ann = rv_ann.dropna()
    valid_raw = rv_raw.dropna()
    assert valid_ann.mean() > valid_raw.mean()
