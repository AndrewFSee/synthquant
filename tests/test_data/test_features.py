"""Tests for FeatureEngine (technical and statistical features)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthquant.data.features import FeatureEngine


@pytest.fixture()
def engine() -> FeatureEngine:
    return FeatureEngine()


class TestRollingReturns:
    def test_same_length(self, engine: FeatureEngine, sample_ohlcv: pd.DataFrame) -> None:
        result = engine.rolling_returns(sample_ohlcv["close"])
        assert len(result) == len(sample_ohlcv)

    def test_first_value_is_nan(self, engine: FeatureEngine, sample_ohlcv: pd.DataFrame) -> None:
        result = engine.rolling_returns(sample_ohlcv["close"], window=1)
        assert pd.isna(result.iloc[0])

    def test_returns_log_returns(self, engine: FeatureEngine, sample_ohlcv: pd.DataFrame) -> None:
        close = sample_ohlcv["close"]
        result = engine.rolling_returns(close, window=1)
        expected = np.log(close / close.shift(1))
        pd.testing.assert_series_equal(result, expected)


class TestRealizedVolatility:
    @pytest.mark.parametrize("estimator", [
        "close_to_close", "parkinson", "garman_klass", "yang_zhang"
    ])
    def test_vol_is_positive(
        self, engine: FeatureEngine, sample_ohlcv: pd.DataFrame, estimator: str
    ) -> None:
        result = engine.realized_volatility(sample_ohlcv, window=21, estimator=estimator)
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid > 0).all(), f"{estimator} returned non-positive vol"

    def test_unknown_estimator_raises(
        self, engine: FeatureEngine, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="estimator"):
            engine.realized_volatility(sample_ohlcv, estimator="bogus")

    def test_annualized_larger_than_not(
        self, engine: FeatureEngine, sample_ohlcv: pd.DataFrame
    ) -> None:
        ann = engine.realized_volatility(sample_ohlcv, window=21, annualize=True).dropna()
        raw = engine.realized_volatility(sample_ohlcv, window=21, annualize=False).dropna()
        # sqrt(252) > 1, so annualized must be larger
        assert (ann.values > raw.values).all()


class TestRSI:
    def test_rsi_range(self, engine: FeatureEngine, sample_ohlcv: pd.DataFrame) -> None:
        result = engine.rsi(sample_ohlcv["close"], period=14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_length(self, engine: FeatureEngine, sample_ohlcv: pd.DataFrame) -> None:
        result = engine.rsi(sample_ohlcv["close"], period=14)
        assert len(result) == len(sample_ohlcv)


class TestMACD:
    def test_macd_columns(self, engine: FeatureEngine, sample_ohlcv: pd.DataFrame) -> None:
        result = engine.macd(sample_ohlcv["close"])
        assert set(result.columns) >= {"macd", "signal", "histogram"}

    def test_histogram_is_macd_minus_signal(
        self, engine: FeatureEngine, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = engine.macd(sample_ohlcv["close"])
        pd.testing.assert_series_equal(
            result["histogram"], result["macd"] - result["signal"],
            check_names=False,
        )


class TestBollingerBands:
    def test_bollinger_columns(self, engine: FeatureEngine, sample_ohlcv: pd.DataFrame) -> None:
        result = engine.bollinger_bands(sample_ohlcv["close"])
        assert {"middle", "upper", "lower", "pct_b", "bandwidth"}.issubset(result.columns)

    def test_upper_above_lower(self, engine: FeatureEngine, sample_ohlcv: pd.DataFrame) -> None:
        result = engine.bollinger_bands(sample_ohlcv["close"])
        valid = result.dropna()
        assert (valid["upper"] > valid["lower"]).all()


class TestRollingSkewKurtosis:
    def test_rolling_skew_length(self, engine: FeatureEngine, sample_ohlcv: pd.DataFrame) -> None:
        result = engine.rolling_skew(sample_ohlcv["close"])
        assert len(result) == len(sample_ohlcv)

    def test_rolling_kurtosis_length(
        self, engine: FeatureEngine, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = engine.rolling_kurtosis(sample_ohlcv["close"])
        assert len(result) == len(sample_ohlcv)
