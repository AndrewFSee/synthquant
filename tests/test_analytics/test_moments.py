"""Tests for rolling moment statistics and normality tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthquant.analytics.moments import (
    jarque_bera_test,
    moment_ratio_test,
    rolling_kurtosis,
    rolling_skewness,
)


@pytest.fixture()
def returns_series(sample_returns: np.ndarray) -> pd.Series:
    return pd.Series(sample_returns)


def test_rolling_skewness_length(returns_series: pd.Series) -> None:
    """rolling_skewness returns series of the same length."""
    result = rolling_skewness(returns_series, window=20)
    assert len(result) == len(returns_series)


def test_rolling_skewness_leading_nans(returns_series: pd.Series) -> None:
    """First window-1 values are NaN due to insufficient data."""
    window = 20
    result = rolling_skewness(returns_series, window=window)
    assert result.iloc[:window - 1].isna().all()


def test_rolling_skewness_accepts_ndarray(sample_returns: np.ndarray) -> None:
    """rolling_skewness accepts raw numpy arrays."""
    result = rolling_skewness(sample_returns, window=20)
    assert len(result) == len(sample_returns)


def test_rolling_kurtosis_length(returns_series: pd.Series) -> None:
    """rolling_kurtosis returns series of the same length."""
    result = rolling_kurtosis(returns_series, window=20)
    assert len(result) == len(returns_series)


def test_rolling_kurtosis_leading_nans(returns_series: pd.Series) -> None:
    """First window-1 values are NaN."""
    window = 20
    result = rolling_kurtosis(returns_series, window=window)
    assert result.iloc[:window - 1].isna().all()


def test_jarque_bera_keys(sample_returns: np.ndarray) -> None:
    """jarque_bera_test returns expected keys."""
    result = jarque_bera_test(sample_returns)
    assert {"statistic", "p_value", "skewness", "kurtosis"} == set(result.keys())


def test_jarque_bera_statistic_nonneg(sample_returns: np.ndarray) -> None:
    """Jarque-Bera statistic is non-negative."""
    result = jarque_bera_test(sample_returns)
    assert result["statistic"] >= 0


def test_jarque_bera_p_value_range(sample_returns: np.ndarray) -> None:
    """p_value is in [0, 1]."""
    result = jarque_bera_test(sample_returns)
    assert 0.0 <= result["p_value"] <= 1.0


def test_jarque_bera_normal_data_high_pvalue() -> None:
    """JB test should not reject normality for truly normal data (large sample)."""
    rng = np.random.default_rng(0)
    normal_data = rng.standard_normal(10_000)
    result = jarque_bera_test(normal_data)
    # With large normal sample, p-value should usually be > 0.01
    assert result["p_value"] > 0.01


def test_moment_ratio_test_keys(sample_returns: np.ndarray) -> None:
    """moment_ratio_test returns expected keys."""
    result = moment_ratio_test(sample_returns)
    assert {"mean", "std", "skewness", "excess_kurtosis", "cv"} == set(result.keys())


def test_moment_ratio_std_positive(sample_returns: np.ndarray) -> None:
    """Standard deviation must be positive for non-constant returns."""
    result = moment_ratio_test(sample_returns)
    assert result["std"] > 0


def test_moment_ratio_zero_mean_cv_is_inf() -> None:
    """cv is inf when mean is exactly zero (integer array with symmetric values)."""
    # Use a symmetric integer-valued array so mean is exactly 0
    data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = moment_ratio_test(data)
    assert result["cv"] == np.inf
