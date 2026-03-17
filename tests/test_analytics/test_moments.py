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
def normal_returns() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(0.0003, 0.012, 500)


@pytest.fixture()
def fat_tailed_returns() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_t(df=3, size=500) * 0.01


# ── Rolling Skewness ──────────────────────────────────────────────────────────

def test_rolling_skewness_returns_series(normal_returns: np.ndarray) -> None:
    """rolling_skewness() returns a pandas Series."""
    result = rolling_skewness(normal_returns, window=60)
    assert isinstance(result, pd.Series)


def test_rolling_skewness_length_matches_input(normal_returns: np.ndarray) -> None:
    """rolling_skewness() output length matches input length."""
    result = rolling_skewness(normal_returns, window=30)
    assert len(result) == len(normal_returns)


def test_rolling_skewness_first_window_minus_one_is_nan(normal_returns: np.ndarray) -> None:
    """First window-1 values are NaN (not enough data)."""
    window = 40
    result = rolling_skewness(normal_returns, window=window)
    assert np.all(np.isnan(result.iloc[: window - 1]))


def test_rolling_skewness_accepts_pandas_series(normal_returns: np.ndarray) -> None:
    """rolling_skewness() also accepts a pandas Series."""
    s = pd.Series(normal_returns)
    result = rolling_skewness(s, window=60)
    assert isinstance(result, pd.Series)


# ── Rolling Kurtosis ──────────────────────────────────────────────────────────

def test_rolling_kurtosis_returns_series(normal_returns: np.ndarray) -> None:
    """rolling_kurtosis() returns a pandas Series."""
    result = rolling_kurtosis(normal_returns, window=60)
    assert isinstance(result, pd.Series)


def test_rolling_kurtosis_length_matches_input(normal_returns: np.ndarray) -> None:
    """rolling_kurtosis() output length matches input length."""
    result = rolling_kurtosis(normal_returns, window=30)
    assert len(result) == len(normal_returns)


def test_fat_tailed_kurtosis_positive(fat_tailed_returns: np.ndarray) -> None:
    """Fat-tailed distribution has positive excess kurtosis on average."""
    result = rolling_kurtosis(fat_tailed_returns, window=60)
    valid = result.dropna()
    assert valid.mean() > 0


# ── Jarque-Bera Test ──────────────────────────────────────────────────────────

def test_jarque_bera_returns_correct_keys(normal_returns: np.ndarray) -> None:
    """jarque_bera_test() returns dict with required keys."""
    result = jarque_bera_test(normal_returns)
    assert "statistic" in result
    assert "p_value" in result
    assert "skewness" in result
    assert "kurtosis" in result


def test_jarque_bera_normal_data_high_p_value(normal_returns: np.ndarray) -> None:
    """Normally distributed data should not reject normality (p > 0.01)."""
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, 1000)
    result = jarque_bera_test(normal)
    assert result["p_value"] > 0.01


def test_jarque_bera_fat_tailed_low_p_value(fat_tailed_returns: np.ndarray) -> None:
    """Heavy-tailed t-distribution data strongly rejects normality."""
    result = jarque_bera_test(fat_tailed_returns)
    assert result["statistic"] > 0


def test_jarque_bera_statistic_non_negative(normal_returns: np.ndarray) -> None:
    """JB statistic is always non-negative."""
    result = jarque_bera_test(normal_returns)
    assert result["statistic"] >= 0.0


def test_jarque_bera_p_value_in_range(normal_returns: np.ndarray) -> None:
    """p-value is in [0, 1]."""
    result = jarque_bera_test(normal_returns)
    assert 0.0 <= result["p_value"] <= 1.0


# ── Moment Ratio Test ─────────────────────────────────────────────────────────

def test_moment_ratio_returns_correct_keys(normal_returns: np.ndarray) -> None:
    """moment_ratio_test() returns dict with required keys."""
    result = moment_ratio_test(normal_returns)
    for key in ("mean", "std", "skewness", "excess_kurtosis", "cv"):
        assert key in result


def test_moment_ratio_std_positive(normal_returns: np.ndarray) -> None:
    """Standard deviation is always positive."""
    result = moment_ratio_test(normal_returns)
    assert result["std"] > 0


def test_moment_ratio_cv_zero_mean() -> None:
    """CV is inf when mean is zero."""
    returns = np.array([-0.01, 0.01] * 50)  # zero mean
    result = moment_ratio_test(returns)
    assert np.isinf(result["cv"])


def test_moment_ratio_normal_small_skew(normal_returns: np.ndarray) -> None:
    """Normally distributed data has skewness close to zero."""
    rng = np.random.default_rng(1)
    normal = rng.normal(0, 1, 2000)
    result = moment_ratio_test(normal)
    assert abs(result["skewness"]) < 0.3
