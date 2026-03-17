"""Rolling moment statistics and normality tests."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

__all__ = [
    "rolling_skewness",
    "rolling_kurtosis",
    "jarque_bera_test",
    "moment_ratio_test",
]


def rolling_skewness(returns: pd.Series | np.ndarray, window: int = 60) -> pd.Series:
    """Compute rolling skewness of a return series.

    Args:
        returns: 1-D return series.
        window: Rolling window length.

    Returns:
        Pandas Series of rolling skewness values.
    """
    s = pd.Series(np.asarray(returns, dtype=float))
    result = s.rolling(window).skew()
    logger.debug(f"Rolling skewness computed, window={window}")
    return result


def rolling_kurtosis(returns: pd.Series | np.ndarray, window: int = 60) -> pd.Series:
    """Compute rolling excess kurtosis of a return series.

    Args:
        returns: 1-D return series.
        window: Rolling window length.

    Returns:
        Pandas Series of rolling excess kurtosis values.
    """
    s = pd.Series(np.asarray(returns, dtype=float))
    result = s.rolling(window).kurt()
    logger.debug(f"Rolling kurtosis computed, window={window}")
    return result


def jarque_bera_test(returns: np.ndarray) -> dict[str, float]:
    """Perform the Jarque-Bera normality test.

    Args:
        returns: 1-D array of returns.

    Returns:
        Dict with keys: 'statistic', 'p_value', 'skewness', 'kurtosis'.
    """
    r = np.asarray(returns, dtype=float)
    jb_stat, p_value = stats.jarque_bera(r)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=True))
    result = {
        "statistic": float(jb_stat),
        "p_value": float(p_value),
        "skewness": skew,
        "kurtosis": kurt,
    }
    logger.debug(f"Jarque-Bera: stat={jb_stat:.4f}, p={p_value:.4f}")
    return result


def moment_ratio_test(returns: np.ndarray) -> dict[str, float]:
    """Compute moment-ratio statistics for return distribution analysis.

    Returns the first four standardised moments and the coefficient of variation.

    Args:
        returns: 1-D array of returns.

    Returns:
        Dict with keys: 'mean', 'std', 'skewness', 'excess_kurtosis', 'cv'.
    """
    r = np.asarray(returns, dtype=float)
    mean = float(np.mean(r))
    std = float(np.std(r, ddof=1))
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=True))
    cv = std / abs(mean) if mean != 0 else np.inf
    result = {
        "mean": mean,
        "std": std,
        "skewness": skew,
        "excess_kurtosis": kurt,
        "cv": cv,
    }
    logger.debug(f"Moment ratios: mean={mean:.6f}, std={std:.6f}, skew={skew:.4f}")
    return result
