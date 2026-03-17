"""Risk metrics computed from Monte Carlo path arrays."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "value_at_risk",
    "conditional_var",
    "expected_shortfall",
    "max_drawdown_distribution",
    "tail_ratio",
]


def value_at_risk(
    paths: np.ndarray,
    alpha: float = 0.05,
    holding_period: int | None = None,
) -> float:
    """Compute Value-at-Risk from simulated price paths.

    VaR is expressed as a negative number representing a loss level
    that is exceeded with probability alpha.

    Args:
        paths: Array of shape (n_paths, n_steps+1) with price paths.
        alpha: Tail probability (e.g., 0.05 for 95% VaR).
        holding_period: Number of steps for the holding period.
            If None, uses the full path horizon.

    Returns:
        VaR as a (negative) return value at the alpha quantile.
    """
    S0 = paths[:, 0]
    end_col = holding_period if holding_period is not None else paths.shape[1] - 1
    terminal = paths[:, end_col]
    log_returns = np.log(terminal / S0)
    var = float(np.quantile(log_returns, alpha))
    logger.debug(f"VaR({alpha:.3f}) = {var:.6f}")
    return var


def conditional_var(
    paths: np.ndarray,
    alpha: float = 0.05,
    holding_period: int | None = None,
) -> float:
    """Compute Conditional VaR (Expected Shortfall) from simulated paths.

    CVaR is the expected loss given that the loss exceeds VaR.

    Args:
        paths: Array of shape (n_paths, n_steps+1).
        alpha: Tail probability.
        holding_period: Number of steps for the holding period.

    Returns:
        CVaR (negative value).
    """
    S0 = paths[:, 0]
    end_col = holding_period if holding_period is not None else paths.shape[1] - 1
    terminal = paths[:, end_col]
    log_returns = np.log(terminal / S0)
    var_threshold = np.quantile(log_returns, alpha)
    tail = log_returns[log_returns <= var_threshold]
    cvar = float(np.mean(tail)) if len(tail) > 0 else var_threshold
    logger.debug(f"CVaR({alpha:.3f}) = {cvar:.6f}")
    return cvar


def expected_shortfall(
    paths: np.ndarray,
    alpha: float = 0.05,
    holding_period: int | None = None,
) -> float:
    """Alias for conditional_var (Expected Shortfall = CVaR).

    Args:
        paths: Array of shape (n_paths, n_steps+1).
        alpha: Tail probability.
        holding_period: Number of steps for the holding period.

    Returns:
        Expected Shortfall (negative value).
    """
    return conditional_var(paths, alpha=alpha, holding_period=holding_period)


def max_drawdown_distribution(paths: np.ndarray) -> np.ndarray:
    """Compute the maximum drawdown for each path.

    Maximum drawdown is the largest peak-to-trough decline, expressed
    as a fraction (always <= 0).

    Args:
        paths: Array of shape (n_paths, n_steps+1).

    Returns:
        Array of shape (n_paths,) with max drawdown per path (non-positive).
    """
    cummax = np.maximum.accumulate(paths, axis=1)
    drawdown = (paths - cummax) / cummax
    max_dd = np.min(drawdown, axis=1)
    logger.debug(
        f"Max drawdown distribution: mean={max_dd.mean():.4f}, min={max_dd.min():.4f}"
    )
    return max_dd


def tail_ratio(
    paths: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Compute the tail ratio: upper-tail gain / lower-tail loss.

    Args:
        paths: Array of shape (n_paths, n_steps+1).
        alpha: Tail probability (symmetric, applied to both tails).

    Returns:
        Tail ratio (positive float). Values > 1 indicate fatter upper tails.
    """
    S0 = paths[:, 0]
    terminal = paths[:, -1]
    log_returns = np.log(terminal / S0)
    upper = float(np.quantile(log_returns, 1 - alpha))
    lower = float(np.quantile(log_returns, alpha))
    ratio = abs(upper) / abs(lower) if lower != 0.0 else np.inf
    logger.debug(f"Tail ratio({alpha:.3f}) = {ratio:.4f}")
    return ratio
