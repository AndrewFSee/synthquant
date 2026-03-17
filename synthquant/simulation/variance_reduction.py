"""Variance reduction techniques for Monte Carlo simulation."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "antithetic_variates",
    "control_variates",
    "stratified_sampling",
    "importance_sampling",
]


def antithetic_variates(paths: np.ndarray) -> np.ndarray:
    """Generate antithetic paths to reduce variance.

    Reflects the log-returns of each path around zero. The antithetic
    paths are appended to the original, doubling the path count.

    Args:
        paths: Array of shape (n_paths, n_steps+1) with price paths.

    Returns:
        Array of shape (2*n_paths, n_steps+1) containing original and antithetic paths.
    """
    S0 = paths[:, 0:1]
    log_returns = np.diff(np.log(paths), axis=1)
    antithetic_log_returns = -log_returns
    anti_log_paths = np.cumsum(antithetic_log_returns, axis=1)
    zeros = np.zeros((paths.shape[0], 1))
    anti_paths = S0 * np.exp(np.concatenate([zeros, anti_log_paths], axis=1))
    combined = np.concatenate([paths, anti_paths], axis=0)
    logger.debug(f"Antithetic variates: {paths.shape[0]} -> {combined.shape[0]} paths")
    return combined


def control_variates(
    paths: np.ndarray,
    control_paths: np.ndarray,
    control_mean: float,
) -> np.ndarray:
    """Apply control variate correction to path terminal values.

    Reduces variance by exploiting the known mean of a correlated quantity.

    Args:
        paths: Array of shape (n_paths, n_steps+1). Target paths.
        control_paths: Array of shape (n_paths, n_steps+1). Control paths
            (e.g., GBM paths when correcting a jump-diffusion).
        control_mean: Known theoretical mean of the control variate's terminal value.

    Returns:
        Corrected terminal values, shape (n_paths,).
    """
    Y = paths[:, -1]
    X = control_paths[:, -1]
    cov = np.cov(Y, X)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
    corrected = Y - beta * (X - control_mean)
    logger.debug(f"Control variates applied: beta={beta:.4f}")
    return corrected


def stratified_sampling(
    n_paths: int,
    n_strata: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate stratified uniform samples for variance reduction.

    Divides [0,1] into n_strata equal intervals and samples one uniform
    random variable per stratum, ensuring better coverage.

    Args:
        n_paths: Total number of samples to generate.
        n_strata: Number of strata (must divide n_paths evenly).
        rng: Optional random generator for reproducibility.

    Returns:
        Array of shape (n_paths,) with stratified uniform samples in [0,1].
    """
    if rng is None:
        rng = np.random.default_rng()
    n_per_stratum = n_paths // n_strata
    strata_samples = []
    for i in range(n_strata):
        lo = i / n_strata
        hi = (i + 1) / n_strata
        samples = lo + (hi - lo) * rng.random(n_per_stratum)
        strata_samples.append(samples)
    result = np.concatenate(strata_samples)
    logger.debug(f"Stratified sampling: {n_paths} samples across {n_strata} strata")
    return result


def importance_sampling(
    paths: np.ndarray,
    target_quantile: float = 0.05,
) -> np.ndarray:
    """Compute importance weights for tail-focused estimation.

    Assigns higher weights to paths whose terminal value falls in the
    tail below the given quantile.

    Args:
        paths: Array of shape (n_paths, n_steps+1) with price paths.
        target_quantile: Lower tail quantile (e.g., 0.05 for 5% tail).

    Returns:
        Array of shape (n_paths,) with importance weights (normalized).
    """
    terminal = paths[:, -1]
    threshold = np.quantile(terminal, target_quantile)
    weights = np.where(terminal <= threshold, 1.0 / target_quantile, 1.0)
    weights = weights / weights.sum() * len(weights)
    logger.debug(
        f"Importance sampling: target_quantile={target_quantile}, "
        f"threshold={threshold:.4f}"
    )
    return weights
