"""Position sizing algorithms."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["KellyCriterion", "RiskParitySizer", "CVaROptimalSizer"]


class KellyCriterion:
    """Kelly Criterion position sizing.

    Computes the optimal fraction of capital to invest in an asset
    based on the empirical return distribution.
    """

    def full_kelly(self, returns: np.ndarray) -> float:
        """Compute the full Kelly fraction.

        For a continuous return distribution:
            f* = mu / sigma^2

        where mu is the mean return and sigma^2 is the variance.

        Args:
            returns: 1-D array of historical or simulated returns.

        Returns:
            Full Kelly fraction (may exceed 1; consider fractional Kelly in practice).
        """
        r = np.asarray(returns, dtype=float)
        mu = float(np.mean(r))
        sigma2 = float(np.var(r, ddof=1))
        if sigma2 < 1e-15:
            raise ValueError("Zero variance returns - cannot compute Kelly fraction")
        fk = mu / sigma2
        logger.debug(f"Full Kelly: mu={mu:.6f}, sigma2={sigma2:.6f}, f*={fk:.4f}")
        return fk

    def fractional_kelly(self, returns: np.ndarray, fraction: float = 0.5) -> float:
        """Compute a fractional Kelly position size.

        Args:
            returns: 1-D array of historical or simulated returns.
            fraction: Fraction of the full Kelly to use (0 < fraction <= 1).

        Returns:
            Fractional Kelly position size.

        Raises:
            ValueError: If fraction is not in (0, 1].
        """
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        fk = self.full_kelly(returns) * fraction
        logger.debug(f"Fractional Kelly ({fraction:.2f}): {fk:.4f}")
        return fk


class RiskParitySizer:
    """Risk parity position sizing.

    Allocates capital so each asset contributes equally to portfolio volatility.
    """

    def compute_weights(
        self,
        returns_matrix: np.ndarray,
        target_vol: float = 0.10,
    ) -> np.ndarray:
        """Compute risk-parity weights.

        Args:
            returns_matrix: Array of shape (n_obs, n_assets) with asset returns.
            target_vol: Target annualised portfolio volatility (informational;
                returned weights are normalised to sum to 1).

        Returns:
            Array of shape (n_assets,) with portfolio weights (sum = 1).
        """
        R = np.asarray(returns_matrix, dtype=float)
        vols = np.std(R, axis=0, ddof=1) * np.sqrt(252)
        inv_vol = 1.0 / (vols + 1e-12)
        weights = inv_vol / inv_vol.sum()

        port_vol = float(np.sqrt(weights @ np.cov(R.T, ddof=1) * 252 @ weights))
        logger.debug(f"RiskParity weights: {weights.round(4)}, port_vol={port_vol:.4f}")
        return weights


class CVaROptimalSizer:
    """CVaR-constrained position sizing.

    Finds the largest position size such that the CVaR of the return distribution
    does not exceed a specified limit.
    """

    def compute_size(
        self,
        returns: np.ndarray,
        target_cvar: float = -0.02,
        alpha: float = 0.05,
    ) -> float:
        """Find the maximum position size satisfying a CVaR constraint.

        Args:
            returns: 1-D array of simulated per-unit returns.
            target_cvar: Maximum allowed CVaR (negative, e.g., -0.02 = -2%).
            alpha: CVaR tail probability.

        Returns:
            Scalar position size (fraction of capital).
        """
        r = np.asarray(returns, dtype=float)
        var_threshold = np.quantile(r, alpha)
        tail = r[r <= var_threshold]
        cvar_unit = float(np.mean(tail)) if len(tail) > 0 else var_threshold

        if cvar_unit >= 0:
            logger.warning("Unit CVaR is non-negative; returning size 1.0")
            return 1.0

        size = abs(target_cvar) / abs(cvar_unit)
        logger.debug(
            f"CVaR-optimal size: target={target_cvar:.4f}, "
            f"unit_cvar={cvar_unit:.6f}, size={size:.4f}"
        )
        return float(np.clip(size, 0.0, 1.0))
