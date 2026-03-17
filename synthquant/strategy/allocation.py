"""Portfolio allocation optimizers."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["MeanCVaROptimizer", "RiskParityAllocator"]


class MeanCVaROptimizer:
    """Mean-CVaR portfolio optimizer.

    Maximises expected return subject to a CVaR constraint,
    or minimises CVaR subject to a minimum return constraint.

    Args:
        alpha: CVaR tail probability (e.g., 0.05 for 95% CVaR).
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def optimize(
        self,
        returns_matrix: np.ndarray,
        target_return: float | None = None,
        max_cvar: float | None = None,
    ) -> np.ndarray:
        """Find optimal weights minimising CVaR.

        Uses scipy linear programming (Rockafellar-Uryasev formulation).

        Args:
            returns_matrix: Array of shape (n_scenarios, n_assets).
            target_return: Minimum required expected portfolio return (annualised).
            max_cvar: Maximum allowed CVaR (negative). If None, minimises CVaR.

        Returns:
            Array of shape (n_assets,) with optimal weights (sum = 1).
        """
        from scipy.optimize import linprog

        R = np.asarray(returns_matrix, dtype=float)
        n_scenarios, n_assets = R.shape
        alpha = self.alpha

        # Variables: [w (n_assets), z (1), u (n_scenarios)]
        # Minimise CVaR: z + (1/(alpha*n)) * sum(u)
        # s.t. u_i >= -R_i.w - z, u_i >= 0, sum(w) = 1, w >= 0
        n_vars = n_assets + 1 + n_scenarios
        c = np.zeros(n_vars)
        c[n_assets] = 1.0  # z
        c[n_assets + 1:] = 1.0 / (alpha * n_scenarios)  # u_i

        # u_i >= -R_i.w - z  =>  R_i.w + z + u_i >= 0
        # As <= constraint: -R_i.w - z - u_i <= 0
        A_ub = np.zeros((n_scenarios, n_vars))
        A_ub[:, :n_assets] = -R
        A_ub[:, n_assets] = -1.0
        A_ub[:, n_assets + 1:] = -np.eye(n_scenarios)
        b_ub = np.zeros(n_scenarios)

        # sum(w) = 1
        A_eq = np.zeros((1, n_vars))
        A_eq[0, :n_assets] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0, None)] * n_assets + [(None, None)] + [(0, None)] * n_scenarios

        if target_return is not None:
            A_ret = np.zeros((1, n_vars))
            A_ret[0, :n_assets] = -np.mean(R, axis=0) * 252
            b_ret = np.array([-target_return])
            A_ub = np.vstack([A_ub, A_ret])  # type: ignore[assignment]
            b_ub = np.append(b_ub, b_ret)  # type: ignore[assignment]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

        if not result.success:
            logger.warning(f"MeanCVaROptimizer did not converge: {result.message}")
            return np.ones(n_assets) / n_assets

        weights = result.x[:n_assets]
        weights = np.clip(weights, 0, None)
        weights /= weights.sum()
        logger.info(f"MeanCVaROptimizer: optimal weights={weights.round(4)}")
        return weights


class RiskParityAllocator:
    """Risk Parity portfolio allocator.

    Finds weights such that each asset contributes equally to total
    portfolio variance (Equal Risk Contribution).

    Args:
        max_iter: Maximum iterations for the iterative algorithm.
        tol: Convergence tolerance.
    """

    def __init__(self, max_iter: int = 1000, tol: float = 1e-8) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def allocate(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """Compute Equal Risk Contribution weights.

        Args:
            covariance_matrix: Array of shape (n_assets, n_assets).

        Returns:
            Array of shape (n_assets,) with ERC weights (sum = 1).
        """
        sigma = np.asarray(covariance_matrix, dtype=float)
        n = sigma.shape[0]
        w = np.ones(n) / n

        for _ in range(self.max_iter):
            mrc = sigma @ w  # Marginal Risk Contribution
            # Update weights proportional to inverse marginal risk
            w_new = w / mrc
            w_new /= w_new.sum()
            if np.max(np.abs(w_new - w)) < self.tol:
                break
            w = w_new

        w = np.clip(w, 0, None)
        w /= w.sum()
        logger.info(f"RiskParityAllocator: weights={w.round(4)}")
        return w
