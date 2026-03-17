"""Walk-forward backtesting framework."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["WalkForwardBacktest", "BacktestResults"]


@dataclass
class BacktestResults:
    """Results of a walk-forward backtest.

    Attributes:
        pnl: Daily PnL series.
        cumulative_returns: Cumulative return series.
        sharpe: Annualised Sharpe ratio.
        sortino: Annualised Sortino ratio.
        calmar: Calmar ratio (annualised return / max drawdown).
        max_drawdown: Maximum drawdown (negative).
        total_return: Total return over the backtest period.
        n_trades: Number of position changes.
        weights_history: List of weight arrays at each rebalance date.
    """

    pnl: pd.Series
    cumulative_returns: pd.Series
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    total_return: float
    n_trades: int
    weights_history: list[np.ndarray] = field(default_factory=list)


class WalkForwardBacktest:
    """Walk-forward backtesting engine.

    At each rebalance date, the strategy function is called with the
    trailing training window of prices to produce portfolio weights.
    Out-of-sample returns are then computed and aggregated.

    Args:
        train_window: Number of periods in the training window.
        rebalance_freq: Number of periods between rebalances.
        transaction_cost: One-way transaction cost as a fraction.
    """

    def __init__(
        self,
        train_window: int = 252,
        rebalance_freq: int = 21,
        transaction_cost: float = 0.001,
    ) -> None:
        self.train_window = train_window
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost

    def run(
        self,
        prices: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame], np.ndarray],
    ) -> BacktestResults:
        """Execute the walk-forward backtest.

        Args:
            prices: DataFrame of shape (n_dates, n_assets) with price history.
            strategy_fn: Callable that takes a price DataFrame (training window)
                and returns portfolio weights, shape (n_assets,).

        Returns:
            BacktestResults with performance metrics.
        """
        log_returns = prices.pct_change().apply(np.log1p).dropna()
        n_periods = len(log_returns)
        n_assets = prices.shape[1]

        pnl_list: list[float] = []
        pnl_dates: list[pd.Timestamp] = []
        weights_history: list[np.ndarray] = []
        current_weights = np.ones(n_assets) / n_assets
        n_trades = 0

        for t in range(self.train_window, n_periods):
            # Rebalance at specified frequency
            if (t - self.train_window) % self.rebalance_freq == 0:
                train_prices = prices.iloc[t - self.train_window: t]
                new_weights = strategy_fn(train_prices)
                new_weights = np.asarray(new_weights, dtype=float)
                new_weights = np.clip(new_weights, 0, None)
                if new_weights.sum() > 0:
                    new_weights /= new_weights.sum()
                else:
                    new_weights = np.ones(n_assets) / n_assets

                # Transaction costs
                tc = self.transaction_cost * np.sum(np.abs(new_weights - current_weights))
                current_weights = new_weights
                weights_history.append(current_weights.copy())
                n_trades += 1
            else:
                tc = 0.0

            # Portfolio return for this period
            period_ret = float(log_returns.iloc[t].values @ current_weights) - tc
            pnl_list.append(period_ret)
            pnl_dates.append(log_returns.index[t])

        pnl = pd.Series(pnl_list, index=pnl_dates, name="pnl")
        cumulative = pnl.cumsum().rename("cumulative_return")

        sharpe = self._sharpe(pnl)
        sortino = self._sortino(pnl)
        max_dd = self._max_drawdown(cumulative)
        total_return = float(cumulative.iloc[-1]) if len(cumulative) > 0 else 0.0
        calmar = (total_return / abs(max_dd)) if max_dd != 0 else np.inf

        logger.info(
            f"Backtest complete: Sharpe={sharpe:.3f}, Sortino={sortino:.3f}, "
            f"MaxDD={max_dd:.4f}, Total={total_return:.4f}"
        )
        return BacktestResults(
            pnl=pnl,
            cumulative_returns=cumulative,
            sharpe=sharpe,
            sortino=sortino,
            calmar=calmar,
            max_drawdown=max_dd,
            total_return=total_return,
            n_trades=n_trades,
            weights_history=weights_history,
        )

    @staticmethod
    def _sharpe(pnl: pd.Series, periods: int = 252) -> float:
        """Annualised Sharpe ratio."""
        if pnl.std() == 0:
            return 0.0
        return float(pnl.mean() / pnl.std() * np.sqrt(periods))

    @staticmethod
    def _sortino(pnl: pd.Series, periods: int = 252) -> float:
        """Annualised Sortino ratio."""
        downside = pnl[pnl < 0]
        if len(downside) == 0 or downside.std() == 0:
            return np.inf
        return float(pnl.mean() / downside.std() * np.sqrt(periods))

    @staticmethod
    def _max_drawdown(cumulative_returns: pd.Series) -> float:
        """Maximum drawdown from cumulative returns."""
        roll_max = cumulative_returns.cummax()
        dd = cumulative_returns - roll_max
        return float(dd.min())
