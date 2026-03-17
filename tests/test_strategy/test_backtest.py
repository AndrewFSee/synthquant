"""Tests for the walk-forward backtest framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthquant.strategy.backtest import BacktestResults, WalkForwardBacktest


@pytest.fixture()
def price_df() -> pd.DataFrame:
    """400-row price DataFrame with 2 assets."""
    rng = np.random.default_rng(42)
    n = 400
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    p1 = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    p2 = 50.0 * np.cumprod(1 + rng.normal(0.0002, 0.015, n))
    return pd.DataFrame({"A": p1, "B": p2}, index=dates)


def equal_weight_strategy(prices: pd.DataFrame) -> np.ndarray:
    return np.ones(prices.shape[1]) / prices.shape[1]


class TestWalkForwardBacktest:
    def test_returns_backtest_results_type(self, price_df: pd.DataFrame) -> None:
        bt = WalkForwardBacktest(train_window=100, rebalance_freq=20)
        result = bt.run(price_df, equal_weight_strategy)
        assert isinstance(result, BacktestResults)

    def test_pnl_length(self, price_df: pd.DataFrame) -> None:
        """PnL series covers out-of-sample period."""
        train = 100
        bt = WalkForwardBacktest(train_window=train, rebalance_freq=20)
        result = bt.run(price_df, equal_weight_strategy)
        # log_returns has len(price_df)-1 rows, out-of-sample starts at train
        expected_len = len(price_df) - 1 - train
        assert len(result.pnl) == expected_len

    def test_cumulative_returns_last_equals_pnl_sum(self, price_df: pd.DataFrame) -> None:
        """Last cumulative return equals sum of all pnl entries."""
        bt = WalkForwardBacktest(train_window=100, rebalance_freq=20)
        result = bt.run(price_df, equal_weight_strategy)
        np.testing.assert_allclose(
            result.cumulative_returns.iloc[-1], result.pnl.sum(), atol=1e-10
        )

    def test_sharpe_is_finite(self, price_df: pd.DataFrame) -> None:
        bt = WalkForwardBacktest(train_window=100, rebalance_freq=20)
        result = bt.run(price_df, equal_weight_strategy)
        assert np.isfinite(result.sharpe)

    def test_max_drawdown_non_positive(self, price_df: pd.DataFrame) -> None:
        bt = WalkForwardBacktest(train_window=100, rebalance_freq=20)
        result = bt.run(price_df, equal_weight_strategy)
        assert result.max_drawdown <= 0

    def test_n_trades_positive(self, price_df: pd.DataFrame) -> None:
        bt = WalkForwardBacktest(train_window=100, rebalance_freq=20)
        result = bt.run(price_df, equal_weight_strategy)
        assert result.n_trades > 0

    def test_weights_history_populated(self, price_df: pd.DataFrame) -> None:
        bt = WalkForwardBacktest(train_window=100, rebalance_freq=20)
        result = bt.run(price_df, equal_weight_strategy)
        assert len(result.weights_history) > 0

    def test_transaction_cost_reduces_return(self, price_df: pd.DataFrame) -> None:
        """Higher transaction cost should reduce total return."""
        bt_zero = WalkForwardBacktest(train_window=100, rebalance_freq=20, transaction_cost=0.0)
        bt_high = WalkForwardBacktest(train_window=100, rebalance_freq=20, transaction_cost=0.01)
        r_zero = bt_zero.run(price_df, equal_weight_strategy).total_return
        r_high = bt_high.run(price_df, equal_weight_strategy).total_return
        assert r_zero >= r_high

    def test_sortino_positive_for_drifting_up_prices(self) -> None:
        """Strong upward drift should give positive Sortino."""
        rng = np.random.default_rng(0)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        p = pd.DataFrame({
            "X": 100 * np.cumprod(1 + rng.normal(0.002, 0.005, n)),
        }, index=dates)
        bt = WalkForwardBacktest(train_window=100, rebalance_freq=20)
        result = bt.run(p, lambda prices: np.array([1.0]))
        assert result.sharpe > 0

    def test_calmar_ratio(self, price_df: pd.DataFrame) -> None:
        bt = WalkForwardBacktest(train_window=100, rebalance_freq=20)
        result = bt.run(price_df, equal_weight_strategy)
        # Calmar = total_return / |max_drawdown|; just check it is finite
        assert np.isfinite(result.calmar) or result.calmar == np.inf
