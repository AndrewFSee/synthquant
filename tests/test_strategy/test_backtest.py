"""Tests for walk-forward backtesting framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthquant.strategy.backtest import BacktestResults, WalkForwardBacktest


@pytest.fixture()
def prices() -> pd.DataFrame:
    """Synthetic multi-asset price history with 504 days and 3 assets."""
    rng = np.random.default_rng(42)
    n = 504
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = {}
    for i, sym in enumerate(["A", "B", "C"]):
        data[sym] = 100.0 * np.cumprod(1 + rng.normal(0.0004, 0.012, n))
    return pd.DataFrame(data, index=dates)


def equal_weight_strategy(train_prices: pd.DataFrame) -> np.ndarray:
    """Simple equal-weight strategy."""
    n_assets = train_prices.shape[1]
    return np.ones(n_assets) / n_assets


# ── Basic Correctness ─────────────────────────────────────────────────────────

def test_backtest_returns_results_object(prices: pd.DataFrame) -> None:
    """run() returns a BacktestResults instance."""
    bt = WalkForwardBacktest(train_window=252, rebalance_freq=21)
    results = bt.run(prices, equal_weight_strategy)
    assert isinstance(results, BacktestResults)


def test_backtest_pnl_is_series(prices: pd.DataFrame) -> None:
    """pnl is a pandas Series."""
    bt = WalkForwardBacktest(train_window=252, rebalance_freq=21)
    results = bt.run(prices, equal_weight_strategy)
    assert isinstance(results.pnl, pd.Series)


def test_backtest_cumulative_returns_is_series(prices: pd.DataFrame) -> None:
    """cumulative_returns is a pandas Series."""
    bt = WalkForwardBacktest(train_window=252, rebalance_freq=21)
    results = bt.run(prices, equal_weight_strategy)
    assert isinstance(results.cumulative_returns, pd.Series)


def test_backtest_pnl_length(prices: pd.DataFrame) -> None:
    """PnL length equals n_periods - train_window."""
    train_window = 252
    bt = WalkForwardBacktest(train_window=train_window, rebalance_freq=21)
    results = bt.run(prices, equal_weight_strategy)
    log_returns = prices.pct_change().apply(np.log1p).dropna()
    expected_len = len(log_returns) - train_window
    assert len(results.pnl) == expected_len


def test_backtest_max_drawdown_non_positive(prices: pd.DataFrame) -> None:
    """Maximum drawdown is always <= 0."""
    bt = WalkForwardBacktest(train_window=252, rebalance_freq=21)
    results = bt.run(prices, equal_weight_strategy)
    assert results.max_drawdown <= 0


def test_backtest_n_trades_positive(prices: pd.DataFrame) -> None:
    """Number of trades is positive (at least one rebalance)."""
    bt = WalkForwardBacktest(train_window=252, rebalance_freq=21)
    results = bt.run(prices, equal_weight_strategy)
    assert results.n_trades > 0


def test_backtest_weights_history_non_empty(prices: pd.DataFrame) -> None:
    """Weights history contains weight arrays for each rebalance."""
    bt = WalkForwardBacktest(train_window=252, rebalance_freq=21)
    results = bt.run(prices, equal_weight_strategy)
    assert len(results.weights_history) == results.n_trades


def test_backtest_sharpe_is_float(prices: pd.DataFrame) -> None:
    """Sharpe ratio is a finite float."""
    bt = WalkForwardBacktest(train_window=252, rebalance_freq=21)
    results = bt.run(prices, equal_weight_strategy)
    assert isinstance(results.sharpe, float)
    assert np.isfinite(results.sharpe)


# ── Transaction Costs ─────────────────────────────────────────────────────────

def test_higher_transaction_cost_lowers_total_return(prices: pd.DataFrame) -> None:
    """Higher transaction costs reduce total return."""
    bt_cheap = WalkForwardBacktest(train_window=252, rebalance_freq=21, transaction_cost=0.0)
    bt_expensive = WalkForwardBacktest(
        train_window=252, rebalance_freq=21, transaction_cost=0.01
    )
    r_cheap = bt_cheap.run(prices, equal_weight_strategy).total_return
    r_expensive = bt_expensive.run(prices, equal_weight_strategy).total_return
    assert r_cheap >= r_expensive


# ── Static Metrics ────────────────────────────────────────────────────────────

def test_sharpe_zero_for_constant_pnl() -> None:
    """Sharpe ratio is 0 when PnL has zero standard deviation."""
    pnl = pd.Series(np.zeros(100))
    result = WalkForwardBacktest._sharpe(pnl)
    assert result == 0.0


def test_max_drawdown_zero_for_monotone_increase() -> None:
    """Max drawdown is 0 for a monotonically increasing cumulative return."""
    cumulative = pd.Series(np.linspace(0, 1, 100))
    result = WalkForwardBacktest._max_drawdown(cumulative)
    assert result == pytest.approx(0.0)
