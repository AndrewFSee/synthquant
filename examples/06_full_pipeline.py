"""Example 06: Full Pipeline — End-to-end from data to strategy and backtest."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Demonstrate the complete SynthQuant pipeline end-to-end."""
    from synthquant.analytics.risk_metrics import (
        conditional_var,
        max_drawdown_distribution,
        value_at_risk,
    )
    from synthquant.data.features import FeatureEngine
    from synthquant.regime.hmm import HMMRegimeDetector
    from synthquant.simulation.engines.regime_switching import RegimeSwitchingEngine
    from synthquant.strategy.allocation import RiskParityAllocator
    from synthquant.strategy.backtest import WalkForwardBacktest
    from synthquant.strategy.signals import RegimeSignalGenerator
    from synthquant.strategy.sizing import KellyCriterion, RiskParitySizer

    logger.info("=== Example 06: Full Pipeline ===")

    # ── 1. Synthesise multi-asset price data ──────────────────────────────────
    rng = np.random.default_rng(42)
    n = 1500
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    assets = ["SPY", "QQQ", "GLD", "TLT"]
    mus = [0.08, 0.12, 0.03, 0.02]
    sigmas = [0.18, 0.22, 0.14, 0.10]
    corr = np.array([
        [1.00,  0.85, -0.20, -0.40],
        [0.85,  1.00, -0.15, -0.35],
        [-0.20, -0.15, 1.00,  0.10],
        [-0.40, -0.35, 0.10,  1.00],
    ])
    L = np.linalg.cholesky(corr)
    Z = rng.standard_normal((n, 4)) @ L.T
    ret_matrix = np.column_stack([
        Z[:, i] * sigmas[i] / np.sqrt(252) + mus[i] / 252
        for i in range(4)
    ])
    price_matrix = 100.0 * np.cumprod(1 + ret_matrix, axis=0)
    prices = pd.DataFrame(price_matrix, index=dates, columns=assets)
    logger.info(f"Synthesised {len(assets)}-asset price history: {prices.shape}")

    # ── 2. Feature Engineering on SPY ─────────────────────────────────────────
    fe = FeatureEngine()
    spy_close = prices["SPY"]
    spy_returns = fe.rolling_returns(spy_close, window=1).dropna()
    rv = fe.realized_volatility(
        pd.DataFrame({"open": spy_close, "high": spy_close * 1.001,
                      "low": spy_close * 0.999, "close": spy_close}),
        window=21
    )
    logger.info(f"Current realized vol (SPY): {rv.dropna().iloc[-1]:.4f}")

    # ── 3. Regime Detection ───────────────────────────────────────────────────
    train_n = 1000
    detector = HMMRegimeDetector(n_components=2, random_state=42)
    detector.fit(spy_returns.values[:train_n])
    regimes = detector.predict(spy_returns.values)
    regime_params = detector.get_regime_params()
    logger.info("\nDetected regimes:")
    for i, p in regime_params.items():
        count = int(np.sum(regimes == i))
        logger.info(
            f"  Regime {i}: mu={p['mu']:.6f}, sigma={p['sigma']:.4f}, "
            f"periods={count} ({count/len(regimes)*100:.1f}%)"
        )

    # ── 4. Regime-Switching Monte Carlo ───────────────────────────────────────
    rs_params = [
        {"mu": regime_params[0]["mu"] * 252, "sigma": regime_params[0]["sigma"] * np.sqrt(252)},
        {"mu": regime_params[1]["mu"] * 252, "sigma": regime_params[1]["sigma"] * np.sqrt(252)},
    ]
    transition = np.array([[0.97, 0.03], [0.05, 0.95]])
    current_regime = int(regimes[-1])
    rs_engine = RegimeSwitchingEngine(rs_params, transition, initial_regime=current_regime)
    mc_paths = rs_engine.simulate(
        n_paths=10_000, n_steps=252, dt=1 / 252,
        S0=float(spy_close.iloc[-1]), random_state=42,
    )
    logger.info(f"\nMC simulation: {mc_paths.shape}")

    # ── 5. Risk Metrics ───────────────────────────────────────────────────────
    var95 = value_at_risk(mc_paths, alpha=0.05)
    cvar95 = conditional_var(mc_paths, alpha=0.05)
    dds = max_drawdown_distribution(mc_paths)
    logger.info("\n1-year Risk Metrics (SPY, Regime-Switching MC):")
    logger.info(f"  95% VaR:   {var95:.4f} ({var95*100:.2f}%)")
    logger.info(f"  95% CVaR:  {cvar95:.4f} ({cvar95*100:.2f}%)")
    logger.info(f"  Mean MDD:  {dds.mean():.4f} ({dds.mean()*100:.2f}%)")

    # ── 6. Position Sizing ────────────────────────────────────────────────────
    kelly = KellyCriterion()
    daily_ret = spy_returns.values
    full_k = kelly.full_kelly(daily_ret)
    half_k = kelly.fractional_kelly(daily_ret, fraction=0.5)
    logger.info(f"\nKelly sizing on SPY: full={full_k:.4f}, half={half_k:.4f}")

    rps = RiskParitySizer()
    daily_ret_matrix = np.log(prices / prices.shift(1)).dropna().values
    rp_weights = rps.compute_weights(daily_ret_matrix, target_vol=0.10)
    logger.info(f"Risk-parity weights: {dict(zip(assets, rp_weights.round(4)))}")

    # ── 7. Portfolio Optimization ─────────────────────────────────────────────
    allocator = RiskParityAllocator()
    cov = np.cov(daily_ret_matrix.T) * 252
    erc_weights = allocator.allocate(cov)
    logger.info(f"ERC weights: {dict(zip(assets, erc_weights.round(4)))}")

    # ── 8. Regime Signals ─────────────────────────────────────────────────────
    signal_gen = RegimeSignalGenerator(regime_signal_map={0: 1, 1: 0})
    signals = signal_gen.generate(regimes)
    pct_invested = np.mean(signals["signal"] > 0)
    logger.info(f"\nRegime signal: {int(np.sum(signals['entry']))} entries, "
                f"{pct_invested*100:.1f}% of time invested")

    # ── 9. Walk-Forward Backtest ──────────────────────────────────────────────
    def equal_weight_strategy(train_prices: pd.DataFrame) -> np.ndarray:
        return np.ones(train_prices.shape[1]) / train_prices.shape[1]

    backtest = WalkForwardBacktest(train_window=252, rebalance_freq=21, transaction_cost=0.001)
    results = backtest.run(prices, equal_weight_strategy)

    logger.info("\nWalk-Forward Backtest (Equal Weight):")
    logger.info(f"  Total Return: {results.total_return:.4f} ({results.total_return*100:.2f}%)")
    logger.info(f"  Sharpe:       {results.sharpe:.4f}")
    logger.info(f"  Sortino:      {results.sortino:.4f}")
    logger.info(f"  Calmar:       {results.calmar:.4f}")
    logger.info(f"  Max Drawdown: {results.max_drawdown:.4f} ({results.max_drawdown*100:.2f}%)")
    logger.info(f"  Rebalances:   {results.n_trades}")

    logger.info("=== Example 06 complete ===")


if __name__ == "__main__":
    main()
