"""Example 05: Risk Analytics — Compute VaR, CVaR, drawdown, Greeks from simulated paths."""

from __future__ import annotations

import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Demonstrate risk analytics using simulated GBM paths."""
    from synthquant.analytics.distributions import EmpiricalDistribution
    from synthquant.analytics.moments import jarque_bera_test, moment_ratio_test
    from synthquant.analytics.options import MCOptionPricer
    from synthquant.analytics.risk_metrics import (
        conditional_var,
        expected_shortfall,
        max_drawdown_distribution,
        tail_ratio,
        value_at_risk,
    )
    from synthquant.analytics.scoring import ForecastScorer
    from synthquant.simulation.engines.gbm import GBMEngine

    logger.info("=== Example 05: Risk Analytics ===")

    # --- Simulate paths -------------------------------------------------------
    engine = GBMEngine()
    paths = engine.simulate(
        n_paths=20_000, n_steps=252, dt=1 / 252,
        S0=100.0, mu=0.07, sigma=0.20, random_state=42,
    )
    logger.info(f"Simulated paths: {paths.shape}")

    # --- Risk Metrics ---------------------------------------------------------
    var_95 = value_at_risk(paths, alpha=0.05)
    var_99 = value_at_risk(paths, alpha=0.01)
    cvar_95 = conditional_var(paths, alpha=0.05)
    es_95 = expected_shortfall(paths, alpha=0.05)
    drawdowns = max_drawdown_distribution(paths)
    tr = tail_ratio(paths, alpha=0.05)

    logger.info("\n--- Risk Metrics ---")
    logger.info(f"  95% VaR:        {var_95:.4f}  ({var_95*100:.2f}%)")
    logger.info(f"  99% VaR:        {var_99:.4f}  ({var_99*100:.2f}%)")
    logger.info(f"  95% CVaR:       {cvar_95:.4f}  ({cvar_95*100:.2f}%)")
    logger.info(f"  95% ES:         {es_95:.4f}  ({es_95*100:.2f}%)")
    logger.info(f"  Mean Max DD:    {drawdowns.mean():.4f}  ({drawdowns.mean()*100:.2f}%)")
    logger.info(f"  Worst Max DD:   {drawdowns.min():.4f}  ({drawdowns.min()*100:.2f}%)")
    logger.info(f"  Tail ratio:     {tr:.4f}")
    assert cvar_95 <= var_95, "CVaR must be <= VaR"

    # --- Empirical Distribution -----------------------------------------------
    ed = EmpiricalDistribution()
    ed.fit(paths)
    lo, hi = ed.confidence_interval(alpha=0.05)
    logger.info(f"\n  Empirical 95% CI: [{lo:.4f}, {hi:.4f}]")
    logger.info(f"  Median return:   {float(ed.quantile(np.array([0.5]))):.4f}")

    # --- Return Moments -------------------------------------------------------
    log_ret = np.log(paths[:, -1] / paths[:, 0])
    jb = jarque_bera_test(log_ret)
    mrt = moment_ratio_test(log_ret)
    logger.info("\n--- Moment Analysis ---")
    logger.info(f"  Mean: {mrt['mean']:.6f}, Std: {mrt['std']:.6f}")
    logger.info(f"  Skew: {mrt['skewness']:.4f}, Kurt: {mrt['excess_kurtosis']:.4f}")
    logger.info(f"  Jarque-Bera: stat={jb['statistic']:.2f}, p={jb['p_value']:.4f}")

    # --- Option Pricing -------------------------------------------------------
    pricer = MCOptionPricer()
    S0 = float(paths[0, 0])
    K = S0 * 1.05  # 5% OTM call
    call = pricer.price_european(paths, K=K, T=1.0, r=0.05, option_type="call")
    put = pricer.price_european(paths, K=K, T=1.0, r=0.05, option_type="put")
    asian_call = pricer.price_asian(paths, K=K, T=1.0, r=0.05, option_type="call")
    barrier = pricer.price_barrier(paths, K=S0, B=S0 * 0.80, T=1.0, r=0.05,
                                   barrier_type="knock_out", option_type="call")
    greeks = pricer.compute_greeks(paths, K=K, T=1.0, r=0.05)

    logger.info("\n--- Option Pricing ---")
    logger.info(f"  European Call (K={K:.1f}):  {call:.4f}")
    logger.info(f"  European Put  (K={K:.1f}):  {put:.4f}")
    logger.info(f"  Asian Call:                 {asian_call:.4f}")
    logger.info(f"  Barrier KO Call (B=80):     {barrier:.4f}")
    logger.info(f"  Delta: {greeks['delta']:.4f}, Gamma: {greeks['gamma']:.6f}")

    # --- Forecast Scoring -----------------------------------------------------
    scorer = ForecastScorer()
    rng = np.random.default_rng(42)
    obs = log_ret[:100]
    fc_ensemble = log_ret.reshape(-1, 1).repeat(50, axis=1)[:100]
    fc_ensemble = fc_ensemble + rng.normal(0, 0.01, fc_ensemble.shape)
    crps_val = scorer.crps(fc_ensemble, obs)
    coverage = scorer.coverage_test(fc_ensemble, obs, alpha=0.05)
    ks = scorer.ks_test(log_ret[:500], log_ret[500:])

    logger.info("\n--- Forecast Scoring ---")
    logger.info(f"  CRPS:             {crps_val:.6f}")
    logger.info(f"  Coverage:         {coverage['empirical_coverage']:.4f} "
                f"(nominal {coverage['nominal_coverage']:.4f})")
    logger.info(f"  KS test: stat={ks['statistic']:.4f}, p={ks['p_value']:.4f}")

    logger.info("=== Example 05 complete ===")


if __name__ == "__main__":
    main()
