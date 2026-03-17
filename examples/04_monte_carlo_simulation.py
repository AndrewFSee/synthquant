"""Example 04: Monte Carlo Simulation — Regime-switching MC with fan chart summary."""

from __future__ import annotations

import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run multiple simulation engines and compare terminal distributions."""
    from synthquant.simulation.engines.gbm import GBMEngine
    from synthquant.simulation.engines.heston import HestonEngine
    from synthquant.simulation.engines.merton_jd import MertonJDEngine
    from synthquant.simulation.engines.regime_switching import RegimeSwitchingEngine
    from synthquant.simulation.variance_reduction import antithetic_variates

    logger.info("=== Example 04: Monte Carlo Simulation ===")

    n_paths = 5_000
    n_steps = 252
    dt = 1 / 252
    S0 = 100.0

    # --- GBM ------------------------------------------------------------------
    gbm = GBMEngine()
    gbm_paths = gbm.simulate(n_paths, n_steps, dt, S0=S0, mu=0.07, sigma=0.20, random_state=42)
    logger.info(f"GBM paths: shape={gbm_paths.shape}")
    logger.info(
        f"  Terminal: mean={gbm_paths[:,-1].mean():.2f}, std={gbm_paths[:,-1].std():.2f}"
    )

    # --- GBM + Antithetic Variates -------------------------------------------
    gbm_base = gbm.simulate(n_paths // 2, n_steps, dt, S0=S0, mu=0.07, sigma=0.20, random_state=42)
    gbm_anti = antithetic_variates(gbm_base)
    logger.info(f"Antithetic paths: {gbm_base.shape[0]} -> {gbm_anti.shape[0]}")
    logger.info(
        f"  Antithetic terminal std: {gbm_anti[:,-1].std():.2f} "
        f"vs. base: {gbm_base[:,-1].std():.2f}"
    )

    # --- Merton Jump-Diffusion ------------------------------------------------
    merton = MertonJDEngine()
    merton_paths = merton.simulate(
        n_paths, n_steps, dt,
        S0=S0, mu=0.07, sigma=0.15, lambda_j=2.0, mu_j=-0.04, sigma_j=0.08,
        random_state=42,
    )
    logger.info(f"Merton JD paths: shape={merton_paths.shape}")
    logger.info(
        f"  Terminal: mean={merton_paths[:,-1].mean():.2f}, std={merton_paths[:,-1].std():.2f}"
    )

    # --- Heston ---------------------------------------------------------------
    heston_engine = HestonEngine()
    heston_paths = heston_engine.simulate(
        n_paths, n_steps, dt,
        S0=S0, v0=0.04, mu=0.07, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7,
        random_state=42,
    )
    logger.info(f"Heston paths: shape={heston_paths.shape}")
    logger.info(
        f"  Terminal: mean={heston_paths[:,-1].mean():.2f}, std={heston_paths[:,-1].std():.2f}"
    )

    # --- Regime Switching -----------------------------------------------------
    regime_params = [
        {"mu": 0.10, "sigma": 0.12},  # Bull regime
        {"mu": -0.05, "sigma": 0.35},  # Bear regime
    ]
    transition = np.array([[0.97, 0.03], [0.05, 0.95]])
    rs_engine = RegimeSwitchingEngine(regime_params, transition, initial_regime=0)
    rs_paths = rs_engine.simulate(n_paths, n_steps, dt, S0=S0, random_state=42)
    logger.info(f"Regime-Switching paths: shape={rs_paths.shape}")
    logger.info(
        f"  Terminal: mean={rs_paths[:,-1].mean():.2f}, std={rs_paths[:,-1].std():.2f}"
    )

    # --- Fan chart summary (percentiles) --------------------------------------
    logger.info("\n--- Terminal Price Percentiles (1-year) ---")
    for label, paths in [
        ("GBM", gbm_paths),
        ("Merton JD", merton_paths),
        ("Heston", heston_paths),
        ("Regime-Switch", rs_paths),
    ]:
        p = np.percentile(paths[:, -1], [5, 25, 50, 75, 95])
        logger.info(
            f"  {label:<14} P5={p[0]:.1f} P25={p[1]:.1f} P50={p[2]:.1f} "
            f"P75={p[3]:.1f} P95={p[4]:.1f}"
        )

    logger.info("=== Example 04 complete ===")


if __name__ == "__main__":
    main()
