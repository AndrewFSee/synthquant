"""Example 03: Volatility Modeling — Fit GARCH and Heston, compare forecasts."""

from __future__ import annotations

import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Fit GARCH and Heston models on synthetic data, compare their volatility forecasts."""
    from synthquant.models.garch import GARCHModel
    from synthquant.models.stochastic_vol import HestonModel

    logger.info("=== Example 03: Volatility Modeling ===")

    # --- Synthetic return series ----------------------------------------------
    rng = np.random.default_rng(42)
    n = 1000
    # GARCH(1,1)-like returns with volatility clustering
    sigma2 = np.zeros(n)
    r = np.zeros(n)
    sigma2[0] = 0.0001
    omega, alpha_g, beta_g = 1e-6, 0.08, 0.90
    for t in range(1, n):
        sigma2[t] = omega + alpha_g * r[t - 1] ** 2 + beta_g * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    logger.info(f"Synthetic returns: n={n}, mean={r.mean():.6f}, std={r.std():.6f}")

    # --- GARCH Model ----------------------------------------------------------
    logger.info("\n--- GARCH(1,1) ---")
    garch = GARCHModel(vol_model="GARCH", p=1, q=1)
    garch.fit(r)
    vol_forecast = garch.forecast(horizon=21)
    logger.info(f"GARCH 21-day annualised vol forecast: {vol_forecast.round(4)}")
    logger.info(f"  Day-1: {vol_forecast[0]:.4f}, Day-21: {vol_forecast[-1]:.4f}")

    garch_paths = garch.simulate(n_paths=1000, horizon=21, random_state=42)
    logger.info(f"Simulated GARCH paths: shape={garch_paths.shape}")
    logger.info(
        f"  Mean annualised vol of sims: {np.std(garch_paths) * np.sqrt(252):.4f}"
    )

    # --- EGARCH ---------------------------------------------------------------
    logger.info("\n--- EGARCH(1,1) ---")
    egarch = GARCHModel(vol_model="EGARCH", p=1, q=1)
    egarch.fit(r)
    evol = egarch.forecast(horizon=21)
    logger.info(f"EGARCH 21-day vol forecast (day-1): {evol[0]:.4f}")

    # --- Heston Simulation ----------------------------------------------------
    logger.info("\n--- Heston Model ---")
    heston = HestonModel(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, mu=0.05)
    S, V = heston.simulate(S0=100.0, T=1.0, n_paths=1000, n_steps=252, random_state=42)
    mean_terminal_vol = float(np.mean(np.sqrt(V[:, -1]) * np.sqrt(252)))
    logger.info(f"Heston paths: S shape={S.shape}, V shape={V.shape}")
    logger.info(f"  Mean terminal annualised vol: {mean_terminal_vol:.4f}")
    logger.info(f"  S terminal: mean={S[:,-1].mean():.2f}, std={S[:,-1].std():.2f}")

    logger.info("=== Example 03 complete ===")


if __name__ == "__main__":
    main()
