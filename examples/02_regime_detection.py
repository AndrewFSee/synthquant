"""Example 02: Regime Detection — Detect regimes on returns using HMM, GMM, and Ensemble."""

from __future__ import annotations

import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def make_regime_returns(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic two-regime return series."""
    rng = np.random.default_rng(seed)
    n = 1000
    # Regime 0: bull (low vol, positive drift)
    # Regime 1: bear (high vol, negative drift)
    regimes_true = np.zeros(n, dtype=int)
    r = np.zeros(n)
    regime = 0
    for t in range(n):
        if regime == 0 and rng.random() < 0.02:
            regime = 1
        elif regime == 1 and rng.random() < 0.05:
            regime = 0
        regimes_true[t] = regime
        if regime == 0:
            r[t] = rng.normal(0.0005, 0.008)
        else:
            r[t] = rng.normal(-0.0010, 0.020)
    return r, regimes_true


def main() -> None:
    """Demonstrate regime detection with HMM, GMM, and Ensemble."""
    from synthquant.regime.clustering import ClusteringRegimeDetector
    from synthquant.regime.ensemble import EnsembleRegimeDetector
    from synthquant.regime.hmm import HMMRegimeDetector

    logger.info("=== Example 02: Regime Detection ===")
    returns, true_regimes = make_regime_returns()
    logger.info(f"Returns: n={len(returns)}, mean={returns.mean():.6f}, std={returns.std():.6f}")
    logger.info(f"True regime distribution: {np.bincount(true_regimes)}")

    # --- HMM ------------------------------------------------------------------
    hmm = HMMRegimeDetector(n_components=2, random_state=42)
    hmm.fit(returns)
    hmm_labels = hmm.predict(returns)
    hmm_proba = hmm.predict_proba(returns)
    hmm_params = hmm.get_regime_params()

    logger.info("HMM Results:")
    logger.info(f"  Regime distribution: {np.bincount(hmm_labels)}")
    for i, p in hmm_params.items():
        logger.info(f"  Regime {i}: mu={p['mu']:.6f}, sigma={p['sigma']:.4f}")
    logger.info(f"  Last-step proba: {hmm_proba[-1].round(4)}")

    # --- GMM ------------------------------------------------------------------
    gmm = ClusteringRegimeDetector(n_components=2, random_state=42)
    gmm.fit(returns)
    gmm_labels = gmm.predict(returns)
    logger.info(f"GMM regime distribution: {np.bincount(gmm_labels)}")

    # --- Ensemble -------------------------------------------------------------
    ensemble = EnsembleRegimeDetector(
        detectors=[
            HMMRegimeDetector(n_components=2, random_state=42),
            ClusteringRegimeDetector(n_components=2, random_state=42),
        ],
        method="majority_vote",
        n_regimes=2,
    )
    ensemble.fit(returns)
    ens_labels = ensemble.predict(returns)
    logger.info(f"Ensemble regime distribution: {np.bincount(ens_labels)}")

    # --- Summary --------------------------------------------------------------
    logger.info("\nCurrent regime summary:")
    logger.info(f"  HMM:      regime {hmm_labels[-1]}")
    logger.info(f"  GMM:      regime {gmm_labels[-1]}")
    logger.info(f"  Ensemble: regime {ens_labels[-1]}")
    logger.info("=== Example 02 complete ===")


if __name__ == "__main__":
    main()
