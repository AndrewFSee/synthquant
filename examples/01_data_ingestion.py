"""Example 01: Data Ingestion — Fetch SPY data, compute features, store locally."""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Demonstrate data ingestion, feature engineering, and storage."""
    from synthquant.data.features import FeatureEngine
    from synthquant.data.storage import ParquetStorage

    logger.info("=== Example 01: Data Ingestion ===")

    # --- Synthesise OHLCV data (no internet required) -------------------------
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    close = 400.0 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    spread = rng.uniform(0.002, 0.006, n)
    ohlcv = pd.DataFrame(
        {
            "open": close * (1 - spread / 2),
            "high": close * (1 + spread),
            "low": close * (1 - spread),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )
    logger.info(f"Synthetic SPY data: {len(ohlcv)} rows from {dates[0].date()} to {dates[-1].date()}")

    # --- Feature Engineering --------------------------------------------------
    fe = FeatureEngine()

    returns = fe.rolling_returns(ohlcv["close"], window=1)
    logger.info(f"Log returns: mean={returns.mean():.6f}, std={returns.std():.6f}")

    rv_cc = fe.realized_volatility(ohlcv, window=21, estimator="close_to_close")
    rv_gk = fe.realized_volatility(ohlcv, window=21, estimator="garman_klass")
    rv_yz = fe.realized_volatility(ohlcv, window=21, estimator="yang_zhang")
    logger.info(
        f"Realized Vol (annualised) — "
        f"C2C: {rv_cc.iloc[-1]:.4f}, GK: {rv_gk.iloc[-1]:.4f}, YZ: {rv_yz.iloc[-1]:.4f}"
    )

    rsi = fe.rsi(ohlcv["close"], period=14)
    logger.info(f"RSI(14): last={rsi.iloc[-1]:.2f}")

    macd_df = fe.macd(ohlcv["close"])
    logger.info(f"MACD: {macd_df.iloc[-1].to_dict()}")

    bb = fe.bollinger_bands(ohlcv["close"], window=20)
    logger.info(
        f"Bollinger Bands: upper={bb['upper'].iloc[-1]:.2f}, "
        f"lower={bb['lower'].iloc[-1]:.2f}, %B={bb['pct_b'].iloc[-1]:.4f}"
    )

    # --- Storage --------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = ParquetStorage(base_dir=tmp_dir)
        storage.save("SPY", ohlcv)
        symbols = storage.list_symbols()
        logger.info(f"Stored symbols: {symbols}")

        loaded = storage.load("SPY")
        logger.info(f"Loaded back: {loaded.shape}, index type: {type(loaded.index).__name__}")

    logger.info("=== Example 01 complete ===")


if __name__ == "__main__":
    main()
