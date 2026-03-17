"""WebSocket endpoints for streaming regime and forecast updates."""

from __future__ import annotations

import asyncio
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["register_websocket_routes"]

try:
    from fastapi import WebSocket, WebSocketDisconnect
except ImportError as e:
    raise ImportError(
        "FastAPI is required for WebSocket support. "
        "Install with: pip install synthquant[api]"
    ) from e


async def regime_stream(websocket: WebSocket, symbol: str, interval: float = 2.0) -> None:
    """Stream regime updates over WebSocket.

    Sends regime probabilities every `interval` seconds until the client disconnects.

    Args:
        websocket: The FastAPI WebSocket connection.
        symbol: Ticker symbol to stream.
        interval: Seconds between updates.
    """
    from synthquant.regime.hmm import HMMRegimeDetector

    await websocket.accept()
    rng = np.random.default_rng()
    detector = HMMRegimeDetector(n_components=2, random_state=42)
    seed_returns = rng.normal(0, 0.01, 500)
    detector.fit(seed_returns)

    try:
        while True:
            # Simulate new incoming return
            new_return = rng.normal(0, 0.01, 1)
            proba = detector.predict_proba(new_return)[0]
            current_regime = int(np.argmax(proba))

            message = {
                "symbol": symbol,
                "current_regime": current_regime,
                "regime_proba": proba.tolist(),
                "event": "regime_update",
            }
            await websocket.send_text(json.dumps(message))
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from regime stream for {symbol}")


async def forecast_stream(
    websocket: WebSocket,
    symbol: str,
    n_paths: int = 1000,
    interval: float = 5.0,
) -> None:
    """Stream rolling forecast updates over WebSocket.

    Sends updated forecast percentiles every `interval` seconds.

    Args:
        websocket: The FastAPI WebSocket connection.
        symbol: Ticker symbol to stream.
        n_paths: Number of MC paths per forecast update.
        interval: Seconds between forecast updates.
    """
    from synthquant.analytics.risk_metrics import conditional_var, value_at_risk
    from synthquant.simulation.engines.gbm import GBMEngine

    await websocket.accept()
    engine = GBMEngine()
    rng = np.random.default_rng()

    try:
        while True:
            # Re-simulate with slightly perturbed parameters
            sigma = 0.20 + rng.normal(0, 0.01)
            paths = engine.simulate(
                n_paths=n_paths,
                n_steps=21,
                dt=1 / 252,
                S0=100.0,
                mu=0.07,
                sigma=max(sigma, 0.01),
                random_state=int(rng.integers(0, 2**31)),
            )
            S0 = paths[:, 0]
            terminal = paths[:, -1]
            log_ret = np.log(terminal / S0)

            message = {
                "symbol": symbol,
                "event": "forecast_update",
                "mean": float(np.mean(log_ret)),
                "std": float(np.std(log_ret)),
                "var_95": value_at_risk(paths, alpha=0.05),
                "cvar_95": conditional_var(paths, alpha=0.05),
                "p50": float(np.percentile(log_ret, 50)),
                "p5": float(np.percentile(log_ret, 5)),
                "p95": float(np.percentile(log_ret, 95)),
            }
            await websocket.send_text(json.dumps(message))
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from forecast stream for {symbol}")


def register_websocket_routes(app: object) -> None:
    """Register WebSocket routes on a FastAPI app.

    Args:
        app: FastAPI application instance.
    """
    from fastapi import FastAPI as _FastAPI

    if not isinstance(app, _FastAPI):
        raise TypeError("app must be a FastAPI instance")

    @app.websocket("/ws/regimes/{symbol}")  # type: ignore[misc]
    async def ws_regimes(websocket: WebSocket, symbol: str) -> None:
        """WebSocket endpoint for streaming regime updates."""
        await regime_stream(websocket, symbol)

    @app.websocket("/ws/forecast/{symbol}")  # type: ignore[misc]
    async def ws_forecast(websocket: WebSocket, symbol: str) -> None:
        """WebSocket endpoint for streaming forecast updates."""
        await forecast_stream(websocket, symbol)
