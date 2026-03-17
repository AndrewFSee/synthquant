"""FastAPI application server for SynthQuant."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
except ImportError as e:
    raise ImportError(
        "FastAPI is required for the API server. "
        "Install with: pip install synthquant[api]"
    ) from e

from synthquant import __version__
from synthquant.analytics.risk_metrics import (
    conditional_var,
    max_drawdown_distribution,
    tail_ratio,
    value_at_risk,
)
from synthquant.api.schemas import (
    ForecastResponse,
    HealthResponse,
    RegimeResponse,
    RiskMetrics,
    SimulationRequest,
)
from synthquant.simulation.engines.gbm import GBMEngine
from synthquant.simulation.engines.heston import HestonEngine
from synthquant.simulation.engines.merton_jd import MertonJDEngine

app = FastAPI(
    title="SynthQuant API",
    description="Production-grade synthetic financial data generation and probabilistic forecasting",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ENGINES = {
    "gbm": GBMEngine(),
    "heston": HestonEngine(),
    "merton_jd": MertonJDEngine(),
}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Liveness check endpoint."""
    return HealthResponse(status="ok", version=__version__)


@app.post("/simulate", response_model=ForecastResponse, tags=["Simulation"])
async def simulate(request: SimulationRequest) -> ForecastResponse:
    """Run a Monte Carlo simulation and return probabilistic forecast statistics.

    Args:
        request: Simulation parameters.

    Returns:
        ForecastResponse with mean, std, VaR, CVaR, and percentiles.
    """
    engine = _ENGINES.get(request.model)
    if engine is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{request.model}'. Choose from: {list(_ENGINES.keys())}",
        )

    dt = request.horizon / request.n_steps
    params = {"S0": 100.0, "mu": 0.05, "sigma": 0.20}
    params.update(request.parameters)
    params["random_state"] = request.random_seed

    try:
        paths = engine.simulate(
            n_paths=request.n_paths,
            n_steps=request.n_steps,
            dt=dt,
            **params,
        )
    except Exception as exc:
        logger.error(f"Simulation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    S0 = paths[:, 0]
    terminal = paths[:, -1]
    log_returns = np.log(terminal / S0)

    percentile_keys = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentiles = {
        f"p{p}": float(np.percentile(log_returns, p)) for p in percentile_keys
    }

    return ForecastResponse(
        symbol=request.symbol,
        model=request.model,
        horizon=request.horizon,
        n_paths=request.n_paths,
        mean=float(np.mean(log_returns)),
        std=float(np.std(log_returns)),
        var_95=value_at_risk(paths, alpha=0.05),
        cvar_95=conditional_var(paths, alpha=0.05),
        percentiles=percentiles,
    )


@app.get("/regimes/{symbol}", response_model=RegimeResponse, tags=["Regime Detection"])
async def detect_regimes(
    symbol: str,
    n_regimes: int = 2,
    lookback: int = 500,
) -> RegimeResponse:
    """Detect market regimes for a symbol using HMM.

    This endpoint uses synthetic GBM data for demonstration.
    In production, replace with real market data via DataIngestor.

    Args:
        symbol: Ticker symbol.
        n_regimes: Number of regimes to detect.
        lookback: Number of periods to use.

    Returns:
        RegimeResponse with current regime and probabilities.
    """
    from synthquant.regime.hmm import HMMRegimeDetector

    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.01, lookback)

    detector = HMMRegimeDetector(n_components=n_regimes, random_state=42)
    detector.fit(returns)
    proba = detector.predict_proba(returns)
    current_regime = int(np.argmax(proba[-1]))
    params = detector.get_regime_params()
    regime_params_str = {str(k): v for k, v in params.items()}

    return RegimeResponse(
        symbol=symbol,
        current_regime=current_regime,
        regime_proba=proba[-1].tolist(),
        regime_params=regime_params_str,
        n_regimes=n_regimes,
    )


@app.post("/forecast", response_model=ForecastResponse, tags=["Simulation"])
async def forecast(request: SimulationRequest) -> ForecastResponse:
    """Alias for /simulate with semantic forecast framing."""
    return await simulate(request)


@app.get("/risk/{symbol}", response_model=RiskMetrics, tags=["Analytics"])
async def risk(
    symbol: str,
    n_paths: int = 10_000,
    horizon: float = 1.0,
    mu: float = 0.07,
    sigma: float = 0.20,
) -> RiskMetrics:
    """Compute risk metrics for a symbol via GBM simulation.

    Args:
        symbol: Ticker symbol.
        n_paths: Number of simulation paths.
        horizon: Time horizon in years.
        mu: Asset drift.
        sigma: Asset volatility.

    Returns:
        RiskMetrics with VaR, CVaR, drawdown, tail ratio, and Sharpe.
    """
    dt = horizon / 252
    paths = GBMEngine().simulate(
        n_paths=n_paths,
        n_steps=252,
        dt=dt,
        S0=100.0,
        mu=mu,
        sigma=sigma,
        random_state=42,
    )

    S0 = paths[:, 0]
    terminal = paths[:, -1]
    log_returns = np.log(terminal / S0)
    drawdowns = max_drawdown_distribution(paths)
    sharpe = float(np.mean(log_returns) / (np.std(log_returns) + 1e-12) * np.sqrt(252 / paths.shape[1]))

    return RiskMetrics(
        symbol=symbol,
        var_95=value_at_risk(paths, alpha=0.05),
        cvar_95=conditional_var(paths, alpha=0.05),
        max_drawdown_mean=float(np.mean(drawdowns)),
        tail_ratio=tail_ratio(paths, alpha=0.05),
        sharpe_estimate=sharpe,
    )
