"""Tests for API request/response Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synthquant.api.schemas import (
    ForecastResponse,
    HealthResponse,
    RegimeResponse,
    RiskMetrics,
    SimulationRequest,
)


# ── SimulationRequest ─────────────────────────────────────────────────────────

def test_simulation_request_defaults() -> None:
    """SimulationRequest has sensible defaults."""
    req = SimulationRequest()
    assert req.symbol == "SPY"
    assert req.model == "gbm"
    assert req.n_paths == 10_000
    assert req.horizon == 1.0
    assert req.n_steps == 252
    assert req.parameters == {}
    assert req.random_seed is None


def test_simulation_request_custom_values() -> None:
    """SimulationRequest accepts custom values."""
    req = SimulationRequest(
        symbol="AAPL",
        model="heston",
        n_paths=500,
        horizon=0.5,
        n_steps=126,
        parameters={"v0": 0.04},
        random_seed=42,
    )
    assert req.symbol == "AAPL"
    assert req.model == "heston"
    assert req.n_paths == 500
    assert req.parameters == {"v0": 0.04}
    assert req.random_seed == 42


def test_simulation_request_n_paths_too_small_raises() -> None:
    """n_paths below minimum raises ValidationError."""
    with pytest.raises(ValidationError):
        SimulationRequest(n_paths=10)  # min is 100


def test_simulation_request_n_paths_too_large_raises() -> None:
    """n_paths above maximum raises ValidationError."""
    with pytest.raises(ValidationError):
        SimulationRequest(n_paths=1_000_000)  # max is 500_000


def test_simulation_request_non_positive_horizon_raises() -> None:
    """Non-positive horizon raises ValidationError."""
    with pytest.raises(ValidationError):
        SimulationRequest(horizon=0.0)


def test_simulation_request_n_steps_zero_raises() -> None:
    """n_steps of zero raises ValidationError."""
    with pytest.raises(ValidationError):
        SimulationRequest(n_steps=0)


# ── ForecastResponse ──────────────────────────────────────────────────────────

def test_forecast_response_construction() -> None:
    """ForecastResponse can be constructed with all required fields."""
    resp = ForecastResponse(
        symbol="SPY",
        model="gbm",
        horizon=1.0,
        n_paths=1000,
        mean=0.07,
        std=0.20,
        var_95=-0.25,
        cvar_95=-0.35,
        percentiles={"p5": -0.25, "p50": 0.08, "p95": 0.40},
    )
    assert resp.symbol == "SPY"
    assert resp.mean == 0.07
    assert resp.percentiles["p50"] == 0.08


def test_forecast_response_serializes_to_dict() -> None:
    """ForecastResponse serializes to dict via model_dump()."""
    resp = ForecastResponse(
        symbol="TSLA",
        model="heston",
        horizon=0.5,
        n_paths=500,
        mean=0.05,
        std=0.30,
        var_95=-0.40,
        cvar_95=-0.50,
        percentiles={},
    )
    d = resp.model_dump()
    assert d["symbol"] == "TSLA"
    assert d["model"] == "heston"


# ── RegimeResponse ────────────────────────────────────────────────────────────

def test_regime_response_construction() -> None:
    """RegimeResponse can be constructed with all required fields."""
    resp = RegimeResponse(
        symbol="SPY",
        current_regime=0,
        regime_proba=[0.8, 0.2],
        regime_params={"0": {"mu": 0.001, "sigma": 0.01}},
        n_regimes=2,
    )
    assert resp.current_regime == 0
    assert resp.n_regimes == 2
    assert resp.regime_proba == [0.8, 0.2]


# ── RiskMetrics ───────────────────────────────────────────────────────────────

def test_risk_metrics_construction() -> None:
    """RiskMetrics can be constructed with all required fields."""
    metrics = RiskMetrics(
        symbol="AMZN",
        var_95=-0.30,
        cvar_95=-0.45,
        max_drawdown_mean=-0.20,
        tail_ratio=1.5,
        sharpe_estimate=0.8,
    )
    assert metrics.symbol == "AMZN"
    assert metrics.var_95 == -0.30


# ── HealthResponse ────────────────────────────────────────────────────────────

def test_health_response_default_status() -> None:
    """HealthResponse has default status 'ok'."""
    resp = HealthResponse(version="1.0.0")
    assert resp.status == "ok"
    assert resp.version == "1.0.0"


def test_health_response_custom_status() -> None:
    """HealthResponse accepts custom status."""
    resp = HealthResponse(status="degraded", version="0.9.0")
    assert resp.status == "degraded"
