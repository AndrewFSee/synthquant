"""Tests for GARCH model."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.models.garch import GARCHModel


@pytest.fixture()
def fitted_garch(sample_returns: np.ndarray) -> GARCHModel:
    """Fitted GARCH(1,1) model."""
    model = GARCHModel(vol_model="GARCH", p=1, q=1)
    model.fit(sample_returns)
    return model


def test_fit_returns_self(sample_returns: np.ndarray) -> None:
    """fit() returns self for method chaining."""
    model = GARCHModel()
    result = model.fit(sample_returns)
    assert result is model


def test_forecast_returns_positive_vols(fitted_garch: GARCHModel) -> None:
    """forecast() returns positive volatility values."""
    vols = fitted_garch.forecast(horizon=10)
    assert len(vols) == 10
    assert np.all(vols > 0)


def test_forecast_before_fit_raises() -> None:
    """forecast() raises RuntimeError before fitting."""
    model = GARCHModel()
    with pytest.raises(RuntimeError, match="fitted"):
        model.forecast(horizon=5)


def test_simulate_correct_shape(fitted_garch: GARCHModel) -> None:
    """simulate() returns array of shape (n_paths, horizon)."""
    paths = fitted_garch.simulate(n_paths=50, horizon=21, random_state=42)
    assert paths.shape == (50, 21)


def test_simulate_before_fit_raises() -> None:
    """simulate() raises RuntimeError before fitting."""
    model = GARCHModel()
    with pytest.raises(RuntimeError, match="fitted"):
        model.simulate(n_paths=10, horizon=5)


@pytest.mark.parametrize("vol_model", ["GARCH", "EGARCH", "GJRGARCH"])
def test_vol_model_variants(sample_returns: np.ndarray, vol_model: str) -> None:
    """All three vol model variants fit without error."""
    model = GARCHModel(vol_model=vol_model)  # type: ignore[arg-type]
    model.fit(sample_returns)
    vols = model.forecast(horizon=1)  # Use horizon=1 for EGARCH analytic forecast
    assert len(vols) == 1
    assert np.all(vols > 0)
