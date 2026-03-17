# Getting Started with SynthQuant

## Installation

### Prerequisites

- Python 3.10 or newer
- pip 23+

### Basic Install

```bash
pip install synthquant
```

### With Optional Dependencies

```bash
# Market data via yfinance
pip install "synthquant[data]"

# Bayesian calibration via PyMC
pip install "synthquant[bayesian]"

# REST API server
pip install "synthquant[api]"

# Interactive dashboard
pip install "synthquant[dashboard]"

# Everything
pip install "synthquant[all]"
```

### From Source (Development)

```bash
git clone https://github.com/AndrewFSee/synthquant.git
cd synthquant
pip install -e ".[dev]"
pre-commit install
```

---

## Configuration

SynthQuant uses `pydantic-settings` for configuration. All settings can be set via:

1. Environment variables prefixed with `SYNTHQUANT_`
2. A `.env` file in the working directory

```bash
# .env
SYNTHQUANT_N_PATHS=50000
SYNTHQUANT_RANDOM_SEED=42
SYNTHQUANT_LOG_LEVEL=DEBUG
SYNTHQUANT_DATA_CACHE_DIR=/tmp/sq_cache
```

```python
from synthquant.config import get_settings

settings = get_settings()
print(settings.n_paths)  # 50000
```

---

## Quickstart Tutorial

### Step 1: Fetch Market Data

```python
from synthquant.data.ingest import DataIngestor

ingestor = DataIngestor()
spy = ingestor.fetch("SPY", start="2020-01-01", end="2023-12-31")
print(spy.tail())
#              open    high     low   close    volume
# 2023-12-27  476.0  476.9  474.1  475.9  52314200
```

### Step 2: Compute Features

```python
from synthquant.data.features import FeatureEngine

fe = FeatureEngine()
returns = fe.rolling_returns(spy["close"], window=1)
rv = fe.realized_volatility(spy, window=21, estimator="yang_zhang")
rsi = fe.rsi(spy["close"], period=14)
```

### Step 3: Detect Regimes

```python
from synthquant.regime.hmm import HMMRegimeDetector

detector = HMMRegimeDetector(n_components=2, random_state=42)
detector.fit(returns.dropna().values)

regimes = detector.predict(returns.dropna().values)
proba = detector.predict_proba(returns.dropna().values)
params = detector.get_regime_params()

print(f"Regime 0: mu={params[0]['mu']:.6f}, sigma={params[0]['sigma']:.4f}")
print(f"Regime 1: mu={params[1]['mu']:.6f}, sigma={params[1]['sigma']:.4f}")
```

### Step 4: Run Monte Carlo Simulation

```python
from synthquant.simulation.engines.gbm import GBMEngine

engine = GBMEngine()
paths = engine.simulate(
    n_paths=10_000,
    n_steps=252,
    dt=1/252,
    S0=spy["close"].iloc[-1],
    mu=0.07,
    sigma=0.20,
    random_state=42,
)
print(f"Paths shape: {paths.shape}")  # (10000, 253)
```

### Step 5: Risk Analytics

```python
from synthquant.analytics.risk_metrics import (
    value_at_risk,
    expected_shortfall,
    max_drawdown_distribution,
)

var_95 = value_at_risk(paths, alpha=0.05)
es_95 = expected_shortfall(paths, alpha=0.05)
drawdowns = max_drawdown_distribution(paths)

print(f"1-year 95% VaR:          {var_95:.4f}")
print(f"1-year 95% ES:           {es_95:.4f}")
print(f"Median max drawdown:     {drawdowns.mean():.4f}")
```

### Step 6: Option Pricing

```python
from synthquant.analytics.options import MCOptionPricer

pricer = MCOptionPricer()
call_price = pricer.price_european(
    paths=paths,
    K=spy["close"].iloc[-1] * 1.05,  # 5% OTM call
    T=1.0,
    r=0.05,
    option_type="call",
)
print(f"MC Call Price: {call_price:.4f}")
```

### Step 7: Position Sizing

```python
from synthquant.strategy.sizing import KellyCriterion, RiskParitySizer
import numpy as np

# Simulate daily returns for two assets
ret1 = np.random.normal(0.0005, 0.01, 252)
ret2 = np.random.normal(0.0003, 0.008, 252)

kelly = KellyCriterion()
size_1 = kelly.full_kelly(ret1)
size_half = kelly.fractional_kelly(ret1, fraction=0.5)
print(f"Full Kelly: {size_1:.4f}, Half Kelly: {size_half:.4f}")

rps = RiskParitySizer()
weights = rps.compute_weights(np.column_stack([ret1, ret2]), target_vol=0.10)
print(f"Risk Parity Weights: {weights}")
```

---

## Running the API Server

```bash
pip install "synthquant[api]"
uvicorn synthquant.api.server:app --host 0.0.0.0 --port 8000 --reload
```

Then visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Running the Dashboard

```bash
pip install "synthquant[dashboard]"
streamlit run dashboard/app.py
```

---

## Running Tests

```bash
# Fast tests only
pytest -m "not slow and not integration"

# All tests
pytest

# With coverage report
pytest --cov=synthquant --cov-report=html
```

---

## Next Steps

- See `examples/` for full working scripts
- Read `docs/architecture.md` for a deep dive into each module
- Check `synthquant/config.py` for all configurable settings
