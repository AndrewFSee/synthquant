# SynthQuant Engine

![CI](https://github.com/AndrewFSee/synthquant/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyPI](https://img.shields.io/pypi/v/synthquant)

**SynthQuant** is a production-grade synthetic financial data generation and probabilistic forecasting platform. It combines regime detection, stochastic volatility models, Monte Carlo simulation, and risk analytics into a unified, extensible Python framework.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SynthQuant Engine                        │
├──────────────┬──────────────┬───────────────┬───────────────────┤
│  Data Layer  │ Regime Layer │  Model Layer  │ Simulation Layer  │
│              │              │               │                   │
│ • Ingest     │ • HMM        │ • GARCH       │ • GBM Engine      │
│ • Features   │ • MS-GARCH   │ • Heston      │ • Heston Engine   │
│ • Storage    │ • Clustering │ • Merton JD   │ • Merton JD Eng.  │
│   (Parquet)  │ • Ensemble   │ • Rough Vol   │ • Regime Switch.  │
│              │              │ • Calibration │ • Rough Bergomi   │
│              │              │               │ • Variance Reduc. │
│              │              │               │ • Copula Sim.     │
│              │              │               │ • GPU Engine      │
├──────────────┴──────────────┴───────────────┴───────────────────┤
│                       Analytics Layer                           │
│  • VaR / CVaR / ES   • Drawdown Distribution  • Tail Ratio     │
│  • MC Option Pricing • Implied Vol Surface    • Greeks          │
│  • CRPS / PIT / KS   • Rolling Moments        • Jarque-Bera    │
├──────────────────────────────┬──────────────────────────────────┤
│        Strategy Layer        │          API / Dashboard         │
│  • Kelly Criterion           │  • FastAPI REST + WebSocket      │
│  • Risk Parity               │  • Streamlit Dashboard           │
│  • CVaR-Optimal Sizing       │  • Plotly Visualizations         │
│  • Regime Signals            │                                  │
│  • Delta/Gamma Hedging       │                                  │
│  • Walk-Forward Backtest     │                                  │
└──────────────────────────────┴──────────────────────────────────┘
```

---

## Installation

```bash
# Core installation
pip install synthquant

# With data fetching (yfinance)
pip install "synthquant[data]"

# With Bayesian calibration (PyMC)
pip install "synthquant[bayesian]"

# With REST API (FastAPI)
pip install "synthquant[api]"

# With dashboard (Streamlit + Plotly)
pip install "synthquant[dashboard]"

# Everything
pip install "synthquant[all]"

# Development
pip install "synthquant[dev]"
```

---

## Quickstart

```python
import numpy as np
from synthquant.simulation.engines.gbm import GBMEngine
from synthquant.regime.hmm import HMMRegimeDetector
from synthquant.analytics.risk_metrics import value_at_risk, expected_shortfall

# 1. Simulate 10,000 GBM paths over 1 year (252 steps)
engine = GBMEngine()
paths = engine.simulate(
    n_paths=10_000,
    n_steps=252,
    dt=1/252,
    S0=100.0,
    mu=0.07,
    sigma=0.20,
    random_state=42,
)
print(f"Paths shape: {paths.shape}")  # (10000, 253)

# 2. Detect regimes on historical returns
returns = np.random.normal(0, 0.01, 500)  # replace with real data
detector = HMMRegimeDetector(n_components=2, random_state=42)
detector.fit(returns)
regimes = detector.predict(returns)
params = detector.get_regime_params()
print(f"Regime 0 vol: {params[0]['sigma']:.4f}")

# 3. Compute risk metrics on terminal prices
var_95 = value_at_risk(paths, alpha=0.05)
es_95 = expected_shortfall(paths, alpha=0.05)
print(f"95% VaR: {var_95:.4f}")
print(f"95% ES:  {es_95:.4f}")
```

---

## Module Overview

| Module | Description |
|--------|-------------|
| `synthquant.data` | Market data ingestion, feature engineering, Parquet storage |
| `synthquant.regime` | HMM, Markov-Switching GARCH, GMM clustering, ensemble detectors |
| `synthquant.models` | GARCH, Heston, Merton Jump-Diffusion, Rough Bergomi, calibration |
| `synthquant.simulation` | GBM, Heston, Merton JD, Regime-Switching, Rough Bergomi engines; variance reduction; copulas; GPU |
| `synthquant.analytics` | VaR, CVaR, drawdown, option pricing, implied vol, forecast scoring |
| `synthquant.strategy` | Kelly criterion, risk parity, CVaR sizing, regime signals, hedging, walk-forward backtest |
| `synthquant.api` | FastAPI REST + WebSocket server |
| `dashboard` | Streamlit interactive dashboard with Plotly charts |

---

## Examples

```bash
python examples/01_data_ingestion.py
python examples/02_regime_detection.py
python examples/03_volatility_modeling.py
python examples/04_monte_carlo_simulation.py
python examples/05_risk_analytics.py
python examples/06_full_pipeline.py
```

---

## Configuration

All settings can be overridden via environment variables prefixed with `SYNTHQUANT_`:

```bash
SYNTHQUANT_N_PATHS=50000 \
SYNTHQUANT_RANDOM_SEED=42 \
SYNTHQUANT_LOG_LEVEL=DEBUG \
python my_script.py
```

Or via a `.env` file in the project root.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest -m "not slow and not integration"`
5. Lint: `ruff check synthquant/ tests/`
6. Submit a pull request

---

## License

MIT License — see [LICENSE](LICENSE) for details.
