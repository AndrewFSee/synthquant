# SynthQuant Architecture

## Overview

SynthQuant is structured as a layered pipeline: raw market data flows upward through regime detection and model calibration into Monte Carlo simulation, then into analytics and strategy components.

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

## Module Descriptions

### `synthquant.data`

The data layer handles all market data ingestion, feature engineering, and local storage.

- **`DataSource`** (abstract): Interface for any market data provider. Implementations must return a normalized OHLCV `DataFrame` with a `DatetimeIndex`.
- **`YFinanceSource`**: Concrete implementation using the `yfinance` library.
- **`DataIngestor`**: Orchestrates multiple sources with fallback logic.
- **`FeatureEngine`**: Computes realized volatility (close-to-close, Parkinson, Garman-Klass, Yang-Zhang), RSI, MACD, Bollinger Bands, and rolling higher moments.
- **`ParquetStorage`**: Persists DataFrames to local Parquet files with symbol/date partitioning.

### `synthquant.regime`

Regime detection algorithms identify distinct market states (e.g., bull/bear, low/high volatility).

- **`HMMRegimeDetector`**: Wraps `hmmlearn.GaussianHMM`. Fits on a return series and returns regime labels and posterior probabilities.
- **`MarkovSwitchingGARCH`**: Uses `statsmodels` Markov Regression to model regime-conditional GARCH dynamics.
- **`ClusteringRegimeDetector`**: Uses `sklearn.mixture.GaussianMixture` on feature vectors.
- **`EnsembleRegimeDetector`**: Combines multiple detectors via majority voting or probability averaging.

### `synthquant.models`

Parametric stochastic models for volatility and returns.

- **`GARCHModel`**: Wraps the `arch` library supporting GARCH, EGARCH, GJR-GARCH.
- **`HestonModel`**: Stochastic volatility with mean-reverting variance (Euler-Maruyama simulation).
- **`MertonJumpDiffusion`**: GBM plus compound Poisson jumps; includes closed-form European option pricing.
- **`KouModel`**: Double-exponential jump distribution variant.
- **`RoughBergomi`**: Fractional Brownian motion-driven volatility with Hurst exponent H < 0.5.
- **`MLECalibrator`**: Fits model parameters via `scipy.optimize`.
- **`BayesianCalibrator`**: Fits parameters via PyMC posterior sampling (optional dependency).

### `synthquant.simulation`

Monte Carlo engines and tools for path generation.

- **`SimulationEngine`** (abstract): All engines expose `simulate(n_paths, n_steps, dt, **params) → np.ndarray` returning shape `(n_paths, n_steps+1)`.
- **`GBMEngine`**: Exact GBM solution via log-normal increments.
- **`MertonJDEngine`**: GBM + Poisson jumps.
- **`HestonEngine`**: QE scheme for stochastic variance.
- **`RegimeSwitchingEngine`**: Markov chain over a dict of per-regime parameter sets.
- **`RoughBergomiEngine`**: Hybrid scheme for fractional Brownian motion.
- **`CopulaSimulator`**: Multi-asset copula sampling (Gaussian, Student-t, Clayton).
- **`GPUEngine`**: JAX-accelerated wrapper with NumPy fallback.
- **Variance reduction**: `antithetic_variates`, `control_variates`, `stratified_sampling`, `importance_sampling`.

### `synthquant.analytics`

Downstream risk and forecast analytics on simulated path arrays.

- **`value_at_risk` / `conditional_var` / `expected_shortfall`**: Standard quantile-based risk measures.
- **`max_drawdown_distribution`**: Empirical distribution of max drawdowns across paths.
- **`MCOptionPricer`**: European, Asian, and barrier option pricing from simulated paths; finite-difference Greeks.
- **`ImpliedVolSurface`**: Newton-Raphson inversion to recover implied vols from MC prices.
- **`EmpiricalDistribution`**: KDE-smoothed CDF/PDF/quantile estimation.
- **`ForecastScorer`**: CRPS, PIT histogram, Brier score, coverage test, KS test.

### `synthquant.strategy`

Portfolio construction and trade execution utilities.

- **`KellyCriterion`**: Full and fractional Kelly position sizing.
- **`RiskParitySizer`**: Equal-risk-contribution weighting.
- **`CVaROptimalSizer`**: CVaR-constrained position sizing.
- **`RegimeSignalGenerator`**: Generates entry/exit signals conditioned on detected regime.
- **`DeltaHedger` / `GammaHedger`**: Compute hedge ratios from simulated paths.
- **`MeanCVaROptimizer`**: Mean-CVaR portfolio optimization via `scipy`.
- **`RiskParityAllocator`**: Covariance-based risk parity.
- **`WalkForwardBacktest`**: Expanding/rolling window backtesting with Sharpe, Sortino, Calmar metrics.

### `synthquant.api`

Optional FastAPI server exposing all core functionality as REST and WebSocket endpoints.

- `GET /health` — liveness check
- `POST /simulate` — run a Monte Carlo simulation
- `GET /regimes/{symbol}` — detect current regime
- `POST /forecast` — generate probabilistic forecast
- `GET /risk/{symbol}` — compute risk metrics
- WebSocket `/ws/regimes` — streaming regime updates

### `dashboard`

Streamlit-based interactive dashboard with Plotly charts for fan charts, regime timelines, volatility surfaces, drawdown distributions, and PIT histograms.

---

## Data Flow

```
Market Data
    │
    ▼
DataIngestor ──► FeatureEngine ──► ParquetStorage
    │
    ▼
RegimeDetector (HMM / MS-GARCH / GMM / Ensemble)
    │
    ├──► Regime Labels & Probabilities
    │
    ▼
Model Calibration (MLE / Bayesian)
    │
    ├──► Calibrated Parameters per Regime
    │
    ▼
SimulationEngine (GBM / Heston / Merton / Regime-Switching / Rough Bergomi)
    │
    ├──► Paths: (n_paths, n_steps+1)
    │
    ▼
Analytics (VaR / CVaR / Options / Scoring)
    │
    ▼
Strategy (Sizing / Signals / Backtest / Hedging)
    │
    ▼
API / Dashboard
```
