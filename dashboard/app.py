"""SynthQuant Streamlit Dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError as e:
    raise ImportError(
        "Streamlit is required for the dashboard. "
        "Install with: pip install synthquant[dashboard]"
    ) from e

from dashboard.plots import (
    plot_drawdown_distribution,
    plot_fan_chart,
    plot_regime_timeline,
)
from synthquant.analytics.risk_metrics import (
    conditional_var,
    max_drawdown_distribution,
    value_at_risk,
)
from synthquant.regime.hmm import HMMRegimeDetector
from synthquant.simulation.engines.gbm import GBMEngine
from synthquant.simulation.engines.heston import HestonEngine


def main() -> None:
    """Entry point for the Streamlit dashboard."""
    st.set_page_config(
        page_title="SynthQuant Dashboard",
        page_icon="📈",
        layout="wide",
    )
    st.title("📈 SynthQuant Engine — Interactive Dashboard")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")
        symbol = st.text_input("Symbol", value="SPY")
        model_choice = st.selectbox("Model", ["GBM", "Heston"])
        n_paths = st.slider("Number of Paths", 100, 50_000, 5_000, step=500)
        horizon_years = st.slider("Horizon (years)", 0.25, 5.0, 1.0, step=0.25)
        n_steps = st.slider("Time Steps", 21, 504, 252, step=21)
        st.divider()
        st.subheader("GBM / Heston Parameters")
        S0 = st.number_input("Initial Price (S0)", value=100.0, min_value=1.0)
        mu = st.slider("Drift (mu)", -0.20, 0.30, 0.07, step=0.01)
        sigma = st.slider("Volatility (sigma)", 0.01, 0.80, 0.20, step=0.01)
        if model_choice == "Heston":
            v0 = st.slider("Initial Variance (v0)", 0.001, 0.25, 0.04, step=0.001)
            kappa = st.slider("Mean Reversion (kappa)", 0.1, 10.0, 2.0, step=0.1)
            theta = st.slider("Long-run Variance (theta)", 0.001, 0.25, 0.04, step=0.001)
            sigma_v = st.slider("Vol of Vol (sigma_v)", 0.01, 1.0, 0.3, step=0.01)
            rho = st.slider("Correlation (rho)", -0.99, 0.99, -0.7, step=0.01)
        n_regimes = st.slider("HMM Regimes", 2, 4, 2)
        random_seed = st.number_input("Random Seed", value=42, min_value=0)
        run_button = st.button("🚀 Run Simulation", type="primary")

    if not run_button:
        st.info("Configure parameters in the sidebar and click **Run Simulation**.")
        return

    # ── Simulation ─────────────────────────────────────────────────────────────
    dt = horizon_years / n_steps
    with st.spinner("Running Monte Carlo simulation…"):
        if model_choice == "GBM":
            engine = GBMEngine()
            paths = engine.simulate(
                n_paths=n_paths,
                n_steps=n_steps,
                dt=dt,
                S0=S0,
                mu=mu,
                sigma=sigma,
                random_state=random_seed,
            )
        else:
            engine = HestonEngine()
            paths = engine.simulate(
                n_paths=n_paths,
                n_steps=n_steps,
                dt=dt,
                S0=S0,
                v0=v0,
                mu=mu,
                kappa=kappa,
                theta=theta,
                sigma_v=sigma_v,
                rho=rho,
                random_state=random_seed,
            )

    dates = pd.date_range("today", periods=n_steps + 1, freq="B")

    # ── Regime Detection ───────────────────────────────────────────────────────
    log_returns = np.log(paths[:, 1:] / paths[:, :-1])
    median_returns = np.median(log_returns, axis=0)
    detector = HMMRegimeDetector(n_components=n_regimes, random_state=42)
    detector.fit(median_returns)
    regimes = detector.predict(median_returns)

    # ── Risk Metrics ───────────────────────────────────────────────────────────
    var_95 = value_at_risk(paths, alpha=0.05)
    cvar_95 = conditional_var(paths, alpha=0.05)
    drawdowns = max_drawdown_distribution(paths)

    # ── Layout ─────────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("95% VaR", f"{var_95:.2%}")
    col2.metric("95% CVaR", f"{cvar_95:.2%}")
    col3.metric("Mean Max Drawdown", f"{drawdowns.mean():.2%}")
    col4.metric("Paths", f"{n_paths:,}")

    st.subheader(f"📊 {model_choice} Fan Chart — {symbol}")
    fan_fig = plot_fan_chart(paths, dates, title=f"{symbol} {model_choice} Simulation")
    st.plotly_chart(fan_fig, use_container_width=True)

    st.subheader("🔍 Regime Timeline")
    regime_fig = plot_regime_timeline(dates[1:], regimes, np.median(paths, axis=0)[1:])
    st.plotly_chart(regime_fig, use_container_width=True)

    st.subheader("📉 Max Drawdown Distribution")
    dd_fig = plot_drawdown_distribution(drawdowns)
    st.plotly_chart(dd_fig, use_container_width=True)


if __name__ == "__main__":
    main()
