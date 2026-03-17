"""Plotly figure generators for the SynthQuant dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    raise ImportError(
        "Plotly is required for dashboard plots. "
        "Install with: pip install synthquant[dashboard]"
    ) from e

__all__ = [
    "plot_fan_chart",
    "plot_regime_timeline",
    "plot_vol_surface",
    "plot_drawdown_distribution",
    "plot_pit_histogram",
]


def plot_fan_chart(
    paths: np.ndarray,
    dates: pd.DatetimeIndex,
    title: str = "Monte Carlo Fan Chart",
    percentiles: list[tuple[float, float]] | None = None,
    n_sample_paths: int = 50,
) -> go.Figure:
    """Create a fan chart showing percentile bands of simulated paths.

    Args:
        paths: Array of shape (n_paths, n_steps+1).
        dates: DatetimeIndex of length n_steps+1.
        title: Chart title.
        percentiles: List of (lower, upper) percentile pairs for shading.
            Defaults to [(5,95), (10,90), (25,75)].
        n_sample_paths: Number of individual paths to overlay.

    Returns:
        Plotly Figure.
    """
    if percentiles is None:
        percentiles = [(5, 95), (10, 90), (25, 75)]

    fig = go.Figure()
    colors = ["rgba(68, 114, 196, 0.15)", "rgba(68, 114, 196, 0.20)", "rgba(68, 114, 196, 0.25)"]

    for (lo, hi), color in zip(percentiles, colors):
        lower = np.percentile(paths, lo, axis=0)
        upper = np.percentile(paths, hi, axis=0)
        fig.add_trace(
            go.Scatter(
                x=list(dates) + list(dates[::-1]),
                y=list(upper) + list(lower[::-1]),
                fill="toself",
                fillcolor=color,
                line={"color": "rgba(0,0,0,0)"},
                name=f"P{lo}–P{hi}",
                showlegend=True,
            )
        )

    # Median
    median = np.median(paths, axis=0)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=median,
            line={"color": "rgb(68, 114, 196)", "width": 2},
            name="Median",
        )
    )

    # Sample paths
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(paths.shape[0], size=min(n_sample_paths, paths.shape[0]), replace=False)
    for i in sample_idx:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=paths[i],
                line={"color": "rgba(150,150,150,0.2)", "width": 0.5},
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def plot_regime_timeline(
    dates: pd.DatetimeIndex,
    regimes: np.ndarray,
    prices: np.ndarray,
    title: str = "Regime Timeline",
) -> go.Figure:
    """Plot price series with regime-shaded background.

    Args:
        dates: DatetimeIndex of length n.
        regimes: Integer array of regime labels, shape (n,).
        prices: Price series, shape (n,).
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    palette = [
        "rgba(255, 99, 71, 0.25)",
        "rgba(30, 144, 255, 0.25)",
        "rgba(50, 205, 50, 0.25)",
        "rgba(255, 165, 0, 0.25)",
    ]

    fig = go.Figure()
    n_regimes = int(np.max(regimes)) + 1

    # Shaded regime regions
    for regime_id in range(n_regimes):
        mask = regimes == regime_id
        color = palette[regime_id % len(palette)]
        in_regime = False
        start_idx = 0
        for i, in_r in enumerate(mask):
            if in_r and not in_regime:
                start_idx = i
                in_regime = True
            elif not in_r and in_regime:
                fig.add_vrect(
                    x0=dates[start_idx],
                    x1=dates[i - 1],
                    fillcolor=color,
                    opacity=1.0,
                    layer="below",
                    line_width=0,
                )
                in_regime = False
        if in_regime:
            fig.add_vrect(
                x0=dates[start_idx],
                x1=dates[-1],
                fillcolor=color,
                opacity=1.0,
                layer="below",
                line_width=0,
                annotation_text=f"Regime {regime_id}",
            )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=prices,
            line={"color": "black", "width": 1.5},
            name="Price",
        )
    )
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", template="plotly_white")
    return fig


def plot_vol_surface(
    strikes: np.ndarray,
    maturities: np.ndarray,
    vols: np.ndarray,
    title: str = "Implied Volatility Surface",
) -> go.Figure:
    """Create a 3D implied volatility surface.

    Args:
        strikes: Array of shape (n_strikes,).
        maturities: Array of shape (n_maturities,).
        vols: Array of shape (n_maturities, n_strikes).
        title: Chart title.

    Returns:
        Plotly Figure with 3D surface.
    """
    fig = go.Figure(
        data=[
            go.Surface(
                x=strikes,
                y=maturities,
                z=vols,
                colorscale="Viridis",
                colorbar={"title": "IV"},
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "Strike",
            "yaxis_title": "Maturity (years)",
            "zaxis_title": "Implied Vol",
        },
        template="plotly_white",
    )
    return fig


def plot_drawdown_distribution(
    max_drawdowns: np.ndarray,
    title: str = "Max Drawdown Distribution",
    n_bins: int = 50,
) -> go.Figure:
    """Histogram of maximum drawdowns across simulated paths.

    Args:
        max_drawdowns: Array of shape (n_paths,) with max drawdown values.
        title: Chart title.
        n_bins: Number of histogram bins.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure(
        data=[
            go.Histogram(
                x=max_drawdowns * 100,
                nbinsx=n_bins,
                marker_color="rgba(68, 114, 196, 0.7)",
                name="Max Drawdown",
            )
        ]
    )
    mean_dd = float(np.mean(max_drawdowns) * 100)
    fig.add_vline(
        x=mean_dd,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_dd:.1f}%",
    )
    fig.update_layout(
        title=title,
        xaxis_title="Max Drawdown (%)",
        yaxis_title="Frequency",
        template="plotly_white",
    )
    return fig


def plot_pit_histogram(
    pit_values: np.ndarray,
    title: str = "PIT Histogram (Calibration Check)",
    n_bins: int = 10,
) -> go.Figure:
    """Probability Integral Transform (PIT) histogram for forecast calibration.

    A uniform histogram indicates a well-calibrated forecast.

    Args:
        pit_values: Array of PIT values in [0, 1].
        title: Chart title.
        n_bins: Number of bins.

    Returns:
        Plotly Figure.
    """
    counts, edges = np.histogram(pit_values, bins=n_bins, range=(0, 1))
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    uniform_level = len(pit_values) / n_bins

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=counts,
            width=1.0 / n_bins * 0.9,
            marker_color="rgba(68, 114, 196, 0.7)",
            name="PIT counts",
        )
    )
    fig.add_hline(
        y=uniform_level,
        line_dash="dash",
        line_color="red",
        annotation_text="Uniform (ideal)",
    )
    fig.update_layout(
        title=title,
        xaxis_title="PIT Value",
        yaxis_title="Count",
        template="plotly_white",
    )
    return fig
