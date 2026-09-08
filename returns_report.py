"""Unified returns analysis: single-path view and Monte Carlo simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cumulative_returns import (
    TRADING_DAYS,
    DataSource,
    ReturnSeries,
    build_hover_data,
    cumulative_returns,
    export_html,
    fetch_ticker_returns,
    simulated_returns,
)
from monte_carlo_simulator import (
    MonteCarloResult,
    build_summary_table,
    run_monte_carlo,
)


@dataclass
class ReportConfig:
    use_historical: bool = False
    ticker: str = "SPY"
    annual_drift: float = 0.08
    annual_volatility: float = 0.20
    n_trials: int = 1000
    n_days: int = 1000
    random_seed: int = 42
    show_mc_trials: bool = True
    html_output: Path = Path("returns_report.html")


@dataclass
class ReportData:
    annual_drift: float
    annual_volatility: float
    single_path: ReturnSeries
    tripled_returns: np.ndarray
    monte_carlo: MonteCarloResult
    mc_summary: dict[str, float]


def estimate_annual_params(daily_returns: np.ndarray) -> tuple[float, float]:
    """Estimate annualized drift and volatility from a daily simple-return sample."""
    log_returns = np.log1p(daily_returns)
    annual_drift = float(np.mean(log_returns) * TRADING_DAYS)
    annual_volatility = float(np.std(log_returns, ddof=1) * np.sqrt(TRADING_DAYS))
    return annual_drift, annual_volatility


def resolve_parameters(config: ReportConfig) -> tuple[float, float, ReturnSeries | None]:
    """Return drift, volatility, and optional historical series for the single-path chart."""
    if config.use_historical:
        historical = fetch_ticker_returns(config.ticker, n_days=config.n_days)
        drift, volatility = estimate_annual_params(historical.daily_returns)
        historical.annual_drift = drift
        historical.annual_volatility = volatility
        return drift, volatility, historical

    return config.annual_drift, config.annual_volatility, None


def build_report(config: ReportConfig, rng: np.random.Generator | None = None) -> ReportData:
    rng = rng or np.random.default_rng(config.random_seed)
    drift, volatility, historical = resolve_parameters(config)

    if historical is not None:
        single_path = historical
        base_trace_source = historical
    else:
        single_path = simulated_returns(drift, volatility, config.n_days, rng)
        base_trace_source = single_path

    tripled_returns = 3.0 * single_path.daily_returns
    monte_carlo = run_monte_carlo(
        drift, volatility, n_trials=config.n_trials, n_days=config.n_days, rng=rng
    )
    mc_summary = build_summary_table(monte_carlo)

    return ReportData(
        annual_drift=drift,
        annual_volatility=volatility,
        single_path=base_trace_source,
        tripled_returns=tripled_returns,
        monte_carlo=monte_carlo,
        mc_summary=mc_summary,
    )


def _add_single_path_traces(
    fig: go.Figure,
    base: ReturnSeries,
    tripled_returns: np.ndarray,
    *,
    row: int,
    base_trace_name: str = "Base returns",
    tripled_trace_name: str = "3x daily returns",
) -> None:
    n_days = len(base.daily_returns)
    x_values = base.x_values if base.x_values is not None else np.arange(n_days)
    base_cumulative = cumulative_returns(base.daily_returns)
    tripled_cumulative = cumulative_returns(tripled_returns)
    base_hover = build_hover_data(base.daily_returns)
    tripled_hover = build_hover_data(tripled_returns)

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=base_cumulative,
            mode="lines",
            name=base_trace_name,
            customdata=base_hover,
            hovertemplate=(
                f"{base_trace_name}<br>"
                "Days looking back: %{customdata[0]}<br>"
                "Cumulative return: %{customdata[1]:.2%}<extra></extra>"
            ),
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=tripled_cumulative,
            mode="lines",
            name=tripled_trace_name,
            customdata=tripled_hover,
            hovertemplate=(
                f"{tripled_trace_name}<br>"
                "Days looking back: %{customdata[0]}<br>"
                "Cumulative return: %{customdata[1]:.2%}<extra></extra>"
            ),
        ),
        row=row,
        col=1,
    )


def _add_monte_carlo_traces(
    fig: go.Figure,
    result: MonteCarloResult,
    *,
    row: int,
    show_trials: bool,
) -> None:
    from monte_carlo_simulator import _add_trial_traces

    baseline_mean = result.baseline_cumulative.mean(axis=0)
    baseline_median = np.median(result.baseline_cumulative, axis=0)
    triple_mean = result.triple_cumulative.mean(axis=0)
    triple_median = np.median(result.triple_cumulative, axis=0)

    if show_trials:
        for trial in range(result.baseline_cumulative.shape[0]):
            scatter = go.Scattergl if result.n_trials > 100 else go.Scatter
            fig.add_trace(
                scatter(
                    x=result.days,
                    y=result.baseline_cumulative[trial],
                    mode="lines",
                    line=dict(color="rgba(31, 119, 180, 0.35)", width=0.5),
                    opacity=0.04,
                    legendgroup="Baseline trials",
                    showlegend=trial == 0,
                    name="Baseline trials",
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )
        for trial in range(result.triple_cumulative.shape[0]):
            scatter = go.Scattergl if result.n_trials > 100 else go.Scatter
            fig.add_trace(
                scatter(
                    x=result.days,
                    y=result.triple_cumulative[trial],
                    mode="lines",
                    line=dict(color="rgba(255, 127, 14, 0.35)", width=0.5),
                    opacity=0.04,
                    legendgroup="3x trials",
                    showlegend=trial == 0,
                    name="3x trials",
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )

    summary_traces = [
        (baseline_mean, "Baseline mean", "#1f77b4", "solid"),
        (baseline_median, "Baseline median", "#1f77b4", "dash"),
        (triple_mean, "3x mean", "#ff7f0e", "solid"),
        (triple_median, "3x median", "#ff7f0e", "dash"),
    ]
    for y, name, color, dash in summary_traces:
        fig.add_trace(
            go.Scatter(
                x=result.days,
                y=y,
                mode="lines",
                name=name,
                line=dict(color=color, width=3, dash=dash),
                hovertemplate=(
                    f"{name}<br>Day: %{{x}}<br>Cumulative return: %{{y:.2%}}<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )


def build_combined_figure(
    data: ReportData,
    *,
    show_mc_trials: bool = True,
) -> go.Figure:
    base = data.single_path
    param_note = (
        f"Calibrated from {base.ticker}" if base.source == DataSource.TICKER else "User inputs"
    )
    single_title = (
        f"Single Path — {base.ticker} ({len(base.daily_returns)} days)"
        if base.source == DataSource.TICKER
        else f"Single Path — Simulated ({len(base.daily_returns)} days)"
    )
    mc_title = f"Monte Carlo — {data.monte_carlo.n_trials:,} trials × {data.monte_carlo.n_days} days"

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(single_title, mc_title),
        vertical_spacing=0.10,
        row_heights=[0.38, 0.62],
    )

    base_name = f"{base.ticker} returns" if base.ticker else "Base returns"
    _add_single_path_traces(fig, base, data.tripled_returns, row=1, base_trace_name=base_name)
    _add_monte_carlo_traces(fig, data.monte_carlo, row=2, show_trials=show_mc_trials)

    xaxis_title = "Date" if base.source == DataSource.TICKER else "Day"
    fig.update_xaxes(title_text=xaxis_title, row=1, col=1)
    fig.update_xaxes(title_text="Day", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative return", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative return", tickformat=".0%", row=2, col=1)

    fig.update_layout(
        title=(
            f"Returns Report<br>"
            f"<sup>{param_note}: drift={data.annual_drift:.2%}, "
            f"volatility={data.annual_volatility:.2%}</sup>"
        ),
        height=900,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig


def export_report(fig: go.Figure, path: str | Path) -> Path:
    return export_html(fig, path)
