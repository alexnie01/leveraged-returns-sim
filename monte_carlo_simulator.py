"""Monte Carlo simulation of cumulative return paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from cumulative_returns import (
    DEFAULT_DAYS,
    export_html,
    lognormal_daily_returns,
    prompt_float,
)


@dataclass
class MonteCarloResult:
    baseline_cumulative: np.ndarray
    triple_cumulative: np.ndarray
    days: np.ndarray
    annual_drift: float
    annual_volatility: float
    n_trials: int
    n_days: int


def simulate_daily_returns(
    annual_drift: float,
    annual_volatility: float,
    n_trials: int,
    n_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return baseline daily simple returns with shape (n_trials, n_days)."""
    mu_daily = annual_drift / 252
    sigma_daily = annual_volatility / np.sqrt(252)
    log_returns = rng.normal(mu_daily, sigma_daily, size=(n_trials, n_days))
    return np.exp(log_returns) - 1.0


def cumulative_paths(daily_returns: np.ndarray) -> np.ndarray:
    """Convert daily returns (n_trials, n_days) to cumulative return paths."""
    return np.cumprod(1.0 + daily_returns, axis=1) - 1.0


def run_monte_carlo(
    annual_drift: float,
    annual_volatility: float,
    n_trials: int = 1000,
    n_days: int = DEFAULT_DAYS,
    rng: np.random.Generator | None = None,
) -> MonteCarloResult:
    rng = rng or np.random.default_rng()
    baseline_daily = simulate_daily_returns(
        annual_drift, annual_volatility, n_trials, n_days, rng
    )
    triple_daily = 3.0 * baseline_daily
    return MonteCarloResult(
        baseline_cumulative=cumulative_paths(baseline_daily),
        triple_cumulative=cumulative_paths(triple_daily),
        days=np.arange(n_days),
        annual_drift=annual_drift,
        annual_volatility=annual_volatility,
        n_trials=n_trials,
        n_days=n_days,
    )


def _add_trial_traces(
    fig: go.Figure,
    paths: np.ndarray,
    days: np.ndarray,
    *,
    color: str,
    legend_group: str,
    show_in_legend: bool,
) -> None:
    scatter = go.Scattergl if paths.shape[0] > 100 else go.Scatter
    for trial in range(paths.shape[0]):
        fig.add_trace(
            scatter(
                x=days,
                y=paths[trial],
                mode="lines",
                line=dict(color=color, width=0.5),
                opacity=0.04,
                legendgroup=legend_group,
                showlegend=show_in_legend and trial == 0,
                name=legend_group,
                hoverinfo="skip",
            )
        )


def build_monte_carlo_figure(
    result: MonteCarloResult,
    *,
    show_trials: bool = True,
) -> go.Figure:
    baseline_mean = result.baseline_cumulative.mean(axis=0)
    baseline_median = np.median(result.baseline_cumulative, axis=0)
    triple_mean = result.triple_cumulative.mean(axis=0)
    triple_median = np.median(result.triple_cumulative, axis=0)

    fig = go.Figure()

    if show_trials:
        _add_trial_traces(
            fig,
            result.baseline_cumulative,
            result.days,
            color="rgba(31, 119, 180, 0.35)",
            legend_group="Baseline trials",
            show_in_legend=True,
        )
        _add_trial_traces(
            fig,
            result.triple_cumulative,
            result.days,
            color="rgba(255, 127, 14, 0.35)",
            legend_group="3x trials",
            show_in_legend=True,
        )

    summary_traces = [
        (baseline_mean, "Baseline mean", "#1f77b4", "solid", 3),
        (baseline_median, "Baseline median", "#1f77b4", "dash", 3),
        (triple_mean, "3x mean", "#ff7f0e", "solid", 3),
        (triple_median, "3x median", "#ff7f0e", "dash", 3),
    ]
    for y, name, color, dash, width in summary_traces:
        fig.add_trace(
            go.Scatter(
                x=result.days,
                y=y,
                mode="lines",
                name=name,
                line=dict(color=color, width=width, dash=dash),
                hovertemplate=(
                    f"{name}<br>"
                    "Day: %{x}<br>"
                    "Cumulative return: %{y:.2%}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=(
            f"Monte Carlo Cumulative Returns ({result.n_trials:,} trials, "
            f"{result.n_days} days)<br>"
            f"<sup>Annual drift={result.annual_drift:.2%}, "
            f"annual volatility={result.annual_volatility:.2%}</sup>"
        ),
        xaxis_title="Day",
        yaxis_title="Cumulative return",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig


def build_summary_table(result: MonteCarloResult) -> dict[str, float]:
    """Final-day summary statistics for baseline and triple paths."""
    final_baseline = result.baseline_cumulative[:, -1]
    final_triple = result.triple_cumulative[:, -1]
    return {
        "baseline_mean_final": float(final_baseline.mean()),
        "baseline_median_final": float(np.median(final_baseline)),
        "triple_mean_final": float(final_triple.mean()),
        "triple_median_final": float(np.median(final_triple)),
    }


def main() -> None:
    print("Monte Carlo cumulative return simulator")
    print("Enter annual parameters as decimals (e.g. 0.08 for 8% drift).")
    annual_drift = prompt_float("Annual drift")
    annual_volatility = prompt_float("Annual volatility")

    n_trials_raw = input("Number of trials [1000]: ").strip()
    n_trials = int(n_trials_raw) if n_trials_raw else 1000

    result = run_monte_carlo(annual_drift, annual_volatility, n_trials=n_trials)
    summary = build_summary_table(result)

    print("\nFinal cumulative return summary:")
    print(f"  Baseline mean:   {summary['baseline_mean_final']:+.2%}")
    print(f"  Baseline median: {summary['baseline_median_final']:+.2%}")
    print(f"  3x mean:         {summary['triple_mean_final']:+.2%}")
    print(f"  3x median:       {summary['triple_median_final']:+.2%}")

    fig = build_monte_carlo_figure(result)
    html_path = input("Export HTML path [monte_carlo_report.html]: ").strip()
    if not html_path:
        html_path = "monte_carlo_report.html"
    export_html(fig, html_path)
    print(f"Saved report to {html_path}")
    fig.show()


if __name__ == "__main__":
    main()
