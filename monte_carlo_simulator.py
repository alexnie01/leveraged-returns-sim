"""Monte Carlo simulation of cumulative return paths."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from cumulative_returns import (
    DEFAULT_DAYS,
    export_html,
    prompt_float,
)

_MIN_TRIALS_PER_WORKER = 50


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


def percentile_band(paths: np.ndarray, lower: float = 5.0, upper: float = 95.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) percentile paths along the trial axis."""
    return (
        np.percentile(paths, lower, axis=0),
        np.percentile(paths, upper, axis=0),
    )


def _chunk_sizes(n_trials: int, n_workers: int) -> list[int]:
    n_workers = max(1, min(n_workers, n_trials))
    base, remainder = divmod(n_trials, n_workers)
    return [base + (1 if i < remainder else 0) for i in range(n_workers)]


def _worker_count(n_trials: int, n_workers: int | None) -> int:
    cpu_count = os.cpu_count() or 1
    requested = cpu_count if n_workers is None else max(1, n_workers)
    max_by_size = max(1, n_trials // _MIN_TRIALS_PER_WORKER)
    return min(requested, n_trials, max_by_size)


def _simulate_chunk(
    annual_drift: float,
    annual_volatility: float,
    n_trials: int,
    n_days: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    baseline_daily = simulate_daily_returns(
        annual_drift, annual_volatility, n_trials, n_days, rng
    )
    triple_daily = 3.0 * baseline_daily
    return cumulative_paths(baseline_daily), cumulative_paths(triple_daily)


def run_monte_carlo(
    annual_drift: float,
    annual_volatility: float,
    n_trials: int = 1000,
    n_days: int = DEFAULT_DAYS,
    rng: np.random.Generator | None = None,
    n_workers: int | None = None,
) -> MonteCarloResult:
    rng = rng or np.random.default_rng()
    workers = _worker_count(n_trials, n_workers)
    chunk_sizes = _chunk_sizes(n_trials, workers)
    seeds = rng.integers(0, np.iinfo(np.uint64).max, size=len(chunk_sizes), dtype=np.uint64)

    if len(chunk_sizes) == 1:
        baseline_cumulative, triple_cumulative = _simulate_chunk(
            annual_drift, annual_volatility, chunk_sizes[0], n_days, int(seeds[0])
        )
    else:
        payloads = [
            (annual_drift, annual_volatility, chunk, n_days, int(seed))
            for chunk, seed in zip(chunk_sizes, seeds)
        ]
        with ProcessPoolExecutor(max_workers=len(chunk_sizes)) as pool:
            chunks = list(pool.map(_simulate_chunk_star, payloads))
        baseline_cumulative = np.concatenate([chunk[0] for chunk in chunks], axis=0)
        triple_cumulative = np.concatenate([chunk[1] for chunk in chunks], axis=0)

    return MonteCarloResult(
        baseline_cumulative=baseline_cumulative,
        triple_cumulative=triple_cumulative,
        days=np.arange(n_days),
        annual_drift=annual_drift,
        annual_volatility=annual_volatility,
        n_trials=n_trials,
        n_days=n_days,
    )


def _simulate_chunk_star(
    payload: tuple[float, float, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    return _simulate_chunk(*payload)


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


def _add_percentile_band(
    fig: go.Figure,
    paths: np.ndarray,
    days: np.ndarray,
    *,
    name: str,
    fillcolor: str,
    line_color: str,
) -> None:
    lower, upper = percentile_band(paths)
    fig.add_trace(
        go.Scatter(
            x=days,
            y=upper,
            mode="lines",
            line=dict(width=0, color=line_color),
            legendgroup=name,
            showlegend=False,
            hoverinfo="skip",
            name=name,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=lower,
            mode="lines",
            line=dict(width=0, color=line_color),
            fill="tonexty",
            fillcolor=fillcolor,
            legendgroup=name,
            name=name,
            customdata=np.column_stack((lower, upper)),
            hovertemplate=(
                f"{name}<br>"
                "Day: %{x}<br>"
                "5th percentile: %{customdata[0]:.1%}<br>"
                "95th percentile: %{customdata[1]:.1%}<extra></extra>"
            ),
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
            legend_group="base trials",
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

    _add_percentile_band(
        fig,
        result.baseline_cumulative,
        result.days,
        name="base 5th-95th",
        fillcolor="rgba(31, 119, 180, 0.22)",
        line_color="#1f77b4",
    )
    _add_percentile_band(
        fig,
        result.triple_cumulative,
        result.days,
        name="3x 5th-95th",
        fillcolor="rgba(255, 127, 14, 0.22)",
        line_color="#ff7f0e",
    )

    summary_traces = [
        (baseline_mean, "base mean", "#1f77b4", "solid", 3),
        (baseline_median, "base median", "#1f77b4", "dash", 3),
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
                    "Cumulative return: %{y:.1%}<extra></extra>"
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
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(r=180),
    )
    return fig


def build_summary_table(result: MonteCarloResult) -> dict[str, float]:
    """Final-day summary statistics for baseline and triple paths."""
    final_baseline = result.baseline_cumulative[:, -1]
    final_triple = result.triple_cumulative[:, -1]
    baseline_p05, baseline_p95 = percentile_band(result.baseline_cumulative)
    triple_p05, triple_p95 = percentile_band(result.triple_cumulative)
    return {
        "baseline_mean_final": float(final_baseline.mean()),
        "baseline_median_final": float(np.median(final_baseline)),
        "baseline_p05_final": float(baseline_p05[-1]),
        "baseline_p95_final": float(baseline_p95[-1]),
        "triple_mean_final": float(final_triple.mean()),
        "triple_median_final": float(np.median(final_triple)),
        "triple_p05_final": float(triple_p05[-1]),
        "triple_p95_final": float(triple_p95[-1]),
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
