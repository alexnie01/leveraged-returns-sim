"""Unified returns analysis: single-path view and Monte Carlo simulation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from cumulative_returns import (
    TRADING_DAYS,
    DataSource,
    ReturnSeries,
    build_figure,
    export_html,
    fetch_ticker_returns,
    simulated_returns,
)
from monte_carlo_simulator import (
    MonteCarloResult,
    build_monte_carlo_figure,
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
    n_workers: int | None = None
    html_output: Path = Path("returns_report.html")


@dataclass
class ReportData:
    annual_drift: float
    annual_volatility: float
    single_path: ReturnSeries
    tripled_returns: np.ndarray
    monte_carlo: MonteCarloResult
    mc_summary: dict[str, float]


@dataclass
class ReportArtifacts:
    data: ReportData
    single_path_figure: go.Figure
    monte_carlo_figure: go.Figure
    single_path_html: Path
    monte_carlo_html: Path


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
    else:
        single_path = simulated_returns(drift, volatility, config.n_days, rng)

    tripled_returns = 3.0 * single_path.daily_returns
    monte_carlo = run_monte_carlo(
        drift,
        volatility,
        n_trials=config.n_trials,
        n_days=config.n_days,
        rng=rng,
        n_workers=config.n_workers,
    )
    mc_summary = build_summary_table(monte_carlo)

    return ReportData(
        annual_drift=drift,
        annual_volatility=volatility,
        single_path=single_path,
        tripled_returns=tripled_returns,
        monte_carlo=monte_carlo,
        mc_summary=mc_summary,
    )


def _param_note(data: ReportData) -> str:
    base = data.single_path
    origin = f"Calibrated from {base.ticker}" if base.source == DataSource.TICKER else "User inputs"
    return f"{origin}: drift={data.annual_drift:.2%}, volatility={data.annual_volatility:.2%}"


def build_single_path_figure(data: ReportData) -> go.Figure:
    base = data.single_path
    base_name = f"{base.ticker} returns" if base.ticker else "Base returns"
    fig = build_figure(base, data.tripled_returns, base_trace_name=base_name)
    n_days = len(base.daily_returns)
    if base.source == DataSource.TICKER:
        title = f"Single Path — {base.ticker} ({n_days} days)<br><sup>{_param_note(data)}</sup>"
    else:
        title = f"Single Path — Simulated ({n_days} days)<br><sup>{_param_note(data)}</sup>"
    fig.update_layout(title=title)
    return fig


def build_monte_carlo_report_figure(data: ReportData, *, show_trials: bool) -> go.Figure:
    fig = build_monte_carlo_figure(data.monte_carlo, show_trials=show_trials)
    result = data.monte_carlo
    fig.update_layout(
        title=(
            f"Monte Carlo — {result.n_trials:,} trials × {result.n_days} days"
            f"<br><sup>{_param_note(data)}</sup>"
        )
    )
    return fig


def report_html_paths(config: ReportConfig) -> tuple[Path, Path]:
    output = Path(config.html_output)
    suffix = output.suffix or ".html"
    return (
        output.with_name(f"{output.stem}_single_path{suffix}"),
        output.with_name(f"{output.stem}_monte_carlo{suffix}"),
    )


def export_report(fig: go.Figure, path: str | Path) -> Path:
    return export_html(fig, path)


def _display_figures(*figures: go.Figure) -> None:
    ipython = None
    try:
        from IPython import get_ipython

        ipython = get_ipython()
    except ImportError:
        ipython = None

    if ipython is not None:
        from IPython.display import display

        for fig in figures:
            display(fig)
        return

    for fig in figures:
        fig.show()


def generate_report(
    config: ReportConfig,
    rng: np.random.Generator | None = None,
    *,
    show: bool = True,
) -> ReportArtifacts:
    """Build data, write HTML reports, and optionally display figures."""
    data = build_report(config, rng=rng)
    single_path_figure = build_single_path_figure(data)
    monte_carlo_figure = build_monte_carlo_report_figure(
        data, show_trials=config.show_mc_trials
    )
    single_path_html, monte_carlo_html = report_html_paths(config)
    export_report(single_path_figure, single_path_html)
    export_report(monte_carlo_figure, monte_carlo_html)
    print(f"Single-path report saved to: {single_path_html.resolve()}")
    print(f"Monte Carlo report saved to: {monte_carlo_html.resolve()}")

    artifacts = ReportArtifacts(
        data=data,
        single_path_figure=single_path_figure,
        monte_carlo_figure=monte_carlo_figure,
        single_path_html=single_path_html,
        monte_carlo_html=monte_carlo_html,
    )
    if show:
        _display_figures(artifacts.single_path_figure, artifacts.monte_carlo_figure)
    return artifacts


def parse_args(argv: list[str] | None = None) -> ReportConfig:
    parser = argparse.ArgumentParser(
        description="Generate single-path and Monte Carlo cumulative-return reports."
    )
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Use historical ticker returns and calibrate drift/volatility from the sample.",
    )
    parser.add_argument("--ticker", default="SPY", help="Ticker when --historical is set.")
    parser.add_argument("--drift", type=float, default=0.08, help="Annual drift (simulated mode).")
    parser.add_argument(
        "--volatility",
        type=float,
        default=0.20,
        help="Annual volatility (simulated mode).",
    )
    parser.add_argument("--n-trials", type=int, default=1000, help="Monte Carlo trial count.")
    parser.add_argument("--n-days", type=int, default=1000, help="Number of trading days.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Process workers for Monte Carlo paths. Defaults to available CPUs.",
    )
    parser.add_argument(
        "--hide-mc-trials",
        action="store_true",
        help="Hide individual Monte Carlo trial paths.",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=Path("returns_report.html"),
        help="Base HTML path; _single_path and _monte_carlo reports are written beside it.",
    )
    args = parser.parse_args(argv)
    return ReportConfig(
        use_historical=args.historical,
        ticker=args.ticker,
        annual_drift=args.drift,
        annual_volatility=args.volatility,
        n_trials=args.n_trials,
        n_days=args.n_days,
        random_seed=args.seed,
        show_mc_trials=not args.hide_mc_trials,
        n_workers=args.n_workers,
        html_output=args.html_output,
    )


def main(argv: list[str] | None = None) -> ReportArtifacts:
    return generate_report(parse_args(argv))


if __name__ == "__main__":
    main()
