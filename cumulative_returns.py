#!/usr/bin/env python3
"""Simulate or fetch daily returns and plot cumulative returns."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

TRADING_DAYS = 252
DEFAULT_DAYS = 1000


class DataSource(str, Enum):
    SIMULATED = "simulated"
    TICKER = "ticker"


@dataclass
class ReturnSeries:
    daily_returns: np.ndarray
    source: DataSource
    label: str
    x_values: np.ndarray | None = None
    annual_drift: float | None = None
    annual_volatility: float | None = None
    ticker: str | None = None


def prompt_float(label: str) -> float:
    while True:
        raw = input(f"{label}: ").strip()
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")


def lognormal_daily_returns(
    annual_drift: float,
    annual_volatility: float,
    n_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample simple daily returns from a lognormal gross-return model."""
    mu_daily = annual_drift / TRADING_DAYS
    sigma_daily = annual_volatility / np.sqrt(TRADING_DAYS)
    log_returns = rng.normal(mu_daily, sigma_daily, size=n_days)
    return np.exp(log_returns) - 1.0


def fetch_ticker_returns(ticker: str, n_days: int = DEFAULT_DAYS) -> ReturnSeries:
    """Fetch the most recent n_days of daily simple returns for a ticker."""
    from historical_data_cache import get_ticker_close

    close, _from_cache = get_ticker_close(ticker)
    daily_returns = close.pct_change().dropna()
    if daily_returns.empty:
        raise ValueError(f"No historical data found for ticker '{ticker}'.")

    if len(daily_returns) > n_days:
        daily_returns = daily_returns.iloc[-n_days:]

    return ReturnSeries(
        daily_returns=daily_returns.to_numpy(),
        source=DataSource.TICKER,
        label=ticker.upper(),
        x_values=daily_returns.index.to_numpy(),
        ticker=ticker.upper(),
    )


def simulated_returns(
    annual_drift: float,
    annual_volatility: float,
    n_days: int = DEFAULT_DAYS,
    rng: np.random.Generator | None = None,
) -> ReturnSeries:
    rng = rng or np.random.default_rng()
    daily_returns = lognormal_daily_returns(annual_drift, annual_volatility, n_days, rng)
    return ReturnSeries(
        daily_returns=daily_returns,
        source=DataSource.SIMULATED,
        label="Simulated",
        x_values=np.arange(n_days),
        annual_drift=annual_drift,
        annual_volatility=annual_volatility,
    )


def cumulative_returns(daily_returns: np.ndarray) -> np.ndarray:
    wealth = np.cumprod(1.0 + daily_returns)
    return wealth - 1.0


def lookback_return(daily_returns: np.ndarray, day_index: int) -> tuple[int, float]:
    """Return (days_looking_back, cumulative return) ending at day_index."""
    days_back = day_index + 1
    window = daily_returns[: day_index + 1]
    cumulative = np.prod(1.0 + window) - 1.0
    return days_back, cumulative


def build_hover_data(daily_returns: np.ndarray) -> list[tuple[int, float]]:
    return [lookback_return(daily_returns, int(day)) for day in range(len(daily_returns))]


def build_figure(
    base: ReturnSeries,
    tripled_returns: np.ndarray,
    *,
    base_trace_name: str = "Base returns",
    tripled_trace_name: str = "3x daily returns",
) -> go.Figure:
    n_days = len(base.daily_returns)
    x_values = base.x_values if base.x_values is not None else np.arange(n_days)
    base_cumulative = cumulative_returns(base.daily_returns)
    tripled_cumulative = cumulative_returns(tripled_returns)

    base_hover = build_hover_data(base.daily_returns)
    tripled_hover = build_hover_data(tripled_returns)

    fig = go.Figure()
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
        )
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
        )
    )

    if base.source == DataSource.TICKER:
        title = (
            f"Cumulative Returns — {base.ticker} ({n_days} trading days)<br>"
            f"<sup>Historical daily data; 3x series uses tripled daily returns</sup>"
        )
        xaxis_title = "Date"
    else:
        title = (
            f"Cumulative Returns ({n_days} simulated days)<br>"
            f"<sup>Annual drift={base.annual_drift:.2%}, "
            f"annual volatility={base.annual_volatility:.2%}</sup>"
        )
        xaxis_title = "Day"

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Cumulative return",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(r=180),
    )
    return fig


def export_html(fig: go.Figure, path: str | Path) -> Path:
    output = Path(path)
    fig.write_html(str(output), include_plotlyjs="cdn", full_html=True)
    return output


def load_returns(
    source: DataSource,
    *,
    annual_drift: float = 0.08,
    annual_volatility: float = 0.20,
    ticker: str = "SPY",
    n_days: int = DEFAULT_DAYS,
    rng: np.random.Generator | None = None,
) -> ReturnSeries:
    if source == DataSource.SIMULATED:
        return simulated_returns(annual_drift, annual_volatility, n_days, rng)
    if source == DataSource.TICKER:
        return fetch_ticker_returns(ticker, n_days)
    raise ValueError(f"Unknown data source: {source}")


def main() -> None:
    print("Cumulative return visualizer")
    print("Choose data source:")
    print("  1) Simulated lognormal returns")
    print("  2) Historical ticker data")
    choice = input("Enter 1 or 2 [1]: ").strip() or "1"

    if choice == "2":
        ticker = input("Ticker symbol [SPY]: ").strip().upper() or "SPY"
        base = fetch_ticker_returns(ticker)
        base_trace = f"{ticker} returns"
    else:
        print("Enter annual parameters as decimals (e.g. 0.08 for 8% drift).")
        annual_drift = prompt_float("Annual drift")
        annual_volatility = prompt_float("Annual volatility")
        base = simulated_returns(annual_drift, annual_volatility)
        base_trace = "Base returns"

    tripled_returns = 3.0 * base.daily_returns
    fig = build_figure(base, tripled_returns, base_trace_name=base_trace)

    html_path = input("Export HTML path [cumulative_returns_report.html]: ").strip()
    if not html_path:
        html_path = "cumulative_returns_report.html"
    export_html(fig, html_path)
    print(f"Saved report to {html_path}")

    fig.show()


if __name__ == "__main__":
    main()
