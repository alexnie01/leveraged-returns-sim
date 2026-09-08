"""Local compressed cache for downloaded ticker history."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HISTORICAL_DATA_DIR = Path(__file__).resolve().parent / "historical_data"


def cache_path(ticker: str) -> Path:
    return HISTORICAL_DATA_DIR / f"{ticker.upper()}.csv.gz"


def _load_cached_close(ticker: str) -> pd.Series | None:
    path = cache_path(ticker)
    if not path.exists():
        return None

    frame = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    if frame.empty or "Close" not in frame.columns:
        return None
    return frame["Close"]


def _save_close(ticker: str, close: pd.Series) -> Path:
    HISTORICAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(ticker)
    close.to_frame(name="Close").to_csv(path, compression="gzip")
    return path


def _download_close(ticker: str) -> pd.Series:
    import yfinance as yf

    history = yf.Ticker(ticker).history(period="max", auto_adjust=True)
    if history.empty or "Close" not in history.columns:
        raise ValueError(f"No historical data found for ticker '{ticker}'.")
    return history["Close"]


def get_ticker_close(ticker: str, *, force_download: bool = False) -> tuple[pd.Series, bool]:
    """Return adjusted close prices and whether the data came from cache."""
    symbol = ticker.upper()
    if not force_download:
        cached = _load_cached_close(symbol)
        if cached is not None:
            return cached, True

    close = _download_close(symbol)
    _save_close(symbol, close)
    return close, False
