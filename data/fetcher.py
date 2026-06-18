"""
data/fetcher.py
Thin wrapper around yfinance. Returns clean DataFrames of adjusted close prices
and log returns. Results are cached in-memory for the session.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


_CACHE: dict[tuple, pd.DataFrame] = {}


def fetch_prices(
    tickers: list[str],
    period_years: float = 3,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Download adjusted close prices for *tickers* covering *period_years*.

    Returns
    -------
    pd.DataFrame  shape (T, N)  — DatetimeIndex, columns = tickers
    """
    end = end or datetime.today()
    start = end - timedelta(days=int(period_years * 365))
    key = (tuple(sorted(tickers)), start.date(), end.date())

    if key in _CACHE:
        return _CACHE[key]

    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][tickers]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all").ffill()
    _CACHE[key] = prices
    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns, NaN first row dropped."""
    return np.log(prices / prices.shift(1)).dropna()


def clear_cache() -> None:
    _CACHE.clear()
