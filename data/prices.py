"""
Price data fetcher.

Why this module exists:
- yfinance is free but slow and occasionally fails. We cache aggressively.
- We MUST enforce point-in-time correctness: if as_of_date is 2022-03-15,
  the returned data must not include any rows from 2022-03-16 onwards.
"""

from datetime import date
from pathlib import Path
import pickle
import time

import pandas as pd
import yfinance as yf

from config import DATA_CACHE_DIR


_PRICE_CACHE_DIR = DATA_CACHE_DIR / "prices"
_PRICE_CACHE_DIR.mkdir(exist_ok=True)

_CACHE_TTL_SECONDS = 24 * 60 * 60


def _cache_path(ticker: str) -> Path:
    safe_ticker = ticker.upper().replace("/", "_")
    return _PRICE_CACHE_DIR / f"{safe_ticker}.pkl"


def _is_cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < _CACHE_TTL_SECONDS


def _fetch_full_history(ticker: str) -> pd.DataFrame:
    """Fetch full history for `ticker` from yfinance."""
    yf_ticker = yf.Ticker(ticker)
    df = yf_ticker.history(period="max", auto_adjust=True)

    if df.empty:
        raise ValueError(
            f"yfinance returned no data for ticker {ticker!r}. "
            f"Check the symbol — common gotchas: BRK.B (yfinance wants BRK-B), "
            f"non-US tickers need exchange suffixes (e.g., 'SAP.DE')."
        )

    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    keep_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[keep_cols].copy()
    return df


def _load_or_fetch(ticker: str) -> pd.DataFrame:
    path = _cache_path(ticker)
    if _is_cache_fresh(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    df = _fetch_full_history(ticker)
    with open(path, "wb") as f:
        pickle.dump(df, f)
    return df


def get_prices(
    ticker: str,
    as_of_date: str | date,
    lookback_days: int = 90,
) -> pd.DataFrame:
    """
    Fetch price data for `ticker`, point-in-time as of `as_of_date`.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    The most recent row is guaranteed to be on or before `as_of_date`.
    """
    if isinstance(as_of_date, str):
        as_of = pd.Timestamp(as_of_date).normalize()
    elif isinstance(as_of_date, date):
        as_of = pd.Timestamp(as_of_date).normalize()
    else:
        raise ValueError(
            f"as_of_date must be a string or date, got {type(as_of_date)}"
        )

    full_history = _load_or_fetch(ticker)

    # THE critical line: never return data dated after as_of_date.
    pit_filter = full_history.index <= as_of
    sliced = full_history.loc[pit_filter]

    earliest = as_of - pd.Timedelta(days=lookback_days)
    sliced = sliced.loc[sliced.index >= earliest]

    if sliced.empty:
        raise ValueError(
            f"No price data for {ticker} between "
            f"{earliest.date()} and {as_of.date()}. "
            f"Was the ticker listed yet? Was the market open?"
        )

    return sliced