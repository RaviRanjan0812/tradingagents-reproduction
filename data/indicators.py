"""
Technical indicators.

Reuses the point-in-time price data from prices.py and runs a handful of
classic technical indicators on it. The stockstats library handles the math.

Indicator selection: we pick a representative subset (not all 60 from the
paper) covering trend, momentum, and volatility. Adding more is trivial later.
"""

from datetime import date

import pandas as pd
from stockstats import wrap as stockstats_wrap

from .prices import get_prices


# ============================================================================
# INDICATOR SET
# ============================================================================
# stockstats uses short names. Here's what each one is:
#
# Trend:
#   close_50_sma         50-day simple moving average of close
#   close_200_sma        200-day SMA (the "golden cross / death cross" line)
#   macd                 Moving Average Convergence Divergence
#   macds                MACD signal line
#   macdh                MACD histogram (macd - macds)
#
# Momentum:
#   rsi_14               Relative Strength Index, 14-day
#   wr_14                Williams %R, 14-day
#
# Volatility:
#   boll                 Bollinger Band middle (20-day SMA)
#   boll_ub              Bollinger upper band
#   boll_lb              Bollinger lower band
#   atr_14               Average True Range, 14-day (absolute volatility)

INDICATORS = [
    "close_50_sma",
    "close_200_sma",
    "macd",
    "macds",
    "macdh",
    "rsi_14",
    "wr_14",
    "boll",
    "boll_ub",
    "boll_lb",
    "atr_14",
]


# ============================================================================
# PUBLIC API
# ============================================================================

def get_indicators(
    ticker: str,
    as_of_date: str | date,
    lookback_days: int = 250,
) -> dict:
    """
    Compute technical indicators for `ticker` as of `as_of_date`.

    Args:
        ticker: Stock symbol.
        as_of_date: Simulation date — indicators reflect data through this date.
        lookback_days: How much history to feed the indicators. Default 250
                       (~one trading year), which is enough for 200-day SMA.

    Returns:
        Dict with two sections:
            "latest": dict mapping indicator name -> most recent value
                      (the value AS OF as_of_date)
            "series": pd.DataFrame of all indicators across the lookback window
                      (useful if an agent wants to see the trend)

    Note: 200-day SMA requires 200 trading days of history. If `lookback_days`
    is too small or the ticker is too new, those indicators will be NaN.
    """
    # Fetch the underlying prices with point-in-time correctness already enforced
    prices = get_prices(ticker, as_of_date, lookback_days=lookback_days)

    if len(prices) < 30:
        raise ValueError(
            f"Only {len(prices)} days of price data for {ticker} before "
            f"{as_of_date}. Need at least 30 for indicators to be meaningful. "
            f"Try a later as_of_date or a different ticker."
        )

    # stockstats wraps a DataFrame and adds indicator columns lazily.
    # The library is picky about column names — needs lowercase.
    df = prices.copy()
    df.columns = [c.lower() for c in df.columns]
    sdf = stockstats_wrap(df)

    # Trigger computation of each indicator. stockstats computes lazily;
    # we touch each column to force the calculation.
    for indicator in INDICATORS:
        _ = sdf[indicator]

    # Extract the latest row — the values as of as_of_date
    latest_row = sdf.iloc[-1]
    latest = {
        indicator: (
            float(latest_row[indicator])
            if pd.notna(latest_row[indicator])
            else None
        )
        for indicator in INDICATORS
    }

    # Also return the time series for agents that want to see trends
    series = sdf[INDICATORS].copy()

    return {
        "latest": latest,
        "series": series,
        "as_of": str(prices.index[-1].date()),  # Actual most recent trading day
    }