"""
Data layer package.

Public API:
    get_prices(ticker, as_of_date, lookback_days) -> pd.DataFrame
    get_news(ticker, as_of_date, lookback_days)   -> list[dict]
    get_indicators(ticker, as_of_date, lookback_days) -> dict

All functions are point-in-time safe: they NEVER return data dated after
`as_of_date`. This is the foundation of honest backtesting.
"""

from .prices import get_prices
from .news import get_news
from .indicators import get_indicators

__all__ = ["get_prices", "get_news", "get_indicators"]