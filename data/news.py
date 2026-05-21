"""
News fetcher.

Uses yfinance's news endpoint — free, no API key, decent quality for major
US stocks. Returns news articles published ON OR BEFORE the as_of_date.

Limitations (be honest about these in your writeup):
- yfinance news only goes back ~30-90 days from "now" — older dates have
  sparse news. For 2020/2022 regime testing, news coverage will be thinner
  than for recent dates.
- We can't easily verify the exact publication timestamp; we trust yfinance's
  `providerPublishTime` field.
- This is NOT a true point-in-time news archive. For production research you'd
  want a paid source like Refinitiv. We document this limitation and proceed.
"""

from datetime import date, datetime
from pathlib import Path
import pickle
import time

import yfinance as yf

from config import DATA_CACHE_DIR


# ============================================================================
# CACHE
# ============================================================================

_NEWS_CACHE_DIR = DATA_CACHE_DIR / "news"
_NEWS_CACHE_DIR.mkdir(exist_ok=True)

# News cache TTL: 6 hours. News doesn't change much over hours,
# but we don't want to serve week-old "latest news" either.
_CACHE_TTL_SECONDS = 6 * 60 * 60


def _cache_path(ticker: str) -> Path:
    safe_ticker = ticker.upper().replace("/", "_")
    return _NEWS_CACHE_DIR / f"{safe_ticker}.pkl"


def _is_cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < _CACHE_TTL_SECONDS


# ============================================================================
# YFINANCE NEWS FETCHER
# ============================================================================

def _fetch_news(ticker: str) -> list[dict]:
    """
    Fetch all available news for `ticker` from yfinance.

    Returns a list of dicts, each with keys:
        title, publisher, link, published_at (datetime), summary

    yfinance returns news in a structured format that has changed over
    versions. We normalize to a stable shape here so the rest of the
    project doesn't need to track yfinance's API changes.
    """
    raw = yf.Ticker(ticker).news

    if not raw:
        return []

    normalized = []
    for item in raw:
        # yfinance wraps each item in a "content" sub-dict in newer versions
        content = item.get("content", item)

        # Title and publisher have moved around across yfinance versions
        title = content.get("title") or content.get("headline") or ""
        publisher = (
            content.get("publisher")
            or content.get("provider", {}).get("displayName")
            or "unknown"
        )

        # Publication timestamp — try a few possible field names
        pub_ts = (
            content.get("providerPublishTime")
            or content.get("pubDate")
            or content.get("displayTime")
        )
        if pub_ts is None:
            continue  # Skip items with no date — we can't point-in-time filter them

        # Normalize timestamp to datetime
        if isinstance(pub_ts, (int, float)):
            published_at = datetime.fromtimestamp(pub_ts)
        elif isinstance(pub_ts, str):
            try:
                published_at = datetime.fromisoformat(pub_ts.replace("Z", "+00:00"))
                published_at = published_at.replace(tzinfo=None)
            except ValueError:
                continue
        else:
            continue

        link = content.get("link") or content.get("canonicalUrl", {}).get("url", "")
        summary = content.get("summary", "")

        normalized.append({
            "title": title,
            "publisher": publisher,
            "link": link,
            "published_at": published_at,
            "summary": summary,
        })

    return normalized


def _load_or_fetch(ticker: str) -> list[dict]:
    """Cached fetch."""
    path = _cache_path(ticker)

    if _is_cache_fresh(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    news = _fetch_news(ticker)
    with open(path, "wb") as f:
        pickle.dump(news, f)
    return news


# ============================================================================
# PUBLIC API
# ============================================================================

def get_news(
    ticker: str,
    as_of_date: str | date,
    lookback_days: int = 7,
) -> list[dict]:
    """
    Fetch news for `ticker` published on or before `as_of_date`.

    Args:
        ticker: Stock symbol.
        as_of_date: Simulation date. News from after this date is filtered out.
        lookback_days: Only return news from the last N days before as_of_date.
                       Default 7 (~one week of recent news).

    Returns:
        List of dicts with keys: title, publisher, link, published_at, summary.
        Sorted newest first. Empty list if no news is available.
    """
    if isinstance(as_of_date, str):
        as_of_dt = datetime.fromisoformat(as_of_date)
    elif isinstance(as_of_date, date):
        as_of_dt = datetime.combine(as_of_date, datetime.min.time())
    else:
        raise ValueError(f"Bad as_of_date type: {type(as_of_date)}")

    # End of the day for as_of (so news from that day still counts)
    as_of_end = as_of_dt.replace(hour=23, minute=59, second=59)

    earliest = as_of_end.replace(hour=0, minute=0, second=0)
    earliest = earliest.replace(day=1) if False else earliest  # placeholder
    from datetime import timedelta
    earliest = as_of_dt - timedelta(days=lookback_days)

    all_news = _load_or_fetch(ticker)

    # Point-in-time filter: keep only items published in [earliest, as_of_end]
    filtered = [
        item for item in all_news
        if earliest <= item["published_at"] <= as_of_end
    ]

    # Sort newest first
    filtered.sort(key=lambda x: x["published_at"], reverse=True)

    return filtered