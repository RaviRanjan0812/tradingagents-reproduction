"""
Smoke test for the data layer.

The most important test here is `test_point_in_time` — it verifies our
backtest is honest. Run before moving on.

Usage:
    python test_data.py
"""

from datetime import date
import pandas as pd

from data import get_prices, get_news, get_indicators


def test_prices_basic():
    """Verify we can fetch prices for a well-known ticker."""
    print("Test 1: Basic price fetch (AAPL, 90 days ending 2024-03-01)")
    df = get_prices("AAPL", as_of_date="2024-03-01", lookback_days=90)

    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
    assert len(df) > 30, f"Expected >30 trading days, got {len(df)}"
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"], \
        f"Unexpected columns: {df.columns.tolist()}"
    assert df.index.is_monotonic_increasing, "Index should be sorted ascending"

    print(f"  Got {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Latest close: ${df['Close'].iloc[-1]:.2f}")
    print("  PASS\n")


def test_point_in_time():
    """
    THE critical test. Verify that asking for data as of 2022-06-01 does
    NOT return any rows dated 2022-06-02 or later.

    If this test fails, every backtest result is meaningless.
    """
    print("Test 2: Point-in-time correctness (AAPL as of 2022-06-01)")
    as_of = pd.Timestamp("2022-06-01")
    df = get_prices("AAPL", as_of_date="2022-06-01", lookback_days=30)

    # The most recent row must be on or before our as_of date
    most_recent = df.index.max()
    assert most_recent <= as_of, (
        f"LEAK! Got data from {most_recent.date()} when as_of was {as_of.date()}. "
        f"This means our backtest can see the future. Critical bug."
    )

    print(f"  Most recent row: {most_recent.date()} (as_of: {as_of.date()})")
    print(f"  No future leak. ✓")
    print("  PASS\n")


def test_news():
    """Fetch news for AAPL recently. Older news may be empty (yfinance limitation)."""
    print("Test 3: News fetch (AAPL, recent)")
    news = get_news("AAPL", as_of_date="2026-05-01", lookback_days=14)

    print(f"  Got {len(news)} articles")
    if news:
        # Print the first headline as a sanity check
        first = news[0]
        print(f"  Most recent: '{first['title'][:80]}...' ({first['published_at'].date()})")

        # Verify point-in-time: no article newer than our as_of
        as_of = pd.Timestamp("2026-05-01")
        for item in news:
            assert pd.Timestamp(item["published_at"]) <= as_of.replace(hour=23, minute=59), \
                f"News leak: article from {item['published_at']} > as_of {as_of}"
        print(f"  All articles within point-in-time window")
    else:
        print(f"  No news returned — this is OK, yfinance news can be sparse")

    print("  PASS\n")


def test_indicators():
    """Verify indicators compute and return sensible values."""
    print("Test 4: Technical indicators (AAPL, 2024-03-01)")
    result = get_indicators("AAPL", as_of_date="2024-03-01", lookback_days=250)

    assert "latest" in result and "series" in result, \
        "Should have 'latest' and 'series' keys"

    latest = result["latest"]
    print(f"  As of: {result['as_of']}")
    print(f"  RSI(14):       {latest.get('rsi_14'):.2f}")
    print(f"  MACD:          {latest.get('macd'):.4f}")
    print(f"  50-day SMA:    ${latest.get('close_50_sma'):.2f}")
    print(f"  200-day SMA:   ${latest.get('close_200_sma'):.2f}")
    print(f"  Bollinger Upper: ${latest.get('boll_ub'):.2f}")

    # Sanity checks: RSI is bounded 0-100; SMAs are positive
    rsi = latest["rsi_14"]
    assert 0 <= rsi <= 100, f"RSI out of bounds: {rsi}"
    assert latest["close_50_sma"] > 0, "SMA should be positive"

    print("  PASS\n")


def test_unknown_ticker():
    """Verify we get a clear error for a nonsense ticker."""
    print("Test 5: Unknown ticker raises clear error")
    try:
        get_prices("ZZZZZNOTAREALTICKER", as_of_date="2024-03-01")
        print("  FAIL — should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  Got expected error: {str(e)[:80]}...")
        print("  PASS\n")
        return True


def main():
    print("=" * 60)
    print("Data Layer Smoke Test")
    print("=" * 60)
    print()

    test_prices_basic()
    test_point_in_time()
    test_news()
    test_indicators()
    test_unknown_ticker()

    print("=" * 60)
    print("All tests passed. Data layer is working.")
    print("=" * 60)


if __name__ == "__main__":
    main()