"""
Unit test for portfolio.py — verifies the simulation and metrics math
with hand-checkable numbers. No LLM calls; costs nothing to run.

Usage:
    python test_portfolio.py
"""

import pandas as pd
from portfolio import Portfolio, compute_metrics, buy_and_hold_value_series


def test_buy_then_hold():
    """Buy at $100, hold while price rises to $120 -> +20% return."""
    print("Test 1: buy then hold through a rise")
    p = Portfolio(starting_cash=10_000.0)
    p.step("2024-01-01", "buy", 100.0)    # invest $10k at $100 -> 100 shares
    p.step("2024-01-08", "hold", 110.0)   # value now 100 * 110 = $11,000
    p.step("2024-01-15", "hold", 120.0)   # value now 100 * 120 = $12,000

    final_value = p.value(120.0)
    print(f"  Final value: ${final_value:,.2f} (expected $12,000.00)")
    assert abs(final_value - 12_000.0) < 0.01, "Buy-and-hold math wrong"
    print("  PASS\n")


def test_buy_sell_locks_in_gain():
    """Buy at $100, sell at $120 -> locked in $12k, later price moves don't matter."""
    print("Test 2: sell locks in the gain")
    p = Portfolio(starting_cash=10_000.0)
    p.step("2024-01-01", "buy", 100.0)
    p.step("2024-01-08", "sell", 120.0)   # sell 100 shares at $120 -> $12,000 cash
    p.step("2024-01-15", "hold", 80.0)    # price crashed, but we're in cash

    final_value = p.value(80.0)
    print(f"  Final value: ${final_value:,.2f} (expected $12,000.00)")
    assert abs(final_value - 12_000.0) < 0.01, "Sell did not lock in gain"
    print("  PASS\n")


def test_metrics_simple():
    """A portfolio going 10000 -> 11000 -> 12100 is +10% per period, +21% total."""
    print("Test 3: metrics on a known series")
    series = pd.Series(
        [10_000.0, 11_000.0, 12_100.0],
        index=pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]),
    )
    m = compute_metrics(series, periods_per_year=52.0)

    print(f"  Cumulative return: {m.cumulative_return * 100:.2f}% (expected 21.00%)")
    assert abs(m.cumulative_return - 0.21) < 0.001, "Cumulative return wrong"

    # Both period returns are exactly +10%, so volatility is 0 -> Sharpe = 0
    print(f"  Sharpe: {m.sharpe_ratio:.2f} (expected 0.00, zero volatility)")
    assert abs(m.sharpe_ratio) < 0.001, "Sharpe should be 0 for constant returns"

    # Steadily rising -> no drawdown
    print(f"  Max drawdown: {m.max_drawdown * 100:.2f}% (expected 0.00%)")
    assert abs(m.max_drawdown) < 0.001, "Should be no drawdown on a rising series"
    print("  PASS\n")


def test_drawdown():
    """A portfolio that peaks then falls should show the correct drawdown."""
    print("Test 4: drawdown calculation")
    series = pd.Series(
        [10_000.0, 12_000.0, 9_000.0, 11_000.0],  # peak 12k, trough 9k
        index=pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"]),
    )
    m = compute_metrics(series, periods_per_year=52.0)
    # Worst drawdown: from 12,000 down to 9,000 = -25%
    print(f"  Max drawdown: {m.max_drawdown * 100:.2f}% (expected -25.00%)")
    assert abs(m.max_drawdown - (-0.25)) < 0.001, "Drawdown math wrong"
    print("  PASS\n")


def test_buy_and_hold_baseline():
    """Buy-and-hold baseline: $10k at $100, price doubles -> $20k."""
    print("Test 5: buy-and-hold baseline")
    prices = pd.Series(
        [100.0, 150.0, 200.0],
        index=pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]),
    )
    bh = buy_and_hold_value_series(prices, starting_cash=10_000.0)
    print(f"  Final value: ${bh.iloc[-1]:,.2f} (expected $20,000.00)")
    assert abs(bh.iloc[-1] - 20_000.0) < 0.01, "Buy-and-hold baseline wrong"
    print("  PASS\n")


def main():
    print("=" * 60)
    print("Portfolio Simulator Unit Test")
    print("=" * 60)
    print()
    test_buy_then_hold()
    test_buy_sell_locks_in_gain()
    test_metrics_simple()
    test_drawdown()
    test_buy_and_hold_baseline()
    print("=" * 60)
    print("All portfolio math verified. Ready for the backtest runner.")
    print("=" * 60)


if __name__ == "__main__":
    main()