"""
Smoke test for the backtest runner.

Deliberately TINY: a ~2-week window on AAPL, so it only makes 1-3 pipeline
calls (and the 2024-03-01 one is already cached from earlier turns). This
verifies the runner mechanics — date generation, checkpointing, portfolio
simulation, metrics — WITHOUT committing to a full 52-decision experiment.

Usage:
    python test_backtest.py
"""

from backtest import run_backtest, weekly_decision_dates


def main():
    print("=" * 60)
    print("Backtest Runner Smoke Test")
    print("=" * 60)

    # First, just check date generation (no LLM calls).
    print("\nChecking weekly date generation...")
    dates = weekly_decision_dates("AAPL", "2024-02-15", "2024-03-08")
    print(f"  Generated {len(dates)} weekly decision dates: {dates}")
    assert len(dates) >= 2, "Expected at least 2 weekly dates in this window"

    # Now a tiny real backtest. This window is short on purpose.
    print("\nRunning tiny backtest (this makes a few real pipeline calls)...")
    result = run_backtest(
        ticker="AAPL",
        regime_name="smoketest",
        start="2024-02-15",
        end="2024-03-08",
        verbose=True,
    )

    # --- Verify the result ---
    print("-" * 60)
    print("RESULT CHECKS")
    print("-" * 60)
    assert result.ticker == "AAPL"
    assert len(result.decisions) == len(result.decision_dates)
    for d in result.decisions:
        assert d["decision"] in ("buy", "sell", "hold"), \
            f"Bad decision: {d['decision']}"
        assert d["price"] > 0, "Price should be positive"
    print(f"  All {len(result.decisions)} decisions valid (buy/sell/hold).")
    print(f"  Agent Sharpe: {result.agent_sharpe:.2f}, "
          f"Buy&Hold Sharpe: {result.bh_sharpe:.2f}")

    print("\n" + "=" * 60)
    print("Backtest runner works. Ready for the full experiment (Turn 8).")
    print("Checkpoint + result files were written to the results/ folder.")
    print("=" * 60)


if __name__ == "__main__":
    main()