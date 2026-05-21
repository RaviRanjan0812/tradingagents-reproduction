"""
Multi-regime experiment runner.

Runs the full robustness experiment:
    2 tickers (AAPL, JPM) x 2 regimes (2024 bull, 2022 bear) = 4 backtests.

Each backtest is resumable (backtest.py checkpoints after every decision),
so if this script is interrupted, just run it again — completed decisions
are loaded from disk and not recomputed.

Usage:
    python run_experiment.py            # run all 4
    python run_experiment.py AAPL       # run only AAPL's 2 regimes
    python run_experiment.py JPM        # run only JPM's 2 regimes
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

from config import RESULTS_DIR
from backtest import run_backtest


# ============================================================================
# EXPERIMENT DESIGN
# ============================================================================
# Each entry: (ticker, regime_name, start_date, end_date)

REGIMES = [
    # 2024 bull market — the paper's evaluation window
    ("2024_bull", "2024-01-02", "2024-03-29"),
    # 2022 bear market — the tech selloff / rate-hike drawdown
    ("2022_bear", "2022-01-03", "2022-03-31"),
]

TICKERS = ["AAPL", "JPM"]


# ============================================================================
# RUNNER
# ============================================================================

def run_all(tickers: list[str]) -> list:
    """Run backtests for the given tickers across both regimes."""
    results = []
    total = len(tickers) * len(REGIMES)
    n = 0

    for ticker in tickers:
        for regime_name, start, end in REGIMES:
            n += 1
            print(f"\n{'#'*60}")
            print(f"# EXPERIMENT RUN {n}/{total}: {ticker} {regime_name}")
            print(f"{'#'*60}")

            result = run_backtest(
                ticker=ticker,
                regime_name=regime_name,
                start=start,
                end=end,
                verbose=True,
            )
            results.append(result)

    return results


def print_comparison_table(results: list) -> None:
    """Print the headline comparison: agents vs buy-and-hold, all runs."""
    print("\n\n" + "=" * 72)
    print("EXPERIMENT SUMMARY — Agents vs Buy & Hold")
    print("=" * 72)
    header = (
        f"{'Ticker':<8}{'Regime':<14}"
        f"{'Agent Ret':>12}{'B&H Ret':>12}"
        f"{'Agent Shrp':>12}{'B&H Shrp':>12}"
    )
    print(header)
    print("-" * 72)
    for r in results:
        print(
            f"{r.ticker:<8}{r.regime_name:<14}"
            f"{r.agent_cumulative_return*100:>11.2f}%"
            f"{r.bh_cumulative_return*100:>11.2f}%"
            f"{r.agent_sharpe:>12.2f}{r.bh_sharpe:>12.2f}"
        )
    print("-" * 72)

    # Also show drawdowns — important for the regime story
    print("\nDrawdown comparison:")
    print(f"{'Ticker':<8}{'Regime':<14}{'Agent MaxDD':>14}{'B&H MaxDD':>14}")
    print("-" * 50)
    for r in results:
        print(
            f"{r.ticker:<8}{r.regime_name:<14}"
            f"{r.agent_max_drawdown*100:>13.2f}%"
            f"{r.bh_max_drawdown*100:>13.2f}%"
        )
    print("=" * 72)


def print_decision_breakdown(results: list) -> None:
    """Show what the agents actually DID — count of buy/sell/hold per run.
    Critical sanity check: if everything is 'hold', the experiment is void."""
    print("\nDecision breakdown (did the agents actually trade?):")
    print(f"{'Ticker':<8}{'Regime':<14}{'buy':>6}{'sell':>6}{'hold':>6}")
    print("-" * 40)
    for r in results:
        counts = {"buy": 0, "sell": 0, "hold": 0}
        for d in r.decisions:
            counts[d["decision"]] = counts.get(d["decision"], 0) + 1
        print(
            f"{r.ticker:<8}{r.regime_name:<14}"
            f"{counts['buy']:>6}{counts['sell']:>6}{counts['hold']:>6}"
        )
    print("-" * 40)


def main():
    # Optional ticker filter from command line
    if len(sys.argv) > 1:
        requested = [t.upper() for t in sys.argv[1:]]
        tickers = [t for t in requested if t in TICKERS]
        if not tickers:
            print(f"Unknown ticker(s) {requested}. Valid: {TICKERS}")
            return
    else:
        tickers = TICKERS

    print("=" * 72)
    print("MULTI-REGIME ROBUSTNESS EXPERIMENT")
    print(f"  Tickers: {tickers}")
    print(f"  Regimes: {[r[0] for r in REGIMES]}")
    print("=" * 72)

    results = run_all(tickers)

    print_comparison_table(results)
    print_decision_breakdown(results)

    # Save a combined summary
    summary = [asdict(r) for r in results]
    summary_path = RESULTS_DIR / "experiment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary saved to: {summary_path}")
    print("\nExperiment complete. Review the tables above for the writeup.")


if __name__ == "__main__":
    main()