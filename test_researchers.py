"""
Smoke test for the researcher team (bull, bear, facilitator).

This test runs the full chain so far:
    4 analysts -> bull researcher -> bear researcher -> facilitator

Read the output as a real debate. Judge for yourself:
- Is the bull case honest, or does it ignore bearish data?
- Is the bear case honest?
- Does the facilitator's verdict seem fair given the two arguments?
"""

import time
from agents import (
    AgentState,
    FundamentalsAnalyst, SentimentAnalyst, NewsAnalyst, TechnicalAnalyst,
    BullResearcher, BearResearcher, DebateFacilitator,
)
from agents.researchers import _extract_conviction


def main():
    print("=" * 60)
    print("Researcher Team Smoke Test")
    print("=" * 60)
    print("\nScenario: AAPL on 2024-03-01\n")

    state = AgentState(ticker="AAPL", as_of_date="2024-03-01")

    # --- Run the four analysts to populate state ---
    print("Running 4 analysts to populate state...")
    state.fundamentals_report = FundamentalsAnalyst().run(state)
    state.technical_report = TechnicalAnalyst().run(state)
    state.news_report = NewsAnalyst().run(state)
    state.sentiment_report = SentimentAnalyst().run(state)
    print("  Analyst signals: "
          f"fundamentals={state.fundamentals_report.signal}, "
          f"technical={state.technical_report.signal}, "
          f"news={state.news_report.signal}, "
          f"sentiment={state.sentiment_report.signal}")

    # --- Bull researcher ---
    print("\nRunning bull researcher...")
    start = time.time()
    state.bull_argument = BullResearcher().run(state)
    print(f"  ({time.time() - start:.1f}s)")
    print("\n" + "-" * 60)
    print("BULL ARGUMENT:")
    print("-" * 60)
    print(state.bull_argument)
    print(f"\n  [parsed conviction: {_extract_conviction(state.bull_argument)}]")

    # --- Bear researcher ---
    print("\nRunning bear researcher...")
    start = time.time()
    state.bear_argument = BearResearcher().run(state)
    print(f"  ({time.time() - start:.1f}s)")
    print("\n" + "-" * 60)
    print("BEAR ARGUMENT:")
    print("-" * 60)
    print(state.bear_argument)
    print(f"\n  [parsed conviction: {_extract_conviction(state.bear_argument)}]")

    # --- Facilitator ---
    print("\nRunning debate facilitator...")
    start = time.time()
    judgment = DebateFacilitator().run(state)
    print(f"  ({time.time() - start:.1f}s)")
    state.debate_winner = judgment.winner
    state.debate_rationale = judgment.rationale

    print("\n" + "=" * 60)
    print("FACILITATOR VERDICT:")
    print("=" * 60)
    print(f"  Winner:        {judgment.winner}")
    print(f"  Bull strength: {judgment.bull_strength}/10")
    print(f"  Bear strength: {judgment.bear_strength}/10")
    print(f"  Rationale:     {judgment.rationale}")
    print("=" * 60)
    print("\nDone. Read the debate above and judge whether it's reasonable.")


if __name__ == "__main__":
    main()