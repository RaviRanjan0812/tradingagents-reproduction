"""
Smoke test for the decision layer — and the first END-TO-END pipeline run.

Chain: 4 analysts -> 2 researchers -> facilitator -> trader
       -> 3 risk analysts -> fund manager -> FINAL DECISION

This is the first time the whole system runs start to finish.
"""

import time
from agents import (
    AgentState,
    FundamentalsAnalyst, SentimentAnalyst, NewsAnalyst, TechnicalAnalyst,
    BullResearcher, BearResearcher, DebateFacilitator,
    Trader, AggressiveRiskAnalyst, NeutralRiskAnalyst, ConservativeRiskAnalyst,
    FundManager,
)


def main():
    print("=" * 60)
    print("FULL PIPELINE Smoke Test")
    print("=" * 60)
    print("\nScenario: AAPL on 2024-03-01\n")

    state = AgentState(ticker="AAPL", as_of_date="2024-03-01")
    t0 = time.time()

    # --- Analysts ---
    print("[1/5] Running 4 analysts...")
    state.fundamentals_report = FundamentalsAnalyst().run(state)
    state.technical_report = TechnicalAnalyst().run(state)
    state.news_report = NewsAnalyst().run(state)
    state.sentiment_report = SentimentAnalyst().run(state)
    print(f"      signals: "
          f"F={state.fundamentals_report.signal}, "
          f"T={state.technical_report.signal}, "
          f"N={state.news_report.signal}, "
          f"S={state.sentiment_report.signal}")

    # --- Researchers + facilitator ---
    print("[2/5] Running bull + bear researchers...")
    state.bull_argument = BullResearcher().run(state)
    state.bear_argument = BearResearcher().run(state)
    print("[2/5] Running debate facilitator...")
    judgment = DebateFacilitator().run(state)
    state.debate_winner = judgment.winner
    state.debate_rationale = judgment.rationale
    print(f"      debate winner: {judgment.winner} "
          f"(bull {judgment.bull_strength}/10, bear {judgment.bear_strength}/10)")

    # --- Trader ---
    print("[3/5] Running trader...")
    proposal = Trader().run(state)
    state.trader_proposal = proposal.reasoning
    state.trader_action = proposal.action
    state.trader_confidence = proposal.confidence
    print(f"      trader proposes: {proposal.action.upper()} "
          f"(confidence {proposal.confidence:.2f})")
    print(f"      reasoning: {proposal.reasoning}")

    # --- Risk team ---
    print("[4/5] Running 3 risk analysts...")
    state.risk_aggressive_view = AggressiveRiskAnalyst().run(state)
    state.risk_neutral_view = NeutralRiskAnalyst().run(state)
    state.risk_conservative_view = ConservativeRiskAnalyst().run(state)
    print("\n      --- AGGRESSIVE ---")
    print("      " + state.risk_aggressive_view.replace("\n", "\n      "))
    print("\n      --- NEUTRAL ---")
    print("      " + state.risk_neutral_view.replace("\n", "\n      "))
    print("\n      --- CONSERVATIVE ---")
    print("      " + state.risk_conservative_view.replace("\n", "\n      "))

    # --- Fund manager ---
    print("\n[5/5] Running fund manager...")
    decision = FundManager().run(state)
    state.final_decision = decision.final_decision
    state.final_rationale = decision.rationale

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("FINAL DECISION")
    print("=" * 60)
    print(f"  Ticker:        {state.ticker}")
    print(f"  Date:          {state.as_of_date}")
    print(f"  Trader said:   {state.trader_action}")
    print(f"  Fund manager:  {state.final_decision.upper()}")
    print(f"  Rationale:     {state.final_rationale}")
    print("=" * 60)
    print(f"\nFull pipeline ran in {elapsed:.1f}s. The system works end to end.")


if __name__ == "__main__":
    main()