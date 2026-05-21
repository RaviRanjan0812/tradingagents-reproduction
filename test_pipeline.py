"""
Smoke test for the pipeline orchestrator.

Verifies that run_pipeline() executes the full 11-agent sequence and returns
a well-formed PipelineResult.

Usage:
    python test_pipeline.py
"""

from pipeline import run_pipeline
from agents import AgentState


def main():
    print("=" * 60)
    print("Pipeline Orchestrator Smoke Test")
    print("=" * 60)
    print("\nScenario: AAPL on 2024-03-01 (verbose mode)\n")

    result = run_pipeline("AAPL", "2024-03-01", verbose=True)

    # --- Verify the result object ---
    print("\n" + "-" * 60)
    print("RESULT OBJECT CHECKS")
    print("-" * 60)

    assert result.state is not None, "Result should carry a state"
    assert isinstance(result.state, AgentState), "state should be an AgentState"
    assert result.decision in ("buy", "sell", "hold"), \
        f"decision should be buy/sell/hold, got {result.decision!r}"
    assert result.elapsed_seconds > 0, "elapsed_seconds should be positive"

    # Verify the state is fully populated
    s = result.state
    populated_checks = {
        "fundamentals_report": s.fundamentals_report is not None,
        "technical_report": s.technical_report is not None,
        "news_report": s.news_report is not None,
        "sentiment_report": s.sentiment_report is not None,
        "bull_argument": s.bull_argument is not None,
        "bear_argument": s.bear_argument is not None,
        "debate_winner": s.debate_winner is not None,
        "trader_action": s.trader_action is not None,
        "risk_aggressive_view": s.risk_aggressive_view is not None,
        "risk_neutral_view": s.risk_neutral_view is not None,
        "risk_conservative_view": s.risk_conservative_view is not None,
        "final_decision": s.final_decision is not None,
    }
    for field_name, is_populated in populated_checks.items():
        status = "OK" if is_populated else "MISSING"
        print(f"  {field_name:.<30} {status}")
        assert is_populated, f"State field {field_name} was not populated"

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Final decision:  {result.decision.upper()}")
    print(f"  Total time:      {result.elapsed_seconds:.1f}s")
    print(f"  Stage timings:")
    for stage, secs in result.stage_timings.items():
        print(f"    {stage:.<20} {secs:.1f}s")
    print("=" * 60)
    print("\nPipeline works. run_pipeline() is ready for the backtester.")


if __name__ == "__main__":
    main()