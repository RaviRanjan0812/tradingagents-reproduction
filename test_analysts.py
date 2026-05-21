"""Smoke test for the four analyst agents."""

import time
from agents import (
    AgentState,
    FundamentalsAnalyst,
    SentimentAnalyst,
    NewsAnalyst,
    TechnicalAnalyst,
)


def pretty_print_report(name, report):
    print(f"\n--- {name} ---")
    print(f"  signal:     {report.signal}")
    print(f"  confidence: {report.confidence:.2f}")
    print(f"  summary:    {report.summary}")
    print(f"  key_points:")
    for kp in report.key_points:
        print(f"    - {kp}")
    print(f"  risks:")
    for r in report.risks:
        print(f"    - {r}")


def main():
    print("=" * 60)
    print("Analyst Agents Smoke Test")
    print("=" * 60)
    print("\nScenario: AAPL on 2024-03-01\n")

    state = AgentState(ticker="AAPL", as_of_date="2024-03-01")

    analysts = [
        ("Fundamentals", FundamentalsAnalyst()),
        ("Technical",    TechnicalAnalyst()),
        ("News",         NewsAnalyst()),
        ("Sentiment",    SentimentAnalyst()),
    ]

    for name, analyst in analysts:
        print(f"\nRunning {name} analyst...")
        start = time.time()
        report = analyst.run(state)
        duration = time.time() - start
        print(f"  ({duration:.1f}s)")
        pretty_print_report(name, report)

    print("\n" + "=" * 60)
    print("Done. Read the reports above.")
    print("=" * 60)


if __name__ == "__main__":
    main()