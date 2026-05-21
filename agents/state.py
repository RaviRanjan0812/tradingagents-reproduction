"""
Shared state for the multi-agent pipeline.

How the pipeline works:
    AgentState starts mostly empty.
    Each agent reads what it needs and writes its output back.
    By the end, all fields are populated and the final decision is in `final_decision`.

Why Pydantic instead of a plain dict:
    - Type checking catches "I spelled the field name wrong" bugs at the source
    - Validation catches "the LLM returned garbage" bugs immediately
    - Field defaults make partial states unambiguous (a missing report is `None`,
      not a KeyError later)
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ============================================================================
# REPORT SHAPES
# ============================================================================
# Each analyst returns a report with this shape. Keeping it consistent across
# analysts means the trader can iterate over them uniformly.

Signal = Literal["bullish", "bearish", "neutral"]


class AnalystReport(BaseModel):
    """One analyst's report. Shape is identical across all four analysts."""

    signal: Signal = Field(
        description="The analyst's overall directional view"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How confident the analyst is, 0.0 to 1.0"
    )
    summary: str = Field(
        description="One-paragraph executive summary"
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="Bullet points of supporting evidence (3-5 items)"
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Things that could invalidate this view (1-3 items)"
    )


# ============================================================================
# AGENT STATE
# ============================================================================

class AgentState(BaseModel):
    """
    The state object that flows through the pipeline.

    Read this top-to-bottom and you can see the entire system in one place:
    inputs at the top, intermediate outputs in the middle, final decision at
    the bottom.
    """

    # --- Inputs ---
    ticker: str
    as_of_date: str  # YYYY-MM-DD format
    had_agent_failure: bool = False  # set True if any agent hit an API error

    # --- Analyst outputs (Turn 3 — that's now) ---
    fundamentals_report: Optional[AnalystReport] = None
    sentiment_report: Optional[AnalystReport] = None
    news_report: Optional[AnalystReport] = None
    technical_report: Optional[AnalystReport] = None

    # --- Researcher outputs (Turn 4) ---
    bull_argument: Optional[str] = None
    bear_argument: Optional[str] = None
    debate_winner: Optional[Literal["bull", "bear", "neither"]] = None
    debate_rationale: Optional[str] = None

    # --- Trader output (Turn 5) ---
    trader_proposal: Optional[str] = None
    trader_action: Optional[Literal["buy", "sell", "hold"]] = None
    trader_confidence: Optional[float] = None

    # --- Risk team outputs (Turn 5) ---
    risk_aggressive_view: Optional[str] = None
    risk_neutral_view: Optional[str] = None
    risk_conservative_view: Optional[str] = None

    # --- Final decision (Turn 5) ---
    final_decision: Optional[Literal["buy", "sell", "hold"]] = None
    final_rationale: Optional[str] = None

    def all_analyst_reports(self) -> dict[str, Optional[AnalystReport]]:
        """Return all four analyst reports in a dict, in a stable order."""
        return {
            "fundamentals": self.fundamentals_report,
            "sentiment": self.sentiment_report,
            "news": self.news_report,
            "technical": self.technical_report,
        }