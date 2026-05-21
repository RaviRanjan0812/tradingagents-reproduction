"""
Pipeline orchestrator.

Wraps the full 11-agent sequence into a single callable:

    run_pipeline(ticker, as_of_date) -> AgentState

The returned AgentState has every field populated, with the final
buy/sell/hold in state.final_decision.

Design note: this is a LINEAR orchestrator written in plain Python. The
paper uses LangGraph; we don't, because our flow has no branching or loops
— it is a straight sequence. LangGraph would be the right tool if the
architecture grew conditional paths (e.g. skip the debate when analysts
unanimously agree). Keeping it plain Python keeps it readable and dependency-
free. The agents themselves are framework-agnostic, so swapping in LangGraph
later would be a wrapper change, not a rewrite.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from agents import (
    AgentState,
    FundamentalsAnalyst, SentimentAnalyst, NewsAnalyst, TechnicalAnalyst,
    BullResearcher, BearResearcher, DebateFacilitator,
    Trader, AggressiveRiskAnalyst, NeutralRiskAnalyst, ConservativeRiskAnalyst,
    FundManager,
)


# ============================================================================
# RESULT WRAPPER
# ============================================================================

@dataclass
class PipelineResult:
    """The output of one pipeline run: the final state plus run metadata."""
    state: AgentState
    elapsed_seconds: float
    stage_timings: dict = field(default_factory=dict)

    @property
    def decision(self) -> str:
        """Convenience accessor for the final buy/sell/hold."""
        return self.state.final_decision or "hold"


# ============================================================================
# THE PIPELINE
# ============================================================================

class Pipeline:
    """Runs the full 11-agent decision pipeline for one (ticker, date).

    Agents are instantiated once at construction and reused across runs —
    they are stateless, so this is safe and avoids re-creating objects on
    every backtest step.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # Analyst team
        self.fundamentals = FundamentalsAnalyst()
        self.technical = TechnicalAnalyst()
        self.news = NewsAnalyst()
        self.sentiment = SentimentAnalyst()

        # Research team
        self.bull = BullResearcher()
        self.bear = BearResearcher()
        self.facilitator = DebateFacilitator()

        # Decision team
        self.trader = Trader()
        self.risk_aggressive = AggressiveRiskAnalyst()
        self.risk_neutral = NeutralRiskAnalyst()
        self.risk_conservative = ConservativeRiskAnalyst()
        self.fund_manager = FundManager()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"  {message}")

    def run(self, ticker: str, as_of_date: str) -> PipelineResult:
        """Execute the full pipeline. Returns a PipelineResult."""
        t_start = time.time()
        timings: dict = {}
        state = AgentState(ticker=ticker, as_of_date=as_of_date)

        # --- Stage 1: Analysts ---
        t = time.time()
        self._log("Stage 1/5: analysts")
        state.fundamentals_report = self.fundamentals.run(state)
        state.technical_report = self.technical.run(state)
        state.news_report = self.news.run(state)
        state.sentiment_report = self.sentiment.run(state)
        timings["analysts"] = time.time() - t
        self._log(
            f"  signals: F={state.fundamentals_report.signal} "
            f"T={state.technical_report.signal} "
            f"N={state.news_report.signal} "
            f"S={state.sentiment_report.signal}"
        )

        # --- Stage 2: Researchers + facilitator ---
        t = time.time()
        self._log("Stage 2/5: research debate")
        state.bull_argument = self.bull.run(state)
        state.bear_argument = self.bear.run(state)
        judgment = self.facilitator.run(state)
        state.debate_winner = judgment.winner
        state.debate_rationale = judgment.rationale
        timings["research"] = time.time() - t
        self._log(f"  debate winner: {judgment.winner}")

        # --- Stage 3: Trader ---
        t = time.time()
        self._log("Stage 3/5: trader")
        proposal = self.trader.run(state)
        state.trader_proposal = proposal.reasoning
        state.trader_action = proposal.action
        state.trader_confidence = proposal.confidence
        timings["trader"] = time.time() - t
        self._log(f"  trader proposes: {proposal.action}")

        # --- Stage 4: Risk team ---
        t = time.time()
        self._log("Stage 4/5: risk team")
        state.risk_aggressive_view = self.risk_aggressive.run(state)
        state.risk_neutral_view = self.risk_neutral.run(state)
        state.risk_conservative_view = self.risk_conservative.run(state)
        timings["risk"] = time.time() - t

        # --- Stage 5: Fund manager ---
        t = time.time()
        self._log("Stage 5/5: fund manager")
        decision = self.fund_manager.run(state)
        state.final_decision = decision.final_decision
        state.final_rationale = decision.rationale
        timings["fund_manager"] = time.time() - t
        self._log(f"  FINAL DECISION: {decision.final_decision}")

        elapsed = time.time() - t_start
        return PipelineResult(
            state=state,
            elapsed_seconds=elapsed,
            stage_timings=timings,
        )


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

# A module-level pipeline instance, so callers that just want a one-off
# decision don't have to manage a Pipeline object.
_default_pipeline: Pipeline | None = None


def run_pipeline(ticker: str, as_of_date: str, verbose: bool = False) -> PipelineResult:
    """
    Run the full decision pipeline for one (ticker, date).

    This is the main public entry point. The backtester calls this.

    Args:
        ticker: Stock symbol, e.g. "AAPL".
        as_of_date: Decision date, "YYYY-MM-DD". Point-in-time enforced.
        verbose: If True, print progress through the 5 stages.

    Returns:
        PipelineResult with .decision ("buy"/"sell"/"hold") and .state
        (the full populated AgentState for inspection).
    """
    global _default_pipeline
    if _default_pipeline is None or _default_pipeline.verbose != verbose:
        _default_pipeline = Pipeline(verbose=verbose)
    return _default_pipeline.run(ticker, as_of_date)