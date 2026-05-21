"""Agents package."""

from .state import AgentState, AnalystReport
from .analysts import (
    FundamentalsAnalyst, SentimentAnalyst, NewsAnalyst, TechnicalAnalyst,
)
from .researchers import (
    BullResearcher, BearResearcher, DebateFacilitator, DebateJudgment,
)
from .decision import (
    Trader, TraderProposal,
    AggressiveRiskAnalyst, NeutralRiskAnalyst, ConservativeRiskAnalyst,
    FundManager, FinalDecision,
)

__all__ = [
    "AgentState", "AnalystReport",
    "FundamentalsAnalyst", "SentimentAnalyst", "NewsAnalyst", "TechnicalAnalyst",
    "BullResearcher", "BearResearcher", "DebateFacilitator", "DebateJudgment",
    "Trader", "TraderProposal",
    "AggressiveRiskAnalyst", "NeutralRiskAnalyst", "ConservativeRiskAnalyst",
    "FundManager", "FinalDecision",
]