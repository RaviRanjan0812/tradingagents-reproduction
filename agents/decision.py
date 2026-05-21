"""
The decision layer: Trader, Risk Team (aggressive/neutral/conservative),
and Fund Manager.

Communication protocol (consistent with earlier turns):
- Trader  -> structured JSON (downstream agents must parse the action)
- Risk team -> free text critiques (these are arguments, like the researchers)
- Fund Manager -> structured JSON (the final decision must be machine-readable)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ValidationError

from llm_router import call_llm
from .state import AgentState


_PROMPTS_DIR = Path(__file__).parent / "prompts"


# ============================================================================
# SHARED HELPERS  (same pattern as researchers.py)
# ============================================================================

def _load_template(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _fill_template(template: str, variables: dict) -> str:
    result = template
    for key, value in variables.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def _extract_json(response: str) -> dict:
    """Strip markdown fences and isolate the JSON object."""
    cleaned = response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise json.JSONDecodeError("No JSON object found", cleaned, 0)
    return json.loads(cleaned[first:last + 1])


def _analyst_summary(state: AgentState) -> str:
    """One-line-per-analyst signal summary, used by the risk agents."""
    lines = []
    for name, report in state.all_analyst_reports().items():
        if report is None:
            lines.append(f"  {name}: (no report)")
        else:
            lines.append(
                f"  {name}: {report.signal} (confidence {report.confidence:.2f})"
            )
    return "\n".join(lines)


def _full_analyst_reports(state: AgentState) -> str:
    """Detailed analyst reports, used by the trader."""
    blocks = []
    for name, report in state.all_analyst_reports().items():
        if report is None:
            blocks.append(f"[{name.upper()}]: (no report available)")
            continue
        kp = "; ".join(report.key_points) or "(none)"
        blocks.append(
            f"[{name.upper()}] signal={report.signal} "
            f"confidence={report.confidence:.2f}\n"
            f"  Summary: {report.summary}\n"
            f"  Key points: {kp}"
        )
    return "\n\n".join(blocks)


# ============================================================================
# TRADER
# ============================================================================

class TraderProposal(BaseModel):
    action: str       # "buy" | "sell" | "hold"
    confidence: float
    reasoning: str


class Trader:
    role = "trader"

    def run(self, state: AgentState) -> TraderProposal:
        template = _load_template("trader.txt")
        prompt = _fill_template(template, {
            "ticker": state.ticker,
            "as_of_date": state.as_of_date,
            "analyst_reports": _full_analyst_reports(state),
            "debate_winner": state.debate_winner or "none",
            "debate_rationale": state.debate_rationale or "none",
            "bull_argument": state.bull_argument or "(no bull argument)",
            "bear_argument": state.bear_argument or "(no bear argument)",
        })
        try:
            response = call_llm(prompt=prompt, role=self.role)
        except Exception as e:
            print(f"  [trader] LLM call failed: {e}")
            state.had_agent_failure = True
            return self._fallback(f"LLM call failed: {e}")
        try:
            data = _extract_json(response)
            proposal = TraderProposal(**data)
            if proposal.action not in ("buy", "sell", "hold"):
                raise ValueError(f"Invalid action: {proposal.action}")
            return proposal
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            print(f"  [trader] Parse failed: {e}")
            print(f"  [trader] Raw response: {response[:300]}")
            state.had_agent_failure = True
            return self._fallback(f"Parse failed: {e}")

    def _fallback(self, error_message: str) -> TraderProposal:
        # When the trader fails, default to "hold" — the safe, do-nothing action.
        return TraderProposal(
            action="hold",
            confidence=0.0,
            reasoning=f"[Trader could not decide: {error_message[:100]}] "
                      f"Defaulting to hold.",
        )


# ============================================================================
# RISK TEAM
# ============================================================================

class _RiskAnalyst:
    """Base for the three risk analysts. They differ only by prompt + role."""

    role: str = ""
    prompt_file: str = ""

    def run(self, state: AgentState) -> str:
        if state.trader_action is None:
            raise ValueError("Risk analyst needs trader_action set in state.")
        template = _load_template(self.prompt_file)
        prompt = _fill_template(template, {
            "ticker": state.ticker,
            "as_of_date": state.as_of_date,
            "trader_action": state.trader_action,
            "trader_confidence": (
                f"{state.trader_confidence:.2f}"
                if state.trader_confidence is not None else "n/a"
            ),
            "trader_reasoning": state.trader_proposal or "(no reasoning)",
            "analyst_summary": _analyst_summary(state),
        })
        try:
            return call_llm(prompt=prompt, role=self.role)
        except Exception as e:
            print(f"  [{self.role}] LLM call failed: {e}")
            state.had_agent_failure = True
            return f"[{self.role} unavailable: {e}]\nSTANCE: endorse"


class AggressiveRiskAnalyst(_RiskAnalyst):
    role = "risk_aggressive"
    prompt_file = "risk_aggressive.txt"


class NeutralRiskAnalyst(_RiskAnalyst):
    role = "risk_neutral"
    prompt_file = "risk_neutral.txt"


class ConservativeRiskAnalyst(_RiskAnalyst):
    role = "risk_conservative"
    prompt_file = "risk_conservative.txt"


# ============================================================================
# FUND MANAGER
# ============================================================================

class FinalDecision(BaseModel):
    final_decision: str   # "buy" | "sell" | "hold"
    rationale: str


class FundManager:
    role = "fund_manager"

    def run(self, state: AgentState) -> FinalDecision:
        missing = [
            n for n, v in [
                ("trader_action", state.trader_action),
                ("risk_aggressive_view", state.risk_aggressive_view),
                ("risk_neutral_view", state.risk_neutral_view),
                ("risk_conservative_view", state.risk_conservative_view),
            ] if v is None
        ]
        if missing:
            raise ValueError(
                f"FundManager needs these set in state first: {missing}"
            )

        template = _load_template("fund_manager.txt")
        prompt = _fill_template(template, {
            "ticker": state.ticker,
            "as_of_date": state.as_of_date,
            "trader_action": state.trader_action,
            "trader_confidence": (
                f"{state.trader_confidence:.2f}"
                if state.trader_confidence is not None else "n/a"
            ),
            "trader_reasoning": state.trader_proposal or "(no reasoning)",
            "risk_aggressive": state.risk_aggressive_view,
            "risk_neutral": state.risk_neutral_view,
            "risk_conservative": state.risk_conservative_view,
        })
        try:
            response = call_llm(prompt=prompt, role=self.role)
        except Exception as e:
            print(f"  [fund_manager] LLM call failed: {e}")
            state.had_agent_failure = True
            return self._fallback(f"LLM call failed: {e}", state)
        try:
            data = _extract_json(response)
            decision = FinalDecision(**data)
            if decision.final_decision not in ("buy", "sell", "hold"):
                raise ValueError(f"Invalid decision: {decision.final_decision}")
            return decision
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            print(f"  [fund_manager] Parse failed: {e}")
            print(f"  [fund_manager] Raw response: {response[:300]}")
            state.had_agent_failure = True
            return self._fallback(f"Parse failed: {e}", state)

    def _fallback(self, error_message: str, state: AgentState) -> FinalDecision:
        # When the fund manager fails, fall back to the trader's action if we
        # have it, else "hold". This way a parse failure doesn't erase the
        # decision the rest of the pipeline already produced.
        fallback_action = state.trader_action or "hold"
        return FinalDecision(
            final_decision=fallback_action,
            rationale=f"[Fund manager could not decide: {error_message[:100]}] "
                      f"Falling back to trader's action: {fallback_action}.",
        )