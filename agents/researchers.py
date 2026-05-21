"""
The researcher team: bull researcher, bear researcher, debate facilitator.

Design notes:
- Bull and Bear researchers return FREE TEXT (an argument is prose, not a
  structured report). They each also emit a CONVICTION score we parse out.
- The Facilitator returns STRUCTURED JSON (the trader downstream needs a
  clean winner signal). This is the "hybrid communication protocol" — free
  where reasoning matters, structured where the next agent must parse.
- All three read the shared AgentState. Researchers read analyst reports;
  the facilitator reads the two arguments.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ValidationError

from llm_router import call_llm
from .state import AgentState, AnalystReport


_PROMPTS_DIR = Path(__file__).parent / "prompts"


# ============================================================================
# HELPERS
# ============================================================================

def _load_template(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}. Did you create it?"
        )
    return path.read_text(encoding="utf-8")


def _fill_template(template: str, variables: dict) -> str:
    """Simple {key} substitution. Avoids str.format() which chokes on the
    literal {} in JSON examples inside the prompt files."""
    result = template
    for key, value in variables.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def _format_analyst_reports(state: AgentState) -> str:
    """Render the four analyst reports into readable text for the researchers."""
    blocks = []
    for name, report in state.all_analyst_reports().items():
        if report is None:
            blocks.append(f"[{name.upper()} ANALYST]: (no report available)")
            continue
        kp = "\n".join(f"  - {p}" for p in report.key_points) or "  (none)"
        risks = "\n".join(f"  - {r}" for r in report.risks) or "  (none)"
        blocks.append(
            f"[{name.upper()} ANALYST]\n"
            f"  Signal: {report.signal} (confidence {report.confidence:.2f})\n"
            f"  Summary: {report.summary}\n"
            f"  Key points:\n{kp}\n"
            f"  Risks:\n{risks}"
        )
    return "\n\n".join(blocks)


def _extract_conviction(argument_text: str) -> int | None:
    """Pull the 'CONVICTION: N' line out of a researcher's argument.
    Returns None if not found (we don't crash — conviction is a nice-to-have)."""
    match = re.search(r"CONVICTION:\s*(\d+)", argument_text, re.IGNORECASE)
    if match:
        value = int(match.group(1))
        return max(1, min(10, value))  # clamp to 1-10
    return None


# ============================================================================
# RESEARCHERS
# ============================================================================

class BullResearcher:
    """Argues the strongest honest case to BUY."""

    role = "bull_researcher"

    def run(self, state: AgentState) -> str:
        template = _load_template("bull_researcher.txt")
        prompt = _fill_template(template, {
            "ticker": state.ticker,
            "as_of_date": state.as_of_date,
            "analyst_reports": _format_analyst_reports(state),
        })
        try:
            return call_llm(prompt=prompt, role=self.role)
        except Exception as e:
            print(f"  [bull_researcher] LLM call failed: {e}")
            state.had_agent_failure = True
            return f"[Bull researcher unavailable: {e}]\nCONVICTION: 1"


class BearResearcher:
    """Argues the strongest honest case to SELL/AVOID."""

    role = "bear_researcher"

    def run(self, state: AgentState) -> str:
        template = _load_template("bear_researcher.txt")
        prompt = _fill_template(template, {
            "ticker": state.ticker,
            "as_of_date": state.as_of_date,
            "analyst_reports": _format_analyst_reports(state),
        })
        try:
            return call_llm(prompt=prompt, role=self.role)
        except Exception as e:
            print(f"  [bear_researcher] LLM call failed: {e}")
            state.had_agent_failure = True
            return f"[Bear researcher unavailable: {e}]\nCONVICTION: 1"


# ============================================================================
# DEBATE FACILITATOR
# ============================================================================

class DebateJudgment(BaseModel):
    """Structured output of the facilitator."""
    winner: str          # "bull" | "bear" | "neither"
    rationale: str
    bull_strength: int
    bear_strength: int


class DebateFacilitator:
    """Neutral judge — reads both arguments, declares a winner."""

    role = "fund_manager"  # reuse a capable model slot for judgment

    def run(self, state: AgentState) -> DebateJudgment:
        if state.bull_argument is None or state.bear_argument is None:
            raise ValueError(
                "DebateFacilitator needs both bull_argument and bear_argument "
                "set in state. Run the researchers first."
            )

        template = _load_template("debate_facilitator.txt")
        prompt = _fill_template(template, {
            "ticker": state.ticker,
            "as_of_date": state.as_of_date,
            "bull_argument": state.bull_argument,
            "bear_argument": state.bear_argument,
        })

        try:
            response = call_llm(prompt=prompt, role=self.role)
        except Exception as e:
            print(f"  [debate_facilitator] LLM call failed: {e}")
            state.had_agent_failure = True
            return self._fallback(f"LLM call failed: {e}")

        try:
            return self._parse(response)
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"  [debate_facilitator] Parse failed: {e}")
            print(f"  [debate_facilitator] Raw response: {response[:300]}")
            state.had_agent_failure = True
            return self._fallback(f"Parse failed: {e}")

    def _parse(self, response: str) -> DebateJudgment:
        cleaned = response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first == -1 or last == -1 or last <= first:
            raise json.JSONDecodeError("No JSON object found", cleaned, 0)
        data = json.loads(cleaned[first:last + 1])

        judgment = DebateJudgment(**data)
        # Validate the winner field is one of the allowed values
        if judgment.winner not in ("bull", "bear", "neither"):
            raise ValidationError.from_exception_data("DebateJudgment", [])
        return judgment

    def _fallback(self, error_message: str) -> DebateJudgment:
        """When judgment fails, default to 'neither' — a non-committal stance
        that lets the trader fall back on the raw analyst reports."""
        return DebateJudgment(
            winner="neither",
            rationale=f"[Facilitator could not judge: {error_message[:100]}]",
            bull_strength=5,
            bear_strength=5,
        )