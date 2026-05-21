"""
The four analyst agents.

Architecture:
    AnalystBase is the shared skeleton. Each concrete analyst subclasses it
    and only has to implement `_build_data_section(state)` — the part where
    it pulls and formats its specific slice of data.

    The base class handles:
    - Loading the prompt template from a .txt file
    - Calling the LLM via our router
    - Parsing JSON out of the response (LLMs sometimes wrap it in markdown)
    - Validating against the AnalystReport schema with Pydantic
    - Falling back to a "neutral, low-confidence" report if parsing fails

Why this design:
    - Adding a new analyst is just a new subclass + new prompt file
    - Bugs in JSON parsing get fixed in ONE place
    - Prompt iteration happens in .txt files, not Python strings
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from llm_router import call_llm
from data import get_prices, get_news, get_indicators
from .state import AgentState, AnalystReport


# ============================================================================
# PATHS
# ============================================================================

_PROMPTS_DIR = Path(__file__).parent / "prompts"


# ============================================================================
# BASE CLASS
# ============================================================================

class AnalystBase(ABC):
    """Shared base class for all four analysts.

    Concrete subclasses define class-level `role` and `prompt_file`, and
    implement `_build_data_section`.
    """

    # Subclasses MUST override these
    role: str = ""           # e.g. "fundamentals_analyst" — must match MODEL_ROUTING key
    prompt_file: str = ""    # e.g. "fundamentals_analyst.txt"

    def run(self, state: AgentState) -> AnalystReport:
        """
        Execute this analyst on the given state. Returns an AnalystReport.

        On unrecoverable errors (LLM fails entirely, malformed JSON we
        can't recover from), returns a neutral fallback report instead of
        crashing. This is important: a single bad analyst should not bring
        down the whole pipeline.
        """
        # 1. Load the prompt template
        template = self._load_template()

        # 2. Build the data section (subclass-specific)
        data_vars = self._build_data_section(state)

        # 3. Substitute variables into the template
        # Use simple replacement, not .format(), because prompt files contain
        # JSON examples with literal {} that .format() would try to interpret.
        prompt = template
        for key, value in data_vars.items():
            prompt = prompt.replace("{" + key + "}", str(value))
        # 4. Call the LLM
        try:
           response_text = call_llm(prompt=prompt, role=self.role)
        except Exception as e:
            print(f"  [{self.role}] LLM call failed: {e}")
            state.had_agent_failure = True
            return self._fallback_report(f"LLM call failed: {e}")

        # 5. Parse and validate
        try:
            return self._parse_response(response_text)
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"  [{self.role}] Response parse failed: {e}")
            print(f"  [{self.role}] Raw response was: {response_text[:300]}")
            state.had_agent_failure = True
            return self._fallback_report(f"Response parse failed: {e}")

    # --- helpers shared by all subclasses ---

    def _load_template(self) -> str:
        path = _PROMPTS_DIR / self.prompt_file
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {path}. "
                f"Did you create agents/prompts/{self.prompt_file}?"
            )
        return path.read_text(encoding="utf-8")

    def _parse_response(self, response: str) -> AnalystReport:
        """Extract JSON from response text and validate against schema."""
        # LLMs sometimes wrap JSON in markdown fences despite instructions.
        # Strip ```json ... ``` or ``` ... ``` if present.
        cleaned = response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # Find the JSON object — first { to matching last }
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first == -1 or last == -1 or last <= first:
            raise json.JSONDecodeError("No JSON object found", cleaned, 0)

        json_str = cleaned[first:last + 1]
        data = json.loads(json_str)
        return AnalystReport(**data)

    def _fallback_report(self, error_message: str) -> AnalystReport:
        """A safe report to return when something goes wrong upstream."""
        return AnalystReport(
            signal="neutral",
            confidence=0.0,
            summary=f"[{self.role} produced no analysis: {error_message[:100]}]",
            key_points=[],
            risks=["Analyst output unavailable; downstream agents should "
                   "discount this report entirely."],
        )

    # --- subclasses must implement this ---

    @abstractmethod
    def _build_data_section(self, state: AgentState) -> dict:
        """Return a dict of variables to substitute into the prompt template."""
        ...


# ============================================================================
# CONCRETE ANALYSTS
# ============================================================================

class FundamentalsAnalyst(AnalystBase):
    role = "fundamentals_analyst"
    prompt_file = "fundamentals_analyst.txt"

    def _build_data_section(self, state: AgentState) -> dict:
        # Pull 90 days of prices
        prices = get_prices(state.ticker, state.as_of_date, lookback_days=90)

        # Summarize: show first/last 5 rows and key stats
        first_5 = prices.head(5)
        last_5 = prices.tail(5)
        price_summary = (
            f"First 5 days in window ({first_5.index[0].date()} to {first_5.index[-1].date()}):\n"
            f"{first_5.to_string()}\n\n"
            f"Last 5 days ({last_5.index[0].date()} to {last_5.index[-1].date()}):\n"
            f"{last_5.to_string()}"
        )

        # Derived metrics
        period_return = (prices["Close"].iloc[-1] / prices["Close"].iloc[0] - 1) * 100
        daily_returns = prices["Close"].pct_change().dropna()
        annual_vol = daily_returns.std() * (252 ** 0.5) * 100
        avg_volume = prices["Volume"].mean()
        recent_volume = prices["Volume"].tail(5).mean()
        volume_ratio = recent_volume / avg_volume

        derived_metrics = (
            f"- Period return: {period_return:+.2f}% "
            f"(over {len(prices)} trading days)\n"
            f"- Annualized volatility: {annual_vol:.1f}%\n"
            f"- Average daily volume: {avg_volume:,.0f} shares\n"
            f"- Recent 5-day avg volume: {recent_volume:,.0f} "
            f"({volume_ratio:.2f}x the period average)"
        )

        return {
            "ticker": state.ticker,
            "as_of_date": state.as_of_date,
            "n_days": len(prices),
            "price_summary": price_summary,
            "derived_metrics": derived_metrics,
        }


class TechnicalAnalyst(AnalystBase):
    role = "technical_analyst"
    prompt_file = "technical_analyst.txt"

    def _build_data_section(self, state: AgentState) -> dict:
        result = get_indicators(state.ticker, state.as_of_date, lookback_days=250)

        # Format the "latest" dict as readable lines
        latest = result["latest"]
        indicator_lines = []
        for name, value in latest.items():
            if value is None:
                indicator_lines.append(f"  {name}: <not available>")
            else:
                indicator_lines.append(f"  {name}: {value:.4f}")
        indicators_str = "\n".join(indicator_lines)

        # Show the last 5 days of the time series
        series_tail = result["series"].tail(5)
        trend_str = series_tail.to_string()

        return {
            "ticker": state.ticker,
            "as_of_date": state.as_of_date,
            "indicators": indicators_str,
            "indicator_trend": trend_str,
        }


class NewsAnalyst(AnalystBase):
    role = "news_analyst"
    prompt_file = "news_analyst.txt"

    def _build_data_section(self, state: AgentState) -> dict:
        news = get_news(state.ticker, state.as_of_date, lookback_days=14)

        if not news:
            news_list = "  (No news articles found in the lookback window.)"
        else:
            lines = []
            for i, item in enumerate(news[:15], start=1):  # cap at 15 to keep prompts short
                lines.append(
                    f"  {i}. [{item['published_at'].strftime('%Y-%m-%d')}] "
                    f"({item['publisher']}) {item['title']}"
                )
                if item.get("summary"):
                    summary = item["summary"][:200]
                    lines.append(f"     Summary: {summary}")
            news_list = "\n".join(lines)

        return {
            "ticker": state.ticker,
            "as_of_date": state.as_of_date,
            "n_articles": len(news),
            "news_list": news_list,
        }


class SentimentAnalyst(AnalystBase):
    role = "sentiment_analyst"
    prompt_file = "sentiment_analyst.txt"

    def _build_data_section(self, state: AgentState) -> dict:
        news = get_news(state.ticker, state.as_of_date, lookback_days=14)
        prices = get_prices(state.ticker, state.as_of_date, lookback_days=30)

        # News for tone reading
        if not news:
            news_list = "  (No news articles found.)"
        else:
            lines = []
            for i, item in enumerate(news[:10], start=1):
                lines.append(
                    f"  {i}. [{item['published_at'].strftime('%Y-%m-%d')}] "
                    f"{item['title']}"
                )
            news_list = "\n".join(lines)

        # Price behavior summary for context
        last_5_return = (prices["Close"].iloc[-1] / prices["Close"].iloc[-6] - 1) * 100 \
            if len(prices) >= 6 else 0.0
        period_high = prices["High"].max()
        period_low = prices["Low"].min()
        latest_close = prices["Close"].iloc[-1]
        distance_from_high = (latest_close / period_high - 1) * 100
        distance_from_low = (latest_close / period_low - 1) * 100

        price_behavior = (
            f"- Last 5-day return: {last_5_return:+.2f}%\n"
            f"- 30-day high: ${period_high:.2f}\n"
            f"- 30-day low: ${period_low:.2f}\n"
            f"- Current close: ${latest_close:.2f} "
            f"({distance_from_high:+.1f}% from high, {distance_from_low:+.1f}% from low)"
        )

        return {
            "ticker": state.ticker,
            "as_of_date": state.as_of_date,
            "n_articles": len(news),
            "news_list": news_list,
            "price_behavior": price_behavior,
        }