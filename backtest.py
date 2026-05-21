"""
Backtest runner.

For a given ticker and date range, this:
  1. Builds a list of weekly decision dates.
  2. At each date, runs the full 11-agent pipeline to get a buy/sell/hold.
  3. Feeds each decision into a Portfolio simulator at that day's close price.
  4. Computes performance metrics for the agent strategy AND a buy-and-hold
     baseline over the same dates.
  5. Checkpoints progress to disk after every decision, so a crashed run can
     resume instead of restarting (and not re-burn API quota).

Public entry point: run_backtest(...).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from config import RESULTS_DIR
from data import get_prices
from pipeline import run_pipeline
from portfolio import Portfolio, compute_metrics, buy_and_hold_value_series


# ============================================================================
# DECISION DATE GENERATION
# ============================================================================

def weekly_decision_dates(ticker: str, start: str, end: str) -> list[str]:
    """
    Return a list of weekly decision dates (every 5th trading day) between
    `start` and `end`, using the ticker's actual trading calendar.

    We use real trading days (from price data) so we never try to make a
    decision on a weekend or market holiday.
    """
    # Pull prices across the whole range. as_of = end gives us everything.
    prices = get_prices(ticker, as_of_date=end, lookback_days=3650)
    trading_days = prices.loc[start:end].index

    if len(trading_days) == 0:
        raise ValueError(
            f"No trading days for {ticker} between {start} and {end}."
        )

    # Take every 5th trading day (~weekly).
    weekly = trading_days[::5]
    return [d.strftime("%Y-%m-%d") for d in weekly]


# ============================================================================
# RESULT TYPES
# ============================================================================

@dataclass
class BacktestResult:
    ticker: str
    regime_name: str
    start: str
    end: str
    decision_dates: list
    decisions: list          # list of {date, decision, price}
    # agent strategy metrics
    agent_cumulative_return: float
    agent_annualized_return: float
    agent_sharpe: float
    agent_max_drawdown: float
    # buy-and-hold baseline metrics
    bh_cumulative_return: float
    bh_annualized_return: float
    bh_sharpe: float
    bh_max_drawdown: float

    def pretty(self) -> str:
        return (
            f"=== {self.ticker} | {self.regime_name} "
            f"({self.start} to {self.end}) ===\n"
            f"  Decisions made: {len(self.decisions)}\n"
            f"  {'Metric':<22}{'Agents':>12}{'Buy&Hold':>12}\n"
            f"  {'-'*46}\n"
            f"  {'Cumulative return':<22}"
            f"{self.agent_cumulative_return*100:>11.2f}%"
            f"{self.bh_cumulative_return*100:>11.2f}%\n"
            f"  {'Annualized return':<22}"
            f"{self.agent_annualized_return*100:>11.2f}%"
            f"{self.bh_annualized_return*100:>11.2f}%\n"
            f"  {'Sharpe ratio':<22}"
            f"{self.agent_sharpe:>12.2f}{self.bh_sharpe:>12.2f}\n"
            f"  {'Max drawdown':<22}"
            f"{self.agent_max_drawdown*100:>11.2f}%"
            f"{self.bh_max_drawdown*100:>11.2f}%\n"
        )


# ============================================================================
# CHECKPOINTING
# ============================================================================

def _checkpoint_path(ticker: str, regime_name: str) -> Path:
    safe = f"{ticker}_{regime_name}".replace(" ", "_").replace("/", "_")
    return RESULTS_DIR / f"checkpoint_{safe}.json"


def _load_checkpoint(ticker: str, regime_name: str) -> dict:
    """Load saved decisions for this run, or an empty dict if none."""
    path = _checkpoint_path(ticker, regime_name)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_checkpoint(ticker: str, regime_name: str, decisions: dict) -> None:
    """Persist decisions-so-far to disk."""
    path = _checkpoint_path(ticker, regime_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2)


# ============================================================================
# THE BACKTEST
# ============================================================================

def run_backtest(
    ticker: str,
    regime_name: str,
    start: str,
    end: str,
    starting_cash: float = 10_000.0,
    verbose: bool = True,
) -> BacktestResult:
    """
    Run a full backtest for one ticker over one regime.

    Resumable: if a checkpoint exists, decisions already made are loaded from
    disk and not recomputed (saving API quota).

    Args:
        ticker: stock symbol.
        regime_name: human label, e.g. "2024_bull" — used for checkpoint files.
        start, end: date range "YYYY-MM-DD".
        starting_cash: starting capital.
        verbose: print progress.

    Returns:
        BacktestResult with agent and buy-and-hold metrics.
    """
    dates = weekly_decision_dates(ticker, start, end)
    if verbose:
        print(f"\n{'='*60}")
        print(f"BACKTEST: {ticker} | {regime_name}")
        print(f"  {start} to {end} | {len(dates)} weekly decisions")
        print(f"{'='*60}")

    # Load any decisions already made in a previous (crashed) run.
    saved = _load_checkpoint(ticker, regime_name)
    decisions: dict = dict(saved)  # date -> {decision, price}
    if saved and verbose:
        print(f"  Resuming: {len(saved)} decisions loaded from checkpoint.")

    # --- Make a decision at each date (skip ones we already have) ---
    for i, date in enumerate(dates, start=1):
        if date in decisions:
            if verbose:
                print(f"  [{i}/{len(dates)}] {date}: "
                      f"{decisions[date]['decision']} (cached)")
            continue

        if verbose:
            print(f"  [{i}/{len(dates)}] {date}: running pipeline...", end=" ")

        result = run_pipeline(ticker, date, verbose=False)

        # Price on the decision date = that day's close.
        day_prices = get_prices(ticker, as_of_date=date, lookback_days=10)
        close_price = float(day_prices["Close"].iloc[-1])

        # If any agent failed (e.g. API error), the decision is not trustworthy.
        if result.state.had_agent_failure:
            recorded_decision = "INVALID"
            print(f"  [{i}/{len(dates)}] {date}: INVALID (agent failure)")
        else:
            recorded_decision = result.decision
        decisions[date] = {
            "decision": recorded_decision,
            "price": close_price,
        }
        _save_checkpoint(ticker, regime_name, decisions)  # checkpoint NOW

        if verbose:
            print(f"{result.decision.upper()} @ ${close_price:.2f}")

    # --- Simulate the portfolio over the decisions ---
    portfolio = Portfolio(starting_cash=starting_cash)
    for date in dates:
        d = decisions[date]
        action = d["decision"]
        sim_action = "hold" if action == "INVALID" else action
        portfolio.step(date, sim_action, d["price"])

    agent_values = portfolio.value_series()
    agent_metrics = compute_metrics(agent_values, periods_per_year=52.0)

    # --- Buy-and-hold baseline over the same dates ---
    bh_prices = pd.Series(
        {pd.to_datetime(date): decisions[date]["price"] for date in dates}
    ).sort_index()
    bh_values = buy_and_hold_value_series(bh_prices, starting_cash=starting_cash)
    bh_metrics = compute_metrics(bh_values, periods_per_year=52.0)

    result = BacktestResult(
        ticker=ticker,
        regime_name=regime_name,
        start=start,
        end=end,
        decision_dates=dates,
        decisions=[
            {"date": d, "decision": decisions[d]["decision"],
             "price": decisions[d]["price"]}
            for d in dates
        ],
        agent_cumulative_return=agent_metrics.cumulative_return,
        agent_annualized_return=agent_metrics.annualized_return,
        agent_sharpe=agent_metrics.sharpe_ratio,
        agent_max_drawdown=agent_metrics.max_drawdown,
        bh_cumulative_return=bh_metrics.cumulative_return,
        bh_annualized_return=bh_metrics.annualized_return,
        bh_sharpe=bh_metrics.sharpe_ratio,
        bh_max_drawdown=bh_metrics.max_drawdown,
    )

    # Save the full result too.
    safe = f"{ticker}_{regime_name}".replace(" ", "_")
    with open(RESULTS_DIR / f"result_{safe}.json", "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)

    if verbose:
        print()
        print(result.pretty())

    return result