"""
Portfolio simulator and performance metrics.

Models a simple single-position portfolio:
  - Start with a cash balance (default $10,000).
  - "buy"  -> if currently flat, invest ALL cash into the stock at that day's price.
  - "sell" -> if currently long, sell the ENTIRE position back to cash.
  - "hold" -> do nothing; keep whatever position we have.

This is deliberately simple (no shorting, no partial sizing, no transaction
costs by default). Simplicity makes the backtest easy to verify by hand,
which matters more than realism for a research comparison. Realism extensions
(shorting, sizing, costs) are noted as future work in the writeup.

The four metrics match the TradingAgents paper's Appendix definitions:
  - Cumulative Return
  - Annualized Return
  - Sharpe Ratio
  - Maximum Drawdown
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ============================================================================
# PORTFOLIO SIMULATOR
# ============================================================================

@dataclass
class Portfolio:
    """A single-position long/flat portfolio."""

    starting_cash: float = 10_000.0
    transaction_cost_pct: float = 0.0  # e.g. 0.001 = 10 bps per trade; default off

    # --- internal state ---
    cash: float = field(init=False)
    shares: float = field(init=False, default=0.0)
    # history: list of (date, action, price, portfolio_value)
    history: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self.cash = self.starting_cash

    def is_long(self) -> bool:
        return self.shares > 0

    def value(self, current_price: float) -> float:
        """Total portfolio value = cash + (shares * current price)."""
        return self.cash + self.shares * current_price

    def step(self, date: str, action: str, price: float) -> None:
        """
        Apply one day's decision at the given price.

        action: "buy" | "sell" | "hold"
        price:  the stock's price on `date` (we use the close).
        """
        if price <= 0:
            raise ValueError(f"Invalid price {price} on {date}")

        if action == "buy" and not self.is_long():
            # Invest all cash. Account for transaction cost.
            effective_cash = self.cash * (1 - self.transaction_cost_pct)
            self.shares = effective_cash / price
            self.cash = 0.0

        elif action == "sell" and self.is_long():
            # Liquidate entire position.
            proceeds = self.shares * price * (1 - self.transaction_cost_pct)
            self.cash = proceeds
            self.shares = 0.0

        # "hold", or "buy" while already long, or "sell" while already flat:
        # no change to positions.

        self.history.append({
            "date": date,
            "action": action,
            "price": price,
            "value": self.value(price),
            "position": "long" if self.is_long() else "flat",
        })

    def value_series(self) -> pd.Series:
        """Portfolio value over time, indexed by date."""
        if not self.history:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self.history)
        return pd.Series(
            df["value"].values,
            index=pd.to_datetime(df["date"]),
            name="portfolio_value",
        )


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

@dataclass
class PerformanceMetrics:
    cumulative_return: float      # e.g. 0.12 = +12%
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float           # e.g. -0.08 = -8% worst peak-to-trough
    n_periods: int

    def pretty(self) -> str:
        return (
            f"Cumulative return:  {self.cumulative_return * 100:+.2f}%\n"
            f"Annualized return:  {self.annualized_return * 100:+.2f}%\n"
            f"Sharpe ratio:       {self.sharpe_ratio:.2f}\n"
            f"Max drawdown:       {self.max_drawdown * 100:.2f}%\n"
            f"Periods:            {self.n_periods}"
        )


def compute_metrics(
    value_series: pd.Series,
    periods_per_year: float = 52.0,   # weekly sampling -> 52 periods/year
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """
    Compute the four standard performance metrics from a portfolio value series.

    Args:
        value_series: portfolio value over time (from Portfolio.value_series()).
        periods_per_year: 52 for weekly sampling, 252 for daily.
        risk_free_rate: annual risk-free rate. Default 0 keeps the Sharpe
                        calculation simple and conservative.

    Returns:
        PerformanceMetrics.
    """
    if len(value_series) < 2:
        # Not enough data to compute returns.
        return PerformanceMetrics(0.0, 0.0, 0.0, 0.0, len(value_series))

    values = value_series.values.astype(float)

    # --- Cumulative return: end vs start ---
    cumulative_return = values[-1] / values[0] - 1.0

    # --- Period returns (percent change between consecutive points) ---
    period_returns = np.diff(values) / values[:-1]

    # --- Annualized return ---
    # Geometric: (1 + total)^(periods_per_year / n_periods) - 1
    n_periods = len(period_returns)
    if n_periods > 0 and (1.0 + cumulative_return) > 0:
        annualized_return = (1.0 + cumulative_return) ** (periods_per_year / n_periods) - 1.0
    else:
        annualized_return = 0.0

    # --- Sharpe ratio ---
    # (mean excess return / std of returns), annualized.
    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = period_returns - rf_per_period
    std = excess_returns.std()
    if std > 1e-9:
        sharpe = (excess_returns.mean() / std) * np.sqrt(periods_per_year)
    else:
        # Zero volatility (e.g. portfolio stayed in cash the whole time).
        sharpe = 0.0

    # --- Maximum drawdown ---
    # Largest peak-to-trough decline in portfolio value.
    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max
    max_drawdown = drawdowns.min()  # most negative value

    return PerformanceMetrics(
        cumulative_return=float(cumulative_return),
        annualized_return=float(annualized_return),
        sharpe_ratio=float(sharpe),
        max_drawdown=float(max_drawdown),
        n_periods=int(n_periods),
    )


# ============================================================================
# BUY-AND-HOLD BASELINE
# ============================================================================

def buy_and_hold_value_series(
    prices: pd.Series,
    starting_cash: float = 10_000.0,
) -> pd.Series:
    """
    Compute the portfolio value series for a buy-and-hold strategy:
    invest everything at the first price, never trade again.

    Args:
        prices: stock close prices indexed by date (the decision dates).
        starting_cash: starting capital.

    Returns:
        Portfolio value over time.
    """
    if len(prices) == 0:
        return pd.Series(dtype=float)
    shares = starting_cash / float(prices.iloc[0])
    return prices.astype(float) * shares