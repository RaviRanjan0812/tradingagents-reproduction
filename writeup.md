# Reproducing TradingAgents Across Market Regimes

*A robustness study of a multi-agent LLM trading framework*

---

## TL;DR

I reimplemented the TradingAgents multi-agent LLM framework from scratch and tested it across a 2024 Q1 bull market and a 2022 Q1 bear market, on AAPL and JPM. The framework's behavior turned out to be **asymmetric across regimes**: in bull markets it tracked buy-and-hold (no alpha), but in bear markets it reduced both maximum drawdown and total loss. On JPM 2022  specifically, the agents traded actively (9 directional trades out of 13) and outperformed buy-and-hold by 5 percentage points while cutting drawdown roughly in half. The system looks less like an alpha generator and more like a regime-aware risk overlay — though n=4 regime-runs is too small to claim this is established rather than suggestive.

---

## 1. Background

The TradingAgents paper (Xiao et al., 2024) proposes a multi-agent LLM framework modeled on a real trading firm: specialist analyst agents, a bull-vs-bear research debate, a trader, a risk committee, and a fund manager. The paper reports strong returns and Sharpe ratios well above rule-based baselines on AAPL, GOOGL, and AMZN over roughly January–March 2024.

The paper's evaluation has one notable limitation: it covers a single, relatively favorable 3-month window. That raises the question this project addresses — does the framework's reported edge survive a market regime it wasn't tested in?

---

## 2. What I built

The implementation is roughly 1,500 lines of Python. The agent layer mirrors the paper: four specialist analysts (fundamentals, sentiment, news, technical) producing structured reports, a bull-vs-bear research debate judged by a neutral facilitator, a trader proposing buy/sell/hold, a three-member risk committee (aggressive/neutral/conservative), and a fund manager making the final call. State flows between agents through a single Pydantic-validated `AgentState` object — every field optional initially, populated as the pipeline advances.

Three engineering choices worth calling out:

**Point-in-time data layer.** The single most important line of code in the project is `pit_filter = full_history.index <= as_of`. Every data fetch — prices, news, technical indicators — filters strictly to dates on or before the simulation date. This is unit-tested explicitly: the test verifies that requesting AAPL "as of 2022-06-01" returns no rows from 2022-06-02 onwards. Without this, any result would be meaningless because the agents would have seen the future.

**Tiered model routing.** API cost is the dominant operational constraint when each pipeline run makes ~11 LLM calls. The four analyst roles run on Gemini Flash-Lite (summarization-heavy work); the seven reasoning-critical roles (researchers, trader, risk team, fund manager) run on the stronger Gemini Flash. One central `MODEL_ROUTING` dict in `config.py` makes this a one-line swap. Total experiment cost: approximately $2 of Gemini credit.

**Failure propagation through to the measurement layer.** This one is worth telling as a story. Each agent catches its own exceptions and returns a neutral fallback report instead of crashing — necessary for partial robustness in a long pipeline. But an early version of the backtester *recorded those fallbacks as real "hold" decisions in the result file*. I discovered this when an initial run of the experiment produced a JPM 2022 result of all-HOLD, 0.00% return — a result I was briefly tempted to interpret as "the agents correctly stand aside in a bear market." On closer inspection the all-HOLD pattern coincided with a session in which API quota had partially exhausted: the HOLDs were the error handler's fallbacks, not real agent decisions. The fix was two parts: every agent fallback now sets a `had_agent_failure` flag on the shared state, and the backtester records contaminated decisions as `INVALID` rather than silent holds. After re-running JPM 2022 under this instrumentation, the clean numbers are the ones reported in Section 4: 9 active trades and a -7.46% return that beats buy-and-hold by 5 percentage points. The general lesson — **graceful degradation in agents can silently corrupt backtest results if failure signals don't propagate to the measurement layer** — felt worth recording.

### Honest limitations of the reimplementation

- **Small sample.** Two tickers, two regimes, weekly decisions. n is too small to claim anything universal. The findings here are suggestive, not established.
- **LLM training-data contamination.** The Gemini models I used have read news from 2022 and 2024, and to some degree "know" what happened next. This cannot be fully eliminated for any LLM backtest using a model trained after the test dates. It is a real confound, and the appropriate response is to acknowledge it rather than pretend it isn't there.
- **News data sparsity for older dates.** yfinance's news endpoint returns mostly recent articles. On 2022 dates, the news analyst frequently received zero articles. Fixing this requires a paid news archive (Refinitiv, Bloomberg) which would invalidate the "fully reproducible on free data" property.
- **Simplified portfolio model.** No transaction costs, no shorting, single-position sizing (fully long or fully flat). Realistic for a comparison across regimes, unrealistic as an actual trading system.
- **Model substitution.** I used Gemini 2.5 Flash and Flash-Lite via the paid Gemini API after free-tier quota walls made the experiment otherwise infeasible. Whether a stronger reasoning model (Claude Opus, GPT-5) changes the regime-sensitivity pattern is the natural next experiment.

---

## 3. Experimental design

- **Tickers:** AAPL (large-cap tech), JPM (large-cap financials).
- **Regimes:** 2024 bull market (Jan 2 – Mar 29, 2024) and 2022 bear market (Jan 3 – Mar 31, 2022, the rate-hike tech selloff).
- **Sampling:** one decision per week (13 per regime).
- **Baseline:** buy-and-hold over the same dates.
- **Metrics:** cumulative return, annualized return, Sharpe ratio, maximum drawdown.

The comparison is deliberately controlled: identical agent configuration, identical models, identical prompts across all four runs. Only the ticker and regime change.

---

## 4. Results

### Headline numbers

| Ticker | Regime | Agent Return | B&H Return | Agent Sharpe | B&H Sharpe | Agent MaxDD | B&H MaxDD |
|---|---|---:|---:|---:|---:|---:|---:|
| AAPL | 2024 bull | -6.45% | -7.51% | -2.71 | -1.28 | -6.45% | -13.00% |
| AAPL | 2022 bear | -4.13% | -2.21% | -0.93 | -0.09 | -8.44% | -12.21% |
| JPM | 2024 bull | +17.12% | +17.12% | 4.88 | 4.88 | -2.30% | -2.30% |
| JPM | 2022 bear | -7.46% | -12.56% | -1.51 | -1.74 | -9.59% | -20.25% |

### Decision breakdown

| Ticker | Regime | Buy | Sell | Hold |
|---|---|---:|---:|---:|
| AAPL | 2024 bull | 2 | 8 | 3 |
| AAPL | 2022 bear | 3 | 3 | 7 |
| JPM | 2024 bull | 8 | 0 | 5 |
| JPM | 2022 bear | 3 | 6 | 4 |

Every decision is a real agent decision (zero INVALID flags across 52 pipeline runs).

### Interpretation

The headline finding is not what I expected going in.

On raw returns, the agent system did not beat buy-and-hold in three of the four runs. In JPM 2024 it stayed long the entire rally and matched the baseline exactly (+17.12% to two decimal places — they essentially replicated buy-and-hold by being correct that the rally was real). In AAPL 2024 they did slightly better than the baseline (-6.45% vs -7.51%, both negative because AAPL underperformed the broader market in Q1). In AAPL 2022 they did slightly worse (-4.13% vs -2.21%). None of these gaps is large enough to claim alpha.

The interesting result is **JPM 2022**, the deepest bear regime in the sample. There the agents beat buy-and-hold by 5 percentage points on return (-7.46% vs -12.56%) AND cut maximum drawdown roughly in half (-9.59% vs -20.25%), while making 9 directional trades. This is not a "they sat in cash and got lucky" result. They traded actively and came out ahead.

Across all four runs, the more consistent pattern is **drawdown reduction**:

- AAPL 2024: -6.45% vs -13.00% (gap of 6.5 pp)
- AAPL 2022: -8.44% vs -12.21% (gap of 3.8 pp)
- JPM 2024: -2.30% vs -2.30% (no gap — the rally never tested it)
- JPM 2022: -9.59% vs -20.25% (gap of 10.7 pp)

The agents trade something specific: *upside capture* for *downside protection*. They behave less like an alpha engine and more like a risk overlay on a long-only strategy. This refines rather than confirms the original paper's framing of the framework as a return-generating system.

### Is this skill or structural passivity?

The skeptical question is whether the drawdown reduction is real risk-reading or just an artifact of the agents holding cash a lot. The decision breakdown matters here.

In JPM 2022, the most directly informative case, the agents made 9 directional trades (3 buy, 6 sell) and only 4 holds. A purely passive system would not have produced 9 trades. And in JPM 2024 — a strongly rising market — the agents stayed long (8 buys, 0 sells, 5 holds) and matched the rally. They didn't disengage when participating was correct. In AAPL 2022, the agents traded actively (3 buys, 3 sells, 7 holds) rather than freezing.

So the pattern is not "the agents always hold cash and sometimes that helps." The most honest reading is that the agents lean active when they see a directional case and lean cautious in ambiguous conditions — and in three of the four runs, this calibration happened to add value (on drawdown if not return). Whether the conditions the agents identify as ambiguous *genuinely* have worse risk/return than the ones they engage with is a hypothesis my data is consistent with, not one it establishes. n=4 regime-runs is small.

---

## 5. A specific behavioral observation

During pipeline development, on the AAPL 2024-03-01 scenario where every analyst signal was bearish, the bull researcher's argument included the line: *"Apple's strong brand, diverse product portfolio, and history of innovation position the company for long-term success."*

None of that content was in any analyst report. The bull researcher reached into the LLM's background knowledge of Apple to manufacture a bull case when the actual evidence didn't support one. The facilitator partly caught this — it judged "bear" and noted the bull "relied on speculative 'potential' reversals" — but the underlying behavior is a real failure mode.

**This suggests the framework's behavior is partly a function of ticker fame, not just data.** On a famous mega-cap, the LLM has strong priors that can substitute for evidence. On an obscure ticker, this substitution would be impossible and the bull case would look visibly weaker. Any honest backtest of LLM-based agents needs to consider this; the original paper, which evaluated only on famous mega-caps (AAPL, GOOGL, AMZN), does not.

---

## 6. What I'd do next

**More regimes.** Adding 2008 GFC, March 2020 COVID crash, and 2015 sideways markets would test the "drawdown reduction across regimes" pattern with more statistical power.

**Famous vs. obscure tickers.** Pair each mega-cap with a small-cap in the same sector. If the bull-researcher-uses-priors effect is real, the framework's behavior should differ more for obscure tickers than the LLM has strong priors about.

**Ablation: does the debate help?** Run the framework with and without the bull/bear debate, identical otherwise. If the debate doesn't measurably change outcomes, then the most architecturally interesting piece of the paper isn't doing the work it's claimed to do.

**Stronger reasoning model.** Repeat the experiment with a frontier model for the trader and fund manager. Does regime sensitivity disappear with smarter reasoning, or persist? This separates capability from architecture.

**Contamination probe.** Compare the agents' decisions against a counterfactual where they only see metadata about the analysis date (no actual data). If the decisions correlate strongly with what actually happened next even in the no-data condition, training-data contamination is the dominant signal — and any LLM backtest result is suspect.

---

## 7. Reproducing this

Code: https://github.com/RaviRanjan0812/tradingagents-reproduction.

The repo includes `requirements.txt` (pinned versions), `.env.example` (lists the one API key needed: a Gemini API key from Google AI Studio), and a single command `python run_experiment.py` that reproduces every table in this writeup. Total runtime ~2 hours; total API cost approximately $2 of Gemini credit. Every decision is checkpointed; runs are fully resumable.

A clean run produces zero `INVALID` decisions. If any appear, the failure-propagation system has flagged them and you can either re-run those specific decisions or drop them from the analysis with the knowledge they were never real agent decisions.

---

*Code: https://github.com/RaviRanjan0812/tradingagents-reproduction*