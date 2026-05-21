# TradingAgents Reproduction & Multi-Regime Study

A from-scratch reimplementation of the TradingAgents multi-agent LLM trading framework (Xiao et al., 2024), with a robustness study across 2024 bull and 2022 bear market regimes on AAPL and JPM.

**Full writeup:** see [writeup.md](./writeup.md).

## Quick start

```
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python run_experiment.py
```

Then fill in your `GEMINI_API_KEY` in `.env` (get one at https://aistudio.google.com/api-keys).

Total runtime ~2 hours. Total API cost ~$2 of Gemini credit. Every decision is checkpointed and resumable.

## What's inside

- `agents/` — 11 specialist agents across 5 stages (analysts, researchers, trader, risk team, fund manager)
- `data/` — point-in-time correct data layer (prices, news, technical indicators)
- `pipeline.py` — orchestrator that runs the full 11-agent decision pipeline
- `backtest.py` — resumable backtester with failure-propagation instrumentation
- `portfolio.py` — portfolio simulator and performance metrics (unit-tested)
- `run_experiment.py` — runs the full multi-regime experiment
- `writeup.md` — the findings

## Headline result

The framework's behavior is asymmetric across regimes:

| Ticker | Regime | Agent Return | B&H Return | Agent MaxDD | B&H MaxDD |
|---|---|---:|---:|---:|---:|
| AAPL | 2024 bull | -6.45% | -7.51% | -6.45% | -13.00% |
| AAPL | 2022 bear | -4.13% | -2.21% | -8.44% | -12.21% |
| JPM | 2024 bull | +17.12% | +17.12% | -2.30% | -2.30% |
| JPM | 2022 bear | -7.46% | -12.56% | -9.59% | -20.25% |

In bull markets the agents tracked buy-and-hold (no alpha). In bear markets they reduced both drawdown and total loss — on JPM 2022, beating buy-and-hold by 5 percentage points while making 9 active trades. See [writeup.md](./writeup.md) for the full analysis.

## Reference

Xiao, Sun, Luo & Wang (2024), "TradingAgents: Multi-Agents LLM Financial Trading Framework."