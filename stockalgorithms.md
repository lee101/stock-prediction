# Stock Trading Algorithms - Honest Assessment (March 2026)

This document reviews every major trading approach in this repo, focusing on **simulation realism** and **what we actually trust**.

---

## TL;DR Ranking

| Approach | Sim Realism | Best Claimed Result | Live Result | Trust Level |
|----------|-------------|---------------------|-------------|-------------|
| **Unified Hourly Meta-Selector** | 8/10 | +4-7% ann., Sortino 1.7+ | ~$46k (from $56k) | Medium - live is losing |
| **PufferLib PPO (crypto)** | 5/10 | 2,659x in 30 days | Never deployed | Low - sim too simple |
| **Binance Neural Policy** | 7/10 | Modest per-symbol profits | Not deployed | Medium - honest sim |
| **FastForecaster2** | 6/10 | +0.94% total | Not deployed | Low - sparse signals |
| **LLM Agents (DeepSeek/Opus)** | 6/10 | +$6.65 (1 day) | -$8,662 (7 days) | Very Low - lost money live |
| **Per-Stock Neural** | 7/10 | +2.6% over 150d, 3.4% DD | Part of meta-selector | Medium |
| **ETH Risk PPO** | 7/10 | Under iteration | Not deployed | TBD |

---

## 1. PufferLib PPO (Reinforcement Learning)

**Location:** `pufferlib_market/`

**What it does:** PPO agent trained in a pure-C environment on hourly crypto bars (12 symbols). Observes Chronos2 forecasts + technical features, outputs discrete long/short/flat actions.

**Claimed results:** 2,659x return in 30 days ($10k -> $26.6M) on crypto12 at 300M training steps.

**Simulation concerns:**
- **Instant fills** - no slippage, no bid-ask spread, no partial fills
- **Limit orders fill at exact price** if bar high/low touches it - unrealistic
- **No market impact** - a $26M position in ALGOUSD would move the market
- **No queue priority** - assumes you're first in line at every price level
- **No short borrow costs** or margin mechanics
- **Chronos forecast leakage not audited** - unclear if forecasts use future data in their computation

**Why the 2,659x number is not credible:**
- That's turning $10k into $26.6M in one month trading crypto
- At those position sizes, you'd be a significant fraction of hourly volume on small-cap coins
- The sim assumes zero market impact at any size
- With realistic 5-10bp slippage per trade at those volumes, returns would collapse
- The sim fills limit orders at exact touch price - real fills are worse

**What's good about it:**
- Very fast C environment enables massive-scale training
- Reproducible (within 0.1% across seeds)
- Clean architecture, proper episode termination
- Anneal-LR finding is a genuine RL insight

**Verdict:** The RL framework is solid engineering, but the simulator is too simple for the results to transfer to live trading. Would need slippage modeling, volume-aware position sizing, and market impact before trusting any results.

---

## 2. Unified Hourly Meta-Selector (Stock Trading)

**Location:** `unified_hourly_experiment/`

**What it does:** Runs multiple neural trading policies in parallel, selects the best-performing one per symbol on a trailing window (7-16 days). Trades hourly on Alpaca with limit orders.

**Current live config:**
- 7 strategy checkpoints, `metric=p10`, `selection_mode=sticky`
- `lookback_days=16`, `switch_margin=0.005`, `sit_out_threshold=-0.001`
- Market-order entry, 2x leverage, 5 max positions
- Stocks: NVDA, PLTR, GOOG (long) + DBX, TRIP, MTCH (short)

**Simulation realism (best in repo):**
- Decision lag (1 bar) matching live loop timing
- Limit order fills require bar penetration
- Per-symbol fees, EOD close enforcement
- Cancel/replace delay simulating broker latency
- Reserved cash for pending orders
- Market hours enforcement
- Multi-scenario validation (multiple bar_margin + TTL combos)

**Simulation gaps:**
- No queue priority modeling
- Static slippage (bar_margin 5-13bp), not stochastic
- No volume/liquidity constraints
- Predicted prices used for edge scoring (mild lookahead concern)

**Backtested results (latest March 8 sweep):**
- Best activity-filtered: +4.3% to +6.6% annualized, Sortino 1.7-2.6
- Max drawdown ~0.34%
- But: limit-entry version goes **negative** (-0.56% to -0.77%)
- Only market-order entry is profitable

**Live results:**
- Production account: **$46,460** (started ~$55,935) - **down ~17%**
- This is the most important data point in the whole repo

**Verdict:** Most realistic simulator, most rigorous validation process. But the gap between backtest (+4-7% ann.) and live (-17%) is a red flag. The market-order vs limit-entry sensitivity suggests the edge is thin and execution-dependent. Still the most promising approach due to honest validation methodology.

---

## 3. Binance Neural Policy (Crypto)

**Location:** `binanceneural/`

**What it does:** Transformer-based policy trained end-to-end with differentiable market simulation. Outputs buy/sell prices and intensities for crypto pairs. Uses Chronos2 forecasts as input features.

**Simulation realism:**
- Penetration-based fills (scales by how far bar trades through limit)
- Touch-fill fraction for partial execution
- Temperature-based stochastic fills during training
- Decision lag, maker fees, margin interest
- Max volume fraction caps

**Results:**
- Individual BTC/ETH/SOL models trained (Feb 2026 checkpoints)
- Multi-symbol selector currently **negative** (-6.5% to -8.5% across scenarios)
- Not deployed live

**Concerns:**
- Chronos forecast quality may have degraded
- Selector edge thresholds too tight for crypto volatility
- Single-position mode limits diversification

**Verdict:** Honest simulation with penetration fills is more realistic than most alternatives. But currently not profitable. The differentiable sim approach is architecturally sound - worth continued iteration.

---

## 4. Per-Stock Neural Policies (Alpaca)

**Location:** `fastalgorithms/per_stock/`, feeds into `unified_hourly_experiment/`

**What it does:** Individual transformer policies trained per stock. h512, 6 layers, 8 heads. Trained with Sortino loss on hourly data.

**Best model:** `wd_0.06_s42/epoch_008.pt`
- First model profitable across ALL holdout periods (1d through 150d)
- +1.67% (30d), +2.62% (120d), max DD 3.4%
- 6 symbols: NVDA, PLTR, GOOG, DBX, TRIP, MTCH

**Concerns:**
- **Hold period = 5h is brittle** - 4h or 6h breaks it. This suggests overfitting.
- **Direction mismatch is suspicious** - training all LONG then evaluating TRIP/MTCH as SHORT "works better" than training with proper directions. Why?
- Only 6 symbols - small sample
- NYT was catastrophically wrong (classified SHORT, rallied +51.6%)

**Verdict:** These policies feed into the meta-selector (approach #2). The returns are modest but plausible. The brittleness around hold period and direction mismatch are warning signs of overfitting to specific market conditions rather than learning genuine alpha.

---

## 5. FastForecaster / FastForecaster2

**Location:** `FastForecaster/`, `fastforecaster2/`

**What it does:**
- FastForecaster: Pure price forecasting transformer (8-layer, 384-dim). Predicts 24h horizon from 256h lookback. **No trading simulation at all** - evaluated only on MAE/direction accuracy.
- FastForecaster2: Same forecaster + policy sweep that converts predictions to trading signals, evaluated through binanceneural's market simulator.

**Results:**
- Forecasting: Direction accuracy ~50% (coin flip)
- Trading (FF2 best): +0.94% total return, 5.9% win rate, 26% max drawdown
- Only 47 active signals out of 2347 rows

**Verdict:** Direction accuracy at ~50% means the forecaster isn't predicting anything useful. The trading results confirm this - sparse signals, low win rate, high drawdown relative to returns. Not promising.

---

## 6. LLM Trading Agents (DeepSeek / Claude Opus)

**Location:** `stockagentdeepseek*/`, `stockagentopus/`

**What they do:** Use LLMs (DeepSeek Reasoner, Claude Opus 4.5) to generate daily trading plans with limit entries and exits. Some variants add neural forecasts and signal calibration.

**Results:**
- Best backtest: +$6.65 on one simulated day (+0.08%)
- **Live production: -$8,662 over 7 days** (primarily one bad MSFT position)
- Opus agent: never backtested with real API calls

**Simulation concerns:**
- Daily-resolution execution (miss intraday dynamics)
- LLM responses non-deterministic
- No systematic edge - relies on LLM "reasoning" about markets
- Calibration pipeline is interesting but unvalidated at scale

**Verdict:** LLMs are not good at predicting prices. The live loss confirms this. The calibration and neural-forecast augmentation ideas are interesting infrastructure, but the core premise (LLM generates profitable trading plans) is flawed.

---

## 7. ETH Risk PPO

**Location:** `fastalgorithms/eth_risk_ppo/`

**What it does:** PPO agent focused specifically on ETH/USD with risk-adjusted reward. 10-step iteration plan with live-vs-sim fill comparison.

**What's good:**
- Explicitly compares live fills against simulated fills (honesty check)
- Tests with 5bp and 10bp fill buffers before promoting
- Multi-period stress testing required

**Status:** Under active iteration. No final results yet.

**Verdict:** The methodology is sound - requiring positive results at 5-10bp fill buffers is the right way to validate. Worth watching.

---

## Simulation Realism Summary

| Simulator | Fills | Fees | Slippage | Decision Lag | Market Hours | Volume Limits | Score |
|-----------|-------|------|----------|--------------|--------------|---------------|-------|
| PufferLib C env | Instant at bar price | 0.1% | None | None | Tradable mask | None | 5/10 |
| Unified Portfolio Sim | Limit w/ bar margin | Per-symbol | Static (13bp) | 1 bar | Yes + EOD close | None | 8/10 |
| Binance Neural Sim | Penetration-based | Maker fee | Temperature stochastic | 1-2 bars | N/A (crypto) | Volume fraction | 7/10 |
| HourlyTrader Sim | Limit w/ fill buffer | Per-symbol | Buffer bps | 1 bar + cancel delay | Yes | Optional | 8/10 |
| Frontier Market Sim | Instant + slip_bps | Asset-class | 1.5bp | None | N/A | None | 7/10 |
| LLM Agent Sims | Daily bar OHLC | Fixed % | None | Daily | None | None | 4/10 |

---

## What We Actually Trust

### High confidence
- **The unified hourly meta-selector framework** is the best-validated approach. Multi-scenario execution testing, composite scoring, activity filters - the methodology is right even if current results are mixed.
- **Penetration-based fills** (binanceneural) are more honest than binary touch fills.
- **Decision lag modeling** is essential and correctly implemented in the hourly systems.

### Medium confidence
- **Per-stock neural policies** show modest but plausible returns. The brittleness concerns are real but the 150-day OOS test is meaningful.
- **Market-order entry** being required for profitability is a yellow flag - it means the edge doesn't survive even modest execution friction.

### Low confidence
- **PufferLib 2,659x returns** are a simulation artifact. The RL framework is good but the sim needs major upgrades before results mean anything.
- **LLM agents** for trading. Lost money live. The approach is fundamentally misguided.
- **FastForecaster** direction accuracy at ~50%. Not predicting anything.

### Key takeaway
The live account is down ~17%. That's the ground truth. Everything else is backtesting. The approaches with the most rigorous simulation (unified hourly, binance neural) also tend to show the most modest returns - which is probably more honest. The approaches claiming astronomical returns (PufferLib PPO) have the simplest simulators.

---

## Recommendations

1. **Fix the live trading gap.** The meta-selector backtests positive but the live account is losing. Diagnose why - is it execution quality? Forecast staleness? Regime change?

2. **Upgrade PufferLib sim.** Add slippage, volume-aware sizing, and market impact modeling. If the 2,659x even partially survives, it's worth deploying. If it collapses (likely), at least we know.

3. **Continue ETH Risk PPO methodology.** The live-vs-sim comparison approach is the right way to validate any system before deployment.

4. **Retire LLM agents.** They lost money live and there's no theoretical reason to expect LLMs to predict short-term price movements.

---

## Next Experiment: Autoresearch-Style Stock Planner

The next experiment should borrow the workflow from `external/autoresearch/`, but target **stock trading plans** instead of LLM training.

Core design:
- Fixed **5-minute** training budget per run.
- Only `src/autoresearch_stock/train.py` is mutable during the experiment loop.
- `src/autoresearch_stock/prepare.py` owns the fixed evaluation harness.
- Evaluate on both **hourly** and **daily** data.
- Use **live spread history per symbol** from `dashboards/metrics.db` when available, with conservative fallbacks otherwise.
- Apply **adverse next-bar entry pricing**, **volume-fraction caps**, and **multi-window robust scoring** so training efficiency is judged under realistic costs.

Success criterion:
- maximize `robust_score`, not forecasting accuracy,
- prefer models that improve quickly inside the five-minute budget,
- reject ideas that only look good when costs or execution realism are relaxed.

5. **Focus on execution quality.** The market-order vs limit-order sensitivity across all approaches suggests execution is where alpha is won or lost. Build better execution simulation before building more models.
