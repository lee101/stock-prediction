# Binance Pure Crypto Optimization (2026-03-15)

## Objective

Find the **highest Sortino, lowest max-drawdown, best annualized PnL** strategy for pure crypto trading on Binance. This will run live on Binance — every decision here must be validated in a market simulator that closely matches real execution.

## Previous Key Findings (binanceprogress1-5 + dailyvshourly.md)

| Finding | Source | Impact |
|---------|--------|--------|
| Daily RL 3.3x better than hourly | dailyvshourly.md | +108% vs +32.5% annualized OOS |
| trade_penalty=0.05 is the daily winner | autoresearch_daily | +20% OOS, 100% profitable, Sortino 1.76 |
| trade_penalty=0.10 also strong daily | sweep_daily_combos | +8.86% OOS, 96% profitable, Sortino 1.24 |
| slip_5bps is the hourly winner | autoresearch_hourly | +5.3% OOS, 96% profitable, Sortino 1.62 |
| 5-min timeboxed training prevents overfitting | OOS eval | 200M+ steps → -3% to -16% OOS |
| 5x leverage optimal for hourly LLM | binanceprogress4 | Sortino 74.69, MaxDD 1.157% |
| Cross-learning improves forecasts 42% | binanceprogress5 | BTC MAE 0.405% → 0.234% |
| 1x leverage >> 2x for RL (48% profitable at 2x) | memory | Risk explodes with leverage on RL alone |
| Live-backtest gap ~2-3% per trade | binanceprogress5 | Limit fills at market ≠ bar touch |
| FDUSD = 0 fee, USDT = 10bps fee | Binance tier | Fee structure is a MAJOR strategic lever |

## Binance-Specific Fee Structure

This is the **single most important** detail differentiating our Binance strategy from generic backtesting:

| Quote Asset | Maker Fee | Taker Fee | Available Pairs | Notes |
|-------------|-----------|-----------|-----------------|-------|
| **FDUSD** | **0%** | **0%** | BTC, ETH, SOL, BNB | Promotional zero-fee. Use LIMIT orders exclusively |
| **USDT** | **0.1%** | **0.1%** | All pairs | Standard fee. 10bps round-trip = 20bps per trade |
| **U (futures)** | 0% | varies | BTC, ETH etc | Futures. Different account type |

**Strategic implication**: With FDUSD, the **only cost** is spread/slippage (~2-5bps on majors). This makes hourly trading FAR more attractive on FDUSD than previously tested (where we assumed 10bps fee). Daily's fee advantage shrinks dramatically when fees are zero.

## Experiment Design: Daily vs Hourly on Binance

### Phase 1: Data Preparation

**FDUSD-3 symbols**: BTCUSD, ETHUSD, SOLUSD (BNB lacks sufficient historical data — only 49 hourly bars)
- Hourly train: 37,060 bars (~4.2 years, Mar 2021 - Jun 2025)
- Hourly val: 6,868 bars (~9.5 months, Jun 2025 - Mar 2026)
- Daily train: 1,210 days (Feb 2022 - Jun 2025)
- Daily val: 286 days (Jun 2025 - Mar 2026)
- Hourly uses price-only features (no Chronos2 dependency) via `export_data_hourly_priceonly.py`
- Daily uses price-only features via `export_data_daily.py`

**USDT-extended symbols** (for diversification sweep): LTCUSD, AVAXUSD, DOGEUSD, LINKUSD, AAVEUSD
- These pay 10bps fee → fee_rate=0.001 in C env
- Only worthwhile if edge > 20bps round-trip
- All have 37,000+ hourly bars and 1,400+ daily bars

### Phase 2: Core Autoresearch Sweeps

Run full 35-config autoresearch on EACH combination:

| Experiment | Timeframe | Fee Rate | Symbols | Episode Length |
|------------|-----------|----------|---------|---------------|
| **fdusd_hourly** | Hourly | 0.0 | 4 FDUSD | 720h (30d) |
| **fdusd_daily** | Daily | 0.0 | 4 FDUSD | 90d |
| **usdt_hourly** | Hourly | 0.001 | 4-6 USDT | 720h (30d) |
| **usdt_daily** | Daily | 0.001 | 4-6 USDT | 90d |

All evaluations use 8bps fill slippage (conservative for Binance majors).

### Phase 3: Leverage Sweep

On best configs from Phase 2, sweep leverage:

| Leverage | Max Leverage | Short Borrow APR | Notes |
|----------|-------------|-------------------|-------|
| 1x | 1.0 | 0% | Spot long-only (current prod) |
| 2x | 2.0 | 6.25% | Cross margin |
| 3x | 3.0 | 6.25% | Cross margin |
| 5x | 5.0 | 6.25% | Max cross margin |
| 1x+short | 1.0 | 6.25% | Spot + short capability |
| 3x+short | 3.0 | 6.25% | Full margin both directions |

**Critical**: 1x was 100% profitable; 2x was 48%. Leverage amplifies both returns AND drawdowns. Need Sortino-focused reward shaping to handle leverage.

### Phase 4: Reward Shaping for Low-Drawdown

For leveraged strategies specifically:
- `downside_penalty`: [0.0, 0.2, 0.5, 1.0] — penalize negative returns
- `smooth_downside_penalty`: [0.0, 0.3, 0.5] — smooth version (differentiable)
- `drawdown_penalty`: [0.0, 0.1, 0.3] — penalize equity drops from peak
- `smoothness_penalty`: [0.0, 0.1, 0.3] — penalize return volatility
- Combined with `trade_penalty` sweep [0.01, 0.03, 0.05, 0.08, 0.10]

### Phase 5: Signal Stacking

Once best base models identified:
1. **Chronos2 forecasts** → embed h1/h24 predictions as features in daily data
2. **Gemini 3.1 Flash overlay** → direction filter + entry price refinement
3. **Ensemble daily+hourly** → daily sets direction, hourly times entry
4. **SMA filter** → suppress longs below SMA-24 (already proven in prod)

### Phase 6: Market Simulator Validation

Final candidates run through `marketsimulator.py` with Binance-realistic parameters:
- `maker_fee=0.0` (FDUSD) or `maker_fee=0.001` (USDT)
- `max_hold_hours=6` (hourly) or `max_hold_hours=72` (daily, 3-day max hold)
- Shared cash simulation across symbols
- Slippage: 5bps for BTC/ETH, 8bps for SOL/BNB, 12bps for alts
- Short borrow fee: 6.25% APR (Binance cross margin rate)

## Annualized PnL Comparison Framework

All results will be reported as **annualized** for fair comparison:

```
annualized_return = (1 + period_return) ^ (365 / period_days) - 1
```

| Metric | Target | Minimum Acceptable |
|--------|--------|-------------------|
| Annualized Return | >100% | >30% |
| Sortino Ratio | >2.0 | >1.0 |
| Max Drawdown | <5% | <15% |
| Win Rate | >55% | >50% |
| Profitable Episodes | >95% | >80% |

## Current Baseline Performance

### Daily RL (trade_pen_05, fee=10bps)
- **Annualized OOS**: +108.1%
- **Sortino**: 1.76
- **MaxDD**: <7% (worst 90d episode: +6.84%)
- **Profitable**: 100% (500/500)
- **Fee drag**: ~3% annual

### Hourly RL (slip_5bps, fee=10bps)
- **Annualized OOS**: +32.5% (250-day eval)
- **Sortino**: 1.10
- **Profitable**: 79.3%
- **Fee drag**: ~10% annual

### Hourly LLM (Gemini, position_context, 5x leverage)
- **3-day backtest**: +13.2%, Sortino 81.85, MaxDD 1.4%
- **Annualized (extrapolated)**: ~1600%+ (unreliable, small sample)
- **Live reality**: -8.9% in 48h (execution gap)

## Experiments To Run

### Experiment 1: FDUSD Zero-Fee Hourly RL
**Hypothesis**: Zero fees should dramatically improve hourly RL (previous winner was held back by 10bps fee drag).

```bash
# Export FDUSD-4 hourly data
python -m pufferlib_market.export_data \
    --symbols BTCUSD,ETHUSD,SOLUSD,BNBUSD \
    --output-train pufferlib_market/data/fdusd4_hourly_train.bin \
    --output-val pufferlib_market/data/fdusd4_hourly_val.bin

# Run autoresearch with fee=0
python -u -m pufferlib_market.autoresearch_rl \
    --train-data pufferlib_market/data/fdusd4_hourly_train.bin \
    --val-data pufferlib_market/data/fdusd4_hourly_val.bin \
    --time-budget 300 --max-trials 50 \
    --leaderboard pufferlib_market/autoresearch_fdusd_hourly_leaderboard.csv \
    --checkpoint-root pufferlib_market/checkpoints/autoresearch_fdusd_hourly
```

### Experiment 2: FDUSD Zero-Fee Daily RL
**Hypothesis**: Daily with zero fees may not improve as much (already few trades), but still worth testing since fee_2x was #3 on daily.

```bash
# Export FDUSD-4 daily data
python -m pufferlib_market.export_data_daily \
    --symbols BTCUSD,ETHUSD,SOLUSD,BNBUSD \
    --output-train pufferlib_market/data/fdusd4_daily_train.bin \
    --output-val pufferlib_market/data/fdusd4_daily_val.bin

# Run autoresearch daily with fee=0
python -u -m pufferlib_market.autoresearch_rl \
    --train-data pufferlib_market/data/fdusd4_daily_train.bin \
    --val-data pufferlib_market/data/fdusd4_daily_val.bin \
    --time-budget 300 --max-trials 50 \
    --periods-per-year 365 --max-steps-override 90 \
    --leaderboard pufferlib_market/autoresearch_fdusd_daily_leaderboard.csv \
    --checkpoint-root pufferlib_market/checkpoints/autoresearch_fdusd_daily
```

### Experiment 3: Leverage Sweep on Best Configs
**Hypothesis**: With Sortino-focused reward shaping, moderate leverage (2-3x) may beat 1x without the 48% profitability collapse seen before.

Requires adding `--max-leverage`, `--short-borrow-apr`, and potentially `--disable-shorts` flags to autoresearch, then sweeping:

```bash
# For each leverage level [1, 2, 3, 5]:
python -u -m pufferlib_market.train \
    --data-path pufferlib_market/data/fdusd4_hourly_train.bin \
    --max-leverage $LEV --short-borrow-apr 0.0625 \
    --trade-penalty 0.05 --fee-rate 0.0 \
    --total-timesteps 999999999 --anneal-lr \
    --hidden-size 1024 --max-steps 720 \
    --checkpoint-dir pufferlib_market/checkpoints/leverage_sweep/lev${LEV}
```

### Experiment 4: Sortino-Optimized Reward Shaping
**Hypothesis**: Adding downside/drawdown penalties during training produces models that trade more conservatively but with better risk-adjusted returns.

Sweep matrix (on best fee/timeframe from Experiments 1-2):
- `trade_penalty` × `downside_penalty` × `leverage`
- Focus: Sortino > 2.0, MaxDD < 10%

### Experiment 5: Daily+Hourly Ensemble
**Hypothesis**: Daily model sets direction (long/short/flat), hourly model times entry within the day. Combined should beat either alone.

Architecture:
```
Daily model (midnight UTC):
  → Direction: LONG BTC, SHORT ETH, FLAT SOL, LONG BNB

Hourly model (each hour):
  → Only act if direction matches daily signal
  → Time entry/exit within the daily window
  → Trailing stop: 0.3% (proven in prod)
```

### Experiment 6: Add Chronos2 Forecasts to Daily
**Hypothesis**: Daily models currently use pure technicals (16 features). Adding Chronos2 h24 forecast deltas should improve signal quality significantly (42% MAE improvement from cross-learning).

Requires modifying `export_data_daily.py` to embed daily Chronos2 forecast features.

### Experiment 7: Gemini LLM Direction Filter
**Hypothesis**: LLM overlay with position context improved Sortino from ~30 to 78 on hourly. Should also help daily.

Test on best daily+hourly models with:
- Gemini 2.5 Flash (production model)
- h1_only horizon (outperformed h24 for Sortino)
- Position context prompt (proven superior)

## Implementation Plan

### Step 1: Fee-Rate Support in Autoresearch
Add `--fee-rate` CLI flag to `autoresearch_rl.py` so we can sweep fee=0 vs fee=10bps.

### Step 2: Export FDUSD-Specific Data
Create hourly + daily binary exports for BTC/ETH/SOL/BNB FDUSD pairs.

### Step 3: Run FDUSD Sweeps (Experiments 1 & 2)
~35 configs × 2 timeframes × 5 min each = ~6 hours total. Can run in parallel on GPU.

### Step 4: Leverage Sweep Script
Create `sweep_leverage.py` that tests leverage=[1,2,3,5] × best configs × fee tiers.

### Step 5: Market Simulator Validation
Run top-3 candidates through full `marketsimulator.py` with Binance-realistic params.

### Step 6: Signal Stacking (Chronos2 + Gemini)
Layer on forecast features and LLM overlay on best base models.

### Step 7: Paper Trade 30 Days
Deploy winner on Binance testnet before real capital.

## Market Simulator Realism Checklist

| Aspect | Current Sim | Binance Reality | Gap |
|--------|-------------|----------------|-----|
| Maker fee FDUSD | 0.1% default | **0%** | **Must fix** |
| Maker fee USDT | 0.1% | 0.1% | OK |
| Taker fee | N/A | 0.1% | Use limit orders only |
| Slippage | 5-8bps | 2-5bps majors, 8-15bps alts | OK (conservative) |
| Short borrow | configurable | ~6.25% APR (varies) | OK |
| Leverage | configurable | up to 5x cross | OK |
| Fill probability | configurable | ~95-99% for majors | Set 0.95-0.98 |
| Order book depth | N/A | Deep for majors | N/A (limit orders) |
| Max hold enforcement | configurable | Manual via bot | OK |
| Position sizing | % of cash | Min notional varies | Need to check |

## Early Results: Slippage Sensitivity Analysis

Before the FDUSD-specific sweeps complete, we can evaluate existing models at FDUSD-realistic slippage (3bps instead of 8bps):

| Strategy | 8bps Slippage | 3bps Slippage | Improvement |
|----------|--------------|--------------|-------------|
| **Daily trade_pen_05** | +108.1% ann, Sortino 1.76 | **+132.9% ann, Sortino 1.90** | +23% |
| **Hourly slip_5bps** | +32.5% ann, Sortino 1.10 | **+84.8% ann, Sortino 1.61** | +161% (2.6x!) |

**Critical finding**: Hourly trading benefits FAR more from lower slippage (2.6x improvement) because it trades ~6x more often. At FDUSD-realistic slippage, hourly narrows the gap to 1.6x (vs 3.3x at 8bps). But daily still wins.

Note: These use the 5-symbol crypto6 data (includes LTC/AVAX which would be USDT 10bps in reality). The FDUSD-3 specific sweeps are running now.

## Expected Outcome

Based on prior findings, the prediction is:

1. **FDUSD hourly RL should dramatically improve** — removing 10bps fee removes the main drag. Predicted: +60-80% annualized (up from +32.5%).
2. **FDUSD daily RL should stay similar** — it already makes few trades. Predicted: +100-120% annualized.
3. **Moderate leverage (2-3x) with Sortino shaping should work** — previous 2x failure was without reward shaping. Predicted: 2x → +150-200% annualized with Sortino > 1.5.
4. **Best combination will be**: FDUSD daily direction + hourly entry timing + 2-3x leverage + Gemini filter + trailing stop. Target: **+200-300% annualized, Sortino > 2.0, MaxDD < 10%**.

## Decision Criteria

Deploy the strategy that maximizes:

```
score = annualized_return × min(sortino / 2.0, 1.0) × (1 - max_drawdown / 0.15)
```

This penalizes high-return strategies with poor risk management. A +200% return with Sortino 1.0 and 15% MaxDD scores lower than +150% with Sortino 2.5 and 5% MaxDD.

## Risk Management for Live Deployment

| Control | Setting | Rationale |
|---------|---------|-----------|
| Max position size | 20% of account per symbol | No single-symbol blow-up |
| Max leverage | 3x (even if 5x available) | Leave margin buffer |
| Trailing stop | 0.3% from peak | Proven in hourly prod |
| Daily loss limit | -3% of account | Stop trading for the day |
| Weekly loss limit | -8% of account | Stop trading, review model |
| Max hold (hourly) | 6 hours | Proven optimal |
| Max hold (daily) | 72 hours (3 days) | Prevents multi-day underwater |
| SMA filter | Price > SMA-24 for longs | Suppress trades in downtrends |
| Execution | Limit orders ONLY | Capture maker fee tier |
| Kill switch | Manual abort | Always available |

---

*Status: Experiment design complete. Implementation starting.*
*Data: 5 crypto symbols × hourly+daily × train+val exports needed*
*Compute: ~12 GPU-hours for full sweep*
*Target: Identify production candidate within 24h*
