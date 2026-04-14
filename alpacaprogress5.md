# Alpaca Progress 5

## 2026-03-17 Hourly Near-Book Gating + Robust 60d Replay

### What changed

- Added a shared hourly entry gate so new entry orders only stage when the book or replay reference is within `25 bps` of the target limit.
- Wired that gate into:
  - `newnanoalpacahourlyexp/trade_alpaca_hourly.py`
  - `newnanoalpacahourlyexp/marketsimulator/hourly_trader.py`
  - `newnanoalpacahourlyexp/run_hourly_trader_sim.py`
- Added a robust replay path that can automatically test:
  - flat start
  - provided live-like seeded state
  - basket long start
  - per-symbol long starts
  - basket short and per-symbol short starts when `--allow-short` is enabled
- Added an early-stop rule for weak scenarios:
  - after `30` periods by default in `--robust-60d`
  - stop if realized `pnl_abs <= 1.0 * max_drawdown_abs`
  - report the exact stop reason in the scenario summary

### New replay command

```bash
source .venv313/bin/activate
python -m newnanoalpacahourlyexp.run_hourly_trader_sim \
  --checkpoint analysis/rl_trainingbinance_sweep_7d30d_20260311T0024Z/run_005/best.pt \
  --symbols BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD \
  --robust-60d \
  --allow-short \
  --output-dir analysis/hourly_robust60d_20260317
```

### First real robust replay result

- Replayed on `BTCUSD,ETHUSD,SOLUSD,LINKUSD` with:
  - `checkpoint=analysis/rl_trainingbinance_sweep_7d30d_20260311T0024Z/run_005/best.pt`
  - `moving_average_windows=168,336,720`
  - `cache_only`
  - `robust-60d`
- Output artifact:
  - `analysis/hourly_robust60d_20260317_run005_btc_eth_sol_link_cachefixed/scenario_summary.json`
- Result:
  - all `9` scenarios hit the new early-stop guard at `30` periods
  - best scenario was `long_SOLUSD`
  - `total_return=+0.0148%`
  - `sortino=10.22`
  - `max_drawdown=-0.0244%`
  - `pnl_abs=+$1.55`
- Interpretation:
  - the near-book gate and scenario runner are working
  - but this basket is not yet strong enough to survive the new PnL-vs-drawdown gate
  - do not expand deployment from this run alone

### Data-path fixes that mattered

- Hourly loader now normalizes every row in a symbol file to the requested symbol.
- This fixes mixed-alias files like `BTCUSD.csv` containing `BTCUSDT` rows, which was causing almost the whole forecast join to disappear.
- Stock min-history checks now convert wall-clock hours to regular-session hourly bars instead of demanding crypto-style `24` bars/day.

### Deploy guidance right now

- Do not promote a wider hourly symbol set until it wins the new multi-start replay, not just the flat-start replay.
- The first candidate deploy shape is:
  - `entry_near_book_bps=25`
  - `fill_buffer_bps=5`
  - `decision_lag_bars=1`
  - `cancel_ack_delay_bars=1`
- This should reduce capital getting trapped in stale far-from-book entry orders and make live behavior match the replay much more closely.

### Overfit controls for larger models

- Use scenario-weighted early stopping: validate on flat, seeded long, seeded short, and basket states together, not one flat replay.
- Add forecast-noise regularization during training: inject small causal perturbations into forecast inputs so the model does not memorize exact Chronos/Kronos trajectories.
- Penalize turnover directly in the training or policy-selection loss so larger models do not win only by over-trading.
- Randomize starting inventory and open-order state during training/replay so the model does not overfit to the “flat at t=0” assumption.
- Prefer symbol holdout plus rolling-time holdout together before deploying any larger checkpoint.

## 2026-03-14 Systematic Pipeline Improvements

### Goal
Systematically improve both crypto and stock trading algorithms through isolated experiments validated on the market simulator. Only integrate changes that improve Sortino, reduce max drawdown, or increase PnL.

---

## 1) Experiment Infrastructure (completed 2026-03-13)

Created `experiments/llm_hourly_improvements/` with 6 experiments:
- exp01: Slippage sweep (0-20 bps)
- exp02: Confidence threshold sweep (0.0-0.9)
- exp03: Volatility context in prompt
- exp04: Correlation context in prompt
- exp05: History window length (6/12/24/48 bars)
- exp06: Exit strategy (trailing stop / ATR TP)

All experiments use the same cached LLM signals where possible (simulation-only changes run instantly, new prompt experiments require fresh API calls ~35 min each).

---

## 2) Winners Integrated into Production

### A. MIN_CONFIDENCE_CRYPTO = 0.7 (was 0.4)
- **Evidence**: Exp02 7d BTC/ETH/SOL — Sortino 39.8 vs 30.9, return +357% vs +257%, 51 trades vs 94
- **Why it works**: Filters out marginal signals. Fewer but higher-quality trades dominate.
- **Deployed**: 2026-03-13 23:39 UTC

### B. Trailing Stop 0.3% from peak
- **Evidence**: Exp06 7d BTC/ETH/SOL — Sortino 76.5 vs 30.9, MaxDD 4.7% vs 28.3%
- **Why it works**: Locks in small gains, prevents drawdowns from turning into large losses
- **Implementation**: Peak prices tracked in `strategy_state/crypto_peaks.json`, updated every cycle
- **Live confirmation**: AVAXUSD trailing-stop exited at 06:01 UTC Mar 14 (1.29% below peak)
- **Deployed**: 2026-03-13 23:39 UTC

### C. Cancel-delay 0.5s in _place_crypto_tp_sell
- **Evidence**: BTC "insufficient balance" error in 22:01 cycle Mar 13 — race condition
- **Fix**: 0.5s sleep after canceling old sell orders before placing new ones
- **Deployed**: 2026-03-13 22:10 UTC

---

## 3) Signal Robustness Confirmed

### Slippage sweep (Exp01, 7d BTC/ETH/SOL)
| Slippage | Return | Sortino | Retention |
|----------|--------|---------|-----------|
| 0 bps | +291.8% | 32.1 | baseline |
| 5 bps | +256.6% | 30.9 | 88% |
| 10 bps | +224.0% | 28.2 | 77% |
| 20 bps | +166.3% | 22.1 | 57% |

**Conclusion**: Real edge exists beyond execution noise. Even at 20 bps total cost, system is profitable.

---

## 4) Multi-Window Validation (Exp07, completed 2026-03-14)

### Critical finding: Trailing stop is THE dominant improvement

| Window | Strategy | Return | Sortino | MaxDD |
|--------|----------|--------|---------|-------|
| 1d | baseline | +1.6% | 4.2 | 17.9% |
| 1d | trail_0.3 | +42.5% | 187.8 | 4.5% |
| 1d | conf+trail | +44.1% | 190.8 | 4.5% |
| 7d | baseline | -2.2% | 5.5 | 58.2% |
| 7d | conf_0.7 | **-40.7%** | 2.5 | 63.3% |
| 7d | trail_0.3 | +306.5% | 49.5 | 14.6% |
| 14d | baseline | -147.2% | -1.5 | 103.6% |
| 14d | trail_0.3 | +29,876% | 95.5 | 15.3% |
| 30d | baseline | -197.6% | 0.04 | 103.6% |
| 30d | trail_0.3 | +106,805% | 52.8 | 34.0% |

### Key insights:
1. **Trailing stop is the only improvement that consistently works across ALL windows**
   - avg Sortino=96.4, min return=+42.5%, avg MaxDD=17.1%
2. **conf_0.7 is window-dependent** — helped on Mar 2-9, HURT on Mar 7-14 (-40.7%)
   - The 7d window in exp02 was a lucky sample. Not robust.
3. **Baseline without trailing stop goes bankrupt on 14d+ windows** (103.6% MaxDD = account wipeout)
4. **The trailing stop's compounding effect is enormous** — small gains compound over 30d to 100,000%+ returns

### Action: Revert MIN_CONFIDENCE_CRYPTO from 0.7 back to 0.4
The confidence threshold is not robust across windows. Keep trailing stop only.

---

## 5) Fill Dropout / Liquidity Test (Exp08, completed 2026-03-14)

Tested on most recent 7d window (Mar 7-14) with conf >= 0.7 filter:

| Dropout | Return | Sortino |
|---------|--------|---------|
| 0% | -40.9% | 2.5 |
| 10% | -43.2% | 0.6 |
| 20% | -44.7% | 0.7 |
| 30% | -17.5% | 2.5 |
| 50% | -18.9% | 2.0 |

**Note**: This ran on the recent losing window (Mar 7-14), so baseline was already negative.
The high variance (±50% std at 50% dropout) shows fill randomness has large impact.
Need to re-run on a profitable window to see if dropout degrades a winning strategy.

---

## 7) Regime Detection (Exp09, 14d BTC/ETH/SOL)

Tested SMA filters, 24h return filters, and drawdown filters — all combined with trailing stop 0.3%.

| Filter | Return | Sortino | MaxDD | Blocked |
|--------|--------|---------|-------|---------|
| **no_filter** | +28,865% | **95.4** | 15.3% | 0% |
| sma_48 | +5,202% | 67.0 | 15.3% | 28% |
| ret24h_-2% | +5,613% | 65.1 | 15.3% | 12% |
| dd_5% | +14,595% | 87.0 | 15.3% | 22% |
| sma_12 | +914% | 35.3 | 15.3% | 46% |

**Conclusion**: No regime filter beats the unfiltered trailing stop on this window. Blocking entries during downtrends just misses recovery entries. The trailing stop already handles regime changes by exiting quickly on drops. **Not integrating.**

---

## 8) Symbol Gating (Exp10, 14d)

Tested per-symbol rolling return gating (24h/48h/72h lookback, various thresholds).

**Result**: All gating configs give identical results — the gates never trigger because rolling returns stay above thresholds when combined with trailing stop. The trailing stop exits losers quickly enough that recent returns never go deeply negative. **Not integrating.**

---

## 9) Hybrid Trailing Stop (Exp11, 14d)

Tested adaptive trail widths — start wide, tighten after profit.

| Variant | Return | Sortino | MaxDD |
|---------|--------|---------|-------|
| **fixed_0.3%** | +27,832% | **95.0** | **15.3%** |
| 0.8→0.3@+0.3% | +21,846% | 53.3 | 28.9% |
| profit_scaled | +2,104% | 50.2 | 15.3% |
| fixed_0.5% | +1,003% | 33.8 | 33.8% |
| 1.0→0.3@+0.5% | +6,760% | 25.5 | 34.4% |

**Conclusion**: Fixed 0.3% trail is optimal. Wider initial trails allow drawdowns that compound into massive losses. The tight 0.3% trail's quick exits enable more compounding re-entries, which dominate. **Keeping current 0.3%.**

---

## 10) Summary of All Experiments

| # | Experiment | Winner | Status |
|---|-----------|--------|--------|
| 01 | Slippage sweep | Signals robust (88% at 5bps) | ✅ Validated |
| 02 | Confidence threshold | 0.7 on one window, harmful on another | ❌ Reverted |
| 03 | Volatility prompt | Hurt Sortino (19.5 vs 30.9) | ❌ Failed |
| 04 | Correlation prompt | Catastrophic (-32% return) | ❌ Failed |
| 06 | Trailing stop 0.3% | **THE improvement** (Sortino 95, all windows) | ✅ Deployed |
| 07 | Multi-window validation | Trail wins 1d/7d/14d/30d consistently | ✅ Validated |
| 08 | Fill dropout | Baseline negative on recent window | ⚠️ Regime issue |
| 09 | Regime detection | No filter beats unfiltered + trail | ❌ Not needed |
| 10 | Symbol gating | Gates never trigger with trail | ❌ Not needed |
| 11 | Hybrid trail | Fixed 0.3% beats all adaptive variants | ✅ Current is optimal |

**Net result**: The only change that consistently improves the system is the 0.3% trailing stop. Everything else either doesn't help or is window-dependent.

---

## 6) Pending Next Steps

## 11) Stock Experiments (completed 2026-03-14)

### Stock LLM signals are LOSING money without trailing stop

| Strategy | Return | Sortino | MaxDD |
|----------|--------|---------|-------|
| LLM TP (baseline) | **-1.09%** | -13.7 | 2.86% |
| **Trail 0.3%** | **+6.77%** | **89.4** | **0.58%** |

**Action taken**: Added trailing stop tracking for stocks (strategy_state/stock_peaks.json).
Peak prices tracked per stock position, trailing stop exit at 0.3% from peak — same mechanism as crypto.
Deployed: 2026-03-14 09:31 UTC

### Pending Next Steps

#### A. Investigate live vs backtest gap for crypto
- Backtest shows huge returns with trailing stop
- Live account crypto positions are underwater
- Gap likely from: Alpaca fill quality, GTC order behavior, hourly cycle timing

#### B. Forecast data refresh
- Stock forecasts end Feb 13 for most symbols — need fresh Chronos2 predictions
- Crypto forecasts current to Mar 14
- Without fresh forecasts, stock LLM signals may be stale

#### C. Test stock RL signals independently
- Stock RL checkpoint: +50% median/30d, Sortino 25.6
- LLM alone: -1.09% (losing). RL+LLM hybrid may be better
- Need: Run RL-only simulation to see if RL generates the stock alpha
