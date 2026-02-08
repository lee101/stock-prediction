# PnL Improvement Plan - Feb 2026

## Goal
Push Alpaca cross-learning PnL beyond current best (55% / 70d, sortino 105) by expanding symbols, enabling Chronos2 multivariate group attention, multi-horizon forecasts, modern feature engineering, and multi-position trading.

## Current Best
- **Mixed14 crypto5+TSLA**: 54.8% over 70d, sortino 104.6
- **Crypto-only 5 syms**: 200% over 90d, sortino 66
- Pipeline: Chronos2 LoRA (r=16) -> h1/h24 forecasts -> transformer policy (4L/8H/256d) -> single-position selector

## What We're Doing

### Phase 0: Baseline + Data Prep
- [x] Downloaded AVAXUSD (18k rows), verified DOTUSD (21k), AAVEUSD (40k), COIN (8.6k)
- [ ] Record baseline metrics for comparison

### Phase 1: Expand to 8 Crypto + COIN
- Add AVAXUSD, DOTUSD, AAVEUSD (high-vol alts) + COIN (crypto-correlated stock)
- More alt-coins = more hourly opportunities for selector
- No code changes, just run existing CLI with new symbol set

### Phase 2: Multi-Horizon Forecasts (h1/h4/h12/h24)
- Currently only h1 + h24 - missing intermediate momentum signals
- h4 = intraday momentum, h12 = half-day trend
- Policy gets 4x the forecast signal: agreement across horizons = higher conviction

### Phase 3: Modern Feature Engineering
- **Forecast confidence**: `ref_close / (p90 - p10)` - size bigger when Chronos is certain
- **Multi-horizon agreement**: do h1/h4/h12/h24 all point same direction? (0-1 score)
- **BTC-relative features**: btc_return_1h, btc_correlation_72h, relative_strength_btc_24h
- Cross-symbol context the policy currently lacks entirely

### Phase 4: Chronos2 Grouped Cross-Attention
- Chronos2 has NATIVE `group_ids` + `GroupSelfAttention` - correlated series attend to each other
- Currently unused: each symbol = separate task = no cross-attention
- Group correlated symbols into single tasks during fine-tuning:
  - `crypto_majors`: [BTCUSD, ETHUSD]
  - `crypto_alts`: [SOLUSD, LINKUSD, UNIUSD, AVAXUSD, DOTUSD, AAVEUSD]
- Joint forecasting with ~80% MAE improvement (already proven in wrapper)

### Phase 5: LoRA Rank Sweep
- Only tested r=16/alpha=32 so far
- Sweep r={8, 16, 32, 64} with alpha=2r
- Lower rank = less overfitting, higher rank = more capacity for 24+ symbols

### Phase 6: Multi-Position Selector
- Current selector holds ONE symbol at a time - misses diversification
- Implement top-K allocation (hold 2-3 positions simultaneously)
- Each sized at cash/K, diversification should boost sortino significantly

### Phase 7: Integration
- Combine best LoRA rank + grouped attention + 4 horizons + all features + multi-position
- Full sweep on expanded 8-crypto + COIN universe
- Compare to Phase 0 baseline

## Key Technical Approach
- NanoChat-style transformer (RoPE, RMSNorm, MQA, GELU) already in `BinanceHourlyPolicyNano`
- Differentiable sortino loss through simulated trades
- Chronos2 `GroupSelfAttention` for multivariate cross-series learning
- LoRA for parameter-efficient fine-tuning of foundation model
- Walk-forward evaluation windows (10d/30d/60d/90d)

## Files Modified
| File | What |
|------|------|
| `binanceexp1/config.py` | forecast_horizons=(1,4,12,24), include_cross_symbol |
| `newnanoalpacahourlyexp/config.py` | forecast_horizons=(1,4,12,24) |
| `binanceneural/forecasts.py` | Keep p10/p90 in build_forecast_bundle |
| `binanceexp1/data.py` | Confidence, agreement, BTC cross-symbol features |
| `newnanoalpacahourlyexp/data.py` | Wire cross-symbol features |
| `alpacanewccrosslearning/data.py` | build_grouped_inputs_for_symbols |
| `alpacanewccrosslearning/chronos_finetune_multi.py` | --group-symbols, --symbol-groups |
| `newnanoalpacahourlyexp/marketsimulator/selector.py` | Multi-position simulation |
| `alpacanewccrosslearning/run_global_selector.py` | --max-concurrent-positions |
| `alpacanewccrosslearning/run_selector_sweep.py` | --max-positions-list |
