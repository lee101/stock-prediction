# Binance Optimization Progress (2026-03-14)

## Live vs Backtest Divergence

### Live P&L (48h ending 2026-03-14 08:00 UTC)
| Symbol | Trades | Winners | Gross P&L | Fees | Net P&L |
|--------|--------|---------|-----------|------|---------|
| BTC | 3 closed + 1 open | 2/3 | -$30.0 | ~$19.4 | -$49.3 |
| ETH | 3 closed + 1 open | 0/3 | -$84.6 | ~$15.4 | -$100.1 |
| SOL | 2 closed + 1 open | 0/2 | -$62.1 | ~$7.9 | -$70.0 |
| **Total** | **8 closed** | **2/8** | **-$176.7** | **~$42.7** | **-$219.3** |

### Backtest P&L (3-day, h1_only variant, 5x leverage)
| Symbol | Return% | Sortino | MaxDD% | Trades |
|--------|---------|---------|--------|--------|
| All 3 | +13.2% | 81.85 | 1.4% | 82 |

### Gap Analysis
- Live: -8.9% in 48h. Backtest: +13.2% in 72h
- Live fees ~10bps per side eating 25% of gross trades
- 2/8 win rate live vs backtest showing strong positive expectancy
- ETH bought at $2147 (near local high) then forced to sell at $2085 = -2.9% loss on one trade
- Key issue: backtest limit fills assume low=buy_price triggers fill; live fills at market price with slippage

## Bug Fixes Applied (2026-03-14)
- Missing take-profit sell after buy: bot placed buy limits but only set TP sell on NEXT hourly cycle
- Fix 1: Place immediate sell when buy order shows status=FILLED
- Fix 2: Balance-check fallback -- query actual asset balance 1s after buy to confirm fill when order returns status=NEW

## Chronos2 Cross-Learning Experiment (2026-03-14)

Evaluated joint prediction (BTC+ETH+SOL together) vs univariate (each symbol alone).
Window: 550 hours (Feb 12 - Mar 7 2026), h1 horizon.

| Symbol | Baseline MAE% | Cross-Learning MAE% | Improvement |
|--------|---------------|---------------------|-------------|
| BTC | 0.405% | 0.234% | **-42.1%** |
| ETH | 0.515% | 0.313% | **-39.3%** |

Cross-learning dramatically improves forecast accuracy by letting the model see correlated asset price movements jointly. This is the single biggest forecast quality lever found.

### Context Length Experiment
| BTC Config | MAE% | Note |
|-----------|-------|------|
| ctx=256 | 0.407% | Slightly worse |
| ctx=512 (baseline) | 0.405% | Current prod |
| Conclusion: context length doesn't matter much for h1 BTC.

## Multi-Window Backtest (h1_only, 5x leverage, 2026-03-14)

| Window | Return% | Sortino | MaxDD% | Trades | PnL |
|--------|---------|---------|--------|--------|-----|
| 1-day | +1.505 | 28.17 | 0.745 | 27 | +$46.11 |
| 3-day | +13.218 | 81.85 | 1.415 | 82 | +$315.38 |
| 7-day | +32.693 | 47.72 | 2.283 | 186 | +$782.04 |

All windows positive. Sortino excellent (28-82). MaxDD stays low (0.7-2.3% at 5x leverage). 7-day shows ~4.7%/day annualized return with 186 trades (26.6/day avg). The backtest signals are strong -- live divergence is execution-related.

## Prod Forecast Issue Found (2026-03-14)
The cache refresh supervisor has `CHRONOS2_MODEL_ID_OVERRIDE` pointing to base Chronos2 model, which overrides the per-symbol LoRA-tuned checkpoints. This means prod forecasts use the **base model, not the LoRA fine-tuned models**. Needs investigation on whether removing this override is safe (LoRA checkpoints need to be loadable).

Cross-learning requires processing multiple symbols jointly in `build_forecast_bundle`, but the cache refresh script processes them one at a time. Need to modify the refresh script to support joint prediction mode.

## Correlation Analysis (2026-03-14)

Best covariates for cross-learning (hourly return correlation, 720h window):

| Target | #1 | #2 | #3 | #4 |
|--------|---------|---------|---------|---------|
| BTC | ETH (0.91) | LINK (0.86) | BNB (0.86) | SOL (0.84) |
| ETH | LINK (0.92) | BTC (0.91) | SOL (0.84) | BNB (0.84) |
| SOL | LINK (0.88) | DOGE (0.87) | AVAX (0.84) | ETH (0.84) |

LINK is top-3 covariate for all targets. All crypto pairs are highly correlated (0.74-0.92).

## Next Steps
1. Modify cache refresh to support cross-learning (joint symbol processing)
2. Test cross-learning forecasts end-to-end through LLM backtest
3. Fix remote 5090 forecast MAE eval pipeline (parquet output issue)
4. Retrain LoRA adapters on recent data (last tuned Feb 6, 5+ weeks stale)
5. Extend cross-learning to stock forecasts (NVDA, PLTR, etc)
6. Per-symbol RL policy training sweep for BTC/ETH/SOL

## Experiments Log
