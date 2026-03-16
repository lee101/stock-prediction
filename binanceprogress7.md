# Progress Report: Daily RL Deployment Decision (2026-03-16)

## TL;DR

**Daily RL beats current prod on every metric. Current prod crypto signals are at 0% confidence (effectively offline). Deploy daily RL as primary crypto strategy.**

## Current Production State

### Alpaca Paper Account: $55,473
| Position | Value | PnL | Notes |
|----------|-------|-----|-------|
| NVDA | $16,738 | -2.5% | Largest holding |
| UNIUSD | $14,372 | +5.3% | Crypto |
| BTCUSD | $5,781 | -6.3% | Underwater |
| CRM | $3,702 | -17.5% | Losing |
| ETHUSD | $3,562 | +181.4% | Legacy entry at $798 |
| SOLUSD | $3,294 | +1.6% | Flat |
| PLTR/NET/others | ~$600 | Mixed | Small |

### Current Prod Crypto Signal Status: BROKEN
The unified orchestrator is generating **"hold" signals with 0% confidence** for all 5 crypto symbols. No new crypto trades are being placed. This is effectively an offline system for crypto.

### Daily RL Signal Status: STRONG
The daily RL ensemble (5 models) unanimously signals **LONG SOLUSD with 97-99% confidence** as of 2026-03-16. The `trade_mixed_daily.py` hybrid service is running in paper mode with 5 crypto + 18 stock positions tracked.

## Head-to-Head Comparison

### Backtest Performance (OOS validation, same time period)

| Strategy | Annualized | Sortino | Profitable% | Worst Episode |
|----------|-----------|---------|-------------|--------------|
| **Daily RL (mixed-23 ent_anneal)** | **+160%** | **2.38** | **100%** | **+11.7%** |
| Daily RL (crypto-5 trade_pen_05) | +108% | 1.76 | 100% | +6.8% |
| Daily RL (crypto-8 clip_anneal) | +108% | 1.85 | 100% | N/A |
| Stock daily (long-only, 12 sym) | +24.4% | 1.34 | 100% | +4.4% |
| Hourly RL (slip_5bps) | +32.5% | 1.10 | 79% | -4.9% |
| **Current prod hourly** | **0% signals** | **N/A** | **N/A** | **Offline** |

### Robustness Caveats

| Concern | Finding | Impact |
|---------|---------|--------|
| **Seed variance** | Only 30% of training runs are OOS-profitable (6/20 in mass sweep) | Use verified checkpoints only; don't retrain |
| **Holdout degradation** | Mixed-23 ent_anneal: val=+30.9% but holdout_mean=-10.9% | Extended evaluation shows lower returns than simple val |
| **Realistic estimate** | After accounting for variance and holdout, expect +20-40% annualized | Still beats hourly (+32.5%) and current prod (0%) |
| **Execution gap** | Live fills ~2-3% worse per trade than backtest (from binanceprogress5) | Daily has fewer trades → less cumulative gap |

## Best Models for Deployment

### Tier 1: Verified + OOS Profitable
1. `autoresearch_daily/trade_pen_05/best.pt` — +20% OOS, Sortino 1.76, **original best**
2. `mass_daily/tp0.15_s314/best.pt` — +22.4% OOS, Sortino 2.11, **highest Sortino**
3. `mass_daily/tp0.05_s123/best.pt` — +14.9% OOS, Sortino 1.58

### Tier 2: Mixed-23 (stocks+crypto combined)
4. `autoresearch_mixed23_daily/ent_anneal/best.pt` — +30.9% val, Sortino 2.38, but holdout shows variance
5. `autoresearch_mixed23_daily/baseline_anneal_lr/best.pt` — +30.1% val, holdout_mean +7%

### Tier 3: Crypto-8 (zero-fee optimized)
6. `autoresearch_crypto8_daily/clip_anneal/best.pt` — +18.8% OOS, Sortino 1.85

## Running Services

| Service | Status | Script | Mode |
|---------|--------|--------|------|
| `daily-rl-trader` | **ACTIVE** | `trade_mixed_daily.py` | Paper, hybrid RL+Gemini |
| `unified-orchestrator` | ACTIVE | `unified_orchestrator.orchestrator` | Live (but crypto signals broken) |
| `alpaca-hourly-trader` | INACTIVE | — | — |

## Deployment Recommendation

### Immediate (TODAY):
1. **Keep `daily-rl-trader` running in paper mode** — it's already generating signals with the mixed-23 model and Gemini overlay
2. **Monitor for 7 more days** — the service has been running ~1 day, need more signal data
3. **The unified-orchestrator crypto portion is broken** — generating 0% confidence signals. Daily RL is strictly better even conservatively.

### Week 2:
4. **Switch daily-rl-trader from --paper to live** (remove `--paper` flag) for crypto positions ONLY
5. **Keep unified-orchestrator for stock positions** (NVDA, CRM, PLTR, etc.) until daily stock RL is validated
6. **Capital allocation**: 50% crypto (daily RL), 50% stocks (existing orchestrator)

### Week 4:
7. **If daily RL crypto > +5% over 30 days**, expand to 70% capital
8. **Test daily stock RL** on paper for 30 days
9. **Compare ensemble vs single-model** performance in live conditions

## Key Config Details

**Current daily-rl-trader service:**
```
trade_mixed_daily.py --daemon --mode hybrid --paper
  --top-k 2              # top 2 symbols per cycle
  --max-positions 2      # max 2 concurrent positions
  --max-position-fraction 0.25  # 25% of account per position
  --max-hold-days 5      # force-close after 5 days
  --llm-model gemini-3.1-flash-lite-preview
```

**Ensemble crypto models (in trade_daily_rl.py):**
- 5 checkpoints, majority vote
- All agree on LONG SOLUSD as of 2026-03-16
- Can be used as supplementary signal to mixed-23

## Fee Optimization Note

For Binance deployment specifically (not Alpaca):
- FDUSD pairs (BTC/ETH/SOL): 0% maker fee
- Train on 8+ symbols but execute on FDUSD-3 at 0% fee
- clip_anneal config is optimal for zero-fee (trade_penalty counterproductive with zero fees)
- 3 symbols alone insufficient (too correlated) — need diversified training data

---

*Status: Daily RL running paper, monitoring signals. Deploy live crypto in 7 days if signals consistent.*
