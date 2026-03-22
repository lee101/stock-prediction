# H100 Experiment Plan (Updated 2026-03-22)

## Executive Summary — REVISED

**The original plan targeting stocks20 was based on contaminated data.**

Original 73-config RTX 5090 sweep showed stocks20 winning (2 configs with positive holdout).
After fixing stock split adjustments (NFLX 10:1 Nov 2025, NVDA 10:1, GOOG 20:1, TSLA 3:1,
AMZN 20:1, AVGO 10:1), the "stocks20 positive holdout" result was entirely an artifact of
policies shorting the fake NFLX -93% price drop.

**New conclusion with clean data:**
- stocks20 at 90s/trial → ALL configs negative holdout (no NFLX artifact to exploit)
- stocks12 at 90s/trial → h100_slip_10bps: 90% windows profitable (neg=10%), median=+5.6%
- stocks12 random_mut_2272 (300s training) → ALL 20 windows profitable, median=+10.5%, p10=+5.2%

**H100 strategy: Use stocks12 with longer budget (200s → ~14M steps on H100)**

---

## Data Quality Fixes Applied (2026-03-22)

All binaries have been re-exported with split-adjusted CSVs.

### Stock splits fixed (backups at `.pre_split_backup`)

| Symbol | Split  | Date       | Pre-split rows (daily) |
|--------|--------|------------|------------------------|
| NFLX   | 10:1   | 2025-11-15 | (previous session)     |
| NVDA   | 10:1   | 2024-06-10 | 587                    |
| GOOG   | 20:1   | 2022-07-18 | 110                    |
| TSLA   |  3:1   | 2022-08-25 | 138                    |
| AMZN   | 20:1   | 2022-06-06 | 82 daily + 3306 hourly |
| AVGO   | 10:1   | 2024-07-15 | 667 daily + 6969 hourly|

Remaining "big drops" are verified real market events:
- NFLX: -35% on 2022-04-20 (Q1 2022 subscriber loss earnings)
- META: -26% on 2022-02-03 (Q4 2021 earnings miss, stocks15 only)
- INTC: -26% on 2024-08-01 (Q2 2024 earnings disaster, stocks15 only)

### Re-exported binaries (all clean, verified)

| File                             | Symbols | Train days | Val days |
|----------------------------------|---------|-----------|---------|
| stocks12_daily_{train,val}.bin   | 12      | 1302      | 158     |
| stocks15_daily_{train,val}.bin   | 15      | 1302      | 113     |
| stocks20_daily_train.bin         | 20      | 1302      | —       |
| stocks20_daily_val.bin           | 20      | —         | 158     |
| stocks20_cross_daily_{train,val} | 20      | 1302      | 158     |

---

## Clean Data Sweep Results (2026-03-22, RTX 5090, 90s/trial)

### stocks12 (12 symbols)

| Config              | holdout | neg_rate | median% | p10%   |
|---------------------|---------|----------|---------|--------|
| h100_slip_10bps     | -22.84  | **10%**  | +5.64   | +0.92  |
| h100_ent_05         | -35.80  | **10%**  | +9.62   | +0.71  |
| h100_mut2272_slip5  | -44.56  | 20%      | +4.35   | -4.15  |
| h100_mut2272_wd01   | -78.54  | 50%      | +0.69   | -11.78 |
| h100_mut2272_style  | -102.73 | 75%      | -4.48   | -14.17 |

### stocks20 (20 symbols) — all FAILED with clean data

| Config              | holdout  | neg_rate | median% |
|---------------------|----------|----------|---------|
| h100_ent_05         | -77.11   | 50%      | +0.99   |
| h100_mut2272_style  | -87.69   | 55%      | -2.18   |
| h100_slip_5bps      | -134.29  | 85%      | -14.47  |
| h100_mut2272_slip5  | -138.80  | 65%      | -8.61   |
| h100_slip_10bps     | -139.15  | 75%      | -9.25   |

### Current SOTA baseline (random_mut_2272, 300s training, stocks12)

| Metric                | Value     |
|-----------------------|-----------|
| holdout neg_rate      | **0%** (all 20 windows profitable) |
| holdout median        | +10.5%    |
| holdout p10           | +5.2%     |
| holdout median Sortino| 1.55      |

---

## H100 Configuration — UPDATED

**Dataset**: stocks12 (12 symbols, 1302 train days, 158 val days)
**Binary files**: `pufferlib_market/data/stocks12_daily_{train,val}.bin`
**Symbols**: AAPL, MSFT, NVDA, GOOG, META, TSLA, SPY, QQQ, PLTR, JPM, V, AMZN

### Recommended H100 command

```bash
source /nvme0n1-disk/code/stock-prediction/.venv313/bin/activate
cd /nvme0n1-disk/code/stock-prediction

python -m pufferlib_market.autoresearch_rl \
    --h100-mode \
    --train-data pufferlib_market/data/stocks12_daily_train.bin \
    --val-data pufferlib_market/data/stocks12_daily_val.bin \
    --time-budget 200 \
    --max-trials 150 \
    --leaderboard autoresearch_stock_h100_leaderboard.csv \
    --checkpoint-root pufferlib_market/checkpoints/autoresearch_stock_h100
```

Notes:
- `--h100-mode` uses H100_STOCK_EXPERIMENTS pool, sets periods_per_year=252, fee_rate=0.001, holdout_eval_steps=90
- Explicitly pass `--train-data` and `--val-data` to override the default stocks20 data
- 200s budget × 150 trials = ~8.3 hours total
- H100 expected ~3.5x faster than RTX 5090 (~14M steps vs ~4M at same wall time)
- stocks12 obs_size=209 (smaller than stocks20 obs_size=345) → faster steps/sec → even more steps per trial

---

## Key Findings

- **stocks20 results were invalid**: Original 73-config sweep showed stocks20 winning because
  NFLX had an unadjusted 10:1 forward split creating a fake -93% single-day drop in val data.
  Policies that shorted NFLX generated enormous artificial returns. After split-adjustment, all
  stocks20 configs fail in holdout at 90s budget.

- **stocks12 with slip_10bps is the best clean-data config at 90s**:
  h100_slip_10bps gets 90% windows profitable (only 2/20 lose), median=+5.64%.
  At H100's ~14M steps, expect this to approach 100% (like random_mut_2272 with 300s training).

- **random_mut_2272 still works with clean data**: Evaluated on fixed stocks12 val
  (158 days, no split artifacts) → all 20 windows profitable, median=+10.5%, p10=+5.2%.

- **h2048 configs in H100 pool are stocks12-safe**: The h2048 variants require ~2.5x more memory
  but stocks12's smaller obs space makes them feasible even at H100 speeds.

---

## Expected Outcomes (REVISED)

| Metric               | Current SOTA (random_mut_2272) | Expected H100 Target |
|----------------------|-------------------------------|----------------------|
| holdout neg_rate     | 0%                            | 0%                   |
| holdout median       | +10.5%                        | > +15%               |
| holdout p10          | +5.2%                         | > +8%                |
| holdout Sortino p10  | 1.52                          | > 2.0                |

---

## H100 Experiment Pool Summary

`H100_STOCK_EXPERIMENTS` contains 127 configs (used with `--h100-mode`):
- 27 structured configs (slip_10bps, ent_05, mut2272-style variants, h2048, etc.)
- 100 random mutations for exploration

With stocks12 data, the structured configs that showed promise at 90s local:
1. `h100_slip_10bps` — 90% windows profitable at 90s. Best clean-data config.
2. `h100_ent_05` — 90% windows profitable at 90s, median=+9.6%.

---

## How to Pull Results

```bash
# View leaderboard sorted by holdout_robust_score
python -c "
import csv
rows = list(csv.DictReader(open('autoresearch_stock_h100_leaderboard.csv')))
rows.sort(key=lambda r: float(r.get('holdout_robust_score') or '-inf'), reverse=True)
for r in rows[:10]:
    print(r['description'], r.get('holdout_robust_score'), r.get('holdout_negative_return_rate'))
"

# Or check the best checkpoint
ls -la pufferlib_market/checkpoints/autoresearch_stock_h100/
```

---

## Deployment Condition

If the best H100 checkpoint beats the current Alpaca deployment (random_mut_2272):

```bash
# Find best checkpoint (replace BEST_CONFIG with description from leaderboard)
BEST_CONFIG="h100_slip_10bps"  # or whatever wins

cp pufferlib_market/checkpoints/autoresearch_stock_h100/${BEST_CONFIG}/best.pt \
   pufferlib_market/checkpoints/stocks_deployment_candidate.pt

echo "Deployed ${BEST_CONFIG} as new stocks_deployment_candidate.pt"
```

Deployment conditions:
- holdout_negative_return_rate = 0% (ALL 20 windows profitable)
- holdout_p10 > current (+5.2%)
- holdout_median_sortino > current (1.55)
- No single window exceeds -15% return

---

## Data Files

```
pufferlib_market/data/stocks12_daily_train.bin  (12 symbols, 1302 days: 2022-02-07 to 2025-08-31)
pufferlib_market/data/stocks12_daily_val.bin    (12 symbols,  158 days: 2025-09-01 to 2026-02-05)
```

Symbols: AAPL, MSFT, NVDA, GOOG, META, TSLA, SPY, QQQ, PLTR, JPM, V, AMZN

Data sources: trainingdata/train/{SYM}.csv (split-adjusted, backup at .pre_split_backup)
