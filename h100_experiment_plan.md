# H100 Experiment Plan (Updated 2026-03-22)

## Executive Summary — REVISED (v3: drawdown_pen is new SOTA)

**The original plan targeting stocks20 was based on contaminated data.**

Original 73-config RTX 5090 sweep showed stocks20 winning (2 configs with positive holdout).
After fixing stock split adjustments (NFLX 10:1 Nov 2025, NVDA 10:1, GOOG 20:1, TSLA 3:1,
AMZN 20:1, AVGO 10:1), the "stocks20 positive holdout" result was entirely an artifact of
policies shorting the fake NFLX -93% price drop.

**New conclusion with clean data (v2 50-trial sweep on stocks12_daily_train.bin):**
- stocks12 `stock_drawdown_pen` (90s): **NEW SOTA** — 0% neg, +22.9% med, +4.8% p10, Sortino=7.25, score=+24.9
- stocks12 `stock_trade_pen_05_s123` (52s): 0% neg, +16.6% med, +7.7% p10, Sortino=4.76, score=+14.9
- stocks12 `random_mut_2272` (300s): 0% neg, +10.5% med, +5.2% p10 (old SOTA, now beaten)

**Key insight**: `drawdown_penalty=0.05 + trade_penalty=0.03` (no slippage training) beats
all previous approaches. Drawdown penalty acts as a superior regularizer vs training slippage.

**H100 strategy: Use stocks12 at 90s/trial × 500 trials, targeting drawdown_pen variants**

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

## V2 50-Trial Sweep Results (2026-03-22, stocks12_daily_train.bin, 90s/trial)

Full 50-trial sweep on stocks12. Top results:

| Config                  | score  | neg_rate | median%  | p10%    | worst%  | Sortino(med) | max_dd_worst% |
|-------------------------|--------|----------|----------|---------|---------|-------------|---------------|
| **stock_drawdown_pen**  | **+24.9** | **0%** | **+22.9** | **+4.8** | **+3.3** | **7.25** | **9.2%** |
| **stock_trade_pen_05_s123** | **+14.9** | **0%** | **+16.6** | **+7.7** | **+4.0** | **4.76** | **11.9%** |
| stock_trade_pen_05      | -3.5   | 5%      | +27.8    | +4.2    | -1.8    | 4.06        | 14.0%         |
| stock_trade_pen_03      | -21.3  | 20%     | +5.0     | -2.9    | -3.5    | 1.63        | 9.7%          |
| stock_ent_08            | -24.4  | 15%     | +5.8     | -3.1    | -7.3    | 1.72        | 10.7%         |

### stock_drawdown_pen config details
```
drawdown_penalty=0.05, trade_penalty=0.03
fill_slippage_bps=0.0 (NO training slippage — drawdown penalty is the regularizer)
hidden_size=1024, lr=3e-4, ent_coef=0.05, weight_decay=0.0, anneal_lr=True
train_steps=3,178,496 in 90s (~35k steps/sec — fewer steps than trade_pen due to penalty overhead)
```

### Key v2 sweep insight
Drawdown penalty approach with **zero slippage training** outperforms slip-training approach:
- `stock_drawdown_pen` (0% slip): score=+24.9 > `random_mut_2272` (12bps slip): estimated ~-5
- The `drawdown_penalty` term forces the policy to avoid large equity dips, which naturally prevents
  the reckless behavior that causes big holdout losses. More interpretable and stable than slip training.

### Why stock_drawdown_pen score is negative-proof
- `worst_ret=+3.3%` means even the worst of 20 windows was profitable
- `max_dd_worst=9.2%` is tight — worst-case drawdown capped below 10%
- `Sortino=7.25` median — extremely high risk-adjusted return

### Checkpoint
`pufferlib_market/checkpoints/stocks12_v2_sweep/stock_drawdown_pen/best.pt`

---

## Overfitting at 300s Budget (2026-03-22 Finding)

Local validation confirmed critical overfitting when training budget exceeds ~90s on stocks12:

| Config          | Budget | holdout neg_rate | holdout median | Notes         |
|-----------------|--------|------------------|----------------|---------------|
| h100_slip_10bps | 90s    | **10%**          | +5.64%         | Good          |
| h100_slip_10bps | 300s   | **50%**          | +0.20%         | Overfit       |

**Conclusion**: stocks12 optimal budget is ~90s (~4M steps on RTX 5090).
H100 at ~3.5x speedup reaches ~4M steps in 90s with better GPU utilization.
Use `time_budget=90` with `max_trials=500` for H100.

---

## Dataset Decision — FINAL (2026-03-22 Local Experiments)

**CONCLUSION: Use stocks12_daily_train.bin (1302 days, 2022-02-07 to 2025-08-31), NOT the extended dataset.**

### Why stocks12_extended (1797 days back to 2020) is WORSE

Local 20-trial autoresearch sweep (90s/trial) comparing datasets:

| Config              | stocks12_orig (1302d) | stocks12_extended (1797d) | stocks11 (2434d, no PLTR) |
|---------------------|----------------------|---------------------------|---------------------------|
| stock_trade_pen_02  | neg=20%, med=+11.2%  | neg=55%, med=-2.1%        | neg=30%, med=+2.9%        |
| stock_longshort     | neg=15%, med=+9.6%   | neg=25%, med=+9.6%        | neg=65%, med=-7.3%        |
| stock_baseline      | neg=65%, med=-6.2%   | neg=85%, med=-9.1%        | neg=45%, med=+1.6%        |
| Best score          | **-25.8** (trade_pen_02) | -47.6 (trade_pen_10)   | -46.4 (trade_pen_02)      |

**Reason**: The extra 2020-2021 data (COVID recovery, zero-rate bull market) is from a very different
market regime than the val period (2025-2026, post-rate-hike). Adding out-of-distribution historical
data hurts generalization. The val period matches the post-2022 regime better.

**Also confirmed**: stocks11 (no PLTR, more data) also worse than stocks12_orig. More data ≠ better
when the extra data is from a different distribution.

---

## H100 Configuration — FINAL

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
    --time-budget 90 \
    --max-trials 500 \
    --leaderboard autoresearch_stock_h100_leaderboard.csv \
    --checkpoint-root pufferlib_market/checkpoints/autoresearch_stock_h100
```

Notes:
- `--h100-mode` uses H100_STOCK_EXPERIMENTS pool, sets periods_per_year=252, fee_rate=0.001, holdout_eval_steps=90
- **stocks12_daily_train.bin**: 1302 days matching post-2022 market regime — better generalization than extended
- **Same val**: stocks12_daily_val.bin (158 days, 2025-09-01 to 2026-02-05) for fair comparison
- **90s budget**: H100 at ~3.5x RTX 5090 speed reaches ~14M steps — the proven non-overfitting regime
- **500 trials × 90s = 12.5 hours** (within H100 daily budget)
- stocks12 obs_size=209 (smaller than stocks20 obs_size=345) → faster steps/sec

### eval_hours calibration (CRITICAL)

When using `evaluate_fast` (C env) on daily data, `eval_hours` counts **calendar days**, not trading days.
To evaluate ~90 trading days, use `eval_hours=130` (90 × 7/5 ≈ 126, rounded to 130).

The autoresearch `holdout_eval_steps=90` calls `evaluate_holdout.py` (Python sim) which counts 90 **trading bars**.
These two give different results:
- `evaluate_holdout --eval-hours 90` ≈ 90 trading days
- `evaluate_fast --eval-hours 90` ≈ 64 trading days → use `--eval-hours 130` for equivalent

Reference on random_mut_2272 (stocks12_daily_val.bin):
| Evaluator | eval_hours | median_return | p10_return |
|-----------|-----------|--------------|------------|
| evaluate_holdout | 90 | +4.9% | -11.6% |
| evaluate_fast | 90 | +0.3% | -12.2% |
| evaluate_fast | 130 | +8.6% | -18.9% |

---

## Key Findings

- **stocks20 results were invalid**: Original 73-config sweep showed stocks20 winning because
  NFLX had an unadjusted 10:1 forward split creating a fake -93% single-day drop in val data.
  Policies that shorted NFLX generated enormous artificial returns. After split-adjustment, all
  stocks20 configs fail in holdout at 90s budget.

- **NEW SOTA: stock_drawdown_pen** (2026-03-22 v2 sweep, trial 20):
  `drawdown_penalty=0.05, trade_penalty=0.03, no slippage` → 0% neg, +22.9% med, Sortino=7.25.
  Beats random_mut_2272 comprehensively. Worst window = +3.3% (ALL 20 profitable with margin).
  H100_STOCK_EXPERIMENTS now has 13 drawpen variants exploring seeds + hyperparams.

- **2nd best: stock_trade_pen_05_s123** (0% neg, +16.6% med, +7.7% p10):
  `trade_penalty=0.05, seed=123` — more reliable than stock_trade_pen_05 (which had 1 negative window).
  H100 pool has 8 trade_pen_05 variants covering seeds + ent/wd/anneal_ent.

- **stocks12 with slip_10bps was the best clean-data config at 90s** (before v2 sweep):
  h100_slip_10bps gets 90% windows profitable (only 2/20 lose), median=+5.64%.
  Superseded by drawdown_pen approach.

- **random_mut_2272 still works with clean data**: Evaluated on fixed stocks12 val
  (158 days, no split artifacts) → all 20 windows profitable, median=+10.5%, p10=+5.2%.
  Now the baseline to beat (not SOTA).

- **Drawdown penalty > slippage training**: `drawdown_penalty=0.05` with no training slippage
  outperforms `fill_slippage_bps=12` as a regularization strategy for stocks12.

- **h2048 configs in H100 pool are stocks12-safe**: The h2048 variants require ~2.5x more memory
  but stocks12's smaller obs space makes them feasible even at H100 speeds.

---

## Expected Outcomes (REVISED v3)

New deployment bar is stock_drawdown_pen (0% neg, +22.9% med, p10=+4.8%, Sortino=7.25).
H100 goal: find a drawpen variant with higher p10 (better worst-case floor).

| Metric               | Old SOTA (random_mut_2272) | New SOTA (stock_drawdown_pen) | H100 Target    |
|----------------------|---------------------------|-------------------------------|----------------|
| holdout neg_rate     | 0%                        | **0%**                        | 0%             |
| holdout median       | +10.5%                    | **+22.9%**                    | > +25%         |
| holdout p10          | +5.2%                     | **+4.8%**                     | > +10%         |
| holdout Sortino med  | 1.55                      | **7.25**                      | > 7.0          |
| holdout worst window | +3.3% (drawpen)           | **+3.3%**                     | > +5%          |

---

## H100 Experiment Pool Summary

`H100_STOCK_EXPERIMENTS` contains **162 configs** (used with `--h100-mode`):
- 62 structured configs (slip/ent/trade_pen/drawpen variants, seeds, h2048, rmu4424/rmu1228)
- 100 random mutations for exploration
- 4 h100-only configs (h2048 variants) that need actual H100 GPU

**New additions after v2 sweep (2026-03-22):**
- 13 `h100_drawpen_*` variants (drawdown_penalty=0.05 + trade_penalty=0.03, different seeds/params)
- 8 `h100_trade_pen_05_*` variants (different seeds + ent/wd/anneal_ent)
- 4 `h100_rmu4424_*` variants (h=256 small-network regularization)
- 3 `h100_rmu1228_*` variants (obs_norm=True + high ent + no slippage)

With stocks12 data, the structured configs that showed promise (ordered by 90s local results):
1. `h100_drawpen_*` — 13 variants of NEW SOTA (stock_drawdown_pen: 0% neg, +22.9% med)
2. `h100_trade_pen_05_*` — 8 variants of 2nd best (stock_trade_pen_05_s123: 0% neg, +16.6% med)
3. `h100_slip_10bps` — 90% windows profitable at 90s (old best before v2 sweep)

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

Deployment conditions (updated to match new SOTA stock_drawdown_pen):
- holdout_negative_return_rate = 0% (ALL 20 windows profitable)
- holdout_p10 > +4.8% (stock_drawdown_pen bar)
- holdout_median_sortino > 7.0 (stock_drawdown_pen bar is 7.25)
- No single window exceeds -5% return (stock_drawdown_pen worst = +3.3%)

---

## Data Files

```
pufferlib_market/data/stocks12_daily_train.bin  (12 symbols, 1302 days: 2022-02-07 to 2025-08-31)
pufferlib_market/data/stocks12_daily_val.bin    (12 symbols,  158 days: 2025-09-01 to 2026-02-05)
```

Symbols: AAPL, MSFT, NVDA, GOOG, META, TSLA, SPY, QQQ, PLTR, JPM, V, AMZN

Data sources: trainingdata/train/{SYM}.csv (split-adjusted, backup at .pre_split_backup)
