# H100 Experiment Plan

## Executive Summary

Local RTX 5090 scaling sweep (73 configs, 90s/trial) showed that stocks20 produces the highest
number of configs with positive holdout robustness (2 vs 1 for stocks12, 0 for stocks15). The
H100 will run the `--h100-mode` experiment pool on stocks20 with 200s/trial, targeting 150 trials,
focusing on the two winning config families: `slip_10bps` and `ent_05`.

## Local Sweep Results (RTX 5090)

### Data files tested

| Dataset  | Symbols | Train timesteps | Val timesteps | Common window              |
|----------|---------|-----------------|---------------|----------------------------|
| stocks12 | 12      | 1211            | 194           | 2022-02-07 – 2025-08-31    |
| stocks15 | 15      | 1277            | 178           | 2022-02-07 – 2025-08-31    |
| stocks20 | 20      | 1302            | 158           | 2022-02-07 – 2026-02-05    |

### Top results by holdout_robust_score (73 configs, 90s/trial, ~4.3M steps)

| Config                    | Symbols | val_return | holdout_robust | neg_rate | h_sortino |
|---------------------------|---------|------------|----------------|----------|-----------|
| stocks12 slip_10bps       | 12      | +1.37      | **+21.04**     | **0%**   | 11.99     |
| stocks20 ent_05           | 20      | -0.14      | **+19.19**     | **0%**   | 5.06      |
| stocks20 trade_pen_10     | 20      | +0.03      | **+2.23**      | **0%**   | 3.09      |
| stocks15 trade_pen_08     | 15      | +0.59      | -4.03          | 10%      | 2.62      |
| stocks12 reward_scale_5   | 12      | +0.25      | -6.66          | 10%      | 2.44      |
| stocks12 trade_pen_03     | 12      | +0.25      | -14.43         | 15%      | 1.93      |
| stocks12 ent_08           | 12      | +0.30      | -15.07         | 10%      | 1.68      |

### Symbol count summary

| Dataset  | Configs tested | Best holdout | Configs > 0 |
|----------|----------------|--------------|-------------|
| stocks12 | 23             | +21.04       | 1 (4.3%)    |
| stocks15 | 25             | -4.03        | 0 (0%)      |
| stocks20 | 25             | +19.19       | 2 (8.0%)    |

## Key Findings

- **stocks20 wins for generalization**: stocks20 is the only dataset with multiple configs
  producing positive holdout robustness. More symbols give the policy more diverse trading
  opportunities and reduce regime concentration.

- **stocks15 is a dead zone**: despite sitting between stocks12 and stocks20 in size, stocks15
  produced zero configs with positive holdout across 25 configurations. This suggests the specific
  symbol mix (15 symbols dominated by lower-cap growth names) is unfavorable.

- **Two winning config families on stocks20**:
  1. `ent_05` (default entropy coefficient): positive holdout with 0% negative windows and
     Sortino 5.06. Lower entropy forces more decisive trades on 20 symbols simultaneously.
  2. `trade_pen_10` (heavy trade penalty): also positive holdout with 0% negative windows.
     Reduces excessive churn on daily bars.

- **slip_10bps dominates on stocks12**: with val_return=1.37 and holdout=+21.04, this is the
  single best config overall. Training with 10bps slippage forces the policy to find only
  wide-edge opportunities. Worth testing on stocks20 with longer H100 training.

- **h2048 not tested on RTX 5090** (too slow for 90s budget). H100 ~3-4x faster so h2048 will
  fit in 200s comfortably (~14M steps vs ~4M on RTX 5090).

- **More steps helps**: the 90s runs produced `random_mut_2272` which got val_return=0.9195 with
  300s training. H100 at 200s/trial and ~3.5x speedup = ~14M steps, expected to improve.

## H100 Configuration

**Dataset**: stocks20 (20 symbols, 1302 train days, 158 val days)
**Binary files**: `pufferlib_market/data/stocks20_daily_{train,val}.bin`

### Recommended H100 command

```bash
source /nvme0n1-disk/code/stock-prediction/.venv313/bin/activate
cd /nvme0n1-disk/code/stock-prediction

python -m pufferlib_market.autoresearch_rl \
    --h100-mode \
    --time-budget 200 \
    --max-trials 150 \
    --leaderboard autoresearch_stock_h100_leaderboard.csv \
    --checkpoint-root pufferlib_market/checkpoints/autoresearch_stock_h100
```

Notes:
- `--h100-mode` automatically selects `H100_STOCK_EXPERIMENTS`, uses stocks20 data,
  sets periods_per_year=252, fee_rate=0.001, holdout_eval_steps=90
- 200s budget × 150 trials = ~8.3 hours total
- H100 expected ~3.5x faster than RTX 5090 (~14M steps vs ~4M at same wall time)

### Alternative: stocks12 with slip_10bps focus

```bash
python -m pufferlib_market.autoresearch_rl \
    --stocks \
    --train-data pufferlib_market/data/stocks12_daily_train.bin \
    --val-data pufferlib_market/data/stocks12_daily_val.bin \
    --descriptions stock_slip_10bps,stock_slip_5bps,stock_slip_15bps \
    --time-budget 300 \
    --max-trials 15 \
    --leaderboard autoresearch_stock_h100_slip_leaderboard.csv \
    --checkpoint-root pufferlib_market/checkpoints/autoresearch_stock_h100_slip
```

## Expected Outcomes

| Metric               | Current SOTA (random_mut_2272) | Expected H100 Target |
|----------------------|-------------------------------|----------------------|
| val_return           | 0.9195                        | > 1.2 (≥30% better)  |
| holdout_robust_score | (not available)               | > 10.0               |
| holdout_neg_rate     | (not available)               | < 5%                 |
| Sortino (holdout)    | 3.80                          | > 4.5                |
| All 20 windows prof  | yes                           | yes                  |

## H100 Experiment Pool Summary

The `--h100-mode` pool (`H100_STOCK_EXPERIMENTS`) contains 127 configs:
- 27 structured configs targeting the local sweep winners
- 100 random mutations for exploration

Structured configs include:
- slip_10bps / slip_5bps / slip_15bps (test slippage training on stocks20)
- ent_05 / ent_03 / ent_08 (entropy coefficient sweep)
- trade_pen_10 and cross-combinations with slippage
- h2048 variants (only viable on H100, marked `requires_gpu="h100"`)
- Cosine LR, obs_norm, weight_decay crosses with best local configs
- Multi-seed repeats of top configs

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
- holdout_robust_score > current SOTA
- holdout_negative_return_rate < 10% (< 2 losing windows out of 20)
- holdout_median_sortino > 3.5
- No single window exceeds -15% return

## Data Files

```
pufferlib_market/data/stocks20_daily_train.bin  (20 symbols, 1302 days: 2022-02-07 to 2025-08-31)
pufferlib_market/data/stocks20_daily_val.bin    (20 symbols,  158 days: 2025-09-01 to 2026-02-05)
```

Symbols: AAPL, MSFT, NVDA, GOOG, META, TSLA, AMZN, AMD, JPM, SPY, QQQ, PLTR, NET, NFLX,
         ADBE, CRM, AVGO, V, COST, ADSK

Data sources: trainingdata/train/{SYM}.csv (primary) + trainingdatahourly/stocks/{SYM}.csv
(hourly bars resampled to daily, used to extend timeline).
