# XGB Daily K-Fold Validation - 2026-06-19

Purpose: convert the daily XGB validation path to actual contiguous k-fold CV
and record the first smoke results while comparing classifier and ranker-style
XGBoost variants.

## Setup

- Data root: `/vfast/data/stock/trainingdata`
- Symbols: `symbol_lists/stocks_fresh_213_top64_20260409.txt`
- CV mode: `k_contiguous` leave-one-contiguous-fold-out
- Costs: `fee_rate=0.001` (10 bps per side), `fill_buffer_bps=5`
- Device: CPU
- Chronos cache: `analysis/top2_backtest/forecast_cache`
- Chronos coverage: `0` rows in all runs because that JSON cache is absent
  locally. These results are technical-feature-only.

## Results

| Artifact | Model | CV Window | Top N | Hold Through | Fold Median Monthly Mean | Worst Fold Median | P10 Monthly Mean | Worst Neg-Window Frac | Verdict |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `kfold_20260619_142335.json` | XGB classifier | 2025-01-02 to 2026-04-19 | 1 | no | -6.91% | -9.47% | -14.56% | 1.00 | fail |
| `kfold_20260619_142339.json` | XGB ranker + ranks | 2025-01-02 to 2026-04-19 | 1 | no | -13.48% | -14.65% | -17.28% | 1.00 | fail |
| `kfold_20260619_142528.json` | XGB classifier | 2024-01-02 to 2025-11-13 | 3 | no | -8.11% | -10.21% | -10.31% | 1.00 | fail |
| `kfold_20260619_142527.json` | XGB classifier | 2024-01-02 to 2025-11-13 | 5 | no | -7.36% | -8.52% | -10.49% | 1.00 | fail |
| `kfold_20260619_142611.json` | XGB classifier | 2024-01-02 to 2025-11-13 | 5 | yes | -5.58% | -7.32% | -9.46% | 1.00 | best of this batch, still fail |

## Interpretation

The validator is now a real k-fold harness, but this local smoke batch is not a
deployable result and does not beat the `prediction2` stock frontier. The best
row here is still negative after realistic fees and fill buffer.

The blocker for testing the intended "XGBoost over Chronos2 features" idea is
data availability, not code shape: no compatible daily JSON Chronos cache is
present in this checkout. The k-fold JSON now records `chronos_coverage` so
future runs cannot silently report a Chronos-feature result when the feature
columns are zero-filled.

## Next Run

Use the same harness with a real daily Chronos JSON cache:

```bash
python -m xgbnew.eval_kfold \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --data-root /vfast/data/stock/trainingdata \
  --chronos-cache <daily-json-chronos-cache> \
  --train-start 2021-01-01 \
  --cv-start 2024-01-02 --cv-end 2026-04-19 \
  --n-folds 5 --cv-mode k_contiguous --embargo-days 5 \
  --window-days 30 --stride-days 14 \
  --top-n 1 --hold-through --leverage 1.0 \
  --fee-rate 0.001 --fill-buffer-bps 5 \
  --n-estimators 400 --max-depth 5 --learning-rate 0.03 \
  --learn-chronos-features \
  --device cpu
```
