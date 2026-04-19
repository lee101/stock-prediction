# Hourly XGB × conviction gate (2026-04-19)

## Question
`project_xgb_minscore_conviction_filter.md` showed the `--min-score`
gate rescues the daily signal (2/30 → 0/30 neg, DD 31 → 2.48, med
+38.85 → +61.38%/mo). Does the same lever rescue the **hourly** XGB
path, which memory (`project_xgb_hourly_loses_to_daily.md`) flagged as
a loser (med ~10%, p10 −57%, 30/72 neg)?

## Method
Added `--min-score-sweep` flag to `xgbnew/eval_hourly_multiwindow.py`:
trains the XGB once on the full hourly stocks dataset, then loops the
window-evaluation step over multiple `min_score` values and writes one
JSON per value. Amortises the ~3-minute dataset+train cost across the
whole sweep.

Data: `trainingdatahourly/stocks/` (661 symbols, 320 MB), train
through 2025-09-30, val through 2025-12-31, test through 2026-04-15.
Config: top_n=3, lev=1.0, n_est=400, depth=5, lr=0.03.

## Hourly score distribution (the key calibration finding)

Daily XGB scores on picks range 0.75 → 0.94 (median 0.854).
**Hourly XGB scores top out at 0.65**:

| percentile | score |
| --- | --- |
| p50 | 0.4994 |
| p75 | 0.5086 |
| p90 | 0.5215 |
| p95 | 0.5343 |
| p99 | 0.5726 |
| p99.5 | 0.5828 |
| p99.9 | 0.6033 |
| max | 0.6532 |

Fraction of hourly scores ≥ 0.55: 2.88%. Fraction ≥ 0.60: 0.13%.
Fraction ≥ 0.70: 0.00%.

**This means the hourly predict_proba signal is fundamentally less
confident.** The gate knee must therefore live in [0.55, 0.60], not
near the daily sweet spot of 0.72-0.75.

## ms sweep at top_n=3, lev=1.0 (73 windows)

| cell | med %/mo | p10 %/mo | worst window | worst DD | neg | trades/window | active wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ms=0.00 (baseline) | −17.45 | −32.13 | −44.53 | 35.54 | 67/73 | 224 | 73/73 |
| ms=0.52 | −10.93 | −23.77 | −31.64 | 24.85 | 55/73 | 142 | 73/73 |
| ms=0.55 | −4.40 | −22.68 | −36.02 | 16.54 | 42/73 | 92 | 73/73 |
| ms=0.57 | +11.00 | −25.81 | −49.94 | 14.21 | 30/73 | 46 | 73/73 |
| **ms=0.58** | **+44.78** | **−22.43** | −52.98 | 12.75 | 19/73 | 24 | 73/73 |
| ms=0.60 | +25.25 | −22.87 | −53.79 | 2.09 | 21/73 | 4.4 | 61/73 |

**Knee: ms=0.58.** Median lifts dramatically (−17.45 → +44.78), neg
collapses (67/73 → 19/73), DD shrinks (35.5 → 12.75). But **p10
remains at −22% and the worst window is −53%**. The gate cannot fix
the fat left tail.

Above ms=0.58 (at ms=0.60), only 4.4 trades/window survive and 12/73
windows hold cash entirely — median falls back to +25 because cash
windows pull the distribution toward zero.

Baseline also degraded vs memory's +10% / 30 neg read: the prior
baseline JSONs from 2026-04-19 earlier showed med +24%/mo; the
extended data through 2026-04-15 now shows med −17%. The
post-April-crash windows are worse for the hourly path. Gate still
recovers a positive median.

## Verdict: hourly gate helps median but doesn't close tail

Compared to daily champion (ms=0.75 × N=3 × lev=1.25):

| axis | daily champion | hourly ms=0.58 N=3 |
| --- | ---: | ---: |
| med %/mo | +61.38 | +44.78 |
| p10 %/mo | **+43.24** | **−22.43** |
| worst window | +35.67 | **−52.98** |
| worst DD | 2.48 | 12.75 |
| neg | 0/30 | 19/73 |

Hourly loses on every metric except median is in-the-ballpark. The
gate on hourly is not a path to prod — **concentration (daily) beats
frequency (hourly)** even after the hourly signal is gated.

## Why hourly scores don't climb past 0.65

Hypotheses for the narrower score distribution:

1. **Signal weakness** — intra-day price action carries less
   information than daily close-to-close. Hourly label (next-bar up)
   is noisier than daily label (next-day up).
2. **Calibration issue** — XGB's default logistic loss over-shrinks
   toward 0.5 on noisy labels. Higher n_estimators / different
   objective (focal loss, pairwise rank) may broaden scores.
3. **Feature set mismatch** — `HOURLY_FEATURE_COLS` may not capture
   enough conditional information to drive predict_proba past ~0.65.

None of these are quick fixes. The daily path is the compute priority.

## Next axes (for future cycles, not now)

* **Hourly ensemble**: daily 5-seed collapses 2/30 neg → 0/30 under
  the gate. Does a 5-seed hourly ensemble (random seeds only) close
  the 19/73 neg at ms=0.58?
* **SPY MA50 regime filter**: already deployed in daily prod; does
  it layer on top of hourly-gate to exit bad windows?
* **Different hourly objective**: pairwise rank loss or focal loss
  during training, to spread predict_proba further.

## Files
* `analysis/xgbnew_hourly_ms/hourly_multiwindow_stocks_20260419_192611*.json` — top_n=1 coarse sweep (ms=0.55/0.65/0.72/0.75 all degenerate)
* `analysis/xgbnew_hourly_ms_n3/hourly_multiwindow_stocks_20260419_192926*.json` — top_n=3 fine sweep
* Code: `xgbnew/eval_hourly_multiwindow.py` (+`--min-score-sweep`)
