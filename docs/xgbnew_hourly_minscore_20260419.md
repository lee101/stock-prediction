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

## 5-seed ensemble × ms sweep (2026-04-19 cycle-3 update)

Added `--n-seeds N` flag to `eval_hourly_multiwindow.py` (mean-blend
predict_scores across N consecutively-seeded XGB models). Goal: test
whether ensemble averaging closes the 19/73 neg tail at the hourly
gate, as it does for daily (2/30 → 0/30 under gate).

### Score-shrinkage finding (the key result)

Ensemble averaging **shrinks** the score distribution. Individual vs
3-seed ensemble on the same test windows:

| metric | seed 42 | seed 43 | seed 44 | 3-seed ensemble | shrinkage |
| --- | ---: | ---: | ---: | ---: | ---: |
| max | 0.6532 | 0.6333 | 0.6372 | **0.6392** | −2.1% vs best |
| p99 | 0.5726 | ~0.57 | ~0.57 | **0.5567** | tighter |
| frac ≥ 0.58 | 0.28% | — | — | **0.18%** | fewer survivors |

For daily, individual model max reaches 0.94 so the deploy gate at
0.75 still sits well below the cap — no meaningful shrinkage effect.
For hourly, the single-seed cap is already 0.65 and ensemble drives
it to 0.64, which pushes the usable knee from ms=0.58 down to
ms=0.57 where tail risk is worse-controlled.

### 5-seed (42..46) ms sweep at top_n=3, lev=1.0 (73 windows)

| cell | med %/mo | p10 %/mo | worst window | worst DD | neg | trades/win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ms=0.55 | −1.34 | −29.01 | −38.88 | 16.85 | 39/73 | 63.0 |
| **ms=0.57 (new knee)** | **+39.23** | **−28.28** | −46.40 | 7.17 | 23/73 | 15.5 |
| ms=0.58 | +16.72 | −57.10 | −85.97 | 5.13 | 25/73 | 6.1 |
| ms=0.59 | +10.98 | −36.86 | −88.34 | 3.35 | 18/73 | 2.4 |
| ms=0.60 | +0.00 | −34.88 | −77.73 | 1.09 | 18/73 | 1.3 |

### 1-seed vs 5-seed best cell comparison

| axis | 1-seed best (ms=0.58) | 5-seed best (ms=0.57) | Δ |
| --- | ---: | ---: | ---: |
| med %/mo | +44.78 | **+39.23** | −5.55 |
| p10 %/mo | −22.43 | **−28.28** | −5.85 |
| worst window | −52.98 | **−46.40** | +6.58 |
| worst DD | 12.75 | **7.17** | −5.58 |
| neg | 19/73 | **23/73** | +4 |

**5-seed ensemble LOSES on median, p10, and neg-count. Marginal wins
on worst-window and DD are not a net improvement.** The ensemble
shifts the knee left (wider score distribution post-averaging means
the gate needs to move to compensate), but at the new knee the tail
is not better protected.

### Verdict: hourly ensemble fails to close the tail

Ensemble averaging does NOT generalize as a rescue lever from daily
to hourly:

* **Daily**: cap=0.94, gate at 0.75, ensemble has headroom → 5-seed
  closes 2 neg windows without shifting knee.
* **Hourly**: cap=0.65, gate at 0.58, ensemble shrinks cap to 0.64 →
  knee shifts left to 0.57 where tail is inherently worse-filtered.

The root cause — hourly signal's narrow score distribution — is not
helped by averaging. Next candidates: non-averaging blend (max,
median, top-k of seed scores), or per-window regime filter (SPY MA50
already shown to help daily).

## Next axes (for future cycles, not now)

* **Non-averaging blend**: hourly max-over-seeds or median-over-seeds
  may preserve conviction tail instead of compressing it.
* **SPY MA50 regime filter**: already deployed in daily prod; does
  it layer on top of hourly-gate to exit bad windows?
* **Different hourly objective**: pairwise rank loss or focal loss
  during training, to spread predict_proba further.

## Files
* `analysis/xgbnew_hourly_ms/hourly_multiwindow_stocks_20260419_192611*.json` — top_n=1 coarse sweep (ms=0.55/0.65/0.72/0.75 all degenerate)
* `analysis/xgbnew_hourly_ms_n3/hourly_multiwindow_stocks_20260419_192926*.json` — top_n=3 fine sweep (1-seed, knee ms=0.58)
* `analysis/xgbnew_hourly_ms_5s/hourly_multiwindow_stocks_20260419_193947_ms*.json` — 5-seed ensemble sweep (knee ms=0.57, worse on tail)
* Code: `xgbnew/eval_hourly_multiwindow.py` (+`--min-score-sweep`, +`--n-seeds`)
