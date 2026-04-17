# Ensemble confidence-threshold ablation — DEAD END

## Question
If we require the v7 ensemble's post-mask softmax top-probability to be
above a threshold before taking the argmax action (otherwise force
flat), can we improve sortino without too much median cost?

## Setup
- `scripts/screened32_confidence_gate.py`
- val: `pufferlib_market/data/screened32_single_offset_val_full.bin` (263 windows × 50d)
- fb=5, lev=1.0, lag=2, disable_shorts=True (prod config)
- top-prob computed on softmax of `log(avg_probs) + short_mask`, so
  shorts are excluded from denominator — honest "confidence over
  permissible actions"

## Result — strictly monotone worse

| threshold | med%/mo | p10%/mo | sortino | neg/263 | max_dd% |
|---:|---:|---:|---:|---:|---:|
| **0.00** (baseline) | **7.47** | **3.18** | **6.74** | **10** | **5.71** |
| 0.02 | 7.47 | 3.18 | 6.74 | 10 | 5.71 |
| 0.04 | 7.47 | 3.18 | 6.74 | 10 | 5.71 |
| 0.05 | 6.77 | 2.40 | 6.32 | 12 | 5.11 |
| 0.06 | 3.33 | −0.62 | 3.85 | 42 | 5.98 |
| 0.08 | −0.09 | −3.35 | −0.33 | 148 | 2.71 |
| 0.10 | 0.00 | −0.50 | 0.00 | 110 | 0.26 |

## Interpretation
- **Threshold ≤ 0.04 has zero effect.** The ensemble's post-mask top
  action is always confident ≥ 0.04 when it picks non-flat. We can't
  even *observe* low-confidence trades below 0.04 — they don't exist.
- **0.05 to 0.06 is where the cliff is.** Forcing 0.05+ confidence
  replaces a handful of marginal-but-correct picks with flat → strict
  loss on every metric.
- **0.06+ forces a majority flat**, and remaining non-flat picks are
  wrong more often than right (neg spikes to 42, then 148).

## Why this fails
softmax_avg already integrates the safety-net effect: members that
predict flat add their flat mass to the averaged distribution, pulling
the argmax toward flat on genuinely uncertain days. A hard top-prob
floor over the masked softmax is **redundant at best**, and when it
triggers it removes the *most ambiguous* non-flat picks — which, on
this val set, turn out to be net profitable (the ambiguity is real
but the ensemble's direction is right).

## Conclusion
Don't wire a confidence floor into prod. The softmax_avg mode is
already doing the right thing; adding a hard threshold only makes it
worse. Leaves the three remaining architectural paths to the 27%/mo
target:

1. Learned MoE gate (not a hard threshold — conditional routing)
2. Test-time compute (Chronos2 MC as a second-opinion vote)
3. Symbol expansion / shorts retraining in-distribution
