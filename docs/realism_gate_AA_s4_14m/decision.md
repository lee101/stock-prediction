# A* candidate batch (AA/AB/AC/AD all seeds) realism-gate verdicts

All recent screened32 A* sweep candidates tested as 14th ensemble
member at the deploy-gate cell (fb=5, lev=1, lag=2, 263 windows,
screened32_single_offset_val_full.bin).

## Results

Baseline 13-model: **med +6.89%/mo, p10 +2.34%, sortino 6.10, neg 11/263**

| candidate | med_monthly | p10 | sortino | neg | Δ med | Δ neg | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| AA_s4 | +6.64% | +2.46% | 5.58 | 12 | −0.25% | +1 | ❌ reject |
| AB_s4 | +6.70% | +2.24% | 5.62 | 12 | −0.19% | +1 | ❌ reject |
| AC_s4 | +6.64% | +2.46% | 5.58 | 12 | −0.25% | +1 | ❌ reject (identical to AA) |
| AD_s2 | +6.69% | +1.90% | 5.94 | 16 | −0.20% | +5 | ❌ reject |
| AD_s3 | +6.40% | +1.44% | 5.22 | 16 | −0.49% | +5 | ❌ reject |
| AD_s4 | +6.98% | +2.02% | 6.39 | 10 | +0.09% | −1 | ❌ reject (1.5× knee fails) |
| AD_s5 | +6.05% | +1.00% | 4.85 | 19 | −0.84% | +8 | ❌ reject |
| AD_s6 | +6.01% | +1.32% | 5.29 | 17 | −0.88% | +6 | ❌ reject |
| AD_s7 | +5.47% | +0.45% | 5.23 | 22 | −1.42% | +11 | ❌ reject (worst, doubles negs) |

## Notes

- AA_s4 ≡ AC_s4: bit-for-bit identical results, consistent with prior
  finding that "AA=AC checkpoint-identical at seed=1 (converge
  pre-divergence)" (see `feedback_E1_sweep_results.md`).
- AB_s4 (the "softest overfit" variant per E1 sweep results) is the
  marginally-least-bad of AA/AB/AC but still loses on 3 of 4 metrics.
- AD_s4's 1× cell looks marginal but `docs/realism_gate_AD_s4_14m/decision.md`
  shows the 1.5× Pareto knee degrades (med −0.23%, sortino −0.16). Reject.
- AD seeds standalone leaderboard: s4 is the standout (med 8.11%, neg 12,
  sortino 16.80), all others weak (s5 med 0.83%, s6 med −1.46% actually
  loses money standalone). Standalone strength predicts ensemble lift —
  none of s2/s3/s5/s6/s7 should have been expected to improve.
- AD_s7 (best.pt, training_best_return=0.49) — despite training-time
  best_return being highest in the AD batch, val_best showed −4.66% med
  on 100 sampled windows and the 14m gate confirms: doubling negs is a
  catastrophic failure mode. Train-time return doesn't predict
  ensemble-membership utility.

## Conclusion

None of the AA/AB/AC/AD sweep candidates (9 tested: AA/AB/AC s4 +
AD s2-s7) improve the deploy
gate. A* sweep is exhausted as a 14th-member candidate source.
Continue waiting for new candidates from D/I/U sweeps. Bar for swap-in:
- (a) median-monthly improvement at 1× cell ≥ +0.5%
- (b) neg-count ≤ 11 (not worse)
- (c) 1.5× cell stays at Pareto knee or better (med ≥ +10.19%, sortino ≥ 6.20)
