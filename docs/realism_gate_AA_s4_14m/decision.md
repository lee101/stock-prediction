# A* candidate batch (AA/AB/AC/AD s4) realism-gate verdicts

All four most-recent screened32 sweep candidates tested as 14th
ensemble member at the deploy-gate cell (fb=5, lev=1, lag=2, 263
windows, screened32_single_offset_val_full.bin).

## Results

Baseline 13-model: **med +6.89%/mo, p10 +2.34%, sortino 6.10, neg 11/263**

| candidate | med_monthly | p10 | sortino | neg | Δ med | Δ neg | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| AA_s4 | +6.64% | +2.46% | 5.58 | 12 | −0.25% | +1 | ❌ reject |
| AB_s4 | +6.70% | +2.24% | 5.62 | 12 | −0.19% | +1 | ❌ reject |
| AC_s4 | +6.64% | +2.46% | 5.58 | 12 | −0.25% | +1 | ❌ reject (identical to AA) |
| AD_s4 | +6.98% | +2.02% | 6.39 | 10 | +0.09% | −1 | weak (1.5× knee fails) |

## Notes

- AA_s4 ≡ AC_s4: bit-for-bit identical results, consistent with prior
  finding that "AA=AC checkpoint-identical at seed=1 (converge
  pre-divergence)" (see `feedback_E1_sweep_results.md`).
- AB_s4 (the "softest overfit" variant per E1 sweep results) is the
  marginally-least-bad of AA/AB/AC but still loses on 3 of 4 metrics.
- AD_s4's 1× cell looks marginal but `docs/realism_gate_AD_s4_14m/decision.md`
  shows the 1.5× Pareto knee degrades (med −0.23%, sortino −0.16). Reject.

## Conclusion

None of the recent A* sweep candidates improve the deploy gate.
Continue waiting for new candidates from D/I/U sweeps. Bar for swap-in:
- (a) median-monthly improvement at 1× cell ≥ +0.5%
- (b) neg-count ≤ 11 (not worse)
- (c) 1.5× cell stays at Pareto knee or better (med ≥ +10.19%, sortino ≥ 6.20)
