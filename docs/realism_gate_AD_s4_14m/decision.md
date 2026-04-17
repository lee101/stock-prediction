# AD_s4 as 14th ensemble member — DEPLOY DECISION: NO

## Numbers vs 13-model prod baseline

| metric | 13-model (prod) | 14-model (+AD_s4) | delta | verdict |
|---|---:|---:|---|---|
| **fb=5, lev=1.0 (deploy gate)** | | | | |
| med_monthly | +6.89% | +6.98% | +0.09% | flat |
| p10_monthly | +2.34% | +2.02% | −0.32% | ⚠️ tail worse |
| sortino | 6.10 | 6.39 | +0.29 | better |
| max_dd | 6.28% | 6.40% | +0.12% | flat |
| n_neg | 11/263 | 10/263 | −1 | very slightly better |
| **fb=5, lev=1.5 (Pareto knee)** | | | | |
| med_monthly | +10.19% | +9.96% | −0.23% | ⚠️ worse |
| p10_monthly | +3.24% (approx) | +2.36% | ⚠️ tail worse | |
| sortino | 6.20 | 6.04 | −0.16 | worse |
| n_neg | 13/263 | 12/263 | −1 | very slightly better |

## Deploy criteria (from `project_realism_gate_overrides_headline.md`)

Required for swap-in:
- (a) median-monthly improvement at the 1x cell — **marginally PASS** (+0.09%, essentially noise)
- (b) neg-count not worse — **PASS** (10 vs 11)
- (c) 1.5x cell stays at the Pareto knee or better — **FAIL** (med −0.23%, sortino −0.16, p10 down)

Two of three are pass-or-noise; (c) fails. The Pareto knee is what we'd actually deploy if we bumped `--allocation-pct` toward 100, so a knee degradation is more disqualifying than a 1x-cell tie is qualifying.

## Why the multihorizon eval recommended `promising_additive`

`scripts/eval_multihorizon_candidate.py --candidate-checkpoint .../AD/s4/best.pt`
returned `promising_additive` with `mean_delta_monthly=+0.08%, wins=13 losses=5`.
That eval uses `--recent-within-days 140` which restricts to the most recent
~140 of 313 val days, dropping the Mar-Apr 2026 tariff crash where every
neg window in the realism gate is concentrated. AD_s4 may help in calm
regimes (the recent slice it was scored on) but it doesn't help in the
crash regime that drives the deploy gate.

This is the same trap that produced the misleading 19.57%/mo headline.

## Decision

**Keep prod at 13 models.** Do NOT swap AD_s4 in. Continue waiting for a
candidate that improves the 1x cell by ≥0.5% AND maintains the 1.5x knee.

Active sweeps still pulling new candidates: D (s73-120+), I (s25-50+),
U (s7-20+). Watcher auto-tests hard passes against the realism gate.
