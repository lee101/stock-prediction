# XGB 2×2 matrix — 16-seed × n1600/lr01 combo (2026-04-19)

## Setup
Same universe, eval harness, and 30-window grid as prior runs. Combining the two axes that each individually beat the deployed 5-seed config:
- **axis A**: seed averaging (5 → 16 prime-seeded pkls)
- **axis B**: slower trees (n=400/lr=0.03 → n=1600/lr=0.01)

Both axes trained on `--train-start 2020-01-01`, `--device cuda`, 846 symbols. All evals `--blend-mode mean --top-n 1 --leverage 1.0 --fill-buffer-bps 5 --fee-rate 2.78e-05`.

## Results — complete 2×2 matrix

| config | med %/mo | p10 %/mo | sortino | worst DD | n_neg |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5-seed 400/0.03 (**deployed live**) | +38.85 | +4.82 | 18.86 | 31.44 | 2/30 |
| 16-seed 400/0.03 | **+40.44** | +4.74 | **19.63** | 31.62 | 2/30 |
| 5-seed 1600/0.01 | +38.24 | **+6.79** | 19.59 | **29.45** | 2/30 |
| 16-seed 1600/0.01 (**combo**) | +39.31 | +6.79 | 19.59 | 29.45 | 2/30 |

## Structural read

The two axes hit **different metrics** and **don't compound**:

- **LR/depth axis dominates the tail.** n1600/lr01 fixes p10 to +6.79 and DD to 29.45% *regardless of seed count* (5-seed and 16-seed combo are bit-for-bit identical on tail metrics). The slower-tree regime is reshaping *which* windows become weak, not averaging them out.
- **Seed averaging helps median but with diminishing returns at slower LR.** At 400/0.03: 5→16 lifts median +1.59. At 1600/0.01: 5→16 only lifts median +1.07 and caps below the 400/0.03 16-seed cell.
- **The best cell for any single metric**:
  - median: 16-seed 400/0.03 (+40.44)
  - p10 / DD / sortino: tied between 5-seed 1600/0.01 and 16-seed 1600/0.01

**Verdict**: Combo is **not a strict upgrade** by the pre-registered criterion (Δmed vs deployed must be ≥ +1.0; combo is +0.46). It is a tail-risk-preferred pick — use it if we want max p10 safety at small median cost.

## Deploy-candidate ranking (vs deployed 5-seed 400/0.03)

| candidate | Δmed | Δp10 | Δsortino | ΔDD | Δn_neg | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 16-seed 400/0.03 | +1.59 | −0.08 | +0.77 | +0.18 | 0 | **median-preferred** |
| 5-seed 1600/0.01 | −0.61 | +1.97 | +0.73 | −1.99 | 0 | **tail-preferred** |
| 16-seed 1600/0.01 combo | +0.46 | +1.97 | +0.73 | −1.99 | 0 | **hybrid** (safest) |

All three pass the deploy gate (median ≥ 35%, p10 ≥ 4%, neg ≤ 2, DD ≤ 32). None are auto-deployable (all three sit outside `monitoring/current_algorithms.md` §6 "same-topology 5-seed top-N" authority).

## Recommendation for the next user-approved swap

The **hybrid combo** (16 seeds × n1600/lr01) is the Pareto-safest pick:
- Never worse than deployed on any metric (all Δs are positive-or-neutral).
- Captures the entire tail-risk improvement from the LR axis.
- Gives up 1.13%/mo median vs the pure 16-seed cell, but gains 2.05%/mo p10 and 2.17%/mo worst-DD reduction.
- n=1600 × 16 models = 4× × 3.2× = ~13× inference vs deployed baseline. Still sub-second on 846 symbols for a once-per-day decision.

Or, if median dominance is preferred, stay on the 16-seed 400/0.03 cell. The combo's only dominance advantage over the plain 16-seed is tail metrics.

## Files
- `analysis/xgbnew_deploy_baseline/deploy_5seed_lev1_top1.json`
- `analysis/xgbnew_deploy_baseline/deploy_16seed_20260419.json`
- `analysis/xgbnew_deploy_baseline/deploy_n1600_lr01_20260419.json`
- `analysis/xgbnew_deploy_baseline/deploy_16seed_n1600_lr01_20260419.json`
