# Failed Experiments Log

Track what didn't work to avoid repeating mistakes.

## Failed Approaches

| Date | Experiment | Why it Failed | Lesson |
|------|------------|---------------|--------|
| - | - | - | - |
| 2026-03-26 | `gspo_like_drawdown_mix15_sds03` | Higher smooth-downside penalty kept validation superficially positive, but daily replay `-20.96%` and hourly replay `-21.23%` with `36.37%` hourly max drawdown. | Stronger downside smoothing by itself was over-regularizing the policy and degrading executable replay quality. |
| 2026-03-26 | `gspo_like_drawdown_mix15_h512` | Smaller capacity improved holdout shape somewhat, but both replay horizons stayed negative: daily `-15.18%`, hourly `-11.45%`. | Reducing hidden size to `512` did not fix the replay realism gap; the issue is not just model size. |
| 2026-03-26 | `gspo_like_drawdown_mix15_tp01_dd03` | Combining the two individually useful knobs without extra friction caused collapse: holdout robust `-243.32`, daily replay `-26.33%`, hourly replay `-26.20%`. | `trade_penalty=0.01` and `drawdown_penalty=0.03` are not additive on their own; interaction needs friction support. |
| 2026-03-26 | `gspo_like_drawdown_mix15_slip12` | Hourly replay jumped to `+41.08%`, but daily replay still collapsed to `-23.69%` and holdout robust stayed very poor at `-222.13`. | More slippage training can improve hourly execution realism, but it is too unstable alone and must be paired with other controls. |

## Notes
- Document hyperparameters that led to instability
- Track architectures that didn't converge
- Note data combinations that caused issues
