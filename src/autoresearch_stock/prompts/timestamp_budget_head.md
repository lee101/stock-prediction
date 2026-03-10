Focus on a learned timestamp budget head, not another hand-tuned threshold.

Current baseline:

- The kept hourly path already has:
  - a five-bucket ambiguity plan head,
  - a continuous cost-margin gate,
  - a dynamic cross-sectional rank budget,
  - soft post-selection sizing,
  - a best hourly `robust_score` of `-21.700715`.
- The recent dynamic score-floor branch failed badly, while soft sizing only
  improved the score marginally.

What to try:

- keep the realistic simulator and current selection path,
- add a lightweight auxiliary head that predicts timestamp regime or budget,
- derive labels from the existing cost-aware plan labels:
  - `skip`,
  - `selective`,
  - `broad`,
- use that learned regime signal to modulate sizing or budgeting,
- keep the experiment deterministic and isolated.

Good ideas:

- reconstruct aligned train/validation rows without changing the simulator,
- use grouped timestamp labels instead of per-sample heuristics,
- keep the new head small and low-weight,
- improve `robust_score` first, `val_loss` second.

Bad ideas:

- changing `prepare.py` behavior or simulator costs,
- adding a large transformer or deeper encoder,
- replacing the current dynamic budget with a looser heuristic before testing the learned budget signal.

Success criteria:

- beat the current hourly best `robust_score` of `-21.700715`,
- preserve or improve trade quality under the realistic simulator,
- keep the branch replayable with a single explicit CLI flag.
