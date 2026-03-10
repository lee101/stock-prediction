Focus on replacing the hard budget-class switch with continuous budget-conditioned thresholds.

Current baseline:

- The kept hourly branch already has:
  - a five-bucket ambiguity plan head,
  - a continuous cost-margin gate,
  - a learned timestamp budget head,
  - budget-guided candidate survival,
  - budget-aware post-selection sizing,
  - a best hourly `robust_score` of `1.845075`.
- The remaining discontinuity is that the rank gate still uses `argmax` over
  `skip / selective / broad` to decide how many names survive.

What to try:

- keep the current realistic simulator unchanged,
- keep the current learned budget head unchanged,
- stop snapping each hour to one budget class before gating,
- let the full predicted budget probabilities modulate:
  - `top_k`,
  - `max_keep`,
  - `min_score`,
  - and threshold looseness,
- keep the branch deterministic and isolated behind one explicit flag.

Good ideas:

- continuous interpolation between skip/selective/broad budget settings,
- partial survival or partial sizing for marginal names if that keeps the path deterministic,
- minimal changes to `train.py` with most logic isolated under `src/autoresearch_stock/experiments/`,
- optimize for `robust_score` first and `val_loss` second.

Bad ideas:

- changing simulator assumptions or spread/slippage logic,
- removing the budget head,
- adding a larger encoder before testing whether the continuous gate alone helps,
- mixing this with unrelated feature-engineering changes.

Success criteria:

- beat the current hourly best `robust_score` of `1.845075`,
- keep the experiment replayable with one explicit CLI flag,
- preserve the realistic hourly evaluation harness unchanged.
