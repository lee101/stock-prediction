Focus on dynamic per-timestamp trade budgets instead of a fixed top-k.

Current baseline:

- The kept hourly code already has:
  - a five-bucket ambiguity plan head,
  - a continuous cost-margin gate,
  - a fixed cross-sectional top-k rank gate,
  - a best hourly `robust_score` of `-44.128952`.
- The remaining likely failure mode is that fixed `top_k` can still over-prune strong
  hours and under-prune crowded average hours.

What to try:

- keep the current calibrated rank signal,
- replace the fixed per-timestamp keep count with a deterministic dynamic budget,
- let the keep count depend on score spread or score gaps within the timestamp,
- keep the gate simple and replayable before trying a learned budget head,
- keep the simulator and benchmark command unchanged.

Good ideas:

- threshold relative to top score and lower quantiles,
- score-gap based keep expansion,
- min/max keep bounds with a dynamic threshold inside them,
- no changes to `prepare.py`.

Bad ideas:

- removing the current cost-margin gate,
- adding a large new training objective before checking whether inference-time
  dynamic gating already helps,
- changing execution costs or simulator assumptions.

Success criteria:

- beat the current hourly best `robust_score` of `-44.128952`,
- further reduce low-edge churn,
- keep the experiment isolated and reproducible.
