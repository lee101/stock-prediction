Focus on dynamic score floors, not only dynamic keep counts.

Current baseline:

- The kept hourly code already has:
  - a five-bucket ambiguity plan head,
  - a continuous cost-margin gate,
  - a cross-sectional rank signal,
  - a dynamic per-timestamp rank budget,
  - a best hourly `robust_score` of `-22.151952`.
- The remaining likely failure mode is that the current gate still uses a static
  minimum score floor across weak and strong hours.

What to try:

- keep the current dynamic-rank-budget path,
- make the score floor itself depend on timestamp-level regime strength,
- let weak hours drop out entirely if the best score barely clears costs,
- let strong hours pass more names only when the score spread is wide,
- keep the change deterministic and inference-time before adding a new training head.

Good ideas:

- top-score strength thresholds relative to the current minimum score,
- dynamic floor formulas based on top score and lower quantiles,
- score-dispersion-aware expansion only in strong hours,
- preserving the current realistic simulator and benchmark command.

Bad ideas:

- changing `prepare.py` or simulator assumptions,
- widening the model or adding a large new loss before testing the gate,
- adding stochastic search or hidden hyperparameter sweeps inside one turn.

Success criteria:

- beat the current hourly best `robust_score` of `-22.151952`,
- reduce low-edge churn again,
- keep the experiment isolated and replayable.
