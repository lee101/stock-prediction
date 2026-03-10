Focus on soft cross-sectional sizing, not another harder binary gate.

Current baseline:

- The kept hourly code already has:
  - a five-bucket ambiguity plan head,
  - a continuous cost-margin gate,
  - a dynamic cross-sectional rank budget,
  - a best hourly `robust_score` of `-22.151952`.
- The dynamic score-floor follow-up regressed to roughly `-43`, which suggests
  harder timestamp rejection is now too blunt.

What to try:

- keep the current dynamic-rank-budget selection path,
- do not add another binary filter ahead of the simulator,
- instead scale surviving predictions by:
  - timestamp strength,
  - relative rank within the timestamp,
- let `build_action_frame` translate the scaled predictions into smaller or
  larger trade sizes under the same simulator.

Good ideas:

- deterministic per-timestamp scaling from top score and score quantiles,
- stronger scale for top-ranked names and weaker scale for borderline names,
- preserving replayability and keeping `prepare.py` untouched.

Bad ideas:

- dropping large fractions of timestamps outright,
- changing simulator assumptions,
- adding heavy new training losses before testing inference-time sizing.

Success criteria:

- beat the current hourly best `robust_score` of `-22.151952`,
- improve trade quality without collapsing opportunity count,
- keep the experiment isolated and reproducible.
