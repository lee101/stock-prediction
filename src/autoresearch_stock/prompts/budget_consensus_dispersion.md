Focus on cross-symbol agreement inside each timestamp, not only entropy of the mean budget distribution.

Current baseline:

- The kept hourly branch already has:
  - a five-bucket ambiguity plan head,
  - a continuous cost-margin gate,
  - a learned timestamp budget head,
  - probability-aware budget-conditioned breadth and score thresholds,
  - a best hourly `robust_score` of `3.042843`.
- The recent entropy-confidence branch stayed positive at `2.448601`, but it
  over-contracted breadth and underperformed the current best.
- The remaining weakness is that the gate only looks at the mean budget
  distribution per timestamp, not whether the symbol-level budget predictions
  actually agree with each other.

What to try:

- keep the current realistic simulator unchanged,
- keep the current continuous budget-threshold branch as the baseline,
- compute a timestamp-level agreement or dispersion score across symbol budget probabilities,
- allow broader expansion only when multiple symbols agree on a broad regime,
- contract breadth when the timestamp mean looks broad but symbol-level predictions disagree,
- keep the branch deterministic and isolated behind one explicit flag.

Good ideas:

- agreement-aware modulation layered on top of the current continuous gate,
- using dispersion or pairwise disagreement of per-symbol budget probabilities,
- preserving profitable broad hours while filtering noisy pseudo-broad hours,
- improving `robust_score` first and `val_loss` second.

Bad ideas:

- changing simulator costs or fill assumptions,
- replacing the current budget head,
- adding large new encoders before testing whether agreement-aware gating helps,
- mixing this with unrelated feature or data changes.

Success criteria:

- beat the current hourly best `robust_score` of `3.042843`,
- keep the experiment replayable with one explicit CLI flag,
- preserve useful breadth while cutting only the noisy expansion hours.
