Focus on budget-head confidence, not only average budget probabilities.

Current baseline:

- The kept hourly branch already has:
  - a five-bucket ambiguity plan head,
  - a continuous cost-margin gate,
  - a learned timestamp budget head,
  - probability-aware budget-conditioned breadth and score thresholds,
  - budget-aware sizing,
  - a best hourly `robust_score` of `3.042843`.
- The remaining weakness is that the gate uses mean budget probabilities per
  hour, but it does not distinguish between confident broad hours and uncertain
  high-entropy hours.

What to try:

- keep the current realistic simulator unchanged,
- keep the current continuous budget-threshold branch as the baseline,
- compute a timestamp-level confidence or entropy signal from the budget head,
- only allow broad expansion when the budget prediction is confident,
- contract `top_k`, `max_keep`, or sizing when the budget distribution is high-entropy,
- keep the branch deterministic and isolated behind one explicit flag.

Good ideas:

- confidence-aware interpolation instead of another hard class switch,
- using entropy to penalize uncertain broad hours,
- modest scaling of marginal names when confidence is low,
- improving `robust_score` first and `val_loss` second.

Bad ideas:

- changing simulator costs or fill assumptions,
- replacing the current budget head,
- adding large new encoders before testing whether confidence-aware gating helps,
- mixing this with unrelated data or feature changes.

Success criteria:

- beat the current hourly best `robust_score` of `3.042843`,
- keep the experiment replayable with one explicit CLI flag,
- reduce over-allocation during uncertain hours without collapsing trade count.
