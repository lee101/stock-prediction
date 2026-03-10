Focus on soft broad-support quorum, not full-basket consensus.

Current baseline:

- The kept hourly branch already has:
  - a five-bucket ambiguity plan head,
  - a continuous cost-margin gate,
  - a learned timestamp budget head,
  - probability-aware budget-conditioned breadth and score thresholds,
  - a best hourly `robust_score` of `3.042843`.
- The entropy-confidence branch regressed to `2.448601`.
- The consensus-dispersion branch regressed further to `0.333672`.
- The likely issue is that full-basket confidence or consensus is too strict:
  some profitable broad hours seem to come from a few strong broad-supporting
  symbols rather than near-unanimous basket agreement.

What to try:

- keep the current realistic simulator unchanged,
- keep the current continuous budget-threshold branch as the baseline,
- compute a soft broad-support score or quorum from the per-symbol broad probabilities,
- allow broader expansion when a few symbols strongly support broad even if full-basket consensus is imperfect,
- keep contraction for obviously noisy pseudo-broad hours,
- keep the branch deterministic and isolated behind one explicit flag.

Good ideas:

- weighted or soft-count broad-support quorum,
- leader-tail broad support rather than unanimity,
- preserving mixed but profitable broad hours,
- improving `robust_score` first and `val_loss` second.

Bad ideas:

- changing simulator costs or fill assumptions,
- replacing the current budget head,
- requiring near-unanimous broad agreement before allowing expansion,
- mixing this with unrelated data or feature changes.

Success criteria:

- beat the current hourly best `robust_score` of `3.042843`,
- keep the experiment replayable with one explicit CLI flag,
- preserve profitable broad windows without reopening obviously noisy ones.
