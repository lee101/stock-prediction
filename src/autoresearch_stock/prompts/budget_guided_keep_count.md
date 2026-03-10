Focus on letting the learned budget head control candidate count, not only post-selection size.

Current baseline:

- The kept hourly branch already has:
  - a five-bucket ambiguity plan head,
  - a continuous cost-margin gate,
  - a dynamic cross-sectional rank budget,
  - a learned timestamp budget head,
  - a best hourly `robust_score` of `0.886226`.
- The current budget head only rescales names after the fixed dynamic budget has
  already decided who survives.

What to try:

- keep the learned `skip / selective / broad` budget head,
- let that regime class influence candidate count directly,
- allow `broad` hours to pass more names,
- force `skip` hours to stay concentrated,
- keep the realistic simulator and current sizing path unchanged otherwise.

Good ideas:

- budget-class-dependent `top_k`,
- budget-class-dependent `max_keep`,
- tighter gap thresholds in `skip` hours and looser ones in `broad` hours,
- preserving replayability behind one explicit flag.

Bad ideas:

- removing the learned budget head,
- changing simulator assumptions,
- adding a larger encoder before checking whether the learned keep-count policy already helps.

Success criteria:

- beat the current hourly best `robust_score` of `0.886226`,
- keep the branch deterministic and isolated,
- improve trade allocation rather than just inflating trade count.
