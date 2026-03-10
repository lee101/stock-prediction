Focus on calibrated trade/no-trade margins on top of the existing five-bucket plan head.

Current baseline:

- The kept hourly code already has a five-bucket plan gate in
  `src/autoresearch_stock/experiments/ambiguity_plan_gate/`.
- It improved hourly `robust_score` to `-91.756334`, but the simulator is still negative.
- The remaining weakness is likely trades whose predicted edge is too close to
  spread + slippage + fee.

What to try:

- keep the current five-bucket plan head,
- add one small, explicit calibration mechanism that suppresses trades unless
  expected edge clears a safety margin above round-trip cost,
- make that safety margin depend only on existing deterministic inputs such as:
  - spread,
  - symbol embedding,
  - volatility,
  - recent ambiguity or opportunity features,
- prefer a simple learned scalar or threshold head over a large architecture change,
- make the calibration connect directly to execution behavior under
  `build_action_frame`, not just raw regression loss.

Do not waste time on:

- re-deriving the same label distributions unless needed for a specific threshold,
- broader Chronos or LLM-style expansions,
- changes that increase trade count,
- changes that require editing `src/autoresearch_stock/prepare.py`.

Success criteria:

- beat the current hourly best `robust_score` of `-91.756334`,
- ideally reduce low-edge churn further,
- keep the experiment isolated and replayable.
