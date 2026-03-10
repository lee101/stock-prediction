Focus on portfolio ranking, not a bigger per-trade model.

Current baseline:

- The kept hourly code already has:
  - a five-bucket ambiguity plan head,
  - a continuous cost-margin gate,
  - a best hourly `robust_score` of `-75.984517`.
- The remaining likely failure mode is still over-allocation across many tradable windows.

What to try:

- keep the current ambiguity plan head and cost-margin gate,
- add a cross-sectional rank or top-k gate so only the strongest windows survive within
  each timestamp,
- rank using the calibrated excess-edge signal, not raw predicted return,
- prefer a simple deterministic inference-time rank gate before adding a heavier
  training-time ranking loss,
- keep the simulator and benchmark command unchanged.

Good ideas:

- top-k per timestamp using the current calibrated margin score,
- soft rank decay for lower-ranked candidates,
- symbol-aware score calibration only if it stays simple,
- respecting `max_positions` and realistic execution costs.

Bad ideas:

- bigger hidden layers or deeper encoders without changing execution behavior,
- relaxing spread/slippage assumptions,
- adding non-deterministic search or hyperparameter sweeps inside one turn.

Success criteria:

- beat the current hourly best `robust_score` of `-75.984517`,
- reduce total trade count or improve average edge quality,
- keep the experiment isolated and replayable.
