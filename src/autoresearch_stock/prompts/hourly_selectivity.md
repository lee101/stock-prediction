Focus on hourly trade selectivity and cost-aware action quality.

Priority:

- improve `robust_score`,
- reduce low-edge churn,
- only secondarily reduce regression loss.

What to inspect:

- `src/autoresearch_stock/prepare.py` action construction and simulator entry costs,
- current trade count in recent turns,
- target asymmetry, opportunity, and ambiguity distributions,
- whether the model is learning when to stay flat.

Good idea directions:

- confidence or selectivity heads,
- cost-aware calibration against spread plus slippage,
- explicit no-trade gating,
- ranking or margin losses that focus capacity on actionable windows,
- action-magnitude suppression on ambiguous two-sided targets,
- symbol or regime conditioning that reduces unnecessary trades.

Bad idea directions:

- optimizing only `val_loss`,
- adding complexity that does not connect to execution behavior,
- changes that increase trade count without clearly improving holdout robustness.
