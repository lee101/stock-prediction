# Soft Rank Sizing

Hypothesis:
- The current dynamic rank budget is the best hourly path so far, but the failed
  dynamic score-floor branch suggests harder rejection is now too blunt.
- A softer cross-sectional sizing pass should preserve coverage while shifting
  more capital toward the strongest names and strongest hours.

Exact files changed:
- `src/autoresearch_stock/train.py`
- `src/autoresearch_stock/experiments/soft_rank_sizing/__init__.py`
- `src/autoresearch_stock/experiments/soft_rank_sizing/sizing.py`
- `tests/test_autoresearch_stock_cross_sectional_rank_gate.py`

Core idea:
- Keep the ambiguity plan head, cost-margin gate, top-k prefilter, and dynamic
  rank budget unchanged.
- After the dynamic gate, scale surviving prediction magnitudes by:
  - timestamp-level score strength,
  - within-timestamp relative rank.
- Let `build_action_frame` convert the scaled predictions into smaller or larger
  position strengths under the same realistic simulator.

Replay command:

```bash
.venv312/bin/python -m autoresearch_stock.train --frequency hourly --soft-rank-sizing
```

Observed hourly benchmark:
- Run log: `analysis/autoresearch_stock_manual/soft_rank_sizing_hourly_20260310T044300Z/train_hourly.log`
- `robust_score`: `-21.700715`
- `val_loss`: `0.000307`
- `total_trade_count`: `138`

Interpretation:
- Soft sizing beat the prior best `-22.151952` dynamic-budget baseline without
  collapsing opportunity count.
- The gain is small, but it is the best hourly result so far under the realistic
  five-minute simulator.
