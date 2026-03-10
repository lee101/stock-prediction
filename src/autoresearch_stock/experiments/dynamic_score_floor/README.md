# Dynamic Score Floor

Hypothesis:
- The current rank-budget path still lets weak hours through because it uses a
  static minimum score floor before the dynamic keep-count stage.
- A regime-aware score floor should skip weak timestamps entirely and only widen
  the candidate set when the top score is genuinely strong.

Exact files changed:
- `src/autoresearch_stock/train.py`
- `src/autoresearch_stock/experiments/cross_sectional_rank_gate/rank_gate.py`
- `src/autoresearch_stock/experiments/dynamic_score_floor/__init__.py`
- `src/autoresearch_stock/experiments/dynamic_score_floor/score_floor.py`
- `tests/test_autoresearch_stock_cross_sectional_rank_gate.py`

Core idea:
- Keep the current ambiguity plan head, cost-margin gate, and cross-sectional
  rank signal unchanged.
- After the top-k prefilter, compute a per-timestamp score floor from:
  - the top score,
  - a lower score quantile,
  - a strength-dependent gap scale.
- Skip timestamps whose best score barely clears the base floor.
- Feed the surviving names into the existing dynamic rank-budget gate.

Borrowed repo ideas:
- deterministic policy gating from prior portfolio-selection experiments,
- keep the simulator, costs, spreads, fills, and sizing unchanged.

Replay command:

```bash
.venv312/bin/python -m autoresearch_stock.train --frequency hourly --dynamic-score-floor
```

Observed hourly benchmark:
- Run log: `analysis/autoresearch_stock_manual/dynamic_score_floor_hourly_20260310T043516Z/train_hourly.log`
- `robust_score`: `-43.231049`
- `val_loss`: `0.000307`
- `total_trade_count`: `118`

Interpretation:
- The dynamic floor did reduce trade count sharply, but it gave up too much
  profitable coverage and regressed badly versus the `-22.151952` dynamic-budget
  baseline.
- Keep it as a documented failed branch, not the default path.
