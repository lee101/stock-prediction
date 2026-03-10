# Budget Guided Keep Count

Hypothesis:
- The current best timestamp-budget-head branch only uses the learned regime
  signal after the dynamic budget has already decided which names survive.
- Letting the learned skip/selective/broad regime influence candidate count
  should improve trade allocation more directly.

Exact files changed:
- `src/autoresearch_stock/train.py`
- `src/autoresearch_stock/experiments/budget_guided_keep_count/__init__.py`
- `src/autoresearch_stock/experiments/budget_guided_keep_count/keep_count.py`
- `tests/test_autoresearch_stock_budget_guided_keep_count.py`
- `tests/test_autoresearch_stock_train_smoke.py`

Core idea:
- Keep the timestamp budget head from the current best branch.
- Replace the fixed cross-sectional `top_k + max_keep` path with a regime-aware
  rank gate:
  - `skip` hours keep only the top name,
  - `selective` hours keep fewer names,
  - `broad` hours are allowed to keep more names.
- Continue to apply budget-aware sizing after the learned keep-count gate.

Replay command:

```bash
.venv312/bin/python -m autoresearch_stock.train --frequency hourly --budget-guided-keep-count
```

Observed hourly benchmark:
- Run log: `analysis/autoresearch_stock_manual/budget_guided_keep_count_hourly_20260310T051603Z/train_hourly.log`
- `robust_score`: `1.845075`
- `val_loss`: `0.000320`
- `total_trade_count`: `213`

Interpretation:
- This beat the prior positive `0.886226` timestamp-budget-head baseline.
- The learned regime signal appears more useful when it influences both
  candidate survival and post-selection size, even though plain validation loss
  worsened slightly again.
