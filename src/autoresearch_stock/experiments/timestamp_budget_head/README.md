# Timestamp Budget Head

Hypothesis:
- The current best hourly branch still gets all of its gains from inference-time
  allocation rules.
- A lightweight budget head trained on timestamp-level labels should help the
  model learn when an hour is a skip, a one-name hour, or a broader opportunity
  hour under the realistic simulator.

Exact files changed:
- `src/autoresearch_stock/train.py`
- `src/autoresearch_stock/experiments/timestamp_budget_head/__init__.py`
- `src/autoresearch_stock/experiments/timestamp_budget_head/budget_head.py`
- `tests/test_autoresearch_stock_timestamp_budget_head.py`
- `tests/test_autoresearch_stock_train_smoke.py`

Core idea:
- Reconstruct train and validation sample rows using `prepare.py` internals so
  timestamps stay aligned with the fixed feature tensors.
- Derive timestamp budget labels from the existing cost-aware plan labels:
  - `skip`,
  - `selective`,
  - `broad`.
- Train a small auxiliary head on those labels.
- Use the predicted regime signal to modulate the current soft rank sizing path
  instead of adding another hard gate.

Replay command:

```bash
.venv312/bin/python -m autoresearch_stock.train --frequency hourly --timestamp-budget-head
```

Observed hourly benchmark:
- Run log: `analysis/autoresearch_stock_manual/timestamp_budget_head_hourly_20260310T050221Z/train_hourly.log`
- `robust_score`: `0.886226`
- `val_loss`: `0.000319`
- `total_trade_count`: `164`

Interpretation:
- This is the first hourly branch to push `robust_score` above zero under the
  realistic simulator.
- The fit metric worsened slightly versus the soft-sizing branch, but the
  trading objective improved sharply, which reinforces that the budget signal is
  learning something the plain validation loss does not capture.
