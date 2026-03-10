# Cost Margin Gate

Hypothesis:
- The current five-bucket plan head removes some ambiguous trades, but it still lets
  through too many windows whose predicted edge is only marginally above execution cost.
- A second calibration head should learn a continuous trade/no-trade margin using the
  existing spread, volatility, and symbol context.

Exact files changed:
- `src/autoresearch_stock/train.py`
- `src/autoresearch_stock/experiments/cost_margin_gate/__init__.py`
- `src/autoresearch_stock/experiments/cost_margin_gate/margin_head.py`
- `tests/test_autoresearch_stock_cost_margin_gate.py`

Core idea:
- Keep the turn-4 ambiguity plan head.
- Add a margin head that predicts a continuous gate in `[0, 1]`.
- Train that gate against a deterministic excess-edge target derived from:
  - target best edge after costs,
  - a train-split edge floor/ceiling,
  - spread-scaled safety margin,
  - volatility-scaled safety margin.

Why this should help the realistic simulator:
- `build_action_frame` only matters when the model's predicted edge exceeds cost.
- This head explicitly suppresses trades whose target edge is too close to the cost floor,
  so the remaining predictions should be more selective without changing `prepare.py`.

Observed hourly benchmark:
- Manual follow-up run: `analysis/autoresearch_stock_manual/cost_margin_gate_hourly_20260310T035051Z/train_hourly.log`
- `robust_score`: `-75.984517`
- `val_loss`: `0.000309`
- `total_trade_count`: `443`
