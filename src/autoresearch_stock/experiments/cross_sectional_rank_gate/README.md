# Cross Sectional Rank Gate

Hypothesis:
- The current ambiguity plan head plus cost-margin gate still allows too many
  simultaneous tradable windows through.
- A deterministic top-k gate applied within each timestamp should allocate trade
  budget to fewer, stronger windows without weakening the simulator.

Exact files changed:
- `src/autoresearch_stock/train.py`
- `src/autoresearch_stock/experiments/cross_sectional_rank_gate/__init__.py`
- `src/autoresearch_stock/experiments/cross_sectional_rank_gate/rank_gate.py`
- `tests/test_autoresearch_stock_cross_sectional_rank_gate.py`

Core idea:
- Reuse the current calibrated excess-edge signal:
  - five-bucket ambiguity plan scale,
  - continuous cost-margin scale.
- For each timestamp in each evaluation scenario, keep only the top-ranked
  candidates above a minimum score.
- Let the fixed simulator handle fills, spread, slippage, and portfolio sizing
  exactly as before.

Observed hourly benchmarks:
- Fixed top-k run: `analysis/autoresearch_stock_manual/cross_sectional_rank_gate_hourly_20260310T040353Z/train_hourly.log`
- Fixed top-k `robust_score`: `-44.128952`
- Fixed top-k `val_loss`: `0.000306`
- Fixed top-k `total_trade_count`: `269`
- Dynamic-budget run: `analysis/autoresearch_stock_manual/dynamic_rank_budget_hourly_20260310T041350Z/train_hourly.log`
- Dynamic-budget `robust_score`: `-22.151952`
- Dynamic-budget `val_loss`: `0.000308`
- Dynamic-budget `total_trade_count`: `158`
