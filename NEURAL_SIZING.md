## Neural Sizing Playbook

### 1. Verify the production loop
1. Activate the `.venv313` environment (or whichever runtime you use for live trading).
2. Launch `trade_stock_e2e.py` with normal settings. Neural sizing is on by default (`MARKETSIM_ENABLE_NEURAL_SIZING=1`), so no extra flags are needed.
3. Watch the logs during start‑up for a line similar to:
   ```
   Neural sizing scale 0.78 from strategytrainingneural/reports/run_20251114_093401 (weights: VolAdjusted_10pct_StockDirShutdown=0.82@2025-09-17, VolAdjusted_10pct_UnprofitShutdown=0.74@2025-09-17, ...)
   ```
   That confirms the `run_20251114_093401` checkpoint is loaded and the reference strategies match what we validated in `strategytraining/test_sizing_on_precomputed_pnl.py`.
4. The trade process now pulls its symbol universe from `strategytraining/sizing_strategy_fast_test_results.json`. Regenerate that file whenever you refresh the sizing dataset so live trading and training remain aligned.

### 2. Exercise the simulator with the same configuration
1. `marketsimulator/run_trade_loop.py` mirrors the production wiring. When `--symbols` is omitted it now pulls the same experiment symbols (falling back to the historical basket only if the JSON is missing).
2. Run a quick smoke test to ensure the mocked Alpaca stack and neural scaler agree with production:
   ```bash
   source .venv313/bin/activate
   python marketsimulator/run_trade_loop.py --steps 5 --fast-sim
   ```
   The simulator will emit the same “Neural sizing scale …” log line on day zero because it imports `trade_stock_e2e`.
3. CI / regression runs can pass an explicit `--symbols` list when you intentionally want to target a subset; otherwise leave it blank to stay in lockstep with the training artifacts.

These steps keep the portfolio‑level neural allocator, gating logic, borrow-cost model, and tradable pairs identical between `strategytraining`, `trade_stock_e2e.py`, and `marketsimulator/run_trade_loop.py`. Update the `MARKETSIM_NEURAL_*` environment variables only when you deliberately change checkpoints or reference strategies.***
