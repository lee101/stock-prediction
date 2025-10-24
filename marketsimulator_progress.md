# Marketsimulator Progress Log — 24 Oct 2025

## What We Did
- Hardened numerical coercion in `trade_stock_e2e.py` by routing all real-analysis metrics through `coerce_numeric(..., prefer="mean")`. This eliminates the `TypeError: cannot convert the series to <class 'float'>` crash when pandas Series leak through the forecasting pipeline.
- Patched the simulator’s `PriceSeries` stack (`marketsimulator/state.py`) to use the same coercion helpers for bid/ask lookups, spread calculations, and trade-cash accounting—closing the remaining gaps that surfaced once real runs reloaded the module under the simulator harness.
- Re-ran `run_trade_loop.py` with the full 17-symbol book under a fresh `TRADE_STATE_SUFFIX=sim_realfix`, GPU enabled, and `SIM_LOGURU_LEVEL=INFO`. Kronos/Toto still fall back because several price windows lack high/low columns, but the loop now completes cleanly and emits the new annualised PnL summary.

## Current Snapshot
- **PnL:** \$0.00 on the 2-step smoke test (cash = \$100k, no open positions). All candidates remain blocked by the walk-forward Sharpe / dollar-volume gates until the real analytics path is restored.
- **Forecasting status:** `predict_stock_forecasting` import still fails; Toto compile disabled to sidestep missing `torch_inductor` cache artifacts; Kronos aborts when OHLC slices are incomplete.
- **State hygiene:** `TRADE_STATE_SUFFIX=sim_realfix` keeps the new run isolated, avoiding the stale trade history that distorted earlier PnL breakdowns.

## Open Issues / Next Steps
1. **Regenerate OHLC data for lagging symbols.** Without consistent `High/Low` columns, both Kronos and Toto refuse to return predictions, forcing us onto the fallback simulator metrics.
2. **Bootstrap the compiled Toto cache.** `_ensure_compilation_artifacts()` needs a writable `compiled_models/torch_inductor/…` tree so torch.compile can succeed. Once the data gap is fixed, re-enable compilation with higher `TOTO_NUM_SAMPLES` (target 2048).
3. **Re-test probe transitions with real forecasts.** After the analytics stack is stable, focus on NVDA/AAPL/MSFT probe clean-up and measure the Kelly/risk-dial impact under genuine GPU inference.
4. **Document long-run behaviour once analytics are live.** The 30-step run still produced the same blocked behaviour; repeat it after the data/compile fixes so we can capture meaningful per-strategy PnL and inventory profiles.
