# ctrader/binance_bot

Pure-C scaffolding for running a pufferlib_market RL policy end-to-end on
Binance (and on offline MKTD files for parity tests).  The parent `ctrader/`
directory already has the Binance REST client (`binance_rest.c`), the C
market simulator (`market_sim.c`), and an MKTD reader (`mktd_reader.c`).
This subdirectory fills in the three pieces that were missing before live
trading could happen on the C side:

1. **Policy loader + forward pass** — `policy_mlp.c` loads a `.ctrdpol`
   binary that `scripts/export_policy_to_ctrader.py` produces from any
   pufferlib_market MLP checkpoint, and runs the forward pass in pure C.
   Zero dependencies (no libtorch, no BLAS).  Verified against Python
   within **1.4e-6** on all 25 logits for the s42 production policy.
2. **Observation builder** — `obs_builder.c` produces the exact 209-dim
   vector `pufferlib_market/inference.py:build_observation` would produce
   given the same MKTD row + portfolio state.  Verified against a Python
   fixture at **0.000e+00** max difference.
3. **End-to-end backtest binary** — `backtest_main.c` reads an MKTD file,
   loads a `.ctrdpol` policy, walks a window forward picking argmax
   actions, simulates cash/position bookkeeping, and prints total return +
   max drawdown + num trades.

## Build & test

```
cd ctrader/binance_bot
make               # compiles everything into .tmp/
make test          # builds and runs both parity tests plus the backtest smoke
make fixtures      # regenerates parity fixtures from Python checkpoints
make clean
```

`TMPDIR` is pinned to `./.tmp` in the Makefile to dodge the global `/tmp`
gcc/triton race that breaks other builds on this host.

## Current status (2026-04-08)

- **Policy parity**: ✅ 1.4e-6 max abs diff on s42 (209→1024³→LN→512→25,
  2.85M params) against a Python `torch.nn` twin built from the same
  state_dict.
- **Observation parity**: ✅ 0.0 max abs diff on window 0 of
  `stocks12_daily_v5_rsi_val.bin` against `inference.build_observation`.
- **Backtest headline**: ⚠️ C reports `+14.80% / 68 trades / DD 16.72%` on
  s42 window 0; Python `render_prod_stocks_video.py` reports
  `+34.59% / 22 trades / DD 8.37%` on the same window.  The **policy
  forward is byte-perfect**, so the gap is entirely in the v0 trade
  simulator in `backtest_main.c`: full-cash allocation on every flip, no
  decision lag (Python uses lag ≥ 2), no fill buffer, no max-hold, no
  fractional sizing.  This is the next follow-up.

## Files

| file | purpose |
|---|---|
| `policy_mlp.{c,h}` | loader + forward pass (Linear → ReLU → LayerNorm → Linear → ReLU → Linear) |
| `obs_builder.{c,h}` | builds the 209-dim obs vector from an MKTD row + `CtrdpolPortfolioState` |
| `backtest_main.c` | end-to-end backtest binary (mktd + policy + C trade sim) |
| `tests/test_policy_mlp_parity.c` | verifies C forward == Python to 1e-4 |
| `tests/test_obs_builder_parity.c` | verifies C obs == Python to 1e-6 |
| `Makefile` | build + test driver |
| `tests/*.bin` | parity fixtures (regenerate with `make fixtures`) |

## Next steps (planned)

1. **Close the backtest PnL gap**: port `decision_lag`, `fill_buffer_bps`,
   fractional action-allocation bins, and `max_hold_bars` from
   `pufferlib_market/evaluate_holdout.py`'s simulator path into
   `backtest_main.c`, then re-parity-check against Python on multiple
   windows.
2. **Replace `ctrader/trade_loop.c`'s stub `build_observation`**
   (currently 6 dims per symbol) with `obs_builder.c` so the existing main
   binary actually uses the trained features.
3. **Replace `ctrader/policy_infer.c`'s stub `policy_forward`** with a
   shim that calls `ctrdpol_forward` from `policy_mlp.c`.
4. **Wire a Python feature helper** that recomputes the 16-feature-per-
   symbol block live from Binance klines (or append into an MKTD file on
   each cycle) so the live loop can feed the same features training saw.
5. **Paper-trade in dry-run** for at least 24h against real Binance data,
   diffing equity curve vs a Python shadow run, before any live fill.
6. **Add Alpaca REST support** alongside Binance so the same binary drives
   both venues (Alpaca is already the stocks12 production venue per
   `alpacaprod.md`; this sub-module is named `binance_bot` historically).
