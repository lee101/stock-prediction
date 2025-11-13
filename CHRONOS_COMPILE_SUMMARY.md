# Chronos2 Torch.compile Optimization Summary

**Date:** 2025-11-12  
**Status:** ✅ Applied & verified  
**Scope:** `chronos-forecasting/src/chronos/chronos2/model.py`

---

## What Was Happening?

Enabling `torch.compile` for Chronos2 triggered graph breaks inside
`Chronos2Model._prepare_patched_future`. The offending line used a
Python `if torch.isnan(...).any():` guard to validate future covariates.
During compilation this introduces data-dependent control flow, forcing
Dynamo to abandon CUDA graphs and replay the Python branch every call.

## The Fix

We detect `torch.compile` tracing via `torch.compiler.is_compiling()`
and skip the Python-side NaN check when the graph is being captured.
The validation still runs during eager execution (uncompiled mode), so
mis-specified masks continue to raise a `ValueError`. This keeps the
compiled graph free of data-dependent branching without masking real
bugs in non-compiled workflows.

On top of the model change, `Chronos2OHLCWrapper` now retries failed
compiled inferences by restoring the eager model and re-running the
request. Any `KeyboardInterrupt` is propagated immediately, but Dynamo
internal crashes (like the `tracer_output` UnboundLocalError) get turned
into a one-time warning plus automatic fallback.

## Validation (trainingdata/)

`tests/test_chronos_compile_accuracy.py` now performs back-to-back runs
against real CSVs under `trainingdata/` for both eager and compiled
models:

| Symbol | Uncompiled MAE | Compiled MAE | ΔMAE | Uncompiled ms | Compiled ms |
|--------|----------------|--------------|------|---------------|-------------|
| BTCUSD | 4,070.6275 | 4,070.6307 | 0.0032 | 650.86 | 6,518.23 |
| ETHUSD | 364.0123 | 364.0124 | 0.0001 | 60.90 | 46.14 |

MAE drift stays well below the `5e-3` tolerance, so accuracy on real
data is unchanged. Baselines are written to
`tests/chronos_mae_baseline.txt` for future comparisons.

## How to Re-run

```bash
source .venv313/bin/activate
python tests/test_chronos_compile_accuracy.py
```

Set `TORCH_LOGS="graph_breaks,cudagraphs"` if you want to inspect the
remaining guard locations; `_prepare_patched_future` no longer shows up.

---

Next targets:
1. Profile the large latency delta for BTCUSD’s first compiled run
   (mostly compilation+autotune) and consider caching compiled graphs.
2. Audit other `torch.isnan(...).any()` guards in Chronos2 to ensure
   they are either compile-safe or gated similarly.
