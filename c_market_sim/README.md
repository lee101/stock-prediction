# c_market_sim/

End-to-end GPU training + validation via XGBoost's C API and a custom
CUDA market-simulator as the early-stopping metric.

## Why this exists

Two complementary "all-GPU" paths live in this repo now:

1. **Python GPU path** (`boostbaseline/gpu_core.py`) — CuPy arrays,
   XGBoost `device='cuda'`, same backtest math, all under the Python
   orchestrator. This is the path to iterate on daily.
2. **C/CUDA path** (this directory) — the tight-loop version. Uses the
   XGBoost C API to drive training, pulls predictions directly from a
   CUDA array interface each round, and runs the validation
   market-simulator as a custom CUDA kernel. This is the path to lift
   into production inference if the Python loop overhead ever matters,
   and the experimentation ground for fused sim + booster-slice
   strategies (early-stop on total_return / Sharpe / drawdown, not on
   RMSE).

The two paths should produce numerically close metrics when fed the
same features; the .cu build here is a scaffold, not a replacement.

## Prerequisites

The PyPI `xgboost` wheel ships `libxgboost.so` but **no headers**, so
you need a source build with CUDA enabled to compile this .cu file.

```bash
git clone --recursive https://github.com/dmlc/xgboost.git "$HOME/src/xgboost"
cmake -S "$HOME/src/xgboost" -B "$HOME/src/xgboost/build" -DUSE_CUDA=ON -GNinja
cmake --build "$HOME/src/xgboost/build" -j
export XGBOOST_HOME="$HOME/src/xgboost"
```

For an RTX 5090 (Blackwell) the Makefile targets `sm_120` by default.
Ada = `sm_89`, Ampere = `sm_86`, Hopper = `sm_90` — override with
`make ARCH=sm_89`.

## Build and run

```bash
cd c_market_sim
make                                  # compiles xgb_cuda_market_sim_example
make run                              # runs with LD_LIBRARY_PATH set
```

Expected output (synthetic data, 200 rounds, early-stop patience 30):

```
[iter   0] total_return=+0.0031  sharpe=+0.12
[iter   1] total_return=+0.0047  sharpe=+0.18
…
[early stop] no improvement for 30 rounds; best iter=47

[done] best validation total_return=+0.0193 at iter=47
[done] saved xgb_cuda_market_sim.json
```

The saved JSON can be loaded by Python (`xgboost.Booster().load_model`)
or by the C API (`XGBoosterLoadModel`). Inference via RAPIDS FIL is a
drop-in replacement if you care about serving throughput.

## Swapping in real features

Replace the synthetic generator block (lines marked "Generate synthetic
data on device") with a CUDA malloc + copy of your real feature tensor.
For Chronos-2 + cross-sectional features the tensor is already on-device
if you ran the forecaster with `device='cuda'` — pass the raw device
pointer via `cai_json()` and you skip the host bounce entirely.

## Known caveats

* `XGBoosterPredictFromCudaArray`'s `out_result` has ambiguous residency
  in the public docs ("copy before use"). The sample calls
  `cudaPointerGetAttributes` and only copies to device when the runtime
  reports host-resident memory — robust either way, no reliance on
  undocumented behavior.
* `XGBoosterSlice` returns a fresh booster cut to `[0, best_iter + 1)`.
  Free the original separately.
* Compile-time `ARCH` must match your GPU. Building for the wrong SM is
  a silent runtime cliff (no kernels launch).
