# FastForecaster

FastForecaster is a new forecasting experiment focused on low validation MAE for stock/crypto CSV data in this repo.

It combines:
- Fast hybrid architecture: depthwise temporal mixing + causal attention + ReLU^2 feed-forward blocks.
- Attention stabilization: optional QK-norm with per-head learned scaling.
- MAE-first objective: optimize directly on predicted price MAE (with RMSE/MAPE/directional metrics tracked).
- Objective shaping: optional return-MAE and directional margin losses to reduce validation MAE drift.
- EMA evaluation: optional exponential moving average weights for validation/test checkpoint selection.
- Throughput controls: BF16/FP16, optional `torch.compile`, fused AdamW, and optional C++/CUDA MAE kernels.
- MLOps integration: logs to TensorBoard and W&B through `wandboard.py` (`WandBoardLogger`).
- Data hygiene: ignores invalid symbol filenames (e.g., comma-joined basket names) to avoid contaminating MAE evaluation.

## Training

Hourly data (default):

```bash
source .venv/bin/activate
python -m FastForecaster.run_training \
  --dataset hourly \
  --epochs 20 \
  --lookback 256 \
  --horizon 24 \
  --batch-size 128 \
  --torch-compile \
  --precision bf16 \
  --wandb-project stock
```

Daily data:

```bash
source .venv/bin/activate
python -m FastForecaster.run_training --dataset daily --epochs 30 --lookback 128 --horizon 10
```

`--dataset daily` now auto-adjusts `min_rows_per_symbol` to a practical lower bound
when you do not pass `--min-rows-per-symbol`, so daily runs can include more symbols.

Disable advanced objective components for ablation:

```bash
source .venv/bin/activate
python -m FastForecaster.run_training \
  --dataset hourly \
  --return-loss-weight 0.0 \
  --direction-loss-weight 0.0 \
  --no-ema-eval \
  --no-qk-norm
```

Custom symbol subset:

```bash
source .venv/bin/activate
python -m FastForecaster.run_training --dataset hourly --symbols NVDA,GOOG,PLTR,DBX --max-symbols 0
```

## Optional C++/CUDA kernel

By default, the extension is not built. Enable it explicitly:

```bash
source .venv/bin/activate
python -m FastForecaster.run_training --use-cpp-kernels --build-cpp-extension
```

If compilation fails or CUDA toolchain is unavailable, kernel-enabled paths automatically fall back to pure PyTorch ops.

Current status:
- C++/CUDA kernels are available for MAE and weighted-MAE ops.
- Training objective path now supports optional compiled weighted-MAE forward with explicit autograd-safe backward.
- Default behavior is still pure PyTorch unless `--use-cpp-kernels` is enabled.

## Benchmark

```bash
source .venv/bin/activate
python -m FastForecaster.benchmark --dataset hourly --batch-size 128 --iters 200
```

Weighted-loss microbenchmark:

```bash
source .venv/bin/activate
python -m FastForecaster.benchmark --dataset hourly --batch-size 128 --iters 200 --weighted-loss
```

Seed sweep helper:

```bash
source .venv/bin/activate
python -m FastForecaster.seed_sweep --dataset hourly --seeds 1337,1701,2026,4242 --epochs 3
```

Each seed run is saved under its own directory (`seed_<seed>/`) inside the sweep folder, with
aggregate ranking written to `seed_sweep_results.json`.

## Artifacts

Outputs are written under `FastForecaster/artifacts/`:
- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `metrics/summary.json`
- `metrics/epoch_metrics.json`
- `metrics/test_per_symbol.json`

## Recent local experiment

Hourly 5-symbol run (`NVDA,GOOG,PLTR,DBX,MTCH`, 4 epochs, same data/window budget):

- Baseline (no return/direction losses, no EMA, no QK-norm): best val MAE `1.9301`, test MAE `2.1444`.
- Enhanced objective + EMA + QK-norm: best val MAE `1.8042`, test MAE `2.1077`.

This was a `-6.52%` best-validation-MAE improvement and `-1.71%` test-MAE improvement in that controlled comparison.

Hourly 11-symbol cleaned-universe sweep + seed scan:

- Historical same-split best remains `2.6082` (`seed=1701`) from the Frontier7 run family.
- In the current trainer path, the best re-tuned result is now `2.6158` (`seed=1701`) with
  `hidden_dim=384, num_layers=6, ff_multiplier=4, lr=2.55e-4, weight_decay=0.01, dropout=0.05, horizon_weight_power=0.26, return_loss_weight=0.2, direction_loss_weight=0.01, EMA, QK-norm, 5 epochs`.
- Frontier15 seed sweep showed high sensitivity on the `hwp=0.26` config family:
  - `1701: 2.6160` (best)
  - `3333: 2.6222`
  - `1337: 2.6276`
  - `2026: 2.6277`
- C++-kernel objective ablation on the `hwp=0.28` config produced identical metrics to pure-PyTorch objective mode in local runs.
- Frontier13 local refinement around the `hwp=0.30` winner did not beat it (best `2.6221`).
- Frontier14 architecture scan (depth/width/FF variants) also did not beat it (best `2.6204` at `hidden_dim=384, layers=6, ff_multiplier=4`).
- Frontier15 targeted regularization/horizon scan found the then-best `2.6160` (`hwp=0.26`).
- Frontier16 nearby horizon/dropout scan (`hwp=0.24/0.25/0.27` and `dropout=0.07` at `hwp=0.26`) did not beat Frontier15 (best `2.6215`).
- Frontier17 `wd/lr` scan around that winner did not improve it (best `2.6184` at `wd=0.011`, `lr=2.5e-4`).
- Frontier18 objective-weight scan (`return_loss_weight`, `direction_loss_weight`) nearly tied but still did not beat Frontier15:
  best `2.6161` at `return_loss_weight=0.2`, `direction_loss_weight=0.01`.
- Frontier19 micro-scan around that near-tie did improve current-path MAE:
  best `2.6158` at `lr=2.55e-4` with `return_loss_weight=0.2`, `direction_loss_weight=0.01`.
- Validation-optimized configs in this family do not always improve test MAE on the same split.

Expanded hourly training budget (22 symbols / 71,494 train windows):

- Best validation MAE observed: `2.1926` (`seed=1701`) with
  `lr=2.5e-4, weight_decay=0.01, dropout=0.05, horizon_weight_power=0.30, return_loss_weight=0.2, EMA, QK-norm`.
- This value is not directly comparable to the 11-symbol setup because the symbol universe and split sizes differ.

## Important data note

If your `trainingdatahourly/stocks` directory contains synthetic/basket files with non-symbol names
(for example `AAPL,MSFT,...csv`), FastForecaster now skips them automatically.

In local testing this fix reduced large-run MAE dramatically by removing a malformed high-error pseudo-symbol.
