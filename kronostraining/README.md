# Kronos Fine-Tuning Harness

This package wires the upstream [`external/kronos`](../external/kronos) foundation
model into the trading-bot repository. It provides a CSV driven data loader,
GPU training loop, and evaluation helpers tailored to the synthetic dataset
under `trainingdata/`.

## Quickstart

1. Install the Kronos dependencies (must use `uv pip` per repo policy):

   ```bash
   uv pip install -r external/kronos/requirements.txt
   ```

2. Launch training (defaults target the Kronos-small checkpoint):

   ```bash
   python -m kronostraining.run_training \
     --data-dir trainingdata \
     --output-dir kronostraining/artifacts \
     --lookback 64 \
     --horizon 30 \
     --validation-days 30 \
     --epochs 3
   ```

   The script requires a CUDA capable GPU. It saves checkpoints under
   `kronostraining/artifacts/checkpoints` and writes evaluation metrics to
   `kronostraining/artifacts/metrics/evaluation.json`.

3. After training, the script automatically evaluates the fine-tuned model on
   the last 30 unseen days per symbol, printing MAE, RMSE, and MAPE as well as
   aggregated scores across symbols.

## Configuration Highlights

- `KronosTrainingConfig` (see `kronostraining/config.py`) centralises all
  hyperparameters.
- `KronosMultiTickerDataset` samples sliding windows across every CSV in
  `trainingdata/`, handling feature normalisation and time-embedding creation.
- `KronosTrainer` orchestrates optimiser setup, checkpointing, and hold-out
  evaluation to keep the workflow reproducible.

All components are designed to run inside the repo without modifying the
upstream Kronos sources.

### Adapter-based fine-tuning

Set `--adapter lora` to train low-rank adapters instead of updating the full
Kronos backbone. By default the script freezes the base weights, injects LoRA
modules into the attention/FFN layers, and stores the resulting adapter at
`<output-dir>/adapters/<adapter-name>/adapter.pt`. Example:

```bash
python -m kronostraining.run_training \
  --data-dir trainingdata/AAPL \
  --output-dir kronostraining/artifacts \
  --adapter lora \
  --adapter-name AAPL \
  --adapter-r 8 \
  --adapter-alpha 16 \
  --adapter-dropout 0.05
```

Disable backbone freezing with `--no-freeze-backbone` when you deliberately
want to fine-tune the full model alongside the adapters.
