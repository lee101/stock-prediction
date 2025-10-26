PufferLib RL Training (Amazon Toto Enhanced)
============================================

Overview
--------

- Multi-stage training pipeline for portfolio RL:
  1. **Generic forecaster** trained on all equities in `trainingdata/` with Amazon Toto features.
  2. **Per-stock specialists** fine-tuned on individual tickers.
  3. **Differentiable portfolio RL** that allocates across stock pairs with leverage-aware profit.
- Uses the new multi-asset Gymnasium environment backed by Torch tensors, enforcing 2× leverage limits and 6.75 % annual financing costs.

Installation
------------

- Python 3.10+ with PyTorch (GPU optional).
- From the repo root run: `uv pip install -r requirements.txt`
- Optional: verify GPU availability  
  `python -c "import torch; print('CUDA:', torch.cuda.is_available())"`

Data
----

- Place raw OHLCV CSVs under `trainingdata/` (one file per symbol) or provide a custom folder via `--trainingdata-dir`.
- If files already live in `tototraining/trainingdata/train`, the trainer discovers them automatically.
- The pipeline augments each asset with Toto forecasts (falling back to statistical features if Toto is unavailable).

Quick Start
-----------

Run the full pipeline on five base symbols, fine-tune AAPL/AMZN/MSFT, and train portfolio RL on adjacent pairs:

```
python pufferlibtraining/train_ppo.py \
  --base-stocks AAPL,AMZN,MSFT,NVDA,GOOGL \
  --specialist-stocks AAPL,AMZN,MSFT \
  --trainingdata-dir trainingdata \
  --output-dir pufferlibtraining/models \
  --tensorboard-dir pufferlibtraining/logs \
  --wandb-project pufferlib \
  --wandb-entity stock
```

The `--wandb-*` switches let you redirect runs into dedicated projects (for example `stock/pufferlib` for RL or `stock/hftraining` for the supervised stacks) while the logger continues to write TensorBoard events locally.

Key Outputs
-----------

- Base, specialist, and portfolio checkpoints land in `pufferlibtraining/models/`.
- TensorBoard logs are written to `pufferlibtraining/logs/`.
- A JSON summary (`pipeline_summary.json`) captures all checkpoints, metrics, and configuration.

Environment Highlights
----------------------

- Supports arbitrarily many assets per episode; observations include Toto features, allocations, balance ratios, and leverage.
- Enforces 2× gross exposure, charges 6.75 % annualised borrowing costs, and tracks per-trade net profit for downstream analytics.
- Rewards are computed with differentiable Torch operations, making the setup compatible with gradient-based optimisation outside of standard RL loops.
