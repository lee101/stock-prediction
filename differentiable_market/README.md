# Differentiable Market RL

## Overview

`differentiable_market` provides an end-to-end differentiable OHLC market simulator,
GRPO-style policy trainer, and backtesting utilities designed for fast iteration.
The core components are:

- Differentiable environment with smooth turnover and risk penalties (`env.py`).
- Dirichlet-based GRU policy that emits simplex-constrained portfolio weights (`policy.py`).
- GRPO training loop with Muon/AdamW optimizers, `torch.compile`, bf16 autocast, and
  EMA-stabilised reference policy (`trainer.py`).
- Evaluation backtester that replays checkpoints on real OHLC data and writes summary
  reports plus optional trade logs (`marketsimulator/backtester.py`).

## Quick Start

All dependency management is handled through `uv`. Sync the environment after adding
the package entry in `pyproject.toml`:

```bash
uv sync
```

### Training

```bash
uv run python -m differentiable_market.train \
  --data-root trainingdata \
  --lookback 192 \
  --batch-windows 128 \
  --rollout-groups 4 \
  --epochs 2000
```

Options of interest:

- `--device` / `--dtype` for hardware overrides.
- `--no-muon` and `--no-compile` to disable Muon or `torch.compile` when debugging.
- `--save-dir` to control where run folders and checkpoints are written.
- `--microbatch-windows` and `--gradient-checkpointing` help keep peak VRAM near a target (e.g., 10 GB on an RTX 3090) while retaining large effective batch sizes.
- `--risk-aversion` and `--drawdown-lambda` tune turnover/variance penalties and add a differentiable max drawdown term to the objective when you need tighter risk control.
- `--include-cash` appends a cash asset (zero return) so the policy can explicitly park capital when risk penalties bite.

Each run produces `<save-dir>/<timestamp>/` containing `metrics.jsonl`,
`config.json`, and checkpoints (`checkpoints/latest.pt`, `checkpoints/best.pt`).

### Backtesting / Evaluation

```bash
uv run python -m differentiable_market.marketsimulator.run \
  --checkpoint differentiable_market/runs/<timestamp>/checkpoints/best.pt \
  --window-length 256 \
  --stride 64
```

The backtester writes aggregated metrics to `differentiable_market/evals/report.json`
and per-window metrics to `windows.json`. Trade logs (`trades.jsonl`) are optional and
can be disabled with `--no-trades`.

Training metrics now include `peak_mem_gb`, `microbatch`, and `windows` to make it easy
to verify the effective batch size and GPU memory footprint.

## Testing

Unit tests cover data ingestion, training loop plumbing, and the evaluation pipeline.
Run them with:

```bash
uv run pytest tests/differentiable_market -q
```

Synthetic OHLC fixtures ensure tests remain fast and deterministic while exercising
the full training/backtesting flow.
