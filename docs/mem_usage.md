# GPU Memory Usage Cheatsheet

This reference summarises the automatic batch-size heuristics introduced for the
training and inference pipelines. The defaults target 24 GiB GPUs (RTX 3090/4090 class)
while staying conservative on smaller cards. Adjust thresholds in
`src/gpu_utils.py` if you collect better telemetry.

## Autotune Overview

- Detection uses `src.gpu_utils.detect_total_vram_bytes`, preferring PyTorch and
  falling back to NVIDIA NVML when available.
- Each pipeline keeps manual overrides: passing the corresponding CLI flag (for
  example `--batch-size` or `--rl-batch-size`) disables automatic increases but
  still allows protective down-scaling to avoid OOMs.
- HuggingFace configs gained `system.auto_batch_size` (default `True`) and
  `training.max_auto_batch_size` for tighter caps. Set `system.auto_batch_size = False`
  to keep a fixed batch size.
- Inference defaults (Kronos sample count) now scale with VRAM; set
  `MARKETSIM_KRONOS_SAMPLE_COUNT` to force a specific value.

## Recommended Values (24 GiB GPUs)

| Pipeline | Setting | Autotuned target | Notes |
| --- | --- | --- | --- |
| `tototraining/train.py` | `--batch-size` | **4** | Dynamic windowing also caps oversized buckets to honour user-requested window sizes. |
| `hftraining/run_training.py` | `training.batch_size` | **24** | Applies when `system.auto_batch_size` is enabled and the batch size has not been explicitly overridden. |
| `pufferlibtraining/train_ppo.py` | `--base-batch-size` | **48** | CLI override keeps the requested value unless it exceeds the safe threshold. |
| `pufferlibtraining/train_ppo.py` | `--rl-batch-size` | **128** | Ensures PPO rollouts remain GPU bound without frequent OOM recoveries. |
| `predict_stock_forecasting.py` | Kronos `sample_count` | **48** | Adjusted automatically at import; environment variable overrides still win. |

## Manual Overrides

- **Toto training**: run with `--batch-size` to enforce a specific value.
- **HF training**: set `config.system.auto_batch_size = False` or `config.training.max_auto_batch_size`
  before calling `run_training`. CLI `--batch_size` also prevents upward scaling.
- **PufferLib PPO**: pass `--base-batch-size` / `--rl-batch-size` for manual control.
- **Kronos inference**: export `MARKETSIM_KRONOS_SAMPLE_COUNT`.

These heuristics are conservative starting points. Capture telemetry from your
next production run and fine-tune the threshold tables if you can sustain
larger batches without paging.
