Optimization Ideas for HFTraining (GPU + Pipeline)

Goals
- Maximize GPU utilization and throughput while keeping training stable.
- Reduce dataloader and CPU bottlenecks so the GPU stays busy.
- Choose model/training settings that match your hardware.

What’s instrumented now
- Step time and samples/sec via in-loop timers.
- CUDA memory (allocated/reserved/max) via torch.cuda APIs.
- Optional GPU utilization and memory used via nvidia-smi if available.
- Metrics logged to TensorBoard under train/* and system/*.

How to benchmark
- Run quick test: `python hftraining/run_training.py --config_type quick_test --debug`
- Open TensorBoard: `tensorboard --logdir hftraining/logs`
- Watch: train/step_time_s, train/samples_per_sec, system/gpu_memory_allocated_mb, system/gpu_utilization_pct.

Auto-tuning (meta optimizer)
- Enable the auto tuner to pick a batch size that maximizes throughput under memory limits:
  - Set env var: `AUTO_TUNE=1` for a one-off run, or
  - Programmatically: set `HFTrainingConfig.auto_tune = True` and optionally `HFTrainingConfig.tuning_steps` and `HFTrainingConfig.target_effective_batch_size`.
- The tuner tries nearby batch sizes (0.5x, 1x, 2x, 4x), measures short forward/backward steps, and selects the best throughput.
- If `target_effective_batch_size` is set, it adjusts `gradient_accumulation_steps` to approximate that target.

High‑impact optimizations
- Batch size and accumulation
  - Increase `training.batch_size` until GPU memory is close to full, then use `training.gradient_accumulation_steps` to simulate larger batches.
  - Target 85–95% memory usage during the steady state of training.

- Mixed precision and TF32
  - Keep `training.use_mixed_precision=true` on NVIDIA GPUs; expect 1.5–2.5x speedups.
  - If using Ampere+ GPUs, enable TF32 for matmul/convs: `torch.backends.cuda.matmul.allow_tf32 = True` and `torch.backends.cudnn.allow_tf32 = True` (safe for training most transformer workloads).

- Model dimensions
  - Ensure `hidden_size % num_heads == 0` to avoid hidden reshapes and underutilized kernels.
  - Prefer multiples of 64/128 for `hidden_size` on NVIDIA GPUs; avoids odd kernel shapes.

- Sequence length and horizon
  - Self‑attention is O(L^2). Keep `data.sequence_length` as small as accuracy allows.
  - Consider downsampling or windowing strategies if you need long context.

- Dataloader pipeline
  - Use more workers (`training.dataloader_num_workers`) until CPU saturates; start with num_cores/2.
  - Enable `dataloader_pin_memory=true` and use `.to(device, non_blocking=True)` when moving tensors.
  - Precompute heavy feature engineering offline; cache normalized features.

- Memory pressure
  - Gradient checkpointing (`training.gradient_checkpointing=true`) reduces activation memory at small compute cost.
  - Clip gradients (`training.max_grad_norm`) to improve stability at larger batch sizes.

- Optimizer and scheduler
  - Start with AdamW or Lion; tune LR with cosine warmup. Keep `weight_decay` modest (0.01–0.05).
  - Check `train/learning_rate` and loss curves; if noisy, increase warmup or reduce initial LR.

- Logging and checkpoints
  - Set `evaluation.logging_steps` to a coarser interval (e.g., 50–200) to reduce host overhead.
  - Save less frequently (e.g., `evaluation.save_steps=1000+`) during long runs.

Advanced tweaks
- Fused ops and kernels
  - If using PyTorch nightly / NVIDIA libs, enable fused optimizers or bias+activation fusions (Apex/FlashAttention when applicable).
  - Consider FlashAttention for long sequences if you adopt compatible blocks.

- Data layout and memory format
  - Keep tensors contiguous on creation; avoid excessive `.contiguous()` in hot paths.
  - Use channels‑last (float16/bfloat16) mainly for CNNs; for transformers default format is fine.

- Profiling
  - Use `torch.profiler` for a brief window (50–100 steps) to locate bottlenecks; keep disabled by default.
  - Record start/end step timestamps to compute variance in step time; large swings often indicate dataloader stalls.

Operational tips
- Watch for GPU idle
  - Low `system/gpu_utilization_pct` with high CPU load suggests dataloader bottlenecks.
  - If utilization dips at intervals matching eval/checkpoint steps, cache eval batches and checkpoint less often.

- Stability at scale
  - If loss spikes appear when increasing batch size, lower LR or increase warmup.
  - Use `label_smoothing` (0.05–0.1) for classification heads to reduce overconfidence.

Suggested configs by hardware
- 1x mid‑range GPU (e.g., 12–16 GB)
  - batch_size: 16–64, gradient_accumulation_steps: 1–4
  - hidden_size: 256–512, num_layers: 4–8, num_heads: 4–8, sequence_length: 30–60

- 1x high‑end GPU (e.g., 24–48 GB)
  - batch_size: 64–256, gradient_accumulation_steps: 1–2
  - hidden_size: 512–768, num_layers: 8–12, num_heads: 8–12, sequence_length: 60–120

How to act on metrics
- If samples/sec is flat but utilization is low: increase num_workers, enable pin_memory, reduce Python work in collate.
- If memory is near OOM: enable gradient checkpointing; reduce batch size or sequence length.
- If step_time spikes: throttle logging/eval frequency; inspect I/O and augmentation hotspots.
