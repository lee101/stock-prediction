# Toto Training Quick Start Guide

## TL;DR - Start Training in 30 Seconds

```bash
# Quick 10-epoch test run (to verify everything works):
python tototraining/run_improved_training.py --max-epochs 10 --run-name quick_test

# Full 100-epoch training run (recommended):
python tototraining/run_improved_training.py --max-epochs 100 --run-name full_training_v1

# With WandB logging (for experiment tracking):
python tototraining/run_improved_training.py \
    --max-epochs 100 \
    --wandb-project stock-toto \
    --run-name experiment_v1
```

## What Was Fixed?

The previous training had **critical misconfigurations** that made the model perform **6x worse than a naive baseline**. Here's what was fixed:

| Setting | ❌ Old (Bad) | ✅ New (Fixed) | Why It Matters |
|---------|--------------|----------------|----------------|
| Patch Size | 64 | **32** | Toto paper uses 32; 64 loses too much temporal resolution |
| Context Length | 192 | **512+** | Paper recommends 512+; 192 is too short for patterns |
| Gradient Clip | 0.1 | **1.0** | 0.1 was WAY too aggressive, preventing learning |
| Training Epochs | 8-24 | **100+** | Need longer training for convergence |
| Loss Function | Huber only | **Quantile** | Better for forecasting uncertainty |
| Effective Batch | 16 | **64** | Larger batches = more stable gradients |

**Result:** Previous model had R² = -2021 (worse than random). New config should achieve **R² > 0** and **beat naive baseline**.

## Training Options

### Basic Usage

```bash
# Default (recommended):
python tototraining/run_improved_training.py

# Custom epochs:
python tototraining/run_improved_training.py --max-epochs 50

# Custom batch size:
python tototraining/run_improved_training.py --device-bs 16 --grad-accum 4

# Resume training:
python tototraining/run_improved_training.py --resume

# Resume from specific checkpoint:
python tototraining/run_improved_training.py --resume-from /path/to/checkpoint.pt
```

### Advanced Options

```bash
# Enable CUDA graphs (20% faster after warmup):
python tototraining/run_improved_training.py --enable-cuda-graphs

# Enable gradient checkpointing (for larger batches if OOM):
python tototraining/run_improved_training.py --gradient-checkpointing

# Change context/prediction length:
python tototraining/run_improved_training.py --context-length 1024 --pred-length 128

# Disable torch.compile (if issues):
python tototraining/run_improved_training.py --no-compile

# Use traditional Huber loss instead of quantile:
python tototraining/run_improved_training.py --no-quantile-loss

# Train on limited symbols (for testing):
python tototraining/run_improved_training.py --max-symbols 20
```

## Performance Expectations

### GPU Training (Recommended)
- **Speed:** ~10-12 min per epoch (8 epochs)
- **Memory:** ~8-12 GB VRAM
- **Speedup:** ~30x faster than CPU

### CPU Training (Not Recommended)
- **Speed:** ~4-6 hours per epoch
- **Memory:** 16+ GB RAM
- **Use case:** Only for testing on laptop

## Monitoring Training

### Check Progress in Real-Time

```bash
# Watch log file:
tail -f tototraining/checkpoints/improved/latest/training.log

# Monitor GPU usage:
nvidia-smi -l 1

# Check tensorboard (if enabled):
tensorboard --logdir tototraining/checkpoints/improved/latest/tensorboard
```

### Key Metrics to Watch

**During Training:**
- **Loss:** Should decrease steadily
- **pct_mae:** Should decrease below 1.0
- **price_mae:** Should decrease below naive_mae
- **Learning Rate:** Should follow cosine schedule

**After Training:**
- **Validation R²:** Should be **> 0** (negative means worse than naive)
- **Test MAE vs Naive MAE:** Model should **beat** naive
- **DM p-value:** Should be **< 0.05** for statistical significance

## Interpreting Results

### Good Training Run ✅
```
Validation Metrics:
  loss: 0.0015
  pct_mae: 0.42
  price_mae: 0.08
  naive_mae: 0.09
  pct_r2: 0.35
  dm_pvalue_vs_naive: 0.001  ← significantly better than naive!
```

### Bad Training Run ❌
```
Validation Metrics:
  loss: 0.0095
  pct_mae: 0.95
  price_mae: 0.59
  naive_mae: 0.09  ← model 6x worse than naive!
  pct_r2: -2021    ← extremely negative R²
  dm_pvalue_vs_naive: 0.0
```

## Optimizations Already Enabled

Your training automatically uses:

1. **✅ torch.compile** (mode="max-autotune") - ~20% faster
2. **✅ Mixed precision** (bfloat16/float16) - 2x faster, less memory
3. **✅ Muon optimizer** (muon_mix) - state-of-the-art for transformers
4. **✅ EMA** (Exponential Moving Average) - better generalization
5. **✅ Gradient accumulation** - simulate larger batches
6. **✅ Data augmentation** - better robustness
7. **✅ Prefetching** - minimize data loading bottleneck

## Optional Advanced Optimizations

### CUDA Graphs (Experimental)
Adds ~20% speedup but requires:
- Fixed input shapes
- No dynamic control flow
- Warmup phase

```bash
python tototraining/run_improved_training.py --enable-cuda-graphs
```

### Gradient Checkpointing
Trades compute for memory (use if OOM):

```bash
python tototraining/run_improved_training.py --gradient-checkpointing
```

### Longer Context for Better Results
```bash
python tototraining/run_improved_training.py --context-length 1024
```

## Data Information

### Available Data
- **138 symbols** in `trainingdata/train/`
- Mix of stocks (AAPL, AMZN, NVDA, etc.) and crypto (BTC, ETH, ADA, etc.)
- Hourly OHLCV data
- By default: trains on **all symbols** for maximum diversity

### Limiting Symbols (for Testing)
```bash
# Train on 20 symbols only:
python tototraining/run_improved_training.py --max-symbols 20
```

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size:
python tototraining/run_improved_training.py --device-bs 4 --grad-accum 16

# Or enable gradient checkpointing:
python tototraining/run_improved_training.py --gradient-checkpointing
```

### Training Too Slow
```bash
# Ensure CUDA is available:
python -c "import torch; print(torch.cuda.is_available())"

# Enable CUDA graphs:
python tototraining/run_improved_training.py --enable-cuda-graphs

# Reduce context length:
python tototraining/run_improved_training.py --context-length 256
```

### Model Not Learning
- Check that loss is decreasing (not stuck)
- Ensure data is loading correctly (check log for dataset sizes)
- Try reducing learning rate: `--lr 0.0001`
- Increase warmup: `--warmup-steps 10000`

## Next Steps After Training

1. **Evaluate best checkpoint:**
   ```python
   from toto_trainer import TotoTrainer
   trainer.load_checkpoint("tototraining/checkpoints/improved/latest/best/rank1_*.pt")
   test_metrics = trainer.evaluate("test")
   ```

2. **Use for inference:**
   The model is automatically exported to HuggingFace format in:
   `tototraining/checkpoints/improved/latest/hf_export/`

3. **Compare with kronos/other models:**
   Run comparative evaluation scripts

## Configuration Files

All training parameters are in:
- **Training config:** `tototraining/run_improved_training.py` (TrainerConfig)
- **Data config:** `tototraining/run_improved_training.py` (DataLoaderConfig)
- **Logs:** `tototraining/checkpoints/improved/latest/training.log`
- **Checkpoints:** `tototraining/checkpoints/improved/latest/*.pt`

## Recommended Training Pipeline

```bash
# 1. Quick sanity check (10 epochs, ~2 hours):
python tototraining/run_improved_training.py \
    --max-epochs 10 \
    --run-name sanity_check

# 2. If results look good, full training (100 epochs, ~20 hours):
python tototraining/run_improved_training.py \
    --max-epochs 100 \
    --wandb-project stock-toto \
    --run-name full_v1 \
    --enable-cuda-graphs

# 3. Experiment with different settings:
python tototraining/run_improved_training.py \
    --max-epochs 100 \
    --context-length 1024 \
    --pred-length 128 \
    --device-bs 4 \
    --grad-accum 16 \
    --run-name experiment_long_context
```

## References

- **Full improvement plan:** `docs/TOTO_TRAINING_IMPROVEMENTS.md`
- **Toto paper:** https://arxiv.org/html/2407.07874v1
- **Original script:** `tototraining/run_gpu_training.py` (old)
- **Improved script:** `tototraining/run_improved_training.py` (new)
