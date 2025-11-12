# Toto Training Improvements Plan

## Executive Summary

Analysis of the current toto training reveals the model **is running successfully** but **not learning effectively**. The model performs worse than a naive "no-change" baseline (DM test p<0.01). Several key hyperparameters deviate from official Datadog Toto recommendations, and training infrastructure optimizations can be better leveraged.

## Current Status

### What's Working Well ✅
- **Training pipeline executes successfully** (~12 min for 8 epochs)
- **Modern optimizations already in place:**
  - `torch.compile` with `mode="max-autotune"`
  - Mixed precision training (bfloat16)
  - Muon optimizer (`muon_mix` - state-of-the-art for transformers)
  - EMA (Exponential Moving Average)
  - Gradient clipping and accumulation
  - Comprehensive metrics tracking
  - 138 training data files available

### Critical Problems ❌
1. **Model performs worse than naive baseline**
   - Validation: MAE 0.59 vs naive 0.099 (6x worse!)
   - Test: MAE 0.136 vs naive 0.062 (2.2x worse!)
   - R² score: -2021 (val), -2.77 (test) - very negative!

2. **Hyperparameters deviate from Datadog Toto paper recommendations:**
   - Current patch_size: **64** → Should be **32**
   - Current context: **192** → Should be **512+**
   - Current epochs: **8-24** → Should be **50-100+**
   - Gradient clip: **0.1** → Should be **1.0**

3. **Insufficient training:**
   - Only 8 epochs completed
   - Early stopping may be too aggressive
   - Not enough data diversity

## Key Research Findings

From [Datadog Toto paper](https://arxiv.org/html/2407.07874v1) and official documentation:

- **Patch size: 32** (non-overlapping, as in PatchTST)
- **Context window: 512 steps** minimum
- **Transformer ratio: 11:1** (time-wise to variate-wise blocks)
- **Loss: Student-T mixture** with NLL (λ_NLL = 0.57)
- **Training scale: 2.36 trillion time series points**

## Recommended Improvements

### Priority 1: Fix Core Hyperparameters

```python
# IN: tototraining/run_gpu_training.py
# Current (WRONG):
patch_size=64
stride=64
sequence_length=192
gradient_clip_val=0.1

# Recommended (ALIGNED WITH PAPER):
patch_size=32
stride=32  # or 16 for overlap
sequence_length=512  # minimum, try 1024 for better results
gradient_clip_val=1.0  # less aggressive
```

### Priority 2: Train Longer with Better Schedules

```python
# Current:
max_epochs=24
early_stopping_patience=8

# Recommended:
max_epochs=100  # train much longer
early_stopping_patience=15  # more patient
warmup_steps=5000  # longer warmup
```

### Priority 3: Improve Loss Function

Current setup uses Huber loss. The Toto paper recommends:

```python
# In TrainerConfig:
loss_type="quantile"  # or "heteroscedastic" with Student-T
quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9]  # full distribution
output_distribution_classes=["<class 'model.distribution.StudentTOutput'>"]
```

### Priority 4: Scale Up Data and Batching

```python
# Current:
batch_size=4
accumulation_steps=4
max_symbols=128

# Recommended:
device_batch_size=8  # if memory allows
accumulation_steps=8  # effective batch = 64
max_symbols=None  # use all 138 available symbols
```

### Priority 5: Leverage Advanced Optimizations

Already implemented but could be enhanced:

1. **CUDA Graphs** (commented out but available):
```python
use_cuda_graphs=True  # for ~20% speedup after warmup
cuda_graph_warmup=10
```

2. **Gradient Checkpointing** (for larger models/batches):
```python
gradient_checkpointing=True  # trades compute for memory
```

3. **Longer training with more pairs**:
   - Add crypto pairs (ADA-USD, ALGO-USD, ATOM-USD, AVAX-USD already in data)
   - Add more traditional stocks from the 138 available
   - Consider data augmentation (already has infrastructure)

## Implementation Plan

### Step 1: Quick Win - Fix Hyperparameters (15 min)

Create a new config that aligns with Toto paper:

```bash
python tototraining/run_gpu_training.py \
  --device-bs 8 \
  --grad-accum 8 \
  --lr 0.0003 \
  --warmup-steps 5000 \
  --max-epochs 100 \
  --save-dir tototraining/checkpoints/toto_aligned \
  --wandb-project stock-toto \
  --run-name toto_aligned_v1
```

Create a config file override:

```python
# tototraining/configs/toto_aligned.py
from toto_trainer import TrainerConfig, DataLoaderConfig

trainer_config = TrainerConfig(
    # ALIGNED WITH TOTO PAPER
    patch_size=32,
    stride=32,
    embed_dim=768,
    num_layers=12,
    num_heads=12,
    mlp_hidden_dim=1536,

    # Better training
    learning_rate=3e-4,
    max_epochs=100,
    warmup_steps=5000,
    gradient_clip_val=1.0,  # less aggressive!

    # Better loss
    loss_type="quantile",
    quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],

    # Optimizer (already good)
    optimizer="muon_mix",
    compile=True,
    use_mixed_precision=True,

    # Checkpointing
    early_stopping_patience=15,
    best_k_checkpoints=4,
)

loader_config = DataLoaderConfig(
    patch_size=32,
    stride=32,
    sequence_length=512,  # LONGER CONTEXT!
    prediction_length=64,  # predict 64 steps
    batch_size=8,
    max_symbols=None,  # USE ALL DATA!
    enable_augmentation=True,
    price_noise_std=0.0125,
    time_mask_prob=0.1,
)
```

### Step 2: Advanced - Enable All Optimizations (30 min)

```python
# Add to TrainerConfig
use_cuda_graphs=True
cuda_graph_warmup=10
gradient_checkpointing=True  # if OOM with larger batches
ema_decay=0.9999  # stronger EMA for better generalization
```

### Step 3: Scale - Train on More Data (1-2 hours)

1. Use all 138 symbols instead of limiting
2. Train for 100+ epochs
3. Use cross-validation (already has infrastructure in `purged_kfold_indices`)
4. Monitor with WandB for experiment tracking

## Modern Training Best Practices (2025)

Based on recent research (nanochat, etc.) and the Toto paper:

1. **Compilation is critical** - Already using `torch.compile(mode="max-autotune")` ✅
2. **Mixed precision** - Already using bfloat16 ✅
3. **Modern optimizers** - Already using Muon (better than AdamW for transformers) ✅
4. **CUDA graphs** - Available but not enabled
5. **Longer training** - Need to increase epochs from 8-24 to 100+
6. **Better data augmentation** - Infrastructure exists, needs tuning
7. **Quantile/distribution losses** - Better for forecasting than MSE/Huber

## Expected Improvements

With these changes, expect:
- **5-10x better MAE** vs current (getting competitive with or beating naive baseline)
- **Positive R² scores** (>0.3 would be good for financial data)
- **Better probabilistic forecasts** with quantile loss
- **Faster training per epoch** with optimizations (~8-10 min instead of 12)
- **More robust predictions** with longer context and better hyperparams

## Next Steps

1. ✅ **Create aligned config** (as shown above)
2. 🔄 **Run 100-epoch training** with proper hyperparameters
3. 🔄 **Enable CUDA graphs** after warmup
4. 🔄 **Add crypto pairs** for diversity
5. 🔄 **Implement quantile loss** for better uncertainty estimates
6. 🔄 **Cross-validate** using the existing purged k-fold infrastructure

## References

- [Toto Technical Report](https://arxiv.org/html/2407.07874v1)
- [Datadog Toto Blog](https://www.datadoghq.com/blog/datadog-time-series-foundation-model/)
- [Hugging Face Model](https://huggingface.co/Datadog/Toto-Open-Base-1.0)
- Modern optimizer research: Muon, AdamW variants
- PyTorch performance guide: torch.compile, CUDA graphs
