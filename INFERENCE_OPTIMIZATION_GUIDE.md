# Inference Hyperparameter Optimization Guide

## Overview

This guide covers optimizing inference-time hyperparameters for both Toto and Kronos models to improve MAE on worst-performing stocks. We have **compiled models** for both which gives 2-5x speedup!

## Current Status

### Worst Performing Stocks (Baseline h64 MAE%)

All have both Toto and Kronos hyperparameter configs:

| Stock  | Baseline MAE% | Has Toto | Has Kronos | Notes |
|--------|--------------|----------|------------|-------|
| UNIUSD | 69.11%       | ✅       | ✅        | Extremely volatile crypto |
| QUBT   | 30.08%       | ✅       | ✅        | High volatility |
| LCID   | 26.25%       | ✅       | ✅        | EV stock, volatile |
| COIN   | 24.10%       | ✅       | ✅        | Crypto exchange |
| TSLA   | 19.13%       | ✅       | ✅        | High volatility |
| NVDA   | 15.43%       | ✅       | ✅        | Tech, volatile |
| AMD    | 14.82%       | ✅       | ✅        | Tech, volatile |

### Compiled Models Available

```
compiled_models/
├── kronos/
│   └── NeoQuasar-Kronos-base/
└── toto/
    └── Datadog-Toto-Open-Base-1.0/
```

## Inference Hyperparameters

### Toto Parameters

**Key Parameters:**
- `num_samples`: Number of trajectories to sample (512-4096)
  - More samples = more robust but slower
  - NVDA: 4096, TSLA: 2048, COIN: 1024, LCID: 1024

- `aggregate`: How to combine samples
  - `"mean"`: Simple average (good for stable stocks)
  - `"trimmed_mean_5"`: Remove top/bottom 5% (good for volatile stocks with outliers)
  - NVDA uses mean, TSLA/COIN/LCID use trimmed_mean_5

- `samples_per_batch`: Batch size for inference (64-256)
  - Trade-off between memory and speed
  - Smaller = less memory, more batches

**Current Configs:**

```json
// NVDA - Best performer of the worst
{
  "num_samples": 4096,
  "aggregate": "mean",
  "samples_per_batch": 256
}

// TSLA - Very volatile
{
  "num_samples": 2048,
  "aggregate": "trimmed_mean_5",
  "samples_per_batch": 256
}

// COIN - Crypto, volatile
{
  "num_samples": 1024,
  "aggregate": "trimmed_mean_5",
  "samples_per_batch": 128
}

// LCID - Highly volatile
{
  "num_samples": 1024,
  "aggregate": "trimmed_mean_5",
  "samples_per_batch": 64
}
```

### Kronos Parameters

**Key Parameters:**
- `temperature`: Sampling randomness (0.1-1.0)
  - Lower = more deterministic, focused on mode
  - Higher = more exploratory, diverse samples
  - NVDA: 0.24 (low, deterministic)

- `top_p`: Nucleus sampling threshold (0.8-0.95)
  - Sample from smallest set of tokens with cumulative prob >= top_p
  - NVDA: 0.88

- `top_k`: Top-k sampling (0 = disabled, 50-200 typical)
  - Sample from top k most likely tokens
  - NVDA: 0 (disabled)

- `sample_count`: Number of trajectories (64-512)
  - More samples = more robust
  - NVDA: 256

- `max_context`: Context window size (192-512)
  - More context = better understanding but slower
  - NVDA: 192

- `clip`: Return clipping threshold (1.0-5.0)
  - Limits extreme returns
  - NVDA: 1.8

**Current Config (NVDA):**

```json
{
  "temperature": 0.24,
  "top_p": 0.88,
  "top_k": 0,
  "sample_count": 256,
  "max_context": 192,
  "clip": 1.8
}
```

## Optimization Strategies

### For Volatile Stocks (UNIUSD, QUBT, TSLA, COIN, LCID)

**Toto:**
1. **Use trimmed_mean_5 or trimmed_mean_10** aggregation
   - Removes outlier predictions
   - More robust to extreme samples

2. **Increase num_samples** (2048-8192)
   - More samples improve ensemble quality
   - Compiled model makes this faster!

3. **Test median aggregation**
   - Even more robust to outliers than trimmed mean

**Kronos:**
1. **Lower temperature** (0.15-0.3)
   - Less randomness for volatile stocks
   - More focused on modal prediction

2. **Increase sample_count** (256-512)
   - Better ensemble averaging

3. **Higher max_context** (384-512)
   - More historical data for pattern recognition

4. **Lower clip** (1.0-2.0)
   - Limit extreme return predictions

### For Semi-Volatile (NVDA, AMD)

**Toto:**
1. **Try both mean and trimmed_mean_5**
   - Test which works better

2. **Optimize num_samples** (2048-4096)
   - Balance speed vs accuracy

**Kronos:**
1. **Fine-tune temperature** (0.2-0.35)
   - Find sweet spot

2. **Test top_k** (50-100)
   - May help vs pure nucleus sampling

## Running Inference Tests

### Using test_kronos_vs_toto.py

**Test with stored hyperparams:**
```bash
# Single stock, 64-step forecast
uv run python test_kronos_vs_toto.py \
  --symbol NVDA \
  --forecast-horizon 64 \
  --use-stored-hyperparams

# Compare multiple configurations
uv run python test_kronos_vs_toto.py \
  --symbol TSLA \
  --forecast-horizon 64
```

**Test custom hyperparams:**
```bash
# Set environment variables
export TOTO_NUM_SAMPLES=8192
export TOTO_AGGREGATE="trimmed_mean_10"
export KRONOS_TEMPERATURE=0.2
export KRONOS_SAMPLE_COUNT=512

uv run python test_kronos_vs_toto.py --symbol COIN --forecast-horizon 64
```

### Using compare_toto_vs_kronos.py

```bash
# Compare on multiple stocks
uv run python tototraining/compare_toto_vs_kronos.py \
  --stocks NVDA AMD TSLA COIN LCID \
  --forecast-horizon 64
```

## Optimization Workflow

### 1. Baseline Test
```bash
# Test current configs
for stock in NVDA AMD TSLA COIN LCID QUBT UNIUSD; do
  echo "Testing $stock..."
  uv run python test_kronos_vs_toto.py \
    --symbol $stock \
    --forecast-horizon 64 \
    --use-stored-hyperparams
done
```

### 2. Explore Aggregation Methods (Toto)
```bash
# Test different aggregations for volatile stocks
for agg in "mean" "trimmed_mean_5" "trimmed_mean_10" "median"; do
  export TOTO_AGGREGATE=$agg
  uv run python test_kronos_vs_toto.py --symbol TSLA --forecast-horizon 64
done
```

### 3. Optimize Sample Counts
```bash
# Test sample count sweep
for samples in 1024 2048 4096 8192; do
  export TOTO_NUM_SAMPLES=$samples
  uv run python test_kronos_vs_toto.py --symbol NVDA --forecast-horizon 64
done
```

### 4. Temperature Sweep (Kronos)
```bash
# Find optimal temperature
for temp in 0.15 0.20 0.25 0.30 0.35; do
  export KRONOS_TEMPERATURE=$temp
  uv run python test_kronos_vs_toto.py --symbol COIN --forecast-horizon 64
done
```

## Expected Improvements

Based on similar optimization work:

- **Aggregation optimization**: 5-15% MAE improvement on volatile stocks
- **Sample count tuning**: 10-20% improvement (with compiled models, this is fast!)
- **Temperature/top_p tuning**: 5-10% improvement
- **Combined optimizations**: 20-35% total MAE reduction possible

## Leveraging Compiled Models

The compiled models in `compiled_models/` provide:
- **2-5x faster inference** vs non-compiled
- **Enables larger sample counts** without timeout
- **Better batch utilization**

To use compiled models, the wrappers automatically detect and load them from:
```
compiled_models/toto/Datadog-Toto-Open-Base-1.0/
compiled_models/kronos/NeoQuasar-Kronos-base/
```

## Next Steps

1. **Run baseline tests** on all 7 worst performers
2. **Identify which model (Toto vs Kronos) performs better** for each stock
3. **Optimize the better model** for each stock
4. **Save optimized configs** back to `hyperparams/toto/` and `hyperparams/kronos/`
5. **Re-run full comparison** to validate improvements

## Key Insights

### Volatile vs Stable Stocks

**Volatile stocks benefit from:**
- Trimmed aggregation (removes outliers)
- More samples (better ensemble)
- Lower temperature (less randomness)
- Return clipping (limit extremes)

**Stable stocks benefit from:**
- Mean aggregation (all samples matter)
- Fewer samples (faster, still accurate)
- Moderate temperature (some exploration)
- More context (capture subtle patterns)

### Model Selection Per Stock

From existing configs, we see:
- **NVDA/AMD**: Using high sample counts with mean aggregation
- **TSLA/COIN/LCID**: Using trimmed_mean_5 (acknowledging volatility)
- **Pattern**: More volatile = more trimming, more samples

## Quick Reference

### Toto Optimization Commands
```bash
# High-quality but slow
export TOTO_NUM_SAMPLES=8192
export TOTO_AGGREGATE="trimmed_mean_10"
export TOTO_SAMPLES_PER_BATCH=256

# Balanced
export TOTO_NUM_SAMPLES=2048
export TOTO_AGGREGATE="trimmed_mean_5"
export TOTO_SAMPLES_PER_BATCH=128

# Fast but less robust
export TOTO_NUM_SAMPLES=512
export TOTO_AGGREGATE="mean"
export TOTO_SAMPLES_PER_BATCH=64
```

### Kronos Optimization Commands
```bash
# Conservative (for volatile)
export KRONOS_TEMPERATURE=0.2
export KRONOS_TOP_P=0.85
export KRONOS_SAMPLE_COUNT=512
export KRONOS_CLIP=1.5

# Balanced
export KRONOS_TEMPERATURE=0.25
export KRONOS_TOP_P=0.90
export KRONOS_SAMPLE_COUNT=256
export KRONOS_CLIP=2.5

# Exploratory (for stable)
export KRONOS_TEMPERATURE=0.35
export KRONOS_TOP_P=0.95
export KRONOS_SAMPLE_COUNT=128
export KRONOS_CLIP=5.0
```
