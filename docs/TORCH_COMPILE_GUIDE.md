# Torch Compile Production Guide

## Overview

This guide covers torch.compile configuration, troubleshooting, and production deployment strategies for the stock prediction models (Toto and Kronos).

## Quick Start

### Disabling torch.compile in Production

If you're experiencing recompilation issues or slowness, you can disable torch.compile:

```bash
# Disable Toto compilation
export TOTO_DISABLE_COMPILE=1

# Alternative
export MARKETSIM_TOTO_DISABLE_COMPILE=1
```

### Running Compile Stress Tests

Before deploying to production, run the integration stress test:

```bash
# Quick test (3 iterations)
python scripts/run_compile_stress_test.py --mode quick

# Full test (10 iterations)
python scripts/run_compile_stress_test.py --mode full

# Production readiness check (20 iterations, strict validation)
python scripts/run_compile_stress_test.py --mode production-check
```

## Common Issues and Solutions

### Issue 1: Excessive Recompilations

**Symptoms:**
```
W1030 09:08:09.060000 torch._dynamo hit config.recompile_limit (8)
skipping cudagraphs due to mutated inputs (2 instances)
```

**Cause:** Dynamic KV cache indices changing during inference, causing torch.compile to recompile the model multiple times.

**Solutions:**

1. **Disable torch.compile (immediate fix):**
   ```bash
   export TOTO_DISABLE_COMPILE=1
   python trade_stock_e2e.py
   ```

2. **Increase recompile limit (temporary workaround):**
   ```bash
   export TORCH_COMPILE_DEBUG=1
   export TORCHDYNAMO_RECOMPILE_LIMIT=16
   ```

3. **Use a different compile mode:**
   ```bash
   # Try reduce-overhead mode (faster compilation, may reduce recompilations)
   export TOTO_COMPILE_MODE=reduce-overhead
   ```

4. **Fix the root cause (long-term):**
   - Static KV cache allocation
   - Use `torch._dynamo.mark_dynamic()` for dynamic dimensions
   - See [KV Cache Optimization](#kv-cache-optimization) below

### Issue 2: Slow First Inference (Compilation Time)

**Symptoms:** First prediction takes 20-60 seconds, subsequent predictions are fast.

**Cause:** torch.compile is compiling the model on first run.

**Solutions:**

1. **Use persistent compilation cache:**
   ```bash
   export COMPILED_MODELS_DIR=/path/to/persistent/cache
   export TORCHINDUCTOR_CACHE_DIR=$COMPILED_MODELS_DIR/torch_inductor
   ```

2. **Warm-up the model during startup:**
   ```python
   # Run a dummy prediction to trigger compilation
   pipeline = load_toto_pipeline()
   dummy_series = np.random.randn(512)
   _ = pipeline.predict(dummy_series, prediction_length=1, num_samples=16)
   ```

3. **Use ahead-of-time compilation (experimental):**
   See [AOT Compilation](#aot-compilation) below.

### Issue 3: Compiled Model is Slower Than Eager

**Symptoms:** Compiled mode inference time > eager mode inference time.

**Cause:** Recompilation overhead exceeds performance gains, or small batch sizes don't benefit from compilation.

**Solutions:**

1. **Disable torch.compile for this workload:**
   ```bash
   export TOTO_DISABLE_COMPILE=1
   ```

2. **Increase batch size or num_samples:**
   - torch.compile benefits from larger batches
   - Try `num_samples=256` or higher

3. **Profile to identify bottlenecks:**
   ```bash
   python scripts/profile_compile_overhead.py
   ```

### Issue 4: MAE Divergence Between Compiled and Eager

**Symptoms:** Predictions differ significantly between compiled and eager modes.

**Cause:** Numerical precision differences or compilation bugs.

**Solutions:**

1. **Run stress test to quantify divergence:**
   ```bash
   python scripts/run_compile_stress_test.py --mode production-check
   ```

2. **Use float32 instead of bfloat16:**
   ```bash
   export REAL_TESTING=1  # Forces float32
   ```

3. **Report to PyTorch if divergence is significant:**
   - Document the issue
   - Provide reproducible example
   - Check PyTorch issue tracker

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TOTO_DISABLE_COMPILE` | `false` | Disable torch.compile for Toto |
| `MARKETSIM_TOTO_DISABLE_COMPILE` | `false` | Alternative disable flag |
| `TOTO_COMPILE` | auto | Explicitly enable compilation |
| `TOTO_COMPILE_MODE` | `max-autotune` | Compilation mode: `default`, `reduce-overhead`, `max-autotune` |
| `TOTO_COMPILE_BACKEND` | `inductor` | Backend: `inductor`, `aot_eager`, etc. |
| `REAL_TOTO_COMPILE_MODE` | - | Override for production |
| `REAL_TOTO_COMPILE_BACKEND` | - | Override for production |
| `COMPILED_MODELS_DIR` | `./compiled_models` | Cache directory for compiled artifacts |
| `TORCHINDUCTOR_CACHE_DIR` | `$COMPILED_MODELS_DIR/torch_inductor` | Inductor cache |
| `TORCH_COMPILE_DEBUG` | `false` | Enable debug logging |
| `TORCHDYNAMO_RECOMPILE_LIMIT` | `8` | Max recompilations before warning |

### Compile Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `default` | Balanced compilation | General use |
| `reduce-overhead` | Fast compilation, minimal overhead | Development, quick iterations |
| `max-autotune` | Aggressive optimization, slow compilation | Production, maximum performance |

## Production Deployment Strategies

### Strategy 1: Disable Compilation (Safest)

**Pros:**
- No recompilation issues
- Predictable performance
- Easier debugging

**Cons:**
- Slower inference (1-3x depending on workload)

**Configuration:**
```bash
export TOTO_DISABLE_COMPILE=1
```

### Strategy 2: Compiled with Persistent Cache (Recommended)

**Pros:**
- Fast inference after warm-up
- Compilation happens once
- Good for long-running services

**Cons:**
- Slow first prediction
- Cache management required

**Configuration:**
```bash
export COMPILED_MODELS_DIR=/var/cache/stock-prediction/compiled_models
export TORCHINDUCTOR_CACHE_DIR=$COMPILED_MODELS_DIR/torch_inductor
mkdir -p $COMPILED_MODELS_DIR
```

### Strategy 3: Hybrid (Compiled for Batch, Eager for Single)

**Pros:**
- Best of both worlds
- Optimized for each use case

**Cons:**
- More complex deployment
- Higher memory usage

**Configuration:**
```python
# In code
if batch_size > 10:
    pipeline = load_toto_pipeline(torch_compile=True)
else:
    pipeline = load_toto_pipeline(torch_compile=False)
```

## Testing and Validation

### Pre-Deployment Checklist

- [ ] Run `python scripts/run_compile_stress_test.py --mode production-check`
- [ ] Verify MAE delta < 5%
- [ ] Check for excessive recompilations (< 10 per run)
- [ ] Measure inference time improvement (compiled should be faster or disabled)
- [ ] Test with production-like data and workloads
- [ ] Monitor memory usage (compiled models may use more memory)
- [ ] Verify compilation cache is persistent and accessible

### Continuous Monitoring

Add these metrics to your monitoring:

1. **Recompilation count** - should be 0 after warm-up
2. **Inference time** - track p50, p99
3. **MAE/RMSE** - detect prediction drift
4. **Memory usage** - watch for leaks
5. **Cache hit rate** - ensure compilation cache is working

## KV Cache Optimization

The recompilation issue is primarily caused by dynamic KV cache indices. Here are potential fixes:

### Option 1: Static KV Cache Allocation (Requires Model Changes)

```python
# In toto/model/attention.py
class KVCache:
    def __init__(self, ...):
        # Pre-allocate maximum size
        self._max_idx = max_seq_len
        self._current_idx = torch.zeros(batch_size, dtype=torch.long)

    def append(self, k, v):
        # Use static indexing
        torch._dynamo.mark_static(self._current_idx)
        ...
```

### Option 2: Mark Dynamic Dimensions (Easier)

```python
# In backtest_test3_inline.py load_toto_pipeline()
if torch_compile_enabled:
    import torch._dynamo
    torch._dynamo.config.automatic_dynamic_shapes = True
    torch._dynamo.config.cache_size_limit = 64
```

### Option 3: Disable CUDA Graphs for KV Cache

```python
# In TotoPipeline
if torch_compile:
    torch._inductor.config.triton.cudagraphs = False
```

## AOT Compilation

Experimental feature to compile the model ahead of time:

```python
# scripts/aot_compile_toto.py
import torch
from src.models.toto_wrapper import TotoPipeline

pipeline = TotoPipeline.from_pretrained(
    "Datadog/Toto-Open-Base-1.0",
    device_map="cuda",
    torch_compile=True,
)

# Warm-up with various input sizes
for length in [256, 512, 1024]:
    dummy = np.random.randn(length)
    _ = pipeline.predict(dummy, prediction_length=1, num_samples=128)

print("AOT compilation complete. Cache saved to $TORCHINDUCTOR_CACHE_DIR")
```

## Debugging Tools

### Enable Debug Logging

```bash
export TORCH_COMPILE_DEBUG=1
export TORCH_LOGS="+dynamo,+inductor,+recompiles"
export TORCHDYNAMO_VERBOSE=1
```

### Profile Compilation

```bash
python -m torch.utils.collect_env  # Check PyTorch setup
python scripts/run_compile_stress_test.py --mode quick 2>&1 | tee compile_debug.log
```

### Check Recompilation Reasons

```python
import torch._dynamo
torch._dynamo.config.verbose = True
torch._dynamo.config.log_file_name = "recompile_log.txt"
```

## Performance Benchmarks

Typical performance characteristics (may vary by hardware):

| Configuration | First Inference | Subsequent Inference | Memory Usage |
|---------------|-----------------|----------------------|--------------|
| Eager | 500ms | 500ms | 650MB |
| Compiled (cold) | 25s | 200ms | 900MB |
| Compiled (warm) | 200ms | 200ms | 900MB |

**Speedup:** 2-3x for compiled (after warm-up) depending on batch size and hardware.

## Support

If you continue to experience issues:

1. Check existing issues: [torch compile issues](https://github.com/pytorch/pytorch/issues?q=is%3Aissue+torch.compile)
2. Run diagnostic: `python scripts/run_compile_stress_test.py --mode production-check`
3. Collect logs with `TORCH_COMPILE_DEBUG=1`
4. Report issue with reproducible example

## References

- [PyTorch torch.compile docs](https://pytorch.org/docs/stable/torch.compiler.html)
- [TorchDynamo troubleshooting](https://pytorch.org/docs/main/torch.compiler_troubleshooting.html)
- [TorchInductor configuration](https://pytorch.org/docs/stable/torch.compiler_api.html)
