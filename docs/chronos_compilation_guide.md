# Chronos2 torch.compile Configuration Guide

## Overview

Chronos2 supports PyTorch's `torch.compile()` for potential performance improvements. However, **compilation is disabled by default** to ensure maximum stability and avoid numerical issues.

This guide explains:
- Why compilation is disabled by default
- How to safely enable compilation
- What settings have been tested
- How to troubleshoot compilation issues

## Quick Start

### Default (Recommended for Production)

By default, Chronos2 runs in **eager mode** (no compilation):

```bash
# Default: compilation disabled
python trade_stock_e2e.py
```

### Enable Compilation

To enable compilation with safe settings:

```bash
export TORCH_COMPILED=1
python trade_stock_e2e.py
```

Or use the configuration helper in code:

```python
from src.chronos_compile_config import apply_production_compiled

apply_production_compiled()
```

## Why Compilation is Disabled by Default

### Numerical Stability Issues

torch.compile can introduce numerical instabilities:

1. **Very small values**: PyTorch's sympy evaluation in torch._inductor can fail with very small values (both positive and negative < 1e-3) during symbolic math operations
2. **NaN/Inf propagation**: Compilation can change how numerical errors propagate
3. **SDPA backend issues**: Scaled Dot Product Attention backends (Flash, Memory-Efficient) can cause crashes or incorrect results
4. **Dtype precision**: Mixed precision (fp16, bf16) often causes issues

### Production Reliability

- Eager mode is more predictable and debuggable
- Compilation adds warmup overhead on first run
- Compiled models can fail in ways that are hard to diagnose
- Fallback mechanisms add complexity

### Testing Results

Extensive fuzzing tests (`tests/test_chronos2_compile_fuzzing.py`) show:
- ✅ Eager mode: 100% success rate across all test scenarios
- ⚠️ Compiled mode: Occasional failures with extreme data

## Safe Compilation Settings

If you need the performance benefit of compilation, use these **tested safe settings**:

### Environment Variables

```bash
# Enable compilation
export TORCH_COMPILED=1

# Safe compile mode (default)
export CHRONOS_COMPILE_MODE=reduce-overhead

# Safe backend (default)
export CHRONOS_COMPILE_BACKEND=inductor

# Safe dtype (default)
export CHRONOS_DTYPE=float32
```

### Compile Modes Comparison

| Mode | Speed | Stability | Recommendation |
|------|-------|-----------|----------------|
| **reduce-overhead** | 1.5-2x faster | ✅ Stable | **Recommended** |
| default | 1.3-1.8x faster | ✅ Mostly stable | Alternative |
| max-autotune | 2-3x faster | ⚠️ Unstable | **Not recommended** |

### Why reduce-overhead?

- **Balance**: Good speedup without aggressive optimizations
- **Tested**: Most extensively tested in production
- **Stable**: Fewer compilation failures
- **Fast warmup**: Quicker first-run compilation

## Configuration Helper Module

Use `src/chronos_compile_config.py` for centralized configuration:

### Production Eager (Default)

```python
from src.chronos_compile_config import apply_production_eager

apply_production_eager()
```

Settings:
- Compilation: **Disabled**
- Dtype: float32
- Attention: eager (no SDPA)

### Production Compiled

```python
from src.chronos_compile_config import apply_production_compiled

apply_production_compiled()
```

Settings:
- Compilation: **Enabled**
- Mode: reduce-overhead
- Backend: inductor
- Dtype: float32
- Attention: eager (no SDPA)
- Cache dir: `compiled_models/chronos2_torch_inductor/`

### Custom Configuration

```python
from src.chronos_compile_config import ChronosCompileConfig

config = ChronosCompileConfig(
    enabled=True,
    mode="reduce-overhead",
    backend="inductor",
    dtype="float32",
    cache_dir="/path/to/cache",
)
config.apply()
config.configure_torch_backends()
```

### Validate Configuration

```python
from src.chronos_compile_config import validate_config, print_current_config

# Print current settings
print_current_config()

# Validate and get warnings
is_valid, warnings = validate_config()
if not is_valid:
    for warning in warnings:
        print(f"⚠️ {warning}")
```

## Testing

### Run Fuzzing Tests

Comprehensive tests covering numerical stability:

```bash
# Run all fuzzing tests
pytest tests/test_chronos2_compile_fuzzing.py -v

# Run specific scenario
pytest tests/test_chronos2_compile_fuzzing.py::test_extreme_data_robustness -v

# Run with specific device
pytest tests/test_chronos2_compile_fuzzing.py -v --device=cuda
```

Test coverage:
- ✅ Basic smoke tests (eager and compiled)
- ✅ Accuracy comparison (MAE < 1e-2)
- ✅ Extreme data: very small, very large, high volatility
- ✅ Edge cases: spikes, drops, near-zero, constant
- ✅ Multiple predictions stability
- ✅ Fallback mechanism validation

### Run Comparison Tests

Compare eager vs compiled performance:

```bash
# Performance comparison
python test_chronos2_compiled_vs_eager.py

# Accuracy regression test
pytest tests/test_chronos_compile_accuracy.py

# End-to-end integration
pytest tests/test_chronos2_e2e_compile.py
```

## Performance Expectations

### Speedup (based on testing)

- **reduce-overhead**: 1.5-2x faster
- **First run penalty**: 30-60s compilation warmup
- **Subsequent runs**: Full speedup applies

### When Compilation Helps

- Long-running inference (many predictions)
- Batch predictions
- Production servers (warmup cost amortized)

### When to Use Eager Mode

- **Development/debugging** (faster iteration)
- **One-off predictions** (compilation overhead not worth it)
- **Numerical precision critical** (avoid any risk)
- **Extreme/unusual data** (better error handling)

## Troubleshooting

### Compilation Fails

If you see compilation errors:

1. **Disable compilation** (fall back to eager):
   ```bash
   export TORCH_COMPILED=0
   ```

2. **Check for very small values** in your data:
   ```python
   # Values < 1e-3 are automatically clamped to 0
   # in chronos2_wrapper.py lines 452-478
   ```

3. **Validate configuration**:
   ```python
   from src.chronos_compile_config import validate_config
   is_valid, warnings = validate_config()
   ```

### Numerical Instability

If predictions differ significantly between eager and compiled:

1. **Check MAE difference**:
   ```bash
   pytest tests/test_chronos_compile_accuracy.py -v
   ```

2. **Run fuzzing tests** on your data:
   ```python
   # Modify test_chronos2_compile_fuzzing.py to use your data
   pytest tests/test_chronos2_compile_fuzzing.py::test_extreme_data_robustness
   ```

3. **Use float32** (not fp16/bf16):
   ```bash
   export CHRONOS_DTYPE=float32
   ```

### Fallback Mechanism

Chronos2 wrapper has automatic fallback (`_call_with_compile_fallback`):

- If compiled mode fails during inference, automatically retries with eager mode
- Logs warning and disables compilation for future calls
- See `chronos2_wrapper.py` lines 378-399

### Debug Mode

Enable debug logging:

```bash
export CHRONOS_INDUCTOR_DEBUG=1
python trade_stock_e2e.py
```

## Implementation Details

### Where Compilation Happens

1. **Model loading** (`backtest_test3_inline.py:2206-2261`):
   - Checks `TORCH_COMPILED` environment variable
   - Applies compile settings if enabled
   - Passes to `Chronos2OHLCWrapper.from_pretrained()`

2. **Wrapper initialization** (`chronos2_wrapper.py:290-312`):
   - Calls `torch.compile()` on the model
   - Sets up cache directory
   - Catches and logs compilation failures

3. **Inference** (`chronos2_wrapper.py:744-757`):
   - Uses `_call_with_compile_fallback()` wrapper
   - Automatically falls back to eager on error

### Numerical Safeguards

1. **Small value clamping** (`chronos2_wrapper.py:452-478`):
   - Values with `abs(x) < 1e-3` are clamped to 0
   - Prevents sympy evaluation errors in torch._inductor

2. **SDPA backend disabling** (`backtest_test3_inline.py:2232-2238`):
   - Disables Flash and Memory-Efficient SDPA
   - Forces math SDPA backend
   - Uses eager attention implementation

3. **Eager attention** (`backtest_test3_inline.py:2240-2242`):
   - Forces `attn_implementation="eager"`
   - Bypasses all SDPA backends
   - Most reliable but slightly slower

## Best Practices

### ✅ Do

- Use eager mode (default) for production unless you've tested compilation thoroughly
- Run fuzzing tests before enabling compilation in production
- Monitor predictions for numerical anomalies when using compilation
- Use `reduce-overhead` mode if you enable compilation
- Keep `dtype=float32` for numerical stability

### ❌ Don't

- Enable compilation without testing on your specific data
- Use `max-autotune` mode in production
- Use fp16/bf16 without extensive testing
- Assume compilation is always faster (warmup cost matters)
- Ignore compilation warnings/errors

## References

- Fuzzing tests: `tests/test_chronos2_compile_fuzzing.py`
- Config module: `src/chronos_compile_config.py`
- Wrapper implementation: `src/models/chronos2_wrapper.py`
- Model loader: `backtest_test3_inline.py:2206-2261`
- Existing tests:
  - `tests/test_chronos_compile_accuracy.py`
  - `test_chronos2_compiled_vs_eager.py`
  - `tests/test_chronos2_e2e_compile.py`

## Summary

**Default: Compilation DISABLED**
- ✅ Maximum stability
- ✅ Predictable behavior
- ✅ Better error handling
- ⚠️ Slower inference

**Optional: Enable with TORCH_COMPILED=1**
- ✅ 1.5-2x faster inference
- ⚠️ Warmup overhead
- ⚠️ Potential numerical issues
- ⚠️ More complex error handling

**Recommendation: Keep disabled unless you have a specific performance need and have tested thoroughly.**
