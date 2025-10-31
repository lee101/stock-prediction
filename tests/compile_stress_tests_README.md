# Compile Stress Tests

This directory contains integration stress tests for validating torch.compile reliability and performance in production environments.

## Quick Start

### Run Quick Test (3 iterations)

```bash
# From project root
python scripts/run_compile_stress_test.py --mode quick
```

### Run Production Readiness Check (20 iterations)

```bash
python scripts/run_compile_stress_test.py --mode production-check
```

### Run Tests via pytest

```bash
# Run Toto compile stress test
pytest tests/test_compile_integration_stress.py::test_toto_compile_stress -v -s

# Run all compile tests
pytest tests/test_compile_integration_stress.py -v -s
```

## What Gets Tested

### 1. Accuracy Validation
- **MAE (Mean Absolute Error)**: Compare predictions between compiled and eager modes
- **RMSE (Root Mean Squared Error)**: Measure prediction quality
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based accuracy
- **Threshold**: MAE delta should be < 5% between compiled and eager

### 2. Performance Metrics
- **Inference Time**: Measure time per prediction (should be faster for compiled)
- **Memory Usage**: Track peak GPU memory (compiled may use more memory)
- **Recompilations**: Count torch.compile recompilations (should be minimal)

### 3. Stability Testing
- **Multi-iteration**: Run predictions multiple times to detect instability
- **Varied Inputs**: Test with different context lengths and sample sizes
- **Recompilation Detection**: Identify excessive recompilations

## Test Configurations

### Quick Mode (3 iterations)
- **Use case**: Fast feedback during development
- **Runtime**: ~1-2 minutes
- **Command**: `--mode quick`

### Full Mode (10 iterations)
- **Use case**: Thorough testing before commits
- **Runtime**: ~5-10 minutes
- **Command**: `--mode full`

### Production Check (20 iterations)
- **Use case**: Pre-deployment validation
- **Runtime**: ~10-20 minutes
- **Command**: `--mode production-check`
- **Validation**: Strict thresholds, fails if issues detected

## Output Files

Results are saved to `tests/compile_stress_results/`:

1. **`*_results.json`**: Raw test results in JSON format
2. **`*_report.md`**: Human-readable markdown report with analysis
3. Timestamped for historical tracking

### Example Report Structure

```markdown
# Compile Integration Stress Test Report

## Toto Model

### Accuracy Metrics
| Compile Mode | MAE (avg) | RMSE (avg) | MAPE (avg) |
|--------------|-----------|------------|------------|
| max-autotune | 0.1234    | 0.2345     | 2.45%      |
| eager        | 0.1230    | 0.2340     | 2.43%      |

### Performance Metrics
| Compile Mode | Inference Time (ms) | Peak Memory (MB) | Recompilations |
|--------------|---------------------|------------------|----------------|
| max-autotune | 245.12              | 892.34           | 8              |
| eager        | 512.34              | 651.23           | 0              |

### Recommendations
✅ MAE delta within acceptable range (<5%)
⚠️  Excessive recompilations detected (8)
```

## Interpreting Results

### ✅ PASS Criteria

1. **MAE delta < 5%**: Compiled and eager predictions are similar
2. **Recompilations < 10**: Minimal recompilations after warm-up
3. **Speedup ≥ 0.8x**: Compiled is not significantly slower

### ⚠️ WARNING Criteria

1. **MAE delta 5-10%**: Noticeable accuracy difference
2. **Recompilations 10-20**: Moderate recompilation overhead
3. **Speedup 0.5-0.8x**: Compiled is moderately slower

### ❌ FAIL Criteria

1. **MAE delta > 10%**: Significant accuracy degradation
2. **Recompilations > 20**: Excessive recompilation overhead
3. **Speedup < 0.5x**: Compiled is significantly slower

## Troubleshooting

### Issue: Tests Fail with CUDA OOM

**Solution 1**: Reduce num_samples
```bash
python scripts/run_compile_stress_test.py --num-samples 64
```

**Solution 2**: Test on CPU (slower)
```bash
python scripts/run_compile_stress_test.py --device cpu
```

### Issue: Excessive Recompilations Detected

**Diagnose**:
```bash
# Capture recompilation logs
TORCH_LOGS="recompiles" python scripts/run_compile_stress_test.py 2>&1 | tee recompile.log

# Analyze
python scripts/analyze_recompilations.py recompile.log
```

**Quick Fix**: Disable torch.compile
```bash
export TOTO_DISABLE_COMPILE=1
```

See `docs/TORCH_COMPILE_GUIDE.md` for detailed solutions.

### Issue: MAE Divergence Between Compiled and Eager

**Investigate**:
1. Check if using bfloat16 (may cause precision differences)
2. Run with float32: `export REAL_TESTING=1`
3. Review PyTorch version and known issues

**Solutions**:
- Use float32 for production if bfloat16 causes issues
- Report significant divergence to PyTorch team
- Fall back to eager mode

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Compile Stress Test

on: [push, pull_request]

jobs:
  test:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: uv pip install -r requirements.txt
      - name: Run compile stress test
        run: python scripts/run_compile_stress_test.py --mode quick
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: compile-stress-results
          path: tests/compile_stress_results/
```

### Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Run quick compile stress test before commit
python scripts/run_compile_stress_test.py --mode quick --model toto
exit $?
```

## Advanced Usage

### Custom Test Configuration

```python
from tests.test_compile_integration_stress import CompileStressTestRunner

runner = CompileStressTestRunner(
    device="cuda",
    num_iterations=10,
    context_length=1024,  # Longer context
    pred_length=7,        # Multi-step prediction
    num_samples=256,      # More samples
)

# Run custom test
series = runner._generate_synthetic_series(1024, seed=123)
targets = np.array([...])  # Your targets

compiled_results, eager_results = runner.test_toto_compiled_vs_eager(series, targets)
```

### Batch Testing Multiple Configurations

```bash
# Test different compile modes
for mode in default reduce-overhead max-autotune; do
    export TOTO_COMPILE_MODE=$mode
    python scripts/run_compile_stress_test.py --mode full
done

# Compare results
ls tests/compile_stress_results/
```

## Related Documentation

- `docs/TORCH_COMPILE_GUIDE.md` - Comprehensive torch.compile guide
- `scripts/compare_toto_compile.py` - Simple eager vs compiled comparison
- `evaltests/compare_compile_modes.py` - Backtest-based comparison

## Support

If you encounter issues:

1. Check `docs/TORCH_COMPILE_GUIDE.md` for common issues
2. Run diagnostic: `python scripts/analyze_recompilations.py <log>`
3. Review PyTorch torch.compile docs
4. Report issue with test results and logs
