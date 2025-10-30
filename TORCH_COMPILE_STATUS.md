# Torch Compile Status Report

**Date**: 2025-10-30
**Status**: ⚠️ RECOMPILATION ISSUES DETECTED

## Current Issues

Based on your production logs, the following issues were identified:

### 1. Excessive Recompilations (Critical)
```
W1030 09:08:09.060000 torch._dynamo hit config.recompile_limit (8)
function: 'positional_embedding' (/path/to/toto/model/attention.py:105)
last reason: 6/7: kv_cache._current_idx[7] == 0
```

**Impact**:
- Slow inference (recompilation overhead)
- Unpredictable latency
- CUDA graphs being skipped

**Root Cause**: Dynamic KV cache indices changing during inference

### 2. CUDA Graphs Skipped
```
skipping cudagraphs due to mutated inputs (2 instances)
```

**Impact**:
- Performance degradation
- Cannot use CUDA graph optimization

## Immediate Actions (Choose One)

### Option 1: Disable torch.compile (SAFEST - Recommended for Production)

```bash
# Quick disable
./scripts/toggle_torch_compile.sh disable
source .env.compile
python trade_stock_e2e.py
```

**Pros**: Stable, predictable performance
**Cons**: 2-3x slower inference

### Option 2: Increase Recompile Limit (Temporary Workaround)

```bash
export TORCHDYNAMO_RECOMPILE_LIMIT=32
python trade_stock_e2e.py
```

**Pros**: Allows more recompilations before warning
**Cons**: Doesn't fix root cause, still slow

### Option 3: Use Reduce-Overhead Mode (Compromise)

```bash
./scripts/toggle_torch_compile.sh mode reduce-overhead
source .env.compile
python trade_stock_e2e.py
```

**Pros**: Less aggressive optimization, fewer recompilations
**Cons**: May still have issues, less speedup

## Testing and Validation

Before making any changes to production, run the stress test:

```bash
# Production readiness check (20 iterations, ~15 min)
python scripts/run_compile_stress_test.py --mode production-check

# Quick test (3 iterations, ~2 min)
python scripts/run_compile_stress_test.py --mode quick
```

This will generate:
- `tests/compile_stress_results/compile_stress_report.md` - Human-readable analysis
- `tests/compile_stress_results/compile_stress_results.json` - Raw data

### Key Metrics to Check

1. **MAE Delta** (compiled vs eager): Should be < 5%
2. **Recompilations**: Should be < 10 per run
3. **Speedup**: Compiled should be faster (not slower)

## Recommended Production Strategy

Based on the current issues, here's the recommended deployment strategy:

### Phase 1: Immediate (Today)
1. ✅ **Disable torch.compile for production**
   ```bash
   export TOTO_DISABLE_COMPILE=1
   ```
2. ✅ Monitor performance and stability
3. ✅ Collect baseline metrics (MAE, latency)

### Phase 2: Investigation (This Week)
1. ⚠️ Run full stress tests
   ```bash
   python scripts/run_compile_stress_test.py --mode full
   ```
2. ⚠️ Analyze recompilation logs
   ```bash
   TORCH_LOGS="recompiles" python trade_stock_e2e.py 2>&1 | tee recompile.log
   python scripts/analyze_recompilations.py recompile.log
   ```
3. ⚠️ Compare accuracy between compiled and eager modes

### Phase 3: Long-term Fix (Future)
1. 📋 Optimize KV cache implementation (static allocation)
2. 📋 Use `torch._dynamo.mark_dynamic()` for dynamic dimensions
3. 📋 Consider model architecture changes to reduce dynamic shapes
4. 📋 Test with newer PyTorch versions (may have fixes)

## New Tools and Infrastructure

I've created comprehensive testing infrastructure:

### 1. Integration Stress Tests
**File**: `tests/test_compile_integration_stress.py`

Validates:
- ✅ Accuracy (MAE, RMSE, MAPE)
- ✅ Performance (inference time, memory)
- ✅ Stability (multi-iteration)
- ✅ Recompilation detection

### 2. Stress Test Runner
**File**: `scripts/run_compile_stress_test.py`

Quick commands:
```bash
# Quick test (3 iterations)
python scripts/run_compile_stress_test.py --mode quick

# Full test (10 iterations)
python scripts/run_compile_stress_test.py --mode full

# Production check (20 iterations, strict validation)
python scripts/run_compile_stress_test.py --mode production-check
```

### 3. Recompilation Analyzer
**File**: `scripts/analyze_recompilations.py`

Analyzes torch logs and provides actionable recommendations:
```bash
TORCH_LOGS="recompiles" python trade_stock_e2e.py 2>&1 | tee recompile.log
python scripts/analyze_recompilations.py recompile.log
```

### 4. Toggle Script
**File**: `scripts/toggle_torch_compile.sh`

Easy switching:
```bash
./scripts/toggle_torch_compile.sh disable   # Disable
./scripts/toggle_torch_compile.sh enable    # Enable
./scripts/toggle_torch_compile.sh status    # Show status
./scripts/toggle_torch_compile.sh test      # Run tests
```

### 5. Comprehensive Guide
**File**: `docs/TORCH_COMPILE_GUIDE.md`

Covers:
- Common issues and solutions
- Configuration reference
- Production deployment strategies
- Debugging tools
- Performance benchmarks

## Next Steps

### Right Now (5 minutes)

1. **Disable torch.compile for production**:
   ```bash
   ./scripts/toggle_torch_compile.sh disable
   source .env.compile
   ```

2. **Restart trading bot**:
   ```bash
   python trade_stock_e2e.py
   ```

### Today (30 minutes)

1. **Run stress test**:
   ```bash
   python scripts/run_compile_stress_test.py --mode quick
   ```

2. **Review results**:
   ```bash
   cat tests/compile_stress_results/compile_stress_report.md
   ```

### This Week (2-3 hours)

1. **Full stress test with production data**:
   ```bash
   python scripts/run_compile_stress_test.py --mode production-check
   ```

2. **Analyze recompilations**:
   ```bash
   TORCH_LOGS="recompiles" python trade_stock_e2e.py 2>&1 | tee recompile.log
   python scripts/analyze_recompilations.py recompile.log
   ```

3. **Compare backtest results**:
   ```bash
   # Run backtest with eager mode
   export TOTO_DISABLE_COMPILE=1
   python backtest_test3_inline.py

   # Run backtest with compiled mode
   export TOTO_DISABLE_COMPILE=0
   python backtest_test3_inline.py

   # Compare
   python evaltests/compare_compile_modes.py
   ```

## Performance Expectations

Based on typical configurations:

| Configuration | First Run | Subsequent | Memory | Stability |
|---------------|-----------|------------|--------|-----------|
| **Eager (current recommendation)** | 500ms | 500ms | 650MB | ✅ Stable |
| **Compiled (with recompilations)** | 25s | 500ms+ | 900MB | ⚠️ Unstable |
| **Compiled (ideal, after fix)** | 25s | 200ms | 900MB | ✅ Stable |

## Support

If you need help:

1. **Check documentation**: `docs/TORCH_COMPILE_GUIDE.md`
2. **Run diagnostic**: `python scripts/analyze_recompilations.py <log>`
3. **Review stress test results**: `tests/compile_stress_results/`

## Summary

**Current Recommendation**: **Disable torch.compile for production** until the KV cache recompilation issue is resolved.

**Risk Level**:
- ⚠️ With compile enabled: HIGH (unstable, unpredictable latency)
- ✅ With compile disabled: LOW (stable, predictable)

**Trade-off**:
- Compile disabled = 2-3x slower but stable
- Compile enabled = potentially faster but currently unreliable due to recompilations

**Action**: Run `./scripts/toggle_torch_compile.sh disable` and restart the bot.
