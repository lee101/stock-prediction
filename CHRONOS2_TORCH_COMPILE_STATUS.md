# Chronos2 torch.compile Integration Status

## Current Configuration

### Default Settings
- `TORCH_COMPILED=1` - **Enabled by default** for faster inference
- `attn_implementation="eager"` - Compatible with torch.compile
- SDPA backends disabled as backup (Flash, MemEfficient)

### Environment Variables
- `TORCH_COMPILED=1` - Enable torch.compile (default)
- `TORCH_COMPILED=0` - Disable torch.compile (eager mode only)
- `CHRONOS_COMPILE_MODE` - Compilation mode (default: "reduce-overhead")
- `CHRONOS_COMPILE_BACKEND` - Backend (default: "inductor")

## Known Issues and Observations

### torch.compile Warnings (TORCH_COMPILED=1)
When running with torch.compile enabled, the following warnings have been observed:

1. **CUDA Graphs Warning**
   ```
   skipping cudagraphs due to mutated inputs (2 instances)
   ```
   - **Cause**: Input tensors are being modified in-place during inference
   - **Impact**: Prevents CUDA graphs optimization, reducing potential speedup
   - **Solution**: Would require refactoring model code to avoid in-place mutations

2. **Symbolic Shapes Warning**
   ```
   W1113 03:20:15.284000 ... symbolic_shapes.py:6833] [6/4] _maybe_guard_rel()
   was called on non-relation expression Eq(s43, 1) | Eq(64*s34, s43)
   ```
   - **Cause**: Dynamic shape handling in torch.compile
   - **Impact**: May trigger recompilations when shapes change
   - **Solution**: Fixed batch sizes could help, but limits flexibility

### Eager Mode (TORCH_COMPILED=0) - Stable Baseline
- ✅ No warnings or errors
- ✅ Consistent performance
- ✅ Verified working with 60+ seconds continuous operation
- ✅ No "Invalid backend" errors
- ⚠️ Potentially slower than optimized compiled mode

## Performance Comparison (TBD)

### Metrics to Measure
1. **Latency**: Average prediction time per inference
2. **MAE**: Mean Absolute Error on walk-forward test
3. **Profitability**: Total return % on simulated trading
4. **Stability**: Success rate without errors

### Test Scripts
- `test_chronos2_load_debug.py` - Debug torch.compile warnings and measure speedup
- `test_chronos2_profitability.py` - Measure MAE and trading profitability

## Integration Implementation

### Key Code Locations

**backtest_test3_inline.py:1930-1993** - `load_chronos2_wrapper()`
```python
# Enable torch.compile by default (can disable with TORCH_COMPILED=0)
torch_compiled_flag = os.getenv("TORCH_COMPILED", "1")  # DEFAULT IS "1"
compile_enabled = torch_compiled_flag in {"1", "true", "yes", "on"}

# Force eager attention (no SDPA) - compatible with torch.compile
attn_implementation = "eager"

# Also disable SDPA backends at torch level as backup
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
```

### SDPA Backend Fix
The "Invalid backend" error was resolved by:
1. Setting `attn_implementation="eager"` in model loading
2. Disabling Flash and MemEfficient SDPA backends
3. Forcing Math SDPA backend as fallback

## Recommendations (Pending Test Results)

### Current Recommendation: TORCH_COMPILED=0 (Eager Mode)
**Use eager mode unless profitability tests show significant benefit from compiled mode**

Reasons:
- ✅ Proven stable (no errors in production testing)
- ✅ No warnings or compilation overhead
- ✅ Simpler to debug and maintain
- ⚠️ torch.compile warnings suggest optimizations are being skipped

### When to Consider TORCH_COMPILED=1
Only enable compiled mode if profitability tests show:
1. Speedup >1.5x without accuracy loss
2. 100% success rate (no errors)
3. MAE difference <2%
4. Equal or better profitability

## Testing Status

### Completed Tests
- ✅ Integration tests with real training data (BTCUSD.csv)
- ✅ Basic smoke tests with TORCH_COMPILED=0
- ✅ Verified no "Invalid backend" errors
- ✅ Multiple consecutive predictions successful

### In Progress
- 🔄 Load test and debug analysis (`test_chronos2_load_debug.py`)
- 🔄 Walk-forward profitability test (`test_chronos2_profitability.py`)

### Pending Analysis
- ⏳ Actual speedup ratio (eager vs compiled)
- ⏳ MAE comparison across modes
- ⏳ Trading profitability comparison
- ⏳ Warning persistence (warmup vs steady-state)

## See Also
- [CHRONOS2_INTEGRATION.md](./CHRONOS2_INTEGRATION.md) - Integration details and fixes
- [CHRONOS2_PERFORMANCE_TEST.md](./CHRONOS2_PERFORMANCE_TEST.md) - Performance testing guide
- [test_chronos2_load_debug.py](../test_chronos2_load_debug.py) - Debug script
- [test_chronos2_profitability.py](../test_chronos2_profitability.py) - Profitability test
