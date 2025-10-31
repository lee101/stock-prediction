# Kronos Torch Compile Analysis

**Date**: 2025-10-30
**Status**: Analysis based on production logs and codebase review

## Question: Should Kronos use torch.compile?

## Evidence Review

### 1. Kronos Compile Support

Kronos DOES support torch.compile:
- File: `external/kronos/kronos_example.py` shows compilation of `decode_s1` and `decode_s2` methods
- File: `differentiable_market_kronos/train.py` has `--kronos-no-compile` flag (line 51)
- Compilation is applied per-method, not full model

**Key difference from Toto**: Kronos compiles specific decode methods, not the entire forward pass.

### 2. Production Logs Analysis

From your `trade_stock_e2e.py` logs:

**BTC Kronos Performance**:
- No recompilation warnings for Kronos
- No "CUDA graphs skipped" for Kronos
- Kronos appears to run without the issues Toto has

**The Issue**: You're seeing synthetic bid/ask because of `ADD_LATEST=False`, not Kronos issues.

### 3. Comparison: Toto vs Kronos Compilation

| Aspect | Toto | Kronos |
|--------|------|--------|
| Compile target | Full model + KV cache | Specific decode methods |
| Recompilations | ✗ 8+ (hitting limit) | ✓ None observed |
| CUDA graphs | ✗ Skipped | ✓ No issues |
| Dynamic shapes | ✗ KV cache indices | ✓ More static |
| Observed issues | ✗ Many | ✓ None |

### 4. Theoretical Analysis

**Why Kronos may handle compilation better**:
1. **Smaller compile scope**: Only decode_s1/decode_s2, not full model
2. **No KV cache**: Kronos doesn't have the dynamic KV cache that causes Toto's issues
3. **Different architecture**: Kronos tokenizer-predictor design is more static

**Potential risks**:
1. First compilation still slow (~10-30s warmup)
2. Memory overhead (compiled models use more memory)
3. Unknown accuracy impact (need data to verify)

## Decision Framework

Without being able to run direct tests (due to `/vfast` permission issues), we use:

### Conservative Approach (Recommended)
**Decision**: **EAGER MODE** for Kronos (no compilation)

**Rationale**:
1. ✅ **Safety first**: Toto showed compilation can cause issues
2. ✅ **No observed problems**: Kronos runs fine in eager mode currently
3. ✅ **Lower memory**: Eager uses less memory
4. ✅ **Simpler debugging**: No compilation complexity
5. ⚠️ **Unknown benefit**: Haven't measured actual Kronos speedup from compilation

**Trade-off**: May be 20-40% slower than compiled, but STABLE.

### Aggressive Approach (If you want performance)
**Decision**: **COMPILED MODE** for Kronos (with monitoring)

**Rationale**:
1. ✅ **No observed issues**: Unlike Toto, Kronos shows no recompilation warnings
2. ✅ **Better architecture**: Smaller compile scope, no KV cache
3. ✅ **Potential speedup**: 20-40% faster inference (theoretical)
4. ⚠️ **Unverified**: Haven't measured accuracy/return impact

**Monitoring required**:
- Watch for recompilation warnings
- Monitor memory usage
- Track strategy returns vs baseline

## Recommendation Based on Your Context

Given:
- ✅ You're already having Toto compilation issues
- ✅ You want optimal returns, not maximum speed
- ✅ ETHUSD MaxDiff is already profitable (10.43% return)
- ✅ Production stability is critical

**RECOMMENDED: EAGER MODE for Kronos**

Rationale:
1. **Risk mitigation**: Don't add compilation complexity when Toto already has issues
2. **Current performance acceptable**: Your profitable strategy (ETHUSD MaxDiff) doesn't need speedup
3. **Easier troubleshooting**: One less variable when debugging
4. **Memory efficiency**: More room for other operations

## Configuration

### Recommended (Eager Mode):

```bash
# In .env.compile

# Kronos model: EAGER mode (default, no compilation)
# Rationale: Prioritize stability over speed
# Kronos doesn't have recompilation issues like Toto,
# but eager mode is simpler and lower memory
# No flag needed - Kronos wrapper doesn't currently
# expose a compile option in production code
```

**No environment variable needed** - Kronos wrapper in `src/models/kronos_wrapper.py` doesn't expose a compile flag.

### If You Want to Enable Compilation Later:

Would need to modify `src/models/kronos_wrapper.py` to:
1. Accept a `compile` parameter in `__init__`
2. Apply `torch.compile` to the predictor's decode methods after loading
3. Add env var `KRONOS_COMPILE` to control it

Example modification needed (not implemented):
```python
# In KronosForecastingWrapper._ensure_predictor():
if self.compile_enabled:
    predictor.model.decode_s1 = torch.compile(predictor.model.decode_s1)
    predictor.model.decode_s2 = torch.compile(predictor.model.decode_s2)
```

## Testing Plan (When Ready)

If you want to test Kronos compilation in the future:

1. **Fix `/vfast` permission issue**:
   ```bash
   export COMPILED_MODELS_DIR=/home/administrator/code/stock-prediction/compiled_models
   ```

2. **Add compile support** to `KronosForecastingWrapper`

3. **Run data-driven test**:
   ```bash
   python scripts/test_kronos_compile_accuracy.py
   ```

4. **Compare backtests**:
   ```bash
   # Eager
   KRONOS_COMPILE=0 python backtest_test3_inline.py --symbol BTCUSD

   # Compiled
   KRONOS_COMPILE=1 python backtest_test3_inline.py --symbol BTCUSD

   # Compare
   python scripts/compare_compile_backtest_returns.py
   ```

## Summary

**Current Decision**: **EAGER MODE for Kronos** (no compilation)

**Configuration**: No changes needed - Kronos already defaults to eager mode.

**Rationale**:
- Prioritize stability and simplicity
- Kronos wrapper doesn't currently expose compile option
- No observed issues with eager mode
- Focus on fixing Toto issues first

**When to Revisit**:
- After Toto compilation is stable
- If Kronos becomes a bottleneck
- When you can run proper A/B tests

**Files Updated**: `.env.compile` (documented decision)

---
**Status**: ✅ Kronos remains in EAGER mode (default, no changes needed)
