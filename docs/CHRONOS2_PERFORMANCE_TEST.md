# Chronos2 Performance Testing

## Quick Start

Run the comprehensive performance comparison:

```bash
uv run python test_chronos2_compiled_vs_eager.py
```

This test compares:
- **TORCH_COMPILED=0** (eager mode, stable) vs **TORCH_COMPILED=1** (compiled mode, faster but may have SDPA errors)
- Measures prediction latency, MAE (Mean Absolute Error), and stability
- Runs 5 iterations per mode on real BTCUSD training data (500 rows)

## Test Overview

The test (`test_chronos2_compiled_vs_eager.py`) performs:

1. **Eager Mode Test** (TORCH_COMPILED=0):
   - Loads Chronos2 with `attn_implementation="eager"`
   - Runs 5 predictions with 7-day forecast horizon
   - Measures latency and MAE against ground truth

2. **Compiled Mode Test** (TORCH_COMPILED=1):
   - Loads Chronos2 with `torch_compile=True`
   - Runs 5 predictions with same parameters
   - Measures latency and MAE against ground truth

3. **Comparison**:
   - Success rate (% of runs without errors)
   - Average prediction latency
   - Average MAE percentage
   - Speedup ratio (if both modes succeed)

## Expected Outcomes

### Eager Mode (TORCH_COMPILED=0)
- **Stability**: ✅ 100% success rate (no SDPA errors)
- **Latency**: ~2-3 seconds per prediction
- **MAE**: Baseline accuracy

### Compiled Mode (TORCH_COMPILED=1)
- **Stability**: ⚠️ May have "Invalid backend" SDPA errors
- **Latency**: Potentially faster (if successful)
- **MAE**: Similar to eager mode

## Running Individual Modes

### Test Only Eager Mode (Recommended)
```bash
TORCH_COMPILED=0 ONLY_CHRONOS2=1 uv run python -c "
import os
os.environ['TORCH_COMPILED'] = '0'
os.environ['ONLY_CHRONOS2'] = '1'
from backtest_test3_inline import resolve_chronos2_params, load_chronos2_wrapper
import pandas as pd
import numpy as np
from pathlib import Path

# Load data
df = pd.read_csv('trainingdata/BTCUSD.csv').tail(200)
df = df.reset_index(drop=True)
df.columns = [c.lower() for c in df.columns]
df['timestamp'] = pd.date_range('2024-01-01', periods=len(df), freq='D')
df['symbol'] = 'BTCUSD'

# Load model
params = resolve_chronos2_params('BTCUSD')
wrapper = load_chronos2_wrapper(params)

# Predict
result = wrapper.predict_ohlc(df, 'BTCUSD', 7, params['context_length'], params['batch_size'])
print(f'✓ Prediction successful: {result.quantile_frames[0.5][\"close\"].values}')
"
```

### Test Compiled Mode (May Fail)
```bash
TORCH_COMPILED=1 ONLY_CHRONOS2=1 uv run python -c "
# Same as above but with TORCH_COMPILED=1
"
```

## Interpreting Results

### Success Rate
- **100%**: Mode is stable and reliable
- **<100%**: Mode has intermittent failures (not production-ready)
- **0%**: Mode completely fails

### Latency Comparison
- **Speedup >1.2x**: Compiled mode provides meaningful performance benefit
- **Speedup <1.2x**: Negligible benefit, not worth instability risk
- **Slowdown**: Compiled mode is actually slower (compilation overhead)

### MAE Comparison
- **Difference <5%**: Accuracy is comparable
- **Difference >5%**: Significant accuracy degradation

## Current Recommendation (as of 2025-11-12)

**Use TORCH_COMPILED=0 (eager mode)** because:
1. ✅ 100% stability (no "Invalid backend" errors)
2. ✅ Verified with 60+ seconds of continuous trading system operation
3. ✅ Verified with 3+ consecutive predictions on real data
4. ⚠️ Compiled mode has shown SDPA backend errors in production

**Only consider TORCH_COMPILED=1 if**:
- You complete this performance test and achieve 100% success rate
- Speedup is >1.5x
- MAE difference is <2%

## Troubleshooting

### "Invalid backend" Error in Compiled Mode
This is expected. The eager mode fix (attn_implementation="eager") may not be compatible with torch.compile. This is why TORCH_COMPILED=0 is the default and recommended setting.

### Test Takes Too Long
- Reduce `num_runs` from 5 to 3 in the test script
- Reduce training data from 500 to 200 rows

### Out of Memory
- Reduce `batch_size` in hyperparams/chronos2/BTCUSD.json
- Reduce `context_length` to 512 instead of 768

## Performance Benchmarks (To Be Measured)

Run the test and record your results here:

```
Date: ____________________
GPU: ____________________

Eager Mode (TORCH_COMPILED=0):
  Success Rate: ____%
  Avg Latency: ____s
  Avg MAE: ____%

Compiled Mode (TORCH_COMPILED=1):
  Success Rate: ____%
  Avg Latency: ____s
  Avg MAE: ____%

Speedup: ____x
Recommendation: ____________
```

## See Also

- [CHRONOS2_INTEGRATION.md](./CHRONOS2_INTEGRATION.md) - Integration details and fixes
- [test_chronos2_compiled_vs_eager.py](../test_chronos2_compiled_vs_eager.py) - Test script
- [run_chronos2_real_data_tests_direct.py](../run_chronos2_real_data_tests_direct.py) - Quick smoke test
