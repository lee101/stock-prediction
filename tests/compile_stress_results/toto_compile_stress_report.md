# Compile Integration Stress Test Report

**Test Configuration:**
- Device: cuda
- Iterations: 3
- Context Length: 256
- Prediction Length: 1
- Num Samples: 64

## Toto Model

### Accuracy Metrics

| Compile Mode | MAE (avg) | RMSE (avg) | MAPE (avg) | Prediction Mean | Std Dev |
|--------------|-----------|------------|------------|-----------------|---------|
| max-autotune | 3.8772 | 3.8772 | 3.52% | 106.37 | 0.00 |
| eager | 0.3439 | 0.3439 | 0.31% | 110.12 | 0.00 |

### Performance Metrics

| Compile Mode | Inference Time (ms) | Peak Memory (MB) | Recompilations |
|--------------|---------------------|------------------|----------------|
| max-autotune | 80082.81 | 783.01 | 0 |
| eager | 2059.40 | 967.25 | 0 |

### Accuracy Delta (Compiled - Eager)

- MAE Delta: 3.5333 (+1027.27%)
- ⚠️ **WARNING**: MAE delta exceeds 5% threshold!

## Recommendations

No major issues detected. ✅
