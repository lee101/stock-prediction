# Pre-Augmentation Sweeps

## Overview

Pre-augmentation sweeps test different data transformation strategies BEFORE training to find which ones improve forecasting MAE (Mean Absolute Error). The idea is that presenting data in different forms (percent changes, log returns, detrended, etc.) may help the model learn better patterns.

## Architecture

### Directory Structure

```
preaug_sweeps/
├── augmentations/          # Augmentation strategy implementations
│   ├── base_augmentation.py
│   ├── strategies.py
│   └── __init__.py
├── augmented_dataset.py    # Dataset builder with augmentation
├── sweep_runner.py         # Main sweep orchestrator
├── run_sweep.sh           # Quick start script
├── results/               # Per-symbol, per-strategy results
│   ├── ETHUSD/
│   ├── UNIUSD/
│   └── BTCUSD/
├── logs/                  # Training logs
├── reports/               # Summary reports and CSVs
└── temp/                  # Temporary augmented datasets

preaugstrategies/
└── best/                  # Best configurations per symbol
    ├── ETHUSD.json
    ├── UNIUSD.json
    └── BTCUSD.json
```

## Augmentation Strategies

### 1. Baseline (No Augmentation)
- **Name**: `baseline`
- **Description**: Control - no transformation applied
- **Use**: Comparison baseline

### 2. Percent Change
- **Name**: `percent_change`
- **Description**: Transform prices to percent changes from first value
- **Formula**: `(price - price[0]) / price[0] * 100`
- **Use**: Removes absolute price levels, focuses on relative movements

### 3. Log Returns
- **Name**: `log_returns`
- **Description**: Logarithmic returns
- **Formula**: `log(price[t] / price[t-1])`
- **Use**: Stationary returns, common in finance

### 4. Differencing
- **Name**: `differencing`
- **Description**: First-order differencing
- **Formula**: `y[t] = x[t] - x[t-1]`
- **Use**: Makes series more stationary
- **Params**: `order` (default: 1)

### 5. Detrending
- **Name**: `detrending`
- **Description**: Remove linear trend, train on residuals
- **Use**: Focuses on deviations from trend
- **Benefits**: Removes long-term drift

### 6. Robust Scaling
- **Name**: `robust_scaling`
- **Description**: Scale using median and IQR instead of mean/std
- **Formula**: `(x - median) / IQR`
- **Use**: More robust to outliers than standard scaling

### 7. MinMax + Standard
- **Name**: `minmax_standard`
- **Description**: Min-max scaling to [0,1] then standardization
- **Use**: Combines benefits of both normalization methods
- **Params**: `feature_range` (default: (0, 1))

### 8. Rolling Window Normalization
- **Name**: `rolling_norm`
- **Description**: Normalize using rolling window statistics
- **Use**: Adapts to recent price dynamics
- **Params**: `window_size` (default: 20)

## Usage

### Quick Start

```bash
# Run full sweep on all symbols with all strategies
cd /nvme0n1-disk/code/stock-prediction
./preaug_sweeps/run_sweep.sh
```

### Custom Configuration

```bash
# Test specific symbols
SYMBOLS="ETHUSD BTCUSD" ./preaug_sweeps/run_sweep.sh

# Adjust training parameters
EPOCHS=5 BATCH_SIZE=32 ./preaug_sweeps/run_sweep.sh

# Test specific strategies
python3 preaug_sweeps/sweep_runner.py \
    --symbols ETHUSD UNIUSD \
    --strategies baseline percent_change log_returns \
    --epochs 3 \
    --batch-size 16
```

### Python API

```python
from preaug_sweeps.sweep_runner import PreAugmentationSweep

sweep = PreAugmentationSweep(
    data_dir="trainingdata",
    symbols=["ETHUSD", "UNIUSD", "BTCUSD"],
    strategies=["baseline", "percent_change", "log_returns"],
    epochs=3,
    batch_size=16,
)

sweep.run_sweep()
```

## Output

### Results

Each strategy run produces:

```json
{
  "status": "success",
  "strategy": "percent_change",
  "symbol": "ETHUSD",
  "mae": 12.345678,
  "rmse": 23.456789,
  "mape": 1.234,
  "best_val_loss": 0.123,
  "epochs": 3,
  "config": {
    "name": "percent_change",
    "params": {},
    "metadata": {
      "open_first": 1234.56,
      "high_first": 1245.67,
      ...
    }
  },
  "timestamp": "2025-11-12T01:23:45.678901"
}
```

### Best Configuration

For each symbol, the best strategy is saved to `preaugstrategies/best/{SYMBOL}.json`:

```json
{
  "symbol": "ETHUSD",
  "best_strategy": "log_returns",
  "mae": 10.123456,
  "rmse": 20.234567,
  "mape": 0.987,
  "config": {
    "name": "log_returns",
    "params": {},
    "metadata": {...}
  },
  "timestamp": "2025-11-12T01:23:45.678901",
  "comparison": {
    "baseline": {"mae": 15.0, "rmse": 25.0, "mape": 1.5},
    "percent_change": {"mae": 12.0, "rmse": 22.0, "mape": 1.2},
    "log_returns": {"mae": 10.123456, "rmse": 20.234567, "mape": 0.987}
  }
}
```

### Reports

Summary reports are generated in `preaug_sweeps/reports/`:

1. **JSON Results**: Full results for all runs
2. **CSV Summary**: Easy-to-analyze table format
3. **Console Output**: Pretty-printed comparison table

Example output:

```
MAE COMPARISON TABLE
================================================================================
strategy              ETHUSD      UNIUSD      BTCUSD
baseline              15.234567   8.765432    25.345678
percent_change        12.123456   7.654321    23.234567
log_returns           10.012345   6.543210    21.123456
differencing          11.234567   7.123456    22.345678
detrending            13.345678   8.234567    24.456789
robust_scaling        14.456789   8.345678    25.567890
minmax_standard       13.567890   7.456789    23.678901
rolling_norm          12.678901   7.567890    22.789012
```

## Implementation Details

### How It Works

1. **Create Augmented Dataset**
   - Load original training data
   - Apply augmentation transformation
   - Save to temporary directory

2. **Train Model**
   - Use existing Kronos/Toto trainer
   - Train on augmented data
   - Standard training pipeline

3. **Evaluate**
   - Run evaluation on validation set
   - **Important**: Predictions are inverse-transformed back to original scale
   - Calculate MAE, RMSE, MAPE

4. **Compare**
   - Compare all strategies
   - Select best by MAE
   - Save configuration

### Inverse Transformation

Critical: When making predictions with an augmented model, predictions must be inverse-transformed:

```python
# During training: original -> augmented
df_aug = augmentation.transform_dataframe(df_original)

# Train model on df_aug...

# During prediction: augmented -> original
predictions_aug = model.predict(context_aug)
predictions_original = augmentation.inverse_transform_predictions(
    predictions_aug,
    context=df_original
)
```

Each augmentation strategy implements:
- `transform_dataframe()`: Apply transformation
- `inverse_transform_predictions()`: Reverse transformation

## Adding New Strategies

To add a new augmentation strategy:

1. **Create Strategy Class**

```python
# In preaug_sweeps/augmentations/strategies.py

class MyCustomAugmentation(BaseAugmentation):
    def name(self) -> str:
        return "my_custom"

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # Apply transformation
        df_aug = df.copy()
        # ... your transformation logic
        return df_aug

    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        context: pd.DataFrame
    ) -> np.ndarray:
        # Reverse transformation
        # ... your inverse logic
        return predictions_original
```

2. **Register Strategy**

```python
# In AUGMENTATION_REGISTRY
AUGMENTATION_REGISTRY = {
    # ... existing strategies
    "my_custom": MyCustomAugmentation,
}
```

3. **Test It**

```bash
python3 preaug_sweeps/sweep_runner.py \
    --symbols ETHUSD \
    --strategies baseline my_custom \
    --epochs 3
```

## Performance Tips

### Fast Testing

For quick iteration:

```bash
# Test one symbol, one strategy, fewer epochs
python3 preaug_sweeps/sweep_runner.py \
    --symbols ETHUSD \
    --strategies baseline percent_change \
    --epochs 1 \
    --batch-size 32
```

### Full Production Run

For comprehensive results:

```bash
# All symbols, all strategies, full training
EPOCHS=10 BATCH_SIZE=16 ./preaug_sweeps/run_sweep.sh
```

### Parallel Execution

To run multiple symbols in parallel:

```bash
# Terminal 1
python3 preaug_sweeps/sweep_runner.py --symbols ETHUSD --epochs 5 &

# Terminal 2
python3 preaug_sweeps/sweep_runner.py --symbols UNIUSD --epochs 5 &

# Terminal 3
python3 preaug_sweeps/sweep_runner.py --symbols BTCUSD --epochs 5 &
```

## Expected Outcomes

### Hypothesis

Different augmentations may improve MAE by:

1. **Stationarity**: Making series more stationary (differencing, log returns)
2. **Scale Normalization**: Better numerical stability (robust scaling, minmax)
3. **Trend Removal**: Focusing on patterns vs. trend (detrending)
4. **Adaptive Scaling**: Adapting to recent dynamics (rolling norm)

### What to Look For

- **Best Strategy**: Which augmentation gives lowest MAE?
- **Consistency**: Does one strategy work well across all symbols?
- **Improvement**: % improvement over baseline
- **Trade-offs**: Some strategies may improve MAE but worsen RMSE/MAPE

### Example Insights

```
ETHUSD Results:
  baseline:         MAE = 15.23  (no augmentation)
  log_returns:      MAE = 10.01  (34.3% improvement!)
  percent_change:   MAE = 12.12  (20.4% improvement)

  → Best: log_returns
  → Takeaway: ETHUSD benefits from log return transformation
```

## Troubleshooting

### Common Issues

**Issue**: Training fails with NaN loss

**Solution**: Some augmentations may create extreme values. Check:
- Clip values in augmentation
- Adjust normalization parameters
- Add safeguards (e.g., `+ 1e-8` for division)

**Issue**: Predictions don't inverse transform correctly

**Solution**:
- Ensure metadata is stored during transform
- Test inverse transform separately
- Check that predictions use same context as training

**Issue**: Out of memory

**Solution**:
- Reduce batch size
- Process one symbol at a time
- Clean up temp directories between runs

## Next Steps

After finding best strategies:

1. **Use Best Config**
   ```python
   # Load best config
   with open("preaugstrategies/best/ETHUSD.json") as f:
       best = json.load(f)

   # Apply to production training
   augmentation = get_augmentation(best["best_strategy"])
   ```

2. **Combine Strategies**
   - Try ensemble of multiple augmentations
   - Average predictions from different augmented models

3. **Fine-tune**
   - Adjust augmentation parameters
   - Test variations (e.g., different window sizes for rolling norm)

4. **Production Integration**
   - Integrate best augmentation into main training pipeline
   - Update model serving to apply inverse transform

## References

- Kronos Training: `kronostraining/`
- Toto Training: `tototraining/`
- Training Data: `trainingdata/`
- Base Augmentation: `preaug_sweeps/augmentations/base_augmentation.py`
- Strategies: `preaug_sweeps/augmentations/strategies.py`
