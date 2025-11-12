# Pre-Augmentation Sweeps - Quick Start

## What is This?

Pre-augmentation sweeps test different data transformations BEFORE training to find which ones improve forecasting MAE (Mean Absolute Error). Think of it as trying different "views" of your data to see which helps the model learn better patterns.

## Quick Start

### 1. Run a Quick Test (Recommended First)

Test with just 2 strategies on one symbol:

```bash
.venv312/bin/python3 preaug_sweeps/sweep_runner.py \
    --symbols ETHUSD \
    --strategies baseline percent_change \
    --epochs 1 \
    --batch-size 16
```

This takes ~10-20 minutes and validates everything works.

### 2. Run Full Sweep

Test ALL strategies on all target symbols:

```bash
./preaug_sweeps/run_full_sweep.sh
```

Or with custom epochs:

```bash
EPOCHS=5 ./preaug_sweeps/run_full_sweep.sh
```

**Warning**: Full sweep takes several hours!

### 3. Analyze Results

```bash
.venv312/bin/python3 preaug_sweeps/analyze_results.py
```

This shows:
- Best strategy for each symbol
- Performance comparison table
- Improvement over baseline

## What You Get

### Best Configurations

For each symbol, the best augmentation strategy is saved to:

```
preaugstrategies/best/ETHUSD.json
preaugstrategies/best/UNIUSD.json
preaugstrategies/best/BTCUSD.json
```

Example:

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
  "comparison": {
    "baseline": {"mae": 15.0, ...},
    "log_returns": {"mae": 10.123456, ...}
  }
}
```

### Full Results

All results are in:
- `preaug_sweeps/reports/` - Summary CSVs and JSON
- `preaug_sweeps/results/` - Per-symbol, per-strategy details
- `preaug_sweeps/logs/` - Training logs

## Available Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `baseline` | No augmentation | Comparison baseline |
| `percent_change` | Percent changes from first value | Relative movements |
| `log_returns` | Logarithmic returns | Stationary returns |
| `differencing` | First-order differencing | Stationarity |
| `detrending` | Remove linear trend | Focus on deviations |
| `robust_scaling` | Median/IQR scaling | Robust to outliers |
| `minmax_standard` | Min-max + standardization | Combined normalization |
| `rolling_norm` | Rolling window normalization | Adaptive to recent dynamics |

## Example Output

```
MAE COMPARISON TABLE
================================================================================
strategy              ETHUSD      UNIUSD      BTCUSD
baseline              15.234567   8.765432    25.345678
percent_change        12.123456   7.654321    23.234567
log_returns           10.012345   6.543210    21.123456  ← BEST
differencing          11.234567   7.123456    22.345678
```

**Result**: `log_returns` gives best MAE improvement across symbols!

## Advanced Usage

### Test Specific Strategies

```bash
.venv312/bin/python3 preaug_sweeps/sweep_runner.py \
    --symbols ETHUSD UNIUSD \
    --strategies baseline log_returns percent_change \
    --epochs 3
```

### Custom Training Parameters

```bash
.venv312/bin/python3 preaug_sweeps/sweep_runner.py \
    --symbols BTCUSD \
    --epochs 5 \
    --batch-size 32 \
    --lookback 128 \
    --horizon 60
```

### Test Augmentation Accuracy

Before running sweeps, test that augmentations work correctly:

```bash
.venv312/bin/python3 preaug_sweeps/test_augmentations.py
```

This verifies roundtrip accuracy (transform → inverse transform).

## Parallel Execution

Run multiple symbols in parallel to save time:

```bash
# Terminal 1
.venv312/bin/python3 preaug_sweeps/sweep_runner.py --symbols ETHUSD --epochs 3 &

# Terminal 2
.venv312/bin/python3 preaug_sweeps/sweep_runner.py --symbols UNIUSD --epochs 3 &

# Terminal 3
.venv312/bin/python3 preaug_sweeps/sweep_runner.py --symbols BTCUSD --epochs 3 &
```

## Using the Best Configuration

Once you've found the best strategy:

```python
import json
from preaug_sweeps.augmentations import get_augmentation

# Load best config
with open("preaugstrategies/best/ETHUSD.json") as f:
    best = json.load(f)

# Create augmentation
augmentation = get_augmentation(best["best_strategy"])

# Apply to your data
df_augmented = augmentation.transform_dataframe(df_original)

# Train model on df_augmented...

# Inverse transform predictions
predictions_original = augmentation.inverse_transform_predictions(
    predictions_augmented,
    context=df_original
)
```

## Troubleshooting

### Out of Memory

Reduce batch size:

```bash
.venv312/bin/python3 preaug_sweeps/sweep_runner.py \
    --symbols ETHUSD \
    --batch-size 8  # Smaller batch
```

### Training Fails

Check logs in `preaug_sweeps/logs/` for errors.

### NaN Loss

Some augmentations may create extreme values. Check:
- Data quality in `trainingdata/`
- Augmentation parameters
- Try different strategies

## Directory Structure

```
preaug_sweeps/
├── README.md                    # This file
├── augmentations/               # Augmentation implementations
│   ├── base_augmentation.py
│   ├── strategies.py
│   └── __init__.py
├── augmented_dataset.py         # Dataset builder
├── sweep_runner.py              # Main sweep orchestrator
├── test_augmentations.py        # Validation tests
├── analyze_results.py           # Results analysis
├── run_sweep.sh                 # Quick start script
├── run_full_sweep.sh            # Production sweep script
├── results/                     # Training results
├── logs/                        # Training logs
├── reports/                     # Summary reports
└── temp/                        # Temporary augmented datasets

preaugstrategies/
└── best/                        # Best configs per symbol
    ├── ETHUSD.json
    ├── UNIUSD.json
    └── BTCUSD.json
```

## Performance Tips

1. **Start Small**: Test with 1-2 strategies first
2. **Use GPU**: Ensure CUDA is available for faster training
3. **Parallel**: Run different symbols in parallel
4. **Monitor**: Check logs in real-time with `tail -f preaug_sweeps/logs/*.log`

## What's Next?

After finding the best strategies:

1. **Validate**: Run longer training (e.g., 10 epochs) with best strategy
2. **Ensemble**: Try combining predictions from multiple augmented models
3. **Fine-tune**: Adjust augmentation parameters
4. **Production**: Integrate best augmentation into main training pipeline

## Documentation

For detailed documentation, see:
- **Main Documentation**: `docs/PREAUGMENTATION_SWEEPS.md`
- **Code**: Read the well-commented source in `preaug_sweeps/augmentations/`

## Examples

### Minimal Example

```python
from preaug_sweeps.sweep_runner import PreAugmentationSweep

sweep = PreAugmentationSweep(
    symbols=["ETHUSD"],
    strategies=["baseline", "log_returns"],
    epochs=1,
)

sweep.run_sweep()
```

### Full Example

```python
sweep = PreAugmentationSweep(
    data_dir="trainingdata",
    symbols=["ETHUSD", "UNIUSD", "BTCUSD"],
    strategies=None,  # All strategies
    epochs=3,
    batch_size=16,
    lookback=64,
    horizon=30,
    validation_days=30,
)

sweep.run_sweep()
```

## Support

For issues or questions:
1. Check `preaug_sweeps/logs/` for error messages
2. Run `test_augmentations.py` to validate setup
3. Review documentation in `docs/PREAUGMENTATION_SWEEPS.md`

## License

Same as the main stock-prediction project.
