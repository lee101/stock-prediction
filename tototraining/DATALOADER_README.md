# Toto OHLC DataLoader System

A comprehensive dataloader system for training the Toto transformer model on OHLC stock data with advanced preprocessing, normalization, and cross-validation capabilities.

## Features

### 🚀 Core Functionality
- **OHLC Data Processing**: Handles Open, High, Low, Close, Volume data
- **Technical Indicators**: RSI, Moving Averages, Price Momentum, Volatility
- **Multi-Symbol Support**: Load and process data from multiple stock symbols
- **Time Series Validation**: Proper train/validation/test splits respecting temporal order
- **Cross-Validation**: Time series cross-validation with configurable folds
- **Batch Processing**: Efficient PyTorch DataLoader integration

### 📊 Data Preprocessing
- **Normalization**: Standard, MinMax, and Robust scaling methods
- **Missing Value Handling**: Interpolation, dropping, or zero-filling
- **Outlier Detection**: Z-score based outlier removal
- **Feature Engineering**: Automatic technical indicator calculation
- **Data Validation**: Ensures proper OHLC relationships and data quality

### ⚙️ Configuration Management
- **JSON Configuration**: Save and load complete configurations
- **Flexible Parameters**: Extensive hyperparameter control
- **Reproducible Results**: Random seed management
- **Environment Adaptation**: Automatic fallbacks for missing dependencies

## Quick Start

### 1. Install Dependencies

```bash
pip install torch pandas scikit-learn numpy
```

### 2. Prepare Data Structure

```
tototraining/
├── trainingdata/
│   ├── train/
│   │   ├── AAPL.csv
│   │   ├── GOOGL.csv
│   │   └── ...
│   └── test/
│       ├── AAPL.csv
│       ├── GOOGL.csv
│       └── ...
```

### 3. Generate Sample Data

```bash
python generate_sample_data.py
```

### 4. Basic Usage

```python
from toto_ohlc_dataloader import TotoOHLCDataLoader, DataLoaderConfig

# Create configuration
config = DataLoaderConfig(
    batch_size=32,
    sequence_length=96,
    prediction_length=24,
    add_technical_indicators=True,
    normalization_method="robust"
)

# Initialize dataloader
dataloader = TotoOHLCDataLoader(config)

# Prepare PyTorch DataLoaders
dataloaders = dataloader.prepare_dataloaders()

# Use in training loop
for batch in dataloaders['train']:
    # batch is a MaskedTimeseries object compatible with Toto model
    series = batch.series  # Shape: (batch_size, n_features, sequence_length)
    # ... training code ...
```

## Configuration Options

### DataLoaderConfig Parameters

#### Data Paths
- `train_data_path`: Path to training data directory
- `test_data_path`: Path to test data directory

#### Model Parameters
- `patch_size`: Size of patches for Toto model (default: 12)
- `stride`: Stride for patch extraction (default: 6)
- `sequence_length`: Input sequence length (default: 96)
- `prediction_length`: Prediction horizon (default: 24)

#### Preprocessing
- `normalization_method`: "standard", "minmax", or "robust" (default: "robust")
- `handle_missing`: "drop", "interpolate", or "zero" (default: "interpolate")
- `outlier_threshold`: Z-score threshold for outlier removal (default: 3.0)

#### Features
- `ohlc_features`: List of OHLC columns (default: ["Open", "High", "Low", "Close"])
- `additional_features`: Additional features like Volume (default: ["Volume"])
- `target_feature`: Target column for prediction (default: "Close")
- `add_technical_indicators`: Enable technical indicators (default: True)

#### Technical Indicators
- `rsi_period`: RSI calculation period (default: 14)
- `ma_periods`: Moving average periods (default: [5, 10, 20])

#### Training Parameters
- `batch_size`: Batch size for training (default: 32)
- `validation_split`: Fraction for validation split (default: 0.2)
- `test_split_days`: Days for test set when splitting (default: 30)

#### Cross-Validation
- `cv_folds`: Number of CV folds (default: 5)
- `cv_gap`: Gap between train/val in CV (default: 24)

## Advanced Usage

### Custom Configuration

```python
# Advanced configuration
config = DataLoaderConfig(
    sequence_length=120,
    prediction_length=30,
    
    # Advanced preprocessing
    normalization_method="robust",
    outlier_threshold=2.5,
    add_technical_indicators=True,
    ma_periods=[5, 10, 20, 50],
    
    # Data filtering
    min_sequence_length=200,
    max_symbols=50,
    
    # Cross-validation
    cv_folds=5,
    cv_gap=48,
    
    # Performance
    batch_size=64,
    num_workers=4,
    pin_memory=True
)

# Save configuration
config.save("my_config.json")

# Load configuration
loaded_config = DataLoaderConfig.load("my_config.json")
```

### Cross-Validation

```python
# Get cross-validation splits
cv_splits = dataloader.get_cross_validation_splits(n_splits=5)

for fold, (train_loader, val_loader) in enumerate(cv_splits):
    print(f"Fold {fold + 1}: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    
    # Train model on this fold
    # ... training code ...
```

### Feature Information

```python
# Get detailed feature information
feature_info = dataloader.get_feature_info()
print(f"Features: {feature_info['feature_columns']}")
print(f"Number of features: {feature_info['n_features']}")
print(f"Target: {feature_info['target_feature']}")
```

### Preprocessor Management

```python
# Save fitted preprocessor
dataloader.save_preprocessor("preprocessor.pth")

# Load preprocessor for inference
new_dataloader = TotoOHLCDataLoader(config)
new_dataloader.load_preprocessor("preprocessor.pth")
```

## Data Format

### Expected CSV Format

```csv
timestamp,Open,High,Low,Close,Volume
2025-01-01 00:00:00,100.0,101.0,99.0,100.5,1000000
2025-01-01 01:00:00,100.5,102.0,100.0,101.5,1200000
...
```

### Required Columns
- `timestamp`: Datetime column (optional, will generate if missing)
- `Open`, `High`, `Low`, `Close`: OHLC price data
- `Volume`: Volume data (optional, will generate dummy values if missing)

### Generated Features (when `add_technical_indicators=True`)
- RSI (Relative Strength Index)
- Moving averages and ratios
- Price momentum (1 and 5 periods)
- Volatility (20-period rolling std)
- OHLC ratios (HL ratio, OC ratio)

## Output Format

The dataloader returns `MaskedTimeseries` objects compatible with the Toto model:

```python
class MaskedTimeseries:
    series: torch.Tensor              # Shape: (batch, features, time)
    padding_mask: torch.Tensor        # Shape: (batch, features, time)
    id_mask: torch.Tensor            # Shape: (batch, features, 1)
    timestamp_seconds: torch.Tensor   # Shape: (batch, features, time)
    time_interval_seconds: torch.Tensor # Shape: (batch, features)
```

## Examples

See the included example files:

- `toto_ohlc_dataloader.py` - Main dataloader with built-in test
- `example_usage.py` - Comprehensive examples
- `generate_sample_data.py` - Sample data generation

Run examples:

```bash
# Test basic functionality
python toto_ohlc_dataloader.py

# Run comprehensive examples
python example_usage.py

# Generate sample data
python generate_sample_data.py
```

## Integration with Toto Model

The dataloader is designed to work seamlessly with the existing Toto trainer:

```python
from toto_ohlc_dataloader import TotoOHLCDataLoader, DataLoaderConfig
from toto_ohlc_trainer import TotoOHLCTrainer, TotoOHLCConfig

# Create compatible configurations
dataloader_config = DataLoaderConfig(
    sequence_length=96,
    prediction_length=24,
    batch_size=32
)

model_config = TotoOHLCConfig(
    sequence_length=96,
    prediction_length=24,
    patch_size=12,
    stride=6
)

# Initialize components
dataloader = TotoOHLCDataLoader(dataloader_config)
trainer = TotoOHLCTrainer(model_config)

# Get dataloaders
dataloaders = dataloader.prepare_dataloaders()

# Train model
# trainer.train_with_dataloaders(dataloaders)
```

## Performance Considerations

### Memory Usage
- Use `batch_size` to control memory usage
- Enable `pin_memory=True` for GPU training
- Adjust `num_workers` based on CPU cores

### Processing Speed
- Increase `num_workers` for faster data loading
- Use `drop_last=True` for consistent batch sizes
- Consider `max_symbols` to limit dataset size during development

### Storage
- CSV files are loaded into memory
- Consider data compression for large datasets
- Use appropriate `min_sequence_length` to filter short series

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'toto'**
   - The dataloader includes fallback implementations for testing
   - Install the Toto model package for full functionality

2. **TypeError: 'type' object is not subscriptable**
   - Older Python versions may have type annotation issues
   - Fallback implementations are included

3. **Memory errors with large datasets**
   - Reduce `batch_size` or `max_symbols`
   - Increase system memory or use data streaming

4. **Slow data loading**
   - Increase `num_workers` (but not too high)
   - Use SSD storage for data files
   - Consider data preprocessing and caching

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Use single worker for debugging:

```python
config = DataLoaderConfig(num_workers=0)
```

## Contributing

When extending the dataloader:

1. Maintain compatibility with `MaskedTimeseries` format
2. Add proper error handling and logging
3. Include tests for new features
4. Update configuration options
5. Document new parameters and usage

## License

This code follows the same license as the Toto model (Apache-2.0).