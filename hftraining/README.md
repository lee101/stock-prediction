# HuggingFace-Style Stock Prediction Training

This directory contains a modern, HuggingFace-style training system for stock prediction using state-of-the-art transformers and optimizers.

## Features

### Modern Optimizers
- **GPro**: Gradient Projection with adaptive preconditioning
- **Lion**: EvoLved Sign Momentum optimizer
- **AdaFactor**: Adaptive learning rates with sublinear memory cost
- **LAMB**: Layer-wise Adaptive Moments optimizer
- **Sophia**: Second-order Clipped Stochastic Optimization
- **Adan**: Adaptive Nesterov Momentum Algorithm
- **AdamW**: Adam with decoupled weight decay

### Model Architecture
- Transformer-based trading model with modern architecture
- Multi-head attention with configurable heads and layers
- Positional encoding for time series data
- Multiple prediction heads:
  - Action prediction (buy/hold/sell)
  - Value estimation
  - Price prediction

### Training Features
- Mixed precision training with automatic scaling
- Gradient accumulation and clipping
- Early stopping with customizable patience
- Multiple learning rate schedulers (cosine, linear, polynomial)
- Comprehensive logging with TensorBoard
- Checkpoint saving and resuming
- Data augmentation for time series

### Data Processing
- Advanced technical indicators (MA, EMA, RSI, MACD, Bollinger Bands)
- Automatic data downloading from Yahoo Finance
- Robust data preprocessing and normalization
- Sequence creation for time series prediction
- Support for multiple data sources

## Quick Start

### 1. Basic Training
```bash
cd hftraining
python run_training.py --config_type quick_test
```

### 2. Production Training
```bash
python run_training.py --config_type production --experiment_name my_production_run
```

### 3. Custom Configuration
```python
from config import create_config

# Create custom config
config = create_config("default")
config.model.hidden_size = 768
config.training.optimizer = "gpro"
config.data.symbols = ["AAPL", "GOOGL", "MSFT"]

# Save config
config.save("my_config.json")

# Use config
python run_training.py --config_file my_config.json
```

## Configuration Types

### Default
- Balanced configuration for general use
- 512 hidden size, 8 layers, 16 heads
- GPro optimizer with cosine scheduling
- Standard data augmentation

### Quick Test
- Lightweight configuration for testing
- 128 hidden size, 4 layers, 8 heads
- 1000 max steps, single stock (AAPL)
- Fast evaluation and checkpointing

### Production
- High-performance configuration
- 768 hidden size, 12 layers, 12 heads
- Multiple stocks, longer training
- Comprehensive evaluation

### Research
- Experimental configuration
- Advanced optimizers and techniques
- Profiling and detailed logging
- Multiple reporting backends

## File Structure

```
hftraining/
├── __init__.py              # Package initialization
├── README.md               # This file
├── config.py               # Configuration management
├── data_utils.py           # Data processing utilities
├── hf_trainer.py           # Core training classes
├── modern_optimizers.py    # State-of-the-art optimizers
├── run_training.py         # Main training script
├── train_hf.py            # HuggingFace-style trainer
├── output/                # Training outputs
├── logs/                  # TensorBoard logs
└── cache/                 # Cached data
```

## Usage Examples

### Command Line Options
```bash
# Quick test with debug mode
python run_training.py --config_type quick_test --debug

# Production training with custom output
python run_training.py --config_type production --output_dir ./my_results

# Resume from checkpoint
python run_training.py --resume_from_checkpoint ./output/checkpoint_step_5000.pth

# Custom experiment name
python run_training.py --experiment_name my_experiment_v2
```

### Programmatic Usage
```python
from run_training import run_training
from config import create_config

# Create and customize config
config = create_config("production")
config.training.optimizer = "lion"
config.training.learning_rate = 5e-5
config.data.symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]

# Run training
model, trainer = run_training(config)
```

### Custom Model Configuration
```python
from config import ExperimentConfig, ModelConfig, TrainingConfig

# Create custom model config
model_config = ModelConfig(
    hidden_size=1024,
    num_layers=16,
    num_heads=16,
    dropout=0.1
)

# Create custom training config
training_config = TrainingConfig(
    optimizer="sophia",
    learning_rate=1e-4,
    batch_size=8,
    max_steps=20000
)

# Combine into experiment config
config = ExperimentConfig(
    model=model_config,
    training=training_config
)
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir hftraining/logs
```

### Key Metrics
- `train/loss`: Training loss
- `train/learning_rate`: Current learning rate
- `eval/loss`: Validation loss
- `eval/action_loss`: Action classification loss
- `eval/price_loss`: Price prediction loss

## Data Requirements

### Supported Formats
- CSV files with OHLCV data
- Yahoo Finance symbols (automatic download)
- Custom data loaders

### Expected Columns
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume

### Data Directory Structure
```
trainingdata/
├── AAPL.csv
├── GOOGL.csv
├── MSFT.csv
└── ...
```

## Advanced Features

### Custom Optimizers
```python
from modern_optimizers import get_optimizer

# Use any supported optimizer
optimizer = get_optimizer("gpro", model.parameters(), lr=1e-4)
optimizer = get_optimizer("lion", model.parameters(), lr=1e-4)
optimizer = get_optimizer("sophia", model.parameters(), lr=1e-4)
```

### Data Augmentation
The system includes advanced time series augmentation:
- Gaussian noise injection
- Random scaling
- Technical indicator variations

### Mixed Precision
Automatic mixed precision training for faster training and reduced memory usage:
```python
config.training.use_mixed_precision = True
```

### Gradient Checkpointing
Memory-efficient training for large models:
```python
config.training.gradient_checkpointing = True
```

## Performance Tips

1. **GPU Optimization**: Use CUDA with mixed precision for best performance
2. **Batch Size**: Increase batch size for better GPU utilization
3. **Data Loading**: Use multiple workers for faster data loading
4. **Memory**: Enable gradient checkpointing for large models
5. **Optimizer Choice**: GPro and Lion often converge faster than Adam

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or enable gradient checkpointing
2. **Slow data loading**: Increase `dataloader_num_workers`
3. **NaN losses**: Reduce learning rate or enable gradient clipping
4. **Poor convergence**: Try different optimizers or learning rate schedules

### Debug Mode
```bash
python run_training.py --debug
```
This enables:
- Reduced training steps
- More frequent logging
- Additional debugging information

## Integration with Existing Training

This HuggingFace-style training system is designed to complement the existing `training/` directory while providing a modern, scalable alternative with state-of-the-art techniques.

### Key Differences from `training/`
- HuggingFace-style configuration management
- Modern optimizers (GPro, Lion, Sophia, etc.)
- Advanced data processing pipeline
- Mixed precision training
- Comprehensive logging and monitoring
- Modular, extensible architecture

### Migration Path
1. Start with `quick_test` configuration
2. Gradually increase model size and training duration
3. Experiment with different optimizers
4. Scale to production configuration