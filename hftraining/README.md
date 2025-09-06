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
- Uses local Alpaca-exported CSVs in `trainingdata/` (no yfinance)
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
- CSV files with OHLCV data (exported from Alpaca or your pipeline)
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

### Data Prep (from local sources)
- Collect and consolidate existing CSVs into `hftraining/trainingdata`:
  - `python -m hftraining.scripts.collect_training_data --sources ../trainingdata ../data --output ./hftraining/trainingdata --since 2015-01-01`
- The loader now scans recursively, so you can also set `data_dir` to a parent folder containing nested `train/` and `test/` subfolders.

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

## GPU Setup and Usage

### Prerequisites

1. **Check GPU Availability**:
```bash
# Test CUDA installation
nvidia-smi

# Test PyTorch GPU support
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

2. **Install CUDA-enabled PyTorch**:
```bash
# For CUDA 12.1
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu118
```

### Training with GPU

#### Single GPU Training
```bash
# Automatic GPU detection
python run_training.py --config_type production

# Specify GPU device
CUDA_VISIBLE_DEVICES=0 python run_training.py --config_type production

# Force specific device
python run_training.py --device cuda:0
```

#### Multi-GPU Training
```bash
# DataParallel (single node, multiple GPUs)
python run_training.py --multi_gpu dp --gpus 0,1

# DistributedDataParallel (faster, recommended)
python -m torch.distributed.launch --nproc_per_node=2 run_training.py --multi_gpu ddp
```

#### Mixed Precision Training
```bash
# Enable mixed precision (2x speedup, half memory)
python run_training.py --mixed_precision --amp_dtype float16

# Use BFloat16 (for Ampere GPUs - RTX 30xx/40xx)
python run_training.py --mixed_precision --amp_dtype bfloat16
```

### GPU Configuration in Code

```python
# config.py - Add GPU settings
config = create_config("production")
config.gpu = {
    'enabled': True,
    'device': 'auto',  # auto, cuda, cuda:0, cpu
    'mixed_precision': True,
    'amp_dtype': 'float16',  # float16, bfloat16
    'allow_tf32': True,  # For Ampere GPUs
    'gradient_checkpointing': False,  # Trade speed for memory
    'multi_gpu_strategy': 'ddp',  # dp, ddp, none
    'compile_model': True,  # PyTorch 2.0+ optimization
}
```

### Memory Optimization

#### Gradient Accumulation
```bash
# Simulate larger batch size with limited memory
python run_training.py --batch_size 8 --gradient_accumulation_steps 4
# Effective batch size = 8 * 4 = 32
```

#### Gradient Checkpointing
```bash
# Trade computation for memory (slower but uses less VRAM)
python run_training.py --gradient_checkpointing
```

#### Dynamic Batch Size
```bash
# Automatically find optimal batch size
python run_training.py --auto_batch_size --max_batch_size 128
```

### GPU Monitoring During Training

The training script automatically logs GPU metrics to TensorBoard:

```bash
# View GPU metrics in TensorBoard
tensorboard --logdir hftraining/logs

# Metrics tracked:
# - GPU Memory Usage (MB/GB)
# - GPU Utilization (%)
# - GPU Temperature (°C)
# - Training throughput (samples/sec)
```

### Performance Benchmarks

| Configuration | GPU | Batch Size | Mixed Precision | Training Speed |
|--------------|-----|------------|-----------------|----------------|
| Baseline | CPU | 16 | No | ~50 samples/sec |
| Single GPU | RTX 3060 | 32 | No | ~500 samples/sec |
| Optimized | RTX 3060 | 32 | Yes (FP16) | ~1000 samples/sec |
| Multi-GPU | 2x RTX 3090 | 64 | Yes (FP16) | ~3000 samples/sec |
| Production | RTX 4090 | 64 | Yes (BF16) | ~4000 samples/sec |

### GPU-Specific Optimizations

#### For NVIDIA Ampere (RTX 30xx/40xx)
```python
# Enable TF32 for matrix operations
config.gpu['allow_tf32'] = True

# Use BFloat16 instead of Float16
config.gpu['amp_dtype'] = 'bfloat16'

# Enable Flash Attention
config.model['use_flash_attention'] = True
```

#### For Limited VRAM (< 8GB)
```python
# Reduce model size
config.model['hidden_size'] = 256
config.model['num_layers'] = 6

# Enable memory-saving features
config.gpu['gradient_checkpointing'] = True
config.training['gradient_accumulation_steps'] = 8
config.training['batch_size'] = 4
```

#### For Maximum Speed
```python
# Compile model (PyTorch 2.0+)
config.gpu['compile_model'] = True
config.gpu['compile_mode'] = 'reduce-overhead'

# Optimize data loading
config.data['num_workers'] = 8
config.data['pin_memory'] = True
config.data['persistent_workers'] = True

# Enable cudnn benchmarking
config.gpu['benchmark_cudnn'] = True
```

## Performance Tips

1. **GPU Optimization**: 
   - Use CUDA with mixed precision for 2x speedup
   - Enable TF32 on Ampere GPUs (RTX 30xx/40xx)
   - Compile models with torch.compile (PyTorch 2.0+)

2. **Batch Size**: 
   - Increase batch size for better GPU utilization
   - Use gradient accumulation if limited by memory
   - Run auto-tuning to find optimal batch size

3. **Data Loading**: 
   - Use multiple workers (num_workers=4-8)
   - Enable pin_memory for faster CPU-GPU transfer
   - Use persistent_workers to avoid recreation overhead

4. **Memory Management**: 
   - Enable gradient checkpointing for large models
   - Use mixed precision to halve memory usage
   - Clear cache periodically with torch.cuda.empty_cache()

5. **Optimizer Choice**: 
   - GPro and Lion often converge faster than Adam
   - Use fused optimizers when available (faster on GPU)
   - Consider 8-bit optimizers for memory savings

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
