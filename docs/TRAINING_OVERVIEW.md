# Stock Prediction Training Setup Overview

## Project Structure

The project contains a comprehensive training infrastructure with two main model frameworks:
- **Toto** (Datadog's foundation model) - primary focus
- **Kronos** (NeoQuasar's time series model) - alternative framework

## 1. Tototraining Directory

**Location:** `/nvme0n1-disk/code/stock-prediction/tototraining/`

### Training Infrastructure
- **train.py** - Main training entry point with comprehensive CLI arguments
- **train_quick.py** - Rapid iteration script with sensible defaults
- **run_gpu_training.py** - GPU-optimized training launcher for longer runs
- **toto_trainer.py** - Core trainer with distributed training support (79k lines)
- **toto_ohlc_dataloader.py** - OHLC data loading with batch support (44k lines)
- **toto_ohlc_trainer.py** - Specialized OHLC training interface

### Training Features
- Multi-GPU distributed training via PyTorch DDP
- Mixed precision training (bf16, fp16, fp32 options)
- Gradient clipping and memory optimization
- Checkpoint management and recovery
- Learning rate scheduling (WarmupCosine, ReduceLROnPlateau, OneCycleLR)
- Validation metrics and evaluation
- LoRA adapter support for parameter-efficient fine-tuning
- torch.compile integration for performance

### Loss Functions Available
- Huber loss (main)
- Heteroscedastic Gaussian NLL
- Pinball loss
- MAE/MSE variants

### Evaluation Metrics
- Loss-based: MSE, RMSE, MAE
- Percentage-based: pct_MSE, pct_RMSE, pct_MAE, pct_MAPE, pct_R2
- Price-based: price_MSE, price_RMSE, price_MAE
- Diebold-Mariano test vs naive forecasting
- Batch timing and throughput metrics

### Checkpoints & Results

**Latest training (unseen15):** Nov 11, 2025
- Location: `tototraining/checkpoints/unseen15/`
- Epochs: 18 completed
- Training time: ~5.1 minutes
- Final validation loss: **0.01156** (epoch 17 best)

**Metrics Summary (unseen15):**
```
Validation Loss: 0.01156
pct_MAE: 1.161 (1.16% prediction error as % of stock price)
pct_RMSE: 1.218
price_MAE: 1.272 (absolute error in dollars)
price_RMSE: 1.317
pct_MAPE: 7341.06 (high on low-priced instruments)
DM Test vs Naive: 11.28 (highly significant improvement)
Batch throughput: 34.64 steps/sec
```

**Model artifacts:**
- Best model (epoch 17): `best_model.pt` (1.7 GB)
- Latest checkpoint: `latest.pt`
- Top 4 checkpoints tracked by validation loss
- HuggingFace export: `hf_export/` directory
- Preprocessor: `preprocessor.pt`

### Training Configuration (from train.py)
```python
Default arguments:
- Context length: 4096 (past steps)
- Prediction length: 64 (future steps to predict)
- Stride: 64 (sliding window)
- Batch size: 2-16 (depends on GPU)
- Learning rate: 3e-4 (default)
- Epochs: 3-20 (configurable)
- Weight decay: 1e-2
- Grad clipping: 1.0
- Precision: bf16 (recommended for Ada GPUs)
- Compile: enabled on CUDA
- Compile mode: max-autotune
```

### Training Outputs Structure
```
checkpoints/
├── unseen15/              # Latest training run
│   ├── checkpoint_epoch_*.pt
│   ├── best_model.pt
│   ├── final_metrics.json
│   ├── best_records.json
│   ├── training.log
│   └── hf_export/
├── unseen15_export/       # Symlinked to latest
├── gpu_run/               # Previous GPU runs
├── quick/                 # Quick iteration runs
└── checkpoint_registry.json
```

---

## 2. Kronostraining Directory

**Location:** `/nvme0n1-disk/code/stock-prediction/kronostraining/`

### Training Infrastructure
- **trainer.py** - Main Kronos trainer (19k lines)
- **config.py** - Configuration dataclass with 50+ parameters
- **dataset.py** - Multi-ticker dataset loader
- **data_utils.py** - Data utilities
- **run_training.py** - Training launcher

### Kronos-Specific Features
- Tokenizer-based input handling (KronosTokenizer)
- LoRA adapter support for parameter-efficient fine-tuning
- Torch compile enabled by default
- Dynamic window batching with bucket warmup
- Mixed precision (bf16, fp16, fp32)
- Per-symbol evaluation and metrics

### Configuration Parameters (config.py)
```python
Model:
- model_name: "NeoQuasar/Kronos-small"
- tokenizer_name: "NeoQuasar/Kronos-Tokenizer-base"

Data:
- lookback_window: 64
- prediction_length: 30
- validation_days: 30
- min_symbol_length: 180

Training:
- batch_size: 16
- epochs: 3
- learning_rate: 4e-5
- weight_decay: 0.01
- grad_clip_norm: 3.0
- grad_accum_steps: 1

Optimization:
- max_tokens_per_batch: 262,144
- length_buckets: (128, 256, 512)
- horizon_buckets: (20, 32, 64)
- torch_compile: True
- precision: "bf16"

LoRA Adapter:
- adapter_type: "none" or "lora"
- adapter_rank: 8
- adapter_alpha: 16.0
- adapter_dropout: 0.05
- adapter_targets: (embedding.fusion_proj, transformer, dep_layer, head)
- freeze_backbone: True
```

### Checkpoints & Results

**Training runs in artifacts/:**

1. **baseline_lr4e5/** (Oct 29, 2025)
   - Learning rate: 4e-5
   - Epochs: 3
   - Status: Completed

2. **lr2e5_stride10_ep8/** (Oct 29, 2025)
   - Learning rate: 2e-5
   - Stride: 10
   - Epochs: 8
   - Status: Completed

3. **lora_r16_lr1e4_ep10/** (Oct 29, 2025)
   - LoRA rank: 16
   - Learning rate: 1e-4
   - Epochs: 10
   - Status: Completed

4. **unseen15/** (Nov 11, 2025) - Latest
   - Location: `artifacts/unseen15/`
   - Checkpoints: `checkpoints/` directory
   - Adapters: `adapters/` directory
   - Metrics: `metrics/evaluation.json`

**Metrics from unseen15 evaluation (15 symbols):**
```
Aggregate Performance:
- MAE: 16.09
- RMSE: 17.62
- MAPE: 10.88%

Per-symbol MAE range:
- Best: 0.037 (ALGO-USD)
- Worst: 62.53 (AVGO)
- Median: ~12-14 (mid-cap stocks)

Top performers (lowest MAE):
- ALGO-USD: 0.038
- ADA-USD: 0.162
- BAC: 2.383
- ABT: 3.674
- ARKG: 1.168
```

### Kronos Evaluation Metrics (evaluation.json)
```
Aggregate:
- symbols_evaluated: 15
- mae: overall Mean Absolute Error
- rmse: Root Mean Squared Error
- mape: Mean Absolute Percentage Error

Per-symbol breakdown with MAE, RMSE, MAPE
```

---

## 3. Training Data

**Location:** `/nvme0n1-disk/code/stock-prediction/trainingdata/`

### Data Available
- **CSV Files:** ~130+ stock/crypto symbols
  - Stocks: AAPL, MSFT, NVDA, TSLA, META, GOOGL, AMZN, etc.
  - Cryptos: BTCUSD, ETHUSD, ADA-USD, SOL-USD, etc.
  - ETFs: SPY, QQQ, DIA, IWM, etc.

- **Data Format:** OHLCV (Open, High, Low, Close, Volume)
- **Sampling:** Minute-level data (normalized/synthetic)
- **Date Coverage:** 2022-present (varies by symbol)
- **Typical Rows:** 1000-4800 per symbol

### Data Structure
```
trainingdata/
├── AAPL.csv              # Raw data files (~40 KB to ~250 KB)
├── MSFT.csv
├── SPY.csv
├── ... (130+ symbol files)
├── train/                # Train/test split
│   ├── AAPL.csv
│   └── ... (all symbols)
├── test/                 # Test sets (30 rows typically)
│   ├── AAPL.csv
│   └── ... (all symbols)
├── unseen15/             # Holdout test set (15 symbols)
│   ├── train/
│   ├── test/
│   └── val/
├── data_summary.csv      # Metadata on all symbols
├── asset_metadata.json   # Symbol metadata
├── cache/                # Cached data
└── puffer_subset/        # Subset for specific training
```

### Data Summary Statistics (data_summary.csv)
```
Fields:
- symbol: Ticker name
- latest_date: Most recent data point
- total_rows: Total observations
- train_rows: Training set size (typically ~97%)
- test_rows: Test set size (typically 30)
- train_file: Path to train CSV
- test_file: Path to test CSV

Examples:
- AAPL: 4801 rows, 4081 train, 720 test (extended history)
- Most symbols: ~1000 rows, ~970 train, 30 test
- Cryptos: ~1450 rows (hourly data more available)
```

---

## 4. Hyperparameter Optimization

**Location:** `/nvme0n1-disk/code/stock-prediction/hyperparamopt/`

### Optimization Infrastructure
- **optimizer.py** - StructuredOpenAIOptimizer using OpenAI structured outputs
- **Storage system** - RunLog for tracking optimization history
- **SuggestionRequest/SuggestionResponse** - Structured I/O with JSON schema

### Features
- LLM-guided hyperparameter search (GPT-4/5 based)
- Objective-based optimization (minimize loss, maximize sharpe ratio, etc.)
- Natural language guidance for constraints
- History-aware suggestions (context window up to 100 trials)
- Batch suggestions (propose multiple candidates at once)

### Other Optimization Files
- `optimize_toto_further.py` - Advanced Toto parameter tuning
- `hyperparameter_optimizer.py` - General optimization framework
- `validate_hyperparams_holdout.py` - Hyperparameter validation
- `test_hyperparamtraining_kronos_toto.py` - Comparative optimization tests
- Benchmark results in `full_optimization_results.json`

---

## 5. Recent Training Activity

### Most Recent Training Run
**Date:** November 11, 2025, 09:12-09:16 UTC

**Toto (unseen15) - 18 epochs:**
```
Epoch progression (sample):
- Epoch 1: Train Loss 0.01151 → Val Loss 0.01289
- Epoch 5: Train Loss 0.01151 → Val Loss 0.01279
- Epoch 10: Train Loss 0.01134 → Val Loss 0.01243
- Epoch 17: Train Loss 0.01078 → Val Loss 0.01169 (best)
- Epoch 18: Train Loss 0.01101 → Val Loss 0.01156 (final)

Per-batch metrics (Epoch 18, sample):
- Batch 0: Loss 0.002073, pct_mae 0.834%
- Batch 10: Loss 0.001580, pct_mae 0.637%
- Batch 30: Loss 0.001316, pct_mae 0.531%
```

**Training Performance:**
- Total time: 5.11 minutes for 18 epochs
- Batch time: ~28.9 ms average
- Throughput: 34.64 steps/second
- No test set evaluation (validation-only)

**Recent Kronos Run:**
- Last update: Nov 11, 2025
- Status: unseen15 checkpoint completed

---

## 6. Model Architectures

### Toto Model
- **Base Model:** Datadog-Toto-Open-Base-1.0
- **Type:** Transformer-based autoregressive time series model
- **Context:** Supports very long contexts (4096+ tokens)
- **Capabilities:**
  - Multi-horizon forecasting (64 steps)
  - Foundation model pre-trained on large financial data
  - Supports fine-tuning with LoRA adapters
  - Inference via TotoForecaster wrapper

### Kronos Model
- **Base Model:** NeoQuasar/Kronos-small
- **Type:** Transformer with specialized tokenizer
- **Input:** Tokenized time series (KronosTokenizer)
- **Key Features:**
  - Looks back 64 steps for prediction
  - Predicts 30 steps ahead
  - Supports dynamic bucketing by context/horizon length
  - LoRA-efficient fine-tuning available

---

## 7. Training Utilities & Logging

### Logging System
- **Training logs:** `training.log` files in checkpoint directories
- **Log format:** Timestamps, epoch/batch info, loss values, metrics
- **Metrics tracked:**
  - Training/validation loss
  - pct_mae (percentage MAE - key metric)
  - price_mae (absolute price error)
  - Learning rates
  - Batch timing

### Supporting Modules (traininglib/)
- `compile_wrap.py` - torch.compile wrapper
- `optim_factory.py` - Optimizer factory (AdamW, etc.)
- `runtime_flags.py` - Runtime optimization flags (bf16_supported, etc.)
- `schedules.py` - Learning rate schedules (WarmupCosine)
- `prof.py` - Profiling utilities
- `prefetch.py` - GPU prefetching (CudaPrefetcher)
- `ema.py` - Exponential Moving Average
- `losses.py` - Custom loss functions
- `dynamic_batcher.py` - Window-based dynamic batching

### Test Infrastructure
- Comprehensive test suite: `test_*.py` files in tototraining/
- Integration tests: `test_integration.py`, `test_toto_integration.py`
- Performance tests: `test_performance.py`
- Regression tests: `test_regression.py`
- Data quality tests: `test_data_quality.py`

---

## 8. Training Strategies & Configurations

### Available Training Modes

1. **Quick Training** (train_quick.py)
   - Purpose: Rapid iteration on architectures
   - Time: ~5-10 minutes per run
   - Use: Early experimentation

2. **Full GPU Training** (run_gpu_training.py)
   - Purpose: Production-grade training
   - Duration: 30+ minutes
   - Features: Top-4 checkpoint tracking
   - Suitable for: Final model selection

3. **Hyperparameter Search**
   - LLM-guided optimization via OpenAI API
   - Objective: Minimize validation loss or maximize metrics
   - History context: Previous trial results

### Key Training Features

**Memory Optimization:**
- CPU offloading for all pipelines
- Attention slicing
- VAE slicing
- Gradient checkpointing
- Dynamic batching by context/horizon length

**Performance Optimization:**
- torch.compile with max-autotune mode
- bf16 mixed precision (RTX 3090/Ada GPU friendly)
- Fused operators where available
- Flash attention support

**Regularization:**
- Weight decay (1e-2 standard)
- Gradient clipping (1.0)
- Dropout in adapters (0.05)
- Learning rate scheduling

---

## 9. Current State Summary

### Models Ready for Training
1. **Toto** - Fully trained, best checkpoint identified
   - Status: Latest run complete (Nov 11)
   - Quality: Good validation metrics
   - Next step: Longer training, hyperparameter tuning

2. **Kronos** - Trained on limited subset
   - Status: unseen15 checkpoint completed
   - Quality: Per-symbol MAE 0.04-62.5 (needs tuning)
   - Next step: LoRA tuning, full symbol training

### Data Status
- Training pairs available: 130+ symbols
- Train/test split: Pre-computed
- Holdout set: 15 symbols (unseen15)
- Quality: Synthetic/normalized OHLCV data

### Infrastructure Ready
- Distributed training support (DDP)
- GPU memory management (bf16, compile)
- Checkpoint recovery
- Evaluation metrics framework
- Logging and tracking system
- Hyperparameter optimization framework

### What Can Be Done Next
1. Continue training Toto on full dataset
2. Tune Kronos hyperparameters for better per-symbol performance
3. Run hyperparameter sweep via OpenAI structured optimization
4. Fine-tune with LoRA adapters on specific high-value symbols
5. Ensemble Toto and Kronos predictions
6. Evaluate on holdout test sets
7. Deploy best models to production

---

## 10. Key Files Reference

### Essential Scripts
- `tototraining/train.py` - Main training entry point
- `tototraining/run_gpu_training.py` - GPU training launcher
- `kronostraining/trainer.py` - Kronos trainer
- `kronostraining/config.py` - Kronos configuration

### Data Loading
- `tototraining/toto_ohlc_dataloader.py` - OHLC data loading
- `tototraining/data.py` - Dataset utilities

### Evaluation
- `tototraining/toto_trainer.py` - Full trainer with evaluation
- Individual symbol metrics in checkpoints

### Configuration
- `traininglib/` - Core training utilities
- `kronostraining/config.py` - Kronos config
- `tototraining/train.py` - Toto argument parser

