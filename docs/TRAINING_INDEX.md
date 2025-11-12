# Training Documentation Index

This directory contains comprehensive documentation about the stock prediction model training setup, including Toto and Kronos frameworks.

## Quick Navigation

### For Beginners
Start here if you're new to the training setup:
1. **TRAINING_QUICKSTART.md** (9 KB) - Quick start commands and examples
2. **TRAINING_OVERVIEW.md** - Section 9 (Current State Summary)
3. **MAE_CALCULATION_GUIDE.md** - Section "MAE Interpretation Guidelines"

### For Training
Running training experiments:
1. **TRAINING_QUICKSTART.md** - "Running Training" section
2. **TRAINING_OVERVIEW.md** - Section 1 (Tototraining) and Section 2 (Kronostraining)
3. **TRAINING_OVERVIEW.md** - Section 8 (Training Strategies)

### For Evaluation & Metrics
Understanding model performance:
1. **MAE_CALCULATION_GUIDE.md** - Complete guide
2. **TRAINING_OVERVIEW.md** - Section 5 (Recent Training Activity)
3. **TRAINING_QUICKSTART.md** - "Evaluating Trained Models" section

### For Hyperparameter Optimization
Tuning model parameters:
1. **TRAINING_OVERVIEW.md** - Section 4 (Hyperparameter Optimization)
2. **TRAINING_QUICKSTART.md** - "Hyperparameter Optimization" section
3. **MAE_CALCULATION_GUIDE.md** - "Improving MAE" section

### For Production Deployment
Getting models ready for use:
1. **TRAINING_QUICKSTART.md** - "Using Trained Models for Inference" section
2. **TRAINING_OVERVIEW.md** - Section 6 (Model Architectures)
3. **TRAINING_QUICKSTART.md** - "Next Steps" section

---

## Document Overview

### TRAINING_OVERVIEW.md (15 KB)
**Comprehensive reference guide for the entire training infrastructure**

Contents:
- Section 1: Tototraining directory structure and features
- Section 2: Kronostraining configuration and setup
- Section 3: Training data availability and format
- Section 4: Hyperparameter optimization infrastructure
- Section 5: Recent training activity and results
- Section 6: Model architectures (Toto and Kronos)
- Section 7: Training utilities and logging
- Section 8: Training strategies and configurations
- Section 9: Current state summary and what's ready
- Section 10: Key files reference

**Best for:** Understanding the full system, finding specific files, checking current status

### MAE_CALCULATION_GUIDE.md (10 KB)
**Deep dive into Mean Absolute Error and evaluation metrics**

Contents:
- MAE variants: price_mae, pct_mae, MAPE
- Implementation in Toto and Kronos code
- Current performance metrics for both models
- Interpretation guidelines (excellent/good/moderate/poor)
- Factors affecting MAE
- Strategies for improving MAE
- Evaluation best practices
- Common issues and solutions
- Metric tracking in checkpoints

**Best for:** Understanding evaluation metrics, improving model performance, debugging metrics

### TRAINING_QUICKSTART.md (9 KB)
**Practical commands and examples for common tasks**

Contents:
- Option 1: Quick training (5-10 minutes)
- Option 2: Full GPU training (30+ minutes)
- Option 3: Kronos training
- Model training results (latest metrics)
- Training data information
- Key training parameters
- Monitoring training in real-time
- Evaluating trained models
- Hyperparameter optimization examples
- Using models for inference
- Troubleshooting guide
- Next steps

**Best for:** Running experiments, copy-paste commands, quick reference

---

## Key Metrics at a Glance

### Toto Model (Latest: Nov 11, 2025)
```
Validation Loss:  0.01156
pct_MAE:         1.161% (Excellent)
price_MAE:       $1.27
Price RMSE:      $1.32
DM Test:         11.28 (p=0.0) - Highly significant
Status:          Ready for production
```

### Kronos Model (Latest: Nov 11, 2025)
```
Aggregate MAE:   $16.09
Symbols tested:  15
Best MAE:        $0.038 (ALGO-USD)
Worst MAE:       $62.53 (AVGO)
Status:          Needs hyperparameter tuning
```

---

## File Locations

### Main Training Scripts
- **Toto:** `/nvme0n1-disk/code/stock-prediction/tototraining/train.py`
- **Kronos:** `/nvme0n1-disk/code/stock-prediction/kronostraining/run_training.py`

### Core Trainers
- **Toto:** `/nvme0n1-disk/code/stock-prediction/tototraining/toto_trainer.py` (79 KB)
- **Kronos:** `/nvme0n1-disk/code/stock-prediction/kronostraining/trainer.py` (19 KB)

### Latest Checkpoints
- **Toto best model:** `tototraining/checkpoints/unseen15/best_model.pt` (1.7 GB)
- **Kronos:** `kronostraining/artifacts/unseen15/`

### Training Data
- **Raw data:** `trainingdata/*.csv` (130+ symbols)
- **Splits:** `trainingdata/train/`, `trainingdata/test/`
- **Holdout:** `trainingdata/unseen15/`
- **Metadata:** `trainingdata/data_summary.csv`

### Configuration
- **Kronos config:** `kronostraining/config.py`
- **Toto CLI:** `tototraining/train.py` (argument parser)

---

## Quick Command Reference

### Start Quick Training
```bash
cd /nvme0n1-disk/code/stock-prediction
uv run python tototraining/train.py \
    --train-root trainingdata/AAPL.csv \
    --val-root trainingdata/AAPL.csv \
    --epochs 5
```

### Start Full GPU Training
```bash
uv run python tototraining/run_gpu_training.py \
    --train-root trainingdata/train/ \
    --val-root trainingdata/test/ \
    --max-epochs 20
```

### Monitor Training
```bash
tail -f tototraining/checkpoints/*/training.log
```

### Check Metrics
```bash
cat tototraining/checkpoints/unseen15/final_metrics.json | python -m json.tool
```

---

## Understanding the Models

### Toto (Datadog Foundation Model)
- **Type:** Transformer-based autoregressive forecaster
- **Context:** 4096 tokens (very long-range dependencies)
- **Horizon:** 64 steps ahead
- **Status:** Production-ready (pct_MAE 1.16%)
- **Best for:** General purpose stock prediction

### Kronos (NeoQuasar Time Series Model)
- **Type:** Tokenized transformer with bucketing
- **Context:** 64 lookback steps
- **Horizon:** 30 steps ahead
- **Status:** Experimental (needs tuning)
- **Best for:** Cryptos and low-priced assets

---

## Performance Expectations

### Toto
- Can achieve <1.2% prediction error on well-behaved stocks
- Takes ~5 minutes to train 18 epochs
- Uses ~1.7 GB VRAM for inference

### Kronos
- Works well on cryptos and low-priced stocks
- Struggles with high-priced stocks (>$100)
- Per-symbol performance varies 0.04-62 dollars

---

## Common Tasks

| Task | Document | Section |
|------|----------|---------|
| Train a model | QUICKSTART | "Running Training" |
| Evaluate metrics | MAE_GUIDE | "Current MAE Performance" |
| Improve MAE | MAE_GUIDE | "Improving MAE" |
| Optimize hyperparams | OVERVIEW | Section 4 |
| Load a checkpoint | QUICKSTART | "Evaluating Trained Models" |
| Run inference | QUICKSTART | "Using Models for Inference" |
| Understand metrics | MAE_GUIDE | "MAE Interpretation Guidelines" |
| Troubleshoot training | QUICKSTART | "Troubleshooting" |

---

## Related Documentation

These guides are also in the docs/ directory:

**Optimization & Performance:**
- OPTIMIZATION_SUMMARY.md
- COMPLETE_OPTIMIZATION_SUMMARY.md
- TORCH_COMPILE_GUIDE.md
- TOTO_OPTIMIZATIONS_SUMMARY.md

**Compilation & Deployment:**
- TOTO_COMPILE_FIXES.md
- INFERENCE_OPTIMIZATION_GUIDE.md
- COMPILATION_OPTIMIZATION_SUMMARY.md

**Strategy & Backtesting:**
- RETRAINING_GUIDE.md
- RETRAINING_QUICKSTART.md

**GPU & Hardware:**
- GPU_SETUP_GUIDE.md

---

## Updated: November 11, 2025

Latest checkpoint: `unseen15`
Latest documentation: This file + referenced files

For the most current information, check the checkpoint logs:
- `tototraining/checkpoints/*/training.log`
- `kronostraining/artifacts/*/metrics/evaluation.json`

