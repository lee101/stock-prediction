# Hyperparameter Sweep System - Quick Start

## TL;DR - 3 Commands to Get Started

```bash
# 1. Run hyperparameter sweep for Toto:
python run_sweep.py --model toto --mode priority --max-runs 3

# 2. Select best model:
python select_best_model.py --model toto

# 3. Use in inference:
python select_best_model.py --export-path
# (loads from .best_model_path in your code)
```

## What This System Does

**Unified hyperparameter tracking** across all your forecasting models (Toto, Kronos, Chronos2) with automatic best-model selection based on `pct_mae`.

**Key Features:**
- ✅ Systematic hyperparameter sweeps with research-backed configs
- ✅ Unified JSON database tracks all runs across all models
- ✅ Automatic best-model selection for inference
- ✅ Compare models apples-to-apples on pct_mae
- ✅ Generate reports and analyze hyperparameter impact

## Complete Workflow

### 1. Run Initial Sweeps

```bash
# Start with priority configs (recommended by research papers):
python run_sweep.py --model toto --mode priority --max-runs 3
python run_sweep.py --model kronos --mode priority --max-runs 3

# Each run:
# - Trains model with specific hyperparameters
# - Logs val_pct_mae, test_pct_mae, R², and more
# - Saves checkpoint path
# - Takes ~5-6 hours for 30-100 epochs
```

### 2. Compare Results

```bash
# View all results:
python select_best_model.py --top-k 10

# Interactive selection:
python select_best_model.py --interactive

# Compare specific model:
python select_best_model.py --model toto --top-k 5
```

### 3. Use Best Model

```python
from hparams_tracker import HyperparamTracker

# Load tracker
tracker = HyperparamTracker("hyperparams/sweep_results.json")

# Get best model
best_toto = tracker.get_best_model(metric="val_pct_mae", model_name="toto")
best_kronos = tracker.get_best_model(metric="val_pct_mae", model_name="kronos")

# Pick overall best
all_models = [best_toto, best_kronos]
best = min(all_models, key=lambda m: m.metrics.get("val_pct_mae", float('inf')))

print(f"Best model: {best.model_name}")
print(f"Val pct_MAE: {best.metrics['val_pct_mae']:.4f}")
print(f"Checkpoint: {best.checkpoint_path}")

# Load and use...
import torch
checkpoint = torch.load(best.checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
```

### 4. Iterate & Improve

```bash
# After analyzing results, run focused sweeps:
# (edit hyperparams/sweep_configs.py to adjust parameter ranges)

python run_sweep.py --model toto --mode full --max-runs 20
```

## Current Results

**Toto Training (30 epochs with improved hyperparams):**
- Val pct_MAE: **0.721%** (was ~5% before improvements!)
- Test pct_MAE: 2.237%
- Status: Still underperforming vs naive (0.076%) but **much better**
- Next: Need more hyperparameter tuning via sweeps

## Sweep Modes

### Priority Mode (Recommended First)
```bash
python run_sweep.py --model toto --mode priority --max-runs 3
```
- Runs 3 research-backed configurations
- Based on Toto/Kronos/Chronos2 papers
- Fastest way to find good starting point
- **Time:** ~5-6 hours per run (3 runs = 15-18 hours)

### Quick Mode (Testing)
```bash
python run_sweep.py --model toto --mode quick --max-runs 5
```
- Reduced parameter space for fast iteration
- Lower epochs (30 instead of 100)
- Good for validating sweep setup
- **Time:** ~2 hours per run (5 runs = 10 hours)

### Full Mode (Comprehensive)
```bash
python run_sweep.py --model toto --mode full --max-runs 20
```
- Grid search over full parameter space
- Randomly samples up to max-runs configs
- Use after priority configs to explore
- **Time:** ~5-6 hours per run (20 runs = 100-120 hours)

## Priority Configurations

### Toto

**Config 1: Paper-Aligned**
- Patch size: 32 (Datadog Toto paper recommendation)
- Context: 512 (paper minimum)
- Learning rate: 3e-4
- Loss: Quantile (better for forecasting)
- **Why:** Directly from research paper

**Config 2: Longer Context**
- Patch size: 32
- Context: 1024 (2x longer)
- Prediction: 128 steps
- **Why:** More historical data = better patterns

**Config 3: Conservative LR**
- Learning rate: 1e-4 (lower)
- Loss: Huber (more stable)
- **Why:** Financial data is noisy, conservative helps

### Kronos

**Config 1: Balanced**
- Context: 512
- LR: 3e-4
- Batch: 32
- **Why:** Proven baseline

**Config 2: Larger Context**
- Context: 1024
- LR: 1e-4 (scaled down)
- Batch: 16 (memory constraint)
- **Why:** Longer history for stocks

## Metrics Tracked

For each run, the system tracks:

**Performance Metrics:**
- `val_pct_mae` - Validation percentage MAE (PRIMARY METRIC)
- `test_pct_mae` - Test percentage MAE
- `val_r2` - Validation R² score
- `test_r2` - Test R² score
- `val_price_mae` - Validation absolute price MAE
- `naive_mae` - Naive baseline for comparison
- `dm_pvalue_vs_naive` - Statistical significance vs naive

**Training Info:**
- All hyperparameters used
- Checkpoint path
- Training time
- Git commit (for reproducibility)
- Custom notes

## File Structure

```
stock-prediction/
├── run_sweep.py                    # Main sweep runner
├── select_best_model.py            # Best model selector
├── hparams_tracker.py              # Tracking system core
├── hyperparams/
│   ├── sweep_configs.py            # Parameter grids
│   ├── sweep_results.json          # Database (auto-created)
│   └── sweep_report_*.md           # Generated reports
├── docs/
│   ├── HYPERPARAM_SWEEP_GUIDE.md   # Full guide
│   └── TOTO_TRAINING_IMPROVEMENTS.md  # Toto-specific improvements
└── SWEEP_QUICKSTART.md             # This file
```

## Common Tasks

### Task: Find Best Model for Trading

```bash
# Get best overall:
python select_best_model.py

# Export path:
python select_best_model.py --export-path

# In your trading code:
model_path = open('.best_model_path').read().strip()
model = load_model(model_path)
```

### Task: Compare Toto vs Kronos

```python
from hparams_tracker import HyperparamTracker

tracker = HyperparamTracker()
df = tracker.compare_models(
    metrics=["val_pct_mae", "test_pct_mae", "val_r2"],
    model_names=["toto", "kronos"]
)
print(df)
```

### Task: See Which LR Works Best

```python
from hparams_tracker import HyperparamTracker

tracker = HyperparamTracker()
df = tracker.get_hyperparameter_impact(
    model_name="toto",
    hyperparam="learning_rate",
    metric="val_pct_mae"
)
print(df)
# Shows: LR vs pct_mae table
```

### Task: Generate Report

```bash
python -c "
from hparams_tracker import HyperparamTracker
t = HyperparamTracker()
t.generate_report('hyperparams/report.md')
"
cat hyperparams/report.md
```

## Next Steps

1. **Run Priority Sweeps** for all models:
   ```bash
   python run_sweep.py --model toto --mode priority --max-runs 3
   python run_sweep.py --model kronos --mode priority --max-runs 3
   ```

2. **Analyze Results** and identify promising hyperparameter ranges:
   ```bash
   python select_best_model.py --top-k 10
   ```

3. **Run Focused Sweeps** around best configs:
   ```bash
   # Edit hyperparams/sweep_configs.py with focused ranges
   python run_sweep.py --model toto --mode full --max-runs 20
   ```

4. **Select Best** and deploy:
   ```bash
   python select_best_model.py --export-path
   # Use .best_model_path in your forecaster/trading system
   ```

5. **Iterate** based on live trading performance

## Tips & Best Practices

1. **Start with Priority** - Don't skip straight to full grid search
2. **Track pct_mae** - It's normalized and comparable across symbols
3. **Check vs Naive** - If `dm_pvalue > 0.05`, model isn't better than naive
4. **Use Validation for Selection** - Pick on val_pct_mae, report test_pct_mae
5. **Ensemble Top 3-5** - Often outperforms single best
6. **Document Learnings** - Add notes when logging runs
7. **Version Control** - System tracks git commits automatically

## Troubleshooting

**Problem: Sweep taking too long**
```bash
# Use quick mode or reduce epochs:
python run_sweep.py --model toto --mode quick --max-runs 3

# Or edit hyperparams/sweep_configs.py:
# Change "max_epochs": [100] to "max_epochs": [30]
```

**Problem: No models found**
```bash
# Check database:
cat hyperparams/sweep_results.json

# Run a test sweep:
python run_sweep.py --model toto --mode quick --max-runs 1
```

**Problem: Models not improving**
```bash
# Check training logs in:
tototraining/checkpoints/gpu_run/*/training.log

# Verify data quality
# Ensure enough training data (>100 symbols recommended)
# Try longer training (more epochs)
```

## Advanced: Manual Logging

You can also log runs manually from your own training scripts:

```python
from hparams_tracker import HyperparamTracker

tracker = HyperparamTracker()

# After training:
tracker.log_run(
    model_name="toto",
    hyperparams={
        "patch_size": 32,
        "learning_rate": 0.0003,
        "context_length": 512,
    },
    metrics={
        "val_pct_mae": 0.721,
        "test_pct_mae": 2.237,
        "val_r2": -85.04,
    },
    checkpoint_path="path/to/checkpoint.pt",
    training_time_seconds=1800,
    notes="Manual run with custom config"
)
```

## References

- **Full Guide:** `docs/HYPERPARAM_SWEEP_GUIDE.md`
- **Toto Improvements:** `docs/TOTO_TRAINING_IMPROVEMENTS.md`
- **Toto Paper:** https://arxiv.org/html/2407.07874v1
- **Sweep Configs:** `hyperparams/sweep_configs.py`
- **Tracker API:** `hparams_tracker.py`
