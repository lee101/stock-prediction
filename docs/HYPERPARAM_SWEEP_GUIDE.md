# Hyperparameter Sweep System Guide

## Overview

This system provides unified hyperparameter tracking and sweeping across all forecasting models (Toto, Kronos, Chronos2) with automatic best-model selection based on `pct_mae` and other metrics.

## Quick Start

### 1. Run a Sweep

```bash
# Priority configs (recommended first):
python run_sweep.py --model toto --mode priority --max-runs 3

# Quick test sweep:
python run_sweep.py --model toto --mode quick --max-runs 5

# Full grid search (20 configs):
python run_sweep.py --model kronos --mode full --max-runs 20
```

### 2. Select Best Model

```bash
# Get best model overall:
python select_best_model.py

# Get best Toto model:
python select_best_model.py --model toto

# Interactive selection:
python select_best_model.py --interactive

# Export best model path for easy loading:
python select_best_model.py --export-path
```

### 3. Use in Inference

```python
from hparams_tracker import HyperparamTracker

tracker = HyperparamTracker("hyperparams/sweep_results.json")
best = tracker.get_best_model(metric="val_pct_mae", model_name="toto")

print(f"Loading best model from: {best.checkpoint_path}")
# Load and use the model...
```

## Architecture

### Components

1. **`hparams_tracker.py`** - Core tracking system
   - Stores all hyperparameter runs in JSON database
   - Tracks metrics (pct_mae, R², price_mae, etc.)
   - Provides query/comparison APIs

2. **`hyperparams/sweep_configs.py`** - Sweep configurations
   - Defines parameter grids for each model
   - Priority configs based on research/best practices
   - Quick vs full sweep options

3. **`run_sweep.py`** - Automated sweep runner
   - Runs multiple training jobs with different configs
   - Logs all results to tracker
   - Generates comparison reports

4. **`select_best_model.py`** - Model selector for inference
   - Finds best model by any metric
   - CLI and interactive modes
   - Exports model path for easy loading

### Database Schema

```json
{
  "runs": [
    {
      "run_id": "toto_20251112_030500",
      "model_name": "toto",
      "timestamp": "2025-11-12T03:05:00",
      "hyperparams": {
        "patch_size": 32,
        "learning_rate": 0.0003,
        "context_length": 512,
        ...
      },
      "metrics": {
        "val_pct_mae": 0.721,
        "test_pct_mae": 2.237,
        "val_r2": -85.04,
        ...
      },
      "checkpoint_path": "path/to/checkpoint.pt",
      "training_time_seconds": 1800.5,
      "notes": "Improved hyperparams aligned with Toto paper"
    }
  ]
}
```

## Sweep Configurations

### Toto Priority Configs

Based on [Datadog Toto paper](https://arxiv.org/html/2407.07874v1):

```python
# Config 1: Paper-aligned
{
    "patch_size": 32,  # Paper recommendation
    "context_length": 512,  # Paper minimum
    "learning_rate": 3e-4,
    "warmup_steps": 5000,
    "max_epochs": 100,
    "loss_type": "quantile",  # Better for forecasting
}

# Config 2: Longer context
{
    "patch_size": 32,
    "context_length": 1024,  # 2x longer
    "prediction_length": 128,
    "learning_rate": 3e-4,
}

# Config 3: Lower LR
{
    "patch_size": 32,
    "context_length": 512,
    "learning_rate": 1e-4,  # More conservative
    "loss_type": "huber",
}
```

### Kronos Priority Configs

```python
# Balanced config
{
    "context_length": 512,
    "learning_rate": 3e-4,
    "epochs": 100,
    "batch_size": 32,
    "loss": "huber",
}

# Larger context
{
    "context_length": 1024,
    "learning_rate": 1e-4,
    "batch_size": 16,
}
```

### Chronos2 Priority Configs

```python
# Fine-tuning with frozen backbone
{
    "context_length": 512,
    "learning_rate": 5e-5,  # Lower for fine-tuning
    "freeze_backbone": True,
    "lora_r": 16,
}

# Full fine-tuning
{
    "context_length": 512,
    "learning_rate": 1e-4,
    "freeze_backbone": False,
}
```

## Usage Examples

### Basic Sweep

```bash
# Run 3 priority Toto configs:
python run_sweep.py --model toto --mode priority --max-runs 3

# Output:
# 🚀 Starting Toto training: sweep_toto_20251112_030500
#    Config: {'patch_size': 32, 'learning_rate': 0.0003, ...}
# ✅ Training completed: toto_20251112_030500
#    Val pct_MAE: 0.7208
#    Test pct_MAE: 2.2374
```

### Compare Models

```python
from hparams_tracker import HyperparamTracker

tracker = HyperparamTracker()

# Get comparison table
df = tracker.compare_models(
    metrics=["val_pct_mae", "test_pct_mae", "val_r2"],
    model_names=["toto", "kronos"]
)
print(df)

# Get top 5 models
top5 = tracker.get_top_k_models(k=5, metric="val_pct_mae")
for i, run in enumerate(top5, 1):
    print(f"{i}. {run.model_name}: val_pct_mae={run.metrics['val_pct_mae']:.4f}")
```

### Analyze Hyperparameter Impact

```python
# See how learning rate affects performance
df = tracker.get_hyperparameter_impact(
    model_name="toto",
    hyperparam="learning_rate",
    metric="val_pct_mae"
)
print(df)
# Output:
#    run_id               learning_rate  val_pct_mae
# 0  toto_20251112_...   0.0001        0.8500
# 1  toto_20251112_...   0.0003        0.7208
# 2  toto_20251112_...   0.0005        0.9123
```

### Generate Report

```python
tracker = HyperparamTracker()
report = tracker.generate_report("hyperparams/sweep_report.md")
print(report)
```

Output:
```markdown
# Hyperparameter Sweep Report

Generated: 2025-11-12T04:00:00

Total runs: 12

## Best Models by Type (val_pct_mae)

### TOTO
- Run ID: toto_20251112_030500
- Val pct_MAE: 0.7208
- Test pct_MAE: 2.2374
- Val R²: -85.04
- Checkpoint: tototraining/checkpoints/gpu_run/.../best/rank1_val0.007159.pt
- Hyperparams: {
    "patch_size": 32,
    "learning_rate": 0.0003,
    ...
  }

...
```

## Integration with Forecasting

### Load Best Model for Trading

```python
from hparams_tracker import HyperparamTracker
import torch

# Get best model
tracker = HyperparamTracker()
best_toto = tracker.get_best_model(metric="val_pct_mae", model_name="toto")
best_kronos = tracker.get_best_model(metric="val_pct_mae", model_name="kronos")
best_chronos = tracker.get_best_model(metric="val_pct_mae", model_name="chronos2")

# Compare across all models
all_models = [best_toto, best_kronos, best_chronos]
best_overall = min(all_models, key=lambda m: m.metrics.get("val_pct_mae", float('inf')))

print(f"Best model overall: {best_overall.model_name}")
print(f"Val pct_MAE: {best_overall.metrics['val_pct_mae']:.4f}")

# Load the model
checkpoint = torch.load(best_overall.checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

# Use in forecaster...
```

### Ensemble Multiple Top Models

```python
# Get top 3 models
top3 = tracker.get_top_k_models(k=3, metric="val_pct_mae")

# Load all three
models = []
for run in top3:
    model = load_model(run.checkpoint_path)
    models.append(model)

# Ensemble predictions
predictions = []
for model in models:
    pred = model.predict(data)
    predictions.append(pred)

# Average or weighted average
ensemble_pred = np.mean(predictions, axis=0)
```

## Sweep Strategies

### Strategy 1: Coarse to Fine

```bash
# 1. Quick sweep to find promising regions:
python run_sweep.py --model toto --mode quick --max-runs 10

# 2. Identify best hyperparameter ranges
python select_best_model.py --top-k 5

# 3. Create focused grid around best configs
# (edit hyperparams/sweep_configs.py)

# 4. Run focused sweep:
python run_sweep.py --model toto --mode full --max-runs 20
```

### Strategy 2: Priority First

```bash
# 1. Run research-backed priority configs:
python run_sweep.py --model toto --mode priority

# 2. If priority configs work, expand grid search:
python run_sweep.py --model toto --mode full --max-runs 50
```

### Strategy 3: Per-Model Sweep

```bash
# Sweep each model independently:
python run_sweep.py --model toto --mode priority --max-runs 5
python run_sweep.py --model kronos --mode priority --max-runs 5
python run_sweep.py --model chronos2 --mode priority --max-runs 5

# Compare across models:
python select_best_model.py --top-k 10
```

## Customization

### Add New Metrics

Edit `hparams_tracker.py` to track additional metrics:

```python
# During training:
tracker.log_run(
    model_name="toto",
    hyperparams=config,
    metrics={
        "val_pct_mae": 0.72,
        "val_sharpe_ratio": 1.5,  # Custom metric
        "val_max_drawdown": -0.15,  # Custom metric
        ...
    }
)

# At inference:
best = tracker.get_best_model(metric="val_sharpe_ratio", minimize=False)
```

### Add New Model Type

1. Add to `sweep_configs.py`:
```python
MY_MODEL_SWEEP_GRID = {
    "param1": [value1, value2],
    "param2": [value3, value4],
}
```

2. Add to `run_sweep.py`:
```python
def run_my_model_training(config, tracker):
    # Training logic
    ...
    tracker.log_run("my_model", hyperparams=config, metrics=metrics)
```

3. Use:
```bash
python run_sweep.py --model my_model --mode full
```

## Best Practices

1. **Start with Priority Configs** - These are research-backed and likely to work
2. **Track pct_mae** - Most important metric for financial forecasting
3. **Compare Against Naive** - Always check `dm_pvalue_vs_naive`
4. **Use Validation for Selection** - Select on val_pct_mae, report test_pct_mae
5. **Ensemble Top Models** - Top 3-5 models often outperform single best
6. **Document Findings** - Add notes when logging runs
7. **Version Control** - Track git commit with each run

## Troubleshooting

### No models found
```bash
# Check database:
python -c "from hparams_tracker import HyperparamTracker; t=HyperparamTracker(); print(f'{len(t.runs)} runs')"

# Run a sweep first:
python run_sweep.py --model toto --mode quick --max-runs 1
```

### Metrics not comparable
- Ensure all models use same dataset split
- Normalize metrics (e.g., pct_mae vs absolute mae)
- Use consistent evaluation procedure

### Sweep taking too long
```bash
# Use quick mode:
python run_sweep.py --model toto --mode quick --max-runs 3

# Or reduce epochs in config:
# Edit sweep_configs.py: "max_epochs": [30]
```

## File Structure

```
stock-prediction/
├── hparams_tracker.py          # Core tracking system
├── run_sweep.py                 # Sweep runner
├── select_best_model.py         # Model selector
├── hyperparams/
│   ├── sweep_configs.py         # Sweep parameter grids
│   ├── sweep_results.json       # Tracking database
│   └── sweep_report_*.md        # Generated reports
├── docs/
│   └── HYPERPARAM_SWEEP_GUIDE.md  # This file
```

## References

- [Toto Paper](https://arxiv.org/html/2407.07874v1) - Datadog Toto technical report
- [Optuna](https://optuna.org/) - Advanced hyperparameter optimization (future integration)
- [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) - Distributed hyperparameter tuning
