# Neural Daily Trading - Complete Training Workflow

## Overview

Two-phase training approach to maximize performance while staying current:

**Phase 1**: Hyperparameter search with validation (find best config)
**Phase 2**: Final fit on ALL data including latest (production model)

## Why Two Phases?

1. **Phase 1** uses train/val split to prevent overfitting and find best hyperparameters
2. **Phase 2** trains on 100% of data (including most recent) to maximize recency
3. This combats performance drift as markets evolve

## Phase 1: Hyperparameter Search

### Goal
Find the best model architecture and training parameters using validation.

### Configuration
- `--validation-days 40` (holdout last 40 days for validation)
- Multiple configs tested systematically
- Track both PnL and Sortino ratio

### Run Commands

```bash
# Quick test (3 experiments)
source .venv313/bin/activate
python auto_improve_loop.py --max-experiments 3

# Full search (all hyperparameter combinations)
nohup python auto_improve_loop.py > training_overnight.log 2>&1 &

# Monitor
tail -f training_overnight.log
tail -f improvement_results.jsonl
```

### What Gets Tested

**Architecture**:
- Transformer dimensions: [256, 384, 512]
- Attention heads: [8, 12, 16]
- Layer depth: [4, 6, 8]
- Sequence length: [128, 256, 384]

**Training**:
- Learning rates: [0.00005, 0.0001, 0.0002]
- Dropout: [0.05, 0.1, 0.15]
- Batch sizes: [16, 32, 64]

**Trading Parameters**:
- Price offset: [0.01, 0.025, 0.05]
- Max trade qty: [1.0, 3.0, 5.0]
- Risk threshold: [0.5, 1.0, 2.0]

### Results Format

Each experiment writes to `improvement_results.jsonl`:
```json
{
  "config": {
    "transformer_dim": 384,
    "transformer_heads": 12,
    "transformer_layers": 6,
    "learning_rate": 0.0001,
    ...
  },
  "checkpoint_path": "neuraldailytraining/checkpoints/...",
  "final_equity": 1.6930,
  "pnl": 0.6930,
  "sortino": 1.8873,
  "training_time": 1234.56
}
```

### Analyzing Results

```python
import json
import pandas as pd

# Load all results
results = []
with open('improvement_results.jsonl') as f:
    for line in f:
        results.append(json.loads(line))

df = pd.DataFrame(results)

# Find best performers
print("Top 5 by PnL:")
print(df.nlargest(5, 'pnl')[['pnl', 'sortino', 'final_equity']])

print("\nTop 5 by Sortino (min PnL > 0):")
best_sortino = df[df['pnl'] > 0].nlargest(5, 'sortino')
print(best_sortino[['pnl', 'sortino', 'final_equity']])

# Get winner config
winner = df[df['pnl'] > 0].loc[df[df['pnl'] > 0]['sortino'].idxmax()]
print(f"\nBest config (Sortino={winner['sortino']:.4f}, PnL={winner['pnl']:.4f}):")
print(json.dumps(winner['config'], indent=2))
```

## Phase 2: Final Fit on All Data

### Goal
Train the winning configuration on 100% of data (no validation split) to get the most up-to-date model.

### Why This Matters

- Markets evolve constantly (performance drift)
- Latest data is most relevant for future predictions
- Validation split wastes recent data during Phase 1
- Final model trained on data up to Nov 17, 2025

### Configuration
- `--validation-days 0` (NEW! Use ALL data for training)
- Best config from Phase 1
- More epochs (100 vs 50) since no validation overfitting risk
- Produces production-ready checkpoint

### Run Commands

```bash
# Automatic: Load best config from improvement_results.jsonl
source .venv313/bin/activate
python final_fit_all_data.py

# Manual: Specify config file
python final_fit_all_data.py --config-file best_config.json --output-name prod_model

# With custom results file
python final_fit_all_data.py --results-file my_experiments.jsonl
```

### Output

```
Final model saved to:
neuraldailytraining/checkpoints/final_model/epoch_XXXX.pt
```

This is your **production model** - deploy this one!

## Complete Workflow Example

```bash
# Step 1: Ensure data is current
python update_daily_data.py

# Step 2: Update Chronos forecasts
python update_key_forecasts.py

# Step 3: Run hyperparameter search (Phase 1)
nohup python auto_improve_loop.py > phase1.log 2>&1 &
# Wait ~6-30 hours depending on experiment count

# Step 4: Analyze results
python -c "
import json, pandas as pd
results = [json.loads(line) for line in open('improvement_results.jsonl')]
df = pd.DataFrame(results)
winner = df[df['pnl'] > 0].loc[df[df['pnl'] > 0]['sortino'].idxmax()]
print(f'Winner: Sortino={winner[\"sortino\"]:.4f}, PnL={winner[\"pnl\"]:.4f}')
print('Config:', json.dumps(winner['config'], indent=2))
"

# Step 5: Final fit on all data (Phase 2)
python final_fit_all_data.py --output-name production_$(date +%Y%m%d)

# Step 6: Test final model
PYTHONPATH=. python neuraldailymarketsimulator/simulator.py \
  --checkpoint neuraldailytraining/checkpoints/final_production_*/epoch_*.pt \
  --days 10 --start-date 2025-10-05

# Step 7: Deploy if performance is good!
```

## Maintenance Schedule

### Daily
```bash
# Update data and retrain final model
python update_daily_data.py
python final_fit_all_data.py --results-file improvement_results.jsonl
```

### Weekly
```bash
# Re-run hyperparameter search to find if better configs exist
python auto_improve_loop.py --max-experiments 20
```

### Monthly
```bash
# Full hyperparameter sweep
python auto_improve_loop.py  # All configs
```

## Monitoring Performance Drift

```python
# Compare models across time
import pandas as pd
from pathlib import Path

checkpoints = sorted(Path("neuraldailytraining/checkpoints").glob("final_*/"))

for ckpt_dir in checkpoints[-5:]:  # Last 5 final models
    latest_epoch = max(ckpt_dir.glob("epoch_*.pt"))
    print(f"\nTesting {ckpt_dir.name}:")
    # Run simulation...
```

## Key Differences: Phase 1 vs Phase 2

| Aspect | Phase 1 (Search) | Phase 2 (Final Fit) |
|--------|------------------|---------------------|
| **Goal** | Find best hyperparameters | Train production model |
| **validation_days** | 40 | 0 (ALL data) |
| **Epochs** | 50 | 100 |
| **Training data** | Up to -40 days from end | Up to latest (Nov 17) |
| **Validation data** | Last 40 days | Minimal (just for trainer) |
| **Overfitting risk** | Low (validated) | N/A (using all data) |
| **Output** | Best config | Production checkpoint |
| **Frequency** | Weekly/monthly | Daily |

## Tips

1. **Start small**: Test with `--max-experiments 3` before full sweep
2. **Monitor GPU**: Use `nvidia-smi` to check VRAM usage
3. **Checkpoint storage**: Each experiment creates ~17MB checkpoint
4. **Time estimates**:
   - Single experiment: ~30-60 minutes
   - 10 experiments: ~6-11 hours
   - Full sweep (50+): ~30-55 hours

5. **Best practices**:
   - Always run Phase 2 after finding good Phase 1 config
   - Re-run Phase 2 daily to stay current
   - Re-run Phase 1 weekly to adapt to regime changes

## Validation

After Phase 2, always validate the final model:

```bash
# Run simulation on most recent data
PYTHONPATH=. python neuraldailymarketsimulator/simulator.py \
  --checkpoint <your_final_checkpoint> \
  --days 10 \
  --start-date 2025-10-05

# Compare to baseline
echo "Baseline: PnL=0.6930, Sortino=1.8873"
```

## Troubleshooting

**"No profitable configs found"**
- Lower the PnL threshold in `final_fit_all_data.py`
- Or manually specify a config file

**"Training too slow"**
- Reduce batch size
- Use fewer transformer layers
- Enable mixed precision (`--use-amp`)

**"Out of memory"**
- Reduce sequence length
- Reduce batch size
- Use smaller transformer_dim

**"Performance degraded"**
- Data is stale → Run `update_daily_data.py`
- Market regime changed → Re-run Phase 1 search
- Need more recent data → Phase 2 trains on latest!
