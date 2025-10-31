# Toto Stock-Specific Retraining & Kronos Comparison

Complete framework for training optimized, stock-specific Toto models and comparing them against Kronos baseline.

---

## Quick Start

### 1. Run Complete Pipeline (Priority Stocks)

```bash
# Train priority stocks (SPY, MSFT, AAPL, QQQ, GOOG, NVDA, AMD, META, TSLA, BTCUSD, ETHUSD)
./tototraining/run_full_optimization.sh true

# This will:
# - Evaluate baseline (naive model)
# - Train stock-specific Toto models
# - Compare vs Kronos
# - Generate summary reports
```

### 2. Run Complete Pipeline (All Stocks)

```bash
# Train ALL 24 stocks (takes several hours!)
./tototraining/run_full_optimization.sh false
```

### 3. Train Specific Stocks

```bash
# Train only specific stocks
uv run python tototraining/toto_retrain_wrapper.py --stocks SPY NVDA AMD TSLA

# Train priority stocks only
uv run python tototraining/toto_retrain_wrapper.py --priority-only
```

### 4. Compare Specific Stocks

```bash
# Compare single stock
uv run python tototraining/compare_toto_vs_kronos.py --symbol SPY --forecast-horizon 64

# Compare multiple stocks
uv run python tototraining/compare_toto_vs_kronos.py --stocks SPY NVDA AMD --forecast-horizon 64

# Compare all trained models
uv run python tototraining/compare_toto_vs_kronos.py --all --forecast-horizon 64
```

---

## Framework Overview

### Components

#### 1. `toto_retrain_wrapper.py`
**Purpose**: Train stock-specific Toto models with optimized hyperparameters

**Features**:
- Automatic hyperparameter selection based on stock characteristics
- Scales configuration based on dataset size
- Adjusts loss function and LoRA rank based on prediction difficulty
- Saves models compatible with comparison framework
- Generates hyperparameter configs for test_kronos_vs_toto.py

**Key Configuration Logic**:
```python
# Sample size → Context/Prediction lengths
400-500 samples   → context=256,  pred=16
500-1000 samples  → context=512,  pred=32
1000-1500 samples → context=768,  pred=48
1500+ samples     → context=1024, pred=64

# Baseline difficulty → Loss function & LoRA rank
< 10% MAE (easy)   → huber loss, rank=8
10-20% MAE (medium)→ heteroscedastic loss, rank=12
> 20% MAE (hard)   → heteroscedastic loss, rank=16
```

**Usage**:
```bash
# Train all stocks
uv run python tototraining/toto_retrain_wrapper.py

# Train specific stocks
uv run python tototraining/toto_retrain_wrapper.py --stocks SPY NVDA AMD

# Train priority stocks only (faster)
uv run python tototraining/toto_retrain_wrapper.py --priority-only
```

**Outputs**:
- `tototraining/stock_models/[SYMBOL]/` - Trained model checkpoints
- `tototraining/stock_models/[SYMBOL]/training_config.json` - Training configuration
- `tototraining/stock_models/[SYMBOL]/training_metrics.json` - Training metrics
- `hyperparams/toto/[SYMBOL].json` - Config for comparison framework
- `tototraining/stock_models/training_summary.json` - Overall summary

#### 2. `compare_toto_vs_kronos.py`
**Purpose**: Systematically compare Toto vs Kronos models

**Features**:
- Uses existing test_kronos_vs_toto.py framework
- Tracks which model performs better per stock
- Computes average improvements
- Identifies best and worst performers
- Saves detailed comparison results

**Usage**:
```bash
# Compare single stock (forecast horizon = 64)
uv run python tototraining/compare_toto_vs_kronos.py --symbol SPY

# Compare multiple stocks
uv run python tototraining/compare_toto_vs_kronos.py --stocks SPY NVDA AMD

# Compare all available stocks
uv run python tototraining/compare_toto_vs_kronos.py --all

# Custom forecast horizon
uv run python tototraining/compare_toto_vs_kronos.py --all --forecast-horizon 128
```

**Outputs**:
- `comparison_results/[SYMBOL]_comparison.txt` - Full comparison output
- `comparison_results/comparison_summary_h[HORIZON].json` - Summary statistics

#### 3. `run_full_optimization.sh`
**Purpose**: End-to-end automation

**Features**:
- Runs complete pipeline from baseline to comparison
- Handles priority vs full training
- Generates summary reports
- Provides next-step recommendations

**Usage**:
```bash
# Priority stocks only
./tototraining/run_full_optimization.sh true

# All stocks
./tototraining/run_full_optimization.sh false

# With custom forecast horizon
FORECAST_HORIZON=128 ./tototraining/run_full_optimization.sh true
```

---

## Workflow

### Phase 1: Baseline Establishment
```bash
python tototraining/baseline_eval_simple.py
```

**Output**: `tototraining/baseline_results.json`

Establishes naive baseline (persistence model) for all stocks:
- Median baseline: **13.44% MAE**
- Easy stocks: 5-8% MAE (SPY, MSFT, AAPL, QQQ, GOOG)
- Hard stocks: 20-70% MAE (COIN, LCID, QUBT, UNIUSD)

### Phase 2: Stock-Specific Training
```bash
uv run python tototraining/toto_retrain_wrapper.py --priority-only
```

**What happens**:
1. Loads baseline results
2. For each stock:
   - Determines optimal hyperparameters
   - Trains model with LoRA adapters
   - Saves model and metrics
   - Creates hyperparam config
3. Generates training summary

**Expected time**:
- Priority stocks (11 stocks): 2-4 hours
- All stocks (24 stocks): 4-8 hours

### Phase 3: Comparison vs Kronos
```bash
uv run python tototraining/compare_toto_vs_kronos.py --all --forecast-horizon 64
```

**What happens**:
1. Finds all trained Toto models
2. Finds corresponding Kronos configs
3. For each stock:
   - Runs test_kronos_vs_toto.py
   - Parses MAE and latency metrics
   - Determines winner
4. Generates comparison summary

**Metrics tracked**:
- Price MAE (main metric)
- Return MAE
- Inference latency
- Winner (toto/kronos/tie)
- Improvement percentage

### Phase 4: Hyperparameter Optimization

Based on comparison results, refine hyperparameters for stocks where Kronos wins:

```bash
# Example: Retrain TSLA with different config
uv run python tototraining/toto_retrain_wrapper.py --stocks TSLA

# Then compare again
uv run python tototraining/compare_toto_vs_kronos.py --symbol TSLA
```

**Optimization strategies**:
- **If Kronos wins on easy stocks**: Increase epochs, try mse loss
- **If Kronos wins on volatile stocks**: Use heteroscedastic loss, increase LoRA rank
- **If close tie**: Try quantile loss for uncertainty estimation

---

## Expected Results

### Training Performance vs Baseline

Based on foundation model capabilities:

| Stock Type | Baseline MAE% | Expected Toto MAE% | Improvement |
|------------|---------------|-------------------|-------------|
| Easy (SPY, MSFT) | 5-6% | 3-4% | 20-40% |
| Medium (NVDA, AMD) | 12-15% | 9-12% | 15-30% |
| Hard (COIN, TSLA) | 20-24% | 16-20% | 10-20% |
| Extreme (UNIUSD, QUBT) | 30-70% | 25-60% | 5-15% |

### Toto vs Kronos

**Expected outcomes**:
- **Toto advantages**: Longer context (1024+ tokens), better for trend following
- **Kronos advantages**: Autoregressive sampling, better for short-term volatility
- **Likely Toto wins**: SPY, MSFT, AAPL, QQQ (stable trends)
- **Likely Kronos wins**: COIN, TSLA, BTCUSD (high volatility)
- **Competitive**: NVDA, AMD, META, GOOG (mixed characteristics)

---

## File Structure

```
tototraining/
├── README_RETRAINING.md           # This document
├── toto_retrain_wrapper.py        # Stock-specific model training
├── compare_toto_vs_kronos.py      # Comparison framework
├── run_full_optimization.sh       # Complete pipeline
├── baseline_eval_simple.py        # Baseline evaluation
├── baseline_results.json          # Baseline metrics
│
├── stock_models/                  # Trained models
│   ├── SPY/
│   │   ├── training_config.json
│   │   ├── training_metrics.json
│   │   ├── training_output.txt
│   │   └── SPY_model/            # Model checkpoint
│   ├── NVDA/
│   └── ...
│   └── training_summary.json      # Overall summary
│
hyperparams/toto/                  # Configs for comparison
├── SPY.json
├── NVDA.json
└── ...

comparison_results/                # Toto vs Kronos results
├── SPY_comparison.txt
├── NVDA_comparison.txt
└── comparison_summary_h64.json
```

---

## Integration with Existing Framework

The trained models are fully compatible with:

- `test_kronos_vs_toto.py` - Uses hyperparams/toto/*.json configs
- `test_hyperparamtraining_kronos_toto.py` - Hyperparameter sweep framework
- `src/models/toto_wrapper.py` - Model loading and inference

**Example integration**:
```python
# Load trained model via wrapper
from src.models.toto_wrapper import TotoPipeline

# Load stock-specific config
import json
with open('hyperparams/toto/SPY.json', 'r') as f:
    config = json.load(f)

# Create pipeline with trained model
model_path = config['config']['model_path']
toto = TotoPipeline.from_pretrained(
    model_path,
    device='cuda',
    torch_dtype='bfloat16'
)

# Run forecasts
forecast = toto.predict(context, prediction_length=64)
```

---

## Troubleshooting

### Issue: Training fails with "No usable windows"
**Cause**: Context + prediction length exceeds sample count

**Solution**:
```bash
# Check sample count first
wc -l trainingdata/SPY.csv  # 973 lines

# Ensure context + pred < samples
# For SPY: use context=512, pred=32 (544 < 973)
```

### Issue: Comparison shows "No configs found"
**Cause**: Hyperparameter config not generated

**Solution**:
```bash
# Check if config exists
ls hyperparams/toto/SPY.json

# If missing, retrain will regenerate
uv run python tototraining/toto_retrain_wrapper.py --stocks SPY
```

### Issue: Models not improving over baseline
**Possible causes & solutions**:
1. **Too few epochs**: Increase from 10 to 15-20
2. **Wrong loss function**: Try heteroscedastic for volatile stocks
3. **LoRA rank too small**: Increase from 8 to 16
4. **Learning rate too high/low**: Try 1e-4, 3e-4, 5e-4

### Issue: Out of GPU memory
**Solutions**:
```bash
# Reduce batch size
--batch-size 2  # instead of 4

# Reduce context length
--context-length 512  # instead of 1024

# Use smaller LoRA rank
--adapter-r 4  # instead of 8
```

---

## Performance Monitoring

### Track Training Progress
```bash
# Watch training in real-time
tail -f tototraining/stock_models/SPY/training_output.txt

# Check current metrics
cat tototraining/stock_models/SPY/training_metrics.json | jq '.final_val_mape'
```

### Compare Multiple Training Runs
```bash
# Use analyze_results.py for experiment comparison
python tototraining/analyze_results.py
```

### Monitor GPU Usage
```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Check VRAM usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

---

## Next Steps

### 1. Initial Training (Priority Stocks)
```bash
./tototraining/run_full_optimization.sh true
```

### 2. Review Results
```bash
# Check training summary
cat tototraining/stock_models/training_summary.json

# Check comparison results
cat comparison_results/comparison_summary_h64.json
```

### 3. Refine Poor Performers
For stocks where Toto underperforms:
```bash
# Example: Refine TSLA
uv run python tototraining/toto_retrain_wrapper.py --stocks TSLA
# (manually edit StockConfig in code to try different hyperparams)

# Compare again
uv run python tototraining/compare_toto_vs_kronos.py --symbol TSLA
```

### 4. Full Training
Once satisfied with priority stocks:
```bash
./tototraining/run_full_optimization.sh false
```

### 5. Production Deployment
Best models can be deployed via existing wrappers:
- `src/models/toto_wrapper.py` for inference
- Hyperparameter configs in `hyperparams/toto/` for reproducibility

---

## Summary

This framework provides:
✅ Automated stock-specific model training with optimal hyperparameters
✅ Systematic comparison against Kronos baseline
✅ Integration with existing testing infrastructure
✅ Clear path to iterative optimization
✅ Production-ready model deployment

**Expected outcome**: 15-30% average improvement over naive baseline, with competitive or better performance than Kronos on most stocks.

---

*Generated: 2025-10-31*
*Framework for stock-specific Toto model optimization and Kronos comparison*
