# Training Quickstart Guide

## Running Training

### Option 1: Quick Training (5-10 minutes)
For rapid experimentation and iteration:

```bash
# Training on a single stock
cd /nvme0n1-disk/code/stock-prediction
uv run python tototraining/train_quick.py --stock AAPL

# Or use train.py with quick defaults
uv run python tototraining/train.py \
    --train-root trainingdata/AAPL.csv \
    --val-root trainingdata/AAPL.csv \
    --epochs 5 \
    --batch-size 4 \
    --learning-rate 3e-4
```

### Option 2: Full GPU Training (30+ minutes)
For production-ready models with checkpoint management:

```bash
cd /nvme0n1-disk/code/stock-prediction
uv run python tototraining/run_gpu_training.py \
    --train-root trainingdata/train/ \
    --val-root trainingdata/test/ \
    --max-epochs 20 \
    --output-dir tototraining/checkpoints/my_run
```

### Option 3: Kronos Training
Fine-tune the Kronos model on multiple symbols:

```bash
cd /nvme0n1-disk/code/stock-prediction

# Create a Kronos training config
python -c "
from kronostraining.config import KronosTrainingConfig
from pathlib import Path

config = KronosTrainingConfig(
    data_dir=Path('trainingdata'),
    epochs=5,
    batch_size=16,
    learning_rate=4e-5,
    adapter_type='lora',
    adapter_rank=8
)
print(config.as_dict())
" > kronos_config.json

# Run training
uv run python kronostraining/run_training.py --config kronos_config.json
```

---

## Model Training Results

### Toto (Nov 11, 2025 - unseen15 checkpoint)
- **Final Val Loss:** 0.01156
- **pct_MAE:** 1.161% (excellent - less than 1.2% prediction error)
- **price_MAE:** $1.27
- **DM Test vs Naive:** 11.28 (p=0.0, highly significant)
- **Training Time:** 5.1 minutes for 18 epochs
- **Throughput:** 34.6 steps/second

**Model Location:** `/nvme0n1-disk/code/stock-prediction/tototraining/checkpoints/unseen15/best_model.pt`

### Kronos (Nov 11, 2025 - unseen15 checkpoint)
- **Aggregate MAE:** $16.09
- **Symbols:** 15 evaluated
- **Best Performers:**
  - ALGO-USD: $0.038
  - ADA-USD: $0.162
  - ARKG: $1.168
- **Status:** Needs hyperparameter tuning for high-priced stocks

**Model Location:** `/nvme0n1-disk/code/stock-prediction/kronostraining/artifacts/unseen15/`

---

## Training Data

### Available Symbols (130+)
Major stocks: AAPL, MSFT, NVDA, TSLA, META, GOOGL, AMZN, NFLX, etc.
Cryptos: BTCUSD, ETHUSD, ADA-USD, SOL-USD, etc.
ETFs: SPY, QQQ, DIA, IWM, etc.

### Data Files
- **Raw data:** `trainingdata/*.csv` (130+ symbols)
- **Train/test splits:** `trainingdata/train/` and `trainingdata/test/`
- **Holdout set:** `trainingdata/unseen15/` (15 completely unseen symbols)
- **Metadata:** `trainingdata/data_summary.csv`

---

## Key Training Parameters

### Toto Training
```bash
python tototraining/train.py \
    --train-root trainingdata/train/ \
    --val-root trainingdata/test/ \
    --context-length 4096 \
    --prediction-length 64 \
    --batch-size 8 \
    --epochs 20 \
    --learning-rate 3e-4 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --precision bf16 \
    --compile \
    --output-dir tototraining/checkpoints/experiment1
```

### Kronos Training
```python
from kronostraining.config import KronosTrainingConfig

config = KronosTrainingConfig(
    data_dir=Path("trainingdata"),
    lookback_window=64,
    prediction_length=30,
    batch_size=16,
    epochs=5,
    learning_rate=4e-5,
    weight_decay=0.01,
    grad_clip_norm=3.0,
    torch_compile=True,
    precision="bf16",
    adapter_type="lora",
    adapter_rank=8,
    freeze_backbone=True
)
```

---

## Monitoring Training

### View Training Logs
```bash
# Real-time monitoring
tail -f tototraining/checkpoints/unseen15/training.log

# View metrics
cat tototraining/checkpoints/unseen15/final_metrics.json | python -m json.tool
```

### Key Metrics to Watch
1. **Validation Loss:** Should decrease steadily
2. **pct_MAE:** Target < 2% for good models
3. **LR (Learning Rate):** Should be decreasing with schedule
4. **Batch Time:** Consistency indicates stable training

### Example Output
```
Epoch 18 - Train Loss: 0.011010, Val Loss: 0.011563
Batch 30/40, Loss 0.001316, pct_mae 0.531%, price_mae 0.19, LR 0.00029666
Steps/sec: 34.64, Batch time: 28.9ms
```

---

## Evaluating Trained Models

### Load and Evaluate Toto
```python
import torch
from toto.inference.forecaster import TotoForecaster
from traininglib.dynamic_batcher import WindowBatcher

# Load best model
model_path = "tototraining/checkpoints/unseen15/best_model.pt"
model = torch.load(model_path)
model.eval()

# Load data
from tototraining.toto_ohlc_dataloader import TotoOHLCDataLoader
dataloader = TotoOHLCDataLoader(config)
train_dl, val_dl = dataloader.prepare_dataloaders()

# Evaluate
with torch.no_grad():
    total_mae = 0
    count = 0
    for batch in val_dl:
        preds = model(batch)
        mae = torch.mean(torch.abs(preds - batch['targets']))
        total_mae += mae.item()
        count += 1
    
    avg_mae = total_mae / count
    print(f"Validation MAE: {avg_mae:.4f}")
```

### Load and Evaluate Kronos
```python
from kronostraining.trainer import KronosTrainer
from kronostraining.config import KronosTrainingConfig

config = KronosTrainingConfig(
    output_dir=Path("kronostraining/artifacts/unseen15")
)
trainer = KronosTrainer(config)

# Model is already loaded
# Evaluate on validation set
metrics = trainer.evaluate(val_dataset)
print(f"MAE: {metrics['mae']:.2f}")
print(f"RMSE: {metrics['rmse']:.2f}")
```

---

## Hyperparameter Optimization

### Using OpenAI Structured Optimization
```python
from hyperparamopt.optimizer import StructuredOpenAIOptimizer, SuggestionRequest

optimizer = StructuredOpenAIOptimizer()

# Define what to optimize
request = SuggestionRequest(
    hyperparam_schema={
        "type": "object",
        "properties": {
            "learning_rate": {"type": "number", "minimum": 1e-5, "maximum": 1e-3},
            "weight_decay": {"type": "number", "minimum": 1e-3, "maximum": 1e-1},
            "batch_size": {"type": "integer", "enum": [4, 8, 16]},
        },
        "required": ["learning_rate", "weight_decay", "batch_size"]
    },
    objective="minimize validation_loss",
    guidance="Prioritize learning_rate in range 1e-4 to 5e-4",
    n=5  # Get 5 suggestions
)

# Get suggestions
response = optimizer.suggest(request)
for i, suggestion in enumerate(response.suggestions):
    print(f"Option {i+1}: {suggestion}")
```

---

## Using Trained Models for Inference

### Toto Inference
```python
import torch
import numpy as np

# Load model
model_path = "tototraining/checkpoints/unseen15/best_model.pt"
model = torch.load(model_path)
model.eval()

# Prepare input (context_length x features)
context = torch.randn(1, 4096, 5)  # (batch, time, features)

# Predict
with torch.no_grad():
    predictions = model(context)  # (batch, prediction_length, 1)

print(f"Predicted prices: {predictions.squeeze().numpy()}")
```

### Kronos Inference
```python
from external.kronos.model import Kronos, KronosPredictor, KronosTokenizer

# Load model
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
predictor = KronosPredictor(model, tokenizer)

# Prepare time series
prices = np.array([100.5, 100.8, 101.2, 100.9, ...])

# Predict next 30 steps
future = predictor.predict(prices, horizon=30)
print(f"30-step forecast: {future}")
```

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch-size 2

# Enable gradient accumulation
--grad-accum 4  # Effective batch size = 2 * 4 = 8

# Reduce context length
--context-length 2048
```

### Loss Not Decreasing
```bash
# Check learning rate - try lower
--learning-rate 1e-5

# Add weight decay
--weight-decay 1e-1

# Increase warmup
--warmup-steps 1000
```

### Training Too Slow
```bash
# Enable torch.compile
--compile

# Reduce context length
--context-length 1024

# Increase batch size (if VRAM available)
--batch-size 16
```

### Model Not Generalizing
```bash
# Use LoRA for regularization
--adapter lora --freeze-backbone

# Increase weight decay
--weight-decay 5e-2

# Use smaller learning rate
--learning-rate 1e-5
```

---

## Next Steps

1. **Baseline Comparison:** Evaluate Toto vs Kronos on same test set
2. **Hyperparameter Tuning:** Run OpenAI optimization for better MAE
3. **Symbol-Specific Models:** Train separate adapters for high-value symbols
4. **Ensemble:** Combine predictions for better accuracy
5. **Production Deployment:** Export models and integrate into trading system

---

## References

- Full documentation: `/nvme0n1-disk/code/stock-prediction/docs/TRAINING_OVERVIEW.md`
- MAE guide: `/nvme0n1-disk/code/stock-prediction/docs/MAE_CALCULATION_GUIDE.md`
- Training logs: `tototraining/checkpoints/*/training.log`
- Metrics: `tototraining/checkpoints/*/final_metrics.json`

