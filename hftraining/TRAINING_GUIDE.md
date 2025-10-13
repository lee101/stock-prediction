# HF Training System - Complete Guide

## 🎉 System Successfully Implemented!

All requested features have been implemented and tested:

✅ **Profit tracking integrated with TensorBoard logging**  
✅ **Organized directory structure (hftraining/models, tensorboard, etc.)**  
✅ **Base model training on multiple stocks**  
✅ **Fine-tuning pipeline for individual stocks & stock pairs**  
✅ **Amazon Toto forecasts baked into every dataset**  
✅ **Differentiable profit-loss objective driving the core trainer**  
✅ **Portfolio RL module that learns allocations with rebalancing**

## Directory Structure

```
hftraining/
├── models/
│   ├── base/               # Base models trained on all stocks
│   ├── finetuned/          # Stock-specific fine-tuned models
│   └── finetuned/portfolio_pairs/  # Differentiable portfolio RL checkpoints
├── tensorboard/
│   ├── base/           # Base model training logs
│   └── finetuned/      # Fine-tuning logs per stock
├── logs/               # Text logs
├── data/
│   ├── raw/            # Downloaded stock data
│   └── processed/      # Processed features
├── reports/            # Training reports
└── checkpoints/        # Training checkpoints
```

## Quick Start

### 1. Train Single Stock with Profit Tracking + Toto Features

```bash
cd hftraining
python train_with_profit.py --stock AAPL --steps 5000
```

### 2. Train Base Model + Fine-tune + Portfolio RL

```bash
# Train base model on multiple stocks, then fine-tune
python train_with_profit.py --stocks AAPL GOOGL MSFT TSLA --base-model

# After base training, run pairwise portfolio optimisation
python -c "from hftraining.base_model_trainer import BaseModelTrainer;\
trainer = BaseModelTrainer();\
trainer.run_complete_pipeline(pair_symbols=[('AAPL','MSFT'),('GOOGL','AMZN')])"
```

### 3. View Training Metrics in TensorBoard

```bash
tensorboard --logdir hftraining/hftraining/tensorboard
```

Then open browser to http://localhost:6006

**Key Metrics to Monitor:**
- `train/loss` - Training loss (should decrease)
- `eval/loss` - Validation loss
- `profit/total_return` - Simulated profit returns
- `profit/sharpe_ratio` - Risk-adjusted returns
- `profit/win_rate` - Percentage of profitable trades
- `profit/max_drawdown` - Maximum loss from peak

## Profit Tracking & Differentiable Profit Loss

### What's Being Tracked

During training, the system simulates trading based on model predictions:

1. **Total Return**: Cumulative profit/loss percentage
2. **Sharpe Ratio**: Risk-adjusted return metric (higher is better)
3. **Win Rate**: Percentage of profitable trades
4. **Max Drawdown**: Largest peak-to-trough decline
5. **Trade Count**: Number of trades executed

### How It Works

During training the allocation head produces continuous portfolio weights in
[-1, 1]. Realised returns are computed from de-normalised close prices, and the
loss function directly maximises profit while regularising with a Sharpe-like
term. A transaction-cost penalty (configurable via `transaction_cost_bps`) is
applied so allocations remain realistic. TensorBoard logging continues to track
PnL, Sharpe, drawdown, and win rate.

### Amazon Toto Forecast Integration

- Set `config.data.use_toto_forecasts = True` (default) to enrich every sample
  with Toto forecast means and standard deviations for the configured horizon.
- Additional parameters (`toto_model_id`, `toto_device`, `toto_horizon`,
  `toto_num_samples`) live under `config.data.*`.
- If Toto dependencies are missing, the pipeline automatically falls back to a
  differentiable statistical approximation so training can still proceed.

### Portfolio Allocation RL

Use `BaseModelTrainer.train_pair_portfolio()` (passing a list/tuple of symbols)
or the helper `run_complete_pipeline` to learn cross-asset allocations. Example:

```python
trainer.train_pair_portfolio(['AAPL', 'MSFT', 'TSLA'])
```

The differentiable RL trainer:

- Consumes `PairStockDataset` (already Toto-enhanced and normalised)
- Rebalances at the start of each horizon by emitting softmax allocation
  weights that sum to 1
- Optimises real profit with optional risk penalty (`PortfolioRLConfig.risk_penalty`)
- Writes checkpoints under `hftraining/models/finetuned/portfolio_pairs/`

## Training Pipeline Options

### Option 1: Individual Models per Stock

```python
# Train separate models for each stock
from hftraining.train_with_profit import train_single_stock_with_profit

stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
for stock in stocks:
    model, path = train_single_stock_with_profit(stock)
    print(f"Trained {stock}: {path}")
```

### Option 2: Base Model + Fine-tuning

```python
# Train base model on all stocks, then specialize
from hftraining.base_model_trainer import BaseModelTrainer

trainer = BaseModelTrainer(
    base_stocks=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
)

# Train base model
base_model, base_checkpoint = trainer.train_base_model(
    max_steps=10000,
    batch_size=32
)

# Fine-tune for specific stock
finetuned_model, path = trainer.finetune_for_stock(
    stock_symbol='AAPL',
    base_checkpoint_path=base_checkpoint,
    num_epochs=10
)
```

## Configuration

### Model Configuration

```python
config.model.hidden_size = 512      # Model capacity
config.model.num_layers = 8         # Transformer layers
config.model.num_heads = 16         # Attention heads
```

### Training Configuration

```python
config.training.learning_rate = 1e-4
config.training.batch_size = 32
config.training.max_steps = 10000
config.training.warmup_steps = 1000
```

### Profit Tracking Configuration

```python
profit_tracker = ProfitTracker(
    initial_capital=10000,      # Starting capital
    commission=0.001,            # 0.1% per trade
    slippage=0.0005,            # 0.05% slippage
    max_position_size=0.3,      # Max 30% per trade
    stop_loss=0.02,             # 2% stop loss
    take_profit=0.05            # 5% take profit
)
```

## Scaling Up

### For Production Training

```bash
# Large-scale training with more data and longer training
python base_model_trainer.py \
    --base-stocks AAPL GOOGL MSFT AMZN TSLA META NVDA JPM V JNJ \
    --base-steps 50000 \
    --finetune-epochs 20 \
    --batch-size 64
```

### Performance Tips

1. **Use GPU**: Training is ~10x faster on GPU
2. **Mixed Precision**: Enabled by default for memory efficiency
3. **Gradient Accumulation**: For larger effective batch sizes
4. **Data Parallel**: Automatically uses multiple GPUs if available

## Results Analysis

### Loss vs Profit Correlation

The system tracks both:
- **Prediction Loss**: How accurately the model predicts prices
- **Profit Metrics**: How well predictions translate to profits

Key insight: Lower loss doesn't always mean higher profits!
The profit tracking helps optimize for actual trading performance.

### Interpreting Metrics

- **Sharpe Ratio > 1.0**: Good risk-adjusted returns
- **Sharpe Ratio > 2.0**: Excellent performance
- **Win Rate > 55%**: Better than random
- **Max Drawdown < 10%**: Good risk management

## Next Steps

### 1. Backtesting
Test trained models on historical data:
```python
from hfinference import backtest_model
results = backtest_model(model_path, start_date, end_date)
```

### 2. Live Trading
Deploy models for paper/live trading:
```python
from hfinference import LiveTrader
trader = LiveTrader(model_path)
trader.start_paper_trading()
```

### 3. Model Ensemble
Combine multiple models for better predictions:
```python
models = [load_model(path) for path in model_paths]
ensemble_prediction = average_predictions(models)
```

## Troubleshooting

### Issue: Training Loss Not Decreasing
- Reduce learning rate
- Increase warmup steps
- Check data normalization

### Issue: Profit Metrics Negative
- Normal in early training
- Check if model is overfitting (eval loss increasing)
- Adjust position sizing and risk parameters

### Issue: TensorBoard Not Showing Metrics
- Check correct log directory: `hftraining/hftraining/tensorboard`
- Ensure training has progressed past first logging step (100 steps)

## Summary

The training system now:
1. ✅ Tracks actual profit metrics during training
2. ✅ Logs everything to organized TensorBoard locations
3. ✅ Supports base model + fine-tuning pipeline
4. ✅ Scales to multiple stocks efficiently
5. ✅ Correlates loss reduction with profit improvement

Ready for production use! 🚀

---

**View live metrics:**
```bash
tensorboard --logdir hftraining/hftraining/tensorboard
```

**Example training command:**
```bash
python train_with_profit.py --stocks AAPL GOOGL MSFT --base-model --steps 10000
```
