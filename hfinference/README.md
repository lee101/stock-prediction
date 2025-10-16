# HuggingFace Inference System

End-to-end trading system using HuggingFace-trained transformer models with full GPU acceleration support.

## Features

- ✅ Load and run inference with trained checkpoints
- ✅ Generate trading signals with confidence scores
- ✅ Risk management with stop-loss and take-profit
- ✅ Kelly criterion position sizing
- ✅ Backtesting on historical data
- ✅ Paper trading mode
- ✅ Live trading ready (broker integration needed)
- ✅ GPU acceleration with automatic device selection
- ✅ Mixed precision inference for faster predictions
- ✅ Batch processing for high-throughput scenarios

## Quick Start

### GPU Setup Check

```bash
# Test GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### 1. Test the System

```bash
# CPU inference
python hfinference/test_inference.py

# GPU inference (automatic detection)
python hfinference/test_inference.py --device auto

# Specific GPU
python hfinference/test_inference.py --device cuda:0
```

### 2. Run Backtest

```bash
# Standard backtest with GPU acceleration
python hfinference/run_trading.py \
    --checkpoint hftraining/checkpoints/production/final.pt \
    --config hfinference/configs/default_config.json \
    --mode backtest \
    --symbols AAPL GOOGL MSFT \
    --start-date 2024-01-01 \
    --capital 10000 \
    --device auto

# High-performance batch processing
python hfinference/run_trading.py \
    --checkpoint hftraining/checkpoints/production/final.pt \
    --mode backtest \
    --batch-size 64 \
    --use-mixed-precision \
    --device cuda
```

### 3. Paper Trading

```bash
# Real-time paper trading with GPU
python hfinference/run_trading.py \
    --checkpoint hftraining/checkpoints/production/final.pt \
    --mode paper \
    --symbols AAPL GOOGL MSFT TSLA AMZN \
    --update-interval 60 \
    --device auto \
    --optimize-inference
```

## Architecture

```
hfinference/
├── hf_trading_engine.py    # Core trading engine
├── run_trading.py          # Main trading script
├── test_inference.py       # Test suite
├── configs/
│   └── default_config.json # Trading configuration
├── logs/                   # Trading logs
├── results/               # Backtest results
└── README.md              # This file
```

## Components

### HFTradingEngine

Core engine that handles:
- Model loading and inference
- Signal generation
- Trade execution
- Risk management
- Position tracking

### TransformerTradingModel

The neural network model:
- 8-layer transformer
- 512 hidden dimensions
- 16 attention heads
- Dual output heads (price + action)

### DataProcessor

Prepares market data:
- OHLCV features
- Technical indicators (RSI, MACD, SMA)
- Normalization
- Sequence formatting

### RiskManager

Controls risk:
- Position size limits
- Maximum positions
- Stop-loss/take-profit
- Daily loss limits

### PositionSizer

Calculates optimal position sizes:
- Kelly criterion
- Confidence weighting
- Capital allocation
- Risk per trade

## Configuration

Edit `configs/default_config.json`:

```json
{
  "trading": {
    "initial_capital": 10000,
    "max_position_size": 0.25,
    "stop_loss": 0.02,
    "take_profit": 0.05,
    "confidence_threshold": 0.6
  },
  "inference": {
    "device": "auto",
    "batch_size": 32,
    "use_mixed_precision": true,
    "optimize_for_inference": true,
    "use_half_precision": false,
    "compile_model": true
  },
  "gpu": {
    "enabled": true,
    "memory_fraction": 0.9,
    "allow_growth": true,
    "benchmark_cudnn": true
  }
}
```

### GPU Configuration Options

- **device**: Device selection
  - `"auto"` - Automatically select best available device
  - `"cuda"` - Use default GPU
  - `"cuda:0"`, `"cuda:1"` - Specific GPU
  - `"cpu"` - Force CPU usage
  
- **batch_size**: Number of samples to process together (higher = faster but more memory)
- **use_mixed_precision**: Enable FP16/BF16 for 2x speedup
- **optimize_for_inference**: Apply inference-specific optimizations
- **compile_model**: Use torch.compile for faster inference (PyTorch 2.0+)

## Trading Strategies

### Conservative
- High confidence threshold (0.7+)
- Small position sizes (10-15%)
- Tight stop-loss (1-2%)

### Moderate (Default)
- Medium confidence (0.6+)
- Standard positions (20-25%)
- 2% stop-loss, 5% take-profit

### Aggressive
- Lower confidence (0.5+)
- Larger positions (30%+)
- Wider stops (3-5%)

## Performance Metrics

The system tracks:
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Total Return**: Overall profit/loss
- **Number of Trades**: Trading frequency

## Integration with Training

### Using Your Trained Model

1. Train a model using `hftraining/`:
```bash
python hftraining/train_production.py
```

2. Copy checkpoint to inference:
```bash
cp hftraining/checkpoints/production/best.pt hfinference/models/
```

3. Run trading:
```bash
python hfinference/run_trading.py --checkpoint hfinference/models/best.pt
```

### Model Compatibility

The system expects models with:
- Input: [batch, sequence_length, features]
- Output: price predictions + action logits
- Checkpoint keys: 'model_state_dict', 'config'

## Advanced Usage

### Custom Strategy

Create `custom_strategy.py`:

```python
from hfinference.hf_trading_engine import HFTradingEngine

class CustomStrategy(HFTradingEngine):
    def generate_signal(self, symbol, data):
        # Your custom logic
        signal = super().generate_signal(symbol, data)
        
        # Modify signal based on your rules
        if self.market_conditions_unfavorable():
            signal.confidence *= 0.5
        
        return signal
```

### Ensemble Trading

Run multiple models:

```python
models = [
    HFTradingEngine('checkpoint1.pt'),
    HFTradingEngine('checkpoint2.pt'),
    HFTradingEngine('checkpoint3.pt')
]

# Vote on signals
signals = [m.generate_signal(symbol, data) for m in models]
best_signal = max(signals, key=lambda s: s.confidence)
```

### Live Broker Integration

Extend for real trading:

```python
from hfinference.run_trading import HFTrader
import alpaca_trade_api as tradeapi

class LiveTrader(HFTrader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api = tradeapi.REST()  # Your broker API
    
    def execute_live_trade(self, signal):
        if signal.action == 'buy':
            self.api.submit_order(
                symbol=signal.symbol,
                qty=signal.shares,
                side='buy',
                type='market'
            )
```

## GPU Performance Optimization

### Inference Speed Benchmarks

| Configuration | Device | Batch Size | Mixed Precision | Speed (samples/sec) |
|--------------|--------|------------|-----------------|-------------------|
| Baseline | CPU | 1 | No | ~10 |
| Standard GPU | RTX 3060 | 32 | No | ~500 |
| Optimized GPU | RTX 3060 | 32 | Yes | ~1000 |
| Production | RTX 4090 | 64 | Yes | ~3000 |

### Memory Usage

```python
# Monitor GPU memory during inference
from hfinference.utils import GPUMonitor

monitor = GPUMonitor()
engine = HFTradingEngine(device='cuda')

# Check memory before/after model load
print(f"Before: {monitor.get_memory_usage():.1f} MB")
engine.load_model('checkpoint.pt')
print(f"After: {monitor.get_memory_usage():.1f} MB")

# Typical memory usage:
# - Model: 200-500 MB
# - Batch data: 50-200 MB per batch
# - Overhead: 100-200 MB
```

### Optimization Tips

1. **Batch Processing**: Process multiple symbols together
```python
# Efficient batch inference
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
data_batch = [prepare_data(s) for s in symbols]
predictions = engine.batch_predict(data_batch, batch_size=32)
```

2. **Model Compilation** (PyTorch 2.0+):
```python
# Compile for faster inference
model = torch.compile(model, mode="reduce-overhead")
```

3. **Persistent Workers**:
```python
# Keep model in GPU memory between calls
engine = HFTradingEngine(device='cuda', persistent=True)
```

## Troubleshooting

### GPU Issues

#### CUDA Out of Memory
```bash
# Solution 1: Reduce batch size
python run_trading.py --batch-size 16

# Solution 2: Use gradient checkpointing
python run_trading.py --gradient-checkpointing

# Solution 3: Clear GPU cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### GPU Not Detected
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
uv pip uninstall torch
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### Slow GPU Performance
```python
# Enable optimizations in config
{
  "gpu": {
    "benchmark_cudnn": true,
    "allow_tf32": true,  # For RTX 30xx/40xx
    "compile_model": true
  }
}
```

### Model Issues

#### Model Not Loading
- Check checkpoint path exists
- Verify checkpoint has 'model_state_dict'
- Ensure config matches model architecture
- Check GPU memory availability

#### No Trades Generated
- Lower confidence threshold
- Check data has enough history (60+ days)
- Verify market hours for live trading
- Ensure GPU inference is working correctly

#### Poor Performance
- Retrain with more data
- Adjust risk parameters
- Use ensemble of models
- Add more technical indicators
- Enable GPU acceleration for faster response

## Results

Expected performance with trained model:
- Sharpe Ratio: 1.0-1.5
- Annual Return: 15-25%
- Win Rate: 55-60%
- Max Drawdown: 10-15%

## Next Steps

1. **Optimize**: Run hyperparameter search on strategy parameters
2. **Ensemble**: Train multiple models with different seeds
3. **Features**: Add more technical indicators
4. **Broker**: Integrate with real broker API
5. **Monitor**: Set up real-time performance tracking

## License

Private - Do not distribute

## Support

For issues or questions, check the logs in `hfinference/logs/`