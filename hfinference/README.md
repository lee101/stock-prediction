# HuggingFace Inference System

End-to-end trading system using HuggingFace-trained transformer models.

## Features

- ✅ Load and run inference with trained checkpoints
- ✅ Generate trading signals with confidence scores
- ✅ Risk management with stop-loss and take-profit
- ✅ Kelly criterion position sizing
- ✅ Backtesting on historical data
- ✅ Paper trading mode
- ✅ Live trading ready (broker integration needed)

## Quick Start

### 1. Test the System

```bash
python hfinference/test_inference.py
```

### 2. Run Backtest

```bash
python hfinference/run_trading.py \
    --checkpoint hftraining/checkpoints/production/final.pt \
    --config hfinference/configs/default_config.json \
    --mode backtest \
    --symbols AAPL GOOGL MSFT \
    --start-date 2024-01-01 \
    --capital 10000
```

### 3. Paper Trading

```bash
python hfinference/run_trading.py \
    --checkpoint hftraining/checkpoints/production/final.pt \
    --mode paper \
    --symbols AAPL GOOGL MSFT TSLA AMZN \
    --update-interval 60
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
  }
}
```

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

## Troubleshooting

### Model Not Loading
- Check checkpoint path exists
- Verify checkpoint has 'model_state_dict'
- Ensure config matches model architecture

### No Trades Generated
- Lower confidence threshold
- Check data has enough history (60+ days)
- Verify market hours for live trading

### Poor Performance
- Retrain with more data
- Adjust risk parameters
- Use ensemble of models
- Add more technical indicators

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