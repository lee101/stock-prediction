# RL-Based Trading System

A complete reinforcement learning-based trading system that uses trained PPO models to make trading decisions in real-time.

## Overview

This system replaces traditional forecasting methods with RL agents trained to maximize trading profitability. The agents output continuous actions between -1 and 1, where:
- Positive values = long position (buy)
- Negative values = short position (sell)  
- Magnitude = position size

## Features

- **Multi-stock support**: Train and run separate models for each stock
- **Ensemble predictions**: Use top-k profitable models for more robust predictions
- **Risk management**: Stop-loss, take-profit, position limits, circuit breakers
- **Paper & live trading**: Support for both paper and live trading via Alpaca
- **Real-time data**: Fetches live market data using yfinance
- **Portfolio tracking**: Comprehensive performance metrics and logging

## Directory Structure

```
rlinference/
├── configs/          # Configuration classes
├── utils/           # Data preprocessing, model management, risk, portfolio tracking
├── strategies/      # RL trading strategy implementation
├── brokers/        # Broker interfaces (Alpaca)
├── logs/           # Trading logs
├── run_trading.py  # Main trading engine entry point
└── train_stock_models.py  # Train models for multiple stocks
```

## Setup

### 1. Install Dependencies

```bash
pip install torch pandas numpy yfinance alpaca-py loguru
```

### 2. Set Environment Variables

For paper trading:
```bash
export ALP_KEY_ID_PAPER="your_paper_api_key"
export ALP_SECRET_KEY_PAPER="your_paper_secret_key"
```

For live trading:
```bash
export ALP_KEY_ID_PROD="your_live_api_key"
export ALP_SECRET_KEY_PROD="your_live_secret_key"
```

### 3. Train Models

Train models for individual stocks:

```bash
# Train for specific stocks
python rlinference/train_stock_models.py --symbols AAPL NVDA TSLA --num-episodes 500

# Train with custom parameters
python rlinference/train_stock_models.py \
  --symbols AAPL NVDA TSLA SPY QQQ \
  --num-episodes 1000 \
  --window-size 30 \
  --lr-actor 3e-4 \
  --parallel 4
```

## Usage

### Basic Usage (Paper Trading)

```bash
python rlinference/run_trading.py --symbols AAPL NVDA --paper
```

### Advanced Configuration

```bash
python rlinference/run_trading.py \
  --symbols AAPL NVDA TSLA SPY \
  --initial-balance 100000 \
  --max-positions 2 \
  --max-position-size 0.47 \
  --stop-loss 0.05 \
  --take-profit 0.20 \
  --use-ensemble \
  --interval 300 \
  --paper
```

### Dry Run Mode (No Trades)

```bash
python rlinference/run_trading.py --symbols AAPL --dry-run
```

### Live Trading (Use with Caution!)

```bash
python rlinference/run_trading.py --symbols AAPL --live
# Will prompt for confirmation
```

## Configuration Options

### Trading Parameters
- `--symbols`: List of symbols to trade
- `--initial-balance`: Starting account balance
- `--max-positions`: Maximum concurrent positions (default: 2)
- `--max-position-size`: Max position as fraction of equity (default: 0.47)

### Risk Management
- `--stop-loss`: Stop loss percentage (default: 5%)
- `--take-profit`: Take profit percentage (default: 20%)
- `--max-drawdown`: Maximum drawdown before stopping (default: 10%)
- `--circuit-breaker`: Daily loss limit (default: 15%)

### Model Options
- `--models-dir`: Directory containing trained models
- `--use-ensemble`: Use ensemble of top-k models for predictions

### Execution
- `--interval`: Trading interval in seconds (default: 300)
- `--paper/--live`: Trading mode
- `--dry-run`: Simulate without placing orders

## Model Training

The system uses PPO (Proximal Policy Optimization) to train trading agents. Key features:

- **State space**: Price history, technical indicators, position info, P&L
- **Action space**: Continuous [-1, 1] for position sizing and direction
- **Reward**: Based on trading returns with transaction costs
- **Top-k tracking**: Automatically saves the k most profitable models

### Training a Single Stock

```bash
cd training
python train_rl_agent.py \
  --symbol AAPL \
  --num_episodes 500 \
  --save_dir ../models/AAPL \
  --top_k 5
```

## Safety Features

1. **Circuit Breaker**: Stops trading if daily loss exceeds threshold
2. **Position Limits**: Maximum position size and count restrictions
3. **Risk Checks**: Validates all trades against risk parameters
4. **Paper Trading Default**: Defaults to paper trading to prevent accidents
5. **Confirmation Required**: Live trading requires explicit confirmation

## Performance Monitoring

The system tracks:
- Portfolio equity and returns
- Sharpe ratio and max drawdown
- Win rate and trade count
- Individual position P&L
- Daily and total returns

Logs are saved to `rlinference/logs/` with detailed trade information.

## API Integration

Currently supports Alpaca for order execution. The broker interface is modular, allowing easy addition of other brokers.

## Important Notes

- Always test thoroughly in paper trading before going live
- Start with small position sizes when transitioning to live trading
- Monitor the system closely, especially during initial deployment
- Review logs regularly for any anomalies
- Ensure models are properly trained with sufficient data

## Troubleshooting

### Common Issues

1. **No model found**: Ensure models are trained and saved in the correct directory
2. **API connection errors**: Check API keys and network connection
3. **Insufficient data**: Models need at least 30 days of history (window_size)
4. **Memory issues**: Reduce parallel training processes or batch size

### Logs

Check logs in `rlinference/logs/` for detailed debugging information.

## Disclaimer

This is an experimental trading system. Trading involves risk of loss. Always:
- Test thoroughly in paper trading
- Start with small amounts
- Never risk more than you can afford to lose
- Consider this as educational/research code