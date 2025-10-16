# Toto RL Retraining System

Multi-asset reinforcement learning system that leverages pretrained transformer embeddings for stock market trading across multiple pairs.

## Architecture

### 1. Toto Embeddings (`../totoembedding/`)
- **Purpose**: Reuses pretrained transformer weights for market understanding
- **Key Features**:
  - Symbol-specific embeddings for different stocks/crypto
  - Market regime awareness (bull/bear/volatile/sideways)
  - Cross-asset correlation modeling
  - Time-based contextual features

### 2. Multi-Asset Environment (`multi_asset_env.py`)
- **Assets**: All 21 symbols from your trainingdata (AAPL, BTCUSD, etc.)
- **Action Space**: Continuous position weights [-1, 1] for each asset
- **Observation Space**: 
  - Toto embeddings (128 dim)
  - Portfolio state (positions, P&L, balance)
  - Market features (technical indicators, correlations)
  - Global context (time, volatility, etc.)

### 3. RL Agent (`rl_trainer.py`)
- **Architecture**: Dueling DQN with continuous actions
- **Features**:
  - Separate processing for embedding vs. other features
  - Risk-adjusted reward function
  - Experience replay with prioritization
  - Target network soft updates

## Usage

### Quick Start
```bash
# Basic training with defaults
python train_toto_rl.py

# Custom configuration
python train_toto_rl.py --episodes 3000 --balance 50000 --train-embeddings

# Specific symbols only
python train_toto_rl.py --symbols AAPL TSLA BTCUSD ETHUSD --episodes 1000
```

### Configuration
The system uses a comprehensive configuration system covering:

```json
{
  "data": {
    "train_dir": "../trainingdata/train",
    "symbols": ["AAPL", "BTCUSD", ...]
  },
  "embedding": {
    "pretrained_model": "../training/models/modern_best_sharpe.pth",
    "freeze_backbone": true
  },
  "environment": {
    "initial_balance": 100000,
    "max_positions": 10,
    "transaction_cost": 0.001
  },
  "training": {
    "episodes": 2000,
    "learning_rate": 1e-4,
    "batch_size": 128
  }
}
```

## Key Features

### Pretrained Weight Reuse
- Automatically loads best available model from `../training/models/`
- Freezes transformer backbone, trains only new layers
- Preserves learned market patterns while adapting to multi-asset trading

### Multi-Asset Trading
- Simultaneous trading across stocks and crypto
- Dynamic correlation tracking
- Position sizing based on volatility and correlation
- Diversification incentives in reward function

### Risk Management
- Transaction cost modeling (commission, spread, slippage)
- Maximum position limits
- Drawdown-based circuit breakers
- Risk-adjusted Sharpe ratio optimization

### Real Market Modeling
- Time-varying volatility and correlations
- Market regime detection
- Realistic execution costs
- Portfolio rebalancing constraints

## Output Structure

```
totoembedding-rlretraining/
├── models/
│   ├── toto_rl_best.pth          # Best performing model
│   ├── toto_rl_final.pth         # Final trained model
│   └── toto_embeddings.pth       # Trained embeddings
├── results/
│   ├── training_results.json     # Training metrics
│   ├── evaluation_results.json   # Test performance
│   └── config.json              # Used configuration
├── plots/
│   └── training_results.png      # Performance visualizations
└── runs/                         # TensorBoard logs
```

## Performance Monitoring

The system tracks comprehensive metrics:
- **Returns**: Total return, Sharpe ratio, max drawdown
- **Trading**: Number of trades, fees, win rate
- **Risk**: Volatility, correlation exposure, position concentration
- **Real-time**: TensorBoard integration for live monitoring

## Integration with Existing System

### Pretrained Models
- Automatically detects and loads best model from `../training/models/`
- Supports both modern transformer and legacy architectures
- Graceful fallback if pretrained loading fails

### Data Pipeline
- Uses existing trainingdata structure
- Supports both train/test splits
- Compatible with your existing data preprocessing

### Model Export
- Trained models compatible with `../rlinference/` system
- Embeddings can be exported for other use cases
- Standard PyTorch format for easy integration

## Advanced Features

### Ensemble Learning
- Multiple agent training with different seeds
- Model averaging for robust predictions
- Uncertainty quantification

### Online Learning
- Continuous adaptation to new market data
- Experience replay with recent data prioritization
- Model drift detection and retraining triggers

### Portfolio Optimization
- Mean-variance optimization integration
- Risk parity constraint options
- ESG and sector exposure limits

## Next Steps

1. **Run Initial Training**: Start with default configuration
2. **Hyperparameter Tuning**: Adjust learning rate, network size, reward function
3. **Symbol Selection**: Focus on best-performing asset combinations
4. **Risk Management**: Calibrate position limits and stop-losses
5. **Live Integration**: Connect to `../rlinference/` for paper trading

The system is designed to be production-ready while maintaining flexibility for research and experimentation.