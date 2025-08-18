# RL Trading Agent with PPO

This system implements a reinforcement learning-based trading agent using Proximal Policy Optimization (PPO) with an actor-critic architecture, inspired by the Toto model design.

## Components

### 1. **TradingAgent** (`trading_agent.py`)
- Actor-Critic neural network with separate heads for:
  - **Actor**: Outputs continuous trading actions (-1 to 1, representing short to long positions)
  - **Critic**: Estimates expected returns (value function)
- Can use pre-trained Toto backbone or custom architecture
- Gaussian policy for continuous action space

### 2. **DailyTradingEnv** (`trading_env.py`)
- OpenAI Gym-compatible trading environment
- Features:
  - Daily trading simulation with configurable window size
  - Transaction costs and position sizing
  - Comprehensive metrics tracking (Sharpe ratio, drawdown, win rate)
  - Normalized observations with position and P&L information

### 3. **PPOTrainer** (`ppo_trainer.py`)
- Implements PPO algorithm with:
  - Generalized Advantage Estimation (GAE)
  - Clipped surrogate objective
  - Value function loss
  - Entropy bonus for exploration
- Automatic checkpointing and evaluation

### 4. **Training Script** (`train_rl_agent.py`)
- Complete training pipeline with:
  - Data loading and preprocessing
  - Feature engineering (RSI, SMA, volume ratios)
  - Train/test splitting
  - Performance visualization
  - Results logging in JSON format

## Quick Start

### Test the System
```bash
cd training
python quick_test.py
```

### Train on Real Data
```bash
python train_rl_agent.py --symbol AAPL --num_episodes 500 --window_size 30
```

### Custom Training
```bash
python train_rl_agent.py \
    --symbol BTCUSD \
    --data_dir ../data \
    --num_episodes 1000 \
    --lr_actor 1e-4 \
    --lr_critic 5e-4 \
    --gamma 0.995 \
    --window_size 50 \
    --initial_balance 100000
```

## Key Features

### Reward Function
The agent receives rewards based on:
- Daily P&L from positions
- Transaction costs (penalized)
- Position changes (to prevent overtrading)

### Action Space
- Continuous: -1.0 to 1.0
  - -1.0 = Full short position
  - 0.0 = No position (cash)
  - 1.0 = Full long position

### Observation Space
Each observation includes:
- Historical OHLCV data
- Technical indicators (RSI, moving averages)
- Current position
- Portfolio balance ratio
- Unrealized P&L

## Training Process

1. **Data Preparation**: Load historical price data and compute technical indicators
2. **Environment Setup**: Create training and testing environments
3. **Model Initialization**: Build actor-critic network with appropriate architecture
4. **PPO Training Loop**:
   - Collect trajectories by running agent in environment
   - Compute advantages using GAE
   - Update policy using clipped PPO objective
   - Evaluate periodically on validation data
5. **Evaluation**: Test final model on held-out data

## Output Files

After training, the system generates:
- `models/best_model.pth`: Best performing model checkpoint
- `models/checkpoint_epN.pth`: Periodic checkpoints
- `models/test_results.png`: Visualization of test performance
- `models/results.json`: Complete metrics and hyperparameters

## Hyperparameters

Key hyperparameters to tune:
- `window_size`: Historical context (default: 30)
- `lr_actor/lr_critic`: Learning rates (default: 3e-4, 1e-3)
- `gamma`: Discount factor (default: 0.99)
- `eps_clip`: PPO clipping parameter (default: 0.2)
- `k_epochs`: PPO update epochs (default: 4)
- `entropy_coef`: Exploration bonus (default: 0.01)

## Performance Metrics

The system tracks:
- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Number of Trades**: Trading frequency

## Integration with Toto

To use pre-trained Toto model:
```python
agent = TradingAgent(use_pretrained_toto=True)
```

This loads Datadog's Toto transformer backbone and adds trading-specific heads.

## Requirements

- PyTorch
- NumPy
- Pandas
- Gym (or Gymnasium)
- Matplotlib
- Scikit-learn