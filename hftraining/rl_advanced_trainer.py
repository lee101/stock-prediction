#!/usr/bin/env python3
"""
Advanced RL Trading System with Learned Position Sizing and Risk Management
Uses deep RL to learn optimal trading strategies including:
- Position sizing based on confidence and market conditions
- Dynamic stop-loss and take-profit levels
- Risk management strategies
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import json
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.append(os.path.dirname(current_dir))

from data_utils import StockDataProcessor, split_data
from train_hf import StockDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class RLTradingConfig:
    """Configuration for RL trading system"""
    # Model architecture
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    dropout: float = 0.1
    
    # RL specific
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Soft update rate
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    sequence_length: int = 30
    prediction_horizon: int = 5
    
    # Trading
    initial_capital: float = 10000
    max_position_size: float = 0.3  # Maximum 30% of capital in one position
    min_position_size: float = 0.05  # Minimum 5% position
    commission: float = 0.001
    slippage: float = 0.0005


class RLTradingEnvironment:
    """Trading environment for RL agent"""
    
    def __init__(self, data, config: RLTradingConfig):
        self.data = data
        self.config = config
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = self.config.sequence_length
        self.capital = self.config.initial_capital
        self.position = 0
        self.entry_price = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0
        self.peak_capital = self.capital
        self.trade_history = []
        
        return self.get_state()
    
    def get_state(self):
        """Get current state observation"""
        # Get historical data
        start_idx = self.current_step - self.config.sequence_length
        end_idx = self.current_step
        
        historical_data = self.data[start_idx:end_idx]
        
        # Add position information
        position_info = np.array([
            self.position / self.config.max_position_size,  # Normalized position
            (self.capital - self.config.initial_capital) / self.config.initial_capital,  # P&L ratio
            self.total_trades / 100,  # Normalized trade count
            self.winning_trades / (self.total_trades + 1),  # Win rate
        ])
        
        return historical_data, position_info
    
    def step(self, action: Dict[str, torch.Tensor]):
        """Execute action and return next state, reward, done"""
        # Extract actions
        trade_action = action['trade']  # 0: hold, 1: buy, 2: sell
        position_size = action['position_size']  # 0-1 normalized
        stop_loss = action['stop_loss']  # Percentage below entry
        take_profit = action['take_profit']  # Percentage above entry
        
        current_price = self.data[self.current_step, 3]  # Close price
        
        # Execute trade
        reward = 0
        trade_executed = False
        
        if trade_action == 1 and self.position == 0 and self.capital > 0:  # Buy signal (no short positions)
            # Calculate actual position size (ensure we don't spend more than available capital)
            position_value = min(
                self.capital * position_size * self.config.max_position_size,
                self.capital * 0.95  # Keep 5% reserve
            )
            position_value = max(position_value, self.capital * self.config.min_position_size)
            
            # Execute buy only if we have enough capital
            if position_value > 0 and position_value <= self.capital:
                shares = position_value / (current_price * (1 + self.config.commission + self.config.slippage))
                self.position = shares
                self.entry_price = current_price * (1 + self.config.commission + self.config.slippage)
                actual_cost = shares * self.entry_price
                
                # Ensure we don't go negative on capital
                if actual_cost <= self.capital:
                    self.capital -= actual_cost
                    trade_executed = True
                    
                    self.trade_history.append({
                        'action': 'buy',
                        'price': self.entry_price,
                        'shares': shares,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    })
            
        elif trade_action == 2 and self.position > 0:  # Sell signal
            # Execute sell
            exit_price = current_price * (1 - self.config.commission - self.config.slippage)
            proceeds = self.position * exit_price
            cost_basis = self.position * self.entry_price
            trade_return = proceeds - cost_basis
            
            self.capital += proceeds
            
            # Track trade statistics
            self.total_trades += 1
            if trade_return > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            reward = trade_return / self.config.initial_capital  # Normalized reward
            self.position = 0
            self.entry_price = 0
            trade_executed = True
            
        # Check stop-loss and take-profit if position exists
        if self.position > 0 and self.entry_price > 0:
            current_return = (current_price - self.entry_price) / self.entry_price
            
            # Stop-loss triggered (ensure stop_loss is positive)
            if current_return <= -abs(stop_loss):
                # Calculate exit price with stop loss
                exit_price = current_price * (1 - self.config.commission - self.config.slippage)
                proceeds = self.position * exit_price
                trade_return = proceeds - (self.position * self.entry_price)
                
                self.capital += proceeds  # Add back the proceeds from selling
                
                self.total_trades += 1
                self.losing_trades += 1
                
                reward = trade_return / self.config.initial_capital
                self.position = 0
                self.entry_price = 0
                
            # Take-profit triggered
            elif current_return >= abs(take_profit):
                # Calculate exit price with take profit
                exit_price = current_price * (1 - self.config.commission - self.config.slippage)
                proceeds = self.position * exit_price
                trade_return = proceeds - (self.position * self.entry_price)
                
                self.capital += proceeds  # Add back the proceeds from selling
                
                self.total_trades += 1
                self.winning_trades += 1
                
                reward = trade_return / self.config.initial_capital
                self.position = 0
                self.entry_price = 0
        
        # Calculate drawdown
        self.peak_capital = max(self.peak_capital, self.capital)
        drawdown = (self.peak_capital - self.capital) / self.peak_capital
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Add holding penalty to encourage decisive action
        if not trade_executed and self.position == 0:
            reward -= 0.0001  # Small penalty for not trading
        
        # Add risk-adjusted reward component
        if trade_executed:
            sharpe_component = reward / (abs(reward) + 0.01)  # Pseudo-Sharpe
            reward = reward * 0.7 + sharpe_component * 0.3
        
        # Survival bonus - reward for maintaining capital
        capital_ratio = self.capital / self.config.initial_capital
        if capital_ratio > 0.8:
            reward += 0.0001  # Small reward for capital preservation
        
        # Check for bankruptcy
        if self.capital <= 100:  # Essentially bankrupt
            reward = -1.0  # Large negative reward
            done = True
            logger.warning(f"Bankrupt! Capital: {self.capital:.2f}, Position: {self.position:.4f}")
        else:
            # Move to next step
            self.current_step += 1
            done = self.current_step >= len(self.data) - 1
        
        next_state = self.get_state() if not done else None
        
        return next_state, reward, done
    
    def get_metrics(self):
        """Get trading performance metrics"""
        total_return = (self.capital - self.config.initial_capital) / self.config.initial_capital
        win_rate = self.winning_trades / (self.total_trades + 1)
        
        return {
            'total_return': total_return * 100,
            'total_trades': self.total_trades,
            'win_rate': win_rate * 100,
            'max_drawdown': self.max_drawdown * 100,
            'final_capital': self.capital
        }


class RLTradingModel(nn.Module):
    """Deep RL model for trading with multiple action heads"""
    
    def __init__(self, config: RLTradingConfig, input_dim: int):
        super().__init__()
        self.config = config
        
        # Shared encoder
        self.input_projection = nn.Linear(input_dim, config.hidden_size)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Position info encoder
        self.position_encoder = nn.Sequential(
            nn.Linear(4, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, config.hidden_size // 2)
        )
        
        # Combine features
        self.feature_combiner = nn.Sequential(
            nn.Linear(config.hidden_size + config.hidden_size // 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Action heads
        # Trade action: hold, buy, sell
        self.trade_action_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 3)
        )
        
        # Position sizing: continuous 0-1
        self.position_size_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Stop-loss level: 0-10% below entry
        self.stop_loss_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Take-profit level: 0-20% above entry
        self.take_profit_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Value head for critic
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
    def forward(self, historical_data, position_info):
        """Forward pass through the model"""
        # Encode historical data
        x = self.input_projection(historical_data)
        x = self.transformer(x)
        
        # Use last hidden state
        market_features = x[:, -1, :]
        
        # Encode position info
        position_features = self.position_encoder(position_info)
        
        # Combine features
        combined = torch.cat([market_features, position_features], dim=-1)
        features = self.feature_combiner(combined)
        
        # Generate actions
        trade_logits = self.trade_action_head(features)
        position_size = self.position_size_head(features)
        stop_loss = self.stop_loss_head(features) * 0.1  # Max 10% stop-loss
        take_profit = self.take_profit_head(features) * 0.2  # Max 20% take-profit
        value = self.value_head(features)
        
        return {
            'trade_logits': trade_logits,
            'position_size': position_size.squeeze(-1),
            'stop_loss': stop_loss.squeeze(-1),
            'take_profit': take_profit.squeeze(-1),
            'value': value.squeeze(-1)
        }


class RLTrainer:
    """Trainer for RL trading model using PPO-style training"""
    
    def __init__(self, model, env, config: RLTradingConfig, device='cuda'):
        self.model = model.to(device)
        self.env = env
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=100,
            T_mult=2
        )
        
        # Tracking
        self.epsilon = config.epsilon_start
        self.best_return = -float('inf')
        self.best_model_state = None
        
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy strategy"""
        historical_data, position_info = state
        
        # Convert to tensors
        hist_tensor = torch.FloatTensor(historical_data).unsqueeze(0).to(self.device)
        pos_tensor = torch.FloatTensor(position_info).unsqueeze(0).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(hist_tensor, pos_tensor)
        
        # Epsilon-greedy for exploration during training
        if training and np.random.random() < self.epsilon:
            trade_action = np.random.choice(3)
            position_size = np.random.random()
            stop_loss = np.random.random() * 0.1
            take_profit = np.random.random() * 0.2
        else:
            trade_action = torch.argmax(outputs['trade_logits']).item()
            position_size = outputs['position_size'].item()
            stop_loss = outputs['stop_loss'].item()
            take_profit = outputs['take_profit'].item()
        
        return {
            'trade': trade_action,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    def train_episode(self):
        """Train one episode"""
        state = self.env.reset()
        episode_reward = 0
        episode_losses = []
        
        while True:
            # Select action
            action = self.select_action(state, training=True)
            
            # Take step
            next_state, reward, done = self.env.step(action)
            episode_reward += reward
            
            # Get model outputs for loss calculation
            historical_data, position_info = state
            hist_tensor = torch.FloatTensor(historical_data).unsqueeze(0).to(self.device)
            pos_tensor = torch.FloatTensor(position_info).unsqueeze(0).to(self.device)
            
            outputs = self.model(hist_tensor, pos_tensor)
            
            # Calculate losses
            # Advantage = reward + gamma * V(s') - V(s)
            with torch.no_grad():
                if next_state is not None:
                    next_hist, next_pos = next_state
                    next_hist_tensor = torch.FloatTensor(next_hist).unsqueeze(0).to(self.device)
                    next_pos_tensor = torch.FloatTensor(next_pos).unsqueeze(0).to(self.device)
                    next_outputs = self.model(next_hist_tensor, next_pos_tensor)
                    next_value = next_outputs['value']
                else:
                    next_value = torch.zeros(1).to(self.device)
            
            # TD error
            td_target = reward + self.config.gamma * next_value * (1 - done)
            td_error = td_target - outputs['value']
            
            # Actor loss (policy gradient with advantage)
            trade_probs = F.softmax(outputs['trade_logits'], dim=-1)
            trade_log_prob = torch.log(trade_probs[0, action['trade']] + 1e-8)
            actor_loss = -trade_log_prob * td_error.detach()
            
            # Critic loss (value function)
            critic_loss = F.mse_loss(outputs['value'], td_target.detach())
            
            # Entropy bonus for exploration
            entropy = -(trade_probs * torch.log(trade_probs + 1e-8)).sum()
            
            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            episode_losses.append(total_loss.item())
            
            if done:
                break
            
            state = next_state
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_end, 
                          self.epsilon * self.config.epsilon_decay)
        
        # Update learning rate
        self.scheduler.step()
        
        return episode_reward, np.mean(episode_losses), self.env.get_metrics()
    
    def train(self, num_episodes=100, max_minutes=2):
        """Train the model"""
        start_time = time.time()
        
        logger.info(f"Starting RL training for {max_minutes} minutes...")
        
        for episode in range(num_episodes):
            # Check time limit
            if (time.time() - start_time) / 60 >= max_minutes:
                break
            
            # Train episode
            episode_reward, avg_loss, metrics = self.train_episode()
            
            # Save best model
            if metrics['total_return'] > self.best_return:
                self.best_return = metrics['total_return']
                self.best_model_state = self.model.state_dict().copy()
                logger.info(f"💰 New best model! Return: {metrics['total_return']:.2f}%")
            
            # Log progress
            if episode % 10 == 0:
                logger.info(
                    f"Episode {episode} | "
                    f"Reward: {episode_reward:.4f} | "
                    f"Return: {metrics['total_return']:.2f}% | "
                    f"Trades: {metrics['total_trades']} | "
                    f"Win Rate: {metrics['win_rate']:.1f}% | "
                    f"Max DD: {metrics['max_drawdown']:.2f}% | "
                    f"ε: {self.epsilon:.3f}"
                )
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"✅ Loaded best model with return: {self.best_return:.2f}%")
        
        elapsed = (time.time() - start_time) / 60
        logger.info(f"Training completed in {elapsed:.2f} minutes")
        
        return self.model


def main():
    """Run advanced RL training"""
    
    # Setup paths
    checkpoint_dir = Path("hftraining/checkpoints/rl_advanced")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = RLTradingConfig()
    
    # Load local data
    logger.info("Loading local stock CSVs...")
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    all_data = []
    data_dir = Path('trainingdata')
    for symbol in stocks[:1]:  # Start with one stock for faster iteration
        candidates = list(data_dir.glob(f"{symbol}.csv"))
        if not candidates:
            candidates = [p for p in data_dir.glob('*.csv') if symbol.lower() in p.stem.lower()]
        if not candidates:
            continue
        df = pd.read_csv(candidates[0])
        df.columns = df.columns.str.lower()
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
            except Exception:
                pass
        logger.info(f"Loaded {len(df)} records for {symbol}")
        all_data.append(df)
    
    # Process data
    combined_df = all_data[0]
    processor = StockDataProcessor()
    features = processor.prepare_features(combined_df)
    processor.fit_scalers(features)
    normalized_data = processor.transform(features)
    
    # Split data
    train_data, val_data, _ = split_data(normalized_data, 0.7, 0.15, 0.15)
    
    # Create environment
    env = RLTradingEnvironment(train_data, config)
    
    # Create model
    input_dim = normalized_data.shape[1]
    model = RLTradingModel(config, input_dim)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Check for existing model
    best_model_path = checkpoint_dir / "best_rl_model.pth"
    if best_model_path.exists():
        logger.info("Loading existing model...")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model with previous return: {checkpoint.get('best_return', 'N/A'):.2f}%")
    
    # Create trainer
    trainer = RLTrainer(model, env, config)
    
    # Train
    trained_model = trainer.train(num_episodes=1000, max_minutes=2)
    
    # Save model
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'best_return': trainer.best_return,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }, best_model_path)
    
    logger.info(f"💾 Model saved to {best_model_path}")
    
    # Final evaluation
    env.reset()
    total_reward = 0
    while True:
        state = env.get_state()
        action = trainer.select_action(state, training=False)
        _, reward, done = env.step(action)
        total_reward += reward
        if done:
            break
    
    final_metrics = env.get_metrics()
    
    # Save report
    report = {
        'best_return': float(trainer.best_return),
        'final_metrics': final_metrics,
        'training_time': datetime.now().isoformat(),
        'stocks_trained': stocks[:1],
        'model_params': sum(p.numel() for p in model.parameters())
    }
    
    report_path = checkpoint_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("\n" + "="*50)
    logger.info("ADVANCED RL TRAINING COMPLETE")
    logger.info(f"Best Return: {trainer.best_return:.2f}%")
    logger.info(f"Final Trades: {final_metrics['total_trades']}")
    logger.info(f"Win Rate: {final_metrics['win_rate']:.1f}%")
    logger.info(f"Max Drawdown: {final_metrics['max_drawdown']:.2f}%")
    logger.info("="*50)


if __name__ == "__main__":
    main()
