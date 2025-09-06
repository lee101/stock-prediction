#!/usr/bin/env python3
"""
Profitable Trading System Trainer
Integrates differentiable training with realistic simulation
Trains until consistent profitability is achieved
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import deque
import sys
sys.path.append('/media/lee/crucial2/code/stock/training')

from differentiable_trainer import (
    DifferentiableTradingModel, 
    DifferentiableTrainer,
    TrainingConfig,
    GradientMonitor
)
from realistic_trading_env import (
    RealisticTradingEnvironment,
    TradingConfig,
    ProfitBasedTrainingReward,
    create_market_data_generator
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProfitableTrainingDataset(Dataset):
    """Dataset that includes profit signals"""
    
    def __init__(self, market_data: pd.DataFrame, seq_len: int = 20, 
                 lookahead: int = 5):
        self.data = market_data
        self.seq_len = seq_len
        self.lookahead = lookahead
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare features and labels with profit targets"""
        
        # Calculate technical indicators
        self.data['sma_5'] = self.data['close'].rolling(5).mean()
        self.data['sma_20'] = self.data['close'].rolling(20).mean()
        self.data['rsi'] = self.calculate_rsi(self.data['close'])
        self.data['volatility'] = self.data['returns'].rolling(20).std()
        self.data['volume_ratio'] = self.data['volume'] / self.data['volume'].rolling(20).mean()
        
        # Calculate profit targets
        self.data['future_return'] = self.data['close'].shift(-self.lookahead) / self.data['close'] - 1
        
        # Define profitable trades
        self.data['profitable_long'] = (self.data['future_return'] > 0.01).astype(int)
        self.data['profitable_short'] = (self.data['future_return'] < -0.01).astype(int)
        
        # Drop NaN values
        self.data = self.data.dropna()
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.lookahead
    
    def __getitem__(self, idx):
        # Get sequence
        seq_data = self.data.iloc[idx:idx + self.seq_len]
        
        # Normalize features
        features = ['close', 'volume', 'sma_5', 'sma_20', 'rsi', 'volatility']
        X = seq_data[features].values
        
        # Normalize
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # Get targets
        target_idx = idx + self.seq_len
        future_return = self.data.iloc[target_idx]['future_return']
        
        # Create action label based on profitability
        if self.data.iloc[target_idx]['profitable_long']:
            action = 0  # Buy
        elif self.data.iloc[target_idx]['profitable_short']:
            action = 2  # Sell  
        else:
            action = 1  # Hold
        
        # Position size based on expected return magnitude
        position_size = np.tanh(future_return * 10)
        
        # Confidence based on trend strength
        trend_strength = abs(seq_data['sma_5'].iloc[-1] - seq_data['sma_20'].iloc[-1]) / seq_data['close'].iloc[-1]
        confidence = min(1.0, trend_strength * 100)
        
        return {
            'inputs': torch.FloatTensor(X),
            'actions': torch.LongTensor([action]).squeeze(),
            'position_sizes': torch.FloatTensor([position_size]).squeeze(),
            'returns': torch.FloatTensor([future_return]).squeeze(),
            'confidence': torch.FloatTensor([confidence]).squeeze()
        }


class ProfitFocusedLoss(nn.Module):
    """Loss function that prioritizes profitable trades"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor],
                env_reward: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        losses = {}
        
        # Standard classification loss
        action_loss = F.cross_entropy(predictions['actions'], targets['actions'])
        losses['action_loss'] = action_loss
        
        # Position sizing loss (weighted by profitability)
        position_loss = F.smooth_l1_loss(
            predictions['position_sizes'], 
            targets['position_sizes']
        )
        
        # Weight position loss by expected returns
        profit_weight = torch.sigmoid(targets['returns'] * 100)
        weighted_position_loss = position_loss * profit_weight.mean()
        losses['position_loss'] = weighted_position_loss
        
        # Confidence calibration
        confidence_loss = F.mse_loss(
            predictions['confidences'],
            torch.sigmoid(torch.abs(targets['returns']) * 50)
        )
        losses['confidence_loss'] = confidence_loss
        
        # Profit-focused component
        predicted_probs = F.softmax(predictions['actions'], dim=-1)
        
        # Penalize wrong decisions on profitable trades
        profitable_mask = torch.abs(targets['returns']) > 0.01
        if profitable_mask.any():
            profit_penalty = F.cross_entropy(
                predictions['actions'][profitable_mask],
                targets['actions'][profitable_mask]
            ) * 2.0  # Double weight for profitable trades
            losses['profit_penalty'] = profit_penalty
        
        # Include environment reward if available
        if env_reward is not None:
            # Convert reward to loss (negative reward)
            env_loss = -env_reward
            losses['env_loss'] = env_loss
        
        # Combine losses
        total_loss = (
            losses['action_loss'] * 0.3 +
            losses.get('position_loss', 0) * 0.2 +
            losses.get('confidence_loss', 0) * 0.1 +
            losses.get('profit_penalty', 0) * 0.2 +
            losses.get('env_loss', 0) * 0.2
        )
        
        return total_loss, losses


class ProfitableSystemTrainer:
    """Trainer that focuses on achieving profitability"""
    
    def __init__(self, model: nn.Module, training_config: TrainingConfig, 
                 trading_config: TradingConfig):
        self.model = model
        self.training_config = training_config
        self.trading_config = trading_config
        
        # Create environments
        self.train_env = RealisticTradingEnvironment(trading_config)
        self.val_env = RealisticTradingEnvironment(trading_config)
        
        # Reward calculator
        self.reward_calc = ProfitBasedTrainingReward()
        
        # Loss function
        self.criterion = ProfitFocusedLoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
        
        # Profitability tracking
        self.profitability_history = []
        self.best_sharpe = -float('inf')
        self.best_return = -float('inf')
        self.patience_counter = 0
        self.max_patience = 10
        
        logger.info("Initialized ProfitableSystemTrainer")
    
    def train_until_profitable(self, train_loader: DataLoader, 
                              val_loader: DataLoader,
                              market_data: pd.DataFrame,
                              target_sharpe: float = 1.0,
                              target_return: float = 0.10,
                              max_epochs: int = 100) -> Dict[str, Any]:
        """Train until profitability targets are met"""
        
        logger.info(f"Training until Sharpe>{target_sharpe} and Return>{target_return:.1%}")
        
        for epoch in range(max_epochs):
            # Training phase
            train_metrics = self.train_epoch(train_loader, market_data[:len(train_loader)*20])
            
            # Validation with trading simulation
            val_performance = self.validate_with_trading(val_loader, market_data[len(train_loader)*20:])
            
            # Check profitability
            current_sharpe = val_performance['sharpe_ratio']
            current_return = val_performance['total_return']
            
            # Update best performance
            if current_sharpe > self.best_sharpe:
                self.best_sharpe = current_sharpe
                self.save_checkpoint(f'best_sharpe_model.pt')
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if current_return > self.best_return:
                self.best_return = current_return
            
            # Log progress
            logger.info(f"Epoch {epoch}: Sharpe={current_sharpe:.3f}, "
                       f"Return={current_return:.2%}, "
                       f"WinRate={val_performance['win_rate']:.1%}, "
                       f"PF={val_performance['profit_factor']:.2f}")
            
            # Store history
            self.profitability_history.append({
                'epoch': epoch,
                'sharpe': current_sharpe,
                'return': current_return,
                'win_rate': val_performance['win_rate'],
                'profit_factor': val_performance['profit_factor'],
                'max_drawdown': val_performance['max_drawdown']
            })
            
            # Check if targets met
            if current_sharpe >= target_sharpe and current_return >= target_return:
                logger.info(f"🎯 PROFITABILITY TARGETS ACHIEVED at epoch {epoch}!")
                logger.info(f"   Sharpe: {current_sharpe:.3f} >= {target_sharpe}")
                logger.info(f"   Return: {current_return:.2%} >= {target_return:.1%}")
                self.save_checkpoint('profitable_model_final.pt')
                break
            
            # Early stopping
            if self.patience_counter >= self.max_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Adjust learning rate if stuck
            if epoch > 0 and epoch % 20 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.5
                logger.info(f"Reduced learning rate to {param_group['lr']:.6f}")
        
        return self.profitability_history
    
    def train_epoch(self, dataloader: DataLoader, market_data: pd.DataFrame) -> Dict[str, float]:
        """Train for one epoch with profit focus"""
        
        self.model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            # Forward pass
            predictions = self.model(batch['inputs'])
            
            # Simulate trading for this batch (simplified)
            env_reward = self.simulate_batch_trading(predictions, batch, market_data)
            
            # Calculate loss
            loss, loss_components = self.criterion(predictions, batch, env_reward)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
        
        return {'train_loss': np.mean(epoch_losses)}
    
    def simulate_batch_trading(self, predictions: Dict[str, torch.Tensor], 
                              batch: Dict[str, torch.Tensor],
                              market_data: pd.DataFrame) -> torch.Tensor:
        """Simulate trading for a batch and return rewards"""
        
        batch_size = predictions['actions'].size(0)
        rewards = []
        
        with torch.no_grad():
            actions = F.softmax(predictions['actions'], dim=-1)
            
            for i in range(min(batch_size, 10)):  # Sample subset for efficiency
                # Convert to trading signal
                action_probs = actions[i]
                if action_probs[0] > 0.6:  # Buy
                    signal = predictions['position_sizes'][i]
                elif action_probs[2] > 0.6:  # Sell
                    signal = -predictions['position_sizes'][i]
                else:  # Hold
                    signal = torch.tensor(0.0)
                
                # Calculate simple reward based on actual returns
                actual_return = batch['returns'][i]
                trade_reward = signal * actual_return * 100  # Scale up
                
                # Ensure tensor and squeeze to scalar
                if not isinstance(trade_reward, torch.Tensor):
                    trade_reward = torch.tensor(trade_reward, dtype=torch.float32)
                
                # Ensure scalar tensor
                if trade_reward.dim() > 0:
                    trade_reward = trade_reward.squeeze()
                if trade_reward.dim() == 0:
                    rewards.append(trade_reward)
                else:
                    rewards.append(trade_reward.mean())
        
        return torch.stack(rewards).mean() if rewards else torch.tensor(0.0)
    
    def validate_with_trading(self, dataloader: DataLoader, 
                             market_data: pd.DataFrame) -> Dict[str, float]:
        """Validate model with full trading simulation"""
        
        self.model.eval()
        self.val_env.reset()
        
        data_idx = 0
        
        with torch.no_grad():
            for batch in dataloader:
                predictions = self.model(batch['inputs'])
                
                # Get batch size
                batch_size = predictions['actions'].size(0)
                
                for i in range(batch_size):
                    if data_idx >= len(market_data) - 1:
                        break
                    
                    # Get market state
                    market_state = {
                        'price': market_data.iloc[data_idx]['close'],
                        'timestamp': data_idx
                    }
                    
                    # Convert model output to trading action
                    action_probs = F.softmax(predictions['actions'][i], dim=-1)
                    
                    if action_probs[0] > 0.5:  # Buy signal
                        signal = predictions['position_sizes'][i].item()
                    elif action_probs[2] > 0.5:  # Sell signal
                        signal = -abs(predictions['position_sizes'][i].item())
                    else:
                        signal = 0.0
                    
                    action = {
                        'signal': torch.tensor(signal),
                        'confidence': predictions['confidences'][i]
                    }
                    
                    # Execute in environment
                    self.val_env.step(action, market_state)
                    data_idx += 1
        
        # Get final performance
        performance = self.val_env.get_performance_summary()
        
        return performance
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'profitability_history': self.profitability_history,
            'best_sharpe': self.best_sharpe,
            'best_return': self.best_return
        }
        
        path = Path('training') / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def plot_training_progress(self):
        """Plot training progress towards profitability"""
        
        if not self.profitability_history:
            return
        
        history = pd.DataFrame(self.profitability_history)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Sharpe ratio progress
        axes[0, 0].plot(history['sharpe'], 'b-', linewidth=2)
        axes[0, 0].axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Target')
        axes[0, 0].set_title('Sharpe Ratio Progress')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Return progress
        axes[0, 1].plot(history['return'] * 100, 'g-', linewidth=2)
        axes[0, 1].axhline(y=10, color='g', linestyle='--', alpha=0.5, label='Target 10%')
        axes[0, 1].set_title('Return Progress')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Return %')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Win rate
        axes[0, 2].plot(history['win_rate'] * 100, 'orange', linewidth=2)
        axes[0, 2].axhline(y=50, color='r', linestyle='--', alpha=0.5)
        axes[0, 2].set_title('Win Rate')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Win Rate %')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Profit factor
        axes[1, 0].plot(history['profit_factor'], 'purple', linewidth=2)
        axes[1, 0].axhline(y=1.5, color='g', linestyle='--', alpha=0.5, label='Good PF')
        axes[1, 0].set_title('Profit Factor')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Profit Factor')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Max drawdown
        axes[1, 1].plot(history['max_drawdown'] * 100, 'r-', linewidth=2)
        axes[1, 1].axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Target <10%')
        axes[1, 1].set_title('Maximum Drawdown')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Drawdown %')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Combined score
        combined_score = (
            history['sharpe'] / 1.5 * 0.4 +
            history['return'] / 0.2 * 0.3 +
            history['win_rate'] * 0.2 +
            (2 - history['max_drawdown'] / 0.1) * 0.1
        )
        axes[1, 2].plot(combined_score, 'black', linewidth=2)
        axes[1, 2].axhline(y=1.0, color='g', linestyle='--', alpha=0.5)
        axes[1, 2].set_title('Combined Profitability Score')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Training Progress Towards Profitability', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('training/profitability_progress.png', dpi=150)
        plt.close()
        
        logger.info("Saved profitability progress plot")


def main():
    """Main training loop for profitable system"""
    
    logger.info("="*60)
    logger.info("PROFITABLE TRADING SYSTEM TRAINER")
    logger.info("="*60)
    
    # Configuration
    training_config = TrainingConfig(
        learning_rate=5e-4,
        batch_size=32,
        num_epochs=100,
        gradient_clip_norm=1.0,
        mixed_precision=False,  # CPU mode
        weight_decay=1e-4
    )
    
    trading_config = TradingConfig(
        initial_capital=100000,
        max_position_size=0.1,
        commission_rate=0.001,
        slippage_factor=0.0005,
        stop_loss_pct=0.02,
        take_profit_pct=0.05
    )
    
    # Create model
    model = DifferentiableTradingModel(
        input_dim=6,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        dropout=0.1
    )
    
    # Generate market data
    logger.info("Generating market data...")
    market_data = create_market_data_generator(n_samples=10000, volatility=0.02)
    
    # Create datasets
    train_size = int(0.7 * len(market_data))
    val_size = int(0.15 * len(market_data))
    
    train_data = market_data[:train_size]
    val_data = market_data[train_size:train_size+val_size]
    test_data = market_data[train_size+val_size:]
    
    train_dataset = ProfitableTrainingDataset(train_data, seq_len=20)
    val_dataset = ProfitableTrainingDataset(val_data, seq_len=20)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create trainer
    trainer = ProfitableSystemTrainer(model, training_config, trading_config)
    
    # Train until profitable
    logger.info("Starting training until profitable...")
    history = trainer.train_until_profitable(
        train_loader,
        val_loader,
        market_data,
        target_sharpe=1.0,
        target_return=0.10,
        max_epochs=50
    )
    
    # Plot progress
    trainer.plot_training_progress()
    
    # Final validation on test data
    logger.info("\n" + "="*60)
    logger.info("FINAL TEST VALIDATION")
    logger.info("="*60)
    
    test_dataset = ProfitableTrainingDataset(test_data, seq_len=20)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    test_performance = trainer.validate_with_trading(test_loader, test_data)
    
    logger.info("Test Set Performance:")
    for key, value in test_performance.items():
        if isinstance(value, float):
            if 'return' in key or 'rate' in key or 'drawdown' in key:
                logger.info(f"  {key}: {value:.2%}")
            else:
                logger.info(f"  {key}: {value:.2f}")
    
    # Save final results
    results = {
        'training_history': history,
        'final_test_performance': test_performance,
        'model_config': {
            'hidden_dim': 128,
            'num_layers': 4,
            'num_heads': 4
        },
        'achieved_profitability': test_performance['sharpe_ratio'] > 1.0 and test_performance['total_return'] > 0.10
    }
    
    with open('training/profitable_training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("\n✅ Training complete! Results saved to training/profitable_training_results.json")
    
    return model, trainer, results


if __name__ == "__main__":
    model, trainer, results = main()