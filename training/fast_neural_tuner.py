#!/usr/bin/env python3
"""
Fast Neural Trading System - Optimized for quick training and learning analysis
Focus on hyperparameter tuning, position sizing, and learning effectiveness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleHyperparameterTuner(nn.Module):
    """Lightweight neural tuner for hyperparameters"""
    
    def __init__(self):
        super().__init__()
        
        # Input: [loss, accuracy, volatility, trend, improvement_rate]
        self.tuner = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)  # [lr_multiplier, batch_size_log, dropout, weight_decay]
        )
        
        logger.info("SimpleHyperparameterTuner initialized")
    
    def forward(self, performance_metrics):
        x = self.tuner(performance_metrics)
        
        # Convert to actual hyperparameter ranges
        lr_mult = torch.sigmoid(x[:, 0]) * 4 + 0.1  # 0.1x to 4.1x multiplier
        batch_size = (torch.sigmoid(x[:, 1]) * 6 + 3).int()  # 8 to 512 (2^3 to 2^9)
        dropout = torch.sigmoid(x[:, 2]) * 0.4 + 0.05  # 0.05 to 0.45
        weight_decay = torch.sigmoid(x[:, 3]) * 0.1  # 0 to 0.1
        
        return {
            'lr_multiplier': lr_mult,
            'batch_size_log': batch_size,
            'dropout': dropout,
            'weight_decay': weight_decay
        }


class SimplePositionSizer(nn.Module):
    """Fast position sizing network"""
    
    def __init__(self):
        super().__init__()
        
        # Input: [price_momentum, volatility, portfolio_heat, win_rate, sharpe]
        self.sizer = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # [position_size, confidence]
        )
        
        logger.info("SimplePositionSizer initialized")
    
    def forward(self, market_state):
        x = self.sizer(market_state)
        
        position_size = torch.tanh(x[:, 0])  # -1 to 1 (short to long)
        confidence = torch.sigmoid(x[:, 1])  # 0 to 1
        
        # Adjust position by confidence
        final_position = position_size * confidence
        
        return {
            'position_size': final_position,
            'confidence': confidence
        }


class SimpleTradingModel(nn.Module):
    """Basic transformer-based trading model for testing"""
    
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(hidden_dim, 3)  # Buy, Hold, Sell
        
        logger.info("SimpleTradingModel initialized")
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.classifier(x[:, -1, :])  # Use last timestep
        return F.softmax(x, dim=-1)


class FastTradingSystem:
    """Fast neural trading system for learning analysis"""
    
    def __init__(self):
        self.device = torch.device('cpu')
        
        # Initialize networks
        self.hyperparameter_tuner = SimpleHyperparameterTuner()
        self.position_sizer = SimplePositionSizer()
        self.trading_model = SimpleTradingModel()
        
        # Optimizers
        self.tuner_optimizer = torch.optim.Adam(self.hyperparameter_tuner.parameters(), lr=1e-3)
        self.sizer_optimizer = torch.optim.Adam(self.position_sizer.parameters(), lr=1e-3)
        
        # Performance tracking
        self.performance_history = {
            'tuner_loss': [],
            'sizer_reward': [],
            'trading_accuracy': [],
            'portfolio_return': [],
            'hyperparameters': [],
            'position_sizes': []
        }
        
        # Current hyperparameters
        self.current_hp = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'dropout': 0.1,
            'weight_decay': 0.01
        }
        
        logger.info("FastTradingSystem initialized")
    
    def generate_market_data(self, n_samples=500, seq_len=20):
        """Generate synthetic market data quickly"""
        
        # Generate price movements
        returns = np.random.normal(0.0005, 0.02, n_samples)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Technical indicators
        volume = np.random.lognormal(10, 0.5, n_samples)
        
        # Simple moving averages
        price_series = pd.Series(prices)
        sma_5 = price_series.rolling(5, min_periods=1).mean()
        sma_20 = price_series.rolling(20, min_periods=1).mean()
        
        # Momentum
        momentum = np.zeros(n_samples)
        for i in range(5, n_samples):
            momentum[i] = (prices[i] - prices[i-5]) / prices[i-5]
        
        # Volatility
        vol_window = 10
        volatility = np.zeros(n_samples)
        for i in range(vol_window, n_samples):
            volatility[i] = np.std(returns[i-vol_window:i])
        
        # Create sequences
        sequences = []
        labels = []
        
        for i in range(seq_len, n_samples - 1):
            # Features: [price, volume, sma_5, sma_20, momentum, volatility]
            seq_features = np.column_stack([
                prices[i-seq_len:i],
                volume[i-seq_len:i],
                sma_5[i-seq_len:i],
                sma_20[i-seq_len:i],
                momentum[i-seq_len:i],
                volatility[i-seq_len:i]
            ])
            
            sequences.append(seq_features)
            
            # Label: future return direction
            future_return = (prices[i+1] - prices[i]) / prices[i]
            if future_return > 0.005:
                labels.append(0)  # Buy
            elif future_return < -0.005:
                labels.append(2)  # Sell
            else:
                labels.append(1)  # Hold
        
        return {
            'sequences': torch.FloatTensor(sequences),
            'labels': torch.LongTensor(labels),
            'prices': prices,
            'returns': returns
        }
    
    def train_trading_model(self, data, epochs=10):
        """Train the basic trading model"""
        
        # Create optimizer with current hyperparameters
        optimizer = torch.optim.Adam(
            self.trading_model.parameters(),
            lr=self.current_hp['learning_rate'],
            weight_decay=self.current_hp['weight_decay']
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            # Simple batching
            batch_size = self.current_hp['batch_size']
            for i in range(0, len(data['sequences']) - batch_size, batch_size):
                batch_x = data['sequences'][i:i+batch_size]
                batch_y = data['labels'][i:i+batch_size]
                
                optimizer.zero_grad()
                
                outputs = self.trading_model(batch_x)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trading_model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                
                pred = outputs.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)
            
            avg_loss = epoch_loss / max(1, len(data['sequences']) // batch_size)
            accuracy = correct / total if total > 0 else 0
            
            losses.append(avg_loss)
            accuracies.append(accuracy)
        
        final_loss = losses[-1] if losses else 1.0
        final_accuracy = accuracies[-1] if accuracies else 0.33
        
        self.performance_history['trading_accuracy'].append(final_accuracy)
        
        return final_loss, final_accuracy
    
    def evaluate_position_sizing(self, data):
        """Evaluate position sizing network"""
        
        portfolio_value = 10000
        positions = []
        returns = []
        
        # Simulate trading
        for i in range(50, len(data['prices']) - 10):
            # Market state: [momentum, volatility, portfolio_heat, win_rate, sharpe]
            recent_returns = data['returns'][i-10:i]
            momentum = (data['prices'][i] - data['prices'][i-5]) / data['prices'][i-5]
            volatility = np.std(recent_returns)
            
            # Portfolio metrics (simplified)
            portfolio_heat = len([p for p in positions if p != 0]) / 5  # Max 5 positions
            win_rate = 0.5  # Simplified
            sharpe = 0.1  # Simplified
            
            market_state = torch.FloatTensor([[momentum, volatility, portfolio_heat, win_rate, sharpe]])
            
            # Get position size
            with torch.no_grad():
                position_output = self.position_sizer(market_state)
                position_size = position_output['position_size'].item()
            
            # Simulate trade
            positions.append(position_size)
            
            # Calculate return
            if i < len(data['prices']) - 1:
                price_change = (data['prices'][i+1] - data['prices'][i]) / data['prices'][i]
                trade_return = position_size * price_change - abs(position_size) * 0.001  # Transaction cost
                returns.append(trade_return)
                portfolio_value *= (1 + trade_return * 0.1)  # 10% of portfolio per trade
        
        avg_return = np.mean(returns) if returns else 0
        sharpe_ratio = avg_return / max(np.std(returns), 1e-6) if returns else 0
        
        self.performance_history['sizer_reward'].append(avg_return)
        self.performance_history['position_sizes'].extend(positions[:10])  # Store sample
        
        return avg_return, sharpe_ratio
    
    def tune_hyperparameters(self, trading_loss, trading_accuracy):
        """Use neural tuner to adjust hyperparameters"""
        
        # Current performance metrics
        recent_accuracy = self.performance_history['trading_accuracy'][-5:] if len(self.performance_history['trading_accuracy']) >= 5 else [0.33]
        
        # Calculate improvement rate
        if len(recent_accuracy) > 1:
            improvement = (recent_accuracy[-1] - recent_accuracy[0]) / max(recent_accuracy[0], 1e-6)
        else:
            improvement = 0
        
        # Market conditions (simplified)
        volatility = 0.02  # Assumed
        trend = 0.001  # Assumed
        
        # Performance metrics: [loss, accuracy, volatility, trend, improvement_rate]
        performance_input = torch.FloatTensor([[
            trading_loss,
            trading_accuracy,
            volatility,
            trend,
            improvement
        ]])
        
        # Get hyperparameter suggestions
        self.hyperparameter_tuner.train()
        hp_suggestions = self.hyperparameter_tuner(performance_input)
        
        # Calculate tuner loss (reward-based)
        reward = trading_accuracy - 0.33  # Above random baseline
        tuner_loss = torch.tensor(-reward, requires_grad=True)  # Negative reward as loss
        
        # Update tuner
        self.tuner_optimizer.zero_grad()
        tuner_loss.backward()
        self.tuner_optimizer.step()
        
        # Apply suggested hyperparameters
        self.current_hp['learning_rate'] *= hp_suggestions['lr_multiplier'].item()
        self.current_hp['learning_rate'] = max(1e-5, min(0.1, self.current_hp['learning_rate']))
        
        new_batch_size = int(2 ** hp_suggestions['batch_size_log'].item())
        self.current_hp['batch_size'] = max(8, min(128, new_batch_size))
        
        self.current_hp['dropout'] = hp_suggestions['dropout'].item()
        self.current_hp['weight_decay'] = hp_suggestions['weight_decay'].item()
        
        # Store results
        self.performance_history['tuner_loss'].append(tuner_loss.item())
        self.performance_history['hyperparameters'].append(self.current_hp.copy())
        
        logger.info(f"Hyperparameters updated: LR={self.current_hp['learning_rate']:.6f}, "
                   f"Batch={self.current_hp['batch_size']}, "
                   f"Dropout={self.current_hp['dropout']:.3f}")
        
        return tuner_loss.item()
    
    def run_learning_experiment(self, cycles=10, epochs_per_cycle=5):
        """Run complete learning experiment"""
        
        logger.info("="*60)
        logger.info("FAST NEURAL TRADING SYSTEM - LEARNING EXPERIMENT")
        logger.info("="*60)
        
        for cycle in range(cycles):
            logger.info(f"\nCycle {cycle+1}/{cycles}")
            
            # Generate fresh data
            data = self.generate_market_data()
            
            # Train trading model
            trading_loss, trading_accuracy = self.train_trading_model(data, epochs=epochs_per_cycle)
            
            # Evaluate position sizing
            avg_return, sharpe = self.evaluate_position_sizing(data)
            
            # Tune hyperparameters
            tuner_loss = self.tune_hyperparameters(trading_loss, trading_accuracy)
            
            # Calculate portfolio performance
            portfolio_return = avg_return * 10  # Simplified
            self.performance_history['portfolio_return'].append(portfolio_return)
            
            logger.info(f"  Trading: Loss={trading_loss:.4f}, Accuracy={trading_accuracy:.3f}")
            logger.info(f"  Position: Return={avg_return:.4f}, Sharpe={sharpe:.2f}")
            logger.info(f"  Tuner Loss: {tuner_loss:.4f}")
            logger.info(f"  Portfolio Return: {portfolio_return:.4f}")
        
        # Final analysis
        self.analyze_learning()
        
        return self.performance_history
    
    def analyze_learning(self):
        """Analyze learning effectiveness"""
        
        logger.info("\n" + "="*60)
        logger.info("LEARNING ANALYSIS")
        logger.info("="*60)
        
        # Trading model learning
        if len(self.performance_history['trading_accuracy']) > 1:
            initial_acc = self.performance_history['trading_accuracy'][0]
            final_acc = self.performance_history['trading_accuracy'][-1]
            acc_improvement = (final_acc - initial_acc) / max(initial_acc, 1e-6) * 100
            logger.info(f"Trading Accuracy: {initial_acc:.3f} → {final_acc:.3f} ({acc_improvement:+.1f}%)")
        
        # Position sizing learning
        if len(self.performance_history['sizer_reward']) > 1:
            initial_return = self.performance_history['sizer_reward'][0]
            final_return = self.performance_history['sizer_reward'][-1]
            return_improvement = (final_return - initial_return) / max(abs(initial_return), 1e-6) * 100
            logger.info(f"Position Sizing: {initial_return:.4f} → {final_return:.4f} ({return_improvement:+.1f}%)")
        
        # Hyperparameter tuning effectiveness
        if len(self.performance_history['tuner_loss']) > 1:
            initial_loss = self.performance_history['tuner_loss'][0]
            final_loss = self.performance_history['tuner_loss'][-1]
            tuner_improvement = (initial_loss - final_loss) / max(abs(initial_loss), 1e-6) * 100
            logger.info(f"Tuner Loss: {initial_loss:.4f} → {final_loss:.4f} ({tuner_improvement:+.1f}%)")
        
        # Overall portfolio performance
        if len(self.performance_history['portfolio_return']) > 1:
            total_return = sum(self.performance_history['portfolio_return'])
            logger.info(f"Total Portfolio Return: {total_return:.4f}")
        
        # Hyperparameter evolution
        if self.performance_history['hyperparameters']:
            initial_hp = self.performance_history['hyperparameters'][0]
            final_hp = self.performance_history['hyperparameters'][-1]
            
            logger.info("\nHyperparameter Evolution:")
            for key in initial_hp:
                initial = initial_hp[key]
                final = final_hp[key]
                change = (final - initial) / max(abs(initial), 1e-6) * 100
                logger.info(f"  {key}: {initial} → {final} ({change:+.1f}%)")
    
    def plot_learning_curves(self):
        """Plot learning progress"""
        
        if not any(self.performance_history.values()):
            logger.warning("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Trading accuracy
        if self.performance_history['trading_accuracy']:
            axes[0, 0].plot(self.performance_history['trading_accuracy'], 'b-o')
            axes[0, 0].set_title('Trading Accuracy Learning')
            axes[0, 0].set_xlabel('Cycle')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Position sizing rewards
        if self.performance_history['sizer_reward']:
            axes[0, 1].plot(self.performance_history['sizer_reward'], 'g-o')
            axes[0, 1].set_title('Position Sizing Returns')
            axes[0, 1].set_xlabel('Cycle')
            axes[0, 1].set_ylabel('Return')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Hyperparameter tuner loss
        if self.performance_history['tuner_loss']:
            axes[1, 0].plot(self.performance_history['tuner_loss'], 'r-o')
            axes[1, 0].set_title('Hyperparameter Tuner Loss')
            axes[1, 0].set_xlabel('Cycle')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Portfolio returns
        if self.performance_history['portfolio_return']:
            cumulative = np.cumsum(self.performance_history['portfolio_return'])
            axes[1, 1].plot(cumulative, 'purple', linewidth=2)
            axes[1, 1].set_title('Cumulative Portfolio Return')
            axes[1, 1].set_xlabel('Cycle')
            axes[1, 1].set_ylabel('Cumulative Return')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training/fast_learning_curves.png', dpi=150)
        plt.close()
        
        logger.info("Learning curves saved to training/fast_learning_curves.png")
    
    def save_results(self):
        """Save experimental results"""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'performance_history': self.performance_history,
            'final_hyperparameters': self.current_hp,
            'summary': {
                'total_cycles': len(self.performance_history['trading_accuracy']),
                'final_accuracy': self.performance_history['trading_accuracy'][-1] if self.performance_history['trading_accuracy'] else 0,
                'total_return': sum(self.performance_history['portfolio_return']),
                'best_position_return': max(self.performance_history['sizer_reward']) if self.performance_history['sizer_reward'] else 0,
            }
        }
        
        save_path = Path('training/fast_learning_results.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {save_path}")


def main():
    """Main experiment runner"""
    
    system = FastTradingSystem()
    
    # Run learning experiment
    results = system.run_learning_experiment(cycles=8, epochs_per_cycle=3)
    
    # Plot and save results
    system.plot_learning_curves()
    system.save_results()
    
    return system, results


if __name__ == "__main__":
    system, results = main()