#!/usr/bin/env python3
"""
Advanced Neural Trading System with Self-Tuning Components
Multiple neural networks that learn to optimize each other and make trading decisions
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
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training/neural_trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TradingState:
    """Current state of the trading system"""
    timestamp: float
    price: float
    volume: float
    position: float  # Current position size
    cash: float
    portfolio_value: float
    recent_returns: List[float]
    volatility: float
    market_regime: str  # 'bull', 'bear', 'sideways'
    confidence: float
    

class HyperparameterTunerNetwork(nn.Module):
    """Neural network that learns to tune hyperparameters for other networks"""
    
    def __init__(self, input_dim=32, hidden_dim=128):
        super().__init__()
        
        # Input: performance metrics, current hyperparams, market conditions
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Attention mechanism to focus on important metrics
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, batch_first=True)
        
        # Output heads for different hyperparameters
        self.learning_rate_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1], will be scaled
        )
        
        self.batch_size_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.dropout_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.momentum_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        logger.info("HyperparameterTunerNetwork initialized")
    
    def forward(self, performance_metrics, current_hyperparams, market_features):
        # Combine all inputs
        x = torch.cat([performance_metrics, current_hyperparams, market_features], dim=-1)
        
        # Encode
        x = self.encoder(x)
        
        # Self-attention to identify important patterns
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1) if x.size(1) == 1 else x.mean(dim=1)
        
        # Generate hyperparameter suggestions
        lr = self.learning_rate_head(x) * 0.01  # Scale to [0, 0.01]
        batch_size = (self.batch_size_head(x) * 256 + 16).int()  # Scale to [16, 272]
        dropout = self.dropout_head(x) * 0.5  # Scale to [0, 0.5]
        momentum = self.momentum_head(x) * 0.99  # Scale to [0, 0.99]
        
        return {
            'learning_rate': lr,
            'batch_size': batch_size,
            'dropout': dropout,
            'momentum': momentum
        }


class PositionSizingNetwork(nn.Module):
    """Neural network that learns optimal position sizing"""
    
    def __init__(self, input_dim=64, hidden_dim=256):
        super().__init__()
        
        # Input: market features, risk metrics, portfolio state
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Risk assessment module
        self.risk_module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Risk score [0, 1]
        )
        
        # Position size predictor
        self.position_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),  # +1 for risk score
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Tanh()  # Position size [-1, 1] where negative is short
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Confidence [0, 1]
        )
        
        logger.info("PositionSizingNetwork initialized")
    
    def forward(self, market_features, portfolio_state, volatility):
        # Combine inputs
        x = torch.cat([market_features, portfolio_state, volatility.unsqueeze(-1)], dim=-1)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Assess risk
        risk_score = self.risk_module(features)
        
        # Predict position size based on features and risk
        position_input = torch.cat([features, risk_score], dim=-1)
        position_size = self.position_predictor(position_input)
        
        # Estimate confidence
        confidence = self.confidence_estimator(features)
        
        # Scale position by confidence
        adjusted_position = position_size * confidence
        
        return {
            'position_size': adjusted_position,
            'risk_score': risk_score,
            'confidence': confidence
        }


class TimingPredictionNetwork(nn.Module):
    """Neural network that learns optimal entry and exit timing"""
    
    def __init__(self, sequence_length=60, input_dim=10, hidden_dim=128):
        super().__init__()
        
        self.sequence_length = sequence_length
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=3, 
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # Transformer for long-range dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,  # Bidirectional LSTM
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Action predictor
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)  # Buy, Hold, Sell
        )
        
        # Timing urgency predictor (how soon to act)
        self.urgency_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Urgency [0, 1]
        )
        
        # Price target predictor
        self.target_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # Entry and exit targets
        )
        
        logger.info("TimingPredictionNetwork initialized")
    
    def forward(self, price_sequence, volume_sequence, indicators):
        # Combine inputs
        x = torch.cat([price_sequence, volume_sequence, indicators], dim=-1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Transformer processing
        trans_out = self.transformer(lstm_out)
        
        # Use last timestep for predictions
        final_features = trans_out[:, -1, :]
        
        # Predictions
        action = self.action_head(final_features)
        urgency = self.urgency_head(final_features)
        targets = self.target_head(final_features)
        
        return {
            'action': F.softmax(action, dim=-1),
            'urgency': urgency,
            'entry_target': targets[:, 0],
            'exit_target': targets[:, 1]
        }


class RiskManagementNetwork(nn.Module):
    """Neural network for dynamic risk management"""
    
    def __init__(self, input_dim=48, hidden_dim=128):
        super().__init__()
        
        # Encode market and portfolio state
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU()
        )
        
        # Stop loss predictor
        self.stop_loss_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Stop loss percentage [0, 1] -> [0%, 10%]
        )
        
        # Take profit predictor
        self.take_profit_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Take profit percentage [0, 1] -> [0%, 20%]
        )
        
        # Maximum position size limiter
        self.max_position_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Max position as fraction of portfolio
        )
        
        # Risk budget allocator
        self.risk_budget_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Daily risk budget [0, 1] -> [0%, 5%]
        )
        
        logger.info("RiskManagementNetwork initialized")
    
    def forward(self, portfolio_metrics, market_volatility, recent_performance):
        # Combine inputs
        x = torch.cat([portfolio_metrics, market_volatility, recent_performance], dim=-1)
        
        # Encode
        features = self.encoder(x)
        
        # Generate risk parameters
        stop_loss = self.stop_loss_head(features) * 0.1  # Scale to [0, 10%]
        take_profit = self.take_profit_head(features) * 0.2  # Scale to [0, 20%]
        max_position = self.max_position_head(features)  # [0, 1]
        risk_budget = self.risk_budget_head(features) * 0.05  # Scale to [0, 5%]
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'max_position': max_position,
            'risk_budget': risk_budget
        }


class MetaLearner(nn.Module):
    """Meta-learning network that coordinates all components"""
    
    def __init__(self, num_components=4, hidden_dim=256):
        super().__init__()
        
        self.num_components = num_components
        
        # Performance encoder for each component
        self.performance_encoder = nn.Sequential(
            nn.Linear(num_components * 10, hidden_dim),  # 10 metrics per component
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Interaction modeling between components
        self.interaction_layer = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=8, 
            batch_first=True
        )
        
        # Weight generator for ensemble
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_components),
            nn.Softmax(dim=-1)
        )
        
        # Learning rate scheduler for each component
        self.lr_scheduler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_components),
            nn.Sigmoid()
        )
        
        logger.info("MetaLearner initialized")
    
    def forward(self, component_performances):
        # Encode performances
        x = self.performance_encoder(component_performances)
        
        # Model interactions
        x = x.unsqueeze(1)
        x, _ = self.interaction_layer(x, x, x)
        x = x.squeeze(1)
        
        # Generate ensemble weights
        weights = self.weight_generator(x)
        
        # Generate learning rates
        learning_rates = self.lr_scheduler(x) * 0.01
        
        return {
            'ensemble_weights': weights,
            'component_lrs': learning_rates
        }


class NeuralTradingSystem:
    """Complete neural trading system with all components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cpu')  # Use CPU to avoid CUDA compatibility issues
        
        # Initialize all networks
        self.hyperparameter_tuner = HyperparameterTunerNetwork().to(self.device)
        self.position_sizer = PositionSizingNetwork().to(self.device)
        self.timing_predictor = TimingPredictionNetwork().to(self.device)
        self.risk_manager = RiskManagementNetwork().to(self.device)
        self.meta_learner = MetaLearner().to(self.device)
        
        # Optimizers for each network
        self.optimizers = {
            'hyperparameter': torch.optim.AdamW(self.hyperparameter_tuner.parameters(), lr=1e-3),
            'position': torch.optim.AdamW(self.position_sizer.parameters(), lr=1e-3),
            'timing': torch.optim.AdamW(self.timing_predictor.parameters(), lr=1e-3),
            'risk': torch.optim.AdamW(self.risk_manager.parameters(), lr=1e-3),
            'meta': torch.optim.AdamW(self.meta_learner.parameters(), lr=1e-4)
        }
        
        # Performance tracking
        self.performance_history = {
            'hyperparameter': deque(maxlen=100),
            'position': deque(maxlen=100),
            'timing': deque(maxlen=100),
            'risk': deque(maxlen=100),
            'overall': deque(maxlen=100)
        }
        
        # Trading state
        self.portfolio_value = 100000  # Starting capital
        self.positions = {}
        self.trade_history = []
        
        logger.info(f"NeuralTradingSystem initialized on {self.device}")
    
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic market data for training"""
        np.random.seed(42)
        
        # Generate price data with realistic patterns
        returns = np.random.normal(0.0002, 0.02, n_samples)
        
        # Add trends
        trend = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 0.001
        returns += trend
        
        # Add volatility clustering
        volatility = np.zeros(n_samples)
        volatility[0] = 0.01
        for i in range(1, n_samples):
            volatility[i] = 0.9 * volatility[i-1] + 0.1 * abs(returns[i-1])
        returns *= (1 + volatility)
        
        # Generate prices
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate volume
        volume = np.random.lognormal(15, 0.5, n_samples)
        
        # Technical indicators
        sma_20 = pd.Series(prices).rolling(20).mean().fillna(prices[0])
        sma_50 = pd.Series(prices).rolling(50).mean().fillna(prices[0])
        rsi = self.calculate_rsi(prices)
        
        return {
            'prices': torch.FloatTensor(prices),
            'returns': torch.FloatTensor(returns),
            'volume': torch.FloatTensor(volume),
            'volatility': torch.FloatTensor(volatility),
            'sma_20': torch.FloatTensor(sma_20.values),
            'sma_50': torch.FloatTensor(sma_50.values),
            'rsi': torch.FloatTensor(rsi)
        }
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi = np.zeros_like(prices)
        rsi[:period] = 50  # Neutral RSI for initial period
        rsi[period] = 100 - 100 / (1 + rs)
        
        for i in range(period + 1, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 100
            rsi[i] = 100 - 100 / (1 + rs)
        
        return rsi
    
    def train_component(self, component_name: str, data: Dict, epochs: int = 10):
        """Train a specific component of the system"""
        logger.info(f"Training {component_name} component...")
        
        component = getattr(self, {
            'hyperparameter': 'hyperparameter_tuner',
            'position': 'position_sizer',
            'timing': 'timing_predictor',
            'risk': 'risk_manager',
            'meta': 'meta_learner'
        }[component_name])
        
        optimizer = self.optimizers[component_name]
        losses = []
        
        for epoch in range(epochs):
            component.train()
            epoch_loss = 0
            
            # Prepare batch data based on component
            if component_name == 'timing':
                # Prepare sequences for timing prediction
                seq_len = 60
                for i in range(len(data['prices']) - seq_len - 1):
                    # Get sequence - combine all features into single tensor
                    features = torch.stack([
                        data['prices'][i:i+seq_len],
                        data['volume'][i:i+seq_len],
                        data['returns'][i:i+seq_len],
                        data['volatility'][i:i+seq_len],
                        data['sma_20'][i:i+seq_len],
                        data['sma_50'][i:i+seq_len],
                        data['rsi'][i:i+seq_len],
                        torch.ones(seq_len) * (i % 24),  # Hour of day
                        torch.ones(seq_len) * ((i // 24) % 7),  # Day of week  
                        torch.ones(seq_len) * (i / len(data['prices']))  # Position in dataset
                    ], dim=-1).unsqueeze(0)  # Shape: (1, seq_len, 10)
                    
                    # Forward pass - now using the combined features
                    output = component(features[:, :, :1], features[:, :, 1:2], features[:, :, 2:])
                    
                    # Calculate loss (simplified - in practice would use actual returns)
                    future_return = data['returns'][i+seq_len]
                    if future_return > 0.001:
                        target_action = torch.tensor([1.0, 0.0, 0.0])  # Buy
                    elif future_return < -0.001:
                        target_action = torch.tensor([0.0, 0.0, 1.0])  # Sell
                    else:
                        target_action = torch.tensor([0.0, 1.0, 0.0])  # Hold
                    
                    loss = F.cross_entropy(output['action'], target_action.unsqueeze(0).to(self.device))
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(component.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
            
            elif component_name == 'position':
                # Train position sizing network
                for i in range(0, len(data['prices']) - 100, 10):
                    # Prepare features
                    market_features = torch.cat([
                        data['prices'][i:i+10],
                        data['volume'][i:i+10],
                        data['rsi'][i:i+10]
                    ]).unsqueeze(0)
                    
                    portfolio_state = torch.tensor([
                        self.portfolio_value / 100000,  # Normalized portfolio value
                        len(self.positions),  # Number of positions
                        0.5  # Risk utilization
                    ]).unsqueeze(0)
                    
                    volatility = data['volatility'][i].unsqueeze(0)
                    
                    # Forward pass
                    output = component(market_features, portfolio_state, volatility)
                    
                    # Calculate reward-based loss
                    position_size = output['position_size'].squeeze()
                    future_return = data['returns'][i+1:i+11].mean()
                    reward = position_size * future_return - abs(position_size) * 0.001  # Transaction cost
                    loss = -reward  # Negative reward as loss
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(component.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
            
            # Log performance
            avg_loss = epoch_loss / max(1, (len(data['prices']) - 100) // 10)
            losses.append(avg_loss)
            self.performance_history[component_name].append(avg_loss)
            
            if epoch % 2 == 0:
                logger.info(f"  Epoch {epoch}/{epochs}: Loss = {avg_loss:.4f}")
        
        return losses
    
    def coordinated_training(self, data: Dict, cycles: int = 5):
        """Train all components in a coordinated manner"""
        logger.info("Starting coordinated training...")
        
        all_losses = {
            'hyperparameter': [],
            'position': [],
            'timing': [],
            'risk': [],
            'meta': []
        }
        
        for cycle in range(cycles):
            logger.info(f"\nTraining Cycle {cycle + 1}/{cycles}")
            
            # Get current performance metrics
            performance_metrics = self.get_performance_metrics()
            
            # Meta-learner decides training strategy
            if cycle > 0:
                self.meta_learner.eval()
                with torch.no_grad():
                    perf_tensor = torch.FloatTensor(performance_metrics).unsqueeze(0).to(self.device)
                    meta_output = self.meta_learner(perf_tensor)
                    
                    # Adjust learning rates based on meta-learner
                    for i, (name, optimizer) in enumerate(self.optimizers.items()):
                        if name != 'meta':
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = meta_output['component_lrs'][0, i].item()
                    
                    logger.info(f"Meta-learner adjusted learning rates: {meta_output['component_lrs'][0].cpu().numpy()}")
            
            # Train each component
            for component_name in ['timing', 'position', 'risk']:
                losses = self.train_component(component_name, data, epochs=5)
                all_losses[component_name].extend(losses)
            
            # Update hyperparameter tuner based on performance
            if cycle > 0:
                self.train_hyperparameter_tuner(performance_metrics)
            
            # Evaluate and log progress
            self.evaluate_system(data)
        
        return all_losses
    
    def train_hyperparameter_tuner(self, performance_metrics):
        """Train the hyperparameter tuner based on system performance"""
        self.hyperparameter_tuner.train()
        
        # Prepare input
        perf_tensor = torch.FloatTensor(performance_metrics[:10]).unsqueeze(0).to(self.device)
        current_hp = torch.FloatTensor([0.001, 32, 0.1, 0.9]).unsqueeze(0).to(self.device)  # Current hyperparams
        market_features = torch.randn(1, 18).to(self.device)  # Simplified market features
        
        # Forward pass
        suggested_hp = self.hyperparameter_tuner(perf_tensor, current_hp, market_features)
        
        # Calculate loss based on whether performance improved
        performance_improvement = performance_metrics[-1] - performance_metrics[-2] if len(performance_metrics) > 1 else 0
        loss = -performance_improvement  # Negative improvement as loss
        
        # Backward pass
        self.optimizers['hyperparameter'].zero_grad()
        loss = torch.tensor(loss, requires_grad=True)
        loss.backward()
        self.optimizers['hyperparameter'].step()
    
    def get_performance_metrics(self) -> List[float]:
        """Get current performance metrics for all components"""
        metrics = []
        
        for component_name in ['hyperparameter', 'position', 'timing', 'risk']:
            history = self.performance_history[component_name]
            if history:
                metrics.extend([
                    np.mean(list(history)),  # Average loss
                    np.std(list(history)),   # Loss variance
                    min(history),             # Best loss
                    max(history),             # Worst loss
                    history[-1] if history else 0,  # Latest loss
                    len(history),             # Number of updates
                    (history[0] - history[-1]) / max(history[0], 1e-6) if len(history) > 1 else 0,  # Improvement
                    0,  # Placeholder for additional metrics
                    0,
                    0
                ])
            else:
                metrics.extend([0] * 10)
        
        return metrics
    
    def evaluate_system(self, data: Dict):
        """Evaluate the complete trading system"""
        self.hyperparameter_tuner.eval()
        self.position_sizer.eval()
        self.timing_predictor.eval()
        self.risk_manager.eval()
        
        total_return = 0
        num_trades = 0
        winning_trades = 0
        
        with torch.no_grad():
            # Simulate trading
            seq_len = 60
            for i in range(seq_len, len(data['prices']) - 10, 5):
                # Get timing prediction
                price_seq = data['prices'][i-seq_len:i].unsqueeze(0).unsqueeze(-1)
                volume_seq = data['volume'][i-seq_len:i].unsqueeze(0).unsqueeze(-1)
                indicators = torch.stack([
                    data['sma_20'][i-seq_len:i],
                    data['sma_50'][i-seq_len:i],
                    data['rsi'][i-seq_len:i]
                ], dim=-1).unsqueeze(0)
                
                timing_output = self.timing_predictor(price_seq, volume_seq, indicators)
                
                # Get position sizing
                market_features = torch.cat([
                    data['prices'][i-10:i],
                    data['volume'][i-10:i],
                    data['rsi'][i-10:i]
                ]).unsqueeze(0)
                
                portfolio_state = torch.tensor([
                    self.portfolio_value / 100000,
                    len(self.positions),
                    0.5
                ]).unsqueeze(0)
                
                position_output = self.position_sizer(
                    market_features, 
                    portfolio_state,
                    data['volatility'][i].unsqueeze(0)
                )
                
                # Make trading decision
                action = timing_output['action'][0].argmax().item()
                if action == 0:  # Buy
                    position_size = position_output['position_size'][0].item()
                    entry_price = data['prices'][i].item()
                    exit_price = data['prices'][min(i+10, len(data['prices'])-1)].item()
                    trade_return = (exit_price - entry_price) / entry_price * position_size
                    total_return += trade_return
                    num_trades += 1
                    if trade_return > 0:
                        winning_trades += 1
        
        # Calculate metrics
        sharpe_ratio = (total_return / max(num_trades, 1)) / 0.02 if num_trades > 0 else 0
        win_rate = winning_trades / max(num_trades, 1)
        
        self.performance_history['overall'].append(total_return)
        
        logger.info(f"Evaluation - Total Return: {total_return:.4f}, "
                   f"Trades: {num_trades}, Win Rate: {win_rate:.2%}, "
                   f"Sharpe: {sharpe_ratio:.2f}")
    
    def save_models(self, path: Path):
        """Save all trained models"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.hyperparameter_tuner.state_dict(), path / 'hyperparameter_tuner.pth')
        torch.save(self.position_sizer.state_dict(), path / 'position_sizer.pth')
        torch.save(self.timing_predictor.state_dict(), path / 'timing_predictor.pth')
        torch.save(self.risk_manager.state_dict(), path / 'risk_manager.pth')
        torch.save(self.meta_learner.state_dict(), path / 'meta_learner.pth')
        
        # Save performance history
        with open(path / 'performance_history.json', 'w') as f:
            json.dump({k: list(v) for k, v in self.performance_history.items()}, f, indent=2)
        
        logger.info(f"Models saved to {path}")
    
    def visualize_learning(self):
        """Visualize the learning progress of all components"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        components = ['hyperparameter', 'position', 'timing', 'risk', 'overall']
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for idx, (component, color) in enumerate(zip(components, colors)):
            row = idx // 3
            col = idx % 3
            
            history = list(self.performance_history[component])
            if history:
                axes[row, col].plot(history, color=color, alpha=0.7)
                axes[row, col].set_title(f'{component.capitalize()} Performance')
                axes[row, col].set_xlabel('Training Step')
                axes[row, col].set_ylabel('Loss/Return')
                axes[row, col].grid(True, alpha=0.3)
                
                # Add trend line
                if len(history) > 10:
                    z = np.polyfit(range(len(history)), history, 1)
                    p = np.poly1d(z)
                    axes[row, col].plot(range(len(history)), p(range(len(history))), 
                                      "--", color=color, alpha=0.5, label=f'Trend: {z[0]:.4f}')
                    axes[row, col].legend()
        
        # Overall system metrics
        axes[1, 2].bar(['HP Tuner', 'Position', 'Timing', 'Risk'], 
                      [len(self.performance_history[c]) for c in ['hyperparameter', 'position', 'timing', 'risk']],
                      color=['blue', 'green', 'red', 'orange'], alpha=0.7)
        axes[1, 2].set_title('Component Update Counts')
        axes[1, 2].set_ylabel('Number of Updates')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Neural Trading System Learning Progress', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = Path('training/neural_system_learning.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        logger.info(f"Learning visualization saved to {save_path}")


def main():
    """Main training and evaluation pipeline"""
    
    # Configuration
    config = {
        'initial_capital': 100000,
        'max_positions': 5,
        'risk_per_trade': 0.02,
        'training_cycles': 5,
        'epochs_per_component': 5
    }
    
    # Initialize system
    logger.info("="*60)
    logger.info("NEURAL TRADING SYSTEM TRAINING")
    logger.info("="*60)
    
    system = NeuralTradingSystem(config)
    
    # Generate training data
    logger.info("\nGenerating synthetic training data...")
    data = system.generate_synthetic_data(n_samples=2000)
    
    # Coordinated training
    logger.info("\nStarting coordinated multi-network training...")
    losses = system.coordinated_training(data, cycles=config['training_cycles'])
    
    # Visualize learning
    system.visualize_learning()
    
    # Save trained models
    system.save_models(Path('training/neural_trading_models'))
    
    # Final evaluation
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE - FINAL EVALUATION")
    logger.info("="*60)
    
    # Performance summary
    for component in ['hyperparameter', 'position', 'timing', 'risk', 'overall']:
        history = list(system.performance_history[component])
        if history:
            improvement = (history[0] - history[-1]) / max(abs(history[0]), 1e-6) * 100
            logger.info(f"{component.capitalize():15s} - "
                       f"Initial: {history[0]:.4f}, "
                       f"Final: {history[-1]:.4f}, "
                       f"Improvement: {improvement:.2f}%")
    
    return system, losses


if __name__ == "__main__":
    system, losses = main()