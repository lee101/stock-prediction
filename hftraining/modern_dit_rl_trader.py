#!/usr/bin/env python3
"""
Modern DiT-based RL Trading System with Learnable Hyperparameters
Uses Diffusion Transformer blocks and learns all trading parameters through RL
"""

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
from typing import Dict, Tuple, Optional, List
import math

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from data_utils import StockDataProcessor, download_stock_data, split_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModernTradingConfig:
    """Configuration for modern RL trading"""
    # Model architecture
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    
    # DiT specific
    patch_size: int = 4  # Treat sequence as patches
    use_adaln: bool = True  # Adaptive Layer Norm (DiT feature)
    
    # RL parameters
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 32
    
    # Trading (all learnable, these are just max bounds)
    initial_capital: float = 25000
    max_possible_position: float = 1.0  # Can use up to 100% of capital (learned)
    min_trade_size: float = 100
    
    # Data
    sequence_length: int = 64
    feature_dim: int = 10


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block with adaptive normalization
    Based on "Scalable Diffusion Models with Transformers" architecture
    """
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Adaptive Layer Norm components
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # Self-attention with RoPE (Rotary Position Embedding)
        self.attn = nn.MultiheadAttention(
            dim, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # MLP with SwiGLU activation
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim * 2),
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Modulation parameters (for adaptive normalization)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        
        # Initialize modulation to identity
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with conditioning
        x: input features (batch, seq_len, dim)
        cond: conditioning vector (batch, dim)
        """
        # Get modulation parameters from conditioning
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(cond).chunk(6, dim=-1)
        
        # Expand for sequence length
        B, L, D = x.shape
        shift_msa = shift_msa.unsqueeze(1).expand(-1, L, -1)
        scale_msa = scale_msa.unsqueeze(1).expand(-1, L, -1)
        gate_msa = gate_msa.unsqueeze(1).expand(-1, L, -1)
        shift_mlp = shift_mlp.unsqueeze(1).expand(-1, L, -1)
        scale_mlp = scale_mlp.unsqueeze(1).expand(-1, L, -1)
        gate_mlp = gate_mlp.unsqueeze(1).expand(-1, L, -1)
        
        # Self-attention with adaptive norm
        norm_x = self.norm1(x)
        norm_x = norm_x * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + gate_msa * attn_out
        
        # MLP with adaptive norm
        norm_x = self.norm2(x)
        norm_x = norm_x * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(norm_x)
        x = x + gate_mlp * mlp_out
        
        return x


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class ModernDiTTrader(nn.Module):
    """
    Modern trading model using DiT blocks with learnable hyperparameters
    """
    
    def __init__(self, config: ModernTradingConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.feature_dim, config.hidden_dim)
        
        # Positional encoding with learnable parameters
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.sequence_length, config.hidden_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Market condition encoder (for adaptive normalization)
        self.market_encoder = nn.Sequential(
            nn.Linear(config.feature_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Stack of DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                config.hidden_dim,
                config.num_heads,
                config.mlp_ratio,
                config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        # Final norm
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        
        # Trading action heads with learnable constraints
        
        # 1. Trade decision (multinomial: hold/buy/sell with learned probabilities)
        self.trade_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 3)
        )
        
        # 2. Position sizing (learned distribution parameters)
        self.position_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 3)  # mean, std, max_position
        )
        
        # 3. Risk management (learned adaptive parameters)
        self.risk_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 4)  # stop_loss_mean, stop_loss_std, tp_mean, tp_std
        )
        
        # 4. Meta-parameters (learn trading style)
        self.meta_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 3)  # aggression, patience, risk_tolerance
        )
        
        # Value head for critic
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, market_state: torch.Tensor):
        """
        x: market data (batch, seq_len, features)
        market_state: current market conditions (batch, features*2)
        """
        B, L, _ = x.shape
        
        # Project input
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :L, :]
        
        # Get market conditioning
        cond = self.market_encoder(market_state)
        
        # Apply DiT blocks
        for block in self.blocks:
            x = block(x, cond)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Pool features (use both mean and max for richer representation)
        x_mean = x.mean(dim=1)
        x_max, _ = x.max(dim=1)
        x_pooled = torch.cat([x_mean, x_max], dim=-1)
        x_pooled = x_pooled[:, :self.config.hidden_dim]  # Ensure correct dimension
        
        # Generate outputs
        trade_logits = self.trade_head(x_pooled)
        
        # Position sizing with learned constraints
        position_params = self.position_head(x_pooled)
        pos_mean = torch.sigmoid(position_params[:, 0])  # 0-1
        pos_std = torch.sigmoid(position_params[:, 1]) * 0.2  # Max 20% std
        max_position = torch.sigmoid(position_params[:, 2])  # Learned max position
        
        # Risk parameters with learned adaptation
        risk_params = self.risk_head(x_pooled)
        stop_loss_mean = torch.sigmoid(risk_params[:, 0]) * 0.1  # Max 10% mean
        stop_loss_std = torch.sigmoid(risk_params[:, 1]) * 0.02  # Max 2% std
        take_profit_mean = torch.sigmoid(risk_params[:, 2]) * 0.2  # Max 20% mean
        take_profit_std = torch.sigmoid(risk_params[:, 3]) * 0.05  # Max 5% std
        
        # Meta parameters for trading style
        meta_params = self.meta_head(x_pooled)
        aggression = torch.sigmoid(meta_params[:, 0])  # How aggressive
        patience = torch.sigmoid(meta_params[:, 1])  # How patient
        risk_tolerance = torch.sigmoid(meta_params[:, 2])  # Risk tolerance
        
        # Value estimation
        value = self.value_head(x_pooled).squeeze(-1)
        
        return {
            'trade_logits': trade_logits,
            'position_mean': pos_mean,
            'position_std': pos_std,
            'max_position': max_position,
            'stop_loss_mean': stop_loss_mean,
            'stop_loss_std': stop_loss_std,
            'take_profit_mean': take_profit_mean,
            'take_profit_std': take_profit_std,
            'aggression': aggression,
            'patience': patience,
            'risk_tolerance': risk_tolerance,
            'value': value
        }


class ImprovedRLEnvironment:
    """Environment with better reward shaping to encourage trading"""
    
    def __init__(self, data: np.ndarray, config: ModernTradingConfig):
        self.data = data
        self.config = config
        self.reset()
        
    def reset(self):
        self.current_step = self.config.sequence_length
        self.capital = self.config.initial_capital
        self.position = 0
        self.entry_price = 0
        self.trades_executed = 0
        self.winning_trades = 0
        self.total_return = 0
        self.peak_capital = self.capital
        
        return self._get_state()
    
    def _get_state(self):
        # Get market data
        start = self.current_step - self.config.sequence_length
        market_data = self.data[start:self.current_step]
        
        # Market statistics for conditioning
        recent_returns = np.diff(self.data[start:self.current_step, 3])
        volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0.01
        trend = np.mean(recent_returns) if len(recent_returns) > 0 else 0
        
        market_state = np.concatenate([
            self.data[self.current_step - 1],  # Current bar
            [volatility, trend, 
             self.position / self.capital if self.capital > 0 else 0,
             self.trades_executed / 100,
             self.winning_trades / (self.trades_executed + 1),
             (self.capital - self.config.initial_capital) / self.config.initial_capital,
             0, 0, 0, 0]  # Padding to match feature_dim * 2
        ])[:self.config.feature_dim * 2]
        
        return market_data, market_state
    
    def step(self, action: Dict) -> Tuple:
        current_price = self.data[self.current_step, 3]
        prev_capital = self.capital + self.position * current_price
        
        # Execute action
        trade_action = action['trade']
        position_size = action.get('position_size', 0.1)
        stop_loss = action.get('stop_loss', 0.02)
        take_profit = action.get('take_profit', 0.05)
        
        reward = 0
        trade_executed = False
        
        # Buy action
        if trade_action == 1 and self.position == 0:
            # Use learned max position
            max_position = action.get('max_position', 0.3)
            trade_value = self.capital * position_size * max_position
            
            if trade_value >= self.config.min_trade_size and trade_value <= self.capital:
                self.position = trade_value / current_price
                self.entry_price = current_price
                self.capital -= trade_value
                self.trades_executed += 1
                trade_executed = True
                
                # Small reward for executing trade
                reward += 0.001
        
        # Sell action
        elif trade_action == 2 and self.position > 0:
            exit_value = self.position * current_price
            trade_return = (current_price - self.entry_price) / self.entry_price
            
            self.capital += exit_value
            
            if trade_return > 0:
                self.winning_trades += 1
                reward += trade_return  # Positive return
            else:
                reward += trade_return * 0.5  # Less penalty for losses
            
            self.total_return += trade_return
            self.position = 0
            self.entry_price = 0
            trade_executed = True
        
        # Check stop-loss/take-profit
        if self.position > 0:
            current_return = (current_price - self.entry_price) / self.entry_price
            
            if current_return <= -stop_loss:
                # Stop loss hit
                exit_value = self.position * current_price
                self.capital += exit_value
                reward += current_return * 0.5  # Reduced penalty
                self.position = 0
                trade_executed = True
                
            elif current_return >= take_profit:
                # Take profit hit
                exit_value = self.position * current_price
                self.capital += exit_value
                self.winning_trades += 1
                reward += current_return * 1.5  # Bonus for hitting TP
                self.position = 0
                trade_executed = True
        
        # Calculate current equity
        current_equity = self.capital + self.position * current_price
        
        # Reward shaping
        # 1. Equity change
        equity_change = (current_equity - prev_capital) / self.config.initial_capital
        reward += equity_change * 10
        
        # 2. Encourage trading (small penalty for not trading)
        if not trade_executed and self.trades_executed < 10:
            reward -= 0.0001
        
        # 3. Risk-adjusted reward
        if self.trades_executed > 5:
            win_rate = self.winning_trades / self.trades_executed
            reward += win_rate * 0.01
        
        # 4. Drawdown penalty
        self.peak_capital = max(self.peak_capital, current_equity)
        drawdown = (self.peak_capital - current_equity) / self.peak_capital
        if drawdown > 0.1:  # More than 10% drawdown
            reward -= drawdown * 0.1
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Terminal reward
        if done:
            final_return = (current_equity - self.config.initial_capital) / self.config.initial_capital
            if final_return > 0:
                reward += final_return * 10  # Big bonus for profit
            
            if self.trades_executed == 0:
                reward -= 1.0  # Big penalty for not trading at all
        
        next_state = self._get_state() if not done else None
        
        info = {
            'trades': self.trades_executed,
            'win_rate': self.winning_trades / (self.trades_executed + 1),
            'total_return': self.total_return,
            'current_equity': current_equity
        }
        
        return next_state, reward, done, info


def train_modern_dit_rl(max_minutes: float = 2):
    """Train the modern DiT-based RL trader"""
    
    config = ModernTradingConfig()
    
    # Setup
    logger.info("Starting Modern DiT RL Training")
    checkpoint_dir = Path("hftraining/checkpoints/modern_dit")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Get data
    logger.info("Downloading data...")
    symbols = ['SPY', 'QQQ', 'AAPL']
    all_data = []
    
    for symbol in symbols:
        stock_data = download_stock_data(symbol, start_date='2020-01-01')
        if symbol in stock_data:
            df = stock_data[symbol]
            all_data.append(df)
            logger.info(f"Downloaded {len(df)} records for {symbol}")
    
    # Process data
    combined_df = pd.concat(all_data, ignore_index=True)
    processor = StockDataProcessor()
    features = processor.prepare_features(combined_df)
    processor.fit_scalers(features)
    normalized_data = processor.transform(features)
    
    # Ensure we have the right feature dimension
    if normalized_data.shape[1] != config.feature_dim:
        # Pad or truncate to match expected dimension
        if normalized_data.shape[1] < config.feature_dim:
            padding = np.zeros((len(normalized_data), config.feature_dim - normalized_data.shape[1]))
            normalized_data = np.concatenate([normalized_data, padding], axis=1)
        else:
            normalized_data = normalized_data[:, :config.feature_dim]
    
    # Split data
    train_data, val_data, _ = split_data(normalized_data, 0.7, 0.15, 0.15)
    
    # Create environments
    train_env = ImprovedRLEnvironment(train_data, config)
    val_env = ImprovedRLEnvironment(val_data, config)
    
    # Create model
    model = ModernDiTTrader(config)
    device = torch.device('cpu')  # Force CPU for stability
    model = model.to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Using device: {device}")
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate * 10,
        total_steps=1000,
        pct_start=0.1
    )
    
    # Training loop
    start_time = time.time()
    best_val_return = -float('inf')
    best_model_state = None
    episode = 0
    
    while (time.time() - start_time) / 60 < max_minutes:
        episode += 1
        
        # Training episode
        state = train_env.reset()
        episode_reward = 0
        episode_steps = 0
        
        while episode_steps < 500:  # Limit steps
            episode_steps += 1
            
            # Prepare inputs
            market_data, market_state = state
            market_tensor = torch.FloatTensor(market_data).unsqueeze(0).to(device)
            state_tensor = torch.FloatTensor(market_state).unsqueeze(0).to(device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = model(market_tensor, state_tensor)
            
            # Sample actions with exploration
            epsilon = max(0.05, 1.0 - episode * 0.01)
            
            if np.random.random() < epsilon:
                # Exploration with smart randomization
                trade_action = np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3])
                position_size = np.random.beta(2, 5)  # Beta distribution favors smaller sizes
                max_position = 0.1 + np.random.random() * 0.4  # 10-50%
                stop_loss = 0.01 + np.random.random() * 0.05
                take_profit = 0.02 + np.random.random() * 0.08
            else:
                # Use model predictions
                trade_probs = F.softmax(outputs['trade_logits'], dim=-1)
                trade_action = torch.multinomial(trade_probs, 1).item()
                
                # Sample from learned distributions
                position_size = torch.clamp(
                    torch.normal(outputs['position_mean'], outputs['position_std']),
                    0, 1
                ).item()
                max_position = outputs['max_position'].item()
                
                stop_loss = torch.clamp(
                    torch.normal(outputs['stop_loss_mean'], outputs['stop_loss_std']),
                    0.005, 0.1
                ).item()
                
                take_profit = torch.clamp(
                    torch.normal(outputs['take_profit_mean'], outputs['take_profit_std']),
                    0.01, 0.2
                ).item()
            
            action = {
                'trade': trade_action,
                'position_size': position_size,
                'max_position': max_position,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
            # Take step
            next_state, reward, done, info = train_env.step(action)
            episode_reward += reward
            
            # Update model
            outputs = model(market_tensor, state_tensor)
            
            # Calculate loss
            with torch.no_grad():
                if next_state is not None:
                    next_market, next_state_data = next_state
                    next_market_tensor = torch.FloatTensor(next_market).unsqueeze(0).to(device)
                    next_state_tensor = torch.FloatTensor(next_state_data).unsqueeze(0).to(device)
                    next_outputs = model(next_market_tensor, next_state_tensor)
                    next_value = next_outputs['value']
                else:
                    next_value = torch.zeros(1).to(device)
            
            # TD target
            td_target = reward + config.gamma * next_value * (1 - done)
            td_error = td_target - outputs['value']
            
            # Policy gradient loss
            log_prob = F.log_softmax(outputs['trade_logits'], dim=-1)[0, trade_action]
            actor_loss = -log_prob * td_error.detach()
            
            # Value loss
            critic_loss = F.smooth_l1_loss(outputs['value'], td_target.detach())
            
            # Entropy bonus for exploration
            entropy = -(F.softmax(outputs['trade_logits'], dim=-1) * 
                       F.log_softmax(outputs['trade_logits'], dim=-1)).sum()
            
            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            if done:
                break
            
            state = next_state
        
        # Log progress
        if episode % 5 == 0:
            logger.info(
                f"Episode {episode} | "
                f"Reward: {episode_reward:.4f} | "
                f"Trades: {info['trades']} | "
                f"Return: {info['total_return']:.2%} | "
                f"Equity: ${info['current_equity']:.2f}"
            )
            
            # Validation
            val_state = val_env.reset()
            val_done = False
            val_steps = 0
            
            while not val_done and val_steps < 500:
                val_steps += 1
                market_data, market_state = val_state
                
                with torch.no_grad():
                    market_tensor = torch.FloatTensor(market_data).unsqueeze(0).to(device)
                    state_tensor = torch.FloatTensor(market_state).unsqueeze(0).to(device)
                    outputs = model(market_tensor, state_tensor)
                
                action = {
                    'trade': torch.argmax(outputs['trade_logits']).item(),
                    'position_size': outputs['position_mean'].item(),
                    'max_position': outputs['max_position'].item(),
                    'stop_loss': outputs['stop_loss_mean'].item(),
                    'take_profit': outputs['take_profit_mean'].item()
                }
                
                val_state, _, val_done, val_info = val_env.step(action)
            
            val_return = val_info['total_return']
            
            if val_return > best_val_return:
                best_val_return = val_return
                best_model_state = model.state_dict().copy()
                logger.info(f"💰 New best model! Val Return: {val_return:.2%}")
    
    # Save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save({
            'model_state_dict': best_model_state,
            'best_return': best_val_return,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_dir / "best_model.pth")
        
        logger.info(f"✅ Saved best model with return: {best_val_return:.2%}")
    
    elapsed = (time.time() - start_time) / 60
    logger.info(f"Training completed in {elapsed:.2f} minutes")
    
    return model, best_val_return


if __name__ == "__main__":
    model, best_return = train_modern_dit_rl(max_minutes=2)
    print(f"\n{'='*50}")
    print(f"MODERN DiT RL TRAINING COMPLETE")
    print(f"Best Validation Return: {best_return:.2%}")
    print(f"Model uses DiT blocks with learnable position limits")
    print(f"{'='*50}")