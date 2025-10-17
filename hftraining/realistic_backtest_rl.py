#!/usr/bin/env python3
"""
Realistic Backtesting RL Trading System
Incorporates real-world trading constraints and feeds backtesting metrics directly into rewards
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
# yfinance removed; rely on local CSVs if needed

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.append(os.path.dirname(current_dir))

from data_utils import StockDataProcessor, split_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class RealisticTradingConfig:
    """Configuration with realistic market constraints"""
    
    # Market microstructure
    commission_rate: float = 0.001  # 0.1% per trade
    bid_ask_spread: float = 0.0002  # 0.02% typical spread
    market_impact: float = 0.0001  # Price impact per $100k traded
    min_trade_size: float = 100  # Minimum $100 per trade
    max_daily_trades: int = 10  # PDT rule consideration
    
    # Slippage model (dynamic based on volatility and volume)
    base_slippage: float = 0.0005
    volatility_slippage_mult: float = 0.5  # Slippage increases with volatility
    size_slippage_mult: float = 0.1  # Slippage increases with trade size
    
    # Risk constraints
    max_position_pct: float = 0.15  # Max 15% in one position (more conservative)
    max_leverage: float = 1.0  # No leverage for retail
    margin_requirement: float = 0.25  # 25% margin requirement
    max_drawdown_limit: float = 0.30  # 30% max drawdown before stopping (allow more room)
    
    # Execution delays
    order_delay_bars: int = 1  # Execute orders at next bar (realistic)
    
    # Capital
    initial_capital: float = 25000  # PDT minimum
    
    # Model architecture
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    dropout: float = 0.1
    
    # RL parameters
    gamma: float = 0.95  # Slightly lower for more immediate rewards
    learning_rate: float = 1e-4
    batch_size: int = 32
    replay_buffer_size: int = 10000
    
    # Data
    sequence_length: int = 60  # 60 bars of history
    prediction_horizon: int = 5


@dataclass
class BacktestMetrics:
    """Comprehensive backtesting metrics"""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0
    avg_holding_period: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    
    def to_dict(self):
        return {k: float(v) if not isinstance(v, int) else v 
                for k, v in self.__dict__.items()}


class RealisticMarketSimulator:
    """Simulates realistic market conditions"""
    
    def __init__(self, data: np.ndarray, config: RealisticTradingConfig):
        self.data = data
        self.config = config
        self.current_bar = config.sequence_length
        
        # Extract OHLCV data
        self.opens = data[:, 0]
        self.highs = data[:, 1]
        self.lows = data[:, 2]
        self.closes = data[:, 3]
        self.volumes = data[:, 4] if data.shape[1] > 4 else np.ones(len(data))
        
        # Calculate market metrics
        self._calculate_market_metrics()
        
    def _calculate_market_metrics(self):
        """Pre-calculate market metrics for efficiency"""
        # Rolling volatility (20-period)
        returns = np.diff(np.log(self.closes + 1e-8))
        self.volatility = pd.Series(returns).rolling(20).std().fillna(0.01).values
        
        # Average volume for liquidity estimation
        self.avg_volume = pd.Series(self.volumes).rolling(20).mean().fillna(
            self.volumes.mean()).values
        
        # Intraday volatility (high-low)
        self.intraday_vol = (self.highs - self.lows) / (self.closes + 1e-8)
        
    def get_execution_price(self, bar_idx: int, is_buy: bool, size: float) -> Tuple[float, float]:
        """
        Get realistic execution price with slippage and spread
        
        Returns:
            (execution_price, total_slippage)
        """
        # Base price (use next bar's open for realism)
        if bar_idx + 1 < len(self.opens):
            base_price = self.opens[bar_idx + 1]
        else:
            base_price = self.closes[bar_idx]
        
        # Bid-ask spread cost
        spread_cost = base_price * self.config.bid_ask_spread
        
        # Dynamic slippage based on volatility
        vol_slippage = self.config.base_slippage * (
            1 + self.config.volatility_slippage_mult * self.volatility[bar_idx]
        )
        
        # Size-based slippage (market impact)
        size_ratio = size / (self.avg_volume[bar_idx] * base_price + 1e-8)
        size_slippage = self.config.size_slippage_mult * size_ratio
        
        # Total slippage
        total_slippage = vol_slippage + size_slippage + self.config.market_impact * (size / 100000)
        
        # Final execution price
        if is_buy:
            execution_price = base_price * (1 + spread_cost/2 + total_slippage)
        else:
            execution_price = base_price * (1 - spread_cost/2 - total_slippage)
        
        return execution_price, total_slippage * base_price
    
    def check_stop_loss_take_profit(self, bar_idx: int, entry_price: float, 
                                   stop_loss: float, take_profit: float) -> Optional[Tuple[str, float]]:
        """
        Check if stop-loss or take-profit triggered using high/low prices
        
        Returns:
            ('stop_loss', exit_price) or ('take_profit', exit_price) or None
        """
        if bar_idx >= len(self.highs):
            return None
            
        high = self.highs[bar_idx]
        low = self.lows[bar_idx]
        
        stop_price = entry_price * (1 - stop_loss)
        profit_price = entry_price * (1 + take_profit)
        
        # Check if stop-loss hit (use low)
        if low <= stop_price:
            # Assume we exit at stop price with some slippage
            exit_price = stop_price * (1 - self.config.base_slippage)
            return ('stop_loss', exit_price)
        
        # Check if take-profit hit (use high)
        if high >= profit_price:
            # Assume we exit at profit price with minimal slippage
            exit_price = profit_price * (1 - self.config.base_slippage * 0.5)
            return ('take_profit', exit_price)
        
        return None


class RealisticTradingEnvironment:
    """Trading environment with realistic constraints and backtesting metrics"""
    
    def __init__(self, data: np.ndarray, config: RealisticTradingConfig):
        self.data = data
        self.config = config
        self.market_sim = RealisticMarketSimulator(data, config)
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_bar = self.config.sequence_length
        self.capital = self.config.initial_capital
        self.position = 0
        self.entry_price = 0
        self.entry_bar = 0
        
        # Pending orders (realistic order execution)
        self.pending_order = None
        
        # Daily trade counter (PDT rule)
        self.daily_trades = 0
        self.last_trade_day = 0
        
        # Track metrics
        self.metrics = BacktestMetrics()
        self.trade_history = []
        self.equity_curve = [self.capital]
        self.peak_equity = self.capital
        
        # Position tracking
        self.position_bars = 0
        self.total_commission = 0
        self.total_slippage = 0
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state observation with market context"""
        # Historical market data
        start_idx = self.current_bar - self.config.sequence_length
        end_idx = self.current_bar
        market_data = self.data[start_idx:end_idx]
        
        # Current market conditions
        current_vol = self.market_sim.volatility[self.current_bar - 1]
        current_volume = self.market_sim.avg_volume[self.current_bar - 1]
        
        # Portfolio state
        position_value = self.position * self.market_sim.closes[self.current_bar - 1]
        portfolio_state = np.array([
            self.position / (self.capital + position_value + 1e-8),  # Position ratio
            (self.capital - self.config.initial_capital) / self.config.initial_capital,  # P&L
            self.daily_trades / self.config.max_daily_trades,  # Trade capacity used
            self.metrics.win_rate,  # Historical win rate
            self.metrics.sharpe_ratio / 3.0,  # Normalized Sharpe
            current_vol / 0.02,  # Normalized volatility
            np.log(current_volume / 1e6 + 1),  # Log volume in millions
            self.position_bars / 100,  # Holding period
        ])
        
        return market_data, portfolio_state
    
    def step(self, action: Dict) -> Tuple:
        """
        Execute action with realistic constraints
        
        Returns:
            (next_state, reward, done, metrics)
        """
        # Execute pending order from previous step
        if self.pending_order is not None:
            self._execute_pending_order()
        
        # Parse action
        trade_action = action['trade']  # 0: hold, 1: buy, 2: sell
        position_size = action['position_size']  # 0-1 normalized
        stop_loss = action.get('stop_loss', 0.02)  # Default 2% stop
        take_profit = action.get('take_profit', 0.05)  # Default 5% profit
        
        # Check daily trade limit
        current_day = self.current_bar // 390  # Assuming 390 bars per day
        if current_day != self.last_trade_day:
            self.daily_trades = 0
            self.last_trade_day = current_day
        
        # Process new order
        if trade_action == 1 and self.position == 0 and self.daily_trades < self.config.max_daily_trades:
            # Buy signal - create pending order
            order_size = min(
                self.capital * position_size * self.config.max_position_pct,
                self.capital - self.config.min_trade_size  # Reserve for min trade
            )
            
            if order_size >= self.config.min_trade_size:
                self.pending_order = {
                    'type': 'buy',
                    'size': order_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'bar': self.current_bar
                }
                
        elif trade_action == 2 and self.position > 0:
            # Sell signal - create pending order
            self.pending_order = {
                'type': 'sell',
                'size': self.position,
                'bar': self.current_bar
            }
        
        # Check stop-loss and take-profit for existing position
        if self.position > 0 and self.entry_price > 0:
            exit_signal = self.market_sim.check_stop_loss_take_profit(
                self.current_bar,
                self.entry_price,
                self.stop_loss,
                self.take_profit
            )
            
            if exit_signal:
                exit_type, exit_price = exit_signal
                self._close_position(exit_price, exit_type)
        
        # Update position holding period
        if self.position > 0:
            self.position_bars += 1
        
        # Calculate reward based on backtesting metrics
        reward = self._calculate_reward()
        
        # Update equity curve
        position_value = self.position * self.market_sim.closes[self.current_bar]
        current_equity = self.capital + position_value
        self.equity_curve.append(current_equity)
        
        # Update peak and drawdown
        self.peak_equity = max(self.peak_equity, current_equity)
        current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        self.metrics.max_drawdown = max(self.metrics.max_drawdown, current_drawdown)
        
        # Check if done
        done = False
        if self.current_bar >= len(self.data) - 2:
            done = True
        elif current_drawdown > self.config.max_drawdown_limit:
            logger.warning(f"Max drawdown limit reached: {current_drawdown:.2%}")
            done = True
        elif current_equity < self.config.min_trade_size:
            logger.warning(f"Insufficient capital: ${current_equity:.2f}")
            done = True
        
        # Move to next bar
        self.current_bar += 1
        
        # Get next state
        next_state = self._get_state() if not done else None
        
        # Calculate final metrics
        self._update_metrics()
        
        return next_state, reward, done, self.metrics
    
    def _execute_pending_order(self):
        """Execute pending order with realistic fills"""
        order = self.pending_order
        self.pending_order = None
        
        if order['type'] == 'buy':
            # Get execution price with slippage
            exec_price, slippage = self.market_sim.get_execution_price(
                order['bar'], True, order['size']
            )
            
            # Calculate shares
            shares = order['size'] / exec_price
            
            # Execute trade
            commission = order['size'] * self.config.commission_rate
            total_cost = shares * exec_price + commission
            
            if total_cost <= self.capital:
                self.position = shares
                self.entry_price = exec_price
                self.entry_bar = order['bar']
                self.stop_loss = order['stop_loss']
                self.take_profit = order['take_profit']
                self.capital -= total_cost
                self.daily_trades += 1
                self.position_bars = 0
                
                # Track costs
                self.total_commission += commission
                self.total_slippage += slippage * shares
                
                # Record trade
                self.trade_history.append({
                    'bar': order['bar'],
                    'type': 'buy',
                    'price': exec_price,
                    'shares': shares,
                    'commission': commission,
                    'slippage': slippage
                })
                
        elif order['type'] == 'sell' and self.position > 0:
            self._close_position(None, 'signal')
    
    def _close_position(self, exit_price: Optional[float], exit_type: str):
        """Close position with realistic execution"""
        if self.position <= 0:
            return
            
        # Get execution price if not provided
        if exit_price is None:
            exit_price, slippage = self.market_sim.get_execution_price(
                self.current_bar, False, self.position * self.market_sim.closes[self.current_bar]
            )
        else:
            slippage = 0
        
        # Calculate proceeds and commission
        proceeds = self.position * exit_price
        commission = proceeds * self.config.commission_rate
        net_proceeds = proceeds - commission
        
        # Calculate return
        cost_basis = self.position * self.entry_price
        trade_return = (net_proceeds - cost_basis) / cost_basis
        
        # Update capital
        self.capital += net_proceeds
        
        # Update metrics
        self.metrics.total_trades += 1
        if trade_return > 0:
            self.metrics.total_return += trade_return
            self.metrics.avg_win = (
                self.metrics.avg_win * (self.metrics.total_trades - 1) + trade_return
            ) / self.metrics.total_trades
        else:
            self.metrics.avg_loss = (
                self.metrics.avg_loss * (self.metrics.total_trades - 1) + abs(trade_return)
            ) / self.metrics.total_trades
        
        # Track costs
        self.total_commission += commission
        self.total_slippage += slippage * self.position
        
        # Record trade
        self.trade_history.append({
            'bar': self.current_bar,
            'type': f'sell_{exit_type}',
            'price': exit_price,
            'shares': self.position,
            'return': trade_return,
            'commission': commission,
            'holding_period': self.position_bars
        })
        
        # Reset position
        self.position = 0
        self.entry_price = 0
        self.daily_trades += 1
        
        # Update average holding period
        self.metrics.avg_holding_period = (
            self.metrics.avg_holding_period * (self.metrics.total_trades - 1) + self.position_bars
        ) / self.metrics.total_trades
        
        self.position_bars = 0
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward incorporating realistic backtesting metrics
        """
        # Get current equity
        position_value = self.position * self.market_sim.closes[self.current_bar]
        current_equity = self.capital + position_value
        
        # Base reward: equity change
        equity_change = (current_equity - self.equity_curve[-1]) / self.config.initial_capital
        
        # Risk-adjusted component (Sharpe-like)
        if len(self.equity_curve) > 20:
            returns = np.diff(self.equity_curve[-20:]) / self.equity_curve[-20:-1]
            if returns.std() > 0:
                sharpe_component = returns.mean() / returns.std()
            else:
                sharpe_component = 0
        else:
            sharpe_component = 0
        
        # Drawdown penalty
        current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        drawdown_penalty = -current_drawdown * 0.5
        
        # Cost penalty (encourage efficient trading)
        cost_ratio = (self.total_commission + self.total_slippage) / self.config.initial_capital
        cost_penalty = -cost_ratio * 10
        
        # Win rate bonus
        win_rate_bonus = self.metrics.win_rate * 0.1 if self.metrics.total_trades > 5 else 0
        
        # Combine reward components
        reward = (
            equity_change * 1.0 +
            sharpe_component * 0.3 +
            drawdown_penalty +
            cost_penalty +
            win_rate_bonus
        )
        
        return reward
    
    def _update_metrics(self):
        """Update comprehensive backtesting metrics"""
        if len(self.trade_history) == 0:
            return
        
        # Calculate returns
        returns = [t['return'] for t in self.trade_history if 'return' in t]
        if returns:
            positive_returns = [r for r in returns if r > 0]
            negative_returns = [r for r in returns if r < 0]
            
            # Win rate
            self.metrics.win_rate = len(positive_returns) / len(returns) if returns else 0
            
            # Profit factor
            gross_profit = sum(positive_returns) if positive_returns else 0
            gross_loss = abs(sum(negative_returns)) if negative_returns else 1e-8
            self.metrics.profit_factor = gross_profit / gross_loss
            
            # Sharpe ratio (annualized)
            if len(returns) > 1:
                returns_array = np.array(returns)
                self.metrics.sharpe_ratio = (
                    np.mean(returns_array) / (np.std(returns_array) + 1e-8) * np.sqrt(252)
                )
            
            # Sortino ratio (downside deviation)
            if negative_returns:
                downside_dev = np.std(negative_returns)
                self.metrics.sortino_ratio = (
                    np.mean(returns) / (downside_dev + 1e-8) * np.sqrt(252)
                )
            
            # Calmar ratio
            if self.metrics.max_drawdown > 0:
                annual_return = self.metrics.total_return * (252 / self.metrics.total_trades)
                self.metrics.calmar_ratio = annual_return / self.metrics.max_drawdown
        
        # Update costs
        self.metrics.total_commission = self.total_commission
        self.metrics.total_slippage = self.total_slippage


class RealisticRLModel(nn.Module):
    """RL model for realistic trading"""
    
    def __init__(self, config: RealisticTradingConfig, input_dim: int):
        super().__init__()
        self.config = config
        
        # Market encoder
        self.market_encoder = nn.LSTM(
            input_dim,
            config.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout
        )
        
        # Portfolio state encoder
        self.portfolio_encoder = nn.Sequential(
            nn.Linear(8, config.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 4, config.hidden_size // 2)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Feature combiner
        self.combiner = nn.Sequential(
            nn.Linear(config.hidden_size + config.hidden_size // 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 128),
            nn.ReLU()
        )
        
        # Trade action output
        self.trade_action = nn.Linear(128, 3)  # hold, buy, sell
        
        # Position size output
        self.position_size = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Risk management outputs
        self.stop_loss = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.take_profit = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1)
        )
    
    def forward(self, market_data, portfolio_state):
        """Forward pass"""
        # Encode market data
        lstm_out, _ = self.market_encoder(market_data)
        
        # Apply self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last hidden state
        market_features = attn_out[:, -1, :]
        
        # Encode portfolio state
        portfolio_features = self.portfolio_encoder(portfolio_state)
        
        # Combine features
        combined = torch.cat([market_features, portfolio_features], dim=-1)
        features = self.combiner(combined)
        
        # Policy outputs
        policy_features = self.policy_head(features)
        
        trade_logits = self.trade_action(policy_features)
        position_size = self.position_size(policy_features).squeeze(-1)
        stop_loss = self.stop_loss(policy_features).squeeze(-1) * 0.1  # Max 10% stop
        take_profit = self.take_profit(policy_features).squeeze(-1) * 0.2  # Max 20% profit
        
        # Value output
        value = self.value_head(features).squeeze(-1)
        
        return {
            'trade_logits': trade_logits,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'value': value
        }


def train_realistic_rl(max_minutes=2):
    """Train with realistic backtesting"""
    
    # Configuration
    config = RealisticTradingConfig()
    
    # Setup paths
    checkpoint_dir = Path("hftraining/checkpoints/realistic_rl")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load local CSVs
    logger.info("Loading local stock CSVs...")
    symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']
    data_dir = Path("trainingdata")
    all_data = []
    for symbol in symbols[:2]:  # Start with 2 for testing
        candidates = list(data_dir.glob(f"{symbol}.csv"))
        if not candidates:
            candidates = [p for p in data_dir.glob("*.csv") if symbol.lower() in p.stem.lower()]
        if not candidates:
            logger.warning(f"No local CSV found for {symbol}")
            continue
        df = pd.read_csv(candidates[0])
        df.columns = df.columns.str.lower()
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
            except Exception:
                pass
        # Ensure volume exists
        if 'volume' not in df.columns:
            df['volume'] = 1e6
        all_data.append(df)
    
    # Process data
    combined_df = pd.concat(all_data, ignore_index=True)
    processor = StockDataProcessor()
    features = processor.prepare_features(combined_df)
    processor.fit_scalers(features)
    normalized_data = processor.transform(features)
    
    # Split data
    train_data, val_data, _ = split_data(normalized_data, 0.7, 0.15, 0.15)
    
    # Create environments
    train_env = RealisticTradingEnvironment(train_data, config)
    val_env = RealisticTradingEnvironment(val_data, config)
    
    # Create model
    input_dim = normalized_data.shape[1]
    model = RealisticRLModel(config, input_dim)
    device = torch.device('cpu')  # Force CPU for stability
    model = model.to(device)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Using device: {device}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)
    
    # Training loop
    start_time = time.time()
    best_sharpe = -float('inf')
    best_model_state = None
    val_metrics: BacktestMetrics = BacktestMetrics()
    episode = 0
    
    logger.info(f"Starting realistic RL training for {max_minutes} minutes...")
    
    while (time.time() - start_time) / 60 < max_minutes:
        episode += 1
        
        # Training episode
        state = train_env.reset()
        episode_reward = 0
        episode_losses = []
        
        if episode % 1 == 0:
            logger.info(f"Starting episode {episode}...")
        
        step_count = 0
        max_steps = 1000  # Prevent infinite loops
        
        while step_count < max_steps:
            step_count += 1
            
            # Get action from model
            market_data, portfolio_state = state
            
            market_tensor = torch.FloatTensor(market_data).unsqueeze(0).to(device)
            portfolio_tensor = torch.FloatTensor(portfolio_state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(market_tensor, portfolio_tensor)
            
            # Sample action (with exploration)
            epsilon = max(0.1, 1.0 - episode * 0.02)  # Faster epsilon decay
            if np.random.random() < epsilon:
                # Bias towards holding during exploration
                trade_action = np.random.choice(3, p=[0.6, 0.2, 0.2])  # 60% hold, 20% buy, 20% sell
                position_size = np.random.random() * 0.5  # Smaller positions during exploration
                stop_loss = 0.02 + np.random.random() * 0.03  # 2-5% stop loss
                take_profit = 0.03 + np.random.random() * 0.07  # 3-10% take profit
            else:
                trade_probs = F.softmax(outputs['trade_logits'], dim=-1)
                trade_action = torch.multinomial(trade_probs, 1).item()
                position_size = outputs['position_size'].item()
                stop_loss = outputs['stop_loss'].item()
                take_profit = outputs['take_profit'].item()
            
            action = {
                'trade': trade_action,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
            # Take step
            next_state, reward, done, metrics = train_env.step(action)
            episode_reward += reward
            
            # Calculate loss and update
            outputs = model(market_tensor, portfolio_tensor)
            
            # Policy gradient loss
            log_prob = F.log_softmax(outputs['trade_logits'], dim=-1)[0, trade_action]
            
            # Value loss
            with torch.no_grad():
                if next_state is not None:
                    next_market, next_portfolio = next_state
                    next_market_tensor = torch.FloatTensor(next_market).unsqueeze(0).to(device)
                    next_portfolio_tensor = torch.FloatTensor(next_portfolio).unsqueeze(0).to(device)
                    next_outputs = model(next_market_tensor, next_portfolio_tensor)
                    next_value = next_outputs['value']
                else:
                    next_value = torch.zeros(1).to(device)
            
            # TD target
            td_target = reward + config.gamma * next_value * (1 - done)
            td_error = td_target - outputs['value']
            
            # Combined loss
            actor_loss = -log_prob * td_error.detach()
            critic_loss = F.smooth_l1_loss(outputs['value'], td_target.detach())
            
            total_loss = actor_loss + 0.5 * critic_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            episode_losses.append(total_loss.item())
            
            if done:
                break
            
            state = next_state
        
        # Update learning rate
        scheduler.step()
        
        # Validation
        if episode % 5 == 0:
            val_state = val_env.reset()
            val_done = False
            
            while not val_done:
                market_data, portfolio_state = val_state
                market_tensor = torch.FloatTensor(market_data).unsqueeze(0).to(device)
                portfolio_tensor = torch.FloatTensor(portfolio_state).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(market_tensor, portfolio_tensor)
                
                trade_action = torch.argmax(outputs['trade_logits']).item()
                
                action = {
                    'trade': trade_action,
                    'position_size': outputs['position_size'].item(),
                    'stop_loss': outputs['stop_loss'].item(),
                    'take_profit': outputs['take_profit'].item()
                }
                
                val_state, _, val_done, val_metrics = val_env.step(action)
            
            # Check if best model
            if val_metrics.sharpe_ratio > best_sharpe:
                best_sharpe = val_metrics.sharpe_ratio
                best_model_state = model.state_dict().copy()
                logger.info(f"💰 New best model! Sharpe: {val_metrics.sharpe_ratio:.3f}")
            
            # Log progress
            logger.info(
                f"Episode {episode} | "
                f"Train Return: {metrics.total_return:.2%} | "
                f"Val Sharpe: {val_metrics.sharpe_ratio:.3f} | "
                f"Trades: {val_metrics.total_trades} | "
                f"Win Rate: {val_metrics.win_rate:.1%} | "
                f"Max DD: {val_metrics.max_drawdown:.2%}"
            )
    
    # Save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save({
            'model_state_dict': best_model_state,
            'best_sharpe': best_sharpe,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_dir / "best_realistic_model.pth")
        
        logger.info(f"✅ Saved best model with Sharpe: {best_sharpe:.3f}")
    
    elapsed = (time.time() - start_time) / 60
    logger.info(f"Training completed in {elapsed:.2f} minutes")
    
    # Final backtesting report
    logger.info("\n" + "="*50)
    logger.info("REALISTIC BACKTESTING RESULTS:")
    logger.info(f"Best Sharpe Ratio: {best_sharpe:.3f}")
    logger.info(f"Final Metrics: {val_metrics.to_dict()}")
    logger.info("="*50)
    
    return model, val_metrics


if __name__ == "__main__":
    model, metrics = train_realistic_rl(max_minutes=2)
    print(f"\nFinal Backtesting Metrics: {json.dumps(metrics.to_dict(), indent=2)}")
