#!/usr/bin/env python3
"""
Realistic Trading Simulation Environment
- Includes transaction costs, slippage, and market impact
- Proper position management and risk controls
- Realistic profit/loss calculation
- Integration with differentiable training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class PositionSide(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class TradingConfig:
    """Configuration for realistic trading simulation"""
    initial_capital: float = 100000.0
    max_position_size: float = 0.2  # Max 20% of capital per position
    max_leverage: float = 2.0  # Max 2x leverage
    
    # Transaction costs
    commission_rate: float = 0.001  # 0.1% per trade
    slippage_factor: float = 0.0005  # 0.05% slippage
    market_impact_factor: float = 0.0001  # Price impact based on volume
    
    # Risk management
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    max_drawdown: float = 0.15  # 15% max drawdown
    position_hold_time: int = 20  # Max bars to hold position
    
    # Market hours (crypto 24/7, stocks 9:30-4:00)
    market_type: str = "crypto"  # "crypto" or "stock"
    
    # Margin requirements
    margin_requirement: float = 0.25  # 25% margin requirement
    margin_call_level: float = 0.15  # Margin call at 15%
    
    # Realistic constraints
    min_trade_size: float = 100.0  # Minimum trade size in dollars
    max_daily_trades: int = 50  # PDT rule consideration
    
    # Performance metrics
    target_sharpe: float = 1.5
    target_annual_return: float = 0.20  # 20% annual return target


@dataclass
class Position:
    """Represents a trading position"""
    entry_price: float
    size: float  # Positive for long, negative for short
    entry_time: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission_paid: float = 0.0
    
    @property
    def side(self) -> PositionSide:
        if self.size > 0:
            return PositionSide.LONG
        elif self.size < 0:
            return PositionSide.SHORT
        return PositionSide.FLAT
    
    @property
    def value(self) -> float:
        return abs(self.size * self.entry_price)


@dataclass
class Trade:
    """Record of a completed trade"""
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    commission: float
    slippage: float
    return_pct: float
    hold_time: int
    exit_reason: str  # "stop_loss", "take_profit", "signal", "time_limit"


class RealisticTradingEnvironment:
    """Realistic trading simulation with all market frictions"""
    
    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()
        self.reset()
        
    def reset(self):
        """Reset the trading environment"""
        self.capital = self.config.initial_capital
        self.initial_capital = self.config.initial_capital
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.current_step = 0
        self.daily_trades = 0
        self.last_trade_day = 0
        
        # Performance tracking
        self.equity_curve = [self.capital]
        self.returns = []
        self.drawdowns = []
        self.max_equity = self.capital
        self.current_drawdown = 0.0
        
        # Risk metrics
        self.var_95 = 0.0  # Value at Risk
        self.cvar_95 = 0.0  # Conditional VaR
        self.max_drawdown_reached = 0.0
        
        logger.info(f"Trading environment reset with ${self.capital:,.2f} capital")
    
    def calculate_transaction_costs(self, size: float, price: float, 
                                   is_entry: bool = True) -> Dict[str, float]:
        """Calculate realistic transaction costs"""
        
        trade_value = abs(size * price)
        
        # Commission
        commission = trade_value * self.config.commission_rate
        
        # Slippage (higher for larger orders)
        size_factor = min(abs(size) / 10000, 1.0)  # Normalize by typical volume
        slippage_pct = self.config.slippage_factor * (1 + size_factor)
        slippage = trade_value * slippage_pct
        
        # Market impact (square root model)
        market_impact = trade_value * self.config.market_impact_factor * np.sqrt(size_factor)
        
        # Direction matters for slippage
        if is_entry:
            # Pay more when entering
            effective_price = price * (1 + slippage_pct + self.config.market_impact_factor)
        else:
            # Receive less when exiting
            effective_price = price * (1 - slippage_pct - self.config.market_impact_factor)
        
        return {
            'commission': commission,
            'slippage': slippage,
            'market_impact': market_impact,
            'total_cost': commission + slippage + market_impact,
            'effective_price': effective_price
        }
    
    def check_risk_limits(self) -> bool:
        """Check if risk limits are breached"""
        
        # Check drawdown
        if self.current_drawdown > self.config.max_drawdown:
            logger.warning(f"Max drawdown breached: {self.current_drawdown:.2%}")
            return False
        
        # Check position concentration
        total_position_value = sum(abs(p.value) for p in self.positions)
        if total_position_value > self.capital * self.config.max_leverage:
            logger.warning(f"Leverage limit breached: {total_position_value/self.capital:.2f}x")
            return False
        
        # Check margin requirements
        margin_used = total_position_value * self.config.margin_requirement
        if margin_used > self.capital * 0.9:  # Leave 10% buffer
            logger.warning(f"Margin limit approaching: {margin_used/self.capital:.2%}")
            return False
        
        # PDT rule check (for stock trading)
        if self.config.market_type == "stock" and self.capital < 25000:
            if self.daily_trades >= 4:
                logger.warning("Pattern Day Trader rule limit reached")
                return False
        
        return True
    
    def enter_position(self, signal: float, price: float, timestamp: int) -> Optional[Position]:
        """Enter a new position with proper risk management"""
        
        if not self.check_risk_limits():
            return None
        
        # Calculate position size with Kelly Criterion adjustment
        base_size = self.capital * self.config.max_position_size
        
        # Adjust size based on signal strength
        size = base_size * abs(signal)
        
        # Ensure minimum trade size
        if size < self.config.min_trade_size:
            return None
        
        # Calculate costs
        costs = self.calculate_transaction_costs(size, price, is_entry=True)
        
        # Check if we have enough capital
        required_capital = size + costs['total_cost']
        if required_capital > self.capital * 0.95:  # Keep 5% buffer
            size = (self.capital * 0.95 - costs['total_cost']) / price
            if size < self.config.min_trade_size:
                return None
        
        # Create position
        position = Position(
            entry_price=costs['effective_price'],
            size=size if signal > 0 else -size,
            entry_time=timestamp,
            commission_paid=costs['commission']
        )
        
        # Set stop loss and take profit
        if signal > 0:  # Long position
            position.stop_loss = position.entry_price * (1 - self.config.stop_loss_pct)
            position.take_profit = position.entry_price * (1 + self.config.take_profit_pct)
        else:  # Short position
            position.stop_loss = position.entry_price * (1 + self.config.stop_loss_pct)
            position.take_profit = position.entry_price * (1 - self.config.take_profit_pct)
        
        # Update capital
        self.capital -= costs['total_cost']
        
        # Add position
        self.positions.append(position)
        
        # Update daily trade count
        current_day = timestamp // 390  # Assuming 390 minutes per trading day
        if current_day != self.last_trade_day:
            self.daily_trades = 1
            self.last_trade_day = current_day
        else:
            self.daily_trades += 1
        
        logger.debug(f"Entered {position.side.name} position: ${size:.2f} @ ${position.entry_price:.2f}")
        
        return position
    
    def exit_position(self, position: Position, price: float, timestamp: int, 
                     reason: str = "signal") -> Trade:
        """Exit a position and record the trade"""
        
        # Calculate costs
        costs = self.calculate_transaction_costs(position.size, price, is_entry=False)
        
        # Calculate PnL
        if position.size > 0:  # Long position
            gross_pnl = (costs['effective_price'] - position.entry_price) * position.size
        else:  # Short position
            gross_pnl = (position.entry_price - costs['effective_price']) * abs(position.size)
        
        net_pnl = gross_pnl - costs['total_cost'] - position.commission_paid
        
        # Create trade record
        trade = Trade(
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price=position.entry_price,
            exit_price=costs['effective_price'],
            size=position.size,
            pnl=net_pnl,
            commission=costs['commission'] + position.commission_paid,
            slippage=costs['slippage'],
            return_pct=net_pnl / abs(position.value),
            hold_time=timestamp - position.entry_time,
            exit_reason=reason
        )
        
        # Update capital
        self.capital += gross_pnl - costs['total_cost']
        
        # Remove position
        self.positions.remove(position)
        
        # Record trade
        self.trades.append(trade)
        
        logger.debug(f"Exited position: PnL=${net_pnl:.2f} ({trade.return_pct:.2%}), Reason: {reason}")
        
        return trade
    
    def update_positions(self, current_price: float, timestamp: int):
        """Update positions with current price and check stops"""
        
        positions_to_exit = []
        
        for position in self.positions:
            # Update unrealized PnL
            if position.size > 0:  # Long
                position.unrealized_pnl = (current_price - position.entry_price) * position.size
                
                # Check stop loss
                if current_price <= position.stop_loss:
                    positions_to_exit.append((position, "stop_loss"))
                # Check take profit
                elif current_price >= position.take_profit:
                    positions_to_exit.append((position, "take_profit"))
                    
            else:  # Short
                position.unrealized_pnl = (position.entry_price - current_price) * abs(position.size)
                
                # Check stop loss
                if current_price >= position.stop_loss:
                    positions_to_exit.append((position, "stop_loss"))
                # Check take profit
                elif current_price <= position.take_profit:
                    positions_to_exit.append((position, "take_profit"))
            
            # Check holding time limit
            if timestamp - position.entry_time > self.config.position_hold_time:
                positions_to_exit.append((position, "time_limit"))
        
        # Exit positions that hit limits
        for position, reason in positions_to_exit:
            self.exit_position(position, current_price, timestamp, reason)
    
    def step(self, action: Dict[str, torch.Tensor], market_data: Dict[str, float]) -> Dict[str, float]:
        """Execute a trading step with the given action"""
        
        current_price = market_data['price']
        timestamp = market_data.get('timestamp', self.current_step)
        
        # Update existing positions
        self.update_positions(current_price, timestamp)
        
        # Parse action
        signal = action['signal'].item() if isinstance(action['signal'], torch.Tensor) else action['signal']
        confidence = action.get('confidence', torch.tensor(1.0)).item()
        
        # Adjust signal by confidence
        adjusted_signal = signal * confidence
        
        # Position management
        if abs(adjusted_signal) > 0.3:  # Threshold for action
            if len(self.positions) == 0:
                # Enter new position
                self.enter_position(adjusted_signal, current_price, timestamp)
            else:
                # Check if we should reverse position
                current_position = self.positions[0]
                if (current_position.size > 0 and adjusted_signal < -0.5) or \
                   (current_position.size < 0 and adjusted_signal > 0.5):
                    # Exit current and enter opposite
                    self.exit_position(current_position, current_price, timestamp, "signal")
                    self.enter_position(adjusted_signal, current_price, timestamp)
        
        # Update metrics
        self.update_metrics(current_price)
        
        # Calculate reward (for training)
        reward = self.calculate_reward()
        
        self.current_step += 1
        
        return {
            'reward': reward,
            'capital': self.capital,
            'positions': len(self.positions),
            'unrealized_pnl': sum(p.unrealized_pnl for p in self.positions),
            'realized_pnl': sum(t.pnl for t in self.trades),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'max_drawdown': self.max_drawdown_reached,
            'win_rate': self.calculate_win_rate(),
            'profit_factor': self.calculate_profit_factor()
        }
    
    def update_metrics(self, current_price: float):
        """Update performance metrics"""
        
        # Calculate current equity
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions)
        current_equity = self.capital + unrealized_pnl
        self.equity_curve.append(current_equity)
        
        # Update max equity and drawdown
        if current_equity > self.max_equity:
            self.max_equity = current_equity
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.max_equity - current_equity) / self.max_equity
            self.max_drawdown_reached = max(self.max_drawdown_reached, self.current_drawdown)
        
        # Calculate return
        if len(self.equity_curve) > 1:
            period_return = (current_equity - self.equity_curve[-2]) / self.equity_curve[-2]
            self.returns.append(period_return)
        
        # Update VaR and CVaR
        if len(self.returns) > 20:
            sorted_returns = sorted(self.returns[-252:])  # Last year of returns
            var_index = int(len(sorted_returns) * 0.05)
            self.var_95 = sorted_returns[var_index]
            self.cvar_95 = np.mean(sorted_returns[:var_index])
    
    def calculate_reward(self) -> float:
        """Calculate reward for reinforcement learning"""
        
        # Base reward components
        components = []
        
        # 1. Profit component (most important)
        if len(self.equity_curve) > 1:
            profit = (self.equity_curve[-1] - self.equity_curve[-2]) / self.initial_capital
            components.append(profit * 100)  # Scale up
        
        # 2. Risk-adjusted return (Sharpe ratio)
        sharpe = self.calculate_sharpe_ratio()
        if sharpe > 0:
            components.append(sharpe * 0.5)
        
        # 3. Drawdown penalty
        dd_penalty = -self.current_drawdown * 10 if self.current_drawdown > 0.05 else 0
        components.append(dd_penalty)
        
        # 4. Win rate bonus
        win_rate = self.calculate_win_rate()
        if win_rate > 0.5:
            components.append((win_rate - 0.5) * 2)
        
        # 5. Profit factor bonus
        pf = self.calculate_profit_factor()
        if pf > 1.5:
            components.append((pf - 1.5) * 0.5)
        
        # 6. Trade efficiency (avoid overtrading)
        if self.daily_trades > 10:
            components.append(-0.1 * (self.daily_trades - 10))
        
        # Combine components
        reward = sum(components)
        
        # Clip reward to reasonable range
        reward = np.clip(reward, -10, 10)
        
        return reward
    
    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.returns) < 20:
            return 0.0
        
        returns = np.array(self.returns[-252:])  # Last year
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        # Annualized Sharpe ratio
        mean_return = np.mean(returns) * 252
        std_return = np.std(returns) * np.sqrt(252)
        
        return mean_return / std_return if std_return > 0 else 0.0
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate of completed trades"""
        if len(self.trades) == 0:
            return 0.5  # Default to 50%
        
        winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        return winning_trades / len(self.trades)
    
    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if len(self.trades) == 0:
            return 1.0
        
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        
        if gross_loss == 0:
            return 3.0 if gross_profit > 0 else 1.0
        
        return gross_profit / gross_loss
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get comprehensive performance summary"""
        
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        return {
            'total_return': total_return,
            'annual_return': total_return * (252 / max(len(self.equity_curve), 1)),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'max_drawdown': self.max_drawdown_reached,
            'win_rate': self.calculate_win_rate(),
            'profit_factor': self.calculate_profit_factor(),
            'total_trades': len(self.trades),
            'avg_trade_pnl': np.mean([t.pnl for t in self.trades]) if self.trades else 0,
            'avg_win': np.mean([t.pnl for t in self.trades if t.pnl > 0]) if any(t.pnl > 0 for t in self.trades) else 0,
            'avg_loss': np.mean([t.pnl for t in self.trades if t.pnl < 0]) if any(t.pnl < 0 for t in self.trades) else 0,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'current_capital': self.capital,
            'current_equity': self.equity_curve[-1] if self.equity_curve else self.initial_capital
        }
    
    def plot_performance(self, save_path: Optional[str] = None):
        """Plot performance metrics"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Equity curve
        axes[0, 0].plot(self.equity_curve, 'b-', linewidth=2)
        axes[0, 0].axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Capital ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Returns distribution
        if self.returns:
            axes[0, 1].hist(self.returns, bins=50, alpha=0.7, color='green')
            axes[0, 1].axvline(x=0, color='r', linestyle='--')
            axes[0, 1].set_title('Returns Distribution')
            axes[0, 1].set_xlabel('Return')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Drawdown
        drawdown_pct = [(self.max_equity - eq) / self.max_equity * 100 
                       for eq in self.equity_curve]
        axes[0, 2].fill_between(range(len(drawdown_pct)), 0, drawdown_pct, 
                                color='red', alpha=0.3)
        axes[0, 2].set_title('Drawdown %')
        axes[0, 2].set_xlabel('Time')
        axes[0, 2].set_ylabel('Drawdown %')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Trade PnL
        if self.trades:
            trade_pnls = [t.pnl for t in self.trades]
            colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnls]
            axes[1, 0].bar(range(len(trade_pnls)), trade_pnls, color=colors, alpha=0.6)
            axes[1, 0].set_title('Trade PnL')
            axes[1, 0].set_xlabel('Trade #')
            axes[1, 0].set_ylabel('PnL ($)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative PnL
        if self.trades:
            cum_pnl = np.cumsum([t.pnl for t in self.trades])
            axes[1, 1].plot(cum_pnl, 'b-', linewidth=2)
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].set_title('Cumulative PnL')
            axes[1, 1].set_xlabel('Trade #')
            axes[1, 1].set_ylabel('Cumulative PnL ($)')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Performance metrics text
        metrics = self.get_performance_summary()
        metrics_text = f"""
        Total Return: {metrics['total_return']:.2%}
        Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
        Max Drawdown: {metrics['max_drawdown']:.2%}
        Win Rate: {metrics['win_rate']:.2%}
        Profit Factor: {metrics['profit_factor']:.2f}
        Total Trades: {metrics['total_trades']}
        """
        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=10, 
                       transform=axes[1, 2].transAxes, verticalalignment='center')
        axes[1, 2].axis('off')
        
        plt.suptitle('Trading Performance Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Performance plot saved to {save_path}")
        
        plt.close()
        
        return fig


class ProfitBasedTrainingReward:
    """Convert trading environment metrics to training rewards"""
    
    def __init__(self, target_sharpe: float = 1.5, target_return: float = 0.20):
        self.target_sharpe = target_sharpe
        self.target_return = target_return
        self.baseline_performance = None
        
    def calculate_training_reward(self, env_metrics: Dict[str, float], 
                                 baseline: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """Calculate differentiable reward for training"""
        
        # Extract key metrics
        sharpe = env_metrics.get('sharpe_ratio', 0)
        total_return = env_metrics.get('reward', 0)
        win_rate = env_metrics.get('win_rate', 0.5)
        profit_factor = env_metrics.get('profit_factor', 1.0)
        max_dd = env_metrics.get('max_drawdown', 0)
        
        # Build reward components
        rewards = []
        
        # 1. Sharpe ratio reward (most important for risk-adjusted returns)
        sharpe_reward = torch.tanh(torch.tensor(sharpe / self.target_sharpe))
        rewards.append(sharpe_reward * 0.3)
        
        # 2. Return reward
        return_reward = torch.tanh(torch.tensor(total_return / 0.01))  # 1% return scale
        rewards.append(return_reward * 0.25)
        
        # 3. Win rate reward
        win_reward = torch.sigmoid(torch.tensor((win_rate - 0.5) * 10))
        rewards.append(win_reward * 0.15)
        
        # 4. Profit factor reward
        pf_reward = torch.tanh(torch.tensor((profit_factor - 1.0) * 2))
        rewards.append(pf_reward * 0.15)
        
        # 5. Drawdown penalty
        dd_penalty = -torch.relu(torch.tensor(max_dd - 0.10)) * 5  # Penalty for DD > 10%
        rewards.append(dd_penalty * 0.15)
        
        # Combine rewards
        total_reward = sum(rewards)
        
        # Add baseline comparison if provided
        if baseline and self.baseline_performance:
            improvement = total_reward - self.baseline_performance
            total_reward = total_reward + improvement * 0.1
        
        return total_reward
    
    def update_baseline(self, performance: float):
        """Update baseline performance for relative rewards"""
        if self.baseline_performance is None:
            self.baseline_performance = performance
        else:
            # Exponential moving average
            self.baseline_performance = 0.9 * self.baseline_performance + 0.1 * performance


def create_market_data_generator(n_samples: int = 10000, 
                                volatility: float = 0.02) -> pd.DataFrame:
    """Generate realistic market data for testing"""
    
    # Generate base price series with trends and volatility clusters
    np.random.seed(42)
    
    # Time series
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H')
    
    # Generate returns with volatility clustering (GARCH-like)
    returns = []
    current_vol = volatility
    
    for i in range(n_samples):
        # Volatility clustering
        vol_shock = np.random.normal(0, 0.01)
        current_vol = 0.95 * current_vol + 0.05 * volatility + vol_shock
        current_vol = max(0.001, min(0.05, current_vol))  # Bound volatility
        
        # Add trend component
        trend = 0.0001 * np.sin(i / 100)  # Sinusoidal trend
        
        # Generate return
        ret = np.random.normal(trend, current_vol)
        returns.append(ret)
    
    # Convert to prices
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Add volume (correlated with volatility)
    volume = np.random.lognormal(15, 0.5, n_samples)
    volume = volume * (1 + np.abs(returns) * 10)  # Higher volume on big moves
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices,
        'volume': volume,
        'returns': returns
    })
    
    return data


def main():
    """Test the realistic trading environment"""
    
    # Create environment
    config = TradingConfig(
        initial_capital=100000,
        max_position_size=0.1,
        commission_rate=0.001,
        slippage_factor=0.0005
    )
    
    env = RealisticTradingEnvironment(config)
    reward_calculator = ProfitBasedTrainingReward()
    
    # Generate market data
    market_data = create_market_data_generator(5000)
    
    logger.info("Starting realistic trading simulation...")
    
    # Simulate trading
    for i in range(1000):
        # Get market state
        market_state = {
            'price': market_data.iloc[i]['close'],
            'timestamp': i
        }
        
        # Generate trading signal (random for testing)
        signal = np.random.normal(0, 0.5)
        confidence = np.random.uniform(0.5, 1.0)
        
        action = {
            'signal': torch.tensor(signal),
            'confidence': torch.tensor(confidence)
        }
        
        # Execute step
        metrics = env.step(action, market_state)
        
        # Calculate training reward
        training_reward = reward_calculator.calculate_training_reward(metrics)
        
        # Log progress
        if i % 100 == 0:
            perf = env.get_performance_summary()
            logger.info(f"Step {i}: Capital=${perf['current_capital']:,.2f}, "
                       f"Return={perf['total_return']:.2%}, "
                       f"Sharpe={perf['sharpe_ratio']:.2f}, "
                       f"Trades={perf['total_trades']}")
    
    # Final performance
    final_performance = env.get_performance_summary()
    
    logger.info("\n" + "="*60)
    logger.info("FINAL PERFORMANCE SUMMARY")
    logger.info("="*60)
    for key, value in final_performance.items():
        if isinstance(value, float):
            if 'return' in key or 'rate' in key or 'drawdown' in key:
                logger.info(f"{key}: {value:.2%}")
            else:
                logger.info(f"{key}: {value:.2f}")
        else:
            logger.info(f"{key}: {value}")
    
    # Plot performance
    env.plot_performance('training/realistic_trading_performance.png')
    
    return env, final_performance


if __name__ == "__main__":
    env, performance = main()