from typing import Dict, List
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger


class PortfolioTracker:
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.current_equity = initial_balance
        self.current_cash = initial_balance
        self.positions = {}
        
        # History tracking
        self.equity_history = [initial_balance]
        self.returns_history = []
        self.trades_history = []
        
        # Daily tracking
        self.daily_start_equity = initial_balance
        self.daily_trades = 0
        
        # Performance metrics
        self.peak_equity = initial_balance
        self.max_drawdown = 0.0
        
    def update(self, equity: float, cash: float, positions: Dict[str, dict]):
        """Update portfolio state."""
        
        self.current_equity = equity
        self.current_cash = cash
        self.positions = positions
        
        # Update history
        self.equity_history.append(equity)
        
        # Calculate return
        if len(self.equity_history) > 1:
            prev_equity = self.equity_history[-2]
            daily_return = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            self.returns_history.append(daily_return)
        
        # Update peak and drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        current_drawdown = (equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0
        if current_drawdown < self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def record_trade(self, symbol: str, side: str, qty: int, price: float, timestamp: datetime):
        """Record a trade."""
        
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'price': price,
            'value': qty * price
        }
        
        self.trades_history.append(trade)
        self.daily_trades += 1
        
        logger.info(f"Trade recorded: {side} {qty} {symbol} @ ${price:.2f}")
    
    def new_day(self):
        """Reset daily tracking."""
        
        self.daily_start_equity = self.current_equity
        self.daily_trades = 0
        logger.info(f"New trading day - Starting equity: ${self.daily_start_equity:,.2f}")
    
    def get_metrics(self) -> Dict:
        """Calculate and return portfolio metrics."""
        
        metrics = {
            'equity': self.current_equity,
            'cash': self.current_cash,
            'positions_value': self.current_equity - self.current_cash,
            'num_positions': len(self.positions),
            'total_return': (self.current_equity - self.initial_balance) / self.initial_balance,
            'daily_return': (self.current_equity - self.daily_start_equity) / self.daily_start_equity,
            'max_drawdown': self.max_drawdown,
            'peak_equity': self.peak_equity,
            'daily_trades': self.daily_trades,
            'total_trades': len(self.trades_history)
        }
        
        # Calculate Sharpe ratio if we have enough data
        if len(self.returns_history) > 20:
            returns_array = np.array(self.returns_history)
            metrics['sharpe_ratio'] = np.mean(returns_array) / (np.std(returns_array) + 1e-8) * np.sqrt(252)
            metrics['volatility'] = np.std(returns_array) * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0
            metrics['volatility'] = 0
        
        # Calculate win rate
        if self.trades_history:
            winning_trades = 0
            for i, trade in enumerate(self.trades_history):
                # Simple approximation: check if next equity increased
                if i < len(self.equity_history) - 1:
                    if self.equity_history[i+1] > self.equity_history[i]:
                        winning_trades += 1
            metrics['win_rate'] = winning_trades / len(self.trades_history)
        else:
            metrics['win_rate'] = 0
        
        return metrics
    
    def get_position_metrics(self, symbol: str) -> Dict:
        """Get metrics for a specific position."""
        
        if symbol not in self.positions:
            return {}
        
        pos = self.positions[symbol]
        
        # Calculate P&L
        if pos.get('current_price') and pos.get('entry_price'):
            price_change = pos['current_price'] - pos['entry_price']
            pnl_pct = price_change / pos['entry_price']
            
            if pos['side'] == 'short':
                pnl_pct = -pnl_pct
            
            pnl_value = pnl_pct * pos['qty'] * pos['entry_price']
        else:
            pnl_pct = 0
            pnl_value = 0
        
        return {
            'symbol': symbol,
            'side': pos.get('side'),
            'qty': pos.get('qty'),
            'entry_price': pos.get('entry_price'),
            'current_price': pos.get('current_price'),
            'pnl_pct': pnl_pct,
            'pnl_value': pnl_value,
            'market_value': pos.get('market_value', 0)
        }
    
    def export_history(self, filepath: str):
        """Export trading history to CSV."""
        
        # Create DataFrame from trades
        if self.trades_history:
            df = pd.DataFrame(self.trades_history)
            df.to_csv(filepath, index=False)
            logger.info(f"Trading history exported to {filepath}")
        else:
            logger.warning("No trades to export")
    
    def get_summary(self) -> str:
        """Get a text summary of portfolio performance."""
        
        metrics = self.get_metrics()
        
        summary = f"""
Portfolio Performance Summary
==============================
Current Equity: ${metrics['equity']:,.2f}
Cash Available: ${metrics['cash']:,.2f}
Positions Value: ${metrics['positions_value']:,.2f}
Number of Positions: {metrics['num_positions']}

Returns
-------
Total Return: {metrics['total_return']:.2%}
Daily Return: {metrics['daily_return']:.2%}
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Volatility: {metrics['volatility']:.2%}

Risk Metrics
------------
Max Drawdown: {metrics['max_drawdown']:.2%}
Peak Equity: ${metrics['peak_equity']:,.2f}

Trading Activity
----------------
Daily Trades: {metrics['daily_trades']}
Total Trades: {metrics['total_trades']}
Win Rate: {metrics['win_rate']:.2%}
"""
        
        return summary