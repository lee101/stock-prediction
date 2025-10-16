#!/usr/bin/env python3
"""
Integrated Profitable Trading System with Smart Risk Management
Combines differentiable training with unprofitable shutdown logic
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
from dataclasses import dataclass
import matplotlib.pyplot as plt
import sys
sys.path.append('/media/lee/crucial2/code/stock/training')

from smart_risk_manager import SmartRiskManager, RiskAwareTradingSystem, TradeDirection
from differentiable_trainer import DifferentiableTradingModel, TrainingConfig
from realistic_trading_env import RealisticTradingEnvironment, TradingConfig, create_market_data_generator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntegratedProfitableSystem:
    """Complete trading system with neural model and smart risk management"""
    
    def __init__(self, model: nn.Module, initial_capital: float = 100000):
        self.model = model
        self.risk_manager = SmartRiskManager(initial_capital)
        self.trading_system = RiskAwareTradingSystem(self.risk_manager)
        
        # Track multiple symbols
        self.symbol_history = {}
        self.active_trades = {}
        
        # Performance tracking
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_pnl = 0.0
        
        logger.info(f"Integrated system initialized with ${initial_capital:,.2f}")
    
    def process_market_data(self, symbol: str, market_data: pd.DataFrame, 
                           start_idx: int = 100, end_idx: int = None):
        """Process market data for a symbol with risk management"""
        
        if end_idx is None:
            end_idx = min(len(market_data) - 1, start_idx + 500)
        
        # Prepare features
        seq_len = 20
        
        # Add technical indicators
        market_data['sma_5'] = market_data['close'].rolling(5).mean()
        market_data['sma_20'] = market_data['close'].rolling(20).mean()
        market_data['rsi'] = self.calculate_rsi(market_data['close'])
        market_data['volatility'] = market_data['returns'].rolling(20).std()
        market_data = market_data.fillna(method='bfill').fillna(method='ffill')
        
        logger.info(f"Processing {symbol} from index {start_idx} to {end_idx}")
        
        for i in range(start_idx, end_idx):
            if i < seq_len:
                continue
            
            # Prepare input sequence
            seq_data = market_data.iloc[i-seq_len:i]
            features = ['close', 'volume', 'sma_5', 'sma_20', 'rsi', 'volatility']
            
            # Normalize features
            X = seq_data[features].values
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            X_tensor = torch.FloatTensor(X).unsqueeze(0)
            
            # Get model prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
            
            # Parse outputs
            action_probs = F.softmax(outputs['actions'], dim=-1).squeeze()
            position_size = outputs['position_sizes'].squeeze().item()
            confidence = outputs['confidences'].squeeze().item()
            
            # Generate trading signal
            if action_probs[0] > 0.5:  # Buy signal
                signal = abs(position_size) * confidence
            elif action_probs[2] > 0.5:  # Sell signal
                signal = -abs(position_size) * confidence
            else:
                signal = 0.0
            
            current_price = market_data.iloc[i]['close']
            
            # Check if we have an active position to close
            if symbol in self.active_trades:
                active_trade = self.active_trades[symbol]
                
                # Simple exit logic (can be enhanced)
                holding_time = i - active_trade['entry_idx']
                price_change = (current_price - active_trade['entry_price']) / active_trade['entry_price']
                
                should_exit = False
                exit_reason = ""
                
                # Exit conditions
                if holding_time > 20:  # Time limit
                    should_exit = True
                    exit_reason = "time_limit"
                elif active_trade['direction'] == TradeDirection.LONG:
                    if price_change > 0.03:  # Take profit
                        should_exit = True
                        exit_reason = "take_profit"
                    elif price_change < -0.02:  # Stop loss
                        should_exit = True
                        exit_reason = "stop_loss"
                else:  # Short position
                    if price_change < -0.03:  # Take profit (price went down)
                        should_exit = True
                        exit_reason = "take_profit"
                    elif price_change > 0.02:  # Stop loss
                        should_exit = True
                        exit_reason = "stop_loss"
                
                # Exit if signal reversed
                if (active_trade['direction'] == TradeDirection.LONG and signal < -0.3) or \
                   (active_trade['direction'] == TradeDirection.SHORT and signal > 0.3):
                    should_exit = True
                    exit_reason = "signal_reversal"
                
                if should_exit:
                    # Close position
                    pnl = self.trading_system.close_position(
                        active_trade['trade_info'], 
                        current_price, 
                        exit_reason
                    )
                    
                    if pnl is not None:
                        self.total_pnl += pnl
                        if pnl > 0:
                            self.profitable_trades += 1
                    
                    del self.active_trades[symbol]
            
            # Enter new position if no active trade
            if symbol not in self.active_trades and abs(signal) > 0.3:
                trade = self.trading_system.execute_trade_decision(
                    symbol, signal, current_price
                )
                
                if trade['executed']:
                    self.active_trades[symbol] = {
                        'trade_info': trade,
                        'entry_idx': i,
                        'entry_price': current_price,
                        'direction': TradeDirection.LONG if signal > 0 else TradeDirection.SHORT
                    }
                    self.total_trades += 1
            
            # Log progress periodically
            if i % 50 == 0:
                self.log_performance()
        
        # Close any remaining positions
        for symbol, trade_data in list(self.active_trades.items()):
            final_price = market_data.iloc[-1]['close']
            pnl = self.trading_system.close_position(
                trade_data['trade_info'],
                final_price,
                "end_of_data"
            )
            if pnl is not None:
                self.total_pnl += pnl
                if pnl > 0:
                    self.profitable_trades += 1
        
        self.active_trades.clear()
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def log_performance(self):
        """Log current performance metrics"""
        risk_report = self.risk_manager.get_risk_report()
        
        win_rate = self.profitable_trades / max(self.total_trades, 1)
        
        logger.info(f"Performance: Capital=${risk_report['current_capital']:,.2f}, "
                   f"PnL=${self.total_pnl:.2f}, "
                   f"Trades={self.total_trades}, "
                   f"WinRate={win_rate:.1%}, "
                   f"Shutdowns={risk_report['active_shutdowns']}")
    
    def get_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        
        risk_report = self.risk_manager.get_risk_report()
        
        return {
            'final_capital': risk_report['current_capital'],
            'total_return': risk_report['total_return'],
            'total_trades': self.total_trades,
            'win_rate': self.profitable_trades / max(self.total_trades, 1),
            'total_pnl': self.total_pnl,
            'risk_report': risk_report,
            'symbol_performance': risk_report['symbol_performance']
        }


def test_integrated_system():
    """Test the integrated profitable system with risk management"""
    
    logger.info("="*60)
    logger.info("TESTING INTEGRATED PROFITABLE SYSTEM")
    logger.info("="*60)
    
    # Create model
    model = DifferentiableTradingModel(
        input_dim=6,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.1
    )
    
    # Initialize system
    system = IntegratedProfitableSystem(model, initial_capital=100000)
    
    # Test with multiple symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    for symbol in symbols:
        logger.info(f"\n--- Processing {symbol} ---")
        
        # Generate synthetic market data
        market_data = create_market_data_generator(
            n_samples=1000, 
            volatility=0.015 if symbol == 'AAPL' else 0.02
        )
        
        # Process the symbol
        system.process_market_data(symbol, market_data, start_idx=100, end_idx=400)
    
    # Get final report
    final_report = system.get_final_report()
    
    logger.info("\n" + "="*60)
    logger.info("FINAL INTEGRATED SYSTEM REPORT")
    logger.info("="*60)
    logger.info(f"Final Capital: ${final_report['final_capital']:,.2f}")
    logger.info(f"Total Return: {final_report['total_return']:.2%}")
    logger.info(f"Total Trades: {final_report['total_trades']}")
    logger.info(f"Win Rate: {final_report['win_rate']:.1%}")
    logger.info(f"Total PnL: ${final_report['total_pnl']:.2f}")
    
    logger.info("\nPer Symbol/Direction Performance:")
    for key, perf in final_report['symbol_performance'].items():
        logger.info(f"  {key}:")
        logger.info(f"    Total PnL: ${perf['total_pnl']:.2f}")
        logger.info(f"    Win Rate: {perf['win_rate']:.1%}")
        logger.info(f"    Sharpe: {perf['sharpe_ratio']:.2f}")
        logger.info(f"    Shutdown: {perf['is_shutdown']}")
        if perf['consecutive_losses'] > 0:
            logger.info(f"    Consecutive Losses: {perf['consecutive_losses']}")
    
    # Check if profitable
    is_profitable = final_report['total_return'] > 0
    
    if is_profitable:
        logger.info("\n✅ SYSTEM IS PROFITABLE WITH RISK MANAGEMENT!")
    else:
        logger.info("\n📊 System needs more training to be profitable")
    
    return system, final_report


def train_until_profitable_with_risk():
    """Train the system until it's profitable with risk management"""
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING WITH RISK MANAGEMENT FEEDBACK")
    logger.info("="*60)
    
    # Create model
    model = DifferentiableTradingModel(
        input_dim=6,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.1
    )
    
    # Training configuration
    config = TrainingConfig(
        learning_rate=1e-3,
        batch_size=32,
        num_epochs=20,
        gradient_clip_norm=1.0,
        weight_decay=1e-4
    )
    
    # Generate training data
    train_data = create_market_data_generator(n_samples=5000, volatility=0.018)
    
    best_return = -float('inf')
    
    for epoch in range(10):
        logger.info(f"\n--- Training Epoch {epoch+1} ---")
        
        # Create new system for testing
        system = IntegratedProfitableSystem(model, initial_capital=100000)
        
        # Test on validation data
        val_data = create_market_data_generator(n_samples=1000, volatility=0.02)
        system.process_market_data('TEST', val_data, start_idx=100, end_idx=500)
        
        # Get performance
        report = system.get_final_report()
        current_return = report['total_return']
        
        logger.info(f"Epoch {epoch+1}: Return={current_return:.2%}, "
                   f"WinRate={report['win_rate']:.1%}")
        
        # Check if improved
        if current_return > best_return:
            best_return = current_return
            torch.save(model.state_dict(), 'training/best_risk_aware_model.pt')
            logger.info(f"💾 Saved new best model with return: {best_return:.2%}")
        
        # Check if profitable enough
        if current_return > 0.05 and report['win_rate'] > 0.55:
            logger.info(f"\n🎯 ACHIEVED PROFITABILITY: {current_return:.2%} return, "
                       f"{report['win_rate']:.1%} win rate")
            break
        
        # Continue training if not profitable
        # (Simplified training loop - in production, use proper DataLoader)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
        for _ in range(50):  # Quick training iterations
            # Generate batch
            batch_size = 32
            seq_len = 20
            
            # Random sampling from data
            idx = np.random.randint(seq_len, len(train_data) - 1)
            seq_data = train_data.iloc[idx-seq_len:idx]
            
            # Prepare features (simplified)
            train_data['sma_5'] = train_data['close'].rolling(5).mean()
            train_data['sma_20'] = train_data['close'].rolling(20).mean()
            X = train_data[['close', 'volume']].iloc[idx-seq_len:idx].values
            X = (X - X.mean()) / (X.std() + 1e-8)
            X = torch.FloatTensor(X).unsqueeze(0)
            
            # Forward pass
            outputs = model(X)
            
            # Simple loss (can be enhanced)
            loss = -outputs['confidences'].mean()  # Maximize confidence
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    
    return model


if __name__ == "__main__":
    # Test integrated system
    system, report = test_integrated_system()
    
    # Train with risk management feedback
    if report['total_return'] < 0.05:
        logger.info("\n🔄 Starting enhanced training with risk feedback...")
        model = train_until_profitable_with_risk()
        
        # Test again with trained model
        logger.info("\n📊 Testing trained model...")
        system2 = IntegratedProfitableSystem(model, initial_capital=100000)
        
        # Test on new data
        test_data = create_market_data_generator(n_samples=1500, volatility=0.018)
        system2.process_market_data('FINAL_TEST', test_data, start_idx=100, end_idx=600)
        
        final_report = system2.get_final_report()
        logger.info(f"\n🏁 Final Result: Return={final_report['total_return']:.2%}, "
                   f"WinRate={final_report['win_rate']:.1%}")