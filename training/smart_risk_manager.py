#!/usr/bin/env python3
"""
Smart Risk Management System with Unprofitable Shutdown
- Tracks performance per symbol/direction
- Implements cooldown after losses
- Uses small test trades to validate recovery
- Gradual position sizing based on confidence
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from enum import Enum
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class SymbolPerformance:
    """Track performance for a specific symbol/direction pair"""
    symbol: str
    direction: TradeDirection
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    total_pnl: float = 0.0
    last_trade_pnl: float = 0.0
    last_trade_time: Optional[datetime] = None
    is_shutdown: bool = False
    test_trade_count: int = 0
    recovery_confidence: float = 0.0
    historical_pnl: deque = field(default_factory=lambda: deque(maxlen=20))
    win_rate: float = 0.5
    avg_win: float = 0.0
    avg_loss: float = 0.0
    sharpe_ratio: float = 0.0


@dataclass
class RiskProfile:
    """Risk parameters that adapt based on performance"""
    max_position_size: float = 0.1  # Max 10% of capital
    current_position_size: float = 0.02  # Start conservative at 2%
    test_position_size: float = 0.001  # 0.1% for test trades
    max_consecutive_losses: int = 3  # Shutdown after 3 consecutive losses
    min_recovery_trades: int = 2  # Minimum successful test trades before full recovery
    cooldown_periods: int = 10  # Periods to wait after shutdown
    confidence_threshold: float = 0.6  # Minimum confidence to exit shutdown
    position_scaling_factor: float = 1.5  # Scale position size by this factor
    max_daily_loss: float = 0.05  # Max 5% daily loss
    max_correlation_exposure: float = 0.3  # Max 30% in correlated trades


class SmartRiskManager:
    """Intelligent risk management with pair-specific shutdown logic"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_profile = RiskProfile()
        
        # Track performance per symbol/direction
        self.symbol_performance: Dict[Tuple[str, TradeDirection], SymbolPerformance] = {}
        
        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_day = datetime.now().date()
        
        # Global risk metrics
        self.total_exposure = 0.0
        self.correlation_matrix = {}
        self.active_positions = {}
        
        # Learning parameters
        self.risk_adjustment_rate = 0.1
        self.confidence_decay = 0.95
        
        logger.info(f"SmartRiskManager initialized with ${initial_capital:,.2f}")
    
    def get_symbol_performance(self, symbol: str, direction: TradeDirection) -> SymbolPerformance:
        """Get or create performance tracker for symbol/direction"""
        key = (symbol, direction)
        if key not in self.symbol_performance:
            self.symbol_performance[key] = SymbolPerformance(symbol, direction)
        return self.symbol_performance[key]
    
    def should_trade(self, symbol: str, direction: TradeDirection, 
                    signal_strength: float) -> Tuple[bool, float, str]:
        """
        Determine if we should trade and what position size
        Returns: (should_trade, position_size, reason)
        """
        
        # Check daily loss limit
        if self.daily_pnl < -self.risk_profile.max_daily_loss * self.current_capital:
            return False, 0.0, "Daily loss limit reached"
        
        # Get symbol performance
        perf = self.get_symbol_performance(symbol, direction)
        
        # Check if in shutdown mode
        if perf.is_shutdown:
            # Only allow test trades during shutdown
            if perf.test_trade_count < self.risk_profile.min_recovery_trades:
                # Place test trade
                return True, self.risk_profile.test_position_size, "Test trade during shutdown"
            
            # Check if ready to exit shutdown
            if perf.recovery_confidence >= self.risk_profile.confidence_threshold:
                perf.is_shutdown = False
                perf.test_trade_count = 0
                logger.info(f"Exiting shutdown for {symbol} {direction.value}")
            else:
                return False, 0.0, f"Still in shutdown (confidence: {perf.recovery_confidence:.2f})"
        
        # Check consecutive losses
        if perf.consecutive_losses >= self.risk_profile.max_consecutive_losses:
            self.enter_shutdown(symbol, direction)
            return True, self.risk_profile.test_position_size, "Entering shutdown with test trade"
        
        # Calculate position size based on performance
        position_size = self.calculate_position_size(perf, signal_strength)
        
        # Check correlation exposure
        if not self.check_correlation_limits(symbol, position_size):
            return False, 0.0, "Correlation exposure limit reached"
        
        return True, position_size, "Normal trade"
    
    def calculate_position_size(self, perf: SymbolPerformance, 
                               signal_strength: float) -> float:
        """Calculate dynamic position size based on performance and confidence"""
        
        base_size = self.risk_profile.current_position_size
        
        # Adjust based on recent performance
        if perf.consecutive_wins > 0:
            # Scale up with wins (Kelly Criterion inspired)
            win_factor = min(1 + (perf.consecutive_wins * 0.2), 2.0)
            base_size *= win_factor
        elif perf.consecutive_losses > 0:
            # Scale down with losses
            loss_factor = max(0.5 ** perf.consecutive_losses, 0.25)
            base_size *= loss_factor
        
        # Adjust based on win rate
        if perf.win_rate > 0.6:
            base_size *= 1.2
        elif perf.win_rate < 0.4:
            base_size *= 0.8
        
        # Adjust based on Sharpe ratio
        if perf.sharpe_ratio > 1.5:
            base_size *= 1.3
        elif perf.sharpe_ratio < 0.5:
            base_size *= 0.7
        
        # Apply signal strength
        base_size *= abs(signal_strength)
        
        # Cap at maximum
        final_size = min(base_size, self.risk_profile.max_position_size)
        
        # Ensure minimum viable size
        min_size = self.risk_profile.test_position_size * 10
        if final_size < min_size:
            final_size = 0.0  # Don't trade if size too small
        
        return final_size
    
    def enter_shutdown(self, symbol: str, direction: TradeDirection):
        """Enter shutdown mode for a symbol/direction pair"""
        perf = self.get_symbol_performance(symbol, direction)
        perf.is_shutdown = True
        perf.test_trade_count = 0
        perf.recovery_confidence = 0.0
        
        logger.warning(f"🚫 Entering shutdown for {symbol} {direction.value} "
                      f"after {perf.consecutive_losses} consecutive losses")
    
    def update_trade_result(self, symbol: str, direction: TradeDirection, 
                          pnl: float, entry_price: float, exit_price: float):
        """Update performance tracking after a trade completes"""
        
        perf = self.get_symbol_performance(symbol, direction)
        
        # Update P&L tracking
        perf.last_trade_pnl = pnl
        perf.total_pnl += pnl
        perf.historical_pnl.append(pnl)
        self.daily_pnl += pnl
        
        # Update win/loss streaks
        if pnl > 0:
            perf.consecutive_wins += 1
            perf.consecutive_losses = 0
            
            # Update recovery confidence if in shutdown
            if perf.is_shutdown:
                perf.recovery_confidence = min(1.0, perf.recovery_confidence + 0.3)
                if perf.test_trade_count < self.risk_profile.min_recovery_trades:
                    perf.test_trade_count += 1
                    logger.info(f"✅ Test trade {perf.test_trade_count}/{self.risk_profile.min_recovery_trades} "
                               f"successful for {symbol} {direction.value}")
        else:
            perf.consecutive_losses += 1
            perf.consecutive_wins = 0
            
            # Decay recovery confidence
            if perf.is_shutdown:
                perf.recovery_confidence *= 0.5
                perf.test_trade_count = 0  # Reset test trades on loss
        
        # Update statistics
        self.update_statistics(perf)
        
        # Update capital
        self.current_capital += pnl
        
        # Log performance
        return_pct = pnl / (entry_price * 100) * 100  # Rough estimate
        logger.info(f"Trade {symbol} {direction.value}: PnL=${pnl:.2f} ({return_pct:.2f}%), "
                   f"Streak: W{perf.consecutive_wins}/L{perf.consecutive_losses}")
    
    def update_statistics(self, perf: SymbolPerformance):
        """Update performance statistics for a symbol/direction"""
        
        if len(perf.historical_pnl) > 0:
            # Calculate win rate
            wins = sum(1 for pnl in perf.historical_pnl if pnl > 0)
            perf.win_rate = wins / len(perf.historical_pnl)
            
            # Calculate average win/loss
            winning_trades = [pnl for pnl in perf.historical_pnl if pnl > 0]
            losing_trades = [pnl for pnl in perf.historical_pnl if pnl < 0]
            
            perf.avg_win = np.mean(winning_trades) if winning_trades else 0
            perf.avg_loss = np.mean(losing_trades) if losing_trades else 0
            
            # Calculate Sharpe ratio (simplified)
            if len(perf.historical_pnl) > 1:
                returns = np.array(list(perf.historical_pnl))
                if np.std(returns) > 0:
                    perf.sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    
    def check_correlation_limits(self, symbol: str, position_size: float) -> bool:
        """Check if adding this position would breach correlation limits"""
        
        # Simplified correlation check
        # In production, use actual correlation matrix
        correlated_exposure = 0.0
        
        for active_symbol, active_size in self.active_positions.items():
            if active_symbol != symbol:
                # Assume some correlation between symbols
                correlation = self.get_correlation(symbol, active_symbol)
                correlated_exposure += abs(active_size * correlation)
        
        total_exposure = correlated_exposure + position_size
        
        return total_exposure <= self.risk_profile.max_correlation_exposure
    
    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols (simplified)"""
        # In production, calculate from historical data
        # For now, use simple heuristics
        
        if symbol1 == symbol2:
            return 1.0
        
        # Tech stocks correlation
        tech_stocks = ['AAPL', 'GOOGL', 'MSFT', 'META', 'NVDA']
        if symbol1 in tech_stocks and symbol2 in tech_stocks:
            return 0.7
        
        # Default low correlation
        return 0.3
    
    def adjust_risk_profile(self):
        """Dynamically adjust risk profile based on performance"""
        
        # Calculate overall performance metrics
        total_pnl = sum(perf.total_pnl for perf in self.symbol_performance.values())
        total_return = total_pnl / self.initial_capital
        
        # Adjust position sizing based on performance
        if total_return > 0.1:  # 10% profit
            self.risk_profile.current_position_size = min(
                self.risk_profile.current_position_size * 1.1,
                self.risk_profile.max_position_size
            )
        elif total_return < -0.05:  # 5% loss
            self.risk_profile.current_position_size = max(
                self.risk_profile.current_position_size * 0.9,
                self.risk_profile.test_position_size * 10
            )
        
        # Adjust max consecutive losses based on market conditions
        avg_volatility = self.estimate_market_volatility()
        if avg_volatility > 0.02:  # High volatility
            self.risk_profile.max_consecutive_losses = 2
        else:
            self.risk_profile.max_consecutive_losses = 3
    
    def estimate_market_volatility(self) -> float:
        """Estimate current market volatility"""
        # Simplified - in production, use VIX or calculate from returns
        recent_pnls = []
        for perf in self.symbol_performance.values():
            recent_pnls.extend(list(perf.historical_pnl)[-5:])
        
        if len(recent_pnls) > 1:
            return np.std(recent_pnls) / (self.current_capital * 0.01)
        return 0.01  # Default volatility
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        
        active_shutdowns = sum(1 for perf in self.symbol_performance.values() if perf.is_shutdown)
        
        report = {
            'current_capital': self.current_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'daily_pnl': self.daily_pnl,
            'active_shutdowns': active_shutdowns,
            'risk_profile': {
                'current_position_size': self.risk_profile.current_position_size,
                'max_position_size': self.risk_profile.max_position_size,
                'max_consecutive_losses': self.risk_profile.max_consecutive_losses
            },
            'symbol_performance': {}
        }
        
        # Add per-symbol performance
        for key, perf in self.symbol_performance.items():
            symbol, direction = key
            report['symbol_performance'][f"{symbol}_{direction.value}"] = {
                'total_pnl': perf.total_pnl,
                'win_rate': perf.win_rate,
                'consecutive_losses': perf.consecutive_losses,
                'is_shutdown': perf.is_shutdown,
                'recovery_confidence': perf.recovery_confidence if perf.is_shutdown else None,
                'sharpe_ratio': perf.sharpe_ratio
            }
        
        return report
    
    def reset_daily_limits(self):
        """Reset daily tracking (call at start of trading day)"""
        current_date = datetime.now().date()
        if current_date != self.current_day:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.current_day = current_date
            logger.info(f"Daily limits reset for {current_date}")


class RiskAwareTradingSystem:
    """Trading system that integrates smart risk management"""
    
    def __init__(self, risk_manager: SmartRiskManager):
        self.risk_manager = risk_manager
        self.trade_history = []
        
    def execute_trade_decision(self, symbol: str, signal: float, 
                              current_price: float) -> Dict[str, Any]:
        """Execute trade with risk management"""
        
        # Determine direction
        direction = TradeDirection.LONG if signal > 0 else TradeDirection.SHORT
        
        # Check with risk manager
        should_trade, position_size, reason = self.risk_manager.should_trade(
            symbol, direction, abs(signal)
        )
        
        if not should_trade:
            return {
                'executed': False,
                'reason': reason,
                'symbol': symbol,
                'direction': direction.value
            }
        
        # Calculate position value
        position_value = self.risk_manager.current_capital * position_size
        shares = position_value / current_price
        
        # Record trade
        trade = {
            'executed': True,
            'symbol': symbol,
            'direction': direction.value,
            'position_size': position_size,
            'shares': shares,
            'entry_price': current_price,
            'reason': reason,
            'timestamp': datetime.now()
        }
        
        self.trade_history.append(trade)
        
        # Log trade
        if "test" in reason.lower():
            logger.info(f"🧪 TEST TRADE: {symbol} {direction.value} "
                       f"${position_value:.2f} @ ${current_price:.2f}")
        else:
            logger.info(f"📈 TRADE: {symbol} {direction.value} "
                       f"${position_value:.2f} @ ${current_price:.2f} "
                       f"(size: {position_size:.1%})")
        
        return trade
    
    def close_position(self, trade: Dict[str, Any], exit_price: float, 
                      exit_reason: str = "signal"):
        """Close a position and update risk manager"""
        
        if not trade['executed']:
            return
        
        # Calculate P&L
        entry_value = trade['shares'] * trade['entry_price']
        exit_value = trade['shares'] * exit_price
        
        if trade['direction'] == TradeDirection.LONG.value:
            pnl = exit_value - entry_value
        else:
            pnl = entry_value - exit_value
        
        # Subtract commission (simplified)
        commission = (entry_value + exit_value) * 0.001
        pnl -= commission
        
        # Update risk manager
        direction = TradeDirection.LONG if trade['direction'] == 'long' else TradeDirection.SHORT
        self.risk_manager.update_trade_result(
            trade['symbol'], direction, pnl, 
            trade['entry_price'], exit_price
        )
        
        # Log result
        if entry_value > 0:
            return_pct = (pnl / entry_value) * 100
        else:
            return_pct = 0.0
        if pnl > 0:
            logger.info(f"✅ CLOSED: {trade['symbol']} {trade['direction']} "
                       f"PnL: ${pnl:.2f} ({return_pct:.2f}%) - {exit_reason}")
        else:
            logger.info(f"❌ CLOSED: {trade['symbol']} {trade['direction']} "
                       f"PnL: ${pnl:.2f} ({return_pct:.2f}%) - {exit_reason}")
        
        return pnl


def test_risk_management():
    """Test the smart risk management system"""
    
    logger.info("="*60)
    logger.info("TESTING SMART RISK MANAGEMENT SYSTEM")
    logger.info("="*60)
    
    # Initialize
    risk_manager = SmartRiskManager(initial_capital=100000)
    trading_system = RiskAwareTradingSystem(risk_manager)
    
    # Simulate trades
    test_scenarios = [
        # Symbol, Signal, Entry Price, Exit Price, Description
        ("AAPL", 0.8, 150, 152, "Win - AAPL Long"),
        ("AAPL", 0.7, 152, 151, "Loss - AAPL Long"),
        ("AAPL", 0.9, 151, 149, "Loss - AAPL Long"),
        ("AAPL", 0.6, 149, 147, "Loss - AAPL Long - Should trigger shutdown"),
        ("AAPL", 0.8, 147, 148, "Test trade during shutdown"),
        ("AAPL", 0.7, 148, 150, "Test trade 2"),
        ("AAPL", 0.8, 150, 153, "Should exit shutdown if profitable"),
        
        ("GOOGL", -0.7, 2800, 2780, "Win - GOOGL Short"),
        ("GOOGL", -0.6, 2780, 2790, "Loss - GOOGL Short"),
        ("GOOGL", 0.8, 2790, 2810, "Win - GOOGL Long (different direction)"),
    ]
    
    for symbol, signal, entry_price, exit_price, description in test_scenarios:
        logger.info(f"\n--- {description} ---")
        
        # Execute trade
        trade = trading_system.execute_trade_decision(symbol, signal, entry_price)
        
        if trade['executed']:
            # Simulate position close
            trading_system.close_position(trade, exit_price, "test")
        
        # Show risk report periodically
        if len(trading_system.trade_history) % 5 == 0:
            report = risk_manager.get_risk_report()
            logger.info(f"\nRisk Report: Active Shutdowns: {report['active_shutdowns']}, "
                       f"Capital: ${report['current_capital']:,.2f}")
    
    # Final report
    final_report = risk_manager.get_risk_report()
    
    logger.info("\n" + "="*60)
    logger.info("FINAL RISK MANAGEMENT REPORT")
    logger.info("="*60)
    logger.info(f"Final Capital: ${final_report['current_capital']:,.2f}")
    logger.info(f"Total Return: {final_report['total_return']:.2%}")
    logger.info(f"Active Shutdowns: {final_report['active_shutdowns']}")
    
    logger.info("\nPer Symbol/Direction Performance:")
    for key, perf in final_report['symbol_performance'].items():
        logger.info(f"  {key}:")
        logger.info(f"    PnL: ${perf['total_pnl']:.2f}")
        logger.info(f"    Win Rate: {perf['win_rate']:.1%}")
        logger.info(f"    Shutdown: {perf['is_shutdown']}")
        if perf['recovery_confidence'] is not None:
            logger.info(f"    Recovery Confidence: {perf['recovery_confidence']:.2f}")
    
    return risk_manager


if __name__ == "__main__":
    risk_manager = test_risk_management()