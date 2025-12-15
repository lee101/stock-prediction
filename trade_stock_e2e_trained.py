#!/usr/bin/env python3
"""
End-to-End Stock Trading System Using Trained RL Models

This script integrates the trained RL models with real trading execution,
including stock selection, position sizing, and portfolio management.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger

# Add paths for module imports
sys.path.extend(['.', './training', './src', './rlinference'])

# Core imports
from src.sizing_utils import get_qty, get_current_symbol_exposure
from src.fixtures import crypto_symbols
from src.logging_utils import setup_logging
from src.date_utils import is_nyse_open_on_date
import alpaca_wrapper

# RL inference imports
from rlinference.utils.model_manager import ModelManager
from rlinference.utils.data_preprocessing import DataPreprocessor
from rlinference.utils.risk_manager import RiskManager
from rlinference.utils.portfolio_tracker import PortfolioTracker
from rlinference.strategies.rl_strategy import RLTradingStrategy
from rlinference.brokers.alpaca_broker import AlpacaBroker

# Training imports for model loading
from training.trading_config import get_trading_costs
from training.best_checkpoints import load_best_model_info


class TradeStockE2ETrained:
    """
    End-to-end trained RL trading system that makes actual buy/sell decisions.
    """
    
    def __init__(self, config_path: Optional[str] = None, paper_trading: bool = True):
        self.logger = setup_logging("trade_e2e_trained.log")
        self.paper_trading = paper_trading
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.model_manager = ModelManager(models_dir=Path("training/models"))
        self.data_preprocessor = DataPreprocessor()
        self.risk_manager = RiskManager(self.config)
        self.portfolio_tracker = PortfolioTracker(self.config.get('initial_balance', 100000))
        
        # Initialize RL strategy
        self.strategy = RLTradingStrategy(self.config, self.model_manager, self.data_preprocessor)
        
        # Load best models
        self._load_best_models()
        
        # Portfolio constraints
        self.max_positions = self.config.get('max_positions', 2)  # Start with 2 as mentioned
        self.max_exposure_per_symbol = self.config.get('max_exposure_per_symbol', 0.6)  # 60%
        self.min_confidence_threshold = self.config.get('min_confidence', 0.4)
        
        # Trading costs
        self.trading_costs = get_trading_costs('stock', 'alpaca')
        
        self.logger.info(f"TradeStockE2ETrained initialized - Paper Trading: {paper_trading}")
        self.logger.info(f"Max positions: {self.max_positions}, Max exposure per symbol: {self.max_exposure_per_symbol:.0%}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load trading configuration."""
        default_config = {
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD', 'AMZN', 'META'],
            'initial_balance': 100000,
            'max_positions': 2,
            'max_exposure_per_symbol': 0.6,
            'min_confidence': 0.4,
            'rebalance_frequency_minutes': 30,
            'risk_management': {
                'max_daily_loss': 0.05,  # 5%
                'max_drawdown': 0.15,    # 15%
                'position_timeout_hours': 24
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _load_best_models(self):
        """Load the best performing models from training."""
        try:
            # Load best checkpoints info
            best_checkpoints_path = Path("training/best_checkpoints.json")
            if best_checkpoints_path.exists():
                with open(best_checkpoints_path) as f:
                    best_models = json.load(f)
                
                self.logger.info(f"Loaded best model info: {best_models}")
                
                # Use the best overall model for trading
                best_model_name = best_models.get('best_sharpe', 'best_advanced_model.pth')
                self.primary_model = best_model_name
                
                # Load model into model manager
                model_path = Path("training/models") / best_model_name
                if model_path.exists():
                    self.logger.info(f"Using primary model: {best_model_name}")
                else:
                    self.logger.warning(f"Best model {best_model_name} not found, using default")
                    self.primary_model = "best_advanced_model.pth"
            else:
                self.logger.warning("No best_checkpoints.json found, using default model")
                self.primary_model = "best_advanced_model.pth"
                
        except Exception as e:
            self.logger.error(f"Error loading best models: {e}")
            self.primary_model = "best_advanced_model.pth"
    
    def get_stock_universe(self) -> List[str]:
        """Get the universe of stocks to consider for trading."""
        # Start with configured symbols
        symbols = self.config['symbols'].copy()
        
        # Can add logic here to dynamically expand/filter universe
        # based on market conditions, liquidity, etc.
        
        # Filter out crypto for this stock-focused system
        symbols = [s for s in symbols if s not in crypto_symbols]
        
        self.logger.info(f"Trading universe: {symbols}")
        return symbols
    
    def analyze_market_opportunity(self, symbol: str) -> Optional[Dict]:
        """Analyze a single symbol for trading opportunities."""
        try:
            # Get current position info
            positions = alpaca_wrapper.get_all_positions()
            current_position = None
            
            for pos in positions:
                if pos.symbol == symbol:
                    current_position = {
                        'symbol': symbol,
                        'qty': float(pos.qty),
                        'side': pos.side,
                        'entry_price': float(pos.avg_entry_price),
                        'market_value': float(pos.market_value) if pos.market_value else 0,
                        'unrealized_pl': float(pos.unrealized_pl) if pos.unrealized_pl else 0
                    }
                    break
            
            # Get market data
            market_data = self.data_preprocessor.fetch_realtime_data(symbol)
            if market_data.empty:
                self.logger.warning(f"No market data for {symbol}")
                return None
            
            # Calculate features
            market_data = self.data_preprocessor.calculate_features(market_data)
            
            # Generate signal using RL strategy
            signal = self.strategy.generate_signals(symbol, market_data, current_position)
            
            # Add additional analysis
            latest_price = market_data['Close'].iloc[-1]
            signal['current_price'] = latest_price
            signal['current_position'] = current_position
            
            # Calculate exposure if we were to enter/modify position
            current_exposure = get_current_symbol_exposure(symbol, positions)
            signal['current_exposure_pct'] = current_exposure
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def select_best_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """Select the best trading opportunities based on RL strategy and constraints."""
        if not opportunities:
            return []
        
        # Filter by minimum confidence
        filtered = [
            opp for opp in opportunities 
            if opp.get('confidence', 0) >= self.min_confidence_threshold
        ]
        
        if not filtered:
            self.logger.info("No opportunities meet minimum confidence threshold")
            return []
        
        # Sort by confidence
        filtered.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Apply portfolio constraints
        current_positions = alpaca_wrapper.get_all_positions()
        current_position_count = len([p for p in current_positions if abs(float(p.market_value or 0)) > 100])
        
        selected = []
        for opp in filtered:
            symbol = opp['symbol']
            
            # Check if we already have a position
            has_position = any(p.symbol == symbol for p in current_positions)
            
            # If we don't have a position, check if we can open new ones
            if not has_position and current_position_count >= self.max_positions:
                self.logger.info(f"Skipping {symbol} - max positions ({self.max_positions}) reached")
                continue
            
            # Check exposure limits
            if opp.get('current_exposure_pct', 0) >= self.max_exposure_per_symbol * 100:
                self.logger.info(f"Skipping {symbol} - max exposure reached")
                continue
            
            selected.append(opp)
            
            # Count this as a position if it's a new one
            if not has_position:
                current_position_count += 1
        
        self.logger.info(f"Selected {len(selected)} opportunities from {len(filtered)} candidates")
        return selected
    
    def calculate_position_sizes(self, opportunities: List[Dict]) -> List[Dict]:
        """Calculate actual position sizes based on RL strategy and risk management."""
        for opp in opportunities:
            symbol = opp['symbol']
            current_price = opp.get('current_price', 0)
            
            if current_price <= 0:
                opp['target_qty'] = 0
                continue
            
            # Use existing position sizing logic but adjusted for RL confidence
            base_qty = get_qty(symbol, current_price)
            
            # Scale by RL confidence
            confidence_multiplier = opp.get('confidence', 0.5)
            adjusted_qty = base_qty * confidence_multiplier
            
            # Apply RL position size recommendation
            rl_position_size = opp.get('position_size', 0.5)  # From RL model
            final_qty = adjusted_qty * rl_position_size
            
            # Final safety checks
            max_value = alpaca_wrapper.equity * self.max_exposure_per_symbol
            max_qty_by_value = max_value / current_price
            final_qty = min(final_qty, max_qty_by_value)
            
            # Round appropriately
            if symbol in crypto_symbols:
                final_qty = round(final_qty, 3)
            else:
                final_qty = int(final_qty)
            
            opp['target_qty'] = max(0, final_qty)
            opp['estimated_value'] = opp['target_qty'] * current_price
            
            self.logger.info(
                f"Position sizing for {symbol}: qty={opp['target_qty']}, "
                f"value=${opp['estimated_value']:,.2f}, confidence={confidence_multiplier:.2%}"
            )
        
        return opportunities
    
    def execute_trades(self, opportunities: List[Dict], dry_run: bool = False) -> List[Dict]:
        """Execute the actual trades."""
        executed_trades = []
        
        for opp in opportunities:
            try:
                symbol = opp['symbol']
                target_qty = opp.get('target_qty', 0)
                side = opp.get('side', 'neutral')
                
                if target_qty <= 0 or side == 'neutral':
                    continue
                
                if dry_run:
                    self.logger.info(f"DRY RUN: Would {side} {target_qty} shares of {symbol}")
                    executed_trades.append({
                        'symbol': symbol,
                        'action': side,
                        'qty': target_qty,
                        'price': opp.get('current_price', 0),
                        'status': 'dry_run',
                        'timestamp': datetime.now()
                    })
                    continue
                
                # Execute real trade
                if side == 'buy':
                    order = alpaca_wrapper.buy_by_target_qty(symbol, target_qty)
                elif side == 'sell':
                    # Check if we have position to sell
                    positions = alpaca_wrapper.get_all_positions()
                    has_position = any(p.symbol == symbol and float(p.qty) > 0 for p in positions)
                    
                    if has_position:
                        order = alpaca_wrapper.sell_by_target_qty(symbol, target_qty)
                    else:
                        self.logger.warning(f"No position to sell for {symbol}")
                        continue
                else:
                    continue
                
                if order:
                    executed_trades.append({
                        'symbol': symbol,
                        'action': side,
                        'qty': target_qty,
                        'price': opp.get('current_price', 0),
                        'order_id': order.id if hasattr(order, 'id') else str(order),
                        'status': 'submitted',
                        'timestamp': datetime.now(),
                        'confidence': opp.get('confidence', 0),
                        'rl_signal': opp.get('recommendation', 'unknown')
                    })
                    
                    self.logger.info(f"✅ Executed {side} order for {symbol}: {target_qty} shares")
                else:
                    self.logger.error(f"❌ Failed to execute {side} order for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error executing trade for {opp.get('symbol', 'unknown')}: {e}")
        
        return executed_trades
    
    def run_trading_cycle(self, dry_run: bool = False) -> Dict:
        """Run one complete trading cycle."""
        cycle_start = datetime.now()
        self.logger.info("="*60)
        self.logger.info(f"Starting trading cycle at {cycle_start}")
        
        # Get current portfolio status
        account_info = alpaca_wrapper.get_account()
        current_positions = alpaca_wrapper.get_all_positions()
        
        self.logger.info(f"Account Equity: ${float(account_info.equity):,.2f}")
        self.logger.info(f"Cash: ${float(account_info.cash):,.2f}")
        self.logger.info(f"Current Positions: {len(current_positions)}")
        
        # Analyze market opportunities
        symbols = self.get_stock_universe()
        opportunities = []
        
        for symbol in symbols:
            opportunity = self.analyze_market_opportunity(symbol)
            if opportunity:
                opportunities.append(opportunity)
        
        self.logger.info(f"Analyzed {len(symbols)} symbols, found {len(opportunities)} opportunities")
        
        # Select best opportunities
        selected_opportunities = self.select_best_opportunities(opportunities)
        
        # Calculate position sizes
        sized_opportunities = self.calculate_position_sizes(selected_opportunities)
        
        # Execute trades
        executed_trades = self.execute_trades(sized_opportunities, dry_run=dry_run)
        
        cycle_result = {
            'timestamp': cycle_start,
            'analyzed_symbols': len(symbols),
            'opportunities_found': len(opportunities),
            'opportunities_selected': len(selected_opportunities),
            'trades_executed': len(executed_trades),
            'account_equity': float(account_info.equity),
            'account_cash': float(account_info.cash),
            'positions_count': len(current_positions),
            'executed_trades': executed_trades
        }
        
        # Log summary
        self.logger.info(f"Cycle completed: {len(executed_trades)} trades executed")
        for trade in executed_trades:
            self.logger.info(f"  {trade['action'].upper()} {trade['symbol']}: {trade['qty']} @ ${trade['price']:.2f}")
        
        return cycle_result
    
    def run_continuous(self, interval_minutes: int = 30, dry_run: bool = False):
        """Run the trading system continuously."""
        self.logger.info(f"Starting continuous trading (interval: {interval_minutes}min, dry_run: {dry_run})")
        
        last_run = datetime.min
        
        try:
            while True:
                current_time = datetime.now()
                
                # Check if it's time for next cycle
                if current_time - last_run >= timedelta(minutes=interval_minutes):
                    
                    # Check if market is open (uses calendar for holidays)
                    if is_nyse_open_on_date(current_time):
                        market_hour = current_time.hour
                        if 9 <= market_hour <= 16:  # Rough market hours
                            cycle_result = self.run_trading_cycle(dry_run=dry_run)
                            last_run = current_time
                        else:
                            self.logger.info("Outside market hours, skipping cycle")
                    else:
                        self.logger.info("Market closed (weekend/holiday), skipping cycle")
                
                # Sleep for a minute before checking again
                time.sleep(60)
                
        except KeyboardInterrupt:
            self.logger.info("Stopping trading system...")
        except Exception as e:
            self.logger.error(f"Unexpected error in continuous trading: {e}")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Trained RL Stock Trading System")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--dry-run', action='store_true', help='Run without executing real trades')
    parser.add_argument('--paper', action='store_true', default=True, help='Use paper trading account')
    parser.add_argument('--continuous', action='store_true', help='Run continuously')
    parser.add_argument('--interval', type=int, default=30, help='Trading interval in minutes')
    parser.add_argument('--single', action='store_true', help='Run single cycle only')
    
    args = parser.parse_args()
    
    # Initialize trading system
    trader = TradeStockE2ETrained(
        config_path=args.config,
        paper_trading=args.paper
    )
    
    if args.single:
        # Run single cycle
        result = trader.run_trading_cycle(dry_run=args.dry_run)
        print(f"Cycle completed. Executed {result['trades_executed']} trades.")
    elif args.continuous:
        # Run continuously
        trader.run_continuous(interval_minutes=args.interval, dry_run=args.dry_run)
    else:
        # Default: run single cycle
        result = trader.run_trading_cycle(dry_run=args.dry_run)
        print(f"Cycle completed. Executed {result['trades_executed']} trades.")


if __name__ == "__main__":
    main()