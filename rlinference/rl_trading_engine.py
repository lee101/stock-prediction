import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

sys.path.append('..')
sys.path.append('../training')

from rlinference.configs.trading_config import TradingConfig, AlpacaConfig, ModelConfig
from rlinference.utils.data_preprocessing import DataPreprocessor
from rlinference.utils.model_manager import ModelManager
from rlinference.brokers.alpaca_broker import AlpacaBroker
from rlinference.strategies.rl_strategy import RLTradingStrategy
from rlinference.utils.risk_manager import RiskManager
from rlinference.utils.portfolio_tracker import PortfolioTracker


class RLTradingEngine:
    def __init__(self, config: TradingConfig, alpaca_config: AlpacaConfig):
        self.config = config
        
        # Setup logging
        logger.add(
            config.log_dir / f"trading_{datetime.now():%Y%m%d_%H%M%S}.log",
            level=config.log_level,
            format="{time} {level} {message}"
        )
        
        # Initialize components
        self.model_manager = ModelManager(models_dir=config.models_dir)
        self.data_preprocessor = DataPreprocessor()
        self.broker = AlpacaBroker(alpaca_config, paper=config.paper_trading)
        self.risk_manager = RiskManager(config)
        self.portfolio_tracker = PortfolioTracker(config.initial_balance)
        self.strategy = RLTradingStrategy(config, self.model_manager, self.data_preprocessor)
        
        # State tracking
        self.positions: Dict[str, dict] = {}
        self.daily_trades = 0
        self.last_data_update = {}
        self.market_data_cache = {}
        
        logger.info(f"RLTradingEngine initialized with {len(config.symbols)} symbols")
        logger.info(f"Trading mode: {'PAPER' if config.paper_trading else 'LIVE'}")
        
    def update_market_data(self, symbol: str) -> pd.DataFrame:
        """Fetch and cache latest market data."""
        
        current_time = time.time()
        
        # Check cache
        if symbol in self.last_data_update:
            if current_time - self.last_data_update[symbol] < self.config.data_refresh_interval:
                logger.debug(f"Using cached data for {symbol}")
                return self.market_data_cache[symbol]
        
        # Fetch new data
        logger.info(f"Fetching market data for {symbol}")
        df = self.data_preprocessor.fetch_realtime_data(symbol)
        
        if not df.empty:
            df = self.data_preprocessor.calculate_features(df)
            self.market_data_cache[symbol] = df
            self.last_data_update[symbol] = current_time
        
        return df
    
    def get_current_positions(self) -> Dict[str, dict]:
        """Get current positions from broker."""
        
        positions = self.broker.get_positions()
        
        # Convert to internal format
        formatted_positions = {}
        for pos in positions:
            formatted_positions[pos.symbol] = {
                'qty': float(pos.qty),
                'side': pos.side,
                'entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price) if hasattr(pos, 'current_price') else None,
                'market_value': float(pos.market_value) if hasattr(pos, 'market_value') else None,
                'unrealized_pl': float(pos.unrealized_pl) if hasattr(pos, 'unrealized_pl') else None
            }
        
        self.positions = formatted_positions
        return formatted_positions
    
    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Analyze a symbol and get trading recommendation."""
        
        # Get market data
        df = self.update_market_data(symbol)
        if df.empty or len(df) < self.data_preprocessor.window_size:
            logger.warning(f"Insufficient data for {symbol}")
            return None
        
        # Get current position info
        current_position = 0.0
        entry_price = 0.0
        if symbol in self.positions:
            pos = self.positions[symbol]
            current_position = pos['qty'] if pos['side'] == 'long' else -pos['qty']
            entry_price = pos['entry_price']
        
        # Get account info
        account = self.broker.get_account()
        current_balance = float(account.equity)
        
        # Prepare observation
        observation = self.data_preprocessor.prepare_observation(
            df,
            current_position=current_position,
            current_balance=current_balance,
            initial_balance=self.config.initial_balance,
            entry_price=entry_price
        )
        
        # Get model recommendation
        recommendation = self.model_manager.get_position_recommendation(
            symbol,
            observation,
            max_position_size=self.config.max_position_size,
            use_ensemble=self.config.ensemble_predictions
        )
        
        # Add current market prices
        bid, ask, last = self.data_preprocessor.get_latest_prices(symbol)
        recommendation['bid'] = bid
        recommendation['ask'] = ask
        recommendation['last_price'] = last
        
        # Add technical indicators for context
        latest_data = df.iloc[-1]
        recommendation['rsi'] = latest_data.get('RSI', 50)
        recommendation['volume_ratio'] = latest_data.get('Volume_Ratio', 1.0)
        
        return recommendation
    
    def execute_recommendations(self, recommendations: List[Dict]):
        """Execute trading recommendations with risk management."""
        
        if self.config.dry_run:
            logger.info("DRY RUN MODE - Not executing trades")
            for rec in recommendations:
                logger.info(f"Would execute: {rec}")
            return
        
        # Apply risk checks
        recommendations = self.risk_manager.filter_recommendations(
            recommendations,
            self.positions,
            self.portfolio_tracker
        )
        
        # Check daily trade limit
        if self.daily_trades >= self.config.max_daily_trades:
            logger.warning(f"Daily trade limit reached ({self.config.max_daily_trades})")
            return
        
        # Execute trades
        for rec in recommendations:
            try:
                symbol = rec['symbol']
                side = rec['side']
                position_size = rec['position_size']
                
                # Calculate order quantity
                if rec['last_price']:
                    account = self.broker.get_account()
                    available_cash = float(account.cash)
                    position_value = min(
                        available_cash * position_size,
                        self.config.max_position_value
                    )
                    qty = int(position_value / rec['last_price'])
                    
                    if qty > 0:
                        # Check if we need to close opposite position first
                        if symbol in self.positions:
                            current_pos = self.positions[symbol]
                            if (current_pos['side'] == 'long' and side == 'sell') or \
                               (current_pos['side'] == 'short' and side == 'buy'):
                                logger.info(f"Closing opposite position for {symbol}")
                                self.broker.close_position(symbol)
                                time.sleep(1)  # Brief pause
                        
                        # Place order
                        logger.info(f"Placing {side} order for {symbol}: {qty} shares")
                        order = self.broker.place_order(
                            symbol=symbol,
                            qty=qty,
                            side=side,
                            order_type='market'
                        )
                        
                        if order:
                            self.daily_trades += 1
                            self.portfolio_tracker.record_trade(
                                symbol=symbol,
                                side=side,
                                qty=qty,
                                price=rec['last_price'],
                                timestamp=datetime.now()
                            )
                            
                            # Set stop-loss and take-profit if configured
                            if self.config.stop_loss or self.config.take_profit:
                                self.set_exit_orders(symbol, side, rec['last_price'])
                    else:
                        logger.warning(f"Calculated quantity is 0 for {symbol}")
                else:
                    logger.warning(f"No price available for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error executing trade for {rec['symbol']}: {e}")
    
    def set_exit_orders(self, symbol: str, side: str, entry_price: float):
        """Set stop-loss and take-profit orders."""
        
        try:
            if self.config.stop_loss:
                stop_price = entry_price * (1 - self.config.stop_loss) if side == 'buy' else \
                            entry_price * (1 + self.config.stop_loss)
                
                self.broker.place_order(
                    symbol=symbol,
                    qty=None,  # Use position quantity
                    side='sell' if side == 'buy' else 'buy',
                    order_type='stop',
                    stop_price=stop_price
                )
                logger.info(f"Stop-loss set for {symbol} at {stop_price:.2f}")
            
            if self.config.take_profit:
                limit_price = entry_price * (1 + self.config.take_profit) if side == 'buy' else \
                             entry_price * (1 - self.config.take_profit)
                
                self.broker.place_order(
                    symbol=symbol,
                    qty=None,  # Use position quantity
                    side='sell' if side == 'buy' else 'buy',
                    order_type='limit',
                    limit_price=limit_price
                )
                logger.info(f"Take-profit set for {symbol} at {limit_price:.2f}")
                
        except Exception as e:
            logger.error(f"Error setting exit orders for {symbol}: {e}")
    
    def run_analysis_cycle(self):
        """Run one complete analysis and trading cycle."""
        
        logger.info("=" * 50)
        logger.info("Starting analysis cycle")
        
        # Update positions
        self.get_current_positions()
        
        # Check circuit breaker
        if self.risk_manager.check_circuit_breaker(self.portfolio_tracker):
            logger.error("Circuit breaker triggered - stopping trading")
            return
        
        # Analyze all symbols
        recommendations = []
        for symbol in self.config.symbols:
            try:
                rec = self.analyze_symbol(symbol)
                if rec and rec['confidence'] > 0.3:  # Min confidence threshold
                    recommendations.append(rec)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Limit to max positions
        recommendations = recommendations[:self.config.max_positions]
        
        # Log recommendations
        logger.info(f"Generated {len(recommendations)} recommendations:")
        for rec in recommendations:
            logger.info(
                f"  {rec['symbol']}: {rec['side']} "
                f"size={rec['position_size']:.2%} "
                f"confidence={rec['confidence']:.2%}"
            )
        
        # Execute trades
        self.execute_recommendations(recommendations)
        
        # Update portfolio tracker
        account = self.broker.get_account()
        self.portfolio_tracker.update(
            equity=float(account.equity),
            cash=float(account.cash),
            positions=self.positions
        )
        
        # Log portfolio status
        metrics = self.portfolio_tracker.get_metrics()
        logger.info(f"Portfolio Status:")
        logger.info(f"  Equity: ${metrics['equity']:,.2f}")
        logger.info(f"  Daily Return: {metrics['daily_return']:.2%}")
        logger.info(f"  Total Return: {metrics['total_return']:.2%}")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"  Positions: {len(self.positions)}")
        
    def run(self, interval_seconds: int = 300):
        """Run the trading engine continuously."""
        
        logger.info("Starting RLTradingEngine")
        
        while True:
            try:
                # Check if market is open (for stocks)
                if not self.broker.is_market_open() and not self.has_crypto_symbols():
                    logger.info("Market is closed, waiting...")
                    time.sleep(60)
                    continue
                
                # Run analysis cycle
                self.run_analysis_cycle()
                
                # Reset daily trades at market open
                if self.broker.is_new_trading_day():
                    self.daily_trades = 0
                    self.portfolio_tracker.new_day()
                
                # Wait for next cycle
                logger.info(f"Waiting {interval_seconds} seconds until next cycle...")
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("Shutting down RLTradingEngine")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def has_crypto_symbols(self) -> bool:
        """Check if any symbols are crypto."""
        crypto_suffixes = ['USD', 'USDT', 'BTC', 'ETH']
        return any(
            any(symbol.endswith(suffix) for suffix in crypto_suffixes)
            for symbol in self.config.symbols
        )