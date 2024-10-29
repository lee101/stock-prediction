import sys
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
from loguru import logger
import pytz
from time import sleep

from backtest_test3_inline import backtest_forecasts
from src.process_utils import backout_near_market, ramp_into_position
from src.fixtures import crypto_symbols
import alpaca_wrapper
from src.date_utils import is_nyse_trading_day_now, is_nyse_trading_day_ending

# Configure logging
class EDTFormatter:
    def __init__(self):
        self.local_tz = pytz.timezone('US/Eastern')

    def __call__(self, record):
        utc_time = record["time"].strftime('%Y-%m-%d %H:%M:%S %Z')
        local_time = datetime.now(self.local_tz).strftime('%Y-%m-%d %H:%M:%S %Z')
        level_colors = {
            "DEBUG": "\033[36m",
            "INFO": "\033[32m",
            "WARNING": "\033[33m",
            "ERROR": "\033[31m",
            "CRITICAL": "\033[35m"
        }
        reset_color = "\033[0m"
        level_color = level_colors.get(record['level'].name, "")
        return f"{utc_time} | {local_time} | {level_color}{record['level'].name}{reset_color} | {record['message']}\n"

logger.remove()
logger.add(sys.stdout, format=EDTFormatter())
logger.add("trade_stock_e2e.log", format=EDTFormatter())

def get_market_hours() -> tuple:
    """Get market open and close times in EST."""
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open, market_close

def analyze_symbols(symbols: List[str]) -> Dict:
    """Run backtest analysis on symbols and return results sorted by Sharpe ratio and determine position side."""
    results = {}
    
    for symbol in symbols:
        try:
            logger.info(f"Analyzing {symbol}")
            backtest_df = backtest_forecasts(symbol)
            
            # Get average metrics
            avg_sharpe = backtest_df['simple_strategy_sharpe'].mean()

            # Only include if Sharpe ratio is positive
            if avg_sharpe <= 0:
                continue

            # Determine position side based on predicted price movement
            last_prediction = backtest_df.iloc[-1]
            predicted_movement = last_prediction['predicted_close'] - last_prediction['close']
            position_side = 'buy' if predicted_movement > 0 else 'sell'
            
            results[symbol] = {
                'sharpe': avg_sharpe,
                'predictions': backtest_df,
                'side': position_side,
                'predicted_movement': predicted_movement
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            continue
            
    # Sort by Sharpe ratio (already filtered for positive only)
    return dict(sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True))

def log_trading_plan(picks: Dict[str, Dict], action: str):
    """Log the trading plan without executing trades."""
    logger.info(f"\n{'='*50}\nTRADING PLAN ({action})\n{'='*50}")
    
    for symbol, data in picks.items():
        logger.info(f"""
Symbol: {symbol}
Direction: {data['side']}
Sharpe Ratio: {data['sharpe']:.3f}
Predicted Movement: {data['predicted_movement']:.3f}
{'='*30}""")

def manage_positions(current_picks: Dict[str, Dict], previous_picks: Dict[str, Dict]):
    """Execute actual position management."""
    positions = alpaca_wrapper.get_all_positions()
    
    logger.info("\nEXECUTING POSITION CHANGES:")
    
    # Close positions that are no longer needed
    for position in positions:
        symbol = position.symbol
        should_close = False
        
        if symbol not in current_picks:
            logger.info(f"Closing position for {symbol} as it's no longer in top picks")
            should_close = True
        elif symbol in current_picks and current_picks[symbol]['side'] != position.side:
            logger.info(f"Closing position for {symbol} to switch direction from {position.side} to {current_picks[symbol]['side']}")
            should_close = True
            
        if should_close:
            backout_near_market(symbol)
            
    # Enter new positions
    for symbol, data in current_picks.items():
        position_exists = any(p.symbol == symbol for p in positions)
        correct_side = any(p.symbol == symbol and p.side == data['side'] for p in positions)
        
        if not position_exists or not correct_side:
            logger.info(f"Entering new {data['side']} position for {symbol}")
            ramp_into_position(symbol, data['side'])

def manage_market_close(symbols: List[str], previous_picks: Dict[str, Dict]):
    """Execute market close position management."""
    logger.info("Managing positions for market close")
    
    # Get next day's analysis before closing positions
    next_day_picks = {
        symbol: data for symbol, data in list(analyze_next_day_positions(symbols).items())[:4]
        if abs(data['sharpe']) > 0.5
    }
    
    positions = alpaca_wrapper.get_all_positions()
    
    # Close positions that won't be needed tomorrow
    for position in positions:
        symbol = position.symbol
        should_close = False
        
        if symbol not in next_day_picks:
            logger.info(f"Closing position for {symbol} as it's not in next day's picks")
            should_close = True
        elif symbol in next_day_picks and next_day_picks[symbol]['side'] != position.side:
            logger.info(f"Closing position for {symbol} to switch direction from {position.side} to {next_day_picks[symbol]['side']} tomorrow")
            should_close = True
            
        if should_close:
            backout_near_market(symbol)
            
    return next_day_picks

def analyze_next_day_positions(symbols: List[str]) -> Dict:
    """Analyze symbols for next day's trading session."""
    logger.info("Analyzing positions for next trading day")
    return analyze_symbols(symbols)  # Reuse existing analysis function

def dry_run_manage_positions(current_picks: Dict[str, Dict], previous_picks: Dict[str, Dict]):
    """Simulate position management without executing trades."""
    positions = alpaca_wrapper.get_all_positions()
    
    logger.info("\nPLANNED POSITION CHANGES:")
    
    # Log position closures
    for position in positions:
        symbol = position.symbol
        should_close = False
        
        if symbol not in current_picks:
            logger.info(f"Would close position for {symbol} as it's no longer in top picks")
            should_close = True
        elif symbol in current_picks and current_picks[symbol]['side'] != position.side:
            logger.info(f"Would close position for {symbol} to switch direction from {position.side} to {current_picks[symbol]['side']}")
            should_close = True
            
    # Log new positions
    for symbol, data in current_picks.items():
        position_exists = any(p.symbol == symbol for p in positions)
        correct_side = any(p.symbol == symbol and p.side == data['side'] for p in positions)
        
        if not position_exists or not correct_side:
            logger.info(f"Would enter new {data['side']} position for {symbol}")



def main():
    symbols = [
        'COUR', 'GOOG', 'TSLA', 'NVDA', 'AAPL', "U", "ADSK", "CRWD", "ADBE", "NET",
        'COIN', 'MSFT', 'NFLX', 'BTCUSD', 'ETHUSD',
    ]
    previous_picks = {}
    initial_analysis_done = False
    
    while True:
        try:
            market_open, market_close = get_market_hours()
            now = datetime.now(pytz.timezone('US/Eastern'))
            
            # Initial analysis when program starts - using dry run
            if not initial_analysis_done:
                logger.info("\nINITIAL ANALYSIS STARTING...")
                results = analyze_symbols(symbols)
                current_picks = {
                    symbol: data for symbol, data in list(results.items())[:4] 
                    if data['sharpe'] > 0  # Only positive Sharpe ratios
                }
                log_trading_plan(current_picks, "INITIAL PLAN")
                dry_run_manage_positions(current_picks, previous_picks)  # Keep dry run here
                previous_picks = current_picks
                initial_analysis_done = True
                market_open_done = False
                market_close_done = False
                
            # Market open analysis - use real trading
            elif (now.hour == market_open.hour and 
                  now.minute >= market_open.minute and 
                  now.minute < market_open.minute + 30 and
                  not market_open_done):
                logger.info("\nMARKET OPEN ANALYSIS STARTING...")
                results = analyze_symbols(symbols)
                current_picks = {
                    symbol: data for symbol, data in list(results.items())[:4] 
                    if data['sharpe'] > 0  # Only positive Sharpe ratios
                }
                log_trading_plan(current_picks, "MARKET OPEN PLAN")
                manage_positions(current_picks, previous_picks)  # Real trading at market open
                previous_picks = current_picks
                market_open_done = True
                
            # Market close analysis - use real trading
            elif is_nyse_trading_day_ending() and not market_close_done:
                logger.info("\nMARKET CLOSE ANALYSIS STARTING...")
                previous_picks = manage_market_close(symbols, previous_picks)  # Real trading at market close
                market_close_done = True
                sleep(300)  # Sleep 5 minutes after close analysis
                    
            sleep(60)  # Check every minute
            
        except Exception as e:
            logger.exception(f"Error in main loop: {str(e)}")
            sleep(60)

if __name__ == "__main__":
    main()