import sys
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
from loguru import logger
import pytz
from time import sleep
import numpy as np
from scipy import stats

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
    """Run backtest analysis on symbols and return results sorted by Sharpe ratio."""
    results = {}
    
    # Calculate Bonferroni-corrected significance level
    base_significance = 0.05  # Standard significance level
    n_tests = len(symbols)  # Number of symbols being tested
    bonferroni_significance = base_significance / n_tests
    
    logger.info(f"Using Bonferroni-corrected significance level: {bonferroni_significance:.4f} "
                f"({n_tests} tests)")
    
    for symbol in symbols:
        try:
            logger.info(f"Analyzing {symbol}")
            num_simulations = 300

            backtest_df = backtest_forecasts(symbol, num_simulations)
            
            # Get average metrics
            avg_sharpe = backtest_df['simple_strategy_sharpe'].mean()
            
            # Calculate p-value for the Sharpe ratio
            n_days = num_simulations
            sharpe_std_error = 1 / np.sqrt(n_days)
            t_stat = avg_sharpe / sharpe_std_error
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_days - 1))
            
            # Determine position side based on predicted price movement
            last_prediction = backtest_df.iloc[-1]
            predicted_movement = last_prediction['predicted_close'] - last_prediction['close']
            position_side = 'buy' if predicted_movement > 0 else 'sell'
            
            results[symbol] = {
                'sharpe': avg_sharpe,
                'p_value': p_value,
                'predictions': backtest_df,
                'side': position_side,
                'predicted_movement': predicted_movement
            }
            
            logger.info(f"Analysis complete for {symbol}: Sharpe={avg_sharpe:.3f}, "
                       f"p-value={p_value:.4f}, side={position_side}")
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            continue
            
    # Sort by Sharpe ratio but include all results
    return dict(sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True))

def log_trading_plan(picks: Dict[str, Dict], action: str):
    """Log the trading plan without executing trades."""
    logger.info(f"\n{'='*50}\nTRADING PLAN ({action})\n{'='*50}")
    
    for symbol, data in picks.items():
        logger.info(f"""
Symbol: {symbol}
Direction: {data['side']}
Sharpe Ratio: {data['sharpe']:.3f}
P-value: {data['p_value']:.4f}
Predicted Movement: {data['predicted_movement']:.3f}
{'='*30}""")

def manage_positions(current_picks: Dict[str, Dict], previous_picks: Dict[str, Dict], all_analyzed_results: Dict[str, Dict]):
    """Execute actual position management."""
    positions = alpaca_wrapper.get_all_positions()
    
    logger.info("\nEXECUTING POSITION CHANGES:")
    
    if not positions:
        logger.info("No positions to analyze")
        return
        
    if not all_analyzed_results:
        logger.warning("No analysis results available - skipping position closure checks")
        return
    
    # Close positions only when forecast shows opposite direction
    for position in positions:
        symbol = position.symbol
        should_close = False
        
        # Only close if we have a new analysis showing opposite direction
        if symbol in all_analyzed_results:
            new_forecast = all_analyzed_results[symbol]
            if new_forecast['side'] != position.side:
                logger.info(f"Closing position for {symbol} due to direction change from {position.side} to {new_forecast['side']}")
                logger.info(f"Predicted movement: {new_forecast['predicted_movement']:.3f}")
                should_close = True
            else:
                logger.info(f"Keeping {symbol} position as forecast matches current {position.side} direction")
        else:
            logger.warning(f"No analysis data for {symbol} - keeping position")
            
        if should_close:
            backout_near_market(symbol)
            
    # Enter new positions from current picks
    if not current_picks:
        logger.warning("No current picks available - skipping new position entry")
        return
        
    for symbol, data in current_picks.items():
        position_exists = any(p.symbol == symbol for p in positions)
        correct_side = any(p.symbol == symbol and p.side == data['side'] for p in positions)
        
        if not position_exists or not correct_side:
            logger.info(f"Entering new {data['side']} position for {symbol}")
            ramp_into_position(symbol, data['side'])

def manage_market_close(symbols: List[str], previous_picks: Dict[str, Dict], all_analyzed_results: Dict[str, Dict]):
    """Execute market close position management."""
    logger.info("Managing positions for market close")
    
    if not all_analyzed_results:
        logger.warning("No analysis results available - keeping all positions open")
        return previous_picks
        
    positions = alpaca_wrapper.get_all_positions()
    if not positions:
        logger.info("No positions to manage for market close")
        return {symbol: data for symbol, data in list(all_analyzed_results.items())[:4] 
                if data['sharpe'] > 0}
    
    # Close positions only when forecast shows opposite direction
    for position in positions:
        symbol = position.symbol
        should_close = False
        
        # Only close if we have a new analysis showing opposite direction
        if symbol in all_analyzed_results:
            next_forecast = all_analyzed_results[symbol]
            if next_forecast['side'] != position.side:
                logger.info(f"Closing position for {symbol} due to predicted direction change from {position.side} to {next_forecast['side']} tomorrow")
                logger.info(f"Predicted movement: {next_forecast['predicted_movement']:.3f}")
                should_close = True
            else:
                logger.info(f"Keeping {symbol} position as tomorrow's forecast matches current {position.side} direction")
        else:
            logger.warning(f"No analysis data for {symbol} - keeping position")
            
        if should_close:
            backout_near_market(symbol)
            
    # Return top picks for next day
    return {symbol: data for symbol, data in list(all_analyzed_results.items())[:4] 
            if data['sharpe'] > 0}

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
        'COIN', 'MSFT', 'NFLX', 
        'BTCUSD', 'ETHUSD', "UNIUSD"
    ]
    previous_picks = {}
    initial_analysis_done = False
    
    while True:
        try:
            market_open, market_close = get_market_hours()
            now = datetime.now(pytz.timezone('US/Eastern'))
            
            # Initial analysis when program starts - using dry run
            if not initial_analysis_done and now.hour == 22 and now.minute >= 0 and now.minute < 30:
                logger.info("\nINITIAL ANALYSIS STARTING...")
                all_analyzed_results = analyze_symbols(symbols)
                current_picks = {
                    symbol: data for symbol, data in list(all_analyzed_results.items())[:4] 
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
                all_analyzed_results = analyze_symbols(symbols)
                current_picks = {
                    symbol: data for symbol, data in list(all_analyzed_results.items())[:4] 
                    if data['sharpe'] > 0  # Only positive Sharpe ratios
                }
                log_trading_plan(current_picks, "MARKET OPEN PLAN")
                manage_positions(current_picks, previous_picks, all_analyzed_results)  # Real trading at market open
                previous_picks = current_picks
                market_open_done = True
                sleep(3600)  # Sleep 1 hour after open analysis
                
            # Market close analysis - use real trading
            elif now.hour == market_close.hour - 1 and now.minute >= market_close.minute + 30 and not market_close_done:
                logger.info("\nMARKET CLOSE ANALYSIS STARTING...")
                all_analyzed_results = analyze_symbols(symbols)
                previous_picks = manage_market_close(symbols, previous_picks, all_analyzed_results)  # Real trading at market close
                market_close_done = True
                sleep(3600)  # Sleep 1 hour after close analysis
                    
            sleep(60)  # Check every minute
            
        except Exception as e:
            logger.exception(f"Error in main loop: {str(e)}")
            sleep(60)

if __name__ == "__main__":
    main()