import sys
from pathlib import Path
import pandas as pd
from loguru import logger
from datetime import datetime, timedelta

import pytz
import alpaca_wrapper
from predict_stock_forecasting import make_predictions, load_stock_data_from_csv
from data_curate_daily import download_daily_stock_data

def show_forecasts(symbol):
    # Set up logging
    logger.remove()
    logger.add(sys.stdout, format="{time} | {level} | {message}")

    # Check if market is open and if symbol is crypto
    from src.fixtures import crypto_symbols
    is_crypto = symbol in crypto_symbols
    market_clock = alpaca_wrapper.get_clock()
    is_market_open = market_clock.is_open
    
    logger.info(f"Market status: {'OPEN' if is_market_open else 'CLOSED'}")
    logger.info(f"Symbol {symbol} is crypto: {is_crypto}")

    # For crypto, always try to get fresh data since crypto markets are always open
    # For stocks, only get fresh data if market is open, otherwise use cached data
    if is_crypto or is_market_open:
        try:
            target_symbols = [symbol.upper()]
            # Download the latest data
            current_time_formatted = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
            data_df = download_daily_stock_data(current_time_formatted, symbols=target_symbols)
            
            # Make predictions
            predictions = make_predictions(
                current_time_formatted,
                alpaca_wrapper=alpaca_wrapper,
                symbols=target_symbols,
            )
            
            # Filter predictions for the given symbol
            symbol_predictions = predictions[predictions['instrument'] == symbol]
            
            if not symbol_predictions.empty:
                logger.info(f"Using fresh predictions for {symbol}")
                display_predictions(symbol, symbol_predictions, data_df)
                return
            else:
                logger.warning(f"No fresh predictions found for {symbol}, falling back to cached data")
                
        except Exception as e:
            import traceback
            logger.error(f"Error getting fresh data: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.info("Falling back to cached predictions...")
    else:
        logger.info(f"Market is closed and {symbol} is not crypto, using cached data")
    
    # Fallback to cached predictions
    cached_predictions = get_cached_predictions(symbol)
    if cached_predictions is not None:
        logger.info(f"Using cached predictions for {symbol}")
        display_predictions(symbol, cached_predictions, None)
    else:
        logger.error(f"No cached predictions found for symbol {symbol}")


def get_cached_predictions(symbol):
    """Get the most recent cached predictions for a symbol"""
    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        return None
    
    # Get all prediction files sorted by modification time (newest first)
    prediction_files = sorted(results_dir.glob("predictions-*.csv"), 
                            key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Add the generic predictions.csv file if it exists
    generic_file = results_dir / "predictions.csv"
    if generic_file.exists():
        prediction_files.insert(0, generic_file)
    
    # Search through files to find the symbol
    for pred_file in prediction_files:
        try:
            predictions = pd.read_csv(pred_file)
            if 'instrument' in predictions.columns:
                symbol_predictions = predictions[predictions['instrument'] == symbol]
                if not symbol_predictions.empty:
                    logger.info(f"Found cached predictions in {pred_file.name}")
                    return symbol_predictions
        except Exception as e:
            logger.warning(f"Error reading {pred_file}: {e}")
            continue
    
    return None


def display_predictions(symbol, symbol_predictions, data_df):
    """Display prediction results for a symbol"""

    # Display forecasts
    logger.info(f"Forecasts for {symbol}:")
    
    # Handle both new and old column formats
    close_price_col = None
    high_price_col = None
    low_price_col = None
    
    for col in symbol_predictions.columns:
        if 'close_predicted_price' in col and 'value' in col:
            close_price_col = col
        elif 'high_predicted_price' in col and 'value' in col:
            high_price_col = col
        elif 'low_predicted_price' in col and 'value' in col:
            low_price_col = col
    
    # Fallback to older column names if new ones not found
    if close_price_col is None:
        close_price_col = 'close_predicted_price'
    if high_price_col is None:
        high_price_col = 'high_predicted_price'
    if low_price_col is None:
        low_price_col = 'low_predicted_price'
    
    try:
        if close_price_col in symbol_predictions.columns:
            close_value = symbol_predictions[close_price_col].values[0]
            # Handle string representations like "(119.93537139892578,)"
            if isinstance(close_value, str) and close_value.startswith('(') and close_value.endswith(')'):
                close_value = float(close_value.strip('()').rstrip(','))
            logger.info(f"Close price: {close_value:.2f}")
        
        if high_price_col in symbol_predictions.columns:
            high_value = symbol_predictions[high_price_col].values[0]
            if isinstance(high_value, str) and high_value.startswith('(') and high_value.endswith(')'):
                high_value = float(high_value.strip('()').rstrip(','))
            logger.info(f"High price: {high_value:.2f}")
        
        if low_price_col in symbol_predictions.columns:
            low_value = symbol_predictions[low_price_col].values[0]
            if isinstance(low_value, str) and low_value.startswith('(') and low_value.endswith(')'):
                low_value = float(low_value.strip('()').rstrip(','))
            logger.info(f"Low price: {low_value:.2f}")
            
    except Exception as e:
        logger.warning(f"Error displaying price predictions: {e}")

    # Display trading strategies if available
    strategy_cols = ['entry_takeprofit_profit', 'maxdiffprofit_profit', 'takeprofit_profit']
    logger.info("\nTrading strategies:")
    for col in strategy_cols:
        if col in symbol_predictions.columns:
            try:
                value = symbol_predictions[col].values[0]
                if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
                    value = float(value.strip('()').rstrip(','))
                logger.info(f"{col.replace('_', ' ').title()}: {value:.4f}")
            except Exception as e:
                logger.warning(f"Error displaying {col}: {e}")

    # Log all data in symbol_predictions
    logger.info("\nAll prediction data:")
    for key, value in symbol_predictions.iloc[0].to_dict().items():
        try:
            if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
                # Handle string representations like "(119.93537139892578,)"
                clean_value = float(value.strip('()').rstrip(','))
                logger.info(f"{key}: {clean_value:.6f}")
            elif isinstance(value, float):
                logger.info(f"{key}: {value:.6f}")
            elif isinstance(value, list):
                logger.info(f"{key}: {value}")
            else:
                logger.info(f"{key}: {value}")
        except Exception as e:
            logger.info(f"{key}: {value}")

    # Get the last timestamp from data_df (only if available)
    if data_df is not None:
        try:
            last_timestamp = data_df.index[-1]
            if isinstance(last_timestamp, pd.Timestamp):
                last_timestamp = last_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(data_df.index, pd.MultiIndex):
                last_timestamp = data_df.index.get_level_values('timestamp')[-1]
            else:
                last_timestamp = data_df['timestamp'].iloc[-1] if 'timestamp' in data_df.columns else None

            if last_timestamp is None:
                logger.warning("Unable to find timestamp in the data")
                return
            logger.info(f"Last timestamp: {last_timestamp}")
            
            # Convert last_timestamp to datetime object
            if isinstance(last_timestamp, str):
                last_timestamp_datetime = datetime.fromisoformat(last_timestamp)
            elif isinstance(last_timestamp, pd.Timestamp):
                last_timestamp_datetime = last_timestamp.to_pydatetime()
            else:
                logger.warning(f"Unexpected timestamp type: {type(last_timestamp)}")
                return

            logger.info(f"Last timestamp datetime: {last_timestamp_datetime}")
            
            # Convert to NZDT
            nzdt = pytz.timezone('Pacific/Auckland')  # NZDT timezone
            last_timestamp_nzdt = last_timestamp_datetime.astimezone(nzdt)
            logger.info(f"Last timestamp NZDT: {last_timestamp_nzdt}")
            
            # Add one day and print
            last_timestamp_nzdt_plus_one = last_timestamp_nzdt + timedelta(days=1)
            logger.info(f"Last timestamp NZDT plus one day: {last_timestamp_nzdt_plus_one}")
        except Exception as e:
            logger.warning(f"Error processing timestamp data: {e}")
    else:
        logger.info("No fresh data available - using cached predictions only")

    # # Display historical data
    # base_dir = Path(__file__).parent
    # data_dir = base_dir / "data" / current_time_formatted
    # csv_file = data_dir / f"{symbol}.csv"

    # if csv_file.exists():
    #     stock_data = load_stock_data_from_csv(csv_file)
    #     last_7_days = stock_data.tail(7)
        
    #     logger.info("\nLast 7 days of historical data:")
    #     logger.info(last_7_days[['Date', 'Open', 'High', 'Low', 'Close']].to_string(index=False))
    # else:
    #     logger.warning(f"No historical data found for {symbol}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python show_forecasts.py <symbol>")
        sys.exit(1)

    symbol = sys.argv[1]
    show_forecasts(symbol)
