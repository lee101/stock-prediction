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

    # Download the latest data
    current_time_formatted = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    data_df = download_daily_stock_data(current_time_formatted)

    # Make predictions
    predictions = make_predictions(current_time_formatted, alpaca_wrapper=alpaca_wrapper)

    # Filter predictions for the given symbol
    symbol_predictions = predictions[predictions['instrument'] == symbol]

    if symbol_predictions.empty:
        logger.error(f"No predictions found for symbol {symbol}")
        return

    # Display forecasts
    logger.info(f"Forecasts for {symbol}:")
    logger.info(f"Close price: {symbol_predictions['close_predicted_price_value'].values[0]:.2f}")
    logger.info(f"High price: {symbol_predictions['high_predicted_price_value'].values[0]:.2f}")
    logger.info(f"Low price: {symbol_predictions['low_predicted_price_value'].values[0]:.2f}")

    logger.info("\nTrading strategies:")
    logger.info(f"Entry TakeProfit: {symbol_predictions['entry_takeprofit_profit'].values[0]:.4f}")
    logger.info(f"MaxDiff Profit: {symbol_predictions['maxdiffprofit_profit'].values[0]:.4f}")
    logger.info(f"TakeProfit: {symbol_predictions['takeprofit_profit'].values[0]:.4f}")

    # Log all data in symbol_predictions
    logger.info("\nAll prediction data:")
    for key, value in symbol_predictions.iloc[0].to_dict().items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.6f}")
        elif isinstance(value, list):
            logger.info(f"{key}: {value}")
        else:
            logger.info(f"{key}: {value}")

    # Get the last timestamp from data_df
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