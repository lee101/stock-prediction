import sys
from pathlib import Path
import pandas as pd
from loguru import logger
from datetime import datetime, timedelta
import alpaca_wrapper
from predict_stock_forecasting import make_predictions, load_stock_data_from_csv
from data_curate_daily import download_daily_stock_data

def show_forecasts(symbol):
    # Set up logging
    logger.remove()
    logger.add(sys.stdout, format="{time} | {level} | {message}")

    # Download the latest data
    current_time_formatted = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    download_daily_stock_data(current_time_formatted)

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

    # Display historical data
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data" / current_time_formatted
    csv_file = data_dir / f"{symbol}.csv"

    if csv_file.exists():
        stock_data = load_stock_data_from_csv(csv_file)
        last_7_days = stock_data.tail(7)
        
        logger.info("\nLast 7 days of historical data:")
        logger.info(last_7_days[['Date', 'Open', 'High', 'Low', 'Close']].to_string(index=False))
    else:
        logger.warning(f"No historical data found for {symbol}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python show_forecasts.py <symbol>")
        sys.exit(1)

    symbol = sys.argv[1]
    show_forecasts(symbol)