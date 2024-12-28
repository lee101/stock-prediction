import numpy as np
import pandas as pd
import torch
from loguru import logger

from loss_utils import calculate_trading_profit_torch_with_entry_buysell
from predict_stock_forecasting import make_predictions, load_pipeline


def backtest(symbol, csv_file, num_simulations=30):
    stock_data = pd.read_csv(csv_file, parse_dates=['Date'], index_col='Date')
    stock_data = stock_data.sort_index()

    if len(stock_data) < num_simulations:
        logger.warning(
            f"Not enough historical data for {num_simulations} simulations. Using {len(stock_data)} instead.")
        num_simulations = len(stock_data)

    results = []

    load_pipeline()

    for i in range(num_simulations):
        simulation_data = stock_data.iloc[:-(i + 1)].copy()

        if simulation_data.empty:
            logger.warning(f"No data left for simulation {i + 1}")
            continue

        current_time_formatted = simulation_data.index[-1].strftime('%Y-%m-%d--%H-%M-%S')

        predictions = make_predictions(current_time_formatted, retrain=False)

        last_preds = predictions[predictions['instrument'] == symbol].iloc[-1]

        close_to_high = last_preds['close_last_price'] - last_preds['high_last_price']
        close_to_low = last_preds['close_last_price'] - last_preds['low_last_price']

        scaler = MinMaxScaler()
        scaler.fit(np.array([last_preds['close_last_price']]).reshape(-1, 1))

        # Calculate profits using different strategies
        entry_profit = calculate_trading_profit_torch_with_entry_buysell(
            scaler, None,
            last_preds["close_actual_movement_values"],
            last_preds['entry_takeprofit_profit_high_multiplier'],
            last_preds["high_actual_movement_values"] + close_to_high,
            last_preds["high_predictions"] + close_to_high + last_preds['entry_takeprofit_profit_high_multiplier'],
            last_preds["low_actual_movement_values"] - close_to_low,
            last_preds["low_predictions"] - close_to_low + last_preds['entry_takeprofit_profit_low_multiplier'],
        ).item()

        maxdiff_trades = (torch.abs(last_preds["high_predictions"] + close_to_high) >
                          torch.abs(last_preds["low_predictions"] - close_to_low)) * 2 - 1
        maxdiff_profit = calculate_trading_profit_torch_with_entry_buysell(
            scaler, None,
            last_preds["close_actual_movement_values"],
            maxdiff_trades,
            last_preds["high_actual_movement_values"] + close_to_high,
            last_preds["high_predictions"] + close_to_high,
            last_preds["low_actual_movement_values"] - close_to_low,
            last_preds["low_predictions"] - close_to_low,
        ).item()

        results.append({
            'date': simulation_data.index[-1],
            'close_price': last_preds['close_last_price'],
            'entry_profit': entry_profit,
            'maxdiff_profit': maxdiff_profit,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    symbol = "AAPL"  # Use AAPL as the stock symbol
    current_time_formatted = "2024-09-24_12-23-05"  # Always use this fixed date
    num_simulations = 30

    backtest_results = backtest(symbol, csv_file, num_simulations)
    print(backtest_results)

    # Calculate and print summary statistics
    total_entry_profit = backtest_results['entry_profit'].sum()
    total_maxdiff_profit = backtest_results['maxdiff_profit'].sum()
    avg_entry_profit = backtest_results['entry_profit'].mean()
    avg_maxdiff_profit = backtest_results['maxdiff_profit'].mean()

    print(f"Total Entry Profit: {total_entry_profit}")
    print(f"Total MaxDiff Profit: {total_maxdiff_profit}")
    print(f"Average Entry Profit: {avg_entry_profit}")
    print(f"Average MaxDiff Profit: {avg_maxdiff_profit}")
