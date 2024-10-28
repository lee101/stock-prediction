import sys
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime, timedelta


import torch
import alpaca_wrapper
from predict_stock_forecasting import load_pipeline, make_predictions, load_stock_data_from_csv, pre_process_data, series_to_tensor
from data_curate_daily import download_daily_stock_data
from loss_utils import calculate_trading_profit_torch_with_buysell, calculate_trading_profit_torch_with_entry_buysell
from src.conversion_utils import unwrap_tensor

ETH_SPREAD = 1.0008711461252937


from chronos import ChronosPipeline

current_date_formatted = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# tb_writer = SummaryWriter(log_dir=f"./logs/{current_date_formatted}")

pipeline = None


def load_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = ChronosPipeline.from_pretrained(
            # "amazon/chronos-t5-large" if not PAPER else "amazon/chronos-t5-tiny",
            # "amazon/chronos-t5-tiny",
            "amazon/chronos-t5-large",
            device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
            # torch_dtype=torch.bfloat16,
        )
        pipeline.model = pipeline.model.eval()
        # pipeline.model = torch.compile(pipeline.model)


def backtest_forecasts(symbol, num_simulations=20):
    logger.remove()
    logger.add(sys.stdout, format="{time} | {level} | {message}")

    # Download the latest data
    current_time_formatted = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    # hardcode repeatable time for testing
    current_time_formatted = "2024-10-18--06-05-32"
    symbols = [symbol]
    symbols = ['MSFT']

    # stock_data = download_daily_stock_data(current_time_formatted, symbols=symbols)
    stock_data = pd.read_csv(f"./data/{current_time_formatted}/{symbol}-{current_time_formatted}.csv")

    base_dir = Path(__file__).parent
    data_dir = base_dir / "data" / current_time_formatted


    # stock_data = load_stock_data_from_csv(csv_file)

    if len(stock_data) < num_simulations:
        logger.warning(f"Not enough historical data for {num_simulations} simulations. Using {len(stock_data)} instead.")
        num_simulations = len(stock_data)

    results = []

    for i in range(num_simulations):
        # Take one day off each iteration
        simulation_data = stock_data.iloc[:-(i+1)].copy()

        if simulation_data.empty:
            logger.warning(f"No data left for simulation {i+1}")
            continue

        last_preds = {
            'instrument': symbol,
            'close_last_price': simulation_data['Close'].iloc[-1],
        }

        for key_to_predict in ['Close', 'Low', 'High', 'Open']:
            data = pre_process_data(simulation_data, key_to_predict)
            price = data[["Close", "High", "Low", "Open"]]

            price = price.rename(columns={"Date": "time_idx"})
            price["ds"] = pd.date_range(start="1949-01-01", periods=len(price), freq="D").values
            price['y'] = price[key_to_predict].shift(-1)
            price['trade_weight'] = (price["y"] > 0) * 2 - 1

            price.drop(price.tail(1).index, inplace=True)
            price['id'] = price.index
            price['unique_id'] = 1
            price = price.dropna()

            training = price[:-7]
            validation = price[-7:]

            load_pipeline()
            predictions = []
            for pred_idx in reversed(range(1, 8)):
                current_context = price[:-pred_idx]
                context = torch.tensor(current_context["y"].values, dtype=torch.float)

                prediction_length = 1
                forecast = pipeline.predict(
                    context,
                    prediction_length,
                    num_samples=20,
                    temperature=1.0,
                    top_k=4000,
                    top_p=1.0,
                )
                low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
                predictions.append(median.item())

            predictions = torch.tensor(predictions)
            actuals = series_to_tensor(validation["y"])
            trading_preds = (predictions[:-1] > 0) * 2 - 1

            error = np.array(validation["y"][:-1].values) - np.array(predictions[:-1])
            mean_val_loss = np.abs(error).mean()

            last_preds[key_to_predict.lower() + "_last_price"] = simulation_data[key_to_predict].iloc[-1]
            last_preds[key_to_predict.lower() + "_predicted_price"] = unwrap_tensor(predictions[-1])
            last_preds[key_to_predict.lower() + "_predicted_price_value"] = unwrap_tensor(last_preds[key_to_predict.lower() + "_last_price"] + (
                    last_preds[key_to_predict.lower() + "_last_price"] * predictions[-1]))
            last_preds[key_to_predict.lower() + "_val_loss"] = mean_val_loss
            last_preds[key_to_predict.lower() + "_actual_movement_values"] = actuals[:-1].view(-1)
            last_preds[key_to_predict.lower() + "_trade_values"] = trading_preds.view(-1)
            last_preds[key_to_predict.lower() + "_predictions"] = predictions[:-1].view(-1)

        validation_size = last_preds["high_actual_movement_values"].numel()
        close_to_high = series_to_tensor(
            abs(1 - (simulation_data["High"].iloc[-validation_size - 2:-2] / simulation_data["Close"].iloc[-validation_size - 2:-2])))
        close_to_low = series_to_tensor(abs(1 - (simulation_data["Low"].iloc[-validation_size - 2:-2] / simulation_data["Close"].iloc[-validation_size - 2:-2])))

        calculated_profit = calculate_trading_profit_torch_with_buysell(None, None,
                                                                        last_preds["close_actual_movement_values"],
                                                                        last_preds["close_trade_values"],
                                                                        last_preds["high_actual_movement_values"] + close_to_high,
                                                                        last_preds["high_predictions"] + close_to_high,
                                                                        last_preds["low_actual_movement_values"] - close_to_low,
                                                                        last_preds["low_predictions"] - close_to_low).item()
        last_preds['takeprofit_profit'] = calculated_profit

        calculated_profit = calculate_trading_profit_torch_with_entry_buysell(None, None,
                                                                              last_preds["close_actual_movement_values"],
                                                                              last_preds["close_trade_values"],
                                                                              last_preds["high_actual_movement_values"] + close_to_high,
                                                                              last_preds["high_predictions"] + close_to_high,
                                                                              last_preds["low_actual_movement_values"] - close_to_low,
                                                                              last_preds["low_predictions"] - close_to_low).item()
        last_preds['entry_takeprofit_profit'] = calculated_profit

        high_diffs = torch.abs(last_preds["high_predictions"] + close_to_high)
        low_diffs = torch.abs(last_preds["low_predictions"] - close_to_low)
        maxdiff_trades = (high_diffs > low_diffs) * 2 - 1
        calculated_profit = calculate_trading_profit_torch_with_entry_buysell(None, None,
                                                                              last_preds["close_actual_movement_values"],
                                                                              maxdiff_trades,
                                                                              last_preds["high_actual_movement_values"] + close_to_high,
                                                                              last_preds["high_predictions"] + close_to_high,
                                                                              last_preds["low_actual_movement_values"] - close_to_low,
                                                                              last_preds["low_predictions"] - close_to_low).item()
        last_preds['maxdiffprofit_profit'] = calculated_profit

        open_price = simulation_data['Open'].iloc[-1]
        close_price = simulation_data['Close'].iloc[-1]
        predicted_close = last_preds['close_predicted_price_value']

        if pd.notna(predicted_close) and pd.notna(open_price) and pd.notna(close_price):
            if predicted_close > open_price:
                entry_hold_profit = (close_price - open_price) / open_price
            else:
                entry_hold_profit = 0
        else:
            entry_hold_profit = 0

        last_preds['entry_hold_profit'] = entry_hold_profit

        result = {
            'date': simulation_data.index[-1],
            'close': last_preds['close_last_price'],
            'predicted_close': last_preds['close_predicted_price_value'],
            'predicted_high': last_preds['high_predicted_price_value'],
            'predicted_low': last_preds['low_predicted_price_value'],
            'entry_takeprofit_profit': last_preds['entry_takeprofit_profit'],
            'maxdiffprofit_profit': last_preds['maxdiffprofit_profit'],
            'takeprofit_profit': last_preds['takeprofit_profit'],
            'entry_hold_profit': last_preds['entry_hold_profit']
        }
        results.append(result)
        print("Result:")
        print(result)

    results_df = pd.DataFrame(results)

    logger.info(f"\nBacktest results for {symbol} over {num_simulations} simulations:")
    logger.info(f"Average Entry TakeProfit: {results_df['entry_takeprofit_profit'].mean():.4f}")
    logger.info(f"Average MaxDiff Profit: {results_df['maxdiffprofit_profit'].mean():.4f}")
    logger.info(f"Average TakeProfit: {results_df['takeprofit_profit'].mean():.4f}")

    # logger.info("\nPrediction accuracy:")
    # logger.info(f"Close price RMSE: {np.sqrt(((results_df['close'] - results_df['predicted_close'])**2).mean()):.2f}")

    return results_df

if __name__ == "__main__":
    if len(sys.argv) != 2:
        symbol = "ETHUSD"
        print("Usage: python backtest_test.py <symbol> defaultint to eth")
    else:
        symbol = sys.argv[1]

    backtest_forecasts(symbol)
