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
from loss_utils import calculate_trading_profit_torch_with_buysell

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


def backtest_forecasts(symbol, num_simulations=100):
    logger.remove()
    logger.add(sys.stdout, format="{time} | {level} | {message}")

    # Download the latest data
    current_time_formatted = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    stock_data = download_daily_stock_data(current_time_formatted, symbols=[symbol])

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
        
        key_to_predict = "Close"
        last_close_price = simulation_data[key_to_predict].iloc[-1]
        
        data = pre_process_data(simulation_data, "High")
        data = pre_process_data(data, "Low")
        data = pre_process_data(data, "Open")
        data = pre_process_data(data, "Close")
        price = data[["Close", "High", "Low", "Open"]]

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
        
        Y_hat_df = pd.DataFrame({'y': predictions})
        
        error = np.array(validation["y"][:-1].values) - np.array(predictions[:-1])
        mean_val_loss = np.abs(error).mean()
        
        last_preds = {
            'instrument': symbol,
            'close_last_price': last_close_price,
            'close_predicted_price': predictions[-1],
            'close_predicted_price_value': last_close_price + (last_close_price * predictions[-1]),
            'close_val_loss': mean_val_loss,
        }
        
        # Repeat similar process for High and Low predictions
        
        # Calculate profits using the functions from loss_utils
        
        results.append({
            'date': simulation_data.index[-1],
            'close': last_close_price,
            'predicted_close': last_preds['close_predicted_price_value'],
            'predicted_high': last_preds.get('high_predicted_price_value', 0),
            'predicted_low': last_preds.get('low_predicted_price_value', 0),
            'entry_takeprofit_profit': last_preds.get('entry_takeprofit_profit', 0),
            'maxdiffprofit_profit': last_preds.get('maxdiffprofit_profit', 0),
            'takeprofit_profit': last_preds.get('takeprofit_profit', 0)
        })

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
        print("Usage: python backtest_test.py <symbol>")
        sys.exit(1)

    symbol = sys.argv[1]
    backtest_forecasts(symbol)
