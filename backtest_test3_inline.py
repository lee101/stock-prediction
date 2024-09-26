import functools
import hashlib
import os
import pickle
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

ETH_SPREAD = 1.0008711461252937
CRYPTO_TRADING_FEE = 0.0015  # 0.15% fee


def disk_cache(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if we're in testing mode
        if os.environ.get('TESTING') == 'True':
            return func(*args, **kwargs)
            
        # Create a unique key based on the function arguments
        key_parts = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                key_parts.append(hashlib.md5(arg.numpy().tobytes()).hexdigest())
            else:
                key_parts.append(str(arg))
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                key_parts.append(f"{k}:{hashlib.md5(v.numpy().tobytes()).hexdigest()}")
            else:
                key_parts.append(f"{k}:{v}")
        
        key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
        cache_dir = os.path.join(os.path.dirname(__file__), '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'{func.__name__}_{key}.pkl')

        # Check if the result is already cached
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        # If not cached, call the function and cache the result
        result = func(*args, **kwargs)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

        return result
    return wrapper


@disk_cache
def cached_predict(context, prediction_length, num_samples, temperature, top_k, top_p):
    global pipeline
    if pipeline is None:
        load_pipeline()
    return pipeline.predict(
        context,
        prediction_length,
        num_samples=num_samples,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

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


def simple_buy_sell_strategy(predictions):
    """Buy if predicted close is up, sell if down."""
    predictions = torch.as_tensor(predictions)
    return (predictions > 0).float() * 2 - 1

def all_signals_strategy(close_pred, high_pred, low_pred, open_pred):
    """Buy if all signals are up, sell if all are down, hold otherwise."""
    close_pred, high_pred, low_pred, open_pred = map(torch.as_tensor, (close_pred, high_pred, low_pred, open_pred))
    buy_signal = (close_pred > 0) & (high_pred > 0) & (low_pred > 0) & (open_pred > 0)
    sell_signal = (close_pred < 0) & (high_pred < 0) & (low_pred < 0) & (open_pred < 0)
    return buy_signal.float() - sell_signal.float()

def buy_hold_strategy(predictions):
    """Always buy and hold strategy."""
    return torch.ones_like(torch.as_tensor(predictions))

def unprofit_shutdown_buy_hold(predictions, actual_returns):
    """Buy and hold strategy that shuts down if the previous trade would have been unprofitable."""
    signals = torch.ones_like(torch.as_tensor(predictions))
    for i in range(1, len(signals)):
        if actual_returns[i-1] <= 0:
            signals[i:] = 0
            break
    return signals

def evaluate_strategy(strategy_signals, actual_returns):
    """Evaluate the performance of a strategy, factoring in trading fees."""
    strategy_signals = strategy_signals.numpy()  # Convert to numpy array
    
    # Calculate fees: apply fee for each trade (both buy and sell)
    fees = np.abs(np.diff(np.concatenate(([0], strategy_signals)))) * CRYPTO_TRADING_FEE
    
    # Apply fees to the strategy returns
    strategy_returns = strategy_signals * actual_returns - fees
    
    cumulative_returns = (1 + strategy_returns).cumprod() - 1
    total_return = cumulative_returns.iloc[-1]
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)  # Assuming daily data
    return total_return, sharpe_ratio

def backtest_forecasts(symbol, num_simulations=20):
    logger.remove()
    logger.add(sys.stdout, format="{time} | {level} | {message}")

    # Download the latest data
    current_time_formatted = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    # use this for testing dataset
    current_time_formatted = '2024-04-18--06-14-26'  # new/ 30 minute data # '2022-10-14 09-58-20'

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
                forecast = cached_predict(
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
            last_preds[key_to_predict.lower() + "_predicted_price"] = predictions[-1]
            last_preds[key_to_predict.lower() + "_predicted_price_value"] = last_preds[key_to_predict.lower() + "_last_price"] + (
                    last_preds[key_to_predict.lower() + "_last_price"] * predictions[-1])
            last_preds[key_to_predict.lower() + "_val_loss"] = mean_val_loss
            last_preds[key_to_predict.lower() + "_actual_movement_values"] = actuals[:-1].view(-1)
            last_preds[key_to_predict.lower() + "_trade_values"] = trading_preds.view(-1)
            last_preds[key_to_predict.lower() + "_predictions"] = predictions[:-1].view(-1)

        # Calculate actual returns
        actual_returns = pd.Series(last_preds["close_actual_movement_values"].numpy())

        # Simple buy/sell strategy
        simple_signals = simple_buy_sell_strategy(last_preds["close_predictions"])
        simple_total_return, simple_sharpe = evaluate_strategy(simple_signals, actual_returns)
        simple_finalday_return = (simple_signals[-1].item() * actual_returns.iloc[-1]) - CRYPTO_TRADING_FEE

        # All signals strategy
        all_signals = all_signals_strategy(
            last_preds["close_predictions"],
            last_preds["high_predictions"],
            last_preds["low_predictions"],
            last_preds["open_predictions"]
        )
        all_signals_total_return, all_signals_sharpe = evaluate_strategy(all_signals, actual_returns)
        all_signals_finalday_return = (all_signals[-1].item() * actual_returns.iloc[-1]) - CRYPTO_TRADING_FEE

        # Buy and hold strategy
        buy_hold_signals = buy_hold_strategy(last_preds["close_predictions"])
        buy_hold_return, buy_hold_sharpe = evaluate_strategy(buy_hold_signals, actual_returns)
        buy_hold_finalday_return = actual_returns.iloc[-1] - CRYPTO_TRADING_FEE

        # Unprofit shutdown buy and hold strategy
        unprofit_shutdown_signals = unprofit_shutdown_buy_hold(last_preds["close_predictions"], actual_returns)
        unprofit_shutdown_return, unprofit_shutdown_sharpe = evaluate_strategy(unprofit_shutdown_signals, actual_returns)
        unprofit_shutdown_finalday_return = (unprofit_shutdown_signals[-1].item() * actual_returns.iloc[-1]) - (CRYPTO_TRADING_FEE if unprofit_shutdown_signals[-1].item() != 0 else 0)

        result = {
            'date': simulation_data.index[-1],
            'close': float(last_preds['close_last_price']),
            'predicted_close': float(last_preds['close_predicted_price_value']),
            'predicted_high': float(last_preds['high_predicted_price_value']),
            'predicted_low': float(last_preds['low_predicted_price_value']),
            'simple_strategy_return': float(simple_total_return),
            'simple_strategy_sharpe': float(simple_sharpe),
            'simple_strategy_finalday': float(simple_finalday_return),
            'all_signals_strategy_return': float(all_signals_total_return),
            'all_signals_strategy_sharpe': float(all_signals_sharpe),
            'all_signals_strategy_finalday': float(all_signals_finalday_return),
            'buy_hold_return': float(buy_hold_return),
            'buy_hold_sharpe': float(buy_hold_sharpe),
            'buy_hold_finalday': float(buy_hold_finalday_return),
            'unprofit_shutdown_return': float(unprofit_shutdown_return),
            'unprofit_shutdown_sharpe': float(unprofit_shutdown_sharpe),
            'unprofit_shutdown_finalday': float(unprofit_shutdown_finalday_return)
        }
        results.append(result)
        print("Result:")
        print(result)

    results_df = pd.DataFrame(results)

    logger.info(f"\nBacktest results for {symbol} over {num_simulations} simulations:")
    logger.info(f"Average Simple Strategy Return: {results_df['simple_strategy_return'].mean():.4f}")
    logger.info(f"Average Simple Strategy Sharpe: {results_df['simple_strategy_sharpe'].mean():.4f}")
    logger.info(f"Average Simple Strategy Final Day Return: {results_df['simple_strategy_finalday'].mean():.4f}")
    logger.info(f"Average All Signals Strategy Return: {results_df['all_signals_strategy_return'].mean():.4f}")
    logger.info(f"Average All Signals Strategy Sharpe: {results_df['all_signals_strategy_sharpe'].mean():.4f}")
    logger.info(f"Average All Signals Strategy Final Day Return: {results_df['all_signals_strategy_finalday'].mean():.4f}")
    logger.info(f"Average Buy and Hold Return: {results_df['buy_hold_return'].mean():.4f}")
    logger.info(f"Average Buy and Hold Sharpe: {results_df['buy_hold_sharpe'].mean():.4f}")
    logger.info(f"Average Buy and Hold Final Day Return: {results_df['buy_hold_finalday'].mean():.4f}")
    logger.info(f"Average Unprofit Shutdown Buy and Hold Return: {results_df['unprofit_shutdown_return'].mean():.4f}")
    logger.info(f"Average Unprofit Shutdown Buy and Hold Sharpe: {results_df['unprofit_shutdown_sharpe'].mean():.4f}")
    logger.info(f"Average Unprofit Shutdown Buy and Hold Final Day Return: {results_df['unprofit_shutdown_finalday'].mean():.4f}")

    return results_df

if __name__ == "__main__":
    if len(sys.argv) != 2:
        symbol = "ETHUSD"
        print("Usage: python backtest_test.py <symbol> defaultint to eth")
    else:
        symbol = sys.argv[1]

    backtest_forecasts(symbol)
