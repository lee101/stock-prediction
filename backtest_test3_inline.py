import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from src.comparisons import is_buy_side
from src.logging_utils import setup_logging

logger = setup_logging("backtest_test3_inline.log")

from data_curate_daily import download_daily_stock_data, fetch_spread
from disk_cache import disk_cache
from predict_stock_forecasting import load_pipeline, pre_process_data, \
    series_to_tensor
from src.fixtures import crypto_symbols
from scripts.alpaca_cli import set_strategy_for_symbol

SPREAD = 1.0008711461252937


@disk_cache
def cached_predict(context, prediction_length, num_samples, temperature, top_k, top_p):
    global pipeline
    if pipeline is None:
        load_pipeline()
    return pipeline.predict(
        context,
        prediction_length,
        # num_samples=num_samples,
        # temperature=temperature,
        # top_k=top_k,
        # top_p=top_p,
    )


from chronos import BaseChronosPipeline

current_date_formatted = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# test data on same dataset
if __name__ == "__main__":
    current_date_formatted = "2024-12-11-18-22-30"

print(f"current_date_formatted: {current_date_formatted}")

tb_writer = SummaryWriter(log_dir=f"./logs/{current_date_formatted}")

pipeline = None


def load_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = BaseChronosPipeline.from_pretrained(
            # "amazon/chronos-t5-large" if not PAPER else "amazon/chronos-t5-tiny",
            # "amazon/chronos-t5-tiny",
            # "amazon/chronos-t5-large",
            "amazon/chronos-bolt-base",
            device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
            # torch_dtype=torch.bfloat16,
        )
        pipeline.model = pipeline.model.eval()
        # pipeline.model = torch.compile(pipeline.model)


def simple_buy_sell_strategy(predictions, is_crypto=False):
    """Buy if predicted close is up; if not crypto, short if down."""
    predictions = torch.as_tensor(predictions)
    if is_crypto:
        # Prohibit shorts for crypto
        return (predictions > 0).float()
    # Otherwise allow buy (1) or sell (-1)
    return (predictions > 0).float() * 2 - 1


def all_signals_strategy(close_pred, high_pred, low_pred, is_crypto=False):
    """
    Buy if all signals are up; if not crypto, sell if all signals are down, else hold.
    If is_crypto=True, no short trades.
    """
    close_pred, high_pred, low_pred = map(torch.as_tensor, (close_pred, high_pred, low_pred))

    # For "buy" all must be > 0
    buy_signal = (close_pred > 0) & (high_pred > 0) & (low_pred > 0)
    if is_crypto:
        return buy_signal.float()

    # For non-crypto, "sell" all must be < 0
    sell_signal = (close_pred < 0) & (high_pred < 0) & (low_pred < 0)

    # Convert to -1, 0, 1
    return buy_signal.float() - sell_signal.float()


def buy_hold_strategy(predictions):
    """Buy when prediction is positive, hold otherwise."""
    predictions = torch.as_tensor(predictions)
    return (predictions > 0).float()


def unprofit_shutdown_buy_hold(predictions, actual_returns, is_crypto=False):
    """Buy and hold strategy that shuts down if the previous trade would have been unprofitable."""
    predictions = torch.as_tensor(predictions)
    signals = torch.ones_like(predictions)
    for i in range(1, len(signals)):
        if signals[i - 1] != 0.0:
            # Check if day i-1 was correct
            was_correct = (
                    (actual_returns[i - 1] > 0 and predictions[i - 1] > 0) or
                    (actual_returns[i - 1] < 0 and predictions[i - 1] < 0)
            )
            if was_correct:
                # Keep same signal direction as predictions[i]
                signals[i] = 1.0 if predictions[i] > 0 else -1.0 if predictions[i] < 0 else 0.0
            else:
                signals[i] = 0.0
        else:
            # If previously no position, open based on prediction direction
            signals[i] = 1.0 if predictions[i] > 0 else -1.0 if predictions[i] < 0 else 0.0
    # For crypto, replace negative signals with 0
    if is_crypto:
        signals[signals < 0] = 0.0
    return signals


def evaluate_strategy(strategy_signals, actual_returns, trading_fee):
    global SPREAD
    """Evaluate the performance of a strategy, factoring in trading fees."""
    strategy_signals = strategy_signals.numpy()  # Convert to numpy array

    # Calculate fees: apply fee for each trade (both buy and sell)
    # Adjust fees: only apply when position changes
    position_changes = np.diff(np.concatenate(([0], strategy_signals)))
    # Trading fee is the sum of the spread cost and any additional trading fee

    # Pay spread once and trading fee twice per position change
    fees = np.abs(position_changes) * trading_fee + np.abs(position_changes) * abs((1 - SPREAD) / 2)
    # logger.info(f'adjusted fees: {fees}')

    # Adjust fees: only apply when position changes
    for i in range(1, len(fees)):
        if strategy_signals[i] == strategy_signals[i - 1]:
            fees[i] = 0

    # logger.info(f'fees after adjustment: {fees}')

    # Apply fees to the strategy returns
    strategy_returns = strategy_signals * actual_returns - fees

    cumulative_returns = (1 + strategy_returns).cumprod() - 1
    total_return = cumulative_returns.iloc[-1]

    if strategy_returns.std() == 0 or np.isnan(strategy_returns.std()):
        sharpe_ratio = 0  # or some other default value
    else:
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)

    return total_return, sharpe_ratio, strategy_returns


def backtest_forecasts(symbol, num_simulations=100):
    # Download the latest data
    current_time_formatted = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    # use this for testing dataset
    if __name__ == "__main__":
        current_time_formatted = '2024-09-07--03-36-27'
    # current_time_formatted = '2024-04-18--06-14-26'  # new/ 30 minute data # '2022-10-14 09-58-20'
    # current_day_formatted = '2024-04-18'  # new/ 30 minute data # '2022-10-14 09-58-20'

    stock_data = download_daily_stock_data(current_time_formatted, symbols=[symbol])
    # hardcode repeatable time for testing
    # current_time_formatted = "2024-10-18--06-05-32"
    if symbol not in crypto_symbols:
        trading_fee = 0.0002  # near no fee on non crypto? 0.003 per share idk how to calc that though
    #     .0000278 per share plus firna 000166 https://files.alpaca.markets/disclosures/library/BrokFeeSched.pdf
    else:
        trading_fee = 0.0023  # 0.15% fee maker but also .25 taker so avg lets say .23 if we are too aggressive

    # 8% margin lending

    # stock_data = download_daily_stock_data(current_time_formatted, symbols=symbols)
    # stock_data = pd.read_csv(f"./data/{current_time_formatted}/{symbol}-{current_day_formatted}.csv")

    base_dir = Path(__file__).parent
    data_dir = base_dir / "data" / current_time_formatted

    spread = fetch_spread(symbol)
    logger.info(f"spread: {spread}")
    SPREAD = spread  #

    # stock_data = load_stock_data_from_csv(csv_file)

    if len(stock_data) < num_simulations:
        logger.warning(
            f"Not enough historical data for {num_simulations} simulations. Using {len(stock_data)} instead.")
        num_simulations = len(stock_data)

    results = []

    is_crypto = symbol in crypto_symbols

    for sim_idx in range(num_simulations):
        # Change from :-(sim_idx + 1) to :-(sim_idx * 3 + 1) to maintain spacing
        simulation_data = stock_data.iloc[:-(sim_idx * 3 + 1)].copy(deep=True)
        if simulation_data.empty:
            logger.warning(f"No data left for simulation {sim_idx + 1}")
            continue

        result = run_single_simulation(simulation_data, symbol, trading_fee, is_crypto, sim_idx)
        results.append(result)

    # Final iteration: use the entire dataset to get the *very* last forecast
    final_data = stock_data.copy(deep=True)
    final_result = run_single_simulation(final_data, symbol, trading_fee, is_crypto, -1)
    results.append(final_result)

    results_df = pd.DataFrame(results)

    # Log final average metrics
    tb_writer.add_scalar(f'{symbol}/final_metrics/simple_avg_return', results_df['simple_strategy_return'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/simple_avg_sharpe', results_df['simple_strategy_sharpe'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/all_signals_avg_return',
                         results_df['all_signals_strategy_return'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/all_signals_avg_sharpe',
                         results_df['all_signals_strategy_sharpe'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/buy_hold_avg_return', results_df['buy_hold_return'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/buy_hold_avg_sharpe', results_df['buy_hold_sharpe'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/unprofit_shutdown_avg_return',
                         results_df['unprofit_shutdown_return'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/unprofit_shutdown_avg_sharpe',
                         results_df['unprofit_shutdown_sharpe'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/entry_takeprofit_avg_return',
                         results_df['entry_takeprofit_return'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/entry_takeprofit_avg_sharpe',
                         results_df['entry_takeprofit_sharpe'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/highlow_avg_return', results_df['highlow_return'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/highlow_avg_sharpe', results_df['highlow_sharpe'].mean(), 0)

    logger.info(f"\nAverage Validation Losses:")
    logger.info(f"Close Val Loss: {results_df['close_val_loss'].mean():.4f}")
    logger.info(f"High Val Loss: {results_df['high_val_loss'].mean():.4f}") 
    logger.info(f"Low Val Loss: {results_df['low_val_loss'].mean():.4f}")

    logger.info(f"\nBacktest results for {symbol} over {num_simulations} simulations:")
    logger.info(f"Average Simple Strategy Return: {results_df['simple_strategy_return'].mean():.4f}")
    logger.info(f"Average Simple Strategy Sharpe: {results_df['simple_strategy_sharpe'].mean():.4f}")
    logger.info(f"Average Simple Strategy Final Day Return: {results_df['simple_strategy_finalday'].mean():.4f}")
    logger.info(f"Average All Signals Strategy Return: {results_df['all_signals_strategy_return'].mean():.4f}")
    logger.info(f"Average All Signals Strategy Sharpe: {results_df['all_signals_strategy_sharpe'].mean():.4f}")
    logger.info(
        f"Average All Signals Strategy Final Day Return: {results_df['all_signals_strategy_finalday'].mean():.4f}")
    logger.info(f"Average Buy and Hold Return: {results_df['buy_hold_return'].mean():.4f}")
    logger.info(f"Average Buy and Hold Sharpe: {results_df['buy_hold_sharpe'].mean():.4f}")
    logger.info(f"Average Buy and Hold Final Day Return: {results_df['buy_hold_finalday'].mean():.4f}")
    logger.info(f"Average Unprofit Shutdown Buy and Hold Return: {results_df['unprofit_shutdown_return'].mean():.4f}")
    logger.info(f"Average Unprofit Shutdown Buy and Hold Sharpe: {results_df['unprofit_shutdown_sharpe'].mean():.4f}")
    logger.info(
        f"Average Unprofit Shutdown Buy and Hold Final Day Return: {results_df['unprofit_shutdown_finalday'].mean():.4f}")
    logger.info(f"Average Entry+Takeprofit Return: {results_df['entry_takeprofit_return'].mean():.4f}")
    logger.info(f"Average Entry+Takeprofit Sharpe: {results_df['entry_takeprofit_sharpe'].mean():.4f}")
    logger.info(
        f"Average Entry+Takeprofit Final Day Return: {results_df['entry_takeprofit_finalday'].mean():.4f}")
    logger.info(f"Average Highlow Return: {results_df['highlow_return'].mean():.4f}")
    logger.info(f"Average Highlow Sharpe: {results_df['highlow_sharpe'].mean():.4f}")
    logger.info(f"Average Highlow Final Day Return: {results_df['highlow_finalday_return'].mean():.4f}")

    # Determine which strategy is best overall
    avg_simple = results_df["simple_strategy_return"].mean()
    avg_allsignals = results_df["all_signals_strategy_return"].mean()
    avg_takeprofit = results_df["entry_takeprofit_return"].mean()
    avg_highlow = results_df["highlow_return"].mean()

    best_return = max(avg_simple, avg_allsignals, avg_takeprofit, avg_highlow)
    if best_return == avg_highlow:
        best_strategy = "highlow"
    elif best_return == avg_takeprofit:
        best_strategy = "takeprofit"
    elif best_return == avg_allsignals:
        best_strategy = "all_signals"
    else:
        best_strategy = "simple"

    # Record which strategy is best for this symbol & day
    set_strategy_for_symbol(symbol, best_strategy)

    return results_df


def run_single_simulation(simulation_data, symbol, trading_fee, is_crypto, sim_idx):
    last_preds = {
        'instrument': symbol,
        'close_last_price': simulation_data['Close'].iloc[-1],
    }
    # not predicting open because nothing todo with it
    for key_to_predict in ['Close', 'Low', 'High']:  # , 'Open']:
        data = pre_process_data(simulation_data, key_to_predict)
        price = data[["Close", "High", "Low", "Open"]]

        price = price.rename(columns={"Date": "time_idx"})
        price["ds"] = pd.date_range(start="1949-01-01", periods=len(price), freq="D").values
        price['y'] = price[key_to_predict].shift(-1)
        price['trade_weight'] = (price["y"] > 0) * 2 - 1

        # Only drop last row if NOT final simulation
        if sim_idx != -1:  # <-- This is the key conditional
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

        # Log validation metrics
        tb_writer.add_scalar(f'{symbol}/{key_to_predict}/val_loss', mean_val_loss, sim_idx)

        # if __name__ == "__main__":
        #     print(f"mean_val_loss: {mean_val_loss}")

        last_preds[key_to_predict.lower() + "_last_price"] = simulation_data[key_to_predict].iloc[-1]
        last_preds[key_to_predict.lower() + "_predicted_price"] = predictions[-1]
        last_preds[key_to_predict.lower() + "_predicted_price_value"] = last_preds[
                                                                            key_to_predict.lower() + "_last_price"] + (
                                                                                last_preds[
                                                                                    key_to_predict.lower() + "_last_price"] *
                                                                                predictions[-1])
        last_preds[key_to_predict.lower() + "_val_loss"] = mean_val_loss
        last_preds[key_to_predict.lower() + "_actual_movement_values"] = actuals[:-1].view(-1)
        last_preds[key_to_predict.lower() + "_trade_values"] = trading_preds.view(-1)
        last_preds[key_to_predict.lower() + "_predictions"] = predictions[:-1].view(-1)

    # Calculate actual returns
    actual_returns = pd.Series(last_preds["close_actual_movement_values"].numpy())

    # Simple buy/sell strategy
    simple_signals = simple_buy_sell_strategy(
        last_preds["close_predictions"],
        is_crypto=is_crypto
    )
    simple_total_return, simple_sharpe, simple_returns = evaluate_strategy(simple_signals, actual_returns,
                                                                           trading_fee)
    simple_finalday_return = (simple_signals[-1].item() * actual_returns.iloc[-1]) - (2 * trading_fee * SPREAD)

    # All signals strategy
    all_signals = all_signals_strategy(
        last_preds["close_predictions"],
        last_preds["high_predictions"],
        last_preds["low_predictions"],
        is_crypto=is_crypto
    )
    all_signals_total_return, all_signals_sharpe, all_signals_returns = evaluate_strategy(all_signals,
                                                                                          actual_returns,
                                                                                          trading_fee)
    all_signals_finalday_return = (all_signals[-1].item() * actual_returns.iloc[-1]) - (2 * trading_fee * SPREAD)

    # Buy and hold strategy
    buy_hold_signals = buy_hold_strategy(last_preds["close_predictions"])
    buy_hold_return, buy_hold_sharpe, buy_hold_returns = evaluate_strategy(buy_hold_signals, actual_returns,
                                                                           trading_fee)
    buy_hold_finalday_return = actual_returns.iloc[-1] - (2 * trading_fee * SPREAD)

    # Unprofit shutdown buy and hold strategy
    unprofit_shutdown_signals = unprofit_shutdown_buy_hold(last_preds["close_predictions"], actual_returns,
                                                           is_crypto=is_crypto)
    unprofit_shutdown_return, unprofit_shutdown_sharpe, unprofit_shutdown_returns = evaluate_strategy(
        unprofit_shutdown_signals,
        actual_returns, trading_fee)
    unprofit_shutdown_finalday_return = (unprofit_shutdown_signals[-1].item() * actual_returns.iloc[-1]) - (
        2 * trading_fee * SPREAD if unprofit_shutdown_signals[-1].item() != 0 else 0)

    # Entry+takeprofit strategy
    entry_takeprofit_return, entry_takeprofit_sharpe, entry_takeprofit_returns = evaluate_entry_takeprofit_strategy(
        last_preds["close_predictions"],
        last_preds["high_predictions"],
        last_preds["low_predictions"],
        last_preds["close_actual_movement_values"],
        last_preds["high_actual_movement_values"],
        last_preds["low_actual_movement_values"],
        trading_fee
    )
    entry_takeprofit_finalday_return = entry_takeprofit_return / len(actual_returns)

    # Highlow strategy
    highlow_return, highlow_sharpe, highlow_returns = evaluate_highlow_strategy(
        last_preds["close_predictions"],
        last_preds["high_predictions"],
        last_preds["low_predictions"],
        last_preds["close_actual_movement_values"],
        last_preds["high_actual_movement_values"],
        last_preds["low_actual_movement_values"],
        trading_fee,
        is_crypto=is_crypto
    )
    highlow_finalday_return = highlow_return / len(actual_returns)

    # Log strategy metrics to tensorboard
    tb_writer.add_scalar(f'{symbol}/strategies/simple/total_return', simple_total_return, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/simple/sharpe', simple_sharpe, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/simple/finalday', simple_finalday_return, sim_idx)

    tb_writer.add_scalar(f'{symbol}/strategies/all_signals/total_return', all_signals_total_return, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/all_signals/sharpe', all_signals_sharpe, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/all_signals/finalday', all_signals_finalday_return, sim_idx)

    tb_writer.add_scalar(f'{symbol}/strategies/buy_hold/total_return', buy_hold_return, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/buy_hold/sharpe', buy_hold_sharpe, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/buy_hold/finalday', buy_hold_finalday_return, sim_idx)

    tb_writer.add_scalar(f'{symbol}/strategies/unprofit_shutdown/total_return', unprofit_shutdown_return, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/unprofit_shutdown/sharpe', unprofit_shutdown_sharpe, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/unprofit_shutdown/finalday', unprofit_shutdown_finalday_return, sim_idx)

    tb_writer.add_scalar(f'{symbol}/strategies/entry_takeprofit/total_return', entry_takeprofit_return, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/entry_takeprofit/sharpe', entry_takeprofit_sharpe, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/entry_takeprofit/finalday', entry_takeprofit_finalday_return, sim_idx)

    tb_writer.add_scalar(f'{symbol}/strategies/highlow/total_return', highlow_return, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/highlow/sharpe', highlow_sharpe, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/highlow/finalday', highlow_finalday_return, sim_idx)

    # Log returns over time
    for t, ret in enumerate(simple_returns):
        tb_writer.add_scalar(f'{symbol}/returns_over_time/simple', ret, t)
    for t, ret in enumerate(all_signals_returns):
        tb_writer.add_scalar(f'{symbol}/returns_over_time/all_signals', ret, t)
    for t, ret in enumerate(buy_hold_returns):
        tb_writer.add_scalar(f'{symbol}/returns_over_time/buy_hold', ret, t)
    for t, ret in enumerate(unprofit_shutdown_returns):
        tb_writer.add_scalar(f'{symbol}/returns_over_time/unprofit_shutdown', ret, t)
    for t, ret in enumerate(entry_takeprofit_returns):
        tb_writer.add_scalar(f'{symbol}/returns_over_time/entry_takeprofit', ret, t)
    for t, ret in enumerate(highlow_returns):
        tb_writer.add_scalar(f'{symbol}/returns_over_time/highlow', ret, t)

    # print(last_preds)
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
        'unprofit_shutdown_finalday': float(unprofit_shutdown_finalday_return),
        'entry_takeprofit_return': float(entry_takeprofit_return),
        'entry_takeprofit_sharpe': float(entry_takeprofit_sharpe),
        'entry_takeprofit_finalday': float(entry_takeprofit_finalday_return),
        'highlow_return': float(highlow_return),
        'highlow_sharpe': float(highlow_sharpe),
        'highlow_finalday_return': float(highlow_finalday_return),
        'close_val_loss': float(last_preds['close_val_loss']),
        'high_val_loss': float(last_preds['high_val_loss']),
        'low_val_loss': float(last_preds['low_val_loss']),
    }

    return result


def evaluate_entry_takeprofit_strategy(
        close_predictions, high_predictions, low_predictions,
        actual_close, actual_high, actual_low,
        trading_fee
):
    """
    Evaluates an entry+takeprofit approach with minimal repeated fees:
      - If close_predictions[idx] > 0 => 'buy'
        - Exit when actual_high >= high_predictions[idx], else exit at actual_close.
      - If close_predictions[idx] < 0 => 'short'
        - Exit when actual_low <= low_predictions[idx], else exit at actual_close.
      - If we remain in the same side as previous day, don't pay another opening fee.
    """

    daily_returns = []
    last_side = None  # track "buy" or "short" from previous day

    for idx in range(len(close_predictions)):
        # determine side
        is_buy = bool(close_predictions[idx] > 0)
        new_side = "buy" if is_buy else "short"

        # if same side as previous day, we are continuing
        continuing_same_side = (last_side == new_side)

        # figure out exit
        if is_buy:
            if actual_high[idx] >= high_predictions[idx]:
                daily_return = high_predictions[idx]  # approximate from 0 to predicted high
            else:
                daily_return = actual_close[idx]
        else:  # short
            if actual_low[idx] <= low_predictions[idx]:
                daily_return = 0 - low_predictions[idx]  # from 0 down to predicted_low
            else:
                daily_return = 0 - actual_close[idx]

        # fees: if it's the first day with new_side, pay one side of the fee
        # if we exit from the previous day (different side or last_side == None?), pay closing fee
        fee_to_charge = 0.0

        # if we changed sides or last_side is None, we pay open fee
        if not continuing_same_side:
            fee_to_charge += trading_fee  # opening fee
            if last_side is not None:
                fee_to_charge += trading_fee  # closing fee for old side

        # apply total fee
        daily_return -= fee_to_charge
        daily_returns.append(daily_return)

        last_side = new_side

    daily_returns = np.array(daily_returns, dtype=float)
    total_return = float(daily_returns.sum())
    if daily_returns.std() == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))

    return total_return, sharpe_ratio, daily_returns


def evaluate_highlow_strategy(
        close_predictions,
        high_predictions,
        low_predictions,
        actual_close,
        actual_high,
        actual_low,
        trading_fee,
        is_crypto=False
):
    """
    Evaluates a 'highlow' approach:
    - If close_predictions[idx] > 0 => attempt a 'buy' at predicted_low, else skip.
    - If is_crypto=False and close_predictions[idx] < 0 => attempt short at predicted_high, else skip.
    - Either way, exit at actual_close by day's end.
    """
    daily_returns = []
    last_side = None  # track "buy"/"short" from previous day

    for idx in range(len(close_predictions)):
        cp = close_predictions[idx]
        if cp > 0:
            # Attempt buy at predicted_low if actual_low <= predicted_low, else buy at actual_close
            entry = low_predictions[idx] if actual_low[idx] <= low_predictions[idx] else actual_close[idx]
            exit_price = actual_close[idx]
            new_side = "buy"
        elif (not is_crypto) and (cp < 0):
            # Attempt short if not crypto
            entry = high_predictions[idx] if actual_high[idx] >= high_predictions[idx] else actual_close[idx]
            # Gains from short are entry - final
            exit_price = actual_close[idx]
            new_side = "short"
        else:
            # Skip if crypto and cp < 0 (no short), or cp == 0
            daily_returns.append(0.0)
            last_side = None
            continue

        # Calculate daily gain
        if is_buy_side(new_side):
            daily_gain = exit_price - entry
        else:
            # short
            daily_gain = entry - exit_price

        # Fees: open if side changed or if None, close prior side if it existed
        fee_to_charge = 0.0
        if new_side != last_side:
            fee_to_charge += trading_fee  # open
            if last_side is not None:
                fee_to_charge += trading_fee  # close old side

        daily_gain -= fee_to_charge
        daily_returns.append(daily_gain)
        last_side = new_side

    daily_returns = np.array(daily_returns, dtype=float)
    total_return = daily_returns.sum()
    if daily_returns.std() == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    return float(total_return), float(sharpe_ratio), daily_returns


if __name__ == "__main__":
    if len(sys.argv) != 2:
        symbol = "ETHUSD"
        print("Usage: python backtest_test.py <symbol> defaultint to eth")
    else:
        symbol = sys.argv[1]

    # backtest_forecasts("NVDA")
    backtest_forecasts(symbol)
    # backtest_forecasts("UNIUSD")
    # backtest_forecasts("AAPL")
    # backtest_forecasts("GOOG")
