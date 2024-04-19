import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS

import alpaca_wrapper
from data_utils import split_data
from loss_utils import calculate_trading_profit_torch, DEVICE, get_trading_profits_list, percent_movements_augment, \
    calculate_trading_profit_torch_buy_only, \
    calculate_trading_profit_torch_with_buysell, calculate_trading_profit_torch_with_entry_buysell, \
    calculate_trading_profit_torch_with_buysell_profit_values, calculate_profit_torch_with_entry_buysell_profit_values
from model import GRU
from src.conversion_utils import unwrap_tensor
from src.fixtures import crypto_symbols


transformers.set_seed(42)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from loguru import logger as loguru_logger

base_dir = Path(__file__).parent
loguru_logger.info(base_dir)
from sklearn.preprocessing import MinMaxScaler

from torch.utils.tensorboard import SummaryWriter

from chronos import ChronosPipeline

current_date_formatted = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
tb_writer = SummaryWriter(log_dir=f"./logs/{current_date_formatted}")

pipeline = None
def load_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = ChronosPipeline.from_pretrained(
            # "amazon/chronos-t5-large",
            "amazon/chronos-t5-tiny",
            device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
            torch_dtype=torch.bfloat16,
        )

def load_stock_data_from_csv(csv_file_path: Path):
    """
    Loads stock data from csv file.
    """
    csv = pd.read_csv(csv_file_path)
    # rename columns to capitalized first letters
    csv.columns = [col.title() for col in csv.columns]
    return csv


def train_test_split(stock_data: pd.DataFrame, test_size=50):
    """
    Splits stock data into train and test sets.
    test_size : int, number of examples be used for test set.
    """
    x_train = stock_data.iloc[:-test_size]
    x_test = stock_data.iloc[-test_size:]
    return x_train, x_test


scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit_transform([[-1, 1]])


def pre_process_data(x_train, key_to_predict):
    # drop useless data
    # x_train = x_train.drop(columns=["Volume",
    #                                 "Ex - Dividend",
    #
    #                                 "Split",
    #                                 "Ratio",
    #                                 "Adj.Open",
    #                                 "Adj.High",
    #                                 "Adj.Low",
    #                                 "Adj.Close",
    #                                 "Adj.Volume",
    #                                 ])
    newdata = x_train.copy()
    newdata[key_to_predict] = percent_movements_augment(x_train[key_to_predict].values.reshape(-1, 1))

    return newdata


pl.seed_everything(42)
torch.autograd.set_detect_anomaly(True)


def series_to_tensor(series_pd):
    return torch.tensor(series_pd.values, dtype=torch.float)  # todo gpu, device=DEVICE)


def series_to_df(series_pd):
    return pd.DataFrame(series_pd.values, columns=series_pd.columns)


def make_predictions(input_data_path=None, pred_name='', retrain=False, alpaca_wrapper=None):
    """
    Make predictions for all csv files in directory.
    """
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_file_name = results_dir / f"predictions-{time}.csv"
    

    headers_written = False

    total_val_loss = 0
    total_forecasted_profit = 0

    if input_data_path:
        input_data_files = base_dir / "data" / input_data_path
    else:
        input_data_files = base_dir / "data"
    loguru_logger.info(f"input_data_files {input_data_files}")

    csv_files = list(input_data_files.glob("*.csv"))
    alpaca_clock = alpaca_wrapper.get_clock()
    for days_to_drop in [0]:  # [1,2,3,4,5,6,7,8,9,10,11]:
        for csv_file in csv_files:
            instrument_name = csv_file.stem.split('-')[0]
            # only trade crypto or stocks currently being traded - dont bother forecasting things that cant be traded.
            if not alpaca_clock.is_open:
                # remove all stock pairs but not crypto
                if instrument_name not in crypto_symbols:
                    continue
            last_preds = {
                "instrument": instrument_name,
            }
            key_to_predict = "Close"
            training_mode = "predict"
            for key_to_predict in [
                "Close",
                'Low',
                'High',
            ]:  # , 'TakeProfit', 'StopLoss']:
                stock_data = load_stock_data_from_csv(csv_file)
                stock_data = stock_data.dropna()
                if stock_data.empty:
                    loguru_logger.info(f"Empty data for {instrument_name}")
                    continue
                # drop last days_to_drop rows
                if days_to_drop:
                    stock_data = stock_data.iloc[:-days_to_drop]

                # x_train, x_test = train_test_split(stock_data)
                last_close_price = stock_data[key_to_predict].iloc[-1]
                data = stock_data.copy()
                data = pre_process_data(data, "High")
                # todo scaler for each, this messes up the scaler
                data = pre_process_data(data, "Low")
                data = pre_process_data(data, "Open")
                data = pre_process_data(data, "Close")
                price = data[["Close", "High", "Low", "Open"]]

                
                price = price.rename(columns={"Date": "time_idx"})
                # not actually important what date it thinks as long as its daily i think
                price["ds"] = pd.date_range(start="1949-01-01", periods=len(price), freq="D").values
                price['y'] = price[key_to_predict].shift(-1)
                price['trade_weight'] = (price["y"] > 0) * 2 - 1

                price.drop(price.tail(1).index, inplace=True)  # drop last row because of percent change augmentation

                price['id'] = price.index
                # add unique_id column to price dataframe (is actually constant so not unique :/ )
                price['unique_id'] = 1
                # drop nan values
                price = price.dropna()
                # final_pred_to_predict = price.tail(1)
                # price.drop(final_pred_to_predict.index, inplace=True)  # drop last row because of percent change augmentation

                # target_to_pred = "y"
                training = price[:-7]
                validation = price[-7:]
                Y_train_df = training
                Y_test_df = validation

                if Y_train_df.empty:
                    loguru_logger.info(f"No training data for {instrument_name}")
                    continue
               
                load_pipeline()
                predictions = []
                # make 7 predictions - todo can batch this all in 1 go
                for pred_idx in reversed(range(1, 8)):
                    
                    current_context = price[:-pred_idx]
                    context = torch.tensor(current_context["y"].values, dtype=torch.float)

                    prediction_length = 1
                    forecast = pipeline.predict(
                        context,
                        prediction_length,
                        num_samples=10,
                        temperature=1.0,
                        top_k=4000,
                        top_p=1.0,
                    )
                    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0) # todo use spread?
                    predictions.append(median.item())
                Y_hat_df = pd.DataFrame({'y': predictions})

                # Y_hat_df = Y_test_df.merge(Y_hat_df, how='left', on=['unique_id', 'ds'])
                # actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
                # baseline_predictions = Baseline().predict(val_dataloader)
                # loguru_logger.info((actuals[:-1] - baseline_predictions[:-1]).abs().mean().item())

                # new prediction based on model
                # actual_list = Y_hat_df['y']  # TODO check this y center transform prints feature neamses warning

                # error_nhits = Y_hat_df['y'] - Y_hat_df['NHITS']
                # error_nbeats = Y_hat_df['y'] - Y_hat_df['NBEATS']
                # lowest_error = None
                # if error_nhits.abs().sum() < error_nbeats.abs().sum():
                #     lowest_error = 'NHITS'
                #     loguru_logger.info(f"Using {lowest_error} as lowest error from nhits")
                # else:
                #     lowest_error = 'NBEATS'
                #     loguru_logger.info(f"Using {lowest_error} as lowest error from nbeats")
                # Y_hat_df['error'] = Y_hat_df['y'] - Y_hat_df[lowest_error]

                # predictions = Y_hat_df["y"]
                error = np.array(validation["y"][:-1].values) - np.array(predictions[:-1]) # last one is not predicted to be anything so not the loss
                mean_val_loss = np.abs(error).mean()
                # loguru_logger.info(f"predictions {predictions} ")
                # loguru_logger.info(f"actuals {validation["y"][:-1]}")
                loguru_logger.info(f"Using {mean_val_loss} as lowest error from chronos")
                # if not added_best_params:
                #     # find best hyperparams
                #     from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
                #
                #     # create study
                #     study = optimize_hyperparameters(
                #         train_dataloader,
                #         val_dataloader,
                #         model_path="optuna_test", # saves over the same place for all models to avoid memory issues
                #         n_trials=100,
                #         max_epochs=200,
                #         gradient_clip_val_range=(0.01, 1.0),
                #         hidden_size_range=(8, 128),
                #         hidden_continuous_size_range=(8, 128),
                #         attention_head_size_range=(1, 4),
                #         learning_rate_range=(0.001, 0.1),
                #         dropout_range=(0.1, 0.3),
                #         trainer_kwargs=dict(limit_train_batches=30),
                #         reduce_on_plateau_patience=4,
                #         use_learning_rate_finder=False,
                #         # use Optuna to find ideal learning rate or use in-built learning rate finder
                #     )
                #
                #     # save study results - also we can resume tuning at a later point in time
                #     with open(best_hyperparams_save_file, "wb") as fout:
                #         pickle.dump(study, fout)
                #
                #     # show best hyperparameters
                #     loguru_logger.info(study.best_trial.params)

                # loguru_logger.info(f"mean val loss:${mean_val_loss}")

                # raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)
                # for idx in range(1):  # plot 10 examples
                #     plot = best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)
                ## display plot
                # note last one at zero actual movement is not true
                # plot.show()
                predictions = torch.tensor(predictions)
                actuals = series_to_tensor(validation["y"])
                # predict trade if last value is above the prediction
                trading_preds = (predictions[:-1] > 0) * 2 - 1
                # last_values = x_test[:, -1, 0]
                calculated_profit = calculate_trading_profit_torch(scaler, None, actuals[:-1], trading_preds).item()

                calculated_profit_buy_only = calculate_trading_profit_torch_buy_only(scaler, None, actuals[:-1],
                                                                                     trading_preds).item()
                calculated_profit_values = get_trading_profits_list(scaler, None, actuals[:-1], trading_preds)
                #

                val_loss = mean_val_loss
                # percent_movement = (
                #     predictions[-1].item() - last_close_price
                # ) / last_close_price
                last_preds[key_to_predict.lower() + "_last_price"] = last_close_price
                last_preds[key_to_predict.lower() + "_predicted_price"] = unwrap_tensor(predictions[
                    -1
                ])
                last_preds[key_to_predict.lower() + "_predicted_price_value"] = unwrap_tensor(last_close_price + (
                            last_close_price * predictions[
                        -1
                    ]))
                last_preds[key_to_predict.lower() + "_val_loss"] = val_loss
                last_preds[key_to_predict.lower() + "min_loss_trading_profit"] = calculated_profit
                last_preds[key_to_predict.lower() + "min_loss_buy_only_trading_profit"] = calculated_profit_buy_only
                last_preds[key_to_predict.lower() + "_actual_movement_values"] = actuals[:-1].view(-1)
                last_preds[key_to_predict.lower() + "_calculated_profit_values"] = list(
                    calculated_profit_values.view(-1).detach().cpu().numpy())
                last_preds[key_to_predict.lower() + "_trade_values"] = trading_preds.view(-1)
                last_preds[key_to_predict.lower() + "_predictions"] = predictions[:-1].view(-1)
                # last_preds[key_to_predict.lower() + "_percent_movement"] = percent_movement
                # last_preds[
                #     key_to_predict.lower() + "_likely_percent_uncertainty"
                # ] = likely_percent_uncertainty
                # last_preds[key_to_predict.lower() + "_minus_uncertainty"] = (
                #     percent_movement - likely_percent_uncertainty
                # )
                total_val_loss += val_loss
                total_forecasted_profit += calculated_profit

            ##### Now train other network to predict buy or sell leverage
            # todo use the same zero shot forecaster for this - also for other stuff

            # key_to_predict = "Close"
            

            # # compute loss when
            # calculate_trading_profit_torch_with_buysell()
            #
            # trading_preds = (predictions[:-1] > 0) * 2 - 1
            # last_values = x_test[:, -1, 0]
            # compute movement to high price
            validation_size = last_preds[
                "high_actual_movement_values"].numel()
            close_to_high = series_to_tensor(
                abs(1 - (stock_data["High"].iloc[-validation_size - 2:-2] / stock_data["Close"].iloc[
                                                                            -validation_size - 2:-2])))
            close_to_low = series_to_tensor(abs(1 - (stock_data["Low"].iloc[
                                                     -validation_size - 2:-2] / stock_data["Close"].iloc[
                                                                                -validation_size - 2:-2])))
            calculated_profit = calculate_trading_profit_torch_with_buysell(scaler, None,
                                                                            last_preds["close_actual_movement_values"],
                                                                            last_preds["close_trade_values"],
                                                                            last_preds[
                                                                                "high_actual_movement_values"] + close_to_high,
                                                                            last_preds[
                                                                                "high_predictions"] + close_to_high,
                                                                            last_preds[
                                                                                "low_actual_movement_values"] - close_to_low,
                                                                            last_preds[
                                                                                "low_predictions"] - close_to_low,

                                                                            ).item()
            loguru_logger.info(f"{instrument_name} calculated_profit: {calculated_profit}")
            last_preds['takeprofit_profit'] = calculated_profit
            last_preds['takeprofit_profit_values'] = list(calculate_trading_profit_torch_with_buysell_profit_values(
                last_preds[
                    "close_actual_movement_values"],
                last_preds[
                    "close_trade_values"],
                last_preds[
                    "high_actual_movement_values"] + close_to_high,
                last_preds[
                    "high_predictions"] + close_to_high,
                last_preds[
                    "low_actual_movement_values"] - close_to_low,
                last_preds[
                    "low_predictions"] - close_to_low,

            ).detach().cpu().numpy())

            # todo margin allocation tests
            current_profit = calculated_profit
            max_profit = float('-Inf')
            for buy_take_profit_multiplier in np.linspace(-.03, .03, 500):
                calculated_profit = calculate_trading_profit_torch_with_buysell(scaler, None,
                                                                                last_preds[
                                                                                    "close_actual_movement_values"],
                                                                                last_preds["close_trade_values"],
                                                                                last_preds[
                                                                                    "high_actual_movement_values"] + close_to_high,
                                                                                last_preds[
                                                                                    "high_predictions"] + close_to_high + buy_take_profit_multiplier,
                                                                                last_preds[
                                                                                    "low_actual_movement_values"] - close_to_low,
                                                                                last_preds[
                                                                                    "low_predictions"] - close_to_low,
                                                                                ).item()
                if calculated_profit > max_profit:
                    max_profit = calculated_profit
                    last_preds['takeprofit_profit_high_multiplier'] = buy_take_profit_multiplier
                    last_preds['takeprofit_high_profit'] = max_profit # high profit cant be trusted because of training the multiplier on valid data
                    # loguru_logger.info(f"{instrument_name} buy_take_profit_multiplier: {buy_take_profit_multiplier} calculated_profit: {calculated_profit}")

            max_profit = float('-Inf')
            for low_take_profit_multiplier in np.linspace(-.03, .03, 500):
                calculated_profit = calculate_trading_profit_torch_with_buysell(scaler, None,
                                                                                last_preds[
                                                                                    "close_actual_movement_values"],
                                                                                last_preds["close_trade_values"],
                                                                                last_preds[
                                                                                    "high_actual_movement_values"] + close_to_high,
                                                                                last_preds[
                                                                                    "high_predictions"] + close_to_high +
                                                                                last_preds[
                                                                                    'takeprofit_profit_high_multiplier'],
                                                                                last_preds[
                                                                                    "low_actual_movement_values"] - close_to_low,
                                                                                last_preds[
                                                                                    "low_predictions"] - close_to_low + low_take_profit_multiplier,
                                                                                ).item()
                if calculated_profit > max_profit:
                    max_profit = calculated_profit
                    last_preds['takeprofit_profit_low_multiplier'] = low_take_profit_multiplier
                    last_preds['takeprofit_low_profit'] = max_profit
                    # loguru_logger.info(f"{instrument_name} low_take_profit_multiplier: {low_take_profit_multiplier} calculated_profit: {calculated_profit}")

            # with maxdiff low or high
            high_diffs = torch.abs(last_preds[
                                       "high_predictions"] + close_to_high)
            low_diffs = torch.abs(last_preds[
                                      "low_predictions"] - close_to_low)
            maxdiff_trades = (high_diffs > low_diffs) * 2 - 1
            calculated_profit = calculate_trading_profit_torch_with_entry_buysell(scaler, None,
                                                                                  last_preds[
                                                                                      "close_actual_movement_values"],
                                                                                  maxdiff_trades,
                                                                                  last_preds[
                                                                                      "high_actual_movement_values"] + close_to_high,
                                                                                  last_preds[
                                                                                      "high_predictions"] + close_to_high,
                                                                                  last_preds[
                                                                                      "low_actual_movement_values"] - close_to_low,
                                                                                  last_preds[
                                                                                      "low_predictions"] - close_to_low,

                                                                                  ).item()
            loguru_logger.info(f"{instrument_name} calculated_profit entry_: {calculated_profit}")
            last_preds['maxdiffprofit_profit'] = calculated_profit
            last_preds['maxdiffprofit_profit_values'] = list(calculate_profit_torch_with_entry_buysell_profit_values(
                last_preds[
                    "close_actual_movement_values"],
                maxdiff_trades,
                last_preds[
                    "high_actual_movement_values"] + close_to_high,
                last_preds[
                    "high_predictions"] + close_to_high,
                last_preds[
                    "low_actual_movement_values"] - close_to_low,
                last_preds[
                    "low_predictions"] - close_to_low,

            ).detach().cpu().numpy())
            latest_close_to_low = abs(1 - (last_preds['low_predicted_price_value'] / last_preds['close_last_price']))
            last_preds['latest_low_diff'] = latest_close_to_low

            latest_close_to_high = abs(1 - (last_preds['high_predicted_price_value'] / last_preds['close_last_price']))
            last_preds['latest_high_diff'] = latest_close_to_high

            # todo margin allocation tests
            current_profit = calculated_profit
            max_profit = float('-Inf')
            for buy_take_profit_multiplier in np.linspace(-.03, .03, 500):
                calculated_profit = calculate_trading_profit_torch_with_entry_buysell(scaler, None,
                                                                                      last_preds[
                                                                                          "close_actual_movement_values"],
                                                                                      maxdiff_trades,
                                                                                      last_preds[
                                                                                          "high_actual_movement_values"] + close_to_high,
                                                                                      last_preds[
                                                                                          "high_predictions"] + close_to_high + buy_take_profit_multiplier,
                                                                                      last_preds[
                                                                                          "low_actual_movement_values"] - close_to_low,
                                                                                      last_preds[
                                                                                          "low_predictions"] - close_to_low,

                                                                                      ).item()
                if calculated_profit > max_profit:
                    max_profit = calculated_profit
                    last_preds['maxdiffprofit_profit_high_multiplier'] = buy_take_profit_multiplier
                    last_preds['maxdiffprofit_high_profit'] = max_profit

            max_profit = float('-Inf')
            for low_take_profit_multiplier in np.linspace(-.03, .03, 500):
                calculated_profit = calculate_trading_profit_torch_with_entry_buysell(scaler, None,
                                                                                      last_preds[
                                                                                          "close_actual_movement_values"],
                                                                                      maxdiff_trades,
                                                                                      last_preds[
                                                                                          "high_actual_movement_values"] + close_to_high,
                                                                                      last_preds[
                                                                                          "high_predictions"] + close_to_high +
                                                                                      last_preds[
                                                                                          'maxdiffprofit_profit_high_multiplier'],
                                                                                      last_preds[
                                                                                          "low_actual_movement_values"] - close_to_low,
                                                                                      last_preds[
                                                                                          "low_predictions"] - close_to_low + low_take_profit_multiplier,

                                                                                      ).item()
                if calculated_profit > max_profit:
                    max_profit = calculated_profit
                    last_preds['maxdiffprofit_profit_low_multiplier'] = low_take_profit_multiplier
                    last_preds['maxdiffprofit_low_profit'] = max_profit

            # with buysellentry:

            calculated_profit = calculate_trading_profit_torch_with_entry_buysell(scaler, None,
                                                                                  last_preds[
                                                                                      "close_actual_movement_values"],
                                                                                  last_preds["close_trade_values"],
                                                                                  last_preds[
                                                                                      "high_actual_movement_values"] + close_to_high,
                                                                                  last_preds[
                                                                                      "high_predictions"] + close_to_high,
                                                                                  last_preds[
                                                                                      "low_actual_movement_values"] - close_to_low,
                                                                                  last_preds[
                                                                                      "low_predictions"] - close_to_low,

                                                                                  ).item()
            loguru_logger.info(f"{instrument_name} calculated_profit entry_: {calculated_profit}")
            last_preds['entry_takeprofit_profit'] = calculated_profit
            last_preds['entry_takeprofit_profit_values'] = list(calculate_profit_torch_with_entry_buysell_profit_values(
                last_preds["close_actual_movement_values"],
                last_preds["close_trade_values"],
                last_preds[
                    "high_actual_movement_values"] + close_to_high,
                last_preds[
                    "high_predictions"] + close_to_high,
                last_preds[
                    "low_actual_movement_values"] - close_to_low,
                last_preds[
                    "low_predictions"] - close_to_low,

            ).detach().cpu().numpy())

            # todo margin allocation tests
            current_profit = calculated_profit
            max_profit = float('-Inf')
            for buy_take_profit_multiplier in np.linspace(-.03, .03, 500):
                calculated_profit = calculate_trading_profit_torch_with_entry_buysell(scaler, None,
                                                                                      last_preds[
                                                                                          "close_actual_movement_values"],
                                                                                      last_preds["close_trade_values"],
                                                                                      last_preds[
                                                                                          "high_actual_movement_values"] + close_to_high,
                                                                                      last_preds[
                                                                                          "high_predictions"] + close_to_high + buy_take_profit_multiplier,
                                                                                      last_preds[
                                                                                          "low_actual_movement_values"] - close_to_low,
                                                                                      last_preds[
                                                                                          "low_predictions"] - close_to_low,
                                                                                      ).item()
                if calculated_profit > max_profit:
                    max_profit = calculated_profit
                    last_preds['entry_takeprofit_profit_high_multiplier'] = buy_take_profit_multiplier
                    last_preds['entry_takeprofit_high_profit'] = max_profit
                    # loguru_logger.info(
                    #     f"{instrument_name} buy_entry_take_profit_multiplier: {buy_take_profit_multiplier} calculated_profit: {calculated_profit}")

            max_profit = float('-Inf')
            for low_take_profit_multiplier in np.linspace(-.03, .03, 500):
                calculated_profit = calculate_trading_profit_torch_with_entry_buysell(scaler, None,
                                                                                      last_preds[
                                                                                          "close_actual_movement_values"],
                                                                                      last_preds["close_trade_values"],
                                                                                      last_preds[
                                                                                          "high_actual_movement_values"] + close_to_high,
                                                                                      last_preds[
                                                                                          "high_predictions"] + close_to_high +
                                                                                      last_preds[
                                                                                          'entry_takeprofit_profit_high_multiplier'],
                                                                                      last_preds[
                                                                                          "low_actual_movement_values"] - close_to_low,
                                                                                      last_preds[
                                                                                          "low_predictions"] - close_to_low + low_take_profit_multiplier,
                                                                                      ).item()
                if calculated_profit > max_profit:
                    max_profit = calculated_profit
                    last_preds['entry_takeprofit_profit_low_multiplier'] = low_take_profit_multiplier
                    last_preds['entry_takeprofit_low_profit'] = max_profit
                    # loguru_logger.info(
                    #     f"{instrument_name} entry_low_take_profit_multiplier: {low_take_profit_multiplier} calculated_profit: {calculated_profit}")
            # TODO break multipliers if we have no data on said side.
            last_preds['entry_takeprofit_low_price'] = last_preds['low_predicted_price_value'] * (1 + last_preds[
                'entry_takeprofit_profit_low_multiplier'])
            last_preds['entry_takeprofit_high_price'] = last_preds['high_predicted_price_value'] * (1 + last_preds[
                'entry_takeprofit_profit_high_multiplier'])
            last_preds['maxdiffprofit_low_price'] = last_preds['low_predicted_price_value'] * (1 + last_preds[
                'maxdiffprofit_profit_low_multiplier'])
            last_preds['maxdiffprofit_high_price'] = last_preds['high_predicted_price_value'] * (1 + last_preds[
                'maxdiffprofit_profit_high_multiplier'])

            last_preds['takeprofit_low_price'] = last_preds['low_predicted_price_value'] * (1 + last_preds[
                'takeprofit_profit_low_multiplier'])
            last_preds['takeprofit_high_price'] = last_preds['high_predicted_price_value'] * (1 + last_preds[
                'takeprofit_profit_high_multiplier'])

            CSV_KEYS = list(last_preds.keys())
            if not headers_written:
                with open(save_file_name, "a") as f:
                    writer = csv.DictWriter(f, CSV_KEYS)
                    writer.writeheader()
                    headers_written = True
            with open(save_file_name, "a") as f:
                writer = csv.DictWriter(f, CSV_KEYS)
                writer.writerow(last_preds)

                # # shift train predictions for plotting
                # trainPredictPlot = np.empty_like(price)
                # trainPredictPlot[:, :] = np.nan
                # trainPredictPlot[lookback:len(y_train_pred_inverted) + lookback, :] = y_train_pred_inverted
                #
                # # shift test predictions for plotting
                # testPredictPlot = np.empty_like(price)
                # testPredictPlot[:, :] = np.nan
                # testPredictPlot[len(y_train_pred_inverted) + lookback - 1:len(price), :] = best_y_test_pred_inverted
                #
                # original = scaler.inverse_transform(price['Close'].values.reshape(-1, 1))
                #
                # predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
                # predictions = np.append(predictions, original, axis=1)
                # result = pd.DataFrame(predictions)

                # # plot
                # import plotly.graph_objects as go
                #
                # fig = go.Figure()
                # fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[0],
                #                                     mode='lines',
                #                                     name='Train prediction')))
                # fig.add_trace(go.Scatter(x=result.index, y=result[1],
                #                          mode='lines',
                #                          name='Test prediction'))
                # fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[2],
                #                                     mode='lines',
                #                                     name='Actual Value')))
                # fig.update_layout(
                #     xaxis=dict(
                #         showline=True,
                #         showgrid=True,
                #         showticklabels=False,
                #         linecolor='white',
                #         linewidth=2
                #     ),
                #     yaxis=dict(
                #         title_text='Close (USD)',
                #         titlefont=dict(
                #             family='Rockwell',
                #             size=12,
                #             color='white',
                #         ),
                #         showline=True,
                #         showgrid=True,
                #         showticklabels=True,
                #         linecolor='white',
                #         linewidth=2,
                #         ticks='outside',
                #         tickfont=dict(
                #             family='Rockwell',
                #             size=12,
                #             color='white',
                #         ),
                #     ),
                #     showlegend=True,
                #     template='plotly_dark'
                #
                # )
                #
                # annotations = []
                # annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                #                         xanchor='left', yanchor='bottom',
                #                         text=csv_file.stem,
                #                         font=dict(family='Rockwell',
                #                                   size=26,
                #                                   color='white'),
                #                         showarrow=False))
                # fig.update_layout(annotations=annotations)
                # 
                # fig.show()

    loguru_logger.info(f"val_loss: {total_val_loss / len(csv_files)}")
    loguru_logger.info(f"total_forecasted_profit: {total_forecasted_profit / len(csv_files)}")
    loguru_logger.info(f"total_val_loss oer symbol: {total_val_loss / len(csv_files)}")
    loguru_logger.info(f"total_forecasted_profit avg per symbol: {total_forecasted_profit / len(csv_files)}")
    return pd.read_csv(save_file_name)


def df_to_torch(df):
    return torch.tensor(df.values, dtype=torch.float)


if __name__ == "__main__":
    make_predictions("2024-04-18--06-14-26")
