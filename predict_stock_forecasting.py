import csv
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data_utils import split_data
from loss_utils import calculate_trading_profit_torch, DEVICE, get_trading_profits_list, percent_movements_augment, \
    TradingLossBinary, TradingLoss, calculate_trading_profit_torch_buy_only, \
    calculate_trading_profit_torch_with_buysell, calculate_trading_profit_torch_with_entry_buysell
from model import GRU

transformers.set_seed(42)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from loguru import logger as loguru_logger

base_dir = Path(__file__).parent
loguru_logger.info(base_dir)
from sklearn.preprocessing import MinMaxScaler

from torch.utils.tensorboard import SummaryWriter

current_date_formatted = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
tb_writer = SummaryWriter(log_dir=f"./logs/{current_date_formatted}")


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
    return torch.tensor(series_pd.values, dtype=torch.float)#todo gpu, device=DEVICE)


def make_predictions(input_data_path=None, pred_name='', retrain=False):
    """
    Make predictions for all csv files in directory.
    """
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_file_name = results_dir / f"predictions-{time}.csv"
    # CSV_KEYS = [
    #     'instrument',
    #     'close_last_price',
    #     'close_predicted_price',
    #     'close_val_loss',
    #     'close_percent_movement',
    #     "close_likely_percent_uncertainty",
    #     "close_minus_uncertainty",
    #     'high_last_price',
    #     'high_predicted_price',
    #     'high_val_loss',
    #     'high_percent_movement',
    #     "high_likely_percent_uncertainty",
    #     "high_minus_uncertainty",
    #     'low_last_price',
    #     'low_predicted_price',
    #     'low_val_loss',
    #     'low_percent_movement',
    #     "low_likely_percent_uncertainty",
    #     "low_minus_uncertainty",
    #
    # ]

    headers_written = False
    # experiment with shared weights roughly a failure
    # input_dim = 4
    # hidden_dim = 32
    # num_layers = 2
    # output_dim = 1
    #
    # model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    # model.to(device)
    # model.load_state_dict(torch.load(base_dir / "data/model.pth"))

    # criterion = torch.nn.L1Loss(reduction='mean')
    # optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    total_val_loss = 0
    total_forecasted_profit = 0
    total_buy_val_loss = 0
    total_profit = 0

    timing_idx = 0

    if input_data_path:
        input_data_files = base_dir / "data" / input_data_path
    else:
        input_data_files = base_dir / "data"
    csv_files = list(input_data_files.glob("*.csv"))

    for days_to_drop in [0]:  # [1,2,3,4,5,6,7,8,9,10,11]:
        for csv_file in csv_files:
            instrument_name = csv_file.stem.split('-')[0]
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

                # x_test = pre_process_data(x_test)

                lookback = 20  # choose sequence length , GTLB only has been open for 27days cant go over that :O
                if len(price) > 40:
                    lookback = 30
                # longer didnt help
                # if len(price) > 100:
                #     lookback = 90
                # if len(price) > 200:
                #     lookback = 180
                # if len(price) > 300:
                #     lookback = 280
                max_prediction_length = 6
                max_encoder_length = 24
                # rename date to time_idx
                price = price.rename(columns={"Date": "time_idx"})
                price['y'] = price[key_to_predict].shift(-1)
                price['trade_weight'] = (price["y"] > 0) * 2 - 1

                price.drop(price.tail(1).index, inplace=True)  # drop last row because of percent change augmentation

                # cuttoff max_prediction_length from price
                # price = price.iloc[:-max_prediction_length]
                # add ascending id to price dataframe
                price['id'] = price.index
                # add constant column to price dataframe
                price['constant'] = 1
                # drop nan values
                price = price.dropna()
                final_pred_to_predict = price.tail(1)
                # price.drop(final_pred_to_predict.index, inplace=True)  # drop last row because of percent change augmentation

                target_to_pred = "y"
                training = TimeSeriesDataSet(
                    price[:-7],
                    time_idx="id",
                    target=target_to_pred,
                    group_ids=['constant'],
                    min_encoder_length=max_encoder_length // 2,
                    # keep encoder length long (as it is in the validation set)
                    max_encoder_length=max_encoder_length,
                    min_prediction_length=1,
                    max_prediction_length=max_prediction_length,
                    # static_categoricals=["agency", "sku"],
                    # static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
                    # time_varying_known_categoricals=["special_days", "month"],
                    # variable_groups={"special_days": []},
                    # group of categorical variables can be treated as one variable
                    # time_varying_known_reals=["time_idx", "price_regular", "discount_in_percent"],
                    time_varying_unknown_categoricals=[],
                    time_varying_known_reals=[
                        "Open",
                        "High",
                        "Low",
                        "Close",
                    ],
                    # target_normalizer=GroupNormalizer(
                    #     groups=["agency", "sku"], transformation="softplus"
                    # ),  # use softplus and normalize by group
                    # add_relative_time_idx=True,
                    add_target_scales=True,
                    add_encoder_length=True,
                )
                validation = TimeSeriesDataSet.from_dataset(training, price, min_prediction_idx=training.index.time.max() + 1, predict=True, stop_randomization=True)

                # create dataloaders for model
                batch_size = 128  # set this between 32 to 128
                train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, pin_memory=True,
                                                          num_workers=0)
                val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, pin_memory=True,
                                                          num_workers=0)
                # actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
                # baseline_predictions = Baseline().predict(val_dataloader)
                # loguru_logger.info((actuals[:, :-1] - baseline_predictions[:, :-1]).abs().mean().item())

                # trainer = pl.Trainer(
                #     gpus=0,
                #     # clipping gradients is a hyperparameter and important to prevent divergance
                #     # of the gradient for recurrent neural networks
                #     gradient_clip_val=0.1,
                # )

                # best hyperpramams look like:
                # {'gradient_clip_val': 0.05690473137493243, 'hidden_size': 50, 'dropout': 0.23151352460442215,
                #  'hidden_continuous_size': 22, 'attention_head_size': 2, 'learning_rate': 0.0816548812864903}
                best_hyperparams_save_file = f"data/test_study{pred_name}{key_to_predict}{instrument_name}.pkl"
                params = None
                try:
                    params = pickle.load(open(best_hyperparams_save_file, "rb"))
                except FileNotFoundError:
                    # logger.info("No best hyperparams found, tuning")
                    best_hyperparams_save_file = f"data/test_study{instrument_name}.pkl"
                    try:
                        params = pickle.load(open(best_hyperparams_save_file, "rb"))
                    except FileNotFoundError:
                        # logger.info("No best hyperparams found, tuning")
                        pass
                added_best_params = {}
                if params and params.best_trial.params:
                    added_best_params = params.best_trial.params
                added_params = {  # not meaningful for finding the learning rate but otherwise very important
                    "learning_rate": 0.03,
                    "hidden_size": 16,  # most important hyperparameter apart from learning rate
                    # number of attention heads. Set to up to 4 for large datasets
                    "attention_head_size": 1,
                    "dropout": 0.1,  # between 0.1 and 0.3 are good values
                    "hidden_continuous_size": 8,  # set to <= hidden_size
                    "output_size": 1,  # 7 quantiles by default
                    "loss": TradingLoss(),
                    # "logging_metrics": TradingLossBinary(),
                    # reduce learning rate if no improvement in validation loss after x epochs
                    "reduce_on_plateau_patience": 4,
                }
                added_params.update(added_best_params)
                if type(added_params['output_size']) == int:
                    if type(target_to_pred) == list:
                        added_params['output_size'] = [added_params['output_size']] * len(target_to_pred)
                gradient_clip_val = added_params.pop("gradient_clip_val", 0.1)
                tft = TemporalFusionTransformer.from_dataset(
                    training,
                    **added_params
                )

                loguru_logger.info(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

                early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=40, verbose=False,
                                                    mode="min")
                model_checkpoint = ModelCheckpoint(
                    monitor="val_loss",
                    mode="min",
                    # save a few of the top models
                    # incase one does great on other metrics than just val_loss
                    save_top_k=1,
                    verbose=True,
                    filename=instrument_name + "_{epoch}_{val_loss:.8f}",
                )
                lr_logger = LearningRateMonitor()  # log the learning rate
                logger = TensorBoardLogger(f"lightning_logs/{pred_name}/{key_to_predict}/{instrument_name}")  # logging results to a tensorboard
                trainer = pl.Trainer(
                    max_epochs=100,
                    gpus=1,
                    weights_summary="top",
                    gradient_clip_val=gradient_clip_val,
                    # track_grad_norm=2,
                    # auto_lr_find=True,
                    # auto_scale_batch_size='binsearch',
                    limit_train_batches=30,  # coment in for training, running valiation every 30 batches
                    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
                    callbacks=[lr_logger, early_stop_callback, model_checkpoint],
                    logger=logger,
                )
                # retrain = True # todo reenable
                # try find specific hl net

                checkpoints_dir = (base_dir / 'lightning_logs' / pred_name / key_to_predict / instrument_name)
                checkpoint_files = list(checkpoints_dir.glob(f"**/*.ckpt"))
                if len(checkpoint_files) == 0:
                    loguru_logger.info("No min+open/low specific checkpoints found, training from other checkpoint")
                    checkpoints_dir = (base_dir / 'lightning_logs' / pred_name / instrument_name)
                    checkpoint_files = list(checkpoints_dir.glob(f"**/*.ckpt"))
                    if len(checkpoint_files) == 0:
                        loguru_logger.info("No min specific checkpoints found, training from other checkpoint")
                        checkpoints_dir = (base_dir / 'lightning_logs' / instrument_name)
                        checkpoint_files = list(checkpoints_dir.glob(f"**/*.ckpt"))
                best_tft = tft

                if checkpoint_files:

                    best_checkpoint_path = checkpoint_files[0]
                    # sort by most recent checkpoint_files
                    checkpoint_files.sort(key=lambda x: os.path.getctime(x))
                    # load the most recent
                    best_checkpoint_path = checkpoint_files[-1]
                    # find best checkpoint
                    # min_current_loss = str(checkpoint_files[0]).split("=")[-1][0:len('.ckpt')]
                    # for file_name in checkpoint_files:
                    #     current_loss = str(file_name).split("=")[-1][0:len('.ckpt')]
                    #     if float(current_loss) < float(min_current_loss):
                    #         min_current_loss = current_loss
                    #         best_checkpoint_path = file_name
                    #         # TODO invalidation for 30minute vs daily data

                    loguru_logger.info(f"Loading best checkpoint from {best_checkpoint_path}")
                    best_tft = TemporalFusionTransformer.load_from_checkpoint(str(best_checkpoint_path))

                if retrain:
                    trainer.fit(
                        best_tft,
                        train_dataloader=train_dataloader,
                        val_dataloaders=val_dataloader,
                    )
                    best_model_path = trainer.checkpoint_callback.best_model_path
                    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

                actual_list = [y[0] for x, y in iter(val_dataloader)] # TODO check this y center transfor prints feature neamses warning

                actuals = torch.cat(actual_list)
                predictions = best_tft.predict(val_dataloader)
                mean_val_loss = (actuals[:, :-1] - predictions[:, :-1]).abs().mean()

                if not added_best_params:
                    # find best hyperparams
                    from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

                    # create study
                    study = optimize_hyperparameters(
                        train_dataloader,
                        val_dataloader,
                        model_path="optuna_test",
                        n_trials=100,
                        max_epochs=200,
                        gradient_clip_val_range=(0.01, 1.0),
                        hidden_size_range=(8, 128),
                        hidden_continuous_size_range=(8, 128),
                        attention_head_size_range=(1, 4),
                        learning_rate_range=(0.001, 0.1),
                        dropout_range=(0.1, 0.3),
                        trainer_kwargs=dict(limit_train_batches=30),
                        reduce_on_plateau_patience=4,
                        use_learning_rate_finder=False,
                        # use Optuna to find ideal learning rate or use in-built learning rate finder
                    )

                    # save study results - also we can resume tuning at a later point in time
                    with open(best_hyperparams_save_file, "wb") as fout:
                        pickle.dump(study, fout)

                    # show best hyperparameters
                    loguru_logger.info(study.best_trial.params)

                loguru_logger.info(f"mean val loss:${mean_val_loss}")

                raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)
                # for idx in range(1):  # plot 10 examples
                #     plot = best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)
                ## display plot
                # note last one at zero actual movement is not true
                # plot.show()

                # predict trade if last value is above the prediction
                trading_preds = (predictions[:, :-1] > 0) * 2 - 1
                # last_values = x_test[:, -1, 0]
                calculated_profit = calculate_trading_profit_torch(scaler, None, actuals[:, :-1], trading_preds).item()

                calculated_profit_buy_only = calculate_trading_profit_torch_buy_only(scaler, None, actuals[:, :-1],
                                                                            trading_preds).item()
                calculated_profit_values = get_trading_profits_list(scaler, None, actuals[:, :-1], trading_preds)
                #
                # x_train, y_train, x_test, y_test = split_data(price, lookback)
                #
                # x_train = torch.from_numpy(x_train).type(torch.Tensor).to(DEVICE)
                # x_test = torch.from_numpy(x_test).type(torch.Tensor).to(DEVICE)
                # y_train = torch.from_numpy(y_train).type(torch.Tensor).to(DEVICE)
                # y_test = torch.from_numpy(y_test).type(torch.Tensor).to(DEVICE)
                #
                # input_dim = 4
                # hidden_dim = 32
                # num_layers = 6
                # output_dim = 1
                # # TODO use pytorch forecasting
                # # from pytorch_forecasting import Baseline, TemporalFusionTransformer
                # model = GRU(
                #     input_dim=input_dim,
                #     hidden_dim=hidden_dim,
                #     output_dim=output_dim,
                #     num_layers=num_layers,
                # )
                # model.to(DEVICE)
                # model.train()
                # criterion = torch.nn.L1Loss(reduction="mean")
                # optimiser = torch.optim.AdamW(model.parameters(), lr=0.01)
                #
                # start_time = datetime.now()
                #
                # num_epochs = 100
                # hist = np.zeros(num_epochs)
                # y_train_pred = None
                # min_val_loss = np.inf
                # best_y_test_pred_inverted = []
                #
                # # Number of steps to unroll
                # for epoc_idx in range(num_epochs):
                #     model.train()
                #     random_aug = torch.rand(x_train.shape) * .0002 - .0001
                #     augmented = x_train + random_aug.to(DEVICE)
                #     y_train_pred = model(augmented)
                #
                #     loss = criterion(y_train_pred, y_train)
                #     loguru_logger.info("Epoch ", epoc_idx, "MSE: ", loss.item())
                #     tb_writer.add_scalar(f"{key_to_predict}/Loss/{instrument_name}/train", loss.item(), epoc_idx)
                #     hist[epoc_idx] = loss.item()
                #
                #     loss.backward()
                #     optimiser.step()
                #     optimiser.zero_grad()
                #     ## test
                #     model.eval()
                #
                #     y_test_pred = model(x_test)
                #     # invert predictions
                #     y_test_pred_inverted = torch_inverse_transform(scaler, y_test_pred)
                #     # y_train_pred_inverted = torch_inverse_transform(scaler, y_train_pred)
                #
                #     # loguru_logger.info(y_test_pred_inverted)
                #     loss = criterion(y_test_pred, y_test)
                #     loguru_logger.info(f"val loss: {loss}")
                #     tb_writer.add_scalar(f"{key_to_predict}/Loss/{instrument_name}/val", loss.item(), epoc_idx)
                #     loguru_logger.info(f"Last prediction: y_test_pred_inverted[-1] = {y_test_pred_inverted[-1]}")
                #     tb_writer.add_scalar(f"{key_to_predict}/Prediction/{instrument_name}last_pred", y_test_pred_inverted[-1], epoc_idx)
                #
                #     # detached_y_test = y_test.detach().cpu().numpy()
                #     last_values = x_test[:, -1, 0]
                #     # predict trade if last value is above the prediction
                #     trading_preds = (y_test_pred > last_values) * 2 - 1
                #     last_values = x_test[:, -1, 0]
                #     calculated_profit = calculate_trading_profit_torch(scaler, last_values, y_test[:, 0], trading_preds[:, 0]).item()
                #     loguru_logger.info(f"{instrument_name}: {key_to_predict} calculated_profit: {calculated_profit}")
                #     tb_writer.add_scalar(f"{key_to_predict}/Profit/{instrument_name}:  calculated_profit", calculated_profit, epoc_idx)
                #     if loss < min_val_loss:
                #         min_val_loss = loss
                #         torch.save(model.state_dict(), "data/model.pth")
                #         best_y_test_pred_inverted = y_test_pred_inverted
                #         best_y_test_pred = y_test_pred
                #         best_y_train_pred = y_train_pred
                #         # percent estimate
                #         y_test_end_scaled_loss = torch_inverse_transform(scaler, torch.add(y_test_pred, loss))
                #
                #         likely_percent_uncertainty = (
                #             (y_test_end_scaled_loss[-1] - y_test_pred_inverted[-1])
                #             / y_test_pred_inverted[-1]
                #         ).item()
                #         min_loss_trading_profit = calculated_profit
                #
                #
                # training_time = datetime.now() - start_time
                # loguru_logger.info("Training time: {}".format(training_time))
                # tb_writer.add_scalar("Time/epoc", training_time.total_seconds(), timing_idx)
                # timing_idx += 1

                # loguru_logger.info(scaler.inverse_transform(y_train_pred.detach().cpu().numpy()))
                # tb_writer.add_scalar("Prediction/train", y_train_pred_inverted[-1], 0)

                val_loss = mean_val_loss
                # percent_movement = (
                #     predictions[-1].item() - last_close_price
                # ) / last_close_price
                last_preds[key_to_predict.lower() + "_last_price"] = last_close_price
                last_preds[key_to_predict.lower() + "_predicted_price"] = predictions[0,
                                                                                      -1
                ].item()
                last_preds[key_to_predict.lower() + "_predicted_price_value"] = last_close_price + (last_close_price * predictions[0,
                                                                                      -1
                ].item())
                last_preds[key_to_predict.lower() + "_val_loss"] = val_loss.item()
                last_preds[key_to_predict.lower() + "min_loss_trading_profit"] = calculated_profit
                last_preds[key_to_predict.lower() + "min_loss_buy_only_trading_profit"] = calculated_profit_buy_only
                last_preds[key_to_predict.lower() + "_actual_movement_values"] = actuals[:, :-1].view(-1)
                last_preds[key_to_predict.lower() + "_calculated_profit_values"] = calculated_profit_values.view(-1)
                last_preds[key_to_predict.lower() + "_trade_values"] = trading_preds.view(-1)
                last_preds[key_to_predict.lower() + "_predictions"] = predictions[:, :-1].view(-1)
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

            key_to_predict = "Close"
            for training_mode in [
                # "BuyOrSell",
                # "Leverage",
            ]:
                loguru_logger.info(f"training mode: {training_mode} {instrument_name}")
                stock_data = load_stock_data_from_csv(csv_file)
                stock_data = stock_data.dropna()
                if stock_data.empty:
                    loguru_logger.info(f"Empty data for {instrument_name}")
                    continue
                # use a quarter of 15min data / hours data
                # drop_n_rows(stock_data, 2)
                # stock_data.reset_index(drop=True, inplace=True)
                # drop_n_rows(stock_data, 2)
                # stock_data.reset_index(drop=True, inplace=True)

                # drop last days_to_drop rows
                if days_to_drop:
                    stock_data = stock_data.iloc[:-days_to_drop]

                # x_train, x_test = train_test_split(stock_data)
                last_close_price = stock_data[key_to_predict].iloc[-1]
                data = pre_process_data(stock_data, "High")
                # todo scaler for each, this messes up the scaler
                data = pre_process_data(data, "Low")
                data = pre_process_data(data, "Open")
                data = pre_process_data(data, "Close")
                price = data[["Close", "High", "Low", "Open"]]
                price.drop(price.tail(1).index, inplace=True)  # drop last row because of percent change augmentation
                # x_test = pre_process_data(x_test)

                lookback = 16  # choose sequence length , GTLB only has been open for 27days cant go over that :O
                if len(price) > 40:
                    lookback = 33
                # longer didnt help
                # if len(price) > 129:
                #     lookback = 129
                # if len(price) > 200:
                #     lookback = 180
                # if len(price) > 300:
                #     lookback = 280
                x_train, y_train, x_test, y_test = split_data(price, lookback)

                x_train = torch.from_numpy(x_train).type(torch.Tensor).to(DEVICE)
                x_test = torch.from_numpy(x_test).type(torch.Tensor).to(DEVICE)
                y_train = torch.from_numpy(y_train).type(torch.Tensor).to(DEVICE)
                y_test = torch.from_numpy(y_test).type(torch.Tensor).to(DEVICE)

                input_dim = 4
                hidden_dim = 32
                num_layers = 6
                output_dim = 1
                # TODO use pytorch forecasting
                # from pytorch_forecasting import Baseline, TemporalFusionTransformer
                model = GRU(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_layers=num_layers,
                )
                model.to(DEVICE)
                model.train()
                criterion = torch.nn.L1Loss(reduction="mean")
                optimiser = torch.optim.AdamW(model.parameters(), lr=0.01)

                start_time = datetime.now()

                num_epochs = 1000  # 100000 TODO more is better
                hist = np.zeros(num_epochs)
                y_train_pred = None
                min_val_loss = np.inf
                # count the amount that the model doesnt improve, terminate if it doesnt improve for a while
                number_of_unsuccessful_epochs = 0
                best_current_profit = np.inf
                best_y_test_pred = []
                best_y_train_pred = []

                # Number of steps to unroll
                for epoc_idx in range(num_epochs):
                    model.train()
                    random_aug = torch.rand(x_train.shape) * .002 - .001
                    augmented = x_train + random_aug.to(DEVICE)
                    y_train_pred = model(augmented)
                    # sciNet gives back 3 values, we only need the first one
                    # y_train_pred = y_train_pred[:, :, 0]

                    # loss = criterion(y_train_pred, y_train)
                    if "BuyOrSell" == training_mode:
                        # sigmoid = torch.nn.Sigmoid()
                        # y_train_pred = sigmoid(y_train_pred)
                        ## map to three trinary predictions -1 0 and 1
                        # y_train_pred = torch.round(y_train_pred) # turn off rounding because ruins gradient
                        # y_train_pred = torch.clamp(y_train_pred, -1, 1)
                        # compute percent movement between y_train and last_values

                        last_values = x_train[:, -1, 0]
                        loss = -calculate_trading_profit_torch(scaler, last_values, y_train[:, 0], y_train_pred[:, 0])

                        ## log if loss is nan
                        if np.isnan(loss.item()):
                            loguru_logger.info(f"{instrument_name} loss is nan")
                            loguru_logger.info(f"{last_values} last_values")
                            loguru_logger.info(f"{y_train} last_values")
                            loguru_logger.info(f"{y_train_pred} last_values")
                            continue

                        # add depreciation loss
                        # loss -= len(y_train) * (.001 / 365)

                        # current_profit [:, 0]= calculate_trading_profit(scaler, x_train, y_train.detach().cpu().numpy()[:, 0],
                        #                                           y_train_pred.detach().cpu().numpy())

                        loguru_logger.info(f"{training_mode} current_profit: {-loss}")
                        tb_writer.add_scalar(f"{instrument_name}/{training_mode}/current_profit/train", -loss, epoc_idx)
                    # elif "Leverage" == training_mode:
                    #     # sigmoid = torch.nn.Sigmoid()
                    #     # y_train_pred = sigmoid(y_train_pred)
                    #     ## map to three trinary predictions -1 0 and 1
                    #     y_train_pred = (y_train_pred * 8) - 4  # how much leveraged? -4x to 4x
                    #     y_train_pred = torch.clamp(y_train_pred, -4, 4)
                    #     # compute percent movement between y_train and last_values
                    #     last_values = x_train[:, -1, 0]
                    #     percent_movements = ((y_train - last_values) / last_values) + 1
                    #     # negative as profit is good
                    #     loss = -torch.prod(1 + (y_train_pred * percent_movements))
                    #     loss += len(y_test) * (.001 / 365)
                    #
                    #     # those where not scaled properly, scale properly for logging purposes
                    #     last_values_scaled = scaler.inverse_transform(
                    #         last_values.detach().cpu().numpy()
                    #     )
                    #     percent_movements_scaled = (
                    #         scaler.inverse_transform(y_train.detach().cpu().numpy())
                    #         - last_values_scaled
                    #     ) / last_values_scaled
                    #     current_profit = np.product(
                    #         y_train_pred.detach().cpu().numpy() * percent_movements_scaled
                    #     )
                    #     loguru_logger.info(f"current_profit: {current_profit}")
                    #     tb_writer.add_scalar(f"{instrument_name}/{training_mode}/current_profit/test", current_profit, epoc_idx)
                    loguru_logger.info("Epoch ", epoc_idx, "MSE: ", loss.item())
                    tb_writer.add_scalar(f"{instrument_name}/{training_mode}/loss", loss.item(), epoc_idx)
                    hist[epoc_idx] = loss.item()

                    loss.backward()
                    optimiser.step()
                    optimiser.zero_grad()

                    ## test
                    model.eval()

                    y_test_pred = model(x_test)
                    # sciNet gives back 3 values, we only need the first one
                    # y_test_pred = y_test_pred[:, :, 0]
                    # dont actually need to invert predictions
                    # y_test_pred_inverted = y_test_pred.detach().cpu().numpy()
                    # y_train_pred_inverted = y_train_pred.detach().cpu().numpy()

                    # loguru_logger.info(y_test_pred_inverted)
                    # loss = criterion(y_test_pred, y_test)
                    if "BuyOrSell" == training_mode:
                        # sigmoid = torch.nn.Sigmoid()
                        # y_test_pred = sigmoid(y_test_pred)
                        ## map to three trinary predictions -1 0 and 1
                        # y_test_pred = torch.round(y_test_pred)  # turn off rounding because ruins gradient
                        y_test_pred = torch.clamp(y_test_pred, -4, 4)  # 4x leverage

                        # y_test_inverted = torch_inverse_transform(scaler, y_test)
                        # plot trading graph

                        # negative as profit is good
                        last_values = x_test[:, -1, 0]
                        loss = -calculate_trading_profit_torch(scaler, last_values, y_test[:, 0], y_test_pred[:, 0])
                        trading_profits_list = get_trading_profits_list(scaler, last_values, y_test[:, 0], y_test_pred[
                                                                                                           :, 0])
                        # add depreciation loss
                        # loss -= len(y_test) * (.001 / 365)

                        # current_profit [:, 0]= calculate_trading_profit(scaler, x_test, y_test.detach().cpu().numpy(), y_test_pred.detach().cpu().numpy()[:, 0])
                        loguru_logger.info(f"{training_mode} current_profit validation: {-loss}")
                        tb_writer.add_scalar(f"{instrument_name}/{training_mode}/current_profit/validation", -loss,
                                             epoc_idx)
                    # if "Leverage" == training_mode:
                    #     # sigmoid = torch.nn.Sigmoid()
                    #     # y_test_pred = sigmoid(y_test_pred)
                    #
                    #     ## map to three trinary predictions -1 0 and 1
                    #     y_test_pred = (y_test_pred * 8) - 4  # how much leveraged? -4x to 4x
                    #     y_test_pred = torch.clamp(y_test_pred, -4, 4)
                    #     # compute percent movement between y_test and last_values
                    #     last_values = x_test[:, -1, 0]
                    #     percent_movements = ((y_test - last_values) / last_values) + 1
                    #     # negative as profit is good
                    #     loss = -torch.prod(1 + (y_test_pred * percent_movements))
                    #     loss += len(y_test) * (.001 / 365)
                    #
                    #     # those where not scaled properly, scale properly for logging purposes
                    #     last_values_scaled = scaler.inverse_transform(
                    #         last_values.detach().cpu().numpy()
                    #     )
                    #     percent_movements_scaled = ((
                    #         scaler.inverse_transform(y_test.detach().cpu().numpy())
                    #         - last_values_scaled
                    #     ) / last_values_scaled) + 1
                    #     current_profit = np.product(
                    #         y_test_pred.detach().cpu().numpy() * percent_movements_scaled
                    #     )
                    #     loguru_logger.info(f"{training_mode} current_profit validation: {current_profit}")
                    #     tb_writer.add_scalar(f"{instrument_name}/{training_mode}/current_profit/validation", current_profit, t)
                    loguru_logger.info(f"{training_mode} val loss: {loss}")
                    loguru_logger.info(
                        f"{training_mode} Last prediction: y_test_pred[-1] = {y_test_pred[-1]}"
                    )
                    if loss < min_val_loss:
                        number_of_unsuccessful_epochs = 0
                        min_val_loss = loss
                        torch.save(model.state_dict(), f"data/model-classify-{instrument_name}.pth")
                        best_y_test_pred = y_test_pred
                        best_current_profit = -loss.item()
                        for i in range(len(y_test_pred)):
                            tb_writer.add_scalar(f"{instrument_name}/{training_mode}/predictions/test", y_test_pred[i],
                                                 i)
                            tb_writer.add_scalar(f"{instrument_name}/{training_mode}/actual/test", y_test[i][0:1], i)
                            # log trading_profits_list
                            tb_writer.add_scalar(f"{instrument_name}/{training_mode}/trading_profits/test",
                                                 trading_profits_list[i], i)
                    else:
                        number_of_unsuccessful_epochs += 1

                    if number_of_unsuccessful_epochs > 40:
                        loguru_logger.info(f"{instrument_name}/{training_mode} Early stopping")
                        break
                training_time = datetime.now() - start_time
                loguru_logger.info("Training time: {}".format(training_time))
                loguru_logger.info("Best val loss: {}".format(min_val_loss))
                loguru_logger.info("Best current profit: {}".format(best_current_profit))

                # loguru_logger.info(scaler.inverse_transform(y_train_pred.detach().cpu().numpy()))

                val_loss = loss.item()
                last_preds[training_mode.lower() + "_buy_no_or_sell"] = best_y_test_pred[
                    -1
                ].item()
                last_preds[training_mode.lower() + "_val_loss_classifier"] = val_loss
                last_preds[training_mode.lower() + "_val_profit"] = best_current_profit
                total_buy_val_loss += val_loss
                total_profit += best_current_profit

            # # compute loss when
            # calculate_trading_profit_torch_with_buysell()
            #
            # trading_preds = (predictions[:, :-1] > 0) * 2 - 1
            # last_values = x_test[:, -1, 0]
            # compute movement to high price
            validation_size = last_preds[
                "high_actual_movement_values"].numel()
            close_to_high = series_to_tensor(abs(1 - (stock_data["High"].iloc[-validation_size -2:-2] / stock_data["Close"].iloc[
                                                                                  -validation_size -2:-2])))
            close_to_low = series_to_tensor(abs(1 - (stock_data["Low"].iloc[
                                    -validation_size -2:-2] / stock_data["Close"].iloc[
                                                         -validation_size -2:-2])))
            calculated_profit = calculate_trading_profit_torch_with_buysell(scaler, None, last_preds["close_actual_movement_values"],
                                                                            last_preds["close_trade_values"],
                                                                            last_preds[
                                                                                "high_actual_movement_values"] + close_to_high,
                                                                            last_preds[
                                                                                "high_predictions"] + close_to_high,
                                                                            last_preds["low_actual_movement_values"] - close_to_low,
                                                                            last_preds["low_predictions"] - close_to_low,

                                                                            ).item()
            loguru_logger.info(f"{instrument_name} calculated_profit: {calculated_profit}")
            last_preds['takeprofit_profit'] = calculated_profit

            # todo margin allocation tests
            current_profit = calculated_profit
            max_profit = float('-Inf')
            for buy_take_profit_multiplier in np.linspace(-.03, .03, 500):
                calculated_profit = calculate_trading_profit_torch_with_buysell(scaler, None,
                                                                                last_preds["close_actual_movement_values"],
                                                                                last_preds["close_trade_values"],
                                                                                last_preds[
                                                                                    "high_actual_movement_values"] + close_to_high,
                                                                                last_preds[
                                                                                    "high_predictions"] + close_to_high + buy_take_profit_multiplier,
                                                                                last_preds["low_actual_movement_values"] - close_to_low,
                                                                                last_preds["low_predictions"] - close_to_low,
                                                                                ).item()
                if calculated_profit > max_profit:
                    max_profit = calculated_profit
                    last_preds['takeprofit_profit_high_multiplier'] = buy_take_profit_multiplier
                    last_preds['takeprofit_high_profit'] = max_profit
                    # loguru_logger.info(f"{instrument_name} buy_take_profit_multiplier: {buy_take_profit_multiplier} calculated_profit: {calculated_profit}")

            max_profit = float('-Inf')
            for low_take_profit_multiplier in np.linspace(-.03, .03, 500):
                calculated_profit = calculate_trading_profit_torch_with_buysell(scaler, None,
                                                                                last_preds["close_actual_movement_values"],
                                                                                last_preds["close_trade_values"],
                                                                                last_preds[
                                                                                    "high_actual_movement_values"] + close_to_high,
                                                                                last_preds[
                                                                                    "high_predictions"] + close_to_high +
                                                                                last_preds[
                                                                                    'takeprofit_profit_high_multiplier'],
                                                                                last_preds["low_actual_movement_values"] - close_to_low,
                                                                                last_preds["low_predictions"] - close_to_low + low_take_profit_multiplier,
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
            loguru_logger.info(f"{instrument_name} calculated_profit entry_: {calculated_profit}")
            last_preds['entry_takeprofit_profit'] = calculated_profit

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
            last_preds['entry_takeprofit_low_price'] = last_preds['low_predicted_price_value'] * (1+ last_preds[
                'entry_takeprofit_profit_low_multiplier'])
            last_preds['entry_takeprofit_high_price'] = last_preds['high_predicted_price_value'] * (1+ last_preds[
                'entry_takeprofit_profit_high_multiplier'])
            last_preds['maxdiffprofit_low_price'] = last_preds['low_predicted_price_value'] * (1+ last_preds[
                'maxdiffprofit_profit_low_multiplier'])
            last_preds['maxdiffprofit_high_price'] = last_preds['high_predicted_price_value'] * (1+ last_preds[
                'maxdiffprofit_profit_high_multiplier'])

            last_preds['takeprofit_low_price'] = last_preds['low_predicted_price_value'] * (1+ last_preds[
                'takeprofit_profit_low_multiplier'])
            last_preds['takeprofit_high_price'] = last_preds['high_predicted_price_value'] * (1+ last_preds[
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
    loguru_logger.info(f"total_buy_val_loss: {total_buy_val_loss / len(csv_files)}")
    loguru_logger.info(f"total_profit avg per symbol: {total_profit / len(csv_files)}")
    return pd.read_csv(save_file_name)


def df_to_torch(df):
    return torch.tensor(df.values, dtype=torch.float)


if __name__ == "__main__":
    make_predictions()
