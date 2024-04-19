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

                # x_test = pre_process_data(x_test)

                # lookback = 20  # choose sequence length , GTLB only has been open for 27days cant go over that :O
                # if len(price) > 40:
                #     lookback = 30
                # longer didnt help
                # if len(price) > 100:
                #     lookback = 90
                # if len(price) > 200:
                #     lookback = 180
                # if len(price) > 300:
                #     lookback = 280
                # max_prediction_length = 6
                # max_encoder_length = 24
                # rename date to time_idx
                price = price.rename(columns={"Date": "time_idx"})
                # not actually important what date it thinks as long as its daily i think
                price["ds"] = pd.date_range(start="1949-01-01", periods=len(price), freq="D").values
                price['y'] = price[key_to_predict].shift(-1)
                price['trade_weight'] = (price["y"] > 0) * 2 - 1

                price.drop(price.tail(1).index, inplace=True)  # drop last row because of percent change augmentation

                # cuttoff max_prediction_length from price
                # price = price.iloc[:-max_prediction_length]
                # add ascending id to price dataframe
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

                # create dataloaders for model
                # batch_size = 128  # set this between 32 to 128
                # compatibitiy

                # nhits_config = {
                #     "max_steps": 100,  # Number of SGD steps
                #     "input_size": 24,  # Size of input window
                #     "learning_rate": tune.loguniform(1e-5, 1e-1),  # Initial Learning rate
                #     "n_pool_kernel_size": tune.choice([[2, 2, 2], [16, 8, 1]]),  # MaxPool's Kernelsize
                #     "n_freq_downsample": tune.choice([[168, 24, 1], [24, 12, 1], [1, 1, 1]]),
                #     # Interpolation expressivity ratios
                #     "val_check_steps": 50,  # Compute validation every 50 steps
                #     "random_seed": tune.randint(1, 10),  # Random seed
                # }
                # horizon = len(Y_test_df) + 1
                # stacks = 3
                # nhits_config.update(dict(
                #     input_size=2 * horizon,
                #     max_steps=700,
                #     stack_types=stacks * ['identity'],
                #     n_blocks=stacks * [1],
                #     mlp_units=[[256, 256] for _ in range(stacks)],
                #     n_pool_kernel_size=stacks * [1],
                #     batch_size=32,
                #     scaler_type='standard',
                #     n_freq_downsample=[12, 4, 1],
                #     max_epochs=5000,
                #     val_check_steps=5,
                # ))
                # fit
                # with much less epocs like 10 or something
                # epocs = 50
                # epocs = 20 if not retrain else 700
                # epocs = 50 if not retrain else 2000
                # models = [NBEATS(input_size=2 * horizon, h=horizon, max_epochs=epocs),
                #           NHITS(
                #               input_size=2 * horizon,
                #               h=horizon,
                #               stack_types=stacks * ['identity'],
                #               n_blocks=stacks * [1],
                #               mlp_units=[[256, 256] for _ in range(stacks)],
                #               n_pool_kernel_size=stacks * [1],
                #               batch_size=32,
                #               scaler_type='standard',
                #               n_freq_downsample=[12, 4, 1],
                #               # loss=TradingLoss(), # TODO fix TradingLoss' object has no attribute 'outputsize_multiplier'
                #               max_epochs=epocs
                #           )]
                # # models = [NBEATS(input_size=2 * horizon, h=horizon, max_epochs=700),
                # #           NHITS(input_size=2 * horizon, h=horizon, max_epochs=700),
                # #           ]
                # nforecast = NeuralForecast(models=models, freq='D')
                # ### Load Checkpoint
                # checkpoints_dir = (base_dir / 'lightning_logs_nforecast' / pred_name / key_to_predict / instrument_name)
                # checkpoints_dir.mkdir(parents=True, exist_ok=True)
                # checkpoint_files = list(checkpoints_dir.glob(f"**/*.ckpt"))
                # if len(checkpoint_files) == 0:
                #     loguru_logger.info("No min+open/low specific checkpoints found, training from other checkpoint")
                #     checkpoints_dir = (base_dir / 'lightning_logs_nforecast' / pred_name / instrument_name)
                #     checkpoint_files = list(checkpoints_dir.glob(f"**/*.ckpt"))
                #     if len(checkpoint_files) == 0:
                #         loguru_logger.info("No min specific checkpoints found, training from other checkpoint")
                #         checkpoints_dir = (base_dir / 'lightning_logs_nforecast' / instrument_name)
                #         checkpoint_files = list(checkpoints_dir.glob(f"**/*.ckpt"))

                # if checkpoint_files:
                #     best_checkpoint_path = checkpoint_files[0]
                #     # sort by most recent checkpoint_files
                #     checkpoint_files.sort(key=lambda x: os.path.getctime(x))
                #     # load the most recent
                #     best_checkpoint_path = checkpoint_files[-1]
                #     # find best checkpoint
                #     # min_current_loss = str(checkpoint_files[0]).split("=")[-1][0:len('.ckpt')]
                #     # for file_name in checkpoint_files:
                #     #     current_loss = str(file_name).split("=")[-1][0:len('.ckpt')]
                #     #     if float(current_loss) < float(min_current_loss):
                #     #         min_current_loss = current_loss
                #     #         best_checkpoint_path = file_name
                #     #         # TODO invalidation for 30minute vs daily data

                #     loguru_logger.info(f"Loading best checkpoint from {best_checkpoint_path}")
                #     nforecast.load(checkpoints_dir)
                if Y_train_df.empty:
                    loguru_logger.info(f"No training data for {instrument_name}")
                    continue
                # if retrain:
                #     nforecast.fit(df=Y_train_df)
                #     # todo save only best during training
                #     nforecast.save(str(checkpoints_dir), save_dataset=False, overwrite=True)
                #     Y_hat_df = nforecast.predict().reset_index()
                # else:
                #     # fit with much less epocs like 10 or something
                #     Y_hat_df = nforecast.predict(df=Y_train_df).reset_index()

                
                # Y_hat_df = Y_test_df.merge(Y_hat_df, how='left', on=['unique_id', 'ds'])
                # actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
                # baseline_predictions = Baseline().predict(val_dataloader)
                # loguru_logger.info((actuals[:-1] - baseline_predictions[:-1]).abs().mean().item())

                # trainer = pl.Trainer(
                #     gpus=0,
                #     # clipping gradients is a hyperparameter and important to prevent divergance
                #     # of the gradient for recurrent neural networks
                #     gradient_clip_val=0.1,
                # )

                # best hyperpramams look like:
                # {'gradient_clip_val': 0.05690473137493243, 'hidden_size': 50, 'dropout': 0.23151352460442215,
                #  'hidden_continuous_size': 22, 'attention_head_size': 2, 'learning_rate': 0.0816548812864903}
                # best_hyperparams_save_file = f"data/test_study{pred_name}{key_to_predict}{instrument_name}.pkl"
                # params = None
                # try:
                #     params = pickle.load(open(best_hyperparams_save_file, "rb"))
                # except FileNotFoundError:
                #     # logger.info("No best hyperparams found, tuning")
                #     best_hyperparams_save_file = f"data/test_study{instrument_name}.pkl"
                #     try:
                #         params = pickle.load(open(best_hyperparams_save_file, "rb"))
                #     except FileNotFoundError:
                #         # logger.info("No best hyperparams found, tuning")
                #         pass
                # added_best_params = {}
                # if params and params.best_trial.params:
                #     added_best_params = params.best_trial.params
                # added_params = {  # not meaningful for finding the learning rate but otherwise very important
                #     "learning_rate": 0.03,
                #     "hidden_size": 16,  # most important hyperparameter apart from learning rate
                #     # number of attention heads. Set to up to 4 for large datasets
                #     "attention_head_size": 1,
                #     "dropout": 0.1,  # between 0.1 and 0.3 are good values
                #     "hidden_continuous_size": 8,  # set to <= hidden_size
                #     "output_size": 1,  # 7 quantiles by default
                #     "loss": TradingLoss(),
                #     # "logging_metrics": TradingLossBinary(),
                #     # reduce learning rate if no improvement in validation loss after x epochs
                #     "reduce_on_plateau_patience": 4,
                # }
                # added_params.update(added_best_params)
                # if type(added_params['output_size']) == int:
                #     if type(target_to_pred) == list:
                #         added_params['output_size'] = [added_params['output_size']] * len(target_to_pred)
                # gradient_clip_val = added_params.pop("gradient_clip_val", 0.1)
                # tft = TemporalFusionTransformer.from_dataset(
                #     training,
                #     **added_params
                # )
                #
                # loguru_logger.info(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")
                #
                # early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=4, verbose=False,
                #                                     mode="min")
                # model_checkpoint = ModelCheckpoint(
                #     monitor="val_loss",
                #     mode="min",
                #     # save a few of the top models
                #     # incase one does great on other metrics than just val_loss
                #     save_top_k=1,
                #     verbose=True,
                #     filename=instrument_name + "_{epoch}_{val_loss:.8f}",
                # )
                # lr_logger = LearningRateMonitor()  # log the learning rate
                # logger = TensorBoardLogger(f"lightning_logs/{pred_name}/{key_to_predict}/{instrument_name}")  # logging results to a tensorboard
                # trainer = pl.Trainer(
                #     max_epochs=100,
                #     gpus=1,
                #     weights_summary="top",
                #     gradient_clip_val=gradient_clip_val,
                #     # track_grad_norm=2,
                #     # auto_lr_find=True,
                #     # auto_scale_batch_size='binsearch',
                #     limit_train_batches=30,  # coment in for training, running valiation every 30 batches
                #     # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
                #     callbacks=[lr_logger, early_stop_callback, model_checkpoint],
                #     logger=logger,
                # )
                # # retrain = False # todo reenable
                # # try find specific hl net
                #
                # checkpoints_dir = (base_dir / 'lightning_logs' / pred_name / key_to_predict / instrument_name)
                # checkpoint_files = list(checkpoints_dir.glob(f"**/*.ckpt"))
                # if len(checkpoint_files) == 0:
                #     loguru_logger.info("No min+open/low specific checkpoints found, training from other checkpoint")
                #     checkpoints_dir = (base_dir / 'lightning_logs' / pred_name / instrument_name)
                #     checkpoint_files = list(checkpoints_dir.glob(f"**/*.ckpt"))
                #     if len(checkpoint_files) == 0:
                #         loguru_logger.info("No min specific checkpoints found, training from other checkpoint")
                #         checkpoints_dir = (base_dir / 'lightning_logs' / instrument_name)
                #         checkpoint_files = list(checkpoints_dir.glob(f"**/*.ckpt"))
                # best_tft = tft
                #
                # if checkpoint_files:
                #
                #     best_checkpoint_path = checkpoint_files[0]
                #     # sort by most recent checkpoint_files
                #     checkpoint_files.sort(key=lambda x: os.path.getctime(x))
                #     # load the most recent
                #     best_checkpoint_path = checkpoint_files[-1]
                #     # find best checkpoint
                #     # min_current_loss = str(checkpoint_files[0]).split("=")[-1][0:len('.ckpt')]
                #     # for file_name in checkpoint_files:
                #     #     current_loss = str(file_name).split("=")[-1][0:len('.ckpt')]
                #     #     if float(current_loss) < float(min_current_loss):
                #     #         min_current_loss = current_loss
                #     #         best_checkpoint_path = file_name
                #     #         # TODO invalidation for 30minute vs daily data
                #
                #     loguru_logger.info(f"Loading best checkpoint from {best_checkpoint_path}")
                #     best_tft = TemporalFusionTransformer.load_from_checkpoint(str(best_checkpoint_path))
                #
                # if retrain:
                #     trainer.fit(
                #         best_tft,
                #         train_dataloader=train_dataloader,
                #         val_dataloaders=val_dataloader,
                #     )
                #     best_model_path = trainer.checkpoint_callback.best_model_path
                #     best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
                # Y_hat_df = nforecast.predict().reset_index()


                load_pipeline()
                predictions = []
                # make 7 predictions - todo can batch this all in 1 go
                for pred_idx in reversed(range(1, 8)):
                    
                    current_context = price[:-pred_idx]
                    context = torch.tensor(current_context["y"].values, dtype=torch.float).unsqueeze(1)

                    prediction_length = 1
                    forecast = pipeline.predict(
                        context,
                        prediction_length,
                        num_samples=1,
                        temperature=1.0,
                        top_k=1,
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
    loguru_logger.info(f"total_buy_val_loss: {total_buy_val_loss / len(csv_files)}")
    loguru_logger.info(f"total_profit avg per symbol: {total_profit / len(csv_files)}")
    return pd.read_csv(save_file_name)


def df_to_torch(df):
    return torch.tensor(df.values, dtype=torch.float)


if __name__ == "__main__":
    make_predictions()
