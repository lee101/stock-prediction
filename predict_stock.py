import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers

from data_utils import split_data, drop_n_rows
from loss_utils import calculate_trading_profit, calculate_trading_profit_torch, DEVICE, torch_inverse_transform, \
    calculate_trading_profit_no_scale, get_trading_profits_list, percent_movements_augment, \
    percent_movements_augment_to, calculate_takeprofit_torch
from model import GRU, GRU

from neuralprophet import NeuralProphet

transformers.set_seed(42)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_dir = Path(__file__).parent
print(base_dir)
from sklearn.preprocessing import MinMaxScaler


from torch.utils.tensorboard import SummaryWriter

current_date_formatted = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
tb_writer = SummaryWriter(log_dir=f"./logs/{current_date_formatted}")

def load_stock_data_from_csv(csv_file_path: Path):
    """
    Loads stock data from csv file.
    """
    return pd.read_csv(csv_file_path)


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


torch.autograd.set_detect_anomaly(True)

def make_predictions(input_data_path=None):
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
            training_mode = "predict"
            for key_to_predict in [
                # "Close",
                # 'High',
                # 'Low',
            ]:  # , 'TakeProfit', 'StopLoss']:
                stock_data = load_stock_data_from_csv(csv_file)
                stock_data = stock_data.dropna()
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

                num_epochs = 100
                hist = np.zeros(num_epochs)
                y_train_pred = None
                min_val_loss = np.inf
                best_y_test_pred_inverted = []

                # Number of steps to unroll
                for epoc_idx in range(num_epochs):
                    model.train()
                    random_aug = torch.rand(x_train.shape) * .0002 - .0001
                    augmented = x_train + random_aug.to(DEVICE)
                    y_train_pred = model(augmented)

                    loss = criterion(y_train_pred, y_train)
                    print("Epoch ", epoc_idx, "MSE: ", loss.item())
                    tb_writer.add_scalar(f"{key_to_predict}/Loss/{instrument_name}/train", loss.item(), epoc_idx)
                    hist[epoc_idx] = loss.item()

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0) # nans caused by overflow in linear la
                    optimiser.step()
                    optimiser.zero_grad()
                    ## test
                    model.eval()

                    y_test_pred = model(x_test)
                    # invert predictions
                    y_test_pred_inverted = torch_inverse_transform(scaler, y_test_pred)
                    # y_train_pred_inverted = torch_inverse_transform(scaler, y_train_pred)

                    # print(y_test_pred_inverted)
                    loss = criterion(y_test_pred, y_test)
                    print(f"val loss: {loss}")
                    tb_writer.add_scalar(f"{key_to_predict}/Loss/{instrument_name}/val", loss.item(), epoc_idx)
                    print(f"Last prediction: y_test_pred_inverted[-1] = {y_test_pred_inverted[-1]}")
                    tb_writer.add_scalar(f"{key_to_predict}/Prediction/{instrument_name}last_pred", y_test_pred_inverted[-1], epoc_idx)

                    # detached_y_test = y_test.detach().cpu().numpy()
                    last_values = x_test[:, -1, 0]
                    # predict trade if last value is above the prediction
                    trading_preds = (y_test_pred > last_values) * 2 - 1
                    last_values = x_test[:, -1, 0]
                    calculated_profit = calculate_trading_profit_torch(scaler, last_values, y_test[:, 0], trading_preds[:, 0]).item()
                    print(f"{instrument_name}: {key_to_predict} calculated_profit: {calculated_profit}")
                    tb_writer.add_scalar(f"{key_to_predict}/Profit/{instrument_name}:  calculated_profit", calculated_profit, epoc_idx)
                    if loss < min_val_loss:
                        min_val_loss = loss
                        torch.save(model.state_dict(), "data/model.pth")
                        best_y_test_pred_inverted = y_test_pred_inverted
                        best_y_test_pred = y_test_pred
                        best_y_train_pred = y_train_pred
                        # percent estimate
                        y_test_end_scaled_loss = torch_inverse_transform(scaler, torch.add(y_test_pred, loss))

                        likely_percent_uncertainty = (
                            (y_test_end_scaled_loss[-1] - y_test_pred_inverted[-1])
                            / y_test_pred_inverted[-1]
                        ).item()
                        min_loss_trading_profit = calculated_profit


                training_time = datetime.now() - start_time
                print("Training time: {}".format(training_time))
                tb_writer.add_scalar("Time/epoc", training_time.total_seconds(), timing_idx)
                timing_idx += 1

                # print(scaler.inverse_transform(y_train_pred.detach().cpu().numpy()))
                # tb_writer.add_scalar("Prediction/train", y_train_pred_inverted[-1], 0)

                val_loss = loss.item()
                # percent_movement = (
                #     best_y_test_pred_inverted[-1].item() - last_close_price
                # ) / last_close_price
                last_preds[key_to_predict.lower() + "_last_price"] = last_close_price
                last_preds[key_to_predict.lower() + "_predicted_price"] = best_y_test_pred_inverted[
                    -1
                ].item()
                last_preds[key_to_predict.lower() + "_val_loss"] = val_loss
                last_preds[key_to_predict.lower() + "min_loss_trading_profit"] = min_loss_trading_profit
                # last_preds[key_to_predict.lower() + "_percent_movement"] = percent_movement
                # last_preds[
                #     key_to_predict.lower() + "_likely_percent_uncertainty"
                # ] = likely_percent_uncertainty
                # last_preds[key_to_predict.lower() + "_minus_uncertainty"] = (
                #     percent_movement - likely_percent_uncertainty
                # )
                total_val_loss += val_loss


            ##### Now train other network to predict buy or sell leverage

            key_to_predict = "Close"
            for training_mode in [
                # "BuyOrSell",
                "TakeProfit",
                # "StopLoss",
            ]:
                print(f"training mode: {training_mode} {instrument_name}")
                stock_data = load_stock_data_from_csv(csv_file)
                stock_data = stock_data.dropna()
                if stock_data.empty:
                    print(f"Empty data for {instrument_name}")
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
                next_day_high_percents = percent_movements_augment_to(stock_data["Open"], stock_data['High'])
                next_day_high_percents.drop(next_day_high_percents.tail(1).index, inplace=True)  # drop last row because of percent change augmentation

                data = pre_process_data(stock_data, "High")
                # todo scaler for each, this messes up the scaler
                data = pre_process_data(data, "Low")
                data = pre_process_data(data, "Open")
                data = pre_process_data(data, key_to_predict)
                price = data[[key_to_predict, "High", "Low", "Open"]]
                # if training_mode == "TakeProfit":
                #     price['movementToHigh'] = percent_movements_augment_to(price["Open"], price['High'])
                # close to high growth in change, a bit wierd
                price.drop(price.tail(1).index, inplace=True) # drop last row because of percent change augmentation
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
                x_ndhp_train, y_ndhp_train, x_ndhp_test, y_ndhp_test = split_data(
                    next_day_high_percents.to_frame(), lookback)

                x_train = torch.from_numpy(x_train).type(torch.Tensor).to(DEVICE)
                x_test = torch.from_numpy(x_test).type(torch.Tensor).to(DEVICE)
                y_train = torch.from_numpy(y_train).type(torch.Tensor).to(DEVICE)
                y_test = torch.from_numpy(y_test).type(torch.Tensor).to(DEVICE)

                x_ndhp_test = torch.from_numpy(x_ndhp_test).type(torch.Tensor).to(DEVICE)
                y_ndhp_test = torch.from_numpy(y_ndhp_test).type(torch.Tensor).to(DEVICE)
                x_ndhp_train = torch.from_numpy(x_ndhp_train).type(torch.Tensor).to(DEVICE)
                y_ndhp_train = torch.from_numpy(y_ndhp_train).type(torch.Tensor).to(DEVICE)

                input_dim = x_train.shape[-1]
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
                optimiser = None
                if "TakeProfit" == training_mode:
                    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
                else:
                    optimiser = torch.optim.AdamW(model.parameters(), lr=0.001)

                start_time = datetime.now()

                num_epochs = 1000 #100000 TODO more is better
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
                            print(f"{instrument_name} loss is nan")
                            print(f"{last_values} last_values")
                            print(f"{y_train} last_values")
                            print(f"{y_train_pred} last_values")
                            continue

                        # add depreciation loss
                        # loss -= len(y_train) * (.001 / 365)

                        # current_profit [:, 0]= calculate_trading_profit(scaler, x_train, y_train.detach().cpu().numpy()[:, 0],
                        #                                           y_train_pred.detach().cpu().numpy())

                        print(f"{training_mode} current_profit: {-loss}")
                        tb_writer.add_scalar(f"{instrument_name}/{training_mode}/current_profit/train", -loss, epoc_idx)
                    elif "TakeProfit" == training_mode:
                        # sigmoid = torch.nn.Sigmoid()
                        # y_train_pred = sigmoid(y_train_pred)
                        ssign = torch.nn.Softsign()
                        y_train_pred = ssign(y_train_pred) * .1
                        ## map to three trinary predictions -1 0 and 1
                        # y_train_pred = torch.round(y_train_pred) # turn off rounding because ruins gradient
                        # y_train_pred = torch.clamp(y_train_pred, -1, 1)
                        # compute percent movement between y_train and last_values

                        last_values = x_train[:, -1, 0]
                        # loss = -calculate_takeprofit_torch(scaler, y_ndhp_train[:, 0], y_train[:, 0],
                        #                                        y_train_pred[:, 0])
                        loss = -calculate_takeprofit_torch(scaler, y_ndhp_train[:, 0], y_train[:, 0],
                                                               y_train_pred[:, 0])

                        ## log if loss is nan
                        if np.isnan(loss.item()):
                            print(f"{instrument_name} loss is nan")
                            print(f"{last_values} last_values")
                            print(f"{y_train} last_values")
                            print(f"{y_train_pred} last_values")
                            continue

                        # add depreciation loss
                        # loss -= len(y_train) * (.001 / 365)

                        # current_profit [:, 0]= calculate_trading_profit(scaler, x_train, y_train.detach().cpu().numpy()[:, 0],
                        #                                           y_train_pred.detach().cpu().numpy())

                        print(f"{training_mode} current_profit: {-loss}")
                        tb_writer.add_scalar(f"{instrument_name}/{training_mode}/current_profit/train", -loss,
                                             epoc_idx)

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
                    #     print(f"current_profit: {current_profit}")
                    #     tb_writer.add_scalar(f"{instrument_name}/{training_mode}/current_profit/test", current_profit, epoc_idx)
                    print("Epoch ", epoc_idx, "MSE: ", loss.item())
                    tb_writer.add_scalar(f"{instrument_name}/{training_mode}/loss", loss.item(), epoc_idx)
                    hist[epoc_idx] = loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0) # nans caused by overflow in linear la
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

                    # print(y_test_pred_inverted)
                    # loss = criterion(y_test_pred, y_test)
                    if "BuyOrSell" == training_mode:
                        # sigmoid = torch.nn.Sigmoid()
                        # y_test_pred = sigmoid(y_test_pred)
                        ssign = torch.nn.Softsign()
                        y_test_pred = ssign(y_test_pred) * 4  # 4x leverage
                        ## map to three trinary predictions -1 0 and 1
                        # y_test_pred = torch.round(y_test_pred)  # turn off rounding because ruins gradient
                        # y_test_pred = torch.clamp(y_test_pred, -4, 4) # 4x leverage

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
                        print(f"{training_mode} current_profit validation: {-loss}")
                        tb_writer.add_scalar(f"{instrument_name}/{training_mode}/current_profit/validation", -loss, epoc_idx)
                    if "TakeProfit" == training_mode:
                        ssign = torch.nn.Softsign()
                        y_test_pred = ssign(y_test_pred) * .1 # max 10% added for sell
                        ## map to three trinary predictions -1 0 and 1
                        # y_test_pred = torch.round(y_test_pred)  # turn off rounding because ruins gradient
                        # y_test_pred = torch.clamp(y_test_pred, -4, 4)  # 4x leverage

                        # y_test_inverted = torch_inverse_transform(scaler, y_test)
                        # plot trading graph

                        # negative as profit is good
                        last_values = x_test[:, -1, 0]
                        loss = -calculate_takeprofit_torch(scaler, y_ndhp_test[:, 0], y_test[:, 0], y_test_pred[:, 0])
                        # trading_profits_list = get_trading_profits_list(scaler, last_values, y_test[:, 0],
                        #                                                 y_test_pred[
                        #                                                 :, 0])
                        # add depreciation loss
                        # loss -= len(y_test) * (.001 / 365)

                        # current_profit [:, 0]= calculate_trading_profit(scaler, x_test, y_test.detach().cpu().numpy(), y_test_pred.detach().cpu().numpy()[:, 0])
                        print(f"{training_mode} current_profit validation: {-loss}")
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
                    #     print(f"{training_mode} current_profit validation: {current_profit}")
                    #     tb_writer.add_scalar(f"{instrument_name}/{training_mode}/current_profit/validation", current_profit, t)
                    print(f"{training_mode} val loss: {loss}")
                    print(
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
                            # tb_writer.add_scalar(f"{instrument_name}/{training_mode}/trading_profits/test", trading_profits_list[i], i)
                    else:
                        number_of_unsuccessful_epochs += 1

                    if number_of_unsuccessful_epochs > 40:
                        print(f"{instrument_name}/{training_mode} Early stopping")
                        break
                training_time = datetime.now() - start_time
                print("Training time: {}".format(training_time))
                print("Best val loss: {}".format(min_val_loss))
                print("Best current profit: {}".format(best_current_profit))

                # print(scaler.inverse_transform(y_train_pred.detach().cpu().numpy()))

                val_loss = loss.item()
                last_preds[training_mode.lower() + "_buy_no_or_sell"] = best_y_test_pred[
                    -1
                ].item()
                last_preds[training_mode.lower() + "_val_loss_classifier"] = val_loss
                last_preds[training_mode.lower() + "_val_profit"] = best_current_profit
                total_buy_val_loss += val_loss
                total_profit += best_current_profit

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

    print(f"val_loss: {total_val_loss / len(csv_files)}")
    print(f"total_buy_val_loss: {total_buy_val_loss / len(csv_files)}")
    print(f"total_profit avg per symbol: {total_profit / len(csv_files)}")

def df_to_torch(df):
    return torch.tensor(df.values, dtype=torch.float)
if __name__ == "__main__":
    make_predictions()
