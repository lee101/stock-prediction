import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers

from model import GRU

transformers.set_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_dir = Path(__file__).parent
print(base_dir)
from sklearn.preprocessing import MinMaxScaler


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

    x_train[key_to_predict] = scaler.fit_transform(x_train[key_to_predict].values.reshape(-1, 1))

    return x_train


def make_predictions():
    """
    Make predictions for all csv files in directory.
    """
    results_dir = base_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_file_name = results_dir / f"predictions-{time}.csv"
    CSV_KEYS = [
        'instrument',
        'close_last_price',
        'close_predicted_price',
        'close_val_loss',
        'close_percent_movement',
        "close_likely_percent_uncertainty",
        "close_minus_uncertainty",
        'high_last_price',
        'high_predicted_price',
        'high_val_loss',
        'high_percent_movement',
        "high_likely_percent_uncertainty",
        "high_minus_uncertainty",
        'low_last_price',
        'low_predicted_price',
        'low_val_loss',
        'low_percent_movement',
        "low_likely_percent_uncertainty",
        "low_minus_uncertainty",
    ]
    with open(save_file_name, "a") as f:
        writer = csv.DictWriter(f, CSV_KEYS)
        writer.writeheader()

    # experiment with shared weights roughly a failure
    # input_dim = 1
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
    csv_files = list((base_dir / "data").glob('*.csv'))

    for days_to_drop in [0]:#[1,2,3,4,5,6,7,8,9,10,11]:
        for csv_file in csv_files:
            last_preds = {
                "instrument": csv_file.stem,
            }
            for key_to_predict in ['Close', 'High', 'Low']:
                stock_data = load_stock_data_from_csv(csv_file)
                stock_data = stock_data.dropna()
                # drop last days_to_drop rows
                if days_to_drop:
                    stock_data = stock_data.iloc[:-days_to_drop]

                # x_train, x_test = train_test_split(stock_data)
                last_close_price = stock_data[key_to_predict].iloc[-1]
                data = pre_process_data(stock_data, key_to_predict)
                price = data[[key_to_predict]]

                # x_test = pre_process_data(x_test)

                def split_data(stock, lookback):
                    data_raw = stock.to_numpy()  # convert to numpy array
                    data = []

                    # create all possible sequences of length seq_len
                    for index in range(lookback, len(data_raw) + 1):
                        data.append(data_raw[index - lookback: index])

                    data = np.array(data)
                    test_set_size = int(np.round(0.1 * data.shape[0]))
                    train_set_size = data.shape[0] - (test_set_size)

                    x_train = data[:train_set_size, :-1, :]
                    y_train = data[:train_set_size, -1, :]

                    x_test = data[train_set_size:, :-1]
                    y_test = data[train_set_size:, -1, :]

                    return [x_train, y_train, x_test, y_test]

                lookback = 20  # choose sequence length , GTLB only has been open for 27days cant go over that :O
                if len(price) > 40:
                    lookback = 30
                x_train, y_train, x_test, y_test = split_data(price, lookback)


                x_train = torch.from_numpy(x_train).type(torch.Tensor).to(device)
                x_test = torch.from_numpy(x_test).type(torch.Tensor).to(device)
                y_train = torch.from_numpy(y_train).type(torch.Tensor).to(device)
                y_test = torch.from_numpy(y_test).type(torch.Tensor).to(device)

                input_dim = 1
                hidden_dim = 32
                num_layers = 2
                output_dim = 1

                model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
                model.to(device)
                model.train()
                criterion = torch.nn.L1Loss(reduction='mean')
                optimiser = torch.optim.AdamW(model.parameters(), lr=0.01)

                start_time = datetime.now()

                num_epochs = 100
                hist = np.zeros(num_epochs)
                y_train_pred = None
                min_val_loss = np.inf
                best_y_test_pred_inverted = []

                # Number of steps to unroll
                for t in range(num_epochs):
                    model.train()
                    y_train_pred = model(x_train)

                    loss = criterion(y_train_pred, y_train)
                    print("Epoch ", t, "MSE: ", loss.item())
                    hist[t] = loss.item()

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    ## test
                    model.eval()

                    y_test_pred = model(x_test)
                    # invert predictions
                    y_test_pred_inverted = scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
                    y_train_pred_inverted = scaler.inverse_transform(y_train_pred.detach().cpu().numpy())

                    # print(y_test_pred_inverted)
                    loss = criterion(y_test_pred, y_test)
                    print(f"val loss: {loss}")
                    print(f"Last prediction: y_test_pred_inverted[-1] = {y_test_pred_inverted[-1]}")
                    if loss < min_val_loss:
                        min_val_loss = loss
                        torch.save(model.state_dict(), "data/model.pth")
                        best_y_test_pred_inverted = y_test_pred_inverted
                        # percent estimate
                        y_test_end_scaled_loss = scaler.inverse_transform(np.add(y_test_pred.detach().cpu().numpy(), loss.detach().cpu().numpy()))
                        likely_percent_uncertainty = ((y_test_end_scaled_loss[-1] - y_test_pred_inverted[-1]) / \
                                                      y_test_pred_inverted[-1]).item()

                training_time = datetime.now() - start_time
                print("Training time: {}".format(training_time))

                print(scaler.inverse_transform(y_train_pred.detach().cpu().numpy()))

                val_loss = loss.item()
                percent_movement = (best_y_test_pred_inverted[-1].item() - last_close_price) / last_close_price
                last_preds[key_to_predict.lower() + "_last_price"] = last_close_price
                last_preds[key_to_predict.lower() + "_predicted_price"] = best_y_test_pred_inverted[-1].item()
                last_preds[key_to_predict.lower() + "_val_loss"] = val_loss
                last_preds[key_to_predict.lower() + "_percent_movement"] = percent_movement
                last_preds[key_to_predict.lower() + "_likely_percent_uncertainty"] = likely_percent_uncertainty
                last_preds[key_to_predict.lower() + "_minus_uncertainty"] = percent_movement - likely_percent_uncertainty
                total_val_loss += val_loss

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


if __name__ == "__main__":
    make_predictions()
