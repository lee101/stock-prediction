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

def train_test_split(stock_data: pd.DataFrame, test_size = 50):
    """
    Splits stock data into train and test sets.
    test_size : int, number of examples be used for test set.
    """
    x_train = stock_data.iloc[:-test_size]
    x_test = stock_data.iloc[-test_size:]
    return x_train, x_test


# Make predictions for all csv files in ectory

def get_labels(x_train):
    """
    Predict the next day closing price "Close"
    :param x_train:
    :return:
    """

    # shift df by 1
    x_train = x_train.shift(-1)
    return x_train["Close"]

scaler = MinMaxScaler(feature_range=(-1, 1))


def pre_process_data(x_train):
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

    x_train['Close'] = scaler.fit_transform(x_train['Close'].values.reshape(-1, 1))

    return x_train

def make_predictions():
    """
    Make predictions for all csv files in directory.
    """
    results_dir = base_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    save_file_name = results_dir / "predictions.csv"
    CSV_KEYS = [
        'instrument',
        'last_close_price',
        'predicted_close_price',
        'val_loss',
    ]
    with open(save_file_name, "a") as f:
        writer = csv.DictWriter(f, CSV_KEYS)
        writer.writeheader()

    last_preds = {}
    for csv_file in (base_dir / "data").glob('*.csv'):
        stock_data = load_stock_data_from_csv(csv_file)
        # x_train, x_test = train_test_split(stock_data)
        last_close_price = stock_data['Close'].iloc[-1]
        data = pre_process_data(stock_data)
        price = data[["Close"]]
        # x_test = pre_process_data(x_test)

        def split_data(stock, lookback):
            data_raw = stock.to_numpy()  # convert to numpy array
            data = []

            # create all possible sequences of length seq_len
            for index in range(len(data_raw) - lookback):
                data.append(data_raw[index: index + lookback])

            data = np.array(data)
            test_set_size = int(np.round(0.2 * data.shape[0]))
            train_set_size = data.shape[0] - (test_set_size)

            x_train = data[:train_set_size, :-1, :]
            y_train = data[:train_set_size, -1, :]

            x_test = data[train_set_size:, :-1]
            y_test = data[train_set_size:, -1, :]

            return [x_train, y_train, x_test, y_test]

        lookback = 20  # choose sequence length
        x_train, y_train, x_test, y_test = split_data(price, lookback)

        # y_train = get_labels(x_train)
        # x_test = get_labels(x_test)

        # x_train = x_train['Close']
        # x_test = x_test['Close']

        x_train = torch.from_numpy(x_train).type(torch.Tensor).to(device)
        x_test = torch.from_numpy(x_test).type(torch.Tensor).to(device)
        y_train = torch.from_numpy(y_train).type(torch.Tensor).to(device)
        y_test = torch.from_numpy(y_test).type(torch.Tensor).to(device)


        input_dim = 1
        hidden_dim = 32
        num_layers = 2
        output_dim = 1
        num_epochs = 100

        model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
        model.to(device)

        criterion = torch.nn.MSELoss(reduction='mean')
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

        start_time = datetime.now()

        num_epochs = 100
        hist = np.zeros(num_epochs)
        y_train_pred = None
        # Number of steps to unroll
        for t in range(num_epochs):
            y_train_pred = model(x_train)

            loss = criterion(y_train_pred, y_train)
            print("Epoch ", t, "MSE: ", loss.item())
            hist[t] = loss.item()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        training_time = datetime.now() - start_time
        print("Training time: {}".format(training_time))

        print(scaler.inverse_transform(y_train_pred.detach().cpu().numpy()))


        ## test
        y_test_pred = model(x_test)
        # invert predictions
        y_test_pred_inverted = scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
        y_train_pred_inverted = scaler.inverse_transform(y_train_pred.detach().cpu().numpy())

        # print(y_test_pred_inverted)
        loss = criterion(y_test_pred, y_test)
        print(f"val loss: {loss}")
        print(f"Last prediction: y_test_pred_inverted[-1] = {y_test_pred_inverted[-1]}")
        last_preds = {
            "instrument": csv_file.stem,
            "last_close_price": last_close_price,
            "predicted_close_price": y_test_pred_inverted[-1].item(),
            "val_loss": loss.item(),
        }
        with open(save_file_name, "a") as f:
            writer = csv.DictWriter(f, CSV_KEYS)
            writer.writerow(last_preds)

        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(price)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[lookback:len(y_train_pred_inverted) + lookback, :] = y_train_pred_inverted

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(price)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(y_train_pred_inverted) + lookback - 1:len(price) - 1, :] = y_test_pred_inverted

        original = scaler.inverse_transform(price['Close'].values.reshape(-1, 1))

        predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
        predictions = np.append(predictions, original, axis=1)
        result = pd.DataFrame(predictions)
        import plotly.express as px
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[0],
                                            mode='lines',
                                            name='Train prediction')))
        fig.add_trace(go.Scatter(x=result.index, y=result[1],
                                 mode='lines',
                                 name='Test prediction'))
        fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[2],
                                            mode='lines',
                                            name='Actual Value')))
        fig.update_layout(
            xaxis=dict(
                showline=True,
                showgrid=True,
                showticklabels=False,
                linecolor='white',
                linewidth=2
            ),
            yaxis=dict(
                title_text='Close (USD)',
                titlefont=dict(
                    family='Rockwell',
                    size=12,
                    color='white',
                ),
                showline=True,
                showgrid=True,
                showticklabels=True,
                linecolor='white',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Rockwell',
                    size=12,
                    color='white',
                ),
            ),
            showlegend=True,
            template='plotly_dark'

        )

        annotations = []
        annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                                xanchor='left', yanchor='bottom',
                                text=csv_file.stem,
                                font=dict(family='Rockwell',
                                          size=26,
                                          color='white'),
                                showarrow=False))
        fig.update_layout(annotations=annotations)

        fig.show()



if __name__ == "__main__":
    make_predictions()
