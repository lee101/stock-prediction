import autoPyTorch
import pandas as pd

import transformers
from pathlib import Path
transformers.set_seed(42)

base_dir = Path(__file__).parent.parent
print(base_dir)
from sklearn.preprocessing import MinMaxScaler

cls = autoPyTorch.api.tabular_classification.TabularRegressionTask()

def load_stock_data_from_csv(csv_file_path: Path):
    """
    Loads stock data from csv file.
    """
    return pd.read_csv(csv_file_path)

def train_test_split(stock_data: pd.DataFrame, test_size = 400):
    """
    Splits stock data into train and test sets.
    test_size : int, number of examples be used for test set.
    """
    X_train = stock_data.iloc[:-test_size]
    X_test = stock_data.iloc[-test_size:]
    return X_train, X_test


# Make predictions for all csv files in ectory

def get_labels(X_train):
    """
    Predict the next day closing price "Close"
    :param X_train:
    :return:
    """



    return X_train["Close"]


def pre_process_data(X_train):
    X_train = X_train.drop(columns=["Volume",
                                    "Ex - Dividend",

                                    "Split",
                                    "Ratio",
                                    "Adj.Open",
                                    "Adj.High",
                                    "Adj.Low",
                                    "Adj.Close",
                                    "Adj.Volume",
                                    ])

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train['Close'] = scaler.fit_transform(X_train['Close'].values.reshape(-1, 1))

    return X_train

def make_predictions():
    """
    Make predictions for all csv files in directory.
    """
    for csv_file in (base_dir / "data").glob('*.csv'):
        print(csv_file)
        stock_data = load_stock_data_from_csv(csv_file)
        X_train, X_test = train_test_split(stock_data)

        X_train = pre_process_data(X_train)
        X_test = pre_process_data(X_test)

        Y_train = get_labels(X_train)
        Y_test = get_labels(X_test)

        cls.search(X_train, Y_train)
        predictions = cls.predict(X_test)





if __name__ == "__main__":
    model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    make_predictions()
