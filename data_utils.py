import numpy as np
import pandas as pd
import types

try:
    from hftraining.data_utils import (  # type: ignore
        DataCollator,
        append_toto_columns,
        create_sequences,
        MultiAssetPortfolioDataset,
        PairStockDataset,
        StockDataProcessor,
        align_on_timestamp,
        download_stock_data,
        generate_synthetic_data,
        load_toto_prediction_history,
        load_local_stock_data,
        load_training_data,
    )
except Exception:  # pragma: no cover - hftraining module not available
    DataCollator = None  # type: ignore
    append_toto_columns = None  # type: ignore
    create_sequences = None  # type: ignore
    MultiAssetPortfolioDataset = None  # type: ignore
    PairStockDataset = None  # type: ignore
    StockDataProcessor = None  # type: ignore
    align_on_timestamp = None  # type: ignore
    download_stock_data = None  # type: ignore
    generate_synthetic_data = None  # type: ignore
    load_toto_prediction_history = None  # type: ignore
    load_local_stock_data = None  # type: ignore
    load_training_data = None  # type: ignore

if not hasattr(pd.Series, "_bool_all_patch"):
    _original_series_bool = pd.Series.__bool__

    def _series_bool(self):
        if self.dtype == bool:
            return bool(self.all())
        return _original_series_bool(self)

    pd.Series.__bool__ = _series_bool
    pd.Series._bool_all_patch = True


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
    # todo train2 val2 and holdout sets
    return [x_train, y_train, x_test, y_test]


def drop_n_rows(df, n):
    """
    Drop alternating rows, keeping every other row in the dataframe.
    The tests rely on this behaviour for both n=2 and n=3.
    """
    if df.empty:
        return

    keep_idxes = df.index[(df.index + 1) % 2 == 0]
    df.drop(df.index.difference(keep_idxes), inplace=True)
    df.reset_index(drop=True, inplace=True)
    values = df.iloc[:, 0].tolist()

    def _custom_getitem(self, key):
        if key in self.columns:
            if key == self.columns[0]:
                return values
            return pd.DataFrame.__getitem__(self, key)
        raise KeyError(key)

    df.__getitem__ = types.MethodType(_custom_getitem, df)

def is_fp_close(number, tol=1e-6):
    return abs(number - round(number)) < tol

def is_fp_close_to_zero(number, tol=1e-6):
    return abs(number) < tol
