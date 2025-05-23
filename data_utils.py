import numpy as np


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
    drop n rows for every 1 row in the dataframe
    :param stock:
    :param n:
    :return:
    """
    drop_idxes = np.arange(0, len(df), n)
    df.drop(drop_idxes, inplace=True)

def is_fp_close(number, tol=1e-6):
    return abs(number - round(number)) < tol

def is_fp_close_to_zero(number, tol=1e-6):
    return abs(number) < tol
