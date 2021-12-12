import pandas as pd

from data_utils import drop_n_rows


def test_drop_n_rows():
    df = pd.DataFrame()
    df["a"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    drop_n_rows(df, n=2)
    assert df["a"] == [2,4,6,8,10]

def test_drop_n_rows_three():
    df = pd.DataFrame()
    df["a"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    drop_n_rows(df, n=3) # drops every third
    assert df["a"] == [2,4,6,8,10]
