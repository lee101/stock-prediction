import pandas as pd
import torch

from data_utils import drop_n_rows
from loss_utils import percent_movements_augment, calculate_takeprofit_torch


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


def test_to_augment_percent():
    assert percent_movements_augment(torch.tensor([100.,150., 50.])) == [1,0.5, -0.666]


def test_calculate_takeprofit_torch():
    profit = calculate_takeprofit_torch(None, torch.tensor([1.2, 1.3]), torch.tensor([1.1, 1.1]), torch.tensor([1.2, 1.05]))
    assert profit == 1.075



def test_calculate_takeprofit_torch_should_be_save_left():
    y_test_pred = torch.tensor([1.5, 1.55])
    leaving_profit = calculate_takeprofit_torch(None, torch.tensor([1.2, 1.3]), torch.tensor([1.1, 1.1]), y_test_pred)
    y_test_pred2 = torch.tensor([1.4, 1.34])

    leaving_profit2 = calculate_takeprofit_torch(None, torch.tensor([1.2, 1.3]), torch.tensor([1.1, 1.1]), y_test_pred2)

    assert leaving_profit == leaving_profit2
