from datetime import datetime

import pandas as pd
import torch

from data_utils import drop_n_rows
from loss_utils import percent_movements_augment, calculate_takeprofit_torch, \
    calculate_trading_profit_torch_with_buysell, calculate_trading_profit_torch_with_entry_buysell


def test_drop_n_rows():
    df = pd.DataFrame()
    df["a"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    drop_n_rows(df, n=2)
    assert df["a"] == [2, 4, 6, 8, 10]


def test_drop_n_rows_three():
    df = pd.DataFrame()
    df["a"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    drop_n_rows(df, n=3)  # drops every third
    assert df["a"] == [2, 4, 6, 8, 10]


def test_to_augment_percent():
    assert percent_movements_augment(torch.tensor([100., 150., 50.])) == [1, 0.5, -0.666]


def test_calculate_takeprofit_torch():
    profit = calculate_takeprofit_torch(None, torch.tensor([1.2, 1.3]), torch.tensor([1.1, 1.1]),
                                        torch.tensor([1.2, 1.05]))
    assert profit == 1.075


def test_calculate_takeprofit_torch_should_be_save_left():
    y_test_pred = torch.tensor([1.5, 1.55])
    leaving_profit = calculate_takeprofit_torch(None, torch.tensor([1.2, 1.3]), torch.tensor([1.1, 1.1]), y_test_pred)
    y_test_pred2 = torch.tensor([1.4, 1.34])

    leaving_profit2 = calculate_takeprofit_torch(None, torch.tensor([1.2, 1.3]), torch.tensor([1.1, 1.1]), y_test_pred2)

    assert leaving_profit == leaving_profit2


def test_takeprofits():
    profits = calculate_trading_profit_torch_with_buysell(None, None, torch.tensor([.2, -.4]), torch.tensor([1, -1]),
                                                          torch.tensor([.4, .1]), torch.tensor([.5, .2]),
                                                          torch.tensor([-.1, -.6]), torch.tensor([-.2, -.8]),
                                                          )

    assert abs(profits - .6) < .002

    # predict the high
    profits = calculate_trading_profit_torch_with_buysell(None, None, torch.tensor([.2, -.4]), torch.tensor([1, -1]),
                                                          torch.tensor([.4, .1]), torch.tensor([.39, .2]),
                                                          torch.tensor([-.1, -.6]), torch.tensor([-.2, -.8]),
                                                          )

    assert (profits - (.39 + .4)) < .002
    # predict the low
    profits = calculate_trading_profit_torch_with_buysell(None, None, torch.tensor([.2, -.4]), torch.tensor([1, -1]),
                                                          torch.tensor([.4, .1]), torch.tensor([.39, .2]),
                                                          torch.tensor([-.1, -.6]), torch.tensor([-.2, -.59]),
                                                          )

    assert (profits - (.39 + .59)) < .002

    # predict the too low
    profits = calculate_trading_profit_torch_with_buysell(None, None, torch.tensor([.2, -.4]), torch.tensor([1, -1]),
                                                          torch.tensor([.4, .1]), torch.tensor([.39, .2]),
                                                          torch.tensor([-.1, -.6]), torch.tensor([.2, .59]),
                                                          )

    assert (profits - (.39 + .59)) < .002
    # predict both the low/high within to sell
    profits = calculate_trading_profit_torch_with_buysell(None, None, torch.tensor([-.4]),
                                                          torch.tensor([-1]),
                                                          torch.tensor([.2]), torch.tensor([.1]),
                                                          # high/highpreds
                                                          torch.tensor([-.6]), torch.tensor([-.59]),
                                                          # low lowpreds
                                                          )

    assert (profits - (.59)) < .002


def test_entry_takeprofits():
    # no one should enter trades/make anything
    profits = calculate_trading_profit_torch_with_entry_buysell(None, None, torch.tensor([.2, -.4]),
                                                                torch.tensor([1, -1]),
                                                                torch.tensor([.4, .1]), torch.tensor([.5, .2]),
                                                                # high/highpreds
                                                                torch.tensor([-.1, -.6]), torch.tensor([-.2, -.8]),
                                                                # lows/preds
                                                                )

    # assert abs(profits - .6) < .002

    # predict the high only but we buy so nothing should happen
    profits = calculate_trading_profit_torch_with_entry_buysell(None, None, torch.tensor([.2, -.4]),
                                                                torch.tensor([1, -1]),
                                                                torch.tensor([.4, .1]), torch.tensor([.39, .2]),
                                                                torch.tensor([-.1, -.6]), torch.tensor([-.2, -.8]),
                                                                )

    # assert (profits - (.39 + .4)) < .002
    # predict the low but we sell so nothing should happen
    profits = calculate_trading_profit_torch_with_entry_buysell(None, None, torch.tensor([.2, -.4]),
                                                                torch.tensor([1, -1]),
                                                                torch.tensor([.4, .1]), torch.tensor([.39, .2]),
                                                                torch.tensor([-.1, -.6]), torch.tensor([-.2, -.59]),
                                                                )

    # assert (profits - (.39 + .59)) < .002

    # predict both the low/high within
    profits = calculate_trading_profit_torch_with_entry_buysell(None, None, torch.tensor([.2, ]),
                                                                torch.tensor([1, ]),
                                                                torch.tensor([.4]), torch.tensor([.39]),
                                                                # high/highpreds
                                                                torch.tensor([-.1, ]), torch.tensor([-.08, ]),
                                                                )
    # predict both the low/high within
    profits = calculate_trading_profit_torch_with_entry_buysell(None, None, torch.tensor([.2, -.4]),
                                                                torch.tensor([1, -1]),
                                                                torch.tensor([.4, .1]), torch.tensor([.39, .2]),
                                                                # high/highpreds
                                                                torch.tensor([-.1, -.6]), torch.tensor([-.08, -.59]),
                                                                )
    # predict both the low/high within to sell
    profits = calculate_trading_profit_torch_with_entry_buysell(None, None, torch.tensor([-.4]),
                                                                torch.tensor([-1]),
                                                                torch.tensor([.2]), torch.tensor([.1]),
                                                                # high/highpreds
                                                                torch.tensor([-.6]), torch.tensor([-.59]),
                                                                # low lowpreds
                                                                )
    assert (profits - (.1 + .59)) < .002  # TODO take away non trades from trading loss


def get_time():
    return datetime.now()
