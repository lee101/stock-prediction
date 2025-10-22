from datetime import datetime

import pandas as pd
import torch

from data_utils import drop_n_rows
from loss_utils import (
    percent_movements_augment,
    calculate_takeprofit_torch,
    calculate_trading_profit_torch_with_buysell,
    calculate_trading_profit_torch_with_entry_buysell,
    calculate_profit_torch_with_entry_buysell_profit_values,
)


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


def test_entry_profit_values_match_total_and_flag_wrong_orders():
    y_test = torch.tensor([0.05, -0.04, 0.02], dtype=torch.float32)
    y_test_pred = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float32)
    y_test_high = torch.tensor([0.08, 0.03, 0.05], dtype=torch.float32)
    y_test_high_pred = torch.tensor([0.06, 0.01, 0.04], dtype=torch.float32)
    y_test_low = torch.tensor([-0.03, -0.06, -0.02], dtype=torch.float32)
    y_test_low_pred = torch.tensor([-0.015, -0.05, -0.01], dtype=torch.float32)

    total = calculate_trading_profit_torch_with_entry_buysell(
        None,
        None,
        y_test,
        y_test_pred,
        y_test_high,
        y_test_high_pred,
        y_test_low,
        y_test_low_pred,
    )
    per_period = calculate_profit_torch_with_entry_buysell_profit_values(
        y_test,
        y_test_high,
        y_test_high_pred,
        y_test_low,
        y_test_low_pred,
        y_test_pred,
    )

    assert per_period.shape == y_test.shape
    assert torch.allclose(total, per_period.sum())
    assert per_period.sum().item() > 0

    # regression guard: old argument order produced inconsistent totals
    buggy_values = calculate_profit_torch_with_entry_buysell_profit_values(
        y_test,
        y_test_pred,
        y_test_high,
        y_test_high_pred,
        y_test_low,
        y_test_low_pred,
    )
    assert not torch.allclose(total, buggy_values.sum())


def get_time():
    return datetime.now()
