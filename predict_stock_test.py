from loss_utils import calculate_trading_profit, calculate_trading_profit_no_scale
import torch

# def test_calculate_trading_profit():
#     x_test = [1., 2.]
#     y_test = [1.5, 3.]
#     y_test_pred = [1.5, 3.]
#     assert calculate_trading_profit(x_test, y_test, y_test_pred) == 0.5


def test_calculate_trading_profit_no_scale_buy_leverage():
    x_test = torch.tensor([1., 2.])
    y_test = torch.tensor([1., 3.])
    y_test_pred = torch.tensor([1., 3.])
    assert calculate_trading_profit_no_scale(x_test, y_test, y_test_pred).item() == 1.5


def test_calculate_trading_profit_no_scale_sell():
    x_test = torch.tensor([1., 2.])
    y_test = torch.tensor([1., 1.])
    y_test_pred = torch.tensor([1., -1.]) # how much portfolio to sell/buy
    assert calculate_trading_profit_no_scale(x_test, y_test, y_test_pred).item() == 0.5

def test_calculate_trading_profit_no_scale_sell_leverage():
    x_test = torch.tensor([1., 2.])
    y_test = torch.tensor([1., 1.])
    y_test_pred = torch.tensor([1., -3.]) # how much portfolio to sell/buy
    assert calculate_trading_profit_no_scale(x_test, y_test, y_test_pred).item() == 1.5
