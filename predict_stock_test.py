import torch

from loss_utils import calculate_trading_profit_no_scale, calculate_trading_profit_torch


# def test_calculate_trading_profit():
#     x_test = [1., 2.]
#     y_test = [1.5, 3.]
#     y_test_pred = [1.5, 3.]
#     assert calculate_trading_profit(x_test, y_test, y_test_pred) == 0.5


def test_calculate_trading_profit_no_scale_buy():
    x_test = torch.tensor([1., 2.])
    y_test = torch.tensor([1., 3.])
    y_test_pred = torch.tensor([1., 1.])
    assert calculate_trading_profit_no_scale(x_test, y_test, y_test_pred).item() == 0.5


def test_calculate_trading_profit_no_scale_buy_leverage():
    x_test = torch.tensor([1., 2.])
    y_test = torch.tensor([1., 3.])
    y_test_pred = torch.tensor([1., 3.])
    assert calculate_trading_profit_no_scale(x_test, y_test, y_test_pred).item() == 1.5


def test_calculate_trading_profit_no_scale_sell():
    x_test = torch.tensor([1., 2.])
    y_test = torch.tensor([1., 1.])
    y_test_pred = torch.tensor([1., -1.])  # how much portfolio to sell/buy
    assert calculate_trading_profit_no_scale(x_test, y_test, y_test_pred).item() == 0.5


def test_calculate_trading_profit_no_scale_sell_leverage():
    x_test = torch.tensor([1., 2.])
    y_test = torch.tensor([1., 1.])
    y_test_pred = torch.tensor([1., -3.])  # how much portfolio to sell/buy
    assert calculate_trading_profit_no_scale(x_test, y_test, y_test_pred).item() == 1.5


# test scaler
def test_calculate_trading_profit_no_scale_buy_leverage_scaler():
    x_test = torch.tensor([1., 2.])
    y_test = torch.tensor([1., 3.])
    y_test_pred = torch.tensor([1., 3.])
    assert calculate_trading_profit_torch(None, x_test, y_test, y_test_pred).item() == 1.5


def test_calculate_trading_profit_no_scale_sell_scaler():
    x_test = torch.tensor([1., 2.])
    y_test = torch.tensor([1., 1.])
    y_test_pred = torch.tensor([1., -1.])  # how much portfolio to sell/buy
    assert calculate_trading_profit_torch(None, x_test, y_test, y_test_pred).item() - 0.5 < .01


def test_calculate_trading_profit_no_scale_sell_leverage_scaler():
    x_test = torch.tensor([1., 2.])
    y_test = torch.tensor([1., 1.])
    y_test_pred = torch.tensor([1., -3.])  # how much portfolio to sell/buy
    assert calculate_trading_profit_torch(None, x_test, y_test, y_test_pred).item() == 1.5


def test_calculate_trading_profit_no_scale_sell_scaler_balanced():
    x_test = torch.tensor([1., 2.])
    y_test = torch.tensor([1., 1.])  # 1.5 per day
    y_test_pred = torch.tensor([1., -1.])  # how much portfolio to sell/buy
    sell_loss = calculate_trading_profit_torch(None, x_test, y_test, y_test_pred).item()
    x_test = torch.tensor([1., 1.])
    y_test = torch.tensor([1., 2.])
    y_test_pred = torch.tensor([1., 1.])  # how much portfolio to sell/buy
    buy_loss = calculate_trading_profit_torch(None, x_test, y_test, y_test_pred).item()
    assert buy_loss == sell_loss


def test_calculate_trading_profit_no_scale_sell_scaler_balanced2():
    x_test = torch.tensor([2.])
    y_test = torch.tensor([1.])  # 1.5 per day
    y_test_pred = torch.tensor([-1.])  # how much portfolio to sell/buy
    sell_loss = calculate_trading_profit_torch(None, x_test, y_test, y_test_pred).item()
    x_test = torch.tensor([1.])
    y_test = torch.tensor([2.])
    y_test_pred = torch.tensor([1.])  # how much portfolio to sell/buy
    buy_loss = calculate_trading_profit_torch(None, x_test, y_test, y_test_pred).item()
    assert buy_loss == sell_loss


def test_calculate_trading_profit_no_scale_sell_scaler_balanced2():
    x_test = torch.tensor([2.])
    y_test = torch.tensor([1.])  # 1.5 per day
    y_test_pred = torch.tensor([-3.])  # how much portfolio to sell/buy
    sell_loss = calculate_trading_profit_torch(None, x_test, y_test, y_test_pred).item()
    x_test = torch.tensor([1.])
    y_test = torch.tensor([2.])
    y_test_pred = torch.tensor([3.])  # how much portfolio to sell/buy
    buy_loss = calculate_trading_profit_torch(None, x_test, y_test, y_test_pred).item()
    assert buy_loss == sell_loss


def test_calculate_trading_profit_no_scale_sell_scaler_balanced2_neg():
    x_test = torch.tensor([2.])
    y_test = torch.tensor([1.])  # 1.5 per day
    y_test_pred = torch.tensor([3.])  # how much portfolio to sell/buy
    sell_loss = calculate_trading_profit_torch(None, x_test, y_test, y_test_pred).item()
    x_test = torch.tensor([1.])
    y_test = torch.tensor([2.])
    y_test_pred = torch.tensor([-3.])  # how much portfolio to sell/buy
    buy_loss = calculate_trading_profit_torch(None, x_test, y_test, y_test_pred).item()
    assert buy_loss == sell_loss


def test_calculate_trading_profit_no_sale():
    x_test = torch.tensor([2.])
    y_test = torch.tensor([1.])  # 1.5 per day
    y_test_pred = torch.tensor([.1])  # how much portfolio to sell/buy
    sell_loss = calculate_trading_profit_torch(None, x_test, y_test, y_test_pred).item()
    x_test = torch.tensor([1.])
    y_test = torch.tensor([2.])
    y_test_pred = torch.tensor([-.1])  # how much portfolio to sell/buy
    buy_loss = calculate_trading_profit_torch(None, x_test, y_test, y_test_pred).item()
    assert buy_loss == sell_loss
    x_test = torch.tensor([2.])
    y_test = torch.tensor([1.])  # 1.5 per day
    y_test_pred = torch.tensor([-.01])  # how much portfolio to sell/buy
    sell_loss = calculate_trading_profit_torch(None, x_test, y_test, y_test_pred).item()
    x_test = torch.tensor([1.])
    y_test = torch.tensor([2.])
    y_test_pred = torch.tensor([.01])  # how much portfolio to sell/buy
    buy_loss = calculate_trading_profit_torch(None, x_test, y_test, y_test_pred).item()
    assert buy_loss == sell_loss
