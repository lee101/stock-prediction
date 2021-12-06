from predict_stock import calculate_trading_profit

def test_calculate_trading_profit():
    x_test = [1., 2.]
    y_test = [1.5, 3.]
    y_test_pred = [1.5, 3.]
    assert calculate_trading_profit(x_test, y_test, y_test_pred) == 0.5
