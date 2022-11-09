from alpaca_wrapper import latest_data


def test_get_latest_data():
    data = latest_data('BTCUSD')
    print(data)
    data = latest_data('COUR')
    print(data)
