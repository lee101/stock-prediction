from alpaca_wrapper import latest_data, has_current_open_position


def test_get_latest_data():
    data = latest_data('BTCUSD')
    print(data)
    data = latest_data('COUR')
    print(data)


def test_has_current_open_position():
    has_position = has_current_open_position('BTCUSD', 'buy') # real
    assert has_position is True
    has_position = has_current_open_position('BTCUSD', 'sell') # real
    assert has_position is False
    has_position = has_current_open_position('LTCUSD', 'buy') # real
    assert has_position is False
