from src.process_utils import backout_near_market


def test_backout_near_market():
    backout_near_market("BTCUSD")
    print('done')


def test_ramp_into_position():
    from src.process_utils import ramp_into_position
    ramp_into_position("TSLA", "buy")
    print('done')
