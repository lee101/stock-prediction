from src.process_utils import backout_near_market


def test_backout_near_market():
    backout_near_market("BTCUSD")
    print('done')
