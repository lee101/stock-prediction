from freezegun import freeze_time

from src.date_utils import is_nyse_trading_day_ending  # replace 'your_module' with the actual module name


@freeze_time("2022-12-15 20:00:00")  # This is 15:00 NYSE time
def test_trading_day_ending():
    assert is_nyse_trading_day_ending() == True


@freeze_time("2022-12-15 23:00:00")  # This is 18:00 NYSE time
def test_trading_day_not_ending():
    assert is_nyse_trading_day_ending() == False
