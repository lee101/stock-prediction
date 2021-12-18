from datetime import datetime

from data_curate import download_daily_stock_data
# from predict_stock import make_predictions
from decorator_utils import timeit
from predict_stock_forecasting import make_predictions

@timeit
def do_forecasting():
    if True:
        current_time_formatted = '2021-12-05 18:20:29'
        current_time_formatted = '2021-12-09 12:16:26'  # new/ more data
        current_time_formatted = '2021-12-11 07:57:21-2'  # new/ less data tickers
        # current_time_formatted = 'min' # new/ less data tickers
    else:
        current_time_formatted = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        download_daily_stock_data(current_time_formatted)
    make_predictions(current_time_formatted)


if __name__ == '__main__':
    # in development, use the following line to avoid re downloading data
    do_forecasting()

