from datetime import datetime

from data_curate import download_daily_stock_data
from predict_stock import make_predictions

if __name__ == '__main__':
    # in development, use the following line to avoid re downloading data
    if False:
        current_time_formatted = '2021-12-05 18:20:29'
    else:
        current_time_formatted = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        download_daily_stock_data(current_time_formatted)
    make_predictions(current_time_formatted)
