import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters

from predict_stock import base_dir

"""
Downloads daily stock data from nasdaq
"""
def download_daily_stock_data():


    symbols = ['COUR', 'GOOG', 'TSLA', 'NVDA', "GTLB", "AMPL", "U", "ADSK", "RBLX", "CRWD", "ADBE", "NET", 'COIN']
    for symbol in symbols:

        start = datetime.datetime(2017, 1, 1)
        end = datetime.datetime.now()
        df = web.DataReader(symbol, 'yahoo', start, end)
        file_save_path = (base_dir / 'data' / '{}-{}.csv'.format(symbol, end))
        df.to_csv(file_save_path)
    return df


def visualize_stock_data(df):
    register_matplotlib_converters()
    df.plot(x='Date', y='Close')
    plt.show()


if __name__ == '__main__':
    df = download_daily_stock_data()
    visualize_stock_data(df)
