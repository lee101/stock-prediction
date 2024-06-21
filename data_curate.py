import datetime

import matplotlib.pyplot as plt
import pandas_datareader.data as web
from pandas.plotting import register_matplotlib_converters

from predict_stock import base_dir

"""
Downloads daily stock data from nasdaq

arqit? ARQQ
SHOP
TEAM?
PFE
MRNA
"""


def download_daily_stock_data(path=None):
    symbols = [
        # 'COUR',
        'GOOG',
        'TSLA',
        'NVDA',
        'AAPL',
        # "GTLB", not quite enough daily data yet :(
        # "AMPL", # ampl cant be sold short for somereason or visualised
        "U",
        "ADSK",
        "RBLX",
        "CRWD",
        "ADBE",
        "NET",
        'COIN',
        # 'QUBT',
        # 'ARQQ',
        # avoiding .6% buffer
        'REA.AX',
        'XRO.AX',
        'SEK.AX',
        'NXL.AX', # data analytics
        'APX.AX', # data collection for ml/labelling
        'CDD.AX',
        'NVX.AX',
        'BRN.AX', # brainchip
        'AV1.AX',
        # 'TEAM',
        # 'PFE',
        # 'MRNA',
        'MSFT',
        'AMD',
    # ]
    # symbols = [
        'BTCUSD',
        'ETHUSD',
        'LTCUSD',
        "PAXGUSD", "UNIUSD"

    ]
    save_path = base_dir / 'data'
    if path:
        save_path = base_dir / 'data' / path
    save_path.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        # longer didnt help
        # download less data for inference mode
        start = datetime.datetime(2021, 6, 6)
        # start = datetime.datetime(2017, 1, 1)
        end = datetime.datetime.now()
        df = web.DataReader(symbol, 'yahoo', start, end)

        file_save_path = (save_path / '{}-{}.csv'.format(symbol, end))
        df.to_csv(file_save_path)
    return df


def visualize_stock_data(df):
    register_matplotlib_converters()
    df.plot(x='Date', y='Close')
    plt.show()


if __name__ == '__main__':
    df = download_daily_stock_data()
    visualize_stock_data(df)
