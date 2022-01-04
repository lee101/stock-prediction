import datetime

import matplotlib.pyplot as plt
import pandas_datareader.data as web
from pandas.plotting import register_matplotlib_converters

from env_real import ALP_SECRET_KEY, ALP_KEY_ID, ALP_ENDPOINT
from predict_stock import base_dir
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.rest import TimeFrameUnit

import pandas as pd
NY = 'America/New_York'
"""
Downloads daily stock data from nasdaq

arqit? ARQQ
SHOP
TEAM?
PFE
MRNA
"""


def download_minute_stock_data(path=None):
    symbols = [
        # 'COUR',
        'GOOG',
        'TSLA',
        'NVDA',
        'AAPL',
        # "GTLB", no data
        # "AMPL",  no data
        "U",
        "ADSK",
        # "RBLX",
        "CRWD",
        "ADBE",
        "NET",
        # 'COIN',
        # 'QUBT',  no data
        # 'ARQQ',  no data
        # avoiding .6% buffer
        # 'REA.AX',
        # 'XRO.AX',
        # 'SEK.AX',
        # 'NXL.AX',  # data anlytics
        # 'APX.AX',  # data collection for ml/labelling
        # 'CDD.AX',
        # 'NVX.AX',
        # 'BRN.AX',  # brainchip
        # 'AV1.AX',
        # 'TEAM',
        # 'PFE',
        # 'MRNA',
    ]
    save_path = base_dir / 'data'
    if path:
        save_path = base_dir / 'data' / path
    save_path.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        api = REST(secret_key=ALP_SECRET_KEY, key_id=ALP_KEY_ID, base_url=ALP_ENDPOINT)

        start = (datetime.datetime.now() - datetime.timedelta(days=3)).strftime('%Y-%m-%d')
        # end = (datetime.datetime.now() - datetime.timedelta(days=2)).strftime('%Y-%m-%d') # todo recent data
        end = (datetime.datetime.now()).strftime('%Y-%m-%d') # todo recent data
        # df = api.get_bars(symbol, TimeFrame.Minute, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), adjustment='raw').df
        # start = pd.Timestamp('2020-08-28 9:30', tz=NY).isoformat()
        # end = pd.Timestamp('2020-08-28 16:00', tz=NY).isoformat()
        ## print(api.get_barset(['AAPL', 'GOOG'], 'minute', start=start, end=end).df)

        minute_df = api.get_bars(symbol, TimeFrame(15, TimeFrameUnit.Minute), start, end,
                                 adjustment='raw').df
        if minute_df.empty:
            print(f"{symbol} has no data")
            continue
        try:
            minute_df.drop(['volume', 'trade_count', 'vwap'], axis=1, inplace=True)
        except KeyError:
            print(f"{symbol} has no volume or something")
            continue


        # rename columns with upper case
        minute_df.rename(columns=lambda x: x.capitalize(), inplace=True)
        print(minute_df)

        file_save_path = (save_path / '{}-{}.csv'.format(symbol, end))
        minute_df.to_csv(file_save_path)
    return minute_df


def visualize_stock_data(df):
    register_matplotlib_converters()
    df.plot(x='Date', y='Close')
    plt.show()


if __name__ == '__main__':
    df = download_minute_stock_data()
    visualize_stock_data(df)
