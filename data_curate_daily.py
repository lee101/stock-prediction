import datetime
from functools import cache

import matplotlib.pyplot as plt
import pandas_datareader.data as web
from cachetools import cache, TTLCache
from pandas import DataFrame
from pandas.plotting import register_matplotlib_converters
from loguru import logger

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


def download_daily_stock_data(path=None, all_data_force=False):
    symbols = [
        'COUR',
        'GOOG',
        'TSLA',
        'NVDA',
        'AAPL',
        # "GTLB", no data
        # "AMPL",  no data
        "U",
        "ADSK",
        # "RBLX", # unpredictable
        "CRWD",
        "ADBE",
        "NET",
        # 'COIN', # unpredictable
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
        'AMD',
        'MSFT',
        'FB',
        'CRM',
        'NFLX',
        'PYPL',
        'SAP',
        'AMD', # tmp consider disabling/felt its model was a bit negative for now
        'SONY',
        # 'PFE',
        # 'MRNA',
    # ]
    # symbols = [
        'BTCUSD',
        'ETHUSD',
        'LTCUSD',

               ]
    api = REST(secret_key=ALP_SECRET_KEY, key_id=ALP_KEY_ID, base_url=ALP_ENDPOINT)

    alpaca_clock = api.get_clock()
    if not alpaca_clock.is_open and not all_data_force:
        logger.info("Market is closed")
        # can trade crypto out of hours
        symbols = [
           'BTCUSD',
           'ETHUSD',
           'LTCUSD',
       ]

    save_path = base_dir / 'data'
    if path:
        save_path = base_dir / 'data' / path
    save_path.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:

        start = (datetime.datetime.now() - datetime.timedelta(days=365*4)).strftime('%Y-%m-%d')
        # end = (datetime.datetime.now() - datetime.timedelta(days=2)).strftime('%Y-%m-%d') # todo recent data
        end = (datetime.datetime.now()).strftime('%Y-%m-%d') # todo recent data
        # df = api.get_bars(symbol, TimeFrame.Minute, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), adjustment='raw').df
        # start = pd.Timestamp('2020-08-28 9:30', tz=NY).isoformat()
        # end = pd.Timestamp('2020-08-28 16:00', tz=NY).isoformat()
        minute_df = download_exchange_historical_data(api, symbol)
        minute_df_last = download_exchange_latest_data(api, symbol)
        # replace the last element of minute_df with last
        if not minute_df_last.empty:
            # can be empty as it could be closed for two days so can skipp getting latest data
            minute_df.iloc[-1] = minute_df_last.iloc[-1]

        if minute_df.empty:
            logger.info(f"{symbol} has no data")
            continue

        # rename columns with upper case
        minute_df.rename(columns=lambda x: x.capitalize(), inplace=True)
        # logger.info(minute_df)

        file_save_path = (save_path / '{}-{}.csv'.format(symbol, end))
        minute_df.to_csv(file_save_path)
    return minute_df

# cache for 4 hours
data_cache = TTLCache(maxsize=100, ttl=14400)
def download_exchange_historical_data(api, symbol):
    cached_result = data_cache.get(symbol, DataFrame())
    if not cached_result.empty:
        return cached_result
    start = (datetime.datetime.now() - datetime.timedelta(days=365 * 4)).strftime('%Y-%m-%d')
    # end = (datetime.datetime.now() - datetime.timedelta(days=2)).strftime('%Y-%m-%d') # todo recent data
    end = (datetime.datetime.now()).strftime('%Y-%m-%d')  # todo recent data
    ## logger.info(api.get_barset(['AAPL', 'GOOG'], 'minute', start=start, end=end).df)
    results = download_stock_data_between_times(api, end, start, symbol)
    if not results.empty:
        data_cache[symbol] = results
    return results


def download_exchange_latest_data(api, symbol):
    start = (datetime.datetime.now() - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
    # end = (datetime.datetime.now() - datetime.timedelta(days=2)).strftime('%Y-%m-%d') # todo recent data
    end = (datetime.datetime.now()).strftime('%Y-%m-%d')  # todo recent data
    ## logger.info(api.get_barset(['AAPL', 'GOOG'], 'minute', start=start, end=end).df)
    return download_stock_data_between_times(api, end, start, symbol)

def download_stock_data_between_times(api, end, start, symbol):
    if symbol in ['BTCUSD', 'ETHUSD', 'LTCUSD']:
        minute_df = api.get_crypto_bars(symbol, TimeFrame(1, TimeFrameUnit.Day), start, end, exchanges=['FTXU']).df
        try:
            minute_df.drop(['exchange'], axis=1, inplace=True)
        except KeyError:
            logger.info(f"{symbol} has no exchange key")
        return minute_df
    else:
        minute_df = api.get_bars(symbol, TimeFrame(1, TimeFrameUnit.Day), start, end,
                                 adjustment='raw').df
        try:
            minute_df.drop(['volume', 'trade_count', 'vwap'], axis=1, inplace=True)
        except KeyError:
            logger.info(f"{symbol} has no volume or something")
        return minute_df


def visualize_stock_data(df):
    register_matplotlib_converters()
    df.plot(x='Date', y='Close')
    plt.show()


if __name__ == '__main__':
    df = download_daily_stock_data()
    visualize_stock_data(df)
