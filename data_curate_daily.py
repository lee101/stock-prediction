import datetime
import traceback

import matplotlib.pyplot as plt
import pytz
from alpaca.data import CryptoBarsRequest, TimeFrame, StockBarsRequest, TimeFrameUnit, CryptoHistoricalDataClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading import TradingClient
from cachetools import TTLCache
from loguru import logger
from pandas import DataFrame
from pandas.plotting import register_matplotlib_converters
from retry import retry

from alpaca_wrapper import latest_data
from env_real import ALP_SECRET_KEY, ALP_KEY_ID, ALP_ENDPOINT, ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, ADD_LATEST
from predict_stock import base_dir
from src.stock_utils import remap_symbols

# work in UTC
# os.environ['TZ'] = 'UTC'
NY = 'America/New_York'
"""
Downloads daily stock data from nasdaq

arqit? ARQQ
SHOP
TEAM?
PFE
MRNA
"""
crypto_client = CryptoHistoricalDataClient()

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
        'COIN', # unpredictable
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
        # 'AMD',
        'MSFT',
        # 'META',
        # 'CRM',
        'NFLX',
        'PYPL',
        'SAP',
        # 'AMD',  # tmp consider disabling/felt its model was a bit negative for now
        'SONY',
        # 'PFE',
        # 'MRNA',
    # ]
    # # only crypto for now TODO change this
    # symbols = [
        'BTCUSD',
        'ETHUSD',
        # 'LTCUSD',
        # "PAXGUSD",
        # "UNIUSD",

    ]
    # client = StockHistoricalDataClient(ALP_KEY_ID, ALP_SECRET_KEY, url_override="https://data.sandbox.alpaca.markets/v2")
    client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
    api = TradingClient(
        ALP_KEY_ID,
        ALP_SECRET_KEY,
        # ALP_ENDPOINT,
        paper=ALP_ENDPOINT != "https://api.alpaca.markets",
    )
    alpaca_clock = api.get_clock()
    if not alpaca_clock.is_open and not all_data_force:
        logger.info("Market is closed")
        # can trade crypto out of hours
        symbols = [
            'BTCUSD',
            'ETHUSD',
            # 'LTCUSD',
            # "PAXGUSD", "UNIUSD"
        ]

    save_path = base_dir / 'data'
    if path:
        save_path = base_dir / 'data' / path
    save_path.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:

        start = (datetime.datetime.now() - datetime.timedelta(days=365 * 4)).strftime('%Y-%m-%d')
        # end = (datetime.datetime.now() - datetime.timedelta(days=2)).strftime('%Y-%m-%d') # todo recent data
        end = (datetime.datetime.now()).strftime('%Y-%m-%d')  # todo recent data
        # df = api.get_bars(symbol, TimeFrame.Minute, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), adjustment='raw').df
        # start = pd.Timestamp('2020-08-28 9:30', tz=NY).isoformat()
        # end = pd.Timestamp('2020-08-28 16:00', tz=NY).isoformat()
        daily_df = download_exchange_historical_data(client, symbol)
        try:
            minute_df_last = download_exchange_latest_data(client, symbol)
        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            print(f"empty new data frame for {symbol}")
            minute_df_last = DataFrame() # weird issue with empty fb data frame
        # replace the last element of daily_df with last
        if not minute_df_last.empty:
            # can be empty as it could be closed for two days so can skipp getting latest data
            daily_df.iloc[-1] = minute_df_last.iloc[-1]

        if daily_df.empty:
            logger.info(f"{symbol} has no data")
            continue

        # rename columns with upper case
        daily_df.rename(columns=lambda x: x.capitalize(), inplace=True)
        # logger.info(daily_df)

        file_save_path = (save_path / '{}-{}.csv'.format(symbol.replace("/", "-"), end))
        file_save_path.parent.mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(file_save_path)
    return daily_df


# cache for 4 hours
data_cache = TTLCache(maxsize=100, ttl=14400)


def download_exchange_historical_data(api, symbol):
    cached_result = data_cache.get(symbol, DataFrame())
    if not cached_result.empty:
        return cached_result
    start = (datetime.datetime.now(tz=pytz.utc) - datetime.timedelta(days=365 * 4))
    # end = (datetime.datetime.now() - datetime.timedelta(days=2)).strftime('%Y-%m-%d') # todo recent data
    end = (datetime.datetime.now(tz=pytz.utc) - datetime.timedelta(minutes=16))  # todo recent data
    ## logger.info(api.get_barset(['AAPL', 'GOOG'], 'minute', start=start, end=end).df)
    results = download_stock_data_between_times(api, end, start, symbol)
    if not results.empty:
        data_cache[symbol] = results
    return results


def download_exchange_latest_data(api, symbol):
    global spreads
    start = (datetime.datetime.now(tz=pytz.utc) - datetime.timedelta(days=10))
    # end = (datetime.datetime.now() - datetime.timedelta(days=2)).strftime('%Y-%m-%d') # todo recent data
    end = (datetime.datetime.now(tz=pytz.utc)) - datetime.timedelta(minutes=16)  # todo recent data
    ## logger.info(api.get_barset(['AAPL', 'GOOG'], 'minute', start=start, end=end).df)
    latest_data_dl = download_stock_data_between_times(api, end, start, symbol)

    if ADD_LATEST: # collect very latest close times, todo extend bars?
        very_latest_data = latest_data(symbol)
        # check if market closed
        ask_price = float(very_latest_data.ask_price)
        bid_price = float(very_latest_data.bid_price)
        if bid_price != 0 and ask_price != 0:
            latest_data_dl["close"] = (bid_price + ask_price) / 2.
            spread = ask_price / bid_price
            logger.info(f"{symbol} spread {spread}")
            spreads[symbol] = spread
            bids[symbol] = bid_price
            asks[symbol] = ask_price
    return latest_data_dl
asks = {}
bids = {}
spreads = {}
def get_spread(symbol):
    return 1 - spreads.get(symbol, 1.05)

def get_ask(symbol):
    ask = asks.get(symbol)
    if not ask:
        logger.error(f"error getting ask price for {symbol}")
        logger.info(asks)
    return ask

def get_bid(symbol):
    bid = bids.get(symbol)
    if not bid:
        logger.error(f"error getting bid price for {symbol}")
        logger.info(bids)
    return bid

def download_stock_data_between_times(api, end, start, symbol):
    if symbol in ['BTCUSD', 'ETHUSD', 'LTCUSD', "PAXGUSD", "UNIUSD"]:
        daily_df = crypto_get_bars(end, start, symbol)
        try:
            daily_df.drop(['exchange'], axis=1, inplace=True)
        except KeyError:
            logger.info(f"{symbol} has no exchange key - this is okay")
        return daily_df
    else:
        daily_df = get_bars(api, end, start, symbol)
        try:
            daily_df.drop(['volume', 'trade_count', 'vwap'], axis=1, inplace=True)
        except KeyError:
            logger.info(f"{symbol} has no volume or something")
        return daily_df

@retry(delay=.1, tries=5)
def get_bars(api, end, start, symbol):
    return api.get_stock_bars(
        StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(1, TimeFrameUnit.Day), start=start, end=end,
                         adjustment='raw')).df


@retry(delay=.1, tries=5)
def crypto_get_bars(end, start, symbol):
    return crypto_client.get_crypto_bars(
        CryptoBarsRequest(symbol_or_symbols=remap_symbols(symbol), timeframe=TimeFrame(1, TimeFrameUnit.Day),
                          start=start, end=end,
                          exchanges=['FTXU'])).df


def visualize_stock_data(df):
    register_matplotlib_converters()
    df.plot(x='Date', y='Close')
    plt.show()


if __name__ == '__main__':
    df = download_daily_stock_data()
    visualize_stock_data(df)
