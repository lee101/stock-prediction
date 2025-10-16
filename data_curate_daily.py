import datetime
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
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
from data_utils import is_fp_close_to_zero
from env_real import ALP_SECRET_KEY, ALP_KEY_ID, ALP_ENDPOINT, ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, ADD_LATEST
from src.fixtures import crypto_symbols
from src.stock_utils import remap_symbols

base_dir = Path(__file__).parent

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


def download_daily_stock_data(path=None, all_data_force=False, symbols=None):
    symbols_provided = symbols is not None
    if symbols is None:
        symbols = [
            'COUR', 'GOOG', 'TSLA', 'NVDA', 'AAPL', "U", "ADSK", "CRWD", "ADBE", "NET",
            'COIN',
            #'MSFT',
            'NFLX', 'PYPL', 'SAP', 'SONY', 'BTCUSD', 'ETHUSD',
        ]
    else:
        symbols = list(symbols)

    client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
    api = TradingClient(
        ALP_KEY_ID,
        ALP_SECRET_KEY,
        paper=ALP_ENDPOINT != "https://api.alpaca.markets",
    )

    save_path = base_dir / 'data'
    if path:
        save_path = base_dir / 'data' / path
    save_path.mkdir(parents=True, exist_ok=True)

    ##test code
    # First check for existing CSV files for each symbol
    found_symbols = {}
    remaining_symbols = []
    end = datetime.datetime.now().strftime('%Y-%m-%d')
    # todo only do this in test mode
    # if False:
    #     for symbol in symbols:
    #         # Look for matching CSV files in save_path
    #         symbol_files = list(save_path.glob(f'{symbol.replace("/", "-")}*.csv'))
    #         if symbol_files:
    #             # Use most recent file if multiple exist
    #             latest_file = max(symbol_files, key=lambda x: x.stat().st_mtime)
    #             found_symbols[symbol] = pd.read_csv(latest_file)
    #         else:
    #             remaining_symbols.append(symbol)

    #     if not remaining_symbols:
    #         return found_symbols[symbols[-1]] if symbols else DataFrame()

    alpaca_clock = api.get_clock()
    if not alpaca_clock.is_open and not all_data_force:
        logger.info("Market is closed")
        if not symbols_provided:
            # Only keep crypto symbols when using the default universe and the market is closed
            symbols = [symbol for symbol in symbols if symbol in crypto_symbols]

    # Use the (potentially filtered) symbols list for downloading
    remaining_symbols = symbols

    # Download data for remaining symbols
    for symbol in remaining_symbols:
        start = (datetime.datetime.now() - datetime.timedelta(days=365 * 4)).strftime('%Y-%m-%d')
        end = (datetime.datetime.now()).strftime('%Y-%m-%d')
        daily_df = download_exchange_historical_data(client, symbol)
        try:
            minute_df_last = download_exchange_latest_data(client, symbol)
        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            print(f"empty new data frame for {symbol}")
            minute_df_last = DataFrame()

        if not minute_df_last.empty:
            daily_df.iloc[-1] = minute_df_last.iloc[-1]

        if daily_df.empty:
            logger.info(f"{symbol} has no data")
            continue

        daily_df.rename(columns=lambda x: x.capitalize(), inplace=True)

        file_save_path = (save_path / '{}-{}.csv'.format(symbol.replace("/", "-"), end))
        file_save_path.parent.mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(file_save_path)
        found_symbols[symbol] = daily_df

    # Return the last processed dataframe or an empty one if none processed
    return found_symbols[symbols[-1]] if symbols else DataFrame()


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

    if ADD_LATEST:  # collect very latest close times, todo extend bars?
        # Try up to 3 times to get valid bid/ask data
        max_retries = 3
        retry_count = 0
        ask_price = None
        bid_price = None
        
        while retry_count < max_retries:
            try:
                very_latest_data = latest_data(symbol)
                ask_price = float(very_latest_data.ask_price)
                bid_price = float(very_latest_data.bid_price)
                logger.info(f"Latest {symbol} bid: {bid_price}, ask: {ask_price} (attempt {retry_count + 1})")
                
                # If both prices are valid, break out of retry loop
                if not is_fp_close_to_zero(bid_price) and not is_fp_close_to_zero(ask_price):
                    break
                    
                # If at least one is invalid, log and retry
                if retry_count < max_retries - 1:
                    logger.warning(f"Invalid bid/ask prices for {symbol} on attempt {retry_count + 1}, retrying...")
                    retry_count += 1
                    time.sleep(0.5)  # Small delay between retries
                    continue
                else:
                    # Final attempt failed
                    break
                    
            except Exception as e:
                logger.error(f"Error getting latest data for {symbol} on attempt {retry_count + 1}: {e}")
                if retry_count < max_retries - 1:
                    retry_count += 1
                    time.sleep(0.5)
                    continue
                else:
                    break
        
        # Handle invalid prices after all retries
        if is_fp_close_to_zero(bid_price) or is_fp_close_to_zero(ask_price):
            if not is_fp_close_to_zero(bid_price) or not is_fp_close_to_zero(ask_price):
                logger.warning(f"Invalid bid/ask prices for {symbol} after {max_retries} attempts, one is zero - using max")
                ask_price = max(bid_price, ask_price)
                bid_price = max(bid_price, ask_price)
            else:
                logger.warning(f"Both bid/ask prices are zero for {symbol} after {max_retries} attempts - using synthetic spread")
                # Both are zero, can't calculate a meaningful price
                ask_price = None
                bid_price = None
        if bid_price is not None and ask_price is not None and not is_fp_close_to_zero(bid_price) and not is_fp_close_to_zero(ask_price):
            # only update the latest row
            latest_data_dl.loc[latest_data_dl.index[-1], 'close'] = (bid_price + ask_price) / 2.
            spread = ask_price / bid_price
            logger.info(f"{symbol} spread {spread}")
            spreads[symbol] = spread
            bids[symbol] = bid_price
            asks[symbol] = ask_price
        else:
            # Use a synthetic spread when we can't get valid bid/ask data
            logger.warning(f"Using synthetic spread of 1.01 for {symbol} due to invalid bid/ask data")
            last_close = latest_data_dl.iloc[-1]['close'] if not latest_data_dl.empty else 100.0
            synthetic_bid = last_close / 1.005  # Assume 0.5% spread around mid
            synthetic_ask = last_close * 1.005
            spreads[symbol] = 1.01  # Use 1.01 as fallback spread
            bids[symbol] = synthetic_bid
            asks[symbol] = synthetic_ask

    logger.info(f"Data timestamp: {latest_data_dl.index[-1]}")
    logger.info(f"Current time: {datetime.datetime.now(tz=pytz.utc)}")
    return latest_data_dl


asks = {}
bids = {}
spreads = {}


def get_spread(symbol):
    return 1 - spreads.get(symbol, 1.05)


def fetch_spread(symbol):
    client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
    minute_df_last = download_exchange_latest_data(client, symbol)
    return spreads.get(symbol, 1.05)


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
            pass
            #logger.info(f"{symbol} has no exchange key - this is okay")
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
    df.plot(x='timestamp', y='close')
    plt.show()


if __name__ == '__main__':
    df = download_daily_stock_data(symbols=['GOOGL'])
    visualize_stock_data(df)
