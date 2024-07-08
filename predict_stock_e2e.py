import random
import traceback
import uuid
from ast import literal_eval
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep

import torch
from alpaca.trading import Position
from loguru import logger
from pandas import DataFrame

import alpaca_wrapper
from data_curate_daily import download_daily_stock_data, get_spread, get_bid, get_ask
# from predict_stock import make_predictions
from decorator_utils import timeit
from jsonshelve import FlatShelf
from loss_utils import CRYPTO_TRADING_FEE
from predict_stock_forecasting import make_predictions
from src.binan import binance_wrapper
# read do_retrain argument from argparse
# do_retrain = True
from src.conversion_utils import convert_string_to_datetime
from src.date_utils import is_nyse_trading_day_ending
from src.fixtures import crypto_symbols
from src.process_utils import backout_near_market
from src.trading_obj_utils import filter_to_realistic_positions
from src.utils import log_time

use_stale_data = False
retrain = True
# dev:
# use_stale_data = True
# retrain = False
# backtesting_simulation = True

# if use_stale_data:
#     alpaca_wrapper.force_open_the_clock = True

daily_predictions = DataFrame()
daily_predictions_time = None

# Configure loguru to print both UTC and EDT time, and write to both stdout and a log file
from loguru import logger
from datetime import datetime
import pytz
import sys

class EDTFormatter:
    def __init__(self):
        self.local_tz = pytz.timezone('US/Eastern')

    def __call__(self, record):
        utc_time = record["time"].strftime('%Y-%m-%d %H:%M:%S %Z')
        local_time = datetime.now(self.local_tz).strftime('%Y-%m-%d %H:%M:%S %Z')
        level_colors = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",   # Green
            "WARNING": "\033[33m", # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m" # Magenta
        }
        reset_color = "\033[0m"
        level_color = level_colors.get(record['level'].name, "")
        return f"{utc_time} | {local_time} | {level_color}{record['level'].name}{reset_color} | {record['message']}\n"

logger.remove()
logger.add(sys.stdout, format=EDTFormatter())
logger.add("logfile.log", format=EDTFormatter())

@timeit
def do_forecasting():
    global daily_predictions
    global daily_predictions_time
    alpaca_clock = alpaca_wrapper.get_clock()
    if daily_predictions.empty and (
            daily_predictions_time is None or daily_predictions_time < datetime.now() - timedelta(days=1)) or (
            'SAP' not in daily_predictions[
        'instrument'].unique() and alpaca_clock.is_open):  # or if we dont have stocks like SAP in there?
        daily_predictions_time = datetime.now()
        if use_stale_data:
            current_time_formatted = '2021-12-05 18-20-29'
            current_time_formatted = '2021-12-09 12-16-26'  # new/ more data
            current_time_formatted = '2021-12-11 07-57-21-2'  # new/ less data tickers
            current_time_formatted = 'min'  # new/ less data tickers
            current_time_formatted = '2021-12-30--20-11-47'  # new/ 30 minute data # '2022-10-14 09-58-20'
            current_time_formatted = '2024-04-04--20-41-41'  # new/ 30 minute data # '2022-10-14 09-58-20'
            current_time_formatted = '2024-04-18--06-14-26'  # new/ 30 minute data # '2022-10-14 09-58-20'
        else:
            current_time_formatted = (datetime.now() - timedelta(days=10)).strftime(
                '%Y-%m-%d--%H-%M-%S')  # but cant be 15 mins?
            if not use_stale_data:
                download_daily_stock_data(current_time_formatted, True)
        # error where daily where downloaded at the wrong time?
        daily_predictions = make_predictions(current_time_formatted, retrain=retrain,
                                             alpaca_wrapper=alpaca_wrapper)  # TODO
        # daily_predictions = make_predictions(current_time_formatted) # TODO

    current_time_formatted = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    if not use_stale_data:  # why are these different?
        download_daily_stock_data(current_time_formatted)
        minute_predictions = make_predictions(current_time_formatted, alpaca_wrapper=alpaca_wrapper)
    else:
        minute_predictions = daily_predictions  # todo fix this

    make_trade_suggestions(daily_predictions, minute_predictions)


def close_profitable_trades(all_preds, positions, orders, change_settings=True):
    # global made_money_recently
    # global made_money_one_before_recently
    # global made_money_recently_tmp

    # close all positions that are not in this current held stock
    already_held_stock = False
    has_traded = False
    for position in positions:

        is_worsening_position = False
        for index, row in all_preds.iterrows():
            # hack convert to tensor. todo fix this earlier
            if isinstance(row['instrument'], torch.Tensor) and row['instrument'].dim() == 0:
                row['instrument'] = float(row['instrument'])
            if isinstance(row['close_predicted_price_minute'], torch.Tensor) and row[
                'close_predicted_price_minute'].dim() == 0:
                row['close_predicted_price_minute'] = float(row['close_predicted_price_minute'])
            if isinstance(row['high_predicted_price_value'], torch.Tensor) and row[
                'high_predicted_price_value'].dim() == 0:
                row['high_predicted_price_value'] = float(row['high_predicted_price_value'])
            if isinstance(row['maxdiffprofit_profit'], torch.Tensor) and row['maxdiffprofit_profit'].dim() == 0:
                row['maxdiffprofit_profit'] = float(row['maxdiffprofit_profit'])
            if isinstance(row['entry_takeprofit_profit'], torch.Tensor) and row['entry_takeprofit_profit'].dim() == 0:
                row['entry_takeprofit_profit'] = float(row['entry_takeprofit_profit'])
            if isinstance(row['entry_takeprofit_high_price'], torch.Tensor) and row[
                'entry_takeprofit_high_price'].dim() == 0:
                row['entry_takeprofit_high_price'] = float(row['entry_takeprofit_high_price'])
            if isinstance(row['maxdiffprofit_high_price'], torch.Tensor) and row['maxdiffprofit_high_price'].dim() == 0:
                row['maxdiffprofit_high_price'] = float(row['maxdiffprofit_high_price'])
            if isinstance(row['entry_takeprofit_profit_high_multiplier'], torch.Tensor) and row[
                'entry_takeprofit_profit_high_multiplier'].dim() == 0:
                row['entry_takeprofit_profit_high_multiplier'] = float(row['entry_takeprofit_profit_high_multiplier'])
            if isinstance(row['maxdiffprofit_profit_high_multiplier'], torch.Tensor) and row[
                'maxdiffprofit_profit_high_multiplier'].dim() == 0:
                row['maxdiffprofit_profit_high_multiplier'] = float(row['maxdiffprofit_profit_high_multiplier'])
            if isinstance(row['entry_takeprofit_low_price'], torch.Tensor) and row[
                'entry_takeprofit_low_price'].dim() == 0:
                row['entry_takeprofit_low_price'] = float(row['entry_takeprofit_low_price'])
            if isinstance(row['low_predicted_price_value'], torch.Tensor) and row[
                'low_predicted_price_value'].dim() == 0:
                row['low_predicted_price_value'] = float(row['low_predicted_price_value'])
            if isinstance(row['maxdiffprofit_low_price'], torch.Tensor) and row['maxdiffprofit_low_price'].dim() == 0:
                row['maxdiffprofit_low_price'] = float(row['maxdiffprofit_low_price'])
            if isinstance(row['entry_takeprofit_profit_low_multiplier'], torch.Tensor) and row[
                'entry_takeprofit_profit_low_multiplier'].dim() == 0:
                row['entry_takeprofit_profit_low_multiplier'] = float(row['entry_takeprofit_profit_low_multiplier'])
            if isinstance(row['maxdiffprofit_profit_low_multiplier'], torch.Tensor) and row[
                'maxdiffprofit_profit_low_multiplier'].dim() == 0:
                row['maxdiffprofit_profit_low_multiplier'] = float(row['maxdiffprofit_profit_low_multiplier'])

            if row['instrument'] == position.symbol:
                # make it reasonably easy to back out of bad trades
                # if (row['close_predicted_price_minute'] < 0) and position.side == 'long':
                #     is_worsening_position = True
                #
                # if (row['close_predicted_price_minute'] > 0) and position.side == 'short':
                #     is_worsening_position = True
                #
                # # if random.choice([True, False, False, False, False, False, False, False, False, False, False, False, False, False, False]) or at_market_open:
                # if is_worsening_position:
                #     alpaca_wrapper.close_position_at_current_price(position, row)
                #     has_traded = True
                #     logger.info(f"Closing predicted to worsen position {position.symbol}")
                # TODO note this is not the real ordered time for manual orders!
                ordered_time = trade_entered_times.get(position.symbol)
                is_crypto = position.symbol in crypto_symbols
                is_trading_day_ending = False  # todo investigate reenabling this logic
                if is_crypto:
                    is_trading_day_ending = datetime.now().hour in [11, 12, 13]  # TODO nzdt specific code here
                else:
                    is_trading_day_ending = is_nyse_trading_day_ending()

                if not ordered_time or ordered_time < datetime.now() - timedelta(minutes=60 * 16):
                    if float(position.unrealized_plpc) < 0 and change_settings:
                        change_time = instrument_strategy_change_times.get(position.symbol)
                        if not change_time or change_time < datetime.now() - timedelta(minutes=30 * 16.):
                            instrument_strategy_change_times[position.symbol] = datetime.now()
                            current_strategy = instrument_strategies.get(position.symbol, 'aggressive_buy')

                            available_strategies = {'aggressive', 'aggressive_buy', 'aggressive_sell', 'entry'} - {
                                current_strategy}
                            if current_strategy.startswith('aggressive'):
                                available_strategies = available_strategies - {'aggressive'}
                            new_strategy = random.choice(list(available_strategies))
                            logger.info(
                                f"Changing strategy for {position.symbol} from {current_strategy} to {new_strategy}")
                            instrument_strategies[position.symbol] = new_strategy
                # todo check time in market not overall time
                trade_length_before_close = timedelta(minutes=60 * 4)
                max_trade_order_length = timedelta(minutes=60 * 30)
                min_trade_order_length = timedelta(minutes=60 * 1)

                if position.symbol in crypto_symbols:
                    trade_length_before_close = timedelta(minutes=60 * 20)

                if abs(float(position.market_value)) < 3000:
                    # closing test positions sooner TODO simulate stuff like this instead of really doing it
                    trade_length_before_close = timedelta(minutes=60 * 4)
                    is_trading_day_ending = True

                close_all_because_of_day_end = is_trading_day_ending and position.symbol not in crypto_symbols
                longer_than_max_order_length = not ordered_time or ordered_time < datetime.now() - max_trade_order_length
                more_recent_than_min_order_length = not ordered_time or ordered_time > datetime.now() - min_trade_order_length
                if ((
                        not ordered_time or ordered_time < datetime.now() - trade_length_before_close) and is_trading_day_ending and change_settings) \
                    or close_all_because_of_day_end \
                    or (longer_than_max_order_length and not more_recent_than_min_order_length):
                    current_time = datetime.now()
                    # at_market_open = False
                    # hourly can close positions at the market open? really?
                    # if current_time.hour == 3 and current_time.minute < 45:
                    #     at_market_open = True  # TODO this only works for NZ time
                    #     # todo properly test if we can close positions at market open
                    #     # for now we wont violently close our positions untill 15mins after open

                    # if not at_market_open:

                    # close other orders for pair
                    for order in orders:
                        if order.symbol == position.symbol:
                            alpaca_wrapper.cancel_order(order)
                            # todo check if we have one open that is trying to close already?
                    # close old position, not been hitting our predictions
                    # todo why cancel order if its still predicted to be successful?

                    logger.info(f"Closing position to reduce risk {position.symbol}")
                    # use bash to run command
                    # python
                    # scripts/alpaca_cli.py
                    # backout_near_market
                    # LTCUSD
                    # todo stop creating lots of
                    # ensure its really closed
                    # alpaca_wrapper.close_position_at_current_price(position, row)
                    backout_near_market(position.symbol)


                else:
                    exit_strategy = 'maxdiff'  # TODO bug - should be based on what entry strategy should be
                    # takeprofit_profit also a thing
                    if float(row['maxdiffprofit_profit']) < float(row['entry_takeprofit_profit']):
                        exit_strategy = 'entry'

                    entry_price = float(position.avg_entry_price)
                    if position.side == 'long':
                        predicted_high = row['entry_takeprofit_high_price']
                        if exit_strategy == 'entry':
                            if abs(row['entry_takeprofit_profit_high_multiplier']) > .01:  # tuned for minutely
                                predicted_high = row['high_predicted_price_value']
                        elif exit_strategy == 'maxdiff':
                            if abs(row['maxdiffprofit_profit_high_multiplier']) > .01:  # tuned for minutely
                                predicted_high = row['maxdiffprofit_high_price']
                        sell_price = predicted_high
                        if not ordered_time or ordered_time > datetime.now() - timedelta(minutes=3 * 60):
                            # close new orders at atleast a profit - crypto needs higher margins to be profitable
                            margin_default_high = entry_price * (1 + .006)
                            sell_price = max(predicted_high, margin_default_high)
                        # only if no other orders already
                        ordered_already = False
                        for order in orders:
                            if order.side == 'sell' and order.symbol == position.symbol:
                                ordered_already = True
                                amount_order_is_closing = order.qty
                                # close the full qty of order
                                if amount_order_is_closing != position.qty:
                                    # cancel order
                                    alpaca_wrapper.cancel_order(order)
                                    alpaca_wrapper.open_take_profit_position(position, row, sell_price, position.qty)
                        if not ordered_already:
                            alpaca_wrapper.open_take_profit_position(position, row, sell_price, position.qty)
                    elif position.side == 'short':
                        predicted_low = row['entry_takeprofit_low_price']
                        if exit_strategy == 'entry':
                            if abs(row['entry_takeprofit_profit_low_multiplier']) > .01:
                                predicted_low = row['low_predicted_price_value']
                        elif exit_strategy == 'maxdiff':
                            if abs(row['maxdiffprofit_profit_low_multiplier']) > .01:
                                predicted_low = row['maxdiffprofit_low_price']
                        sell_price = predicted_low
                        if not ordered_time or ordered_time > datetime.now() - timedelta(minutes=3 * 60):
                            # close new orders at atleast a profit
                            margin_default_low = entry_price * (1 - .003)
                            sell_price = min(predicted_low, margin_default_low)
                        # only if no other orders already
                        ordered_already = False
                        for order in orders:
                            if order.side == 'long' and order.symbol == position.symbol:
                                ordered_already = True
                                amount_order_is_closing = order.qty
                                # close the full qty of order
                                if amount_order_is_closing != position.qty:
                                    # cancel order
                                    alpaca_wrapper.cancel_order(order)
                                    alpaca_wrapper.open_take_profit_position(position, row, sell_price, position.qty)

                        if not ordered_already:
                            alpaca_wrapper.open_take_profit_position(position, row, sell_price, position.qty)
                break
                # else:
                #     pass
                # instant close?
                # if float(position.unrealized_plpc) > 0.004:  ## or float(position.unrealized_pl) > 50:
                #     logger.info(f"Closing good position")
                #     alpaca_wrapper.close_position_violently(position)
                # todo test take profit?
                # alpaca_wrapper.open_take_profit_position(position, row)
                # logger.info(
                #     f"keeping position {position.symbol} - predicted to get better - open takeprofit at {row['close_last_price_minute'] }")


def close_profitable_crypto_binance_trades(all_preds, positions, orders, change_settings=True):
    # global made_money_recently
    # global made_money_one_before_recently
    # global made_money_recently_tmp

    # close all positions that are not in this current held stock
    already_held_stock = False
    has_traded = False
    balances = binance_wrapper.get_account_balances()
    # {'asset': 'BTC', 'free': '0.02332178',
    # get btc balance
    btc_balance = 0
    for balance in balances:
        if balance['asset'] == 'BTC':
            btc_balance = float(balance['free'])
            break
    side = 'long'
    if btc_balance > 0.01:
        side = 'short'
        # need to sell btc on binance
    # otheerwise need to buy btc on binance
    positions = filter_to_realistic_positions(positions)
    for position in positions:

        is_worsening_position = False
        for index, row in all_preds.iterrows():
            if row['instrument'] == position.symbol:
                # make it reasonably easy to back out of bad trades
                # if (row['close_predicted_price_minute'] < 0) and position.side == 'long':
                #     is_worsening_position = True
                #
                # if (row['close_predicted_price_minute'] > 0) and position.side == 'short':
                #     is_worsening_position = True
                #
                # # if random.choice([True, False, False, False, False, False, False, False, False, False, False, False, False, False, False]) or at_market_open:
                # if is_worsening_position:
                #     alpaca_wrapper.close_position_at_current_price(position, row)
                #     has_traded = True
                #     logger.info(f"Closing predicted to worsen position {position.symbol}")
                # TODO note this is not the real ordered time for manual orders!
                ordered_time = trade_entered_times.get(position.symbol)
                is_crypto = position.symbol in crypto_symbols
                is_trading_day_ending = False  # todo investigate reenabling this logic
                if is_crypto:
                    is_trading_day_ending = datetime.now().hour in [11, 12, 13]  # TODO nzdt specific code here
                else:
                    is_trading_day_ending = is_nyse_trading_day_ending()

                if not ordered_time or ordered_time < datetime.now() - timedelta(minutes=60 * 16):
                    if float(position.unrealized_plpc) < 0 and change_settings:
                        change_time = instrument_strategy_change_times.get(position.symbol)
                        if not change_time or change_time < datetime.now() - timedelta(minutes=30 * 16.):
                            instrument_strategy_change_times[position.symbol] = datetime.now()
                            current_strategy = instrument_strategies.get(position.symbol, 'aggressive_buy')

                            available_strategies = {'aggressive', 'aggressive_buy', 'aggressive_sell', 'entry'} - {
                                current_strategy}
                            if current_strategy.startswith('aggressive'):
                                available_strategies = available_strategies - {'aggressive'}
                            new_strategy = random.choice(list(available_strategies))
                            logger.info(
                                f"Changing strategy for {position.symbol} from {current_strategy} to {new_strategy}")
                            instrument_strategies[position.symbol] = new_strategy
                # todo check time in market not overall time
                trade_length_before_close = timedelta(minutes=60 * 22)
                if abs(float(position.market_value)) < 3000:
                    # closing test positions sooner TODO simulate stuff like this instead of really doing it
                    trade_length_before_close = timedelta(minutes=60 * 6)
                    is_trading_day_ending = True
                if (
                        not ordered_time or ordered_time < datetime.now() - trade_length_before_close) and is_trading_day_ending and change_settings:
                    current_time = datetime.now()
                    # at_market_open = False
                    # hourly can close positions at the market open? really?
                    # if current_time.hour == 3 and current_time.minute < 45:
                    #     at_market_open = True  # TODO this only works for NZ time
                    #     # todo properly test if we can close positions at market open
                    #     # for now we wont violently close our positions untill 15mins after open

                    # if not at_market_open:

                    # close other orders for pair on binance?
                    # for order in orders:
                    #     if order.symbol == position.symbol:
                    #         alpaca_wrapper.cancel_order(order)
                    # todo check if we have one open that is trying to close already?
                    # close old position, not been hitting our predictions
                    # todo why cancel order if its still predicted to be successful?

                    binance_wrapper.close_position_at_current_price(position, row)
                    logger.info(f"Closing position to reduce risk on binance {position.symbol}")

                else:
                    exit_strategy = 'maxdiff'  # TODO bug - should be based on what entry strategy should be
                    # takeprofit_profit also a thing
                    if float(row['maxdiffprofit_profit']) < float(row['entry_takeprofit_profit']):
                        exit_strategy = 'entry'

                    entry_price = float(position.avg_entry_price)
                    if position.side == 'long':
                        predicted_high = row['entry_takeprofit_high_price']
                        if exit_strategy == 'entry':
                            if abs(row['entry_takeprofit_profit_high_multiplier']) > .01:  # tuned for minutely
                                predicted_high = row['high_predicted_price_value']
                        elif exit_strategy == 'maxdiff':
                            if abs(row['maxdiffprofit_profit_high_multiplier']) > .01:  # tuned for minutely
                                predicted_high = row['maxdiffprofit_high_price']
                        sell_price = predicted_high
                        if not ordered_time or ordered_time > datetime.now() - timedelta(minutes=3 * 60):
                            # close new orders at atleast a profit - crypto needs higher margins to be profitable
                            margin_default_high = entry_price * (1 + .006)
                            sell_price = max(predicted_high, margin_default_high)
                        # only if no other orders already
                        ordered_already = False
                        for order in orders:
                            if order.side == 'sell' and order.symbol == position.symbol:
                                ordered_already = True
                                amount_order_is_closing = order.qty
                                # close the full qty of order
                                if amount_order_is_closing != position.qty:
                                    # cancel order
                                    binance_wrapper.open_take_profit_position(position, row, sell_price, position.qty)
                        if not ordered_already:
                            binance_wrapper.open_take_profit_position(position, row, sell_price, position.qty)
                    elif position.side == 'short':
                        predicted_low = row['entry_takeprofit_low_price']
                        if exit_strategy == 'entry':
                            if abs(row['entry_takeprofit_profit_low_multiplier']) > .01:
                                predicted_low = row['low_predicted_price_value']
                        elif exit_strategy == 'maxdiff':
                            if abs(row['maxdiffprofit_profit_low_multiplier']) > .01:
                                predicted_low = row['maxdiffprofit_low_price']
                        sell_price = predicted_low
                        if not ordered_time or ordered_time > datetime.now() - timedelta(minutes=3 * 60):
                            # close new orders at atleast a profit
                            margin_default_low = entry_price * (1 - .003)
                            sell_price = min(predicted_low, margin_default_low)
                        # only if no other orders already
                        ordered_already = False
                        for order in orders:
                            if order.side == 'long' and order.symbol == position.symbol:
                                ordered_already = True
                                amount_order_is_closing = order.qty
                                # close the full qty of order
                                if amount_order_is_closing != position.qty:
                                    binance_wrapper.open_take_profit_position(position, row, sell_price, position.qty)

                        if not ordered_already:
                            alpaca_wrapper.open_take_profit_position(position, row, sell_price, position.qty)
                break


data_dir = Path(__file__).parent / 'data'

current_flags = FlatShelf(str(data_dir / f"current_flags.db.json"))
made_money_recently = FlatShelf(str(data_dir / f"made_money_recently.db.json"))
made_money_recently_tmp = FlatShelf(str(data_dir / f"made_money_recently_tmp.db.json"))
made_money_one_before_recently = FlatShelf(str(data_dir / f"made_money_one_before_recently.db.json"))

made_money_recently_shorting = FlatShelf(str(data_dir / f"made_money_recently_shorting.db.json"))
made_money_recently_tmp_shorting = FlatShelf(str(data_dir / f"made_money_recently_tmp_shorting.db.json"))
made_money_one_before_recently_shorting = FlatShelf(str(data_dir / f"made_money_one_before_recently_shorting.db.json"))

trade_entered_times = FlatShelf(str(data_dir / f"trade_entered_times.db.json"))
# all_historical_orders = FlatShelf(str(data_dir / f"all_historical_orders.db"))

instrument_strategies = FlatShelf(str(data_dir / f"instrument_strategies.db.json"))
instrument_strategy_change_times = FlatShelf(str(data_dir / f"instrument_strategy_change_times_.db.json"))

# all keys in _times are stored e.g. 2024-04-16T19:53:01.577838

# convert all to strings
for key in list(instrument_strategy_change_times.keys()):
    instrument_strategy_change_times[str(key)] = convert_string_to_datetime(instrument_strategy_change_times.pop(key))

for key in list(trade_entered_times.keys()):
    trade_entered_times[str(key)] = convert_string_to_datetime(trade_entered_times.pop(key))


def buy_stock(row, all_preds, positions, orders):
    """
    or sell stock
    :param row:
    :return:
    """
    global made_money_recently
    global made_money_recently_tmp
    global made_money_one_before_recently
    global made_money_recently_shorting
    global made_money_recently_tmp_shorting
    global made_money_one_before_recently_shorting
    logger.info("buying stock...")
    current_interest_symbol = row['instrument']
    # close all positions that are not in this current held stock
    already_held_stock = False
    entry_strategy = 'maxdiff'
    # takeprofit_profit is also a thing
    if float(row['maxdiffprofit_profit']) + float(row['maxdiffprofit_profit_minute']) < float(
            row['entry_takeprofit_profit']) + float(row['entry_takeprofit_profit_minute']):
        entry_strategy = 'entry'
        logger.info(f"using entry strategy for {current_interest_symbol}")

    if entry_strategy == 'maxdiff':
        # maxdiff based side similar to simulation
        # already calculated for the minute, but use current price for old low/high
        low_to_close_diff = abs(1 - (row['low_predicted_price_value'] / row['close_last_price_minute'])) + abs(
            row['latest_low_diff_minute'])
        high_to_close_diff = abs(1 - (row['high_predicted_price_value'] / row['close_last_price_minute'])) + abs(
            row['latest_high_diff_minute'])

        new_position_side = 'short' if low_to_close_diff > high_to_close_diff else 'long'  # maxdiff max profit potential
    elif entry_strategy == 'entry':
        now_to_old_pred = 1 - (row['close_predicted_price_value_minute'] / row['close_last_price_minute'])
        new_position_side = 'short' if now_to_old_pred + row[
            'close_predicted_price_minute'] < 0 else 'long'  # just the end price 15min from now- dont worry about the extremes
    # also try the minmax or takeprofit strategy that doesn't trade at said price
    entry_price_strategy = 'minmax'  # at predicted low/high
    if float(row['takeprofit_profit']) + float(row['takeprofit_profit_minute']) < float(
            row['entry_takeprofit_profit']) + float(row['entry_takeprofit_profit_minute']):
        entry_price_strategy = 'entry'  # at current market price

    has_traded = False
    # filter out crypto positions under .01 for eth - this too low amount cannot be traded/is an anomaly
    positions = filter_to_realistic_positions(positions)
    for position in positions:
        if position.side == 'long':
            made_money_recently[position.symbol] = float(position.unrealized_plpc)
            made_money_one_before_recently[position.symbol] = made_money_recently_tmp.get(position.symbol, 0)
        else:
            made_money_recently_shorting[position.symbol] = float(position.unrealized_plpc)
            made_money_one_before_recently_shorting[position.symbol] = made_money_recently_tmp_shorting.get(
                position.symbol, 0)

        if position.symbol == current_interest_symbol:
            # todo could this prevent you from margining upward? should we clear all positions first?
            logger.info("Already holding {}".format(current_interest_symbol))
            already_held_stock = True
            already_held_amount = position.qty
        # may cause overtrading
        # else:
        #     alpaca_wrapper.close_position_at_current_price(position, row)
        #     has_traded = True
        #     logger.info(f"changing stance on {current_interest_symbol} to {new_position_side}")

    if not already_held_stock:
        logger.info(f"{new_position_side} {current_interest_symbol}")  # todo log the previous gains
        margin_multiplier = (1. / 10.0) * .8  # leave some room
        if current_interest_symbol not in crypto_symbols:
            # cant short crypto so turned off for crypto
            if entry_price_strategy == 'entry':
                entry_takeprofit_profit_over_two_trades = sum(literal_eval(row['entry_takeprofit_profit_values'])[:-2])
                logger.info(f"entry_takeprofit_profit_over_two_trades {entry_takeprofit_profit_over_two_trades}")
                if entry_takeprofit_profit_over_two_trades <= 0:
                    logger.info(
                        f"{current_interest_symbol} is loosing money over two days via entry takeprofit, making a small trade {row['entry_takeprofit_profit_values']} {entry_takeprofit_profit_over_two_trades}")

                    margin_multiplier = .001  # (1. / 10.0) * .3 # last trade values are loosing half trade
            else:
                take_profit_profit_over_two_trades = sum(literal_eval(row['takeprofit_profit_values'])[:-2])
                logger.info(f"takeprofit_profit_over_two_trades {take_profit_profit_over_two_trades}")
                if take_profit_profit_over_two_trades <= 0:
                    logger.info(
                        f"{current_interest_symbol} is loosing money over two days via takeprofit, making a small trade {row['takeprofit_profit_values']} {take_profit_profit_over_two_trades}")
                    margin_multiplier = .001  # (1. / 10.0) * .3 # last trade values are loosing half trade

            if entry_strategy == 'maxdiff':
                max_diff_profit_over_two_trades = sum(literal_eval(row['maxdiffprofit_profit_values'])[:-2])
                logger.info(f"maxdiff profit over two trades {max_diff_profit_over_two_trades}")
                if max_diff_profit_over_two_trades <= 0:
                    logger.info(
                        f"{current_interest_symbol} is loosing money over two days via maxdiff, making a small trade {row['maxdiffprofit_profit_values']} {max_diff_profit_over_two_trades}")
                    margin_multiplier = .001  # (1. / 10.0) * .3 # last trade values are loosing half trade

        made_money_recently_shorting_pnl = made_money_recently_shorting.get(current_interest_symbol, 0)
        made_money_recently_pnl = made_money_recently.get(current_interest_symbol, 0)
        if new_position_side == 'long':
            made_money_one_before_recently_pnl = made_money_one_before_recently.get(current_interest_symbol, 0)
            logger.info(
                f"made_money_recently_pnl {made_money_recently_pnl} made_money_one_before_recently_pnl {made_money_one_before_recently_pnl}")
            if (made_money_recently_pnl or 0) + (
                    made_money_one_before_recently_pnl or 0) <= 0:
                # if loosing money over two trades, make a small trade /recalculate
                margin_multiplier = .001
                logger.info(f"{current_interest_symbol} is loosing money over two days, making a small trade")
        else:
            made_money_one_before_recently_shorting_pnl = made_money_one_before_recently_shorting.get(
                current_interest_symbol, 0)
            logger.info(
                f"made_money_recently_shorting_pnl {made_money_recently_shorting_pnl} made_money_one_before_recently_shorting_pnl {made_money_one_before_recently_shorting_pnl}")
            if (made_money_recently_shorting_pnl or 0) + (
                    made_money_one_before_recently_shorting_pnl or 0) <= 0:
                # if loosing money over two trades, make a small trade /recalculate
                margin_multiplier = .001
                logger.info(
                    f"{current_interest_symbol} is loosing money over two days via shorting, making a small trade")

        current_price = row['close_last_price_minute']

        price_to_trade_at = max(current_price, row['high_last_price_minute'])
        current_strategy = instrument_strategies.get(current_interest_symbol, 'aggressive_buy')

        if new_position_side == 'long':
            predicted_low = row['takeprofit_low_price_minute']
            if abs(row['takeprofit_profit_low_multiplier_minute']) > .04:
                predicted_low = row['low_predicted_price_value_minute']
            price_to_trade_at = min(current_price, predicted_low)  # , row['low_last_price_minute'])
        elif new_position_side == 'short':
            predicted_high = row['takeprofit_high_price_minute']
            if abs(row['takeprofit_profit_high_multiplier_minute']) > .04:  # tuned for minutely
                predicted_high = row['high_predicted_price_value_minute']
            price_to_trade_at = max(current_price, predicted_high)

        if entry_price_strategy == 'entry':
            if current_strategy == 'aggressive':
                price_to_trade_at = current_price
            elif current_strategy == 'aggressive_buy' and new_position_side == 'long':
                price_to_trade_at = current_price
            elif current_strategy == 'aggressive_sell' and new_position_side == 'short':
                price_to_trade_at = current_price
        # ONLY trade if we aren't trading in that dir already
        ordered_already = False

        for order in orders:
            # position_side = 'buy' if new_position_side == 'long' else 'sell'
            # only trade if we arent in that market already, let the close positions logic do it otherwise
            if order.symbol == current_interest_symbol:
                ordered_already = True

        if not ordered_already:
            trade_entered_times[current_interest_symbol] = datetime.now()

            if new_position_side == 'long':
                made_money_recently_tmp[current_interest_symbol] = made_money_recently_pnl
            else:
                made_money_recently_tmp_shorting[current_interest_symbol] = made_money_recently_shorting_pnl
            bid = get_bid(current_interest_symbol)
            ask = get_ask(current_interest_symbol)
            return alpaca_wrapper.alpaca_order_stock(current_interest_symbol, row, price_to_trade_at, margin_multiplier,
                                                     new_position_side, bid, ask)
    return False


def make_trade_suggestions(predictions, minute_predictions):
    ### join predictions and minute predictions
    # convert to ints to join
    global made_money_recently
    global made_money_recently_tmp
    global made_money_one_before_recently
    global made_money_recently_shorting
    global made_money_recently_tmp_shorting
    global made_money_one_before_recently_shorting
    global current_flags
    predictions = predictions.merge(minute_predictions, how='outer', on='instrument', suffixes=['', '_minute'])

    with log_time("re setup vars for trade suggestions"):
        alpaca_wrapper.re_setup_vars()

    # sort df by close predicted price
    # where closemin_loss_trading_profit is positive
    # add new absolute movement column
    # predictions['absolute_movement'] = abs(predictions['close_predicted_price'] + predictions['close_predicted_price_minute'] * 3) # movement of both predictions
    predictions['either_profit_movement'] = abs(predictions['entry_takeprofit_profit_minute']) + abs(
        predictions['maxdiffprofit_profit_minute']) + abs(
        predictions['takeprofit_profit_minute']) + abs(predictions['entry_takeprofit_profit']) + abs(
        predictions['maxdiffprofit_profit']) + abs(
        predictions['takeprofit_profit'])
    # sort by close_predicted_price absolute movement
    predictions.sort_values(by=['either_profit_movement'], ascending=False, inplace=True)
    # do_trade = False
    has_traded = False
    # cancel any order open longer than 20 mins/recalculate it
    orders = alpaca_wrapper.get_open_orders()
    leftover_live_orders = []
    for order in orders:
        created_at = order.created_at
        if created_at < datetime.now(created_at.tzinfo) - timedelta(minutes=60 * 7):  # hr ?
            logger.info(
                f"canceling order open a long time {order.symbol} {order.side} {order.qty} {order.type} {order.time_in_force}")
            alpaca_wrapper.cancel_order(order)
        else:
            leftover_live_orders.append(order)
            # todo if amount changes then cancel order and re-place it
    # alpaca_wrapper.close_open_orders() # all orders cancelled/remade
    # todo exec top entry_trading_profit
    # make top 5 trades
    current_trade_count = 0
    with log_time("get positions"):
        all_positions = alpaca_wrapper.get_all_positions()
    # filter out crypto positions under .01 for eth - this too low amount cannot be traded/is an anomaly
    positions = filter_to_realistic_positions(all_positions)
    # # filter out crypto positions manually managed
    # positions = [position for position in positions if position.symbol not in ['BTCUSD', 'ETHUSD', 'LTCUSD', 'BCHUSD']]
    max_concurrent_trades = 13
    # todo go through tiny stock or btc orders and cancel them as they arent worth waiting for just for the data
    # for position in all_positions:
    ordered_or_positioned_instruments = set()
    for position in positions:
        ordered_or_positioned_instruments.add(position.symbol)
        # time_since_entered_position = todo entered time and not edit the data based on old positions
        if position.side == 'long':
            made_money_recently[position.symbol] = float(position.unrealized_plpc)
            made_money_one_before_recently[position.symbol] = made_money_recently_tmp.get(position.symbol, 0)
        else:
            made_money_recently_shorting[position.symbol] = float(position.unrealized_plpc)
            made_money_one_before_recently_shorting[position.symbol] = made_money_recently_tmp_shorting.get(
                position.symbol, 0)
    for order in leftover_live_orders:
        ordered_or_positioned_instruments.add(order.symbol)
    max_trades_available = max_concurrent_trades - len(ordered_or_positioned_instruments)

    ## if all predictions (but not crypto) make over 230 then cancel all trades and close positions for the day

    total_money_made = 0
    for position in positions:
        if position.symbol not in crypto_symbols:
            total_money_made += float(position.unrealized_pl)
    # if total_money_made > 500: # todo reevaluate if we need this?
    #     logger.info("total made today is over 500, closing all positions and not trading anymore today")
    #     total_money_made = total_money_made * -1
    #     current_flags['trading_today'] = False
    #     alpaca_wrapper.close_open_orders()
    #     alpaca_wrapper.backout_all_non_crypto_positions(positions, predictions)

    if datetime.now().hour == 12:
        current_flags['trading_today'] = True

    if current_flags.get('trading_today', True) == False:
        logger.info('not trading today, already made money ')
        return
    for index, row in predictions.iterrows():

        # if row['close_predicted_price'] > 0:
        # check that close_predicted_price and close_predicted_price_minute dont have opposite signs
        #
        # if row['close_predicted_price'] * row['close_predicted_price_minute'] < 0:
        #     logger.info(f"conflicting preds {row['instrument']} {row['close_predicted_price']} {row['close_predicted_price_minute']}")
        #     continue
        # both made profit also sued to use row['takeprofit_profit'] > 0 and
        # extra profit check for buying crypto which has higher fees
        # todo try not to aggressive trade on high spreads
        # todo order at market price meaning the bid price is buying not the ask price
        spread = get_spread(row['instrument'])  # todo high spread fast trading to take advantage of high spread?
        if row['instrument'] not in crypto_symbols and not (
                (row['entry_takeprofit_profit'] > 0 and
                 row[
                     'entry_takeprofit_profit_minute'] > 0) or (
                        row['maxdiffprofit_profit'] > 0 and
                        row[
                            'maxdiffprofit_profit_minute'] > 0)):
            logger.info(
                f"not trading {row['instrument']} takeprofit {row['entry_takeprofit_profit']} takeprofitminute {row['entry_takeprofit_profit_minute']}")
            continue

        if row['instrument'] in crypto_symbols and not (
                (row['entry_takeprofit_profit'] - (CRYPTO_TRADING_FEE * 2) > 0 and
                 row[
                     'entry_takeprofit_profit_minute'] - (CRYPTO_TRADING_FEE * 2) > 0) or (
                        row['maxdiffprofit_profit'] - (CRYPTO_TRADING_FEE * 2) > 0 and
                        row[
                            'maxdiffprofit_profit_minute'] - (CRYPTO_TRADING_FEE * 2) > 0)):
            logger.info(
                f"not trading {row['instrument']} takeprofit {row['entry_takeprofit_profit']} takeprofitminute {row['entry_takeprofit_profit_minute']}")
            continue

        logger.info("Trade suggestion")
        logger.info(row)

        if current_trade_count >= max_trades_available:
            break
        # either most profitable strategy is picked
        if use_stale_data:
            logger.info('using stale data so, not actually trading')
            continue
        has_traded = buy_stock(row, predictions, positions, leftover_live_orders)
        if has_traded:
            current_trade_count += 1
        do_trade = True
        # break
    # if not has_traded:
    #     logger.info("No trade suggestions, trying to exit position")
    close_profitable_trades(predictions, positions, leftover_live_orders)
    # fake position to close any btc in binance smartly
    # TODO remove this hack
    # not going to go through in alpaca is it's a huge order
    btc_position = Position(symbol='BTCUSD', qty='1000', side='long', avg_entry_price='18000', unrealized_plpc='0.1',
                            unrealized_pl='0.1', market_value='5000',
                            asset_id=uuid.uuid4(),
                            exchange='FTXU',
                            asset_class='crypto',
                            cost_basis='1',
                            unrealized_intraday_pl='1',
                            unrealized_intraday_plpc='1',
                            current_price='100000',
                            lastday_price='1',
                            change_today='1',
                            )
    # close_profitable_trades(predictions, [btc_position], leftover_live_orders, False)
    close_profitable_crypto_binance_trades(predictions, [btc_position], leftover_live_orders, False)

    sleep(60)


if __name__ == '__main__':
    while True:
        try:
            # skip running logic if not us stock exchange ?

            do_forecasting()
        except Exception as e:
            traceback.print_exc()

            logger.exception(e)
            logger.info(e)
        # sleep for 1 minutes
        logger.info("Sleeping for 5min")
        sleep(60 * 5)

    # make_trade_suggestions(pd.read_csv('/home/lee/code/stock/results/predictions-2021-12-23_23-04-07.csv'))
