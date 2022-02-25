import random
import traceback
from ast import literal_eval
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep

import numpy as np
import torch
from loguru import logger
from pandas import DataFrame

import alpaca_wrapper
from data_curate_minute import download_minute_stock_data
from data_curate_daily import download_daily_stock_data
# from predict_stock import make_predictions
from decorator_utils import timeit
from jsonshelve import FlatShelf
from predict_stock_forecasting import make_predictions
import shelve

# read do_retrain argument from argparse
# do_retrain = True
from src.fixtures import crypto_symbols

use_stale_data = False

daily_predictions = DataFrame()
daily_predictions_time = None
@timeit
def do_forecasting():
    global daily_predictions
    global daily_predictions_time

    if daily_predictions.empty and (daily_predictions_time is None or daily_predictions_time < datetime.now() - timedelta(days=1)):
        daily_predictions_time = datetime.now()
        if use_stale_data:
            current_time_formatted = '2021-12-05 18:20:29'
            current_time_formatted = '2021-12-09 12:16:26'  # new/ more data
            current_time_formatted = '2021-12-11 07:57:21-2'  # new/ less data tickers
            current_time_formatted = 'min' # new/ less data tickers
            current_time_formatted = '2021-12-30 20:11:47'  # new/ 30 minute data
        else:
            current_time_formatted = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
            download_daily_stock_data(current_time_formatted, True)
        daily_predictions = make_predictions(current_time_formatted, retrain=True) # TODO


    current_time_formatted = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    download_daily_stock_data(current_time_formatted)
    minute_predictions = make_predictions(current_time_formatted)

    make_trade_suggestions(daily_predictions, minute_predictions)

def close_profitable_trades(all_preds, positions, orders):
    global made_money_recently
    global made_money_one_before_recently
    global made_money_recently_tmp

    # close all positions that are not in this current held stock
    already_held_stock = False
    has_traded = False
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
                ordered_time = trade_entered_times.get(position.symbol)
                is_crypto = position.symbol in crypto_symbols
                is_trading_day_ending = False # todo investigate reenabling this logic
                if is_crypto:
                    is_trading_day_ending = datetime.now().hour in [11, 12, 13] # TODO nzdt specific code here
                else:
                    is_trading_day_ending = datetime.now().hour in [9,10,11,12] # last 30 mins
                if not ordered_time or ordered_time < datetime.now() - timedelta(minutes=60 * 16):
                    if float(position.unrealized_plpc) < 0:
                        change_time = instrument_strategy_change_times.get(position.symbol)
                        if not change_time or change_time < datetime.now() - timedelta(minutes=30 * 16.):
                            instrument_strategy_change_times[position.symbol] = datetime.now()
                            current_strategy = instrument_strategies.get(position.symbol, 'aggressive_buy')

                            available_strategies = {'aggressive', 'aggressive_buy', 'aggressive_sell', 'entry'} - { current_strategy }
                            if current_strategy.startswith('aggressive'):
                                available_strategies = available_strategies - {'aggressive'}
                            new_strategy = random.choice(list(available_strategies))
                            logger.info(f"Changing strategy for {position.symbol} from {current_strategy} to {new_strategy}")
                            instrument_strategies[position.symbol] = new_strategy
                # todo check time in market not overall time
                if (not ordered_time or ordered_time < datetime.now() - timedelta(minutes=60 * 23 + 20)) and is_trading_day_ending:
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

                    alpaca_wrapper.close_position_at_current_price(position, row)
                    logger.info(f"Closing position to reduce risk {position.symbol}")

                else:
                    exit_strategy = 'maxdiff' # TODO bug - should be based on what entry strategy should be
                    # takeprofit_profit also a thing
                    if float(row['maxdiffprofit_profit']) < float(row['entry_takeprofit_profit']):
                        exit_strategy = 'entry'

                    entry_price = float(position.avg_entry_price)
                    if position.side == 'long':
                        predicted_high = row['entry_takeprofit_high_price']
                        if exit_strategy == 'entry':
                            if abs(row['entry_takeprofit_profit_high_multiplier']) > .01: # tuned for minutely
                                predicted_high = row['high_predicted_price_value']
                        elif exit_strategy == 'maxdiff':
                            if abs(row['maxdiffprofit_profit_high_multiplier']) > .01: # tuned for minutely
                                predicted_high = row['maxdiffprofit_high_price']
                        sell_price = predicted_high
                        if not ordered_time or ordered_time > datetime.now() - timedelta(minutes=3*60):
                            # close new orders at atleast a profit
                            margin_default_high = entry_price * (1 + .003)
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
                        if not ordered_time or ordered_time > datetime.now() - timedelta(minutes=3*60):
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


data_dir = Path(__file__).parent / 'data'

made_money_recently = FlatShelf(str(data_dir / f"made_money_recently.db.json"))
made_money_recently_tmp = FlatShelf(str(data_dir / f"made_money_recently_tmp.db.json"))
made_money_one_before_recently = FlatShelf(str(data_dir / f"made_money_one_before_recently.db.json"))

made_money_recently_shorting = FlatShelf(str(data_dir / f"made_money_recently_shorting.db.json"))
made_money_recently_tmp_shorting = FlatShelf(str(data_dir / f"made_money_recently_tmp_shorting.db.json"))
made_money_one_before_recently_shorting = FlatShelf(str(data_dir / f"made_money_one_before_recently_shorting.db.json"))

trade_entered_times = shelve.open(str(data_dir / f"trade_entered_times.db"))
# all_historical_orders = shelve.open(str(data_dir / f"all_historical_orders.db"))

instrument_strategies = FlatShelf(str(data_dir / f"instrument_strategies.db.json"))
instrument_strategy_change_times = shelve.open(str(data_dir / f"instrument_strategy_change_times.db.json"))

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

    current_interest_symbol = row['instrument']
    # close all positions that are not in this current held stock
    already_held_stock = False
    entry_strategy = 'maxdiff'
    # takeprofit_profit is also a thing
    if float(row['maxdiffprofit_profit']) + float(row['maxdiffprofit_profit_minute']) < float(row['entry_takeprofit_profit']) + float(row['entry_takeprofit_profit_minute']):
        entry_strategy = 'entry'
        logger.info(f"using entry strategy for {current_interest_symbol}")

    if entry_strategy == 'maxdiff':
        # maxdiff based side similar to simulation
        # already calculated for the minute, but use current price for old low/high
        low_to_close_diff = abs(1 - (row['low_predicted_price_value'] / row['close_last_price_minute'])) + abs(row['latest_low_diff_minute'])
        high_to_close_diff = abs(1 - (row['high_predicted_price_value'] / row['close_last_price_minute'])) + abs(row['latest_high_diff_minute'])

        new_position_side = 'short' if low_to_close_diff > high_to_close_diff else 'long' # maxdiff max profit potential
    elif entry_strategy == 'entry':
        now_to_old_pred = 1 - (row['close_predicted_price_value_minute'] / row['close_last_price_minute'])
        new_position_side = 'short' if now_to_old_pred + row['close_predicted_price_minute'] < 0 else 'long' # just the end price 15min from now- dont worry about the extremes
    # also try the minmax or takeprofit strategy that doesn't trade at said price
    entry_price_strategy = 'minmax' # at predicted low/high
    if float(row['takeprofit_profit']) + float(row['takeprofit_profit_minute']) < float(
            row['entry_takeprofit_profit']) + float(row['entry_takeprofit_profit_minute']):
        entry_price_strategy = 'entry' # at current market price

    has_traded = False
    for position in positions:
        if position.side == 'long':
            made_money_recently[position.symbol] = float(position.unrealized_plpc)
            made_money_one_before_recently[position.symbol] = made_money_recently_tmp.get(position.symbol, 0)
        else:
            made_money_recently_shorting[position.symbol] = float(position.unrealized_plpc)
            made_money_one_before_recently_shorting[position.symbol] = made_money_recently_tmp_shorting.get(position.symbol, 0)

        if position.symbol != current_interest_symbol:
            ## dont trade until we made money
            if float(position.unrealized_pl) < 0 and float(position.unrealized_plpc) < 0:
                pass
                # think more carefully about jumping off positions until we make good profit
                # skip closing bad positions, sometimes wait for a while before jumping between stock
                # if not at market open

                # find stance on current position
                # can close if we predict it to get worse
                # is_worsening_position = False
                # for index, row in all_preds.iterrows():
                #     if row['instrument'] == position.symbol:
                #         # make it reasonably easy to back out of bad trades
                #         if (row['close_predicted_price_minute'] < 0) and position.side == 'long':
                #             is_worsening_position = True
                #             break
                #         if (row['close_predicted_price_minute'] > 0) and position.side == 'short':
                #             is_worsening_position = True
                #             break
                # if random.choice([True, False, False, False, False, False, False, False, False, False, False, False, False, False, False]) or at_market_open:
                # if is_worsening_position:
                #     alpaca_wrapper.close_position_violently(position)
                #     has_traded = True
                #     logger.info(f"Closing worsening bad position {position.symbol}")
                #
                # else:
                # done later
                # already_held_stock = True
                # logger.info(
                #     f"hodling bad position {position.symbol} instead of {new_position_side} {current_interest_symbol} - predicted to get better")

            else:
                # has_traded = True
                # logger.info(f"Closing position {position.symbol}")
                # alpaca_wrapper.close_position_violently(position)
                has_traded = False
                logger.info(f"Not jumping to buy other stock - ")
        elif position.side == new_position_side: # todo could this prevent you from margining upward? should we clear all positions first?
            logger.info("Already holding {}".format(current_interest_symbol))
            already_held_stock = True
            already_held_amount = position.qty
        # may cause overtrading
        # else:
        #     alpaca_wrapper.close_position_at_current_price(position, row)
        #     has_traded = True
        #     logger.info(f"changing stance on {current_interest_symbol} to {new_position_side}")

    if not already_held_stock:
        logger.info(f"{new_position_side} {current_interest_symbol}")
        margin_multiplier = (1. / 10.0) * .8  # leave some room

        if entry_price_strategy == 'entry':
            if sum(literal_eval(row['entry_takeprofit_profit_values'])[:-2]) <= 0:
                logger.info(
                    f"{current_interest_symbol} is loosing money over two days via entry takeprofit, making a small trade")

                margin_multiplier = (1. / 10.0) * .3 # last trade values are loosing half trade
        else:
            if sum(literal_eval(row['takeprofit_profit_values'])[:-2]) <= 0:
                logger.info(
                    f"{current_interest_symbol} is loosing money over two days via takeprofit, making a small trade")
                margin_multiplier = (1. / 10.0) * .3 # last trade values are loosing half trade

        if entry_strategy == 'maxdiff':
            if sum(literal_eval(row['maxdiffprofit_profit_values'])[:-2]) <= 0:
                logger.info(
                    f"{current_interest_symbol} is loosing money over two days via maxdiff, making a small trade")
                margin_multiplier = (1. / 10.0) * .3 # last trade values are loosing half trade


        if new_position_side == 'long':
            if (made_money_recently.get(current_interest_symbol, 0) or 0) + (made_money_one_before_recently.get(current_interest_symbol, 0) or 0) <= 0:
                # if loosing money over two trades, make a small trade /recalculate
                margin_multiplier = .001
                logger.info(f"{current_interest_symbol} is loosing money over two trades, making a small trade")
        else:
            if (made_money_recently_shorting.get(current_interest_symbol, 0) or 0) + (made_money_one_before_recently_shorting.get(current_interest_symbol, 0) or 0) <= 0:
                # if loosing money over two trades, make a small trade /recalculate
                margin_multiplier = .001
                logger.info(f"{current_interest_symbol} is loosing money over two trades via shorting, making a small trade")



        trade_entered_times[current_interest_symbol] = datetime.now()
        current_price = row['close_last_price_minute']

        price_to_trade_at = max(current_price, row['high_last_price_minute'])
        current_strategy = instrument_strategies.get(current_interest_symbol, 'aggressive_buy')

        if new_position_side == 'long':
            predicted_low = row['takeprofit_low_price_minute']
            if abs(row['takeprofit_profit_low_multiplier_minute']) > .04:
                predicted_low = row['low_predicted_price_value_minute']
            price_to_trade_at = min(current_price, predicted_low) #, row['low_last_price_minute'])
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
            position_side = 'buy' if new_position_side == 'long' else 'sell'
            if order.side == position_side and order.symbol == current_interest_symbol:
                ordered_already = True
        if not ordered_already:
            if new_position_side == 'long':
                made_money_recently_tmp[current_interest_symbol] = made_money_recently.get(current_interest_symbol, 0)
            else:
                made_money_recently_tmp_shorting[current_interest_symbol] = made_money_recently_shorting.get(current_interest_symbol, 0)

            alpaca_wrapper.buy_stock(current_interest_symbol, row, price_to_trade_at, margin_multiplier, new_position_side)
            return True
    return has_traded


def make_trade_suggestions(predictions, minute_predictions):
    ### join predictions and minute predictions
    # convert to ints to join
    global made_money_recently
    global made_money_recently_tmp
    global made_money_one_before_recently
    global made_money_recently_shorting
    global made_money_recently_tmp_shorting
    global made_money_one_before_recently_shorting

    predictions = predictions.merge(minute_predictions, how='outer', on='instrument', suffixes=['', '_minute'])


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
    do_trade = False
    has_traded = False
    # cancel any order open longer than 20 mins/recalculate it
    orders = alpaca_wrapper.get_open_orders()
    leftover_live_orders = []
    for order in orders:
        created_at = order.created_at
        if created_at < datetime.now(created_at.tzinfo) - timedelta(minutes=60*7):
            alpaca_wrapper.cancel_order(order)
        else:
            leftover_live_orders.append(order)
            # todo if amount changes then cancel order and re-place it
    # alpaca_wrapper.close_open_orders() # all orders cancelled/remade
    # todo exec top entry_trading_profit
    # make top 5 trades
    current_trade_count = 0
    all_positions = alpaca_wrapper.list_positions()
    # filter out crypto positions under .01 for eth - this too low amount cannot be traded/is an anomaly
    positions = []
    for position in all_positions:
        if position.symbol in ['ETHUSD', 'LTCUSD'] and float(position.qty) >= .01:
            positions.append(position)
        elif position.symbol in ['BTCUSD'] and float(position.qty) >= .001:
            positions.append(position)
        elif position.symbol not in crypto_symbols:
            positions.append(position)
    # # filter out crypto positions manually managed
    # positions = [position for position in positions if position.symbol not in ['BTCUSD', 'ETHUSD', 'LTCUSD', 'BCHUSD']]
    max_concurrent_trades = 13

    ordered_or_positioned_instruments = set()
    for position in positions:
        ordered_or_positioned_instruments.add(position.symbol)

        if position.side == 'long':
            made_money_recently[position.symbol] = float(position.unrealized_plpc)
            made_money_one_before_recently[position.symbol] = made_money_recently_tmp.get(position.symbol, 0)
        else:
            made_money_recently_shorting[position.symbol] = float(position.unrealized_plpc)
            made_money_one_before_recently_shorting[position.symbol] = made_money_recently_tmp_shorting.get(position.symbol, 0)
    for order in leftover_live_orders:
        ordered_or_positioned_instruments.add(order.symbol)
    max_trades_available = max_concurrent_trades - len(ordered_or_positioned_instruments)

    for index, row in predictions.iterrows():

        # if row['close_predicted_price'] > 0:
        # check that close_predicted_price and close_predicted_price_minute dont have opposite signs
        #
        # if row['close_predicted_price'] * row['close_predicted_price_minute'] < 0:
        #     logger.info(f"conflicting preds {row['instrument']} {row['close_predicted_price']} {row['close_predicted_price_minute']}")
        #     continue
        # both made profit also sued to use row['takeprofit_profit'] > 0 and
        if (row['entry_takeprofit_profit'] > 0 and row['entry_takeprofit_profit_minute'] > 0) or (
                row['maxdiffprofit_profit'] > 0 and row['maxdiffprofit_profit_minute'] > 0):
            logger.info("Trade suggestion")
            logger.info(row)

            if current_trade_count >= max_trades_available:
                break
            # either most profitable strategy is picked
            has_traded = buy_stock(row, predictions, positions, leftover_live_orders)
            if has_traded:
                current_trade_count += 1
            do_trade = True
            # break
    # if not has_traded:
    #     logger.info("No trade suggestions, trying to exit position")
    close_profitable_trades(predictions, positions, leftover_live_orders)

    sleep(20)


if __name__ == '__main__':
    # in development, use the following line to avoid re downloading data
    while True:
        try:
            # skip running logic if not us stock exchange ?

            do_forecasting()
        except Exception as e:
            traceback.print_exc()

            logger.exception(e)
            logger.info(e)
        # sleep for 1 minutes
        logger.info("Sleeping for 5sec")
        sleep(60*5)

    # make_trade_suggestions(pd.read_csv('/home/lee/code/stock/results/predictions-2021-12-23_23-04-07.csv'))
