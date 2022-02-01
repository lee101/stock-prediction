import random
import traceback
from collections import defaultdict
from datetime import datetime, timedelta
from time import sleep

from loguru import logger
from pandas import DataFrame

import alpaca_wrapper
from data_curate_minute import download_minute_stock_data
from data_curate_daily import download_daily_stock_data
# from predict_stock import make_predictions
from decorator_utils import timeit
from predict_stock_forecasting import make_predictions


# read do_retrain argument from argparse
# do_retrain = True
use_stale_data = False

daily_predictions = DataFrame()
@timeit
def do_forecasting():
    global daily_predictions

    if daily_predictions.empty:

        if use_stale_data:
            current_time_formatted = '2021-12-05 18:20:29'
            current_time_formatted = '2021-12-09 12:16:26'  # new/ more data
            current_time_formatted = '2021-12-11 07:57:21-2'  # new/ less data tickers
            current_time_formatted = 'min' # new/ less data tickers
            current_time_formatted = '2021-12-30 20:11:47'  # new/ 30 minute data
        else:
            current_time_formatted = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            download_daily_stock_data(current_time_formatted)
        daily_predictions = make_predictions(current_time_formatted)


    current_time_formatted = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    download_minute_stock_data(current_time_formatted)
    minute_predictions = make_predictions(current_time_formatted, pred_name='minute')

    make_trade_suggestions(daily_predictions, minute_predictions)

def close_profitable_trades(all_preds, orders):
    positions = alpaca_wrapper.list_positions()
    global made_money_recently
    # close all positions that are not in this current held stock
    already_held_stock = False
    has_traded = False
    for position in positions:
        if float(position.unrealized_pl) < 0:
            made_money_recently[position.symbol] = False
        else:
            made_money_recently[position.symbol] = True

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
                #     print(f"Closing predicted to worsen position {position.symbol}")
                ordered_time = trade_entered_times.get(position.symbol)
                if ordered_time and ordered_time < datetime.now() - timedelta(minutes=60 * 1.5):
                    # close other orders for pair
                    for order in orders:
                        if order.symbol == position.symbol:
                            alpaca_wrapper.cancel_order(order)
                            # todo check if we have one open that is trying to close already?
                    #close old position, not been hitting out preditctions
                    alpaca_wrapper.close_position_at_current_price(position, row)
                    print(f"Closing bad position to reduce risk {position.symbol}")

                else:
                    exit_strategy = 'maxdiff' # TODO bug - should be based on what entry strategy should be
                    # takeprofit_profit also a thing
                    if float(row['maxdiffprofit_profit']) < float(row['entry_takeprofit_profit']):
                        exit_strategy = 'entry'

                    entry_price = float(position.avg_entry_price)
                    if position.side == 'long':
                        predicted_high = row['entry_takeprofit_high_price_minute']
                        if exit_strategy == 'entry':
                            if abs(row['entry_takeprofit_profit_high_multiplier_minute']) > .01: # tuned for minutely
                                predicted_high = row['high_predicted_price_value_minute']
                        elif exit_strategy == 'maxdiff':
                            if abs(row['maxdiffprofit_profit_high_multiplier_minute']) > .01: # tuned for minutely
                                predicted_high = row['maxdiffprofit_high_price_minute']
                        sell_price = predicted_high
                        if not ordered_time or ordered_time > datetime.now() - timedelta(minutes=25):
                            # close new orders at atleast a profit
                            margin_default_high = entry_price * (1 + .0015)
                            sell_price = max(predicted_high, margin_default_high)
                        # only if no other orders already
                        ordered_already = False
                        for order in orders:
                            if order.side == 'buy' and order.symbol == position.symbol:
                                ordered_already = True
                        if not ordered_already:
                            alpaca_wrapper.open_take_profit_position(position, row, sell_price)
                    elif position.side == 'sell':
                        predicted_low = row['entry_takeprofit_low_price_minute']
                        if exit_strategy == 'entry':
                            if abs(row['entry_takeprofit_profit_low_multiplier_minute']) > .01:
                                predicted_low = row['low_predicted_price_value_minute']
                        elif exit_strategy == 'maxdiff':
                            if abs(row['maxdiffprofit_profit_low_multiplier_minute']) > .01:
                                predicted_low = row['maxdiffprofit_low_price_minute']
                        sell_price = predicted_low
                        if not ordered_time or ordered_time > datetime.now() - timedelta(minutes=25):
                            # close new orders at atleast a profit
                            margin_default_low = entry_price * (1 - .0015)
                            sell_price = min(predicted_low, margin_default_low)
                        # only if no other orders already
                        ordered_already = False
                        for order in orders:
                            if order.side == 'sell' and order.symbol == position.symbol:
                                ordered_already = True
                        if not ordered_already:
                            alpaca_wrapper.open_take_profit_position(position, row, sell_price)

                # else:
                #     pass
                    # instant close?
                    # if float(position.unrealized_plpc) > 0.004:  ## or float(position.unrealized_pl) > 50:
                    #     print(f"Closing good position")
                    #     alpaca_wrapper.close_position_violently(position)
                    # todo test take profit?
                    # alpaca_wrapper.open_take_profit_position(position, row)
                    # print(
                    #     f"keeping position {position.symbol} - predicted to get better - open takeprofit at {row['close_last_price_minute'] }")



made_money_recently = defaultdict(bool)
trade_entered_times = {}

def buy_stock(row, all_preds, positions, orders):
    """
    or sell stock
    :param row:
    :return:
    """
    global made_money_recently

    currentInterestSymbol = row['instrument']
    # close all positions that are not in this current held stock
    already_held_stock = False
    entry_strategy = 'maxdiff'
    # takeprofit_profit is also a thing
    if float(row['maxdiffprofit_profit']) < float(row['entry_takeprofit_profit']):
        entry_strategy = 'entry'

    if entry_strategy == 'maxdiff':
        low_to_close_diff = abs(row['low_predicted_price_minute'] - row['close_predicted_price_minute'])
        high_to_close_diff = abs(row['high_predicted_price_minute'] - row['close_predicted_price_minute'])

        new_position_side = 'short' if low_to_close_diff > high_to_close_diff else 'long' # maxdiff max profit potential
    elif entry_strategy == 'entry':
        new_position_side = 'short' if row['close_predicted_price_minute'] < 0 else 'long' # just the end price 15min from now- dont worry about the extremes

    has_traded = False
    for position in positions:
        if float(position.unrealized_pl) < 0:
            made_money_recently[position.symbol] = False
        else:
            made_money_recently[position.symbol] = True

        if position.symbol != currentInterestSymbol:
            ## dont trade until we made money
            if float(position.unrealized_pl) < 0 and float(position.unrealized_plpc) < 0: # think more carefully about jumping off positions until we make good profit
                # skip closing bad positions, sometimes wait for a while before jumping between stock
                # if not at market open
                current_time = datetime.now()
                at_market_open = False
                # if current_time.hour == 1:
                #     at_market_open = True
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
                #     print(f"Closing worsening bad position {position.symbol}")
                #
                # else:
                # done later
                # already_held_stock = True
                # print(
                #     f"hodling bad position {position.symbol} instead of {new_position_side} {currentInterestSymbol} - predicted to get better")

            else:
                # alpaca_wrapper.close_position_violently(position)
                # has_traded = True
                # print(f"Closing position {position.symbol}")
                # alpaca_wrapper.close_position_violently(position)
                has_traded = False
                print(f"Not jumping to buy other stock - ")
        elif position.side == new_position_side:
            print("Already holding {}".format(currentInterestSymbol))
            already_held_stock = True
        # may cause overtrading
        # else:
        #     alpaca_wrapper.close_position_at_current_price(position, row)
        #     has_traded = True
        #     print(f"changing stance on {currentInterestSymbol} to {new_position_side}")

    if not already_held_stock:
        print(f"{new_position_side} {currentInterestSymbol}")
        margin_multiplier = 1. / 5.0
        # if not made_money_recently[currentInterestSymbol]:
        #     margin_multiplier = .03

        trade_entered_times[currentInterestSymbol] = datetime.now()
        current_price = row['close_last_price_minute']

        price_to_trade_at = max(current_price, row['high_last_price_minute'])
        if new_position_side == 'buy':
            predicted_low = row['entry_takeprofit_low_price_minute']
            if abs(row['entry_takeprofit_profit_low_multiplier_minute']) > .01:
                predicted_low = row['low_predicted_price_value_minute']
            price_to_trade_at = min(current_price, predicted_low)
            price_to_trade_at = min(current_price, row['low_last_price_minute'])
        elif new_position_side == 'sell':
            predicted_high = row['entry_takeprofit_high_price_minute']
            if abs(row['entry_takeprofit_profit_high_multiplier_minute']) > .01:  # tuned for minutely
                predicted_high = row['high_predicted_price_value_minute']
            price_to_trade_at = max(current_price, predicted_high)
        # ONLY trade if we aren't trading in that dir already
        ordered_already = False

        for order in orders:
            position_side = 'buy' if new_position_side == 'long' else 'sell'
            if order.side == position_side and order.symbol == currentInterestSymbol:
                ordered_already = True
        if not ordered_already:
            alpaca_wrapper.buy_stock(currentInterestSymbol, row, price_to_trade_at, margin_multiplier, new_position_side)
            return True
    return has_traded


def make_trade_suggestions(predictions, minute_predictions):
    ### join predictions and minute predictions
    # convert to ints to join

    predictions = predictions.merge(minute_predictions, how='outer', on='instrument', suffixes=['', '_minute'])


    alpaca_wrapper.re_setup_vars()

    # sort df by close predicted price
    # where closemin_loss_trading_profit is positive
    # add new absolute movement column
    # predictions['absolute_movement'] = abs(predictions['close_predicted_price'] + predictions['close_predicted_price_minute'] * 3) # movement of both predictions
    predictions['either_profit_movement'] = abs(predictions['entry_takeprofit_profit_minute']) + abs(
        predictions['maxdiffprofit_profit_minute'])
    # sort by close_predicted_price absolute movement
    predictions.sort_values(by=['either_profit_movement'], ascending=False, inplace=True)
    do_trade = False
    has_traded = False
    # cancel any order open longer than 20 mins/recalculate it
    orders = alpaca_wrapper.get_open_orders()
    for order in orders:
        created_at = order.created_at
        if created_at < datetime.now(created_at.tzinfo) - timedelta(minutes=15):
            alpaca_wrapper.cancel_order(order)
            # todo if amount changes then cancel order and re-place it
    # alpaca_wrapper.close_open_orders() # all orders cancelled/remade
    # todo exec top entry_trading_profit
    # make top 5 trades
    current_trade_count = 0
    positions = alpaca_wrapper.list_positions()
    max_concurrent_trades = 5
    max_trades_available = max_concurrent_trades - len(positions)
    for index, row in predictions.iterrows():

        # if row['close_predicted_price'] > 0:
        # check that close_predicted_price and close_predicted_price_minute dont have opposite signs
        #
        # if row['close_predicted_price'] * row['close_predicted_price_minute'] < 0:
        #     print(f"conflicting preds {row['instrument']} {row['close_predicted_price']} {row['close_predicted_price_minute']}")
        #     continue
        # both made profit also sued to use row['takeprofit_profit'] > 0 and
        if row['entry_takeprofit_profit_minute'] > 0 or row['maxdiffprofit_profit_minute'] > 0:
            print("Trade suggestion")
            print(row)

            if current_trade_count >= max_trades_available:
                break
            # either most profitable strategy is picked
            has_traded = buy_stock(row, predictions, positions, orders)
            if has_traded:
                current_trade_count += 1
            do_trade = True
            # break
    # if not has_traded:
    #     print("No trade suggestions, trying to exit position")
    close_profitable_trades(predictions, orders)

    sleep(5)


if __name__ == '__main__':
    # in development, use the following line to avoid re downloading data
    while True:
        try:
            # skip running logic if not us stock exchange ?

            do_forecasting()
        except Exception as e:
            traceback.print_exc()

            logger.exception(e)
            print(e)
        # sleep for 1 minutes
        print("Sleeping for 5sec")
        sleep(5)

    # make_trade_suggestions(pd.read_csv('/home/lee/code/stock/results/predictions-2021-12-23_23-04-07.csv'))
