import random
from datetime import datetime
from time import sleep

import alpaca_wrapper
from data_curate import download_daily_stock_data
# from predict_stock import make_predictions
from decorator_utils import timeit
from predict_stock_forecasting import make_predictions


# read do_retrain argument from argparse
# do_retrain = True


@timeit
def do_forecasting():
    if False:
        current_time_formatted = '2021-12-05 18:20:29'
        current_time_formatted = '2021-12-09 12:16:26'  # new/ more data
        current_time_formatted = '2021-12-11 07:57:21-2'  # new/ less data tickers
        # current_time_formatted = 'min' # new/ less data tickers
        # current_time_formatted = '2021-12-30 20:11:47'  # new/ 30 minute data
    else:
        current_time_formatted = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        download_daily_stock_data(current_time_formatted)
    predictions = make_predictions(current_time_formatted)

    make_trade_suggestions(predictions)


def buy_stock(row, all_preds):
    """
    or sell stock
    :param row:
    :return:
    """
    positions = alpaca_wrapper.list_positions()
    currentBuySymbol = row['instrument']
    # close all positions that are not in this current held stock
    already_held_stock = False
    new_position_side = 'short' if row['close_predicted_price'] < 0 else 'long'
    for position in positions:
        if position.symbol != currentBuySymbol:
            ## dont trade until we made money
            if float(position.unrealized_pl) < 0 and float(position.unrealized_plpc) < 0.001:
                # skip closing bad positions, sometimes wait for a while before jumping between stock
                # if not at market open
                current_time = datetime.now()
                at_market_open = False
                # if current_time.hour == 1:
                #     at_market_open = True
                # find stance on current position
                # can close if we predict it to get worse
                is_worsening_position = False
                for index, row in all_preds.iterrows():
                    if row['instrument'] == position.symbol:
                        if row['close_predicted_price'] < 0 and position.side == 'long':
                            is_worsening_position = True
                            break
                        if row['close_predicted_price'] > 0 and position.side == 'short':
                            is_worsening_position = True
                            break
                # if random.choice([True, False, False, False, False, False, False, False, False, False, False, False, False, False, False]) or at_market_open:
                if is_worsening_position:
                    alpaca_wrapper.close_position(position)
                    print(f"Closing worsening bad position {position.symbol}")

                else:
                    already_held_stock = True
                    print(
                        f"hodling bad position {position.symbol} instead of {new_position_side} {currentBuySymbol} - predicted to get better")

            else:
                alpaca_wrapper.close_position(position)
        elif position.side == new_position_side:
            print("Already holding {}".format(currentBuySymbol))
            already_held_stock = True
        else:
            alpaca_wrapper.close_position(position)
            print(f"changing stance on {currentBuySymbol} to {new_position_side}")

    if not already_held_stock:
        print(f"{new_position_side} {currentBuySymbol}")
        alpaca_wrapper.buy_stock(currentBuySymbol, row)


def make_trade_suggestions(predictions):
    alpaca_wrapper.re_setup_vars()
    # sort df by close predicted price
    # where closemin_loss_trading_profit is positive
    # add new absolute movement column
    predictions['absolute_movement'] = abs(predictions['close_predicted_price'])
    # sort by close_predicted_price absolute movement
    predictions.sort_values(by=['absolute_movement'], ascending=False, inplace=True)
    for index, row in predictions.iterrows():

        # if row['close_predicted_price'] > 0:
        if row['closemin_loss_trading_profit'] > 0:
            print("Trade suggestion")
            print(row)
            alpaca_wrapper.close_open_orders()
            buy_stock(row, predictions)
            break


if __name__ == '__main__':
    # in development, use the following line to avoid re downloading data
    while True:
        try:
            do_forecasting()
        except Exception as e:
            print(e)
        # sleep for 1 minutes
        print("Sleeping for 5sec")
        sleep(5)

    # make_trade_suggestions(pd.read_csv('/home/lee/code/stock/results/predictions-2021-12-23_23-04-07.csv'))
