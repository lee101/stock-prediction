from datetime import datetime

import alpaca_wrapper
from data_curate import download_daily_stock_data
# from predict_stock import make_predictions
from decorator_utils import timeit
from predict_stock_forecasting import make_predictions


@timeit
def do_forecasting():
    if True:
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


def buy_stock(row):
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
            alpaca_wrapper.close_position(position)
        elif position.side == new_position_side:
            print("Already holding {}".format(currentBuySymbol))
            already_held_stock = True
        else:
            alpaca_wrapper.close_position(position)

    if not already_held_stock:
        print("Buying {}".format(currentBuySymbol))
        alpaca_wrapper.buy_stock(currentBuySymbol, row)


def make_trade_suggestions(predictions):
    # sort df by close predicted price
    # where closemin_loss_trading_profit is positive
    predictions.sort_values(by=['closemin_loss_trading_profit'], ascending=False, inplace=True)
    for index, row in predictions.iterrows():
        print("Trade suggestion")
        print(row)
        # if row['close_predicted_price'] > 0:
        if row['closemin_loss_trading_profit'] > 0:
            buy_stock(row)
        # trade

        break


if __name__ == '__main__':
    # in development, use the following line to avoid re downloading data
    do_forecasting()
    # make_trade_suggestions(pd.read_csv('/home/lee/code/stock/results/predictions-2021-12-23_23-04-07.csv'))
