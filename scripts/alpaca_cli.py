import alpaca_trade_api as tradeapi
import typer
from alpaca.data import StockHistoricalDataClient

import alpaca_wrapper
from data_curate_daily import download_exchange_latest_data, get_bid, get_ask
from env_real import ALP_KEY_ID, ALP_SECRET_KEY, ALP_ENDPOINT, ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD

alpaca_api = tradeapi.REST(
    ALP_KEY_ID,
    ALP_SECRET_KEY,
    ALP_ENDPOINT,
    'v2')


def main(command: str):
    """
    cancel_all_orders - cancel all orders
    close_all_positions - close all positions at near market price
    close_position_violently - close position violently
    :param command:
    :return:
    """
    if command == 'close_all_positions':
        close_all_positions()
    elif command == 'violently_close_all_positions':
        violently_close_all_positions()
    elif command == 'cancel_all_orders':
        alpaca_wrapper.cancel_all_orders()

client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)

def close_all_positions():
    positions = alpaca_wrapper.get_all_positions()

    for position in positions:
        symbol = position.symbol

        # get latest data then bid/ask
        download_exchange_latest_data(client, symbol)
        bid = get_bid(symbol)
        ask = get_ask(symbol)


        current_price = ask if position.side == 'long' else bid
        # close a long with the ask price
        # close a short with the bid price
        # get bid/ask
        # get current price
        alpaca_wrapper.close_position_at_almost_current_price(
            position, {
                'close_last_price_minute': current_price
            }
        )
            # alpaca_order_stock(position.symbol, position.qty)


def violently_close_all_positions():
    positions = alpaca_wrapper.get_all_positions()
    for position in positions:
        alpaca_wrapper.close_position_violently(position)


if __name__ == "__main__":
    # typer.run(main)
    close_all_positions()
