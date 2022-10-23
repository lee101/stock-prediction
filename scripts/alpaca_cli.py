import alpaca_trade_api as tradeapi
import typer

import alpaca_wrapper
from env_real import ALP_KEY_ID, ALP_SECRET_KEY, ALP_ENDPOINT

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


def close_all_positions():
    positions = alpaca_wrapper.get_all_positions()

    for position in positions:
        if position.side == 'long':
            alpaca_wrapper.buy_stock(position.symbol, position.qty)


def violently_close_all_positions():
    positions = alpaca_wrapper.get_all_positions()
    for position in positions:
        alpaca_wrapper.close_position_violently(position)


if __name__ == "__main__":
    typer.run(main)
