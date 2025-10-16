from __future__ import annotations

from typing import Optional

from .logging_utils import logger

from . import alpaca_wrapper_mock as alpaca_wrapper


def backout_near_market(symbol: str):
    positions = alpaca_wrapper.get_all_positions()
    for pos in positions:
        if pos.symbol == symbol:
            logger.info(f"[sim] Closing position for {symbol}")
            alpaca_wrapper.close_position_violently(pos)
            break


def ramp_into_position(symbol: str, side: str = "buy", target_qty: Optional[float] = None):
    quote = alpaca_wrapper.latest_data(symbol)
    price = quote.ask_price if side == "buy" else quote.bid_price
    qty = target_qty if target_qty is not None else 1.0
    logger.info(f"[sim] Entering {side} position for {symbol} qty={qty} @ {price}")
    alpaca_wrapper.open_order_at_price_allow_add_to_position(symbol, qty, side, price)


def spawn_close_position_at_takeprofit(symbol: str, takeprofit_price: float):
    positions = alpaca_wrapper.get_all_positions()
    for pos in positions:
        if pos.symbol == symbol:
            logger.info(f"[sim] Scheduling takeprofit for {symbol} at {takeprofit_price}")
            alpaca_wrapper.open_take_profit_position(pos, {}, takeprofit_price, float(pos.qty))
            break
