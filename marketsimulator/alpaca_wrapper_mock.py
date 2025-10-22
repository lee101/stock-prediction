from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Optional
import os

from src.leverage_settings import get_leverage_settings

from .logging_utils import logger

from .state import SimulationState, SimulatedOrder, SimulatedPosition, get_state

equity: float = 100_000.0
cash: float = 100_000.0
total_buying_power: float = 100_000.0
margin_multiplier: float = get_leverage_settings().max_gross_leverage


def _sync_account_metrics(state: Optional[SimulationState] = None) -> None:
    global equity, cash, total_buying_power
    if state is None:
        state = get_state()
    equity = float(state.equity)
    cash = float(state.cash)
    total_buying_power = float(state.buying_power)


@dataclass
class MockAccount:
    cash: float
    equity: float
    buying_power: float
    multiplier: float


@dataclass
class MockOrder:
    id: str
    symbol: str
    qty: float
    side: str
    limit_price: float
    status: str = "open"


class MockClock:
    def __init__(self, state: SimulationState):
        self._state = state

    @property
    def is_open(self) -> bool:
        if os.getenv("MARKETSIM_FORCE_MARKET_OPEN", "0").lower() in {"1", "true", "yes", "on"}:
            return True
        return self._state.clock.is_open

    @property
    def timestamp(self):
        return self._state.clock.current

    @property
    def next_open(self):
        return self._state.clock.next_open

    @property
    def next_close(self):
        return self._state.clock.next_close


def reset_account(initial_cash: float = 100_000.0) -> None:
    global margin_multiplier
    state = get_state()
    settings = get_leverage_settings()
    margin_multiplier = settings.max_gross_leverage
    state.leverage_settings = settings
    state.cash = initial_cash
    state.positions.clear()
    state.open_orders.clear()
    state.take_profit_targets.clear()
    state.fees_paid = 0.0
    state.financing_cost_paid = 0.0
    state.last_financing_timestamp = state.clock.current
    state.update_market_prices()
    _sync_account_metrics(state)


def get_clock() -> MockClock:
    return MockClock(get_state())


def get_account() -> MockAccount:
    state = get_state()
    _sync_account_metrics(state)
    settings = state.leverage_settings or get_leverage_settings()
    current_multiplier = getattr(settings, "max_gross_leverage", margin_multiplier)
    return MockAccount(
        cash=state.cash,
        equity=state.equity,
        buying_power=state.buying_power,
        multiplier=current_multiplier,
    )


def _to_mock_position(position: SimulatedPosition) -> Any:
    class _Mock:
        pass

    mock = _Mock()
    mock.symbol = position.symbol
    mock.qty = f"{position.qty:.6f}"
    mock.side = "long" if position.side == "buy" else "short"
    mock.avg_entry_price = f"{position.avg_entry_price:.4f}"
    mock.market_value = f"{position.market_value:.4f}"
    mock.unrealized_pl = f"{position.unrealized_pl:.4f}"
    mock.current_price = f"{position.current_price:.4f}"
    return mock


def get_all_positions() -> List[Any]:
    state = get_state()
    state.update_market_prices()
    return [_to_mock_position(pos) for pos in state.positions.values()]


def _store_order(order: SimulatedOrder) -> MockOrder:
    state = get_state()
    state.register_order(order)
    return MockOrder(
        id=order.order_id,
        symbol=order.symbol,
        qty=order.qty,
        side=order.side,
        limit_price=order.limit_price,
        status=order.status,
    )


def get_orders() -> List[MockOrder]:
    state = get_state()
    return [
        MockOrder(
            id=order.order_id,
            symbol=order.symbol,
            qty=order.qty,
            side=order.side,
            limit_price=order.limit_price,
            status=order.status,
        )
        for order in state.open_orders.values()
    ]


def get_open_orders() -> List[MockOrder]:
    return get_orders()


def cancel_order(order: MockOrder) -> None:
    state = get_state()
    state.open_orders.pop(order.id, None)


def cancel_all_orders() -> None:
    state = get_state()
    state.open_orders.clear()


def has_current_open_position(symbol: str, side: str) -> bool:
    state = get_state()
    position = state.positions.get(symbol)
    if not position:
        return False
    from src.comparisons import is_same_side

    stored_side = "buy" if position.side == "buy" else "sell"
    return is_same_side(stored_side, side)


def _execute_order(symbol: str, qty: float, side: str, price: float, replace_existing: bool) -> MockOrder:
    state = get_state()
    if replace_existing:
        state.clear_symbol_orders(symbol)
        if has_current_open_position(symbol, side):
            logger.info(f"Skipping {symbol} order because position already matches {side}")
            return MockOrder("noop", symbol, qty, side, price, status="rejected")
    state.ensure_position(symbol, qty, side, price)
    order = SimulatedOrder(
        order_id=state.next_order_id(),
        symbol=symbol,
        qty=qty,
        side=side,
        limit_price=price,
        status="filled",
    )
    state.register_order(order)
    state.fill_order(order.order_id)
    _sync_account_metrics(state)
    return MockOrder(order.order_id, symbol, qty, side, price, status="filled")


def open_order_at_price(symbol: str, qty: float, side: str, price: float) -> MockOrder:
    return _execute_order(symbol, qty, side, price, replace_existing=False)


def open_order_at_price_or_all(symbol: str, qty: float, side: str, price: float) -> MockOrder:
    return _execute_order(symbol, qty, side, price, replace_existing=True)


def open_order_at_price_allow_add_to_position(symbol: str, qty: float, side: str, price: float) -> MockOrder:
    return _execute_order(symbol, qty, side, price, replace_existing=False)


def execute_portfolio_orders(orders: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for order in orders:
        results[order["symbol"]] = open_order_at_price_or_all(
            order["symbol"],
            order["qty"],
            order["side"],
            order["price"],
        )
    return results


def close_position_violently(position: Any) -> Optional[MockOrder]:
    state = get_state()
    qty = float(position.qty)
    side = "sell" if position.side.lower() in {"buy", "long"} else "buy"
    price = state.prices[position.symbol].price("Close")
    state.close_position(position.symbol, price, qty)
    _sync_account_metrics(state)
    return MockOrder(f"CLOSE-{position.symbol}", position.symbol, qty, side, price, status="filled")


def close_position_near_market(position: Any, pct_above_market: float = 0.0) -> Optional[MockOrder]:
    state = get_state()
    price = state.prices[position.symbol].price("Close")
    adjust = price * pct_above_market
    fill_price = price + adjust
    state.close_position(position.symbol, fill_price)
    _sync_account_metrics(state)
    closing_side = "sell" if position.side.lower() in {"buy", "long"} else "buy"
    return MockOrder(
        f"CLOSE-{position.symbol}",
        position.symbol,
        float(position.qty),
        closing_side,
        fill_price,
        status="filled",
    )


def open_take_profit_position(position: Any, row: Dict[str, Any], price: float, qty: float) -> MockOrder:
    state = get_state()
    side = "sell" if position.side.lower() in {"buy", "long"} else "buy"
    state.place_take_profit(position.symbol, side, price, qty)
    return MockOrder(f"TP-{position.symbol}", position.symbol, qty, side, price, status="open")


def latest_data(symbol: str):
    state = get_state()
    series = state.prices.get(symbol)
    if series is None:
        return None

    class _Quote:
        pass

    quote = _Quote()
    quote.symbol = symbol
    quote.timestamp = series.timestamp
    ask = state.current_ask(symbol)
    bid = state.current_bid(symbol)
    close_price = series.price("Close")
    quote.ask_price = ask if ask is not None else close_price
    quote.bid_price = bid if bid is not None else close_price
    return quote


def get_clock_internal(*_, **__) -> MockClock:
    return get_clock()


def force_open_the_clock_func() -> None:
    state = get_state()
    state.clock.advance(timedelta(minutes=0))


def get_all_positions_summary() -> Dict[str, Any]:
    state = get_state()
    return {
        "cash": state.cash,
        "equity": state.equity,
        "positions": {symbol: pos.qty for symbol, pos in state.positions.items()},
    }


def re_setup_vars() -> None:
    _sync_account_metrics()


def close_open_orders() -> None:
    cancel_all_orders()
