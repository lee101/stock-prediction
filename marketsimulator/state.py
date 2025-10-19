from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional

import pandas as pd
import pytz

from src.leverage_settings import LeverageSettings, get_leverage_settings
from loss_utils import CRYPTO_TRADING_FEE, TRADING_FEE
from .execution import classify_liquidity, simulate_fill
from src.fixtures import crypto_symbols


def _ensure_timezone(value: datetime) -> datetime:
    if value.tzinfo is None:
        return pytz.utc.localize(value)
    return value.astimezone(pytz.utc)


def _east_coast_time(value: datetime) -> datetime:
    return value.astimezone(pytz.timezone("US/Eastern"))


@dataclass
class PriceSeries:
    symbol: str
    frame: pd.DataFrame
    cursor: int = 0

    def __post_init__(self) -> None:
        if "timestamp" not in self.frame.columns:
            raise ValueError(f"Price series for {self.symbol} requires a timestamp column")
        self.frame.sort_values("timestamp", inplace=True)
        self.frame.reset_index(drop=True, inplace=True)

    @property
    def current_row(self) -> pd.Series:
        return self.frame.iloc[self.cursor]

    @property
    def timestamp(self) -> datetime:
        raw = self.current_row["timestamp"]
        if isinstance(raw, str):
            raw = datetime.fromisoformat(raw)
        return _ensure_timezone(raw)

    def advance(self, steps: int = 1) -> None:
        self.cursor = min(self.cursor + steps, len(self.frame) - 1)

    def has_next(self) -> bool:
        return self.cursor < len(self.frame) - 1

    def price(self, column: str = "Close") -> float:
        return float(self.current_row[column])


@dataclass
class SimulatedPosition:
    symbol: str
    qty: float
    side: str  # "buy" for long, "sell" for short
    avg_entry_price: float
    current_price: float

    def update_price(self, price: float) -> None:
        self.current_price = price

    @property
    def market_value(self) -> float:
        value = self.current_price * abs(self.qty)
        return value if self.side == "buy" else -value

    @property
    def unrealized_pl(self) -> float:
        delta = self.current_price - self.avg_entry_price
        multiplier = 1 if self.side == "buy" else -1
        return delta * self.qty * multiplier


@dataclass
class SimulatedOrder:
    order_id: str
    symbol: str
    qty: float
    side: str
    limit_price: float
    order_type: str = "limit"
    status: str = "open"
    tag: Optional[str] = None


@dataclass
class TakeProfitTarget:
    symbol: str
    side: str
    price: float
    qty: float


@dataclass
class TradeExecution:
    timestamp: datetime
    symbol: str
    side: str
    price: float
    qty: float
    notional: float
    fee: float
    cash_delta: float
    slip_bps: float = 0.0


class SimulatedClock:
    def __init__(self, now: datetime):
        self.current = _ensure_timezone(now)

    def set(self, new_time: datetime) -> None:
        self.current = _ensure_timezone(new_time)

    def advance(self, delta: timedelta) -> None:
        self.current += delta

    @property
    def is_open(self) -> bool:
        local = _east_coast_time(self.current)
        if local.weekday() >= 5:
            return False
        open_time = local.replace(hour=9, minute=30, second=0, microsecond=0)
        close_time = local.replace(hour=16, minute=0, second=0, microsecond=0)
        return open_time <= local <= close_time

    @property
    def next_open(self) -> datetime:
        local = _east_coast_time(self.current)
        next_day = local
        while True:
            next_day += timedelta(days=1)
            if next_day.weekday() < 5:
                break
        open_time = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
        return open_time.astimezone(pytz.utc)

    @property
    def next_close(self) -> datetime:
        local = _east_coast_time(self.current)
        close_time = local.replace(hour=16, minute=0, second=0, microsecond=0)
        if local > close_time:
            close_time = self.next_open.astimezone(pytz.timezone("US/Eastern")).replace(
                hour=16, minute=0, second=0, microsecond=0
            )
        return close_time.astimezone(pytz.utc)


@dataclass
class SimulationState:
    clock: SimulatedClock
    prices: Dict[str, PriceSeries]
    cash: float = 100_000.0
    buying_power: float = 0.0
    equity: float = 100_000.0
    leverage_settings: LeverageSettings = field(default_factory=get_leverage_settings)
    gross_exposure: float = 0.0
    financing_cost_paid: float = 0.0
    last_financing_timestamp: Optional[datetime] = None
    positions: Dict[str, SimulatedPosition] = field(default_factory=dict)
    open_orders: Dict[str, SimulatedOrder] = field(default_factory=dict)
    take_profit_targets: List[TakeProfitTarget] = field(default_factory=list)
    order_sequence: int = 1
    fees_paid: float = 0.0
    trade_log: List[TradeExecution] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.leverage_settings is None:
            self.leverage_settings = get_leverage_settings()
        if self.last_financing_timestamp is None:
            self.last_financing_timestamp = self.clock.current
        self._recalculate_equity()

    def _gross_exposure(self) -> float:
        return sum(abs(pos.current_price * pos.qty) for pos in self.positions.values())

    def _accrue_financing_cost(self, delta_seconds: float, gross_exposure: Optional[float] = None) -> None:
        if delta_seconds <= 0:
            self.last_financing_timestamp = self.clock.current
            return
        gross = self._gross_exposure() if gross_exposure is None else gross_exposure
        if gross <= 0:
            self.last_financing_timestamp = self.clock.current
            return
        base_equity = max(self.equity, 0.0)
        borrow_notional = max(0.0, gross - base_equity)
        if borrow_notional <= 0:
            self.last_financing_timestamp = self.clock.current
            return
        day_fraction = delta_seconds / 86400.0
        daily_rate = self.leverage_settings.daily_cost
        cost = borrow_notional * daily_rate * day_fraction
        if cost <= 0:
            self.last_financing_timestamp = self.clock.current
            return
        self.cash -= cost
        self.fees_paid += cost
        self.financing_cost_paid += cost
        self.last_financing_timestamp = self.clock.current

    def update_market_prices(self) -> None:
        for symbol, position in self.positions.items():
            series = self.prices.get(symbol)
            if series is None:
                continue
            position.update_price(series.price("Close"))
        self._recalculate_equity()

    def _recalculate_equity(self) -> None:
        net_position_value = sum(pos.market_value for pos in self.positions.values())
        gross_value = self._gross_exposure()
        self.gross_exposure = gross_value
        self.equity = self.cash + net_position_value
        max_gross_allowed = self.leverage_settings.max_gross_leverage * max(self.equity, 0.0)
        self.buying_power = max(0.0, max_gross_allowed - gross_value)

    def current_bid(self, symbol: str) -> Optional[float]:
        series = self.prices.get(symbol)
        if series is None:
            return None
        return float(series.current_row.get("Low", series.price("Close")))

    def current_ask(self, symbol: str) -> Optional[float]:
        series = self.prices.get(symbol)
        if series is None:
            return None
        return float(series.current_row.get("High", series.price("Close")))

    def place_take_profit(self, symbol: str, side: str, price: float, qty: float) -> None:
        self.take_profit_targets.append(TakeProfitTarget(symbol, side, price, qty))

    def next_order_id(self) -> str:
        order_id = f"SIM-{self.order_sequence}"
        self.order_sequence += 1
        return order_id

    def register_order(self, order: SimulatedOrder) -> None:
        self.open_orders[order.order_id] = order

    def fill_order(self, order_id: str) -> None:
        order = self.open_orders.get(order_id)
        if not order:
            return
        order.status = "filled"
        self.open_orders.pop(order_id, None)

    def clear_symbol_orders(self, symbol: str) -> None:
        to_remove = [oid for oid, order in self.open_orders.items() if order.symbol == symbol]
        for oid in to_remove:
            self.open_orders.pop(oid, None)

    def advance_time(self, steps: int = 1) -> None:
        previous_time = self.clock.current
        previous_gross = self._gross_exposure()
        for series in self.prices.values():
            series.advance(steps)
        timestamps = [series.timestamp for series in self.prices.values()]
        if timestamps:
            self.clock.set(min(timestamps))
        delta_seconds = max(0.0, (self.clock.current - previous_time).total_seconds())
        self._accrue_financing_cost(delta_seconds, gross_exposure=previous_gross)
        self.update_market_prices()
        self._apply_take_profit_targets()

    def _apply_take_profit_targets(self) -> None:
        remaining: List[TakeProfitTarget] = []
        for target in self.take_profit_targets:
            series = self.prices.get(target.symbol)
            if series is None:
                continue
            last_high = float(series.current_row.get("High", series.price("Close")))
            last_low = float(series.current_row.get("Low", series.price("Close")))
            met = False
            if target.side == "sell":
                met = last_high >= target.price
            else:
                met = last_low <= target.price
            if met:
                if target.symbol in self.positions:
                    self.close_position(target.symbol, target.price, target.qty)
            else:
                remaining.append(target)
        self.take_profit_targets = remaining

    def close_position(self, symbol: str, price: Optional[float] = None, qty: Optional[float] = None) -> None:
        position = self.positions.get(symbol)
        if not position:
            return
        fill_qty = qty if qty is not None else position.qty
        price = price if price is not None else position.current_price
        fill_qty = min(fill_qty, position.qty)
        trade_side = "sell" if position.side == "buy" else "buy"
        self._apply_trade_cash(symbol, trade_side, price, fill_qty)
        if fill_qty >= position.qty:
            self.positions.pop(symbol, None)
        else:
            position.qty -= fill_qty
        self.update_market_prices()

    def ensure_position(self, symbol: str, qty: float, side: str, price: float) -> None:
        position = self.positions.get(symbol)
        if position is None:
            self.positions[symbol] = SimulatedPosition(
                symbol=symbol,
                qty=qty,
                side=side,
                avg_entry_price=price,
                current_price=price,
            )
            self._apply_trade_cash(symbol, side, price, qty)
            self.update_market_prices()
            return

        if position.side == side:
            self._apply_trade_cash(symbol, side, price, qty)
            total_qty = position.qty + qty
            position.avg_entry_price = (
                (position.avg_entry_price * position.qty) + (price * qty)
            ) / total_qty
            position.qty = total_qty
            self.update_market_prices()
            return

        # Opposite side order reduces or flips the existing position
        if side == "buy":
            # buy order closes part of a short
            if qty < position.qty:
                self._apply_trade_cash(symbol, side, price, qty)
                position.qty -= qty
                self.update_market_prices()
                return
            elif qty == position.qty:
                self._apply_trade_cash(symbol, side, price, qty)
                self.positions.pop(symbol, None)
                self.update_market_prices()
                return
            else:
                self._apply_trade_cash(symbol, side, price, position.qty)
                remainder = qty - position.qty
                self.positions.pop(symbol, None)
                self.ensure_position(symbol, remainder, side="buy", price=price)
                return
        else:
            # sell order closes part of a long
            if qty < position.qty:
                self._apply_trade_cash(symbol, side, price, qty)
                position.qty -= qty
                self.update_market_prices()
                return
            elif qty == position.qty:
                self._apply_trade_cash(symbol, side, price, qty)
                self.positions.pop(symbol, None)
                self.update_market_prices()
                return
            else:
                self._apply_trade_cash(symbol, side, price, position.qty)
                remainder = qty - position.qty
                self.positions.pop(symbol, None)
                self.ensure_position(symbol, remainder, side="sell", price=price)
                return

    def symbols(self) -> Iterable[str]:
        return self.prices.keys()

    def _apply_trade_cash(self, symbol: str, side: str, price: float, qty: float) -> None:
        if qty <= 0:
            return
        intended_price = price
        series = self.prices.get(symbol)
        mid_price = intended_price
        vol_bps = 0.0
        if series is not None:
            current_row = series.current_row
            high = float(current_row.get("High", series.price("Close")))
            low = float(current_row.get("Low", series.price("Close")))
            mid_price = max(1e-9, (high + low) / 2.0)
            if mid_price > 0:
                vol_bps = abs(high - low) / mid_price * 1e4
        liquidity_tier = classify_liquidity(symbol)
        intended_notional = intended_price * qty
        executed_price, slip_bps = simulate_fill(side, intended_price, mid_price, vol_bps, intended_notional, liquidity_tier)
        price = executed_price
        notional = price * qty
        rate = CRYPTO_TRADING_FEE if symbol.upper() in crypto_symbols else TRADING_FEE
        fee = notional * rate
        cash_delta: float
        if side == "buy":
            cash_delta = -(notional + fee)
        else:
            cash_delta = notional - fee
        self.cash += cash_delta
        self.fees_paid += fee
        self.trade_log.append(
            TradeExecution(
                timestamp=self.clock.current,
                symbol=symbol,
                side=side,
                price=price,
                qty=qty,
                notional=notional,
                fee=fee,
                cash_delta=cash_delta,
                slip_bps=slip_bps,
            )
        )


SIMULATION_STATE: Optional[SimulationState] = None


def set_state(state: SimulationState) -> None:
    global SIMULATION_STATE
    SIMULATION_STATE = state


def get_state() -> SimulationState:
    if SIMULATION_STATE is None:
        raise RuntimeError("Simulation state has not been initialised")
    return SIMULATION_STATE
