from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional

import pandas as pd
import pytz

from stock.data_utils import coerce_numeric
from src.leverage_settings import LeverageSettings, get_leverage_settings
from loss_utils import CRYPTO_TRADING_FEE, TRADING_FEE
from .execution import classify_liquidity, simulate_fill
from src.fixtures import crypto_symbols
from .hourly_utils import load_hourly_bars


def _ensure_timezone(value: datetime) -> datetime:
    if value.tzinfo is None:
        return pytz.utc.localize(value)
    return value.astimezone(pytz.utc)


def _east_coast_time(value: datetime) -> datetime:
    return value.astimezone(pytz.timezone("US/Eastern"))


def _normalise_side(side: str) -> str:
    return "buy" if str(side).lower().startswith("b") else "sell"


def _row_value(row: pd.Series, *names: str, default: float) -> float:
    for name in names:
        if name not in row:
            continue
        raw_value = row[name]
        scalar = coerce_numeric(raw_value, default=default)
        if pd.notna(scalar):
            return float(scalar)
    return coerce_numeric(default, default=default)


def _as_utc_timestamp(value: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


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
        value = self.current_row.get(column)
        if value is None:
            return coerce_numeric(self.current_row.get("Close"), default=0.0)
        return coerce_numeric(value, default=0.0)


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


@dataclass
class MaxDiffEntryWatcher:
    symbol: str
    side: str
    limit_price: float
    target_qty: float
    tolerance_pct: float
    expiry: datetime
    created_at: datetime
    last_checked: datetime
    last_fill: Optional[datetime] = None
    fills: int = 0
    min_interval: timedelta = timedelta(hours=1)
    force_immediate: bool = False
    priority_rank: Optional[int] = None


@dataclass
class MaxDiffExitWatcher:
    symbol: str
    entry_side: str
    takeprofit_price: float
    expiry: datetime
    created_at: datetime
    last_checked: datetime
    tolerance_pct: float = 0.001
    last_fill: Optional[datetime] = None
    fills: int = 0
    min_interval: timedelta = timedelta(hours=1)


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
    peak_equity: float = 100_000.0
    leverage_settings: LeverageSettings = field(default_factory=get_leverage_settings)
    gross_exposure: float = 0.0
    financing_cost_paid: float = 0.0
    last_financing_timestamp: Optional[datetime] = None
    positions: Dict[str, SimulatedPosition] = field(default_factory=dict)
    open_orders: Dict[str, SimulatedOrder] = field(default_factory=dict)
    take_profit_targets: List[TakeProfitTarget] = field(default_factory=list)
    maxdiff_entries: List[MaxDiffEntryWatcher] = field(default_factory=list)
    maxdiff_exits: List[MaxDiffExitWatcher] = field(default_factory=list)
    order_sequence: int = 1
    fees_paid: float = 0.0
    trade_log: List[TradeExecution] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.leverage_settings is None:
            self.leverage_settings = get_leverage_settings()
        if self.last_financing_timestamp is None:
            self.last_financing_timestamp = self.clock.current
        self._recalculate_equity()
        self.peak_equity = max(self.peak_equity, self.equity)

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
        self.peak_equity = max(self.peak_equity, self.equity)
        max_gross_allowed = self.leverage_settings.max_gross_leverage * max(self.equity, 0.0)
        self.buying_power = max(0.0, max_gross_allowed - gross_value)

    def current_bid(self, symbol: str) -> Optional[float]:
        series = self.prices.get(symbol)
        if series is None:
            return None
        close_price = series.price("Close")
        return _row_value(series.current_row, "Low", "low", default=close_price)

    def current_ask(self, symbol: str) -> Optional[float]:
        series = self.prices.get(symbol)
        if series is None:
            return None
        close_price = series.price("Close")
        return _row_value(series.current_row, "High", "high", default=close_price)

    @property
    def drawdown(self) -> float:
        return max(0.0, self.peak_equity - self.equity)

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return self.drawdown / self.peak_equity

    def place_take_profit(self, symbol: str, side: str, price: float, qty: float) -> None:
        self.take_profit_targets.append(TakeProfitTarget(symbol, side, price, qty))

    def register_maxdiff_entry(
        self,
        symbol: str,
        side: str,
        limit_price: float,
        target_qty: float,
        tolerance_pct: float,
        expiry_minutes: int,
        force_immediate: bool = False,
        priority_rank: Optional[int] = None,
    ) -> None:
        if limit_price <= 0 or target_qty <= 0:
            return
        now = self.clock.current
        expiry_minutes = max(1, int(expiry_minutes))
        priority_value: Optional[int] = None
        if priority_rank is not None:
            try:
                priority_value = int(priority_rank)
            except (TypeError, ValueError):
                priority_value = None

        watcher = MaxDiffEntryWatcher(
            symbol=symbol.upper(),
            side=_normalise_side(side),
            limit_price=float(limit_price),
            target_qty=float(abs(target_qty)),
            tolerance_pct=max(0.0, float(tolerance_pct)),
            expiry=now + timedelta(minutes=expiry_minutes),
            created_at=now,
            last_checked=now,
            force_immediate=bool(force_immediate),
            priority_rank=priority_value,
        )
        self.maxdiff_entries.append(watcher)

    def register_maxdiff_exit(
        self,
        symbol: str,
        side: str,
        takeprofit_price: float,
        expiry_minutes: int,
        tolerance_pct: float = 0.001,
    ) -> None:
        if takeprofit_price <= 0:
            return
        now = self.clock.current
        expiry_minutes = max(1, int(expiry_minutes))
        watcher = MaxDiffExitWatcher(
            symbol=symbol.upper(),
            entry_side=_normalise_side(side),
            takeprofit_price=float(takeprofit_price),
            expiry=now + timedelta(minutes=expiry_minutes),
            created_at=now,
            last_checked=now,
            tolerance_pct=max(0.0, float(tolerance_pct)),
        )
        self.maxdiff_exits.append(watcher)

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
        self._process_maxdiff_watchers(previous_time, self.clock.current)
        self._apply_take_profit_targets()

    def _process_maxdiff_watchers(self, start: datetime, end: datetime) -> None:
        if not self.maxdiff_entries and not self.maxdiff_exits:
            return
        if end < start:
            start, end = end, start
        symbols = {w.symbol for w in self.maxdiff_entries} | {w.symbol for w in self.maxdiff_exits}
        if not symbols:
            return
        for symbol in symbols:
            self._process_symbol_watchers(symbol, start, end)
        self._trim_expired_watchers(end)

    def _process_symbol_watchers(self, symbol: str, start: datetime, end: datetime) -> None:
        entries = [w for w in self.maxdiff_entries if w.symbol == symbol]
        exits = [w for w in self.maxdiff_exits if w.symbol == symbol]
        if not entries and not exits:
            return
        earliest = start
        for watcher in entries + exits:
            earliest = min(earliest, watcher.last_checked)
        earliest = max(start, earliest)
        bars = load_hourly_bars(symbol)
        if not bars.empty:
            start_ts = _as_utc_timestamp(earliest)
            end_ts = _as_utc_timestamp(end)
            symbol_bars = bars.loc[(bars["timestamp"] >= start_ts) & (bars["timestamp"] <= end_ts)]
        else:
            symbol_bars = pd.DataFrame()
        if symbol_bars.empty:
            self._process_symbol_bar_fallback(symbol, entries, exits, end)
            for watcher in entries + exits:
                watcher.last_checked = max(watcher.last_checked, end)
            return
        for _, row in symbol_bars.iterrows():
            ts = row["timestamp"]
            if isinstance(ts, pd.Timestamp):
                timestamp = ts.to_pydatetime()
            else:
                timestamp = ts
            if timestamp.tzinfo is None:
                timestamp = pytz.utc.localize(timestamp)
            else:
                timestamp = timestamp.astimezone(pytz.utc)
            self._process_entry_watchers(symbol, entries, row, timestamp)
            self._process_exit_watchers(symbol, exits, row, timestamp)
        last_ts = symbol_bars["timestamp"].max()
        if isinstance(last_ts, pd.Timestamp):
            last_dt = last_ts.to_pydatetime()
        else:
            last_dt = last_ts
        if last_dt.tzinfo is None:
            last_dt = pytz.utc.localize(last_dt)
        else:
            last_dt = last_dt.astimezone(pytz.utc)
        for watcher in entries + exits:
            watcher.last_checked = max(watcher.last_checked, last_dt)
            if watcher.last_checked < end:
                watcher.last_checked = end

    def _process_entry_watchers(
        self,
        symbol: str,
        watchers: List[MaxDiffEntryWatcher],
        row: pd.Series,
        timestamp: datetime,
    ) -> None:
        close = _row_value(row, "Close", "close", default=0.0)
        high = _row_value(row, "High", "high", default=close)
        low = _row_value(row, "Low", "low", default=close)
        for watcher in watchers:
            watcher.last_checked = max(watcher.last_checked, timestamp)
            if timestamp > watcher.expiry:
                continue
            existing = self.positions.get(symbol)
            if existing and existing.side == watcher.side and existing.qty >= watcher.target_qty - 1e-9:
                continue
            limit_price = watcher.limit_price
            tolerance = watcher.tolerance_pct
            if watcher.side == "buy":
                threshold = limit_price * (1.0 + tolerance)
                trigger = low <= threshold
            else:
                threshold = limit_price * (1.0 - tolerance)
                trigger = high >= threshold
            if watcher.last_fill is not None and (timestamp - watcher.last_fill) < watcher.min_interval:
                continue
            if not trigger:
                continue
            qty = watcher.target_qty
            if qty <= 0:
                continue
            self.ensure_position(symbol, qty, watcher.side, limit_price, market_row=row)
            watcher.fills += 1
            watcher.last_fill = timestamp

    def _process_exit_watchers(
        self,
        symbol: str,
        watchers: List[MaxDiffExitWatcher],
        row: pd.Series,
        timestamp: datetime,
    ) -> None:
        close = _row_value(row, "Close", "close", default=0.0)
        high = _row_value(row, "High", "high", default=close)
        low = _row_value(row, "Low", "low", default=close)
        for watcher in watchers:
            watcher.last_checked = max(watcher.last_checked, timestamp)
            if timestamp > watcher.expiry:
                continue
            position = self.positions.get(symbol)
            if not position or position.side != watcher.entry_side:
                continue
            qty = abs(position.qty)
            if qty <= 0:
                continue
            if watcher.last_fill is not None and (timestamp - watcher.last_fill) < watcher.min_interval:
                continue
            tolerance = watcher.tolerance_pct
            target = watcher.takeprofit_price
            if watcher.entry_side == "buy":
                trigger = high >= target * (1.0 - tolerance)
            else:
                trigger = low <= target * (1.0 + tolerance)
            if not trigger:
                continue
            self.close_position(symbol, target, qty, market_row=row)
            watcher.fills += 1
            watcher.last_fill = timestamp

    def _process_symbol_bar_fallback(
        self,
        symbol: str,
        entries: List[MaxDiffEntryWatcher],
        exits: List[MaxDiffExitWatcher],
        reference_time: datetime,
    ) -> None:
        series = self.prices.get(symbol)
        if series is None:
            return
        row = series.current_row
        close = _row_value(row, "Close", "close", default=0.0)
        fallback = pd.Series(
            {
                "Open": _row_value(row, "Open", "open", default=close),
                "High": _row_value(row, "High", "high", default=close),
                "Low": _row_value(row, "Low", "low", default=close),
                "Close": close,
            }
        )
        self._process_entry_watchers(symbol, entries, fallback, reference_time)
        self._process_exit_watchers(symbol, exits, fallback, reference_time)

    def _trim_expired_watchers(self, reference: datetime) -> None:
        self.maxdiff_entries = [w for w in self.maxdiff_entries if w.expiry > reference]
        self.maxdiff_exits = [w for w in self.maxdiff_exits if w.expiry > reference]

    def _apply_take_profit_targets(self) -> None:
        remaining: List[TakeProfitTarget] = []
        for target in self.take_profit_targets:
            series = self.prices.get(target.symbol)
            if series is None:
                continue
            close_price = series.price("Close")
            last_high = _row_value(series.current_row, "High", "high", default=close_price)
            last_low = _row_value(series.current_row, "Low", "low", default=close_price)
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

    def close_position(
        self,
        symbol: str,
        price: Optional[float] = None,
        qty: Optional[float] = None,
        market_row: Optional[pd.Series] = None,
    ) -> None:
        position = self.positions.get(symbol)
        if not position:
            return
        fill_qty = qty if qty is not None else position.qty
        price = price if price is not None else position.current_price
        fill_qty = min(fill_qty, position.qty)
        trade_side = "sell" if position.side == "buy" else "buy"
        self._apply_trade_cash(symbol, trade_side, price, fill_qty, market_row)
        if fill_qty >= position.qty:
            self.positions.pop(symbol, None)
        else:
            position.qty -= fill_qty
        self.update_market_prices()

    def ensure_position(
        self,
        symbol: str,
        qty: float,
        side: str,
        price: float,
        market_row: Optional[pd.Series] = None,
    ) -> None:
        position = self.positions.get(symbol)
        if position is None:
            self.positions[symbol] = SimulatedPosition(
                symbol=symbol,
                qty=qty,
                side=side,
                avg_entry_price=price,
                current_price=price,
            )
            self._apply_trade_cash(symbol, side, price, qty, market_row)
            self.update_market_prices()
            return

        if position.side == side:
            self._apply_trade_cash(symbol, side, price, qty, market_row)
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
                self._apply_trade_cash(symbol, side, price, qty, market_row)
                position.qty -= qty
                self.update_market_prices()
                return
            elif qty == position.qty:
                self._apply_trade_cash(symbol, side, price, qty, market_row)
                self.positions.pop(symbol, None)
                self.update_market_prices()
                return
            else:
                self._apply_trade_cash(symbol, side, price, position.qty, market_row)
                remainder = qty - position.qty
                self.positions.pop(symbol, None)
                self.ensure_position(symbol, remainder, side="buy", price=price, market_row=market_row)
                return
        else:
            # sell order closes part of a long
            if qty < position.qty:
                self._apply_trade_cash(symbol, side, price, qty, market_row)
                position.qty -= qty
                self.update_market_prices()
                return
            elif qty == position.qty:
                self._apply_trade_cash(symbol, side, price, qty, market_row)
                self.positions.pop(symbol, None)
                self.update_market_prices()
                return
            else:
                self._apply_trade_cash(symbol, side, price, position.qty, market_row)
                remainder = qty - position.qty
                self.positions.pop(symbol, None)
                self.ensure_position(symbol, remainder, side="sell", price=price, market_row=market_row)
                return

    def symbols(self) -> Iterable[str]:
        return self.prices.keys()

    def _apply_trade_cash(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        market_row: Optional[pd.Series] = None,
    ) -> None:
        if qty <= 0:
            return
        intended_price = price
        series = self.prices.get(symbol)
        mid_price = intended_price
        vol_bps = 0.0
        reference_row: Optional[pd.Series] = market_row
        if reference_row is None and series is not None:
            reference_row = series.current_row
        if reference_row is not None:
            high = _row_value(reference_row, "High", "high", default=price)
            low = _row_value(reference_row, "Low", "low", default=price)
        else:
            high = low = price
        mid_price = max(1e-9, (high + low) / 2.0)
        side_lower = side.lower()
        if side_lower == "buy":
            mid_price = max(1e-9, min(mid_price, price))
        else:
            mid_price = max(1e-9, max(mid_price, price))
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
