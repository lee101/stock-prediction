from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.allocation_utils import allocation_usd_for_symbol
from src.fees import get_fee_for_symbol
from src.hourly_trader_utils import (
    build_order_intents,
    build_plan_from_action,
    ensure_valid_levels,
    infer_working_order_kind,
)
from src.metrics_utils import annualized_sortino, compute_step_returns
from src.market_sim_early_exit import evaluate_drawdown_vs_profit_early_exit, print_early_exit
from src.symbol_utils import is_crypto_symbol
from src.trade_directions import resolve_trade_directions

from .simulator import _infer_periods_per_year, _market_open_mask, _to_new_york


@dataclass
class HourlyTraderSimulationConfig:
    initial_cash: float = 10_000.0
    initial_positions: Optional[Dict[str, float]] = None
    initial_open_orders: Optional[Sequence["OpenOrder"]] = None
    allocation_usd: Optional[float] = None  # per symbol
    allocation_pct: Optional[float] = 0.05
    allocation_mode: str = "per_symbol"  # "per_symbol" | "portfolio"
    max_leverage: float = 1.0
    intensity_scale: float = 1.0
    price_offset_pct: float = 0.0
    min_gap_pct: float = 0.001
    # Realism: require bar to trade through limit by this many bps before fill.
    # Example: 5 bps means a buy at 100 fills only if low <= 99.95.
    fill_buffer_bps: float = 0.0
    # Execution realism: lag between decision bar close and earliest fill eligibility.
    # For live hourly trading, 1 is appropriate (decide on bar t, fill on bar t+1).
    decision_lag_bars: int = 1
    # Broker realism: same-side cancel/replace is not instantaneous on Alpaca.
    # Keep the old order working through this many additional bars before it is
    # considered fully canceled and replacement can proceed.
    cancel_ack_delay_bars: int = 1
    # When a limit is only lightly touched intrabar, treat the fill as partial
    # unless the bar opens/closes through the limit.
    partial_fill_on_touch: bool = False
    enforce_market_hours: bool = True
    fee_by_symbol: Optional[Dict[str, float]] = None
    periods_per_year_by_symbol: Optional[Dict[str, float]] = None
    allow_short: bool = False
    long_only_symbols: Optional[Sequence[str]] = None
    short_only_symbols: Optional[Sequence[str]] = None
    exit_only_symbols: Optional[Sequence[str]] = None
    allow_position_adds: bool = False
    always_full_exit: bool = True
    symbols: Optional[Sequence[str]] = None

    # Broker constraints (approximate).
    min_notional_crypto: float = 10.0
    min_notional_stock: float = 1.0
    cash_buffer: float = 0.99

    # When True, keep an existing same-side order if it is "similar enough" to avoid
    # unrealistically canceling and freeing reserved cash every bar.
    keep_similar_orders: bool = True
    price_tol_pct: float = 0.0003  # ~3 bps
    qty_tol_pct: float = 0.05
    qty_tol_notional_usd: float = 100.0

    # Separate cash pools for stock vs crypto (unified orchestrator mode).
    # When enabled, stock entries draw from cash_stock and crypto entries from cash_crypto.
    separate_cash_pools: bool = False
    initial_cash_stock: float = 10_000.0
    initial_cash_crypto: float = 5_000.0

    # Backout simulation: minutes before NYSE open to close crypto positions
    # with edge below backout_edge_threshold. 0 = disabled.
    backout_near_market_minutes: int = 0
    backout_edge_threshold: float = 0.005
    max_hold_hours: Optional[int] = None
    trailing_stop_pct: Optional[float] = None
    force_exit_offset_pct: float = 0.1


@dataclass(frozen=True)
class OpenOrder:
    symbol: str
    side: str  # "buy" or "sell"
    qty: float
    limit_price: float
    kind: str  # "entry" or "exit"
    placed_at: pd.Timestamp
    reserved_cash: float = 0.0  # reserved entry notional / buying power
    cancel_requested_at: Optional[pd.Timestamp] = None
    cancel_effective_at: Optional[pd.Timestamp] = None


@dataclass(frozen=True)
class FillRecord:
    timestamp: pd.Timestamp
    symbol: str
    side: str
    price: float
    quantity: float
    fee_paid: float
    cash_after: float
    reserved_cash_after: float
    position_after: float
    kind: str


@dataclass
class HourlyTraderSimulationResult:
    equity_curve: pd.Series
    per_hour: pd.DataFrame
    fills: List[FillRecord]
    final_cash: float
    final_reserved_cash: float
    final_positions: Dict[str, float]
    open_orders: List[OpenOrder]
    metrics: Dict[str, float] = field(default_factory=dict)


class HourlyTraderMarketSimulator:
    """Simulate the live hourly Alpaca trader across multiple symbols with shared cash.

    The simulator:
    - Places limit orders using action rows at timestamp t.
    - Orders become eligible to fill starting at t + decision_lag_bars hours.
    - Open buy orders reserve USD cash immediately (approximating broker behavior).
    - At most one buy and one sell order per symbol are maintained (same-side orders replace).
    """

    def __init__(self, config: Optional[HourlyTraderSimulationConfig] = None) -> None:
        self.config = config or HourlyTraderSimulationConfig()

    def run(self, bars: pd.DataFrame, actions: pd.DataFrame) -> HourlyTraderSimulationResult:
        cfg = self.config
        decision_lag_bars = int(getattr(cfg, "decision_lag_bars", 1) or 0)
        if decision_lag_bars < 0:
            raise ValueError(f"decision_lag_bars must be >= 0, got {cfg.decision_lag_bars}.")
        cancel_ack_delay_bars = int(getattr(cfg, "cancel_ack_delay_bars", 1) or 0)
        if cancel_ack_delay_bars < 0:
            raise ValueError(f"cancel_ack_delay_bars must be >= 0, got {cfg.cancel_ack_delay_bars}.")
        fill_buffer_bps = float(getattr(cfg, "fill_buffer_bps", 0.0) or 0.0)
        if not math.isfinite(fill_buffer_bps) or fill_buffer_bps < 0.0:
            raise ValueError(f"fill_buffer_bps must be finite and >= 0, got {cfg.fill_buffer_bps}.")
        fill_buffer = fill_buffer_bps / 10_000.0
        max_hold_hours = getattr(cfg, "max_hold_hours", None)
        if max_hold_hours is not None:
            max_hold_hours = int(max_hold_hours)
            if max_hold_hours < 0:
                raise ValueError(f"max_hold_hours must be >= 0, got {cfg.max_hold_hours}.")
        trailing_stop_pct = getattr(cfg, "trailing_stop_pct", None)
        if trailing_stop_pct is not None:
            trailing_stop_pct = float(trailing_stop_pct)
            if not math.isfinite(trailing_stop_pct) or trailing_stop_pct < 0.0:
                raise ValueError(f"trailing_stop_pct must be finite and >= 0, got {cfg.trailing_stop_pct}.")
        force_exit_offset_pct = float(getattr(cfg, "force_exit_offset_pct", 0.1) or 0.0)
        if not math.isfinite(force_exit_offset_pct) or force_exit_offset_pct < 0.0:
            raise ValueError(f"force_exit_offset_pct must be finite and >= 0, got {cfg.force_exit_offset_pct}.")
        max_leverage = float(getattr(cfg, "max_leverage", 1.0) or 1.0)
        if not math.isfinite(max_leverage) or max_leverage <= 0.0:
            raise ValueError(f"max_leverage must be finite and > 0, got {cfg.max_leverage}.")
        cash_buffer = float(getattr(cfg, "cash_buffer", 0.99) or 0.99)
        if not math.isfinite(cash_buffer) or cash_buffer <= 0.0 or cash_buffer > 1.0:
            raise ValueError(f"cash_buffer must be in (0, 1], got {cfg.cash_buffer}.")

        bars = bars.copy()
        actions = actions.copy()
        for df, name in ((bars, "bars"), (actions, "actions")):
            if "timestamp" not in df.columns:
                raise ValueError(f"{name} must include a timestamp column.")
            if "symbol" not in df.columns:
                raise ValueError(f"{name} must include a symbol column.")
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["symbol"] = df["symbol"].astype(str).str.upper()

        required_bar_cols = {"high", "low", "close"}
        missing_bar = required_bar_cols - set(bars.columns)
        if missing_bar:
            raise ValueError(f"bars missing required columns: {sorted(missing_bar)}")

        required_action_cols = {"buy_price", "sell_price", "buy_amount", "sell_amount"}
        missing_actions = required_action_cols - set(actions.columns)
        if missing_actions:
            raise ValueError(f"actions missing required columns: {sorted(missing_actions)}")

        symbols = sorted(set(bars["symbol"]).intersection(actions["symbol"]))
        if cfg.symbols:
            allowed = {s.upper() for s in cfg.symbols}
            symbols = [s for s in symbols if s in allowed]
        if not symbols:
            raise ValueError("No overlapping symbols between bars and actions.")

        bars = bars[bars["symbol"].isin(symbols)].copy()
        actions = actions[actions["symbol"].isin(symbols)].copy()

        # Merge action columns into the bar frame for convenience. Missing actions become NaN and
        # are treated as "no new orders" for that timestamp.
        frame = bars.merge(
            actions[["timestamp", "symbol", "buy_price", "sell_price", "buy_amount", "sell_amount"]],
            on=["timestamp", "symbol"],
            how="left",
        )
        frame = frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

        # Build per-symbol meta.
        fee_map = cfg.fee_by_symbol or {}
        periods_map = cfg.periods_per_year_by_symbol or {}
        meta: Dict[str, Dict[str, float | str | bool]] = {}
        for sym in symbols:
            asset_class = "crypto" if is_crypto_symbol(sym) else "stock"
            maker_fee = float(fee_map.get(sym, get_fee_for_symbol(sym)))
            if sym in periods_map:
                periods_per_year = float(periods_map[sym])
            else:
                sym_bars = frame.loc[frame["symbol"] == sym, "timestamp"]
                periods_per_year = _infer_periods_per_year(sym_bars, asset_class)
            directions = resolve_trade_directions(
                sym,
                allow_short=bool(cfg.allow_short),
                long_only_symbols=cfg.long_only_symbols,
                short_only_symbols=cfg.short_only_symbols,
                use_default_groups=True,
            )
            meta[sym] = {
                "asset_class": asset_class,
                "maker_fee": maker_fee,
                "periods_per_year": periods_per_year,
                "can_long": bool(directions.can_long),
                "can_short": bool(directions.can_short),
            }

        # Precompute stock market-open mask per symbol (for placement only).
        stock_open_mask: Dict[Tuple[pd.Timestamp, str], bool] = {}
        if cfg.enforce_market_hours:
            for sym in symbols:
                if meta[sym]["asset_class"] != "stock":
                    continue
                sym_ts = frame.loc[frame["symbol"] == sym, "timestamp"]
                ny_ts = _to_new_york(sym_ts)
                mask = _market_open_mask(ny_ts)
                for ts, is_open in zip(sym_ts, mask):
                    stock_open_mask[(pd.Timestamp(ts), sym)] = bool(is_open)

        exit_only_set = {s.upper() for s in (cfg.exit_only_symbols or ())}

        def _normalize_symbol(value: object) -> str:
            return str(value or "").replace("/", "").replace("-", "").upper()

        def _optional_ts(value: object) -> Optional[pd.Timestamp]:
            ts = pd.to_datetime(value, utc=True, errors="coerce")
            if pd.isna(ts):
                return None
            return pd.Timestamp(ts)

        # Cash management: single pool or separate stock/crypto pools.
        if cfg.separate_cash_pools:
            cash_stock = float(cfg.initial_cash_stock)
            cash_crypto = float(cfg.initial_cash_crypto)
            cash = cash_stock + cash_crypto  # Total for equity calc
        else:
            cash = float(cfg.initial_cash)
            cash_stock = cash
            cash_crypto = cash
        positions: Dict[str, float] = {sym: 0.0 for sym in symbols}
        for raw_symbol, raw_qty in (cfg.initial_positions or {}).items():
            symbol = _normalize_symbol(raw_symbol)
            if symbol not in positions:
                continue
            positions[symbol] = float(raw_qty)
        position_entry_times: Dict[str, pd.Timestamp] = {}
        position_peak_closes: Dict[str, float] = {}
        initial_ts = pd.Timestamp(frame["timestamp"].min()) if not frame.empty else None
        if initial_ts is not None:
            for sym, qty in positions.items():
                if float(qty) > 0.0:
                    position_entry_times[sym] = initial_ts

        open_orders: Dict[Tuple[str, str], OpenOrder] = {}
        for seed_order in (cfg.initial_open_orders or ()):
            symbol = _normalize_symbol(getattr(seed_order, "symbol", ""))
            side = str(getattr(seed_order, "side", "") or "").strip().lower()
            qty = float(getattr(seed_order, "qty", 0.0) or 0.0)
            price = float(getattr(seed_order, "limit_price", 0.0) or 0.0)
            if symbol not in positions or side not in {"buy", "sell"} or qty <= 0.0 or price <= 0.0:
                continue

            placed_at = _optional_ts(getattr(seed_order, "placed_at", None))
            if placed_at is None:
                continue

            kind = str(getattr(seed_order, "kind", "") or "").strip().lower()
            if kind not in {"entry", "exit"}:
                kind = infer_working_order_kind(side=side, position_qty=float(positions.get(symbol, 0.0)))

            fee_rate = float(meta[symbol]["maker_fee"])
            reserved_cash_value = float(getattr(seed_order, "reserved_cash", 0.0) or 0.0)
            if reserved_cash_value < 0.0 or not math.isfinite(reserved_cash_value):
                reserved_cash_value = 0.0

            normalized = OpenOrder(
                symbol=symbol,
                side=side,
                qty=qty,
                limit_price=price,
                kind=kind,
                placed_at=placed_at,
                reserved_cash=reserved_cash_value,
                cancel_requested_at=_optional_ts(getattr(seed_order, "cancel_requested_at", None)),
                cancel_effective_at=_optional_ts(getattr(seed_order, "cancel_effective_at", None)),
            )
            key = (symbol, side)
            existing = open_orders.get(key)
            if existing is None:
                open_orders[key] = normalized
                continue

            merged_qty = float(existing.qty) + float(normalized.qty)
            if merged_qty <= 0.0:
                continue
            merged_price = (
                float(existing.limit_price) * float(existing.qty) + float(normalized.limit_price) * float(normalized.qty)
            ) / merged_qty
            merged_kind = existing.kind if existing.kind == normalized.kind else infer_working_order_kind(
                side=side,
                position_qty=float(positions.get(symbol, 0.0)),
            )
            cancel_requested_at = existing.cancel_requested_at
            new_cancel_requested_at = normalized.cancel_requested_at
            if cancel_requested_at is None or (
                new_cancel_requested_at is not None and pd.Timestamp(new_cancel_requested_at) < pd.Timestamp(cancel_requested_at)
            ):
                cancel_requested_at = new_cancel_requested_at
            cancel_effective_at = existing.cancel_effective_at
            new_cancel_effective_at = normalized.cancel_effective_at
            if cancel_effective_at is None or (
                new_cancel_effective_at is not None and pd.Timestamp(new_cancel_effective_at) < pd.Timestamp(cancel_effective_at)
            ):
                cancel_effective_at = new_cancel_effective_at
            open_orders[key] = OpenOrder(
                symbol=symbol,
                side=side,
                qty=merged_qty,
                limit_price=float(merged_price),
                kind=merged_kind,
                placed_at=min(pd.Timestamp(existing.placed_at), pd.Timestamp(normalized.placed_at)),
                reserved_cash=float(existing.reserved_cash) + float(normalized.reserved_cash),
                cancel_requested_at=cancel_requested_at,
                cancel_effective_at=cancel_effective_at,
            )

        reserved_cash = float(sum(float(order.reserved_cash) for order in open_orders.values()))
        last_close: Dict[str, float] = {}
        fills: List[FillRecord] = []
        equity_values: List[float] = []
        per_hour_rows: List[Dict[str, float | str]] = []

        def _min_notional(sym: str) -> float:
            if meta[sym]["asset_class"] == "crypto":
                return float(cfg.min_notional_crypto)
            return float(cfg.min_notional_stock)

        def _equity() -> float:
            return float(cash) + sum(float(qty) * float(last_close.get(sym, 0.0)) for sym, qty in positions.items())

        def _gross_exposure() -> float:
            return sum(abs(float(qty)) * float(last_close.get(sym, 0.0)) for sym, qty in positions.items())

        def _max_entry_capacity(sym: str | None = None) -> float:
            leverage = max_leverage
            if sym is not None and meta[sym]["asset_class"] == "crypto":
                leverage = min(leverage, 1.0)
            return max(0.0, _equity()) * leverage

        def _available_cash(sym: str | None = None) -> float:
            if cfg.separate_cash_pools and sym is not None:
                # In separate pool mode, each asset class has its own cash
                if meta[sym]["asset_class"] == "crypto":
                    crypto_exposure = sum(abs(float(positions.get(s, 0))) * float(last_close.get(s, 0))
                                          for s in symbols if meta[s]["asset_class"] == "crypto")
                    crypto_reserved = sum(float(o.reserved_cash) for (s, _), o in open_orders.items()
                                          if meta.get(s, {}).get("asset_class") == "crypto")
                    return max(0.0, cash_crypto - crypto_exposure - crypto_reserved)
                else:
                    stock_exposure = sum(abs(float(positions.get(s, 0))) * float(last_close.get(s, 0))
                                         for s in symbols if meta[s]["asset_class"] == "stock")
                    stock_reserved = sum(float(o.reserved_cash) for (s, _), o in open_orders.items()
                                         if meta.get(s, {}).get("asset_class") == "stock")
                    return max(0.0, cash_stock * max_leverage - stock_exposure - stock_reserved)
            used = _gross_exposure() + reserved_cash
            return max(0.0, _max_entry_capacity(sym) - used)

        def _is_eligible(order: OpenOrder, ts: pd.Timestamp) -> bool:
            if decision_lag_bars <= 0:
                return True
            return ts >= (order.placed_at + pd.Timedelta(hours=decision_lag_bars))

        def _cancel_is_effective(order: OpenOrder, ts: pd.Timestamp) -> bool:
            if order.cancel_effective_at is None:
                return False
            return ts >= pd.Timestamp(order.cancel_effective_at)

        def _estimate_fill_fraction(
            *,
            order: OpenOrder,
            bar_open: float,
            bar_high: float,
            bar_low: float,
            bar_close: float,
        ) -> float:
            price = float(order.limit_price)
            if price <= 0:
                return 0.0

            if order.side == "buy":
                buy_trigger = max(0.0, price * (1.0 - fill_buffer))
                if bar_low > buy_trigger:
                    return 0.0
                if not bool(getattr(cfg, "partial_fill_on_touch", True)):
                    return 1.0
                if bar_open <= price or bar_close <= price:
                    return 1.0
                cross_amount = max(0.0, price - bar_low)
            else:
                sell_trigger = price * (1.0 + fill_buffer)
                if bar_high < sell_trigger:
                    return 0.0
                if not bool(getattr(cfg, "partial_fill_on_touch", True)):
                    return 1.0
                if bar_open >= price or bar_close >= price:
                    return 1.0
                cross_amount = max(0.0, bar_high - price)

            bar_range = max(bar_high - bar_low, price * 1e-8, 1e-12)
            return min(1.0, max(0.0, cross_amount / bar_range))

        def _fill_order(
            ts: pd.Timestamp,
            bar_open: float,
            bar_high: float,
            bar_low: float,
            bar_close: float,
            order: OpenOrder,
        ) -> None:
            nonlocal cash, reserved_cash, cash_stock, cash_crypto
            sym = order.symbol
            fee_rate = float(meta[sym]["maker_fee"])
            qty = float(order.qty)
            price = float(order.limit_price)
            if qty <= 0 or price <= 0:
                return

            fill_fraction = _estimate_fill_fraction(
                order=order,
                bar_open=float(bar_open),
                bar_high=float(bar_high),
                bar_low=float(bar_low),
                bar_close=float(bar_close),
            )
            if fill_fraction <= 0.0:
                return

            fill_qty = qty if fill_fraction >= 0.999999 else min(qty, qty * fill_fraction)
            if fill_qty <= 1e-12:
                return

            reserved_released = 0.0
            reserved_released = float(order.reserved_cash) * (fill_qty / qty) if qty > 0 else 0.0
            reserved_cash = max(0.0, reserved_cash - reserved_released)
            if order.side == "buy":
                # Consume only the reserved cash corresponding to the filled slice.
                notional = fill_qty * price
                fee = notional * fee_rate
                cost = notional + fee
                cash -= cost
                if cfg.separate_cash_pools:
                    if meta[sym]["asset_class"] == "crypto":
                        cash_crypto -= cost
                    else:
                        cash_stock -= cost
                positions[sym] = float(positions.get(sym, 0.0)) + fill_qty
            else:
                notional = fill_qty * price
                fee = notional * fee_rate
                proceeds = notional - fee
                cash += proceeds
                if cfg.separate_cash_pools:
                    if meta[sym]["asset_class"] == "crypto":
                        cash_crypto += proceeds
                    else:
                        cash_stock += proceeds
                positions[sym] = float(positions.get(sym, 0.0)) - fill_qty

            new_position = float(positions.get(sym, 0.0))
            if new_position > 0.0:
                position_entry_times.setdefault(sym, ts)
                position_peak_closes.setdefault(sym, float(bar_close))
            else:
                position_entry_times.pop(sym, None)
                position_peak_closes.pop(sym, None)

            fills.append(
                FillRecord(
                    timestamp=ts,
                    symbol=sym,
                    side=order.side,
                    price=price,
                    quantity=float(fill_qty),
                    fee_paid=float(fee),
                    cash_after=float(cash),
                    reserved_cash_after=float(reserved_cash),
                    position_after=float(positions[sym]),
                    kind=order.kind,
                )
            )
            remaining_qty = max(0.0, qty - fill_qty)
            if remaining_qty <= 1e-12:
                open_orders.pop((sym, order.side), None)
                return

            remaining_reserved = max(0.0, float(order.reserved_cash) - reserved_released)
            open_orders[(sym, order.side)] = OpenOrder(
                symbol=order.symbol,
                side=order.side,
                qty=float(remaining_qty),
                limit_price=float(order.limit_price),
                kind=str(order.kind),
                placed_at=order.placed_at,
                reserved_cash=float(remaining_reserved),
                cancel_requested_at=order.cancel_requested_at,
                cancel_effective_at=order.cancel_effective_at,
            )

        def _cancel_same_side(sym: str, side: str) -> None:
            nonlocal reserved_cash, cash_stock, cash_crypto
            existing = open_orders.pop((sym, side), None)
            if existing is None:
                return
            if float(existing.reserved_cash) > 0.0:
                reserved_cash = max(0.0, reserved_cash - float(existing.reserved_cash))

        def _request_cancel_same_side(ts: pd.Timestamp, sym: str, side: str) -> bool:
            existing = open_orders.get((sym, side))
            if existing is None:
                return False
            if cancel_ack_delay_bars <= 0:
                _cancel_same_side(sym, side)
                return False
            if existing.cancel_effective_at is not None:
                return True
            open_orders[(sym, side)] = OpenOrder(
                symbol=existing.symbol,
                side=existing.side,
                qty=float(existing.qty),
                limit_price=float(existing.limit_price),
                kind=str(existing.kind),
                placed_at=existing.placed_at,
                reserved_cash=float(existing.reserved_cash),
                cancel_requested_at=ts,
                cancel_effective_at=ts + pd.Timedelta(hours=cancel_ack_delay_bars),
            )
            return True

        def _order_similar(existing: OpenOrder, *, qty: float, price: float) -> bool:
            if existing.qty <= 0 or existing.limit_price <= 0:
                return False
            price_diff_pct = abs(float(existing.limit_price) - float(price)) / float(price) if price > 0 else float("inf")
            if price_diff_pct >= float(cfg.price_tol_pct):
                return False
            qty_diff_pct = abs(float(existing.qty) - float(qty)) / float(qty) if qty > 0 else float("inf")
            notional_diff = abs((float(existing.qty) - float(qty)) * float(price))
            return (qty_diff_pct < float(cfg.qty_tol_pct)) or (notional_diff < float(cfg.qty_tol_notional_usd))

        def _place_order(ts: pd.Timestamp, *, sym: str, intent_side: str, intent_qty: float, intent_price: float, kind: str) -> None:
            nonlocal reserved_cash, cash_stock, cash_crypto
            side = intent_side.lower()
            if side not in {"buy", "sell"}:
                return
            qty = float(intent_qty)
            price = float(intent_price)
            if qty <= 0 or price <= 0:
                return

            fee_rate = float(meta[sym]["maker_fee"])
            min_notional = _min_notional(sym)

            # Apply open_order_at_price_or_all "target qty" semantics for entry orders only.
            if kind == "entry":
                inv = float(positions.get(sym, 0.0))
                current_same_side = max(0.0, inv) if side == "buy" else max(0.0, -inv)
                remaining = qty - current_same_side
                if current_same_side > 0 and remaining <= qty * 0.01:
                    return
                if remaining <= 0:
                    return
                qty = remaining
                if qty <= 0:
                    return

            # Enforce inventory availability for exit sizing (defensive).
            if kind == "exit":
                inv = float(positions.get(sym, 0.0))
                if side == "sell":
                    qty = min(qty, max(0.0, inv))
                else:
                    qty = min(qty, max(0.0, -inv))
                if qty <= 0:
                    return

            # Enforce broker min notional (approx).
            if min_notional > 0 and (qty * price) < min_notional:
                qty = min_notional / price

            # Manage replace/keep semantics for same-side open order.
            existing = open_orders.get((sym, side))
            if existing is not None and cfg.keep_similar_orders and _order_similar(existing, qty=qty, price=price):
                return
            if _request_cancel_same_side(ts, sym, side):
                return

            # Reserve entry buying power at placement time (broker behavior approximation).
            reserved = 0.0
            if kind == "entry":
                avail = _available_cash(sym)
                cost_per_unit = price * (1 + fee_rate)
                if cost_per_unit <= 0:
                    return
                if qty * cost_per_unit > avail:
                    qty = (cash_buffer * avail) / cost_per_unit
                if qty <= 0:
                    return
                if min_notional > 0 and (qty * price) < min_notional:
                    return
                reserved = float(qty * cost_per_unit)
                reserved_cash += reserved

            open_orders[(sym, side)] = OpenOrder(
                symbol=sym,
                side=side,
                qty=float(qty),
                limit_price=float(price),
                kind=str(kind),
                placed_at=ts,
                reserved_cash=float(reserved),
            )

        def _update_position_trackers(ts: pd.Timestamp, current_group: pd.DataFrame) -> None:
            for row in current_group.itertuples(index=False):
                sym = str(row.symbol).upper()
                qty = float(positions.get(sym, 0.0))
                if qty > 0.0:
                    position_entry_times.setdefault(sym, ts)
                    close_price = float(row.close)
                    prev_peak = float(position_peak_closes.get(sym, close_price))
                    position_peak_closes[sym] = max(prev_peak, close_price)
                else:
                    position_entry_times.pop(sym, None)
                    position_peak_closes.pop(sym, None)

        def _risk_exit_price(close_price: float) -> float:
            return float(close_price) * (1.0 - force_exit_offset_pct / 100.0)

        def _apply_risk_exit_rules(ts: pd.Timestamp, current_group: pd.DataFrame) -> set[str]:
            triggered_symbols: set[str] = set()
            if max_hold_hours is None and trailing_stop_pct is None:
                return triggered_symbols

            for row in current_group.itertuples(index=False):
                sym = str(row.symbol).upper()
                qty = float(positions.get(sym, 0.0))
                if qty <= 0.0:
                    continue

                close_price = float(row.close)
                should_exit = False
                held_since = position_entry_times.get(sym)
                if max_hold_hours is not None and held_since is not None:
                    held_hours = (ts - held_since).total_seconds() / 3600.0
                    if held_hours >= float(max_hold_hours):
                        should_exit = True

                peak_close = float(position_peak_closes.get(sym, close_price))
                if trailing_stop_pct is not None and peak_close > 0.0:
                    drop_pct = (peak_close - close_price) / peak_close * 100.0
                    if drop_pct >= float(trailing_stop_pct):
                        should_exit = True

                if not should_exit:
                    continue

                triggered_symbols.add(sym)
                _cancel_same_side(sym, "buy")
                _cancel_same_side(sym, "sell")
                _place_order(
                    ts,
                    sym=sym,
                    intent_side="sell",
                    intent_qty=abs(qty),
                    intent_price=_risk_exit_price(close_price),
                    kind="exit",
                )

            return triggered_symbols

        # Determine symbol order: if config.symbols is provided, preserve that order.
        symbol_order: List[str]
        if cfg.symbols:
            desired = [str(s).upper() for s in cfg.symbols if str(s).upper() in symbols]
            rest = [s for s in symbols if s not in set(desired)]
            symbol_order = desired + rest
        else:
            symbol_order = list(symbols)

        total_steps = int(frame["timestamp"].nunique())
        for ts, group in frame.groupby("timestamp", sort=True):
            ts = pd.Timestamp(ts)

            # Update last_close first (mark-to-market uses most recent close for symbols present).
            for row in group.itertuples(index=False):
                sym = str(row.symbol).upper()
                last_close[sym] = float(row.close)

            # Fill eligible orders using this bar's range.
            for row in group.itertuples(index=False):
                sym = str(row.symbol).upper()
                bar_open = float(getattr(row, "open", row.close))
                high = float(row.high)
                low = float(row.low)
                close = float(row.close)

                # Fill sell first then buy to avoid optimistic same-bar round-trip assumptions.
                sell_order = open_orders.get((sym, "sell"))
                if sell_order is not None and _is_eligible(sell_order, ts):
                    _fill_order(ts, bar_open, high, low, close, sell_order)

                buy_order = open_orders.get((sym, "buy"))
                if buy_order is not None and _is_eligible(buy_order, ts):
                    _fill_order(ts, bar_open, high, low, close, buy_order)

            # Apply deferred cancels after the bar had one last chance to fill.
            for key, order in list(open_orders.items()):
                if _cancel_is_effective(order, ts):
                    _cancel_same_side(order.symbol, order.side)

            _update_position_trackers(ts, group)
            risk_exit_symbols = _apply_risk_exit_rules(ts, group)

            # Place/update orders using this bar's action row.
            for sym in symbol_order:
                if sym in risk_exit_symbols:
                    continue
                sym_group = group[group["symbol"] == sym]
                if sym_group.empty:
                    continue

                if meta[sym]["asset_class"] == "stock" and cfg.enforce_market_hours:
                    if not stock_open_mask.get((ts, sym), True):
                        continue

                row = sym_group.iloc[-1]
                if not (
                    pd.notna(row.get("buy_price"))
                    and pd.notna(row.get("sell_price"))
                    and pd.notna(row.get("buy_amount"))
                    and pd.notna(row.get("sell_amount"))
                ):
                    continue

                action = {
                    "timestamp": ts,
                    "symbol": sym,
                    "buy_price": float(row.buy_price),
                    "sell_price": float(row.sell_price),
                    "buy_amount": float(row.buy_amount),
                    "sell_amount": float(row.sell_amount),
                }
                plan = build_plan_from_action(action, intensity_scale=float(cfg.intensity_scale))
                buy_price = float(plan.buy_price) * (1.0 - float(cfg.price_offset_pct))
                sell_price = float(plan.sell_price) * (1.0 + float(cfg.price_offset_pct))
                adjusted = ensure_valid_levels(buy_price, sell_price, min_gap_pct=float(cfg.min_gap_pct))
                if adjusted is None:
                    continue
                buy_price, sell_price = adjusted

                inv = float(positions.get(sym, 0.0))
                directions = resolve_trade_directions(
                    sym,
                    allow_short=bool(cfg.allow_short),
                    long_only_symbols=cfg.long_only_symbols,
                    short_only_symbols=cfg.short_only_symbols,
                    use_default_groups=True,
                )
                direction_conflict = (not directions.can_long) and (not directions.can_short)
                exit_only = sym in exit_only_set
                if inv > 0 and not directions.can_long:
                    exit_only = True
                if inv < 0 and not directions.can_short:
                    exit_only = True
                if direction_conflict:
                    exit_only = True

                account_view = SimpleNamespace(
                    cash=float(cash),
                    equity=float(_equity()),
                    buying_power=max(0.0, float(_equity())),
                )
                allocation = allocation_usd_for_symbol(
                    account_view,
                    symbol=sym,
                    allocation_usd=cfg.allocation_usd,
                    allocation_pct=cfg.allocation_pct,
                    allocation_mode=cfg.allocation_mode,
                    symbols_count=len(symbol_order),
                    prefer_cash_for_crypto=True,
                )

                intents = build_order_intents(
                    plan,
                    position_qty=inv,
                    allocation_usd=allocation,
                    buy_price=float(buy_price),
                    sell_price=float(sell_price),
                    can_long=bool(directions.can_long),
                    can_short=bool(directions.can_short),
                    allow_short=bool(cfg.allow_short),
                    exit_only=bool(exit_only),
                    allow_position_adds=bool(cfg.allow_position_adds),
                    always_full_exit=bool(cfg.always_full_exit),
                )

                for intent in intents:
                    _place_order(
                        ts,
                        sym=sym,
                        intent_side=str(intent.side),
                        intent_qty=float(intent.qty),
                        intent_price=float(intent.limit_price),
                        kind=str(intent.kind),
                    )

            # Record combined equity at this timestamp.
            equity = float(_equity())
            gross = float(_gross_exposure())
            equity_values.append(equity)
            per_hour_rows.append(
                {
                    "timestamp": ts,
                    "equity": equity,
                    "cash": float(cash),
                    "reserved_cash": float(reserved_cash),
                    "available_cash": float(_available_cash()),
                    "gross_exposure": float(gross),
                    "open_orders": float(len(open_orders)),
                }
            )
            early_exit = evaluate_drawdown_vs_profit_early_exit(
                equity_values,
                total_steps=total_steps,
                label="newnanoalpacahourlyexp.HourlyTraderMarketSimulator",
            )
            if early_exit.should_stop:
                print_early_exit(early_exit)
                break

        equity_curve = pd.Series(equity_values, index=pd.to_datetime([r["timestamp"] for r in per_hour_rows], utc=True))
        per_hour = pd.DataFrame(per_hour_rows)

        # Annualization: use average periods-per-year across symbols as a rough heuristic.
        periods = [float(meta[s]["periods_per_year"]) for s in symbols if float(meta[s]["periods_per_year"]) > 0]
        periods_per_year = float(np.mean(periods)) if periods else float(24 * 365)
        metrics = self._compute_metrics(equity_curve, periods_per_year)

        return HourlyTraderSimulationResult(
            equity_curve=equity_curve,
            per_hour=per_hour,
            fills=fills,
            final_cash=float(cash),
            final_reserved_cash=float(reserved_cash),
            final_positions={k: float(v) for k, v in positions.items() if abs(float(v)) > 1e-12},
            open_orders=list(open_orders.values()),
            metrics=metrics,
        )

    @staticmethod
    def _compute_metrics(equity_curve: pd.Series, periods_per_year: float) -> Dict[str, float]:
        if equity_curve.empty:
            return {"total_return": 0.0, "sortino": 0.0}
        returns = compute_step_returns(equity_curve.values)
        sortino = annualized_sortino(returns, periods_per_year=periods_per_year)
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        return {
            "total_return": float(total_return),
            "sortino": float(sortino),
            "mean_hourly_return": float(returns.mean() if returns.size else 0.0),
        }


__all__ = [
    "FillRecord",
    "HourlyTraderMarketSimulator",
    "HourlyTraderSimulationConfig",
    "HourlyTraderSimulationResult",
    "OpenOrder",
]
