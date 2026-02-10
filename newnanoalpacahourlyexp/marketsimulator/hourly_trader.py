from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.allocation_utils import allocation_usd_for_symbol
from src.fees import get_fee_for_symbol
from src.hourly_trader_utils import build_order_intents, build_plan_from_action, ensure_valid_levels
from src.metrics_utils import annualized_sortino, compute_step_returns
from src.symbol_utils import is_crypto_symbol
from src.trade_directions import resolve_trade_directions

from .simulator import _infer_periods_per_year, _market_open_mask, _to_new_york


@dataclass
class HourlyTraderSimulationConfig:
    initial_cash: float = 10_000.0
    allocation_usd: Optional[float] = None  # per symbol
    allocation_pct: Optional[float] = 0.05
    allocation_mode: str = "per_symbol"  # "per_symbol" | "portfolio"
    intensity_scale: float = 1.0
    price_offset_pct: float = 0.0
    min_gap_pct: float = 0.001
    # Execution realism: lag between decision bar close and earliest fill eligibility.
    # For live hourly trading, 1 is appropriate (decide on bar t, fill on bar t+1).
    decision_lag_bars: int = 1
    enforce_market_hours: bool = True
    fee_by_symbol: Optional[Dict[str, float]] = None
    periods_per_year_by_symbol: Optional[Dict[str, float]] = None
    allow_short: bool = False
    long_only_symbols: Optional[Sequence[str]] = None
    short_only_symbols: Optional[Sequence[str]] = None
    exit_only_symbols: Optional[Sequence[str]] = None
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


@dataclass(frozen=True)
class OpenOrder:
    symbol: str
    side: str  # "buy" or "sell"
    qty: float
    limit_price: float
    kind: str  # "entry" or "exit"
    placed_at: pd.Timestamp
    reserved_cash: float = 0.0  # only for buy orders


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

        cash = float(cfg.initial_cash)
        reserved_cash = 0.0
        positions: Dict[str, float] = {sym: 0.0 for sym in symbols}
        last_close: Dict[str, float] = {}
        open_orders: Dict[Tuple[str, str], OpenOrder] = {}
        fills: List[FillRecord] = []
        equity_values: List[float] = []
        per_hour_rows: List[Dict[str, float | str]] = []

        def _min_notional(sym: str) -> float:
            if meta[sym]["asset_class"] == "crypto":
                return float(cfg.min_notional_crypto)
            return float(cfg.min_notional_stock)

        def _available_cash() -> float:
            return max(0.0, cash - reserved_cash)

        def _is_eligible(order: OpenOrder, ts: pd.Timestamp) -> bool:
            if decision_lag_bars <= 0:
                return True
            return ts >= (order.placed_at + pd.Timedelta(hours=decision_lag_bars))

        def _fill_order(ts: pd.Timestamp, bar_high: float, bar_low: float, order: OpenOrder) -> None:
            nonlocal cash, reserved_cash
            sym = order.symbol
            fee_rate = float(meta[sym]["maker_fee"])
            qty = float(order.qty)
            price = float(order.limit_price)
            if qty <= 0 or price <= 0:
                return

            if order.side == "buy":
                if bar_low > price:
                    return
                # Consume reserved cash (approx broker behavior). Cost is deterministic in this sim.
                notional = qty * price
                fee = notional * fee_rate
                cost = notional + fee
                cash -= cost
                reserved_cash -= float(order.reserved_cash)
                positions[sym] = float(positions.get(sym, 0.0)) + qty
            else:
                if bar_high < price:
                    return
                notional = qty * price
                fee = notional * fee_rate
                proceeds = notional - fee
                cash += proceeds
                positions[sym] = float(positions.get(sym, 0.0)) - qty

            fills.append(
                FillRecord(
                    timestamp=ts,
                    symbol=sym,
                    side=order.side,
                    price=price,
                    quantity=qty,
                    fee_paid=float(fee),
                    cash_after=float(cash),
                    reserved_cash_after=float(reserved_cash),
                    position_after=float(positions[sym]),
                    kind=order.kind,
                )
            )
            open_orders.pop((sym, order.side), None)

        def _cancel_same_side(sym: str, side: str) -> None:
            nonlocal reserved_cash
            existing = open_orders.pop((sym, side), None)
            if existing is None:
                return
            if existing.side == "buy":
                reserved_cash -= float(existing.reserved_cash)

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
            nonlocal reserved_cash
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
            _cancel_same_side(sym, side)

            # Reserve cash for buys at placement time (broker behavior).
            reserved = 0.0
            if side == "buy":
                avail = _available_cash()
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

        # Determine symbol order: if config.symbols is provided, preserve that order.
        symbol_order: List[str]
        if cfg.symbols:
            desired = [str(s).upper() for s in cfg.symbols if str(s).upper() in symbols]
            rest = [s for s in symbols if s not in set(desired)]
            symbol_order = desired + rest
        else:
            symbol_order = list(symbols)

        for ts, group in frame.groupby("timestamp", sort=True):
            ts = pd.Timestamp(ts)

            # Update last_close first (mark-to-market uses most recent close for symbols present).
            for row in group.itertuples(index=False):
                sym = str(row.symbol).upper()
                last_close[sym] = float(row.close)

            # Fill eligible orders using this bar's range.
            for row in group.itertuples(index=False):
                sym = str(row.symbol).upper()
                high = float(row.high)
                low = float(row.low)

                # Fill sell first then buy to avoid optimistic same-bar round-trip assumptions.
                sell_order = open_orders.get((sym, "sell"))
                if sell_order is not None and _is_eligible(sell_order, ts):
                    _fill_order(ts, high, low, sell_order)

                buy_order = open_orders.get((sym, "buy"))
                if buy_order is not None and _is_eligible(buy_order, ts):
                    _fill_order(ts, high, low, buy_order)

            # Place/update orders using this bar's action row.
            for sym in symbol_order:
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
                    equity=float(cash) + sum(float(q) * float(last_close.get(s, 0.0)) for s, q in positions.items()),
                    buying_power=float(cash),  # conservative default (no margin) unless caller models otherwise
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
            equity = float(cash) + sum(float(qty) * float(last_close.get(sym, 0.0)) for sym, qty in positions.items())
            gross = sum(abs(float(qty)) * float(last_close.get(sym, 0.0)) for sym, qty in positions.items())
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

