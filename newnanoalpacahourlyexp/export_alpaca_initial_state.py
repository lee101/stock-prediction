from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

import alpaca_wrapper
from alpaca.trading import GetOrdersRequest
from src.fees import get_fee_for_symbol
from src.hourly_trader_utils import infer_working_order_kind


def _normalize_symbol(value: object) -> str:
    return str(value or "").replace("/", "").replace("-", "").upper()


def _coerce_ts(value: object) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp value: {value!r}")
    return pd.Timestamp(ts)


def _parse_symbols(raw: str) -> list[str]:
    return [_normalize_symbol(token) for token in str(raw).split(",") if str(token).strip()]


def _fetch_orders(symbols: Iterable[str], *, fetch_after: pd.Timestamp, fetch_until: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        slash_symbol = alpaca_wrapper.remap_symbols(symbol)
        req = GetOrdersRequest(
            status="all",
            symbols=[slash_symbol],
            after=fetch_after.to_pydatetime(),
            until=fetch_until.to_pydatetime(),
            direction="asc",
            limit=500,
        )
        for order in alpaca_wrapper.alpaca_api.get_orders(filter=req):
            rows.append(
                {
                    "id": str(getattr(order, "id", "")),
                    "symbol": _normalize_symbol(getattr(order, "symbol", symbol)),
                    "created_at": pd.to_datetime(getattr(order, "created_at", None), utc=True, errors="coerce"),
                    "filled_at": pd.to_datetime(getattr(order, "filled_at", None), utc=True, errors="coerce"),
                    "canceled_at": pd.to_datetime(getattr(order, "canceled_at", None), utc=True, errors="coerce"),
                    "side": str(getattr(getattr(order, "side", None), "value", getattr(order, "side", ""))).lower(),
                    "status": str(getattr(getattr(order, "status", None), "value", getattr(order, "status", ""))).lower(),
                    "qty": float(getattr(order, "qty", 0.0) or 0.0),
                    "filled_qty": float(getattr(order, "filled_qty", 0.0) or 0.0),
                    "limit_price": float(getattr(order, "limit_price", 0.0) or 0.0),
                    "filled_price": float(getattr(order, "filled_avg_price", 0.0) or 0.0),
                }
            )
    return pd.DataFrame(rows)


def reconstruct_positions_at_start(
    *,
    current_positions: dict[str, float],
    orders: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    symbols: Iterable[str],
) -> dict[str, float]:
    positions = {symbol: float(current_positions.get(symbol, 0.0)) for symbol in symbols}
    if orders.empty:
        return positions

    fills = orders[
        (orders["filled_qty"] > 0.0)
        & pd.to_datetime(orders["filled_at"], utc=True, errors="coerce").between(window_start, window_end, inclusive="both")
    ].copy()
    if fills.empty:
        return positions

    fills["signed_qty"] = fills.apply(
        lambda row: float(row["filled_qty"]) if str(row["side"]).lower() == "buy" else -float(row["filled_qty"]),
        axis=1,
    )
    delta_by_symbol = fills.groupby("symbol")["signed_qty"].sum().to_dict()
    for symbol, delta in delta_by_symbol.items():
        positions[symbol] = float(positions.get(symbol, 0.0)) - float(delta)
    return positions


def reconstruct_cash_at_start(
    *,
    current_cash: float,
    orders: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> float:
    cash = float(current_cash)
    if orders.empty:
        return cash

    fills = orders[
        (orders["filled_qty"] > 0.0)
        & pd.to_datetime(orders["filled_at"], utc=True, errors="coerce").between(window_start, window_end, inclusive="both")
    ].copy()
    if fills.empty:
        return cash

    delta = 0.0
    for row in fills.itertuples(index=False):
        symbol = _normalize_symbol(row.symbol)
        fee_rate = float(get_fee_for_symbol(symbol))
        notional = float(row.filled_qty) * float(row.filled_price)
        if str(row.side).lower() == "buy":
            delta -= notional * (1.0 + fee_rate)
        else:
            delta += notional * (1.0 - fee_rate)
    return cash - delta


def reconstruct_open_orders_at_start(
    *,
    orders: pd.DataFrame,
    positions_at_start: dict[str, float],
    window_start: pd.Timestamp,
    reserve_buy_notional: bool,
) -> list[dict[str, object]]:
    if orders.empty:
        return []

    created_at = pd.to_datetime(orders["created_at"], utc=True, errors="coerce")
    filled_at = pd.to_datetime(orders["filled_at"], utc=True, errors="coerce")
    canceled_at = pd.to_datetime(orders["canceled_at"], utc=True, errors="coerce")
    active = orders[
        created_at.le(window_start)
        & (filled_at.isna() | filled_at.gt(window_start))
        & (canceled_at.isna() | canceled_at.gt(window_start))
    ].copy()
    if active.empty:
        return []

    merged: dict[tuple[str, str], dict[str, object]] = {}
    for row in active.itertuples(index=False):
        symbol = _normalize_symbol(row.symbol)
        side = str(row.side).lower()
        qty = float(row.qty)
        price = float(row.limit_price)
        if side not in {"buy", "sell"} or qty <= 0.0 or price <= 0.0:
            continue
        position_qty = float(positions_at_start.get(symbol, 0.0))
        kind = infer_working_order_kind(side=side, position_qty=position_qty)
        reserved_cash = 0.0
        if reserve_buy_notional and kind == "entry" and side == "buy":
            reserved_cash = qty * price * (1.0 + float(get_fee_for_symbol(symbol)))

        key = (symbol, side)
        payload = {
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            "price": price,
            "kind": kind,
            "created_at": pd.Timestamp(row.created_at).isoformat(),
            "reserved_cash": reserved_cash,
        }
        existing = merged.get(key)
        if existing is None:
            merged[key] = payload
            continue
        existing_qty = float(existing["quantity"])
        total_qty = existing_qty + qty
        if total_qty <= 0.0:
            continue
        existing["price"] = (float(existing["price"]) * existing_qty + price * qty) / total_qty
        existing["quantity"] = total_qty
        existing["reserved_cash"] = float(existing["reserved_cash"]) + reserved_cash
        existing["created_at"] = min(str(existing["created_at"]), payload["created_at"])
    return list(merged.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct a live Alpaca initial-state snapshot for hourly replay.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols to reconstruct.")
    parser.add_argument("--window-start", required=True, help="UTC replay window start timestamp.")
    parser.add_argument(
        "--window-end",
        default=None,
        help="UTC replay window end timestamp. Defaults to now, and should normally be close to current time.",
    )
    parser.add_argument(
        "--prefetch-hours",
        type=float,
        default=168.0,
        help="Extra order lookback before window_start so active pre-window orders can be reconstructed.",
    )
    parser.add_argument(
        "--reserve-buy-notional",
        action="store_true",
        help="Reserve cash for reconstructed entry buy orders at window start.",
    )
    parser.add_argument("--output", required=True, help="Path to write the initial-state JSON.")
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    window_start = _coerce_ts(args.window_start)
    window_end = _coerce_ts(args.window_end) if args.window_end else pd.Timestamp.now(tz="UTC")
    now = pd.Timestamp.now(tz="UTC")
    if window_end < now - pd.Timedelta(minutes=10):
        raise ValueError(
            f"window_end={window_end.isoformat()} is stale relative to now={now.isoformat()}; "
            "this reconstruction assumes the current live account matches window_end."
        )
    if window_end <= window_start:
        raise ValueError("window_end must be after window_start.")

    fetch_after = window_start - pd.Timedelta(hours=float(args.prefetch_hours))

    account = alpaca_wrapper.get_account(use_cache=False)
    current_cash = float(getattr(account, "cash", 0.0) or 0.0)
    current_positions = {
        _normalize_symbol(getattr(position, "symbol", "")): float(getattr(position, "qty", 0.0) or 0.0)
        for position in alpaca_wrapper.get_all_positions()
        if _normalize_symbol(getattr(position, "symbol", "")) in set(symbols)
    }
    orders = _fetch_orders(symbols, fetch_after=fetch_after, fetch_until=window_end)

    positions_at_start = reconstruct_positions_at_start(
        current_positions=current_positions,
        orders=orders,
        window_start=window_start,
        window_end=window_end,
        symbols=symbols,
    )
    cash_at_start = reconstruct_cash_at_start(
        current_cash=current_cash,
        orders=orders,
        window_start=window_start,
        window_end=window_end,
    )
    open_orders_at_start = reconstruct_open_orders_at_start(
        orders=orders,
        positions_at_start=positions_at_start,
        window_start=window_start,
        reserve_buy_notional=bool(args.reserve_buy_notional),
    )

    output = {
        "window_start_utc": window_start.isoformat(),
        "window_end_utc": window_end.isoformat(),
        "reconstructed_at_utc": now.isoformat(),
        "cash": float(cash_at_start),
        "positions": [
            {"symbol": symbol, "quantity": float(positions_at_start.get(symbol, 0.0))}
            for symbol in symbols
            if abs(float(positions_at_start.get(symbol, 0.0))) > 1e-12
        ],
        "open_orders": open_orders_at_start,
        "current_cash": current_cash,
        "current_positions": current_positions,
        "orders_fetched": int(len(orders)),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
