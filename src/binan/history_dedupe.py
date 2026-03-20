from __future__ import annotations

from typing import Any, Iterable, Mapping


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def dedupe_margin_trades(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for raw in rows:
        row = dict(raw)
        trade_id = _safe_int(row.get("id"))
        if trade_id is not None:
            key = ("id", trade_id)
        else:
            key = (
                "fallback",
                _safe_int(row.get("orderId")),
                _safe_int(row.get("time")),
                str(row.get("price", "")),
                str(row.get("qty", "")),
                bool(row.get("isBuyer")),
            )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def dedupe_margin_orders(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_order_id: dict[int, dict[str, Any]] = {}
    fallback: list[dict[str, Any]] = []
    fallback_seen: set[tuple[Any, ...]] = set()

    for raw in rows:
        row = dict(raw)
        order_id = _safe_int(row.get("orderId"))
        if order_id is None:
            key = (
                str(row.get("symbol", "")),
                str(row.get("side", "")),
                _safe_int(row.get("time")),
                str(row.get("price", "")),
                str(row.get("origQty", "")),
                str(row.get("executedQty", "")),
                str(row.get("status", "")),
            )
            if key in fallback_seen:
                continue
            fallback_seen.add(key)
            fallback.append(row)
            continue

        current = by_order_id.get(order_id)
        if current is None:
            by_order_id[order_id] = row
            continue

        current_ts = max(
            _safe_int(current.get("updateTime")) or -1,
            _safe_int(current.get("time")) or -1,
        )
        next_ts = max(
            _safe_int(row.get("updateTime")) or -1,
            _safe_int(row.get("time")) or -1,
        )
        if next_ts >= current_ts:
            by_order_id[order_id] = row

    deduped = list(by_order_id.values()) + fallback
    deduped.sort(
        key=lambda row: (
            _safe_int(row.get("time")) or 0,
            _safe_int(row.get("orderId")) or 0,
        )
    )
    return deduped


__all__ = ["dedupe_margin_orders", "dedupe_margin_trades"]
