from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import TypedDict

STATUS_SYMBOL_PREVIEW_LIMIT = 5


class LocalDataHealthStatus(StrEnum):
    USABLE = "usable"
    STALE = "stale"
    MISSING = "missing"
    INVALID = "invalid"


LOCAL_DATA_STATUSES = tuple(LocalDataHealthStatus)


class LocalDataStatusCounts(TypedDict):
    usable: int
    stale: int
    missing: int
    invalid: int


def _detail_value(detail: object, field: str) -> object:
    if isinstance(detail, Mapping):
        return detail.get(field)
    return getattr(detail, field, None)


def _compact_detail_value(value: object) -> str:
    return " ".join(str(value).split())


def _symbols_with_status(
    symbol_details: Mapping[str, object],
    *,
    status: str | LocalDataHealthStatus,
    detail_field: str | None = None,
) -> list[str]:
    symbols: list[str] = []
    for symbol, detail in symbol_details.items():
        if _detail_value(detail, "status") != status:
            continue
        label = str(symbol)
        if detail_field is not None:
            detail_value = _compact_detail_value(_detail_value(detail, detail_field))
            if detail_value and detail_value.lower() != "none":
                label = f"{label} ({detail_value})"
        symbols.append(label)
    return symbols


def _format_status_symbol_preview(symbols: list[str]) -> str:
    if len(symbols) <= STATUS_SYMBOL_PREVIEW_LIMIT:
        return ", ".join(symbols)
    preview = ", ".join(symbols[:STATUS_SYMBOL_PREVIEW_LIMIT])
    remaining = len(symbols) - STATUS_SYMBOL_PREVIEW_LIMIT
    return f"{preview} (+{remaining} more)"


def local_data_status_counts(
    symbol_details: Mapping[str, object],
) -> LocalDataStatusCounts:
    counts: LocalDataStatusCounts = {
        "usable": 0,
        "stale": 0,
        "missing": 0,
        "invalid": 0,
    }
    for detail in symbol_details.values():
        status = _detail_value(detail, "status")
        if isinstance(status, LocalDataHealthStatus):
            counts[status.value] += 1
        elif status in counts:
            counts[str(status)] += 1
    return counts


def format_local_data_health_lines(
    *,
    symbol_details: Mapping[str, object],
    usable_symbol_count: int,
    latest_local_data_date: str | None,
) -> list[str]:
    if not symbol_details:
        return []

    lines = [
        "Local data health:",
        f"- usable symbols: {usable_symbol_count}/{len(symbol_details)}",
    ]
    if latest_local_data_date:
        lines.append(f"- latest local data date: {latest_local_data_date}")

    stale_symbols = _symbols_with_status(
        symbol_details,
        status="stale",
        detail_field="local_data_date",
    )
    if stale_symbols:
        lines.append("- stale symbols: " + _format_status_symbol_preview(stale_symbols))

    missing_symbols = _symbols_with_status(symbol_details, status="missing")
    if missing_symbols:
        lines.append("- missing symbols: " + _format_status_symbol_preview(missing_symbols))

    invalid_symbols = _symbols_with_status(
        symbol_details,
        status="invalid",
        detail_field="reason",
    )
    if invalid_symbols:
        lines.append("- invalid symbols: " + _format_status_symbol_preview(invalid_symbols))

    return lines


__all__ = [
    "LocalDataHealthStatus",
    "LocalDataStatusCounts",
    "format_local_data_health_lines",
    "local_data_status_counts",
]
