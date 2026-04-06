from __future__ import annotations

from collections.abc import Mapping


def _detail_value(detail: object, field: str) -> object:
    if isinstance(detail, Mapping):
        return detail.get(field)
    return getattr(detail, field, None)


def _symbols_with_status(
    symbol_details: Mapping[str, object],
    *,
    status: str,
    include_local_data_date: bool = False,
) -> list[str]:
    symbols: list[str] = []
    for symbol, detail in symbol_details.items():
        if _detail_value(detail, "status") != status:
            continue
        if include_local_data_date:
            symbols.append(f"{symbol} ({_detail_value(detail, 'local_data_date')})")
        else:
            symbols.append(str(symbol))
    return symbols


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
        include_local_data_date=True,
    )
    if stale_symbols:
        lines.append("- stale symbols: " + ", ".join(stale_symbols))

    missing_symbols = _symbols_with_status(symbol_details, status="missing")
    if missing_symbols:
        lines.append("- missing symbols: " + ", ".join(missing_symbols))

    invalid_symbols = _symbols_with_status(symbol_details, status="invalid")
    if invalid_symbols:
        lines.append("- invalid symbols: " + ", ".join(invalid_symbols))

    return lines


__all__ = ["format_local_data_health_lines"]
