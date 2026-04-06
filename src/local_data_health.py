"""Format local data health information for status reports."""
from __future__ import annotations


def format_local_data_health_lines(
    symbol_details: dict,
    usable_symbol_count: int = 0,
    latest_local_data_date: str | None = None,
) -> list[str]:
    lines: list[str] = []
    total = len(symbol_details)
    if latest_local_data_date:
        lines.append(f"Local data through {latest_local_data_date} ({usable_symbol_count}/{total} symbols usable)")
    stale = []
    missing = []
    for sym, detail in sorted(symbol_details.items()):
        if isinstance(detail, dict):
            status = detail.get("status", "")
            if status == "missing":
                missing.append(sym)
            elif status == "stale":
                stale.append(sym)
    if missing:
        lines.append(f"  Missing: {', '.join(missing)}")
    if stale:
        lines.append(f"  Stale: {', '.join(stale)}")
    return lines
