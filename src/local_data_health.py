"""Formatting helpers for local daily-data health diagnostics."""

from __future__ import annotations


def format_local_data_health_lines(
    *,
    symbol_details: dict[str, dict],
    usable_symbol_count: int,
    latest_local_data_date: str | None,
) -> list[str]:
    """Return human-readable lines summarising per-symbol local data health.

    Parameters
    ----------
    symbol_details:
        Mapping of symbol -> {status, file_path, local_data_date, row_count, reason}.
        ``status`` is one of ``"usable"``, ``"stale"``, ``"missing"``, ``"unreadable"``.
    usable_symbol_count:
        Number of symbols whose data is usable.
    latest_local_data_date:
        ISO-date string of the freshest local data across all symbols, or *None*.
    """
    total = len(symbol_details)
    lines: list[str] = ["Local data health:"]
    lines.append(f"- usable symbols: {usable_symbol_count}/{total}")
    if latest_local_data_date:
        lines.append(f"- latest local data date: {latest_local_data_date}")

    stale: list[str] = []
    missing: list[str] = []
    for sym, detail in symbol_details.items():
        status = detail.get("status", "")
        if status == "stale":
            date = detail.get("local_data_date") or "?"
            stale.append(f"{sym} ({date})")
        elif status in ("missing", "unreadable"):
            missing.append(sym)

    if stale:
        lines.append(f"- stale symbols: {', '.join(stale)}")
    if missing:
        lines.append(f"- missing symbols: {', '.join(missing)}")

    # Actionable next-step lines
    stale_names = [
        sym for sym, d in symbol_details.items() if d.get("status") == "stale"
    ]
    if stale_names:
        lines.append(f"- Refresh stale local daily CSVs for: {', '.join(stale_names)}.")
    if missing:
        lines.append(
            "- Add the missing local daily CSV files under the resolved data directory."
        )

    return lines
