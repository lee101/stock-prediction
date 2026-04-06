from __future__ import annotations

from dataclasses import dataclass

from src.local_data_health import format_local_data_health_lines


@dataclass(frozen=True)
class _Detail:
    status: str
    local_data_date: str | None


def test_format_local_data_health_lines_supports_mapping_details() -> None:
    lines = format_local_data_health_lines(
        symbol_details={
            "AAPL": {"status": "stale", "local_data_date": "2026-04-03"},
            "MSFT": {"status": "missing", "local_data_date": None},
            "GOOG": {"status": "invalid", "local_data_date": None},
            "NVDA": {"status": "usable", "local_data_date": "2026-04-04"},
        },
        usable_symbol_count=1,
        latest_local_data_date="2026-04-04",
    )

    assert lines == [
        "Local data health:",
        "- usable symbols: 1/4",
        "- latest local data date: 2026-04-04",
        "- stale symbols: AAPL (2026-04-03)",
        "- missing symbols: MSFT",
        "- invalid symbols: GOOG",
    ]


def test_format_local_data_health_lines_supports_object_details() -> None:
    lines = format_local_data_health_lines(
        symbol_details={
            "BTCUSD": _Detail(status="stale", local_data_date="2026-04-02"),
            "ETHUSD": _Detail(status="usable", local_data_date="2026-04-04"),
        },
        usable_symbol_count=1,
        latest_local_data_date="2026-04-04",
    )

    assert lines == [
        "Local data health:",
        "- usable symbols: 1/2",
        "- latest local data date: 2026-04-04",
        "- stale symbols: BTCUSD (2026-04-02)",
    ]


def test_format_local_data_health_lines_returns_empty_list_without_symbols() -> None:
    assert (
        format_local_data_health_lines(
            symbol_details={},
            usable_symbol_count=0,
            latest_local_data_date=None,
        )
        == []
    )
