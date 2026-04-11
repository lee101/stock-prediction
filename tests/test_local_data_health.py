from __future__ import annotations

from dataclasses import dataclass

from src.local_data_health import LocalDataHealthStatus, format_local_data_health_lines, local_data_status_counts


@dataclass(frozen=True)
class _Detail:
    status: str
    local_data_date: str | None
    reason: str | None = None


def test_format_local_data_health_lines_supports_mapping_details() -> None:
    lines = format_local_data_health_lines(
        symbol_details={
            "AAPL": {"status": "stale", "local_data_date": "2026-04-03"},
            "MSFT": {"status": "missing", "local_data_date": None},
            "GOOG": {
                "status": "invalid",
                "local_data_date": None,
                "reason": "Daily frame missing columns:\n timestamp/date + ['open']",
            },
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
        "- invalid symbols: GOOG (Daily frame missing columns: timestamp/date + ['open'])",
    ]


def test_format_local_data_health_lines_supports_object_details() -> None:
    lines = format_local_data_health_lines(
        symbol_details={
            "BTCUSD": _Detail(status="stale", local_data_date="2026-04-02"),
            "DOGEUSD": _Detail(
                status="invalid",
                local_data_date=None,
                reason="could not parse\ncsv payload",
            ),
            "ETHUSD": _Detail(status="usable", local_data_date="2026-04-04"),
        },
        usable_symbol_count=1,
        latest_local_data_date="2026-04-04",
    )

    assert lines == [
        "Local data health:",
        "- usable symbols: 1/3",
        "- latest local data date: 2026-04-04",
        "- stale symbols: BTCUSD (2026-04-02)",
        "- invalid symbols: DOGEUSD (could not parse csv payload)",
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


def test_format_local_data_health_lines_truncates_large_status_lists() -> None:
    lines = format_local_data_health_lines(
        symbol_details={
            "AAPL": {"status": "missing", "local_data_date": None},
            "MSFT": {"status": "missing", "local_data_date": None},
            "NVDA": {"status": "missing", "local_data_date": None},
            "GOOG": {"status": "missing", "local_data_date": None},
            "META": {"status": "missing", "local_data_date": None},
            "TSLA": {"status": "missing", "local_data_date": None},
            "AMZN": {"status": "usable", "local_data_date": "2026-04-04"},
        },
        usable_symbol_count=1,
        latest_local_data_date="2026-04-04",
    )

    assert lines == [
        "Local data health:",
        "- usable symbols: 1/7",
        "- latest local data date: 2026-04-04",
        "- missing symbols: AAPL, MSFT, NVDA, GOOG, META (+1 more)",
    ]


def test_local_data_status_counts_supports_mapping_and_object_details() -> None:
    counts = local_data_status_counts(
        {
            "AAPL": {"status": LocalDataHealthStatus.USABLE},
            "MSFT": _Detail(status=LocalDataHealthStatus.STALE, local_data_date="2026-04-02"),
            "NVDA": {"status": "missing"},
            "GOOG": _Detail(status="invalid", local_data_date=None, reason="bad csv"),
            "META": {"status": "unexpected"},
        }
    )

    assert counts == {
        "usable": 1,
        "stale": 1,
        "missing": 1,
        "invalid": 1,
    }
