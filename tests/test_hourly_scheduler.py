from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.hourly_scheduler import (
    HourlyRunCoordinator,
    extract_symbols_from_text,
    load_symbols_from_file,
    resolve_hourly_symbols,
)


def test_extract_symbols_from_text_dedupes_and_ignores_noise():
    blob = """
    symbols = ['AAPL', 'msft', "ETHUSD"]
    # 'IGNORED'
    fallback = "btcusd"
    """
    assert extract_symbols_from_text(blob) == ["AAPL", "MSFT", "ETHUSD", "BTCUSD"]


def test_load_symbols_from_missing_file_returns_empty(tmp_path: Path):
    missing = tmp_path / "absent.txt"
    assert load_symbols_from_file(missing) == []


def test_resolve_hourly_symbols_prefers_env():
    defaults = ["AAPL", "MSFT"]
    resolved = resolve_hourly_symbols("tsla, ethusd, TsLa", [], defaults)
    assert resolved == ["TSLA", "ETHUSD"]


def test_resolve_hourly_symbols_uses_first_file(tmp_path: Path):
    file_one = tmp_path / "symbols.txt"
    file_two = tmp_path / "later.txt"
    file_one.write_text("['NVDA', 'AMZN']")
    file_two.write_text("['IGNORE']")
    defaults = ["BTCUSD"]

    resolved = resolve_hourly_symbols(None, [file_one, file_two], defaults)
    assert resolved == ["NVDA", "AMZN"]


def test_resolve_hourly_symbols_falls_back_to_defaults(tmp_path: Path):
    defaults = ["SOLUSD", "LINKUSD"]
    resolved = resolve_hourly_symbols(None, [tmp_path / "missing.txt"], defaults)
    assert resolved == ["SOLUSD", "LINKUSD"]


def test_hourly_run_coordinator_allows_single_run_per_hour():
    coordinator = HourlyRunCoordinator(analysis_window_minutes=5)
    t0 = datetime(2025, 1, 1, 13, 0, tzinfo=timezone.utc)

    assert coordinator.should_run(t0) is True
    coordinator.mark_executed(t0)

    assert coordinator.should_run(t0 + timedelta(minutes=2)) is False
    assert coordinator.should_run(t0 + timedelta(minutes=30)) is False

    next_hour = t0 + timedelta(hours=1, minutes=2)
    assert coordinator.should_run(next_hour) is True

    coordinator.mark_executed(next_hour)
    assert coordinator.should_run(next_hour + timedelta(minutes=10)) is False


def test_hourly_run_coordinator_blocks_runs_outside_window():
    coordinator = HourlyRunCoordinator(analysis_window_minutes=10, allow_immediate_start=False)
    first_hour = datetime(2025, 5, 1, 9, 20, tzinfo=timezone.utc)
    assert coordinator.should_run(first_hour) is False

    allowed = datetime(2025, 5, 1, 10, 5, tzinfo=timezone.utc)
    coordinator.mark_executed(allowed)
    later = datetime(2025, 5, 1, 11, 20, tzinfo=timezone.utc)
    assert coordinator.should_run(later) is False


def test_hourly_run_coordinator_catches_up_outside_window():
    coordinator = HourlyRunCoordinator(analysis_window_minutes=5, allow_catch_up=True)
    start = datetime(2025, 3, 1, 14, 0, tzinfo=timezone.utc)

    assert coordinator.should_run(start) is True
    coordinator.mark_executed(start)

    late_next_hour = start + timedelta(hours=1, minutes=45)
    assert coordinator.should_run(late_next_hour) is True
    coordinator.mark_executed(late_next_hour)

    assert coordinator.should_run(late_next_hour + timedelta(minutes=1)) is False
