from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import scripts.backfill_stock_hourly as mod


def _status(symbol: str, path: Path, staleness_hours: float = 1.0) -> mod.HourlyDataStatus:
    return mod.HourlyDataStatus(
        symbol=symbol,
        path=path,
        latest_timestamp=datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc),
        latest_close=123.45,
        staleness_hours=staleness_hours,
    )


def _issue(symbol: str, reason: str) -> mod.HourlyDataIssue:
    return mod.HourlyDataIssue(symbol=symbol, reason=reason, detail=f"{symbol}-{reason}")


def test_resolve_symbols_supports_file_and_inline(tmp_path: Path) -> None:
    path = tmp_path / "symbols.txt"
    path.write_text("odD\nnflx\n")
    symbols = mod._resolve_symbols(["crm, msft"], symbols_file=str(path))
    assert symbols == ["ODD", "NFLX", "CRM", "MSFT"]


def test_build_summary_tracks_ready_and_missing(tmp_path: Path) -> None:
    path = tmp_path / "ODD.csv"
    summary = mod.build_summary(
        symbols=["ODD", "CRM"],
        before_statuses=[],
        before_issues=[_issue("ODD", "missing"), _issue("CRM", "missing")],
        after_statuses=[_status("ODD", path)],
        after_issues=[_issue("CRM", "missing")],
    )
    assert summary["symbol_count"] == 2
    assert summary["before"]["ready_count"] == 0
    assert summary["after"]["ready_count"] == 1
    assert summary["newly_ready_symbols"] == ["ODD"]
    assert summary["still_missing_symbols"] == ["CRM"]


def test_main_check_only_writes_summary(monkeypatch, tmp_path: Path, capsys) -> None:
    path = tmp_path / "CRM.csv"

    class FakeValidator:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def filter_ready(self, symbols):
            assert list(symbols) == ["CRM", "ODD"]
            return [_status("CRM", path)], [_issue("ODD", "missing")]

    class FakeRefresher:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("Refresher should not be constructed in check-only mode")

    monkeypatch.setattr(mod, "HourlyDataValidator", FakeValidator)
    monkeypatch.setattr(mod, "HourlyDataRefresher", FakeRefresher)

    out = tmp_path / "summary.json"
    rc = mod.main(
        [
            "--symbols",
            "crm, odd",
            "--check-only",
            "--json-out",
            str(out),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "Hourly coverage 1/2 -> 1/2" in captured.out
    assert out.exists()
