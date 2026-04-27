from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts import collect_5min_bars


def test_collection_cycle_continues_after_symbol_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fetched: list[str] = []
    appended: list[str] = []

    def fake_fetch(symbol: str, *, limit: int) -> pd.DataFrame:
        fetched.append(symbol)
        if symbol == "BADUSDT":
            raise RuntimeError("temporary fetch failure")
        return pd.DataFrame([{"timestamp": pd.Timestamp("2026-01-01T00:00:00Z")}])

    def fake_append(path: Path, new: pd.DataFrame) -> int:
        appended.append(path.stem)
        assert not new.empty
        return len(new)

    monkeypatch.setattr(collect_5min_bars, "fetch_recent_5m", fake_fetch)
    monkeypatch.setattr(collect_5min_bars, "append_bars", fake_append)
    monkeypatch.setattr(collect_5min_bars, "daily_backfill", lambda symbols, out_root: None)

    last_backfill = collect_5min_bars.run_collection_cycle(
        ["BADUSDT", "GOODUSDT"],
        tmp_path,
        None,
    )

    assert fetched == ["BADUSDT", "GOODUSDT"]
    assert appended == ["GOODUSDT"]
    assert isinstance(last_backfill, datetime)


def test_collection_cycle_preserves_backfill_time_after_scheduler_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    previous_backfill = datetime(2026, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(
        collect_5min_bars,
        "fetch_recent_5m",
        lambda symbol, *, limit: pd.DataFrame(),
    )
    monkeypatch.setattr(collect_5min_bars, "append_bars", lambda path, new: 0)

    def fail_backfill(symbols: list[str], out_root: Path) -> None:
        raise RuntimeError("backfill scheduler failure")

    monkeypatch.setattr(collect_5min_bars, "daily_backfill", fail_backfill)

    last_backfill = collect_5min_bars.run_collection_cycle(
        ["DOGEUSDT"],
        tmp_path,
        previous_backfill,
    )

    assert last_backfill == previous_backfill
