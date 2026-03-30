from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from traininglib.report import _format_generated_timestamp, write_report_markdown


def test_format_generated_timestamp_normalizes_to_utc() -> None:
    eastern = timezone(timedelta(hours=-4))
    current = datetime(2026, 3, 30, 8, 15, tzinfo=eastern)

    assert _format_generated_timestamp(current) == "2026-03-30 12:15 UTC"


def test_write_report_markdown_uses_utc_generated_label(tmp_path: Path, monkeypatch) -> None:
    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            current = cls(2026, 3, 30, 12, 34, tzinfo=timezone.utc)
            if tz is None:
                return current.replace(tzinfo=None)
            return current.astimezone(tz)

    monkeypatch.setattr("traininglib.report.datetime.datetime", _FrozenDateTime)

    out_path = tmp_path / "report.md"
    write_report_markdown(
        str(out_path),
        title="Smoke Report",
        args={"alpha": 1},
        train_metrics={"loss": 0.1},
    )

    text = out_path.read_text(encoding="utf-8")
    assert "*Generated:* 2026-03-30 12:34 UTC" in text
    assert "# Smoke Report" in text
