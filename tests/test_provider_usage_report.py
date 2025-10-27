from __future__ import annotations

from datetime import datetime

from scripts.provider_usage_report import build_timeline, load_usage, render_report, main as provider_main
import sys


def test_load_usage_sorted(tmp_path):
    log = tmp_path / "provider_usage.csv"
    log.write_text(
        "timestamp,provider,count\n"
        "2025-10-24T19:00:00+00:00,yahoo,16\n"
        "2025-10-23T19:00:00+00:00,stooq,16\n",
        encoding="utf-8",
    )

    rows = load_usage(log)
    assert [row.provider for row in rows] == ["stooq", "yahoo"]


def test_build_timeline_window(tmp_path):
    log = tmp_path / "provider_usage.csv"
    log.write_text(
        "timestamp,provider,count\n"
        "2025-10-22T00:00:00+00:00,stooq,16\n"
        "2025-10-23T00:00:00+00:00,yahoo,16\n"
        "2025-10-24T00:00:00+00:00,yahoo,16\n",
        encoding="utf-8",
    )
    rows = load_usage(log)
    timeline = build_timeline(rows, window=2)
    assert timeline == "YY"


def test_render_report_includes_latest(tmp_path):
    log = tmp_path / "provider_usage.csv"
    log.write_text(
        "timestamp,provider,count\n"
        "2025-10-24T00:00:00+00:00,yahoo,16\n",
        encoding="utf-8",
    )
    rows = load_usage(log)
    output = render_report(rows, timeline_window=5, sparkline=True)
    assert "Total runs: 1" in output
    assert "provider=yahoo" in output


def test_main_writes_output(tmp_path, monkeypatch):
    log = tmp_path / "provider_usage.csv"
    log.write_text(
        "timestamp,provider,count\n"
        "2025-10-24T00:00:00+00:00,yahoo,16\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "summary.txt"
    argv = [
        "provider_usage_report.py",
        "--log",
        str(log),
        "--output",
        str(output_path),
        "--timeline-window",
        "5",
        "--no-sparkline",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    provider_main()
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "provider=yahoo" in content
