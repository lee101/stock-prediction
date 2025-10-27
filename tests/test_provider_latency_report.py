from __future__ import annotations

import sys
from pathlib import Path

from scripts.provider_latency_report import load_latency, percentile, render_summary
from scripts.provider_latency_report import main as latency_main
import sys


def write_latency_log(tmp_path: Path, rows: list[tuple[str, str, str, float]]) -> Path:
    log = tmp_path / "provider_latency.csv"
    with log.open("w", encoding="utf-8") as handle:
        handle.write("timestamp,symbol,provider,latency_ms\n")
        for timestamp, symbol, provider, latency in rows:
            handle.write(f"{timestamp},{symbol},{provider},{latency}\n")
    return log


def test_load_latency_parses_and_sorts(tmp_path):
    log = write_latency_log(
        tmp_path,
        [
            ("2025-10-24T12:00:01+00:00", "QQQ", "yahoo", 110.0),
            ("2025-10-24T12:00:00+00:00", "XLF", "stooq", 90.0),
        ],
    )
    samples = load_latency(log)
    assert samples[0].symbol == "XLF"
    assert samples[1].provider == "yahoo"


def test_percentile_interpolation():
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 50) == 25.0
    assert percentile(values, 100) == 40.0


def test_render_summary_contains_stats(tmp_path):
    log = write_latency_log(
        tmp_path,
        [
            ("2025-10-24T12:00:00+00:00", "QQQ", "yahoo", 120.0),
            ("2025-10-24T12:00:00+00:00", "XLF", "yahoo", 80.0),
        ],
    )
    samples = load_latency(log)
    summary = render_summary(samples)
    assert "avg" in summary
    assert "Latest sample" in summary


def test_render_summary_alert(tmp_path):
    log = write_latency_log(
        tmp_path,
        [
            ("2025-10-24T12:00:00+00:00", "QQQ", "yahoo", 600.0),
            ("2025-10-24T12:00:01+00:00", "QQQ", "yahoo", 700.0),
        ],
    )
    samples = load_latency(log)
    summary = render_summary(samples, p95_threshold=500.0)
    assert "[alert] yahoo" in summary


def test_main_writes_rollup(tmp_path, monkeypatch):
    log = write_latency_log(
        tmp_path,
        [
            ("2025-10-24T12:00:00+00:00", "QQQ", "yahoo", 600.0),
            ("2025-10-24T12:00:00+00:00", "QQQ", "stooq", 400.0),
        ],
    )
    summary_path = tmp_path / "latency_summary.txt"
    rollup_path = tmp_path / "latency_rollup.csv"
    argv = [
        "provider_latency_report.py",
        "--log",
        str(log),
        "--output",
        str(summary_path),
        "--p95-threshold",
        "500",
        "--rollup-csv",
        str(rollup_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    latency_main()
    assert summary_path.exists()
    content = rollup_path.read_text(encoding="utf-8").splitlines()
    assert content[0] == "timestamp,provider,avg_ms,p50_ms,p95_ms,max_ms,count"
    assert any("yahoo" in line for line in content[1:])
