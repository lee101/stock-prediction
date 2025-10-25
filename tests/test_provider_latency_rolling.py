from __future__ import annotations

import sys
from pathlib import Path

import json
import sys
from scripts.provider_latency_rolling import compute_rolling, load_rollup, render_markdown, main as rolling_main


def write_rollup(tmp_path: Path, rows: list[tuple[str, str, float, float, float, float, int]]):
    path = tmp_path / "rollup.csv"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("timestamp,provider,avg_ms,p50_ms,p95_ms,max_ms,count\n")
        for row in rows:
            timestamp, provider, avg_ms, p50_ms, p95_ms, max_ms, count = row
            handle.write(
                f"{timestamp},{provider},{avg_ms},{p50_ms},{p95_ms},{max_ms},{count}\n"
            )
    return path


def test_compute_rolling(tmp_path):
    rollup_path = write_rollup(
        tmp_path,
        [
            ("2025-10-24T12:00:00+00:00", "yahoo", 300.0, 290.0, 320.0, 340.0, 16),
            ("2025-10-25T12:00:00+00:00", "yahoo", 310.0, 300.0, 330.0, 350.0, 16),
        ],
    )
    rows = load_rollup(rollup_path)
    aggregates = compute_rolling(rows, window=2)
    assert "yahoo" in aggregates
    assert aggregates["yahoo"]["window"] == 2
    assert abs(aggregates["yahoo"]["avg_ms"] - 305.0) < 1e-6
    assert abs(aggregates["yahoo"]["delta_avg_ms"] - 5.0) < 1e-6
    assert abs(aggregates["yahoo"]["delta_p95_ms"] - 5.0) < 1e-6


def test_render_markdown(tmp_path):
    rollup_path = write_rollup(
        tmp_path,
        [
            ("2025-10-24T12:00:00+00:00", "yahoo", 300.0, 290.0, 320.0, 340.0, 16),
        ],
    )
    rows = load_rollup(rollup_path)
    aggregates = compute_rolling(rows, window=5)
    markdown = render_markdown(aggregates, window=5)
    assert "Rolling Provider Latency" in markdown
    assert "yahoo" in markdown
    assert "ΔAvg" in markdown


def test_main_writes_json(tmp_path, monkeypatch):
    rollup_path = write_rollup(
        tmp_path,
        [
            ("2025-10-24T12:00:00+00:00", "yahoo", 300.0, 290.0, 320.0, 340.0, 16),
        ],
    )
    md_path = tmp_path / "rolling.md"
    json_path = tmp_path / "rolling.json"
    history_path = tmp_path / "history.jsonl"
    argv = [
        "provider_latency_rolling.py",
        "--rollup",
        str(rollup_path),
        "--output",
        str(md_path),
        "--json-output",
        str(json_path),
        "--window",
        "3",
        "--history-jsonl",
        str(history_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rolling_main()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "yahoo" in data
    assert "avg_ms" in data["yahoo"]
    history_lines = history_path.read_text(encoding="utf-8").splitlines()
    assert history_lines
