from __future__ import annotations

import sys
from pathlib import Path

from scripts.provider_latency_history_report import load_history, main as history_main, render_history


def write_history(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "history.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            # ensure deterministic ordering
            handle.write(__import__("json").dumps(row, sort_keys=True) + "\n")
    return path


def test_render_history_outputs_sparkline(tmp_path):
    history_path = write_history(
        tmp_path,
        [
            {
                "timestamp": "2025-10-24T20:00:00+00:00",
                "window": 5,
                "aggregates": {
                    "yahoo": {"avg_ms": 300.0, "delta_avg_ms": 0.0, "p95_ms": 320.0, "delta_p95_ms": 0.0},
                },
            },
            {
                "timestamp": "2025-10-24T20:05:00+00:00",
                "window": 5,
                "aggregates": {
                    "yahoo": {"avg_ms": 320.0, "delta_avg_ms": 20.0, "p95_ms": 340.0, "delta_p95_ms": 20.0},
                },
            },
        ],
    )
    entries = load_history(history_path)
    markdown = render_history(entries, window=2)
    assert "yahoo" in markdown
    assert "Sparkline" in markdown


def test_main_produces_markdown(tmp_path, monkeypatch):
    history_path = write_history(
        tmp_path,
        [
            {
                "timestamp": "2025-10-24T20:00:00+00:00",
                "window": 5,
                "aggregates": {
                    "yahoo": {"avg_ms": 310.0, "delta_avg_ms": 10.0, "p95_ms": 335.0, "delta_p95_ms": 15.0},
                },
            }
        ],
    )
    output_path = tmp_path / "history.md"
    argv = [
        "provider_latency_history_report.py",
        "--history",
        str(history_path),
        "--output",
        str(output_path),
        "--window",
        "5",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    history_main()
    content = output_path.read_text(encoding="utf-8")
    assert "Provider Latency History" in content
