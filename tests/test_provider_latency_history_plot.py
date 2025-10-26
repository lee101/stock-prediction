from __future__ import annotations

import sys
from pathlib import Path

from scripts.provider_latency_history_plot import load_history, main as plot_main


def write_history(tmp_path: Path) -> Path:
    path = tmp_path / "history.jsonl"
    path.write_text(
        "{\"timestamp\":\"2025-10-24T20:00:00+00:00\",\"aggregates\":{\"yahoo\":{\"avg_ms\":310.0,\"p95_ms\":340.0}}}\n"
        "{\"timestamp\":\"2025-10-24T20:05:00+00:00\",\"aggregates\":{\"yahoo\":{\"avg_ms\":320.0,\"p95_ms\":350.0}}}\n",
        encoding="utf-8",
    )
    return path


def test_load_history(tmp_path):
    history_path = write_history(tmp_path)
    providers = load_history(history_path, window=2)
    assert "yahoo" in providers
    assert len(providers["yahoo"]["timestamps"]) == 2


def test_main_writes_html(tmp_path, monkeypatch):
    history_path = write_history(tmp_path)
    output_path = tmp_path / "plot.html"
    argv = [
        "provider_latency_history_plot.py",
        "--history",
        str(history_path),
        "--output",
        str(output_path),
        "--window",
        "10",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    plot_main()
    content = output_path.read_text(encoding="utf-8")
    assert "Plotly" in content
    assert "yahoo" in content
