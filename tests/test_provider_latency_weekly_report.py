from __future__ import annotations

import json
from pathlib import Path

from scripts.provider_latency_weekly_report import build_report, load_history, compute_trend


def write_history(tmp_path: Path) -> Path:
    path = tmp_path / "history.jsonl"
    entries = [
        {
            "timestamp": "2025-10-24T20:00:00+00:00",
            "provider_severity": {"YAHOO": {"CRIT": 1}},
        },
        {
            "timestamp": "2025-10-25T20:00:00+00:00",
            "provider_severity": {"YAHOO": {"CRIT": 2}},
        },
        {
            "timestamp": "2025-10-26T20:00:00+00:00",
            "provider_severity": {"SOXX": {"WARN": 1}},
        },
        {
            "timestamp": "2025-10-27T20:00:00+00:00",
            "provider_severity": {"SOXX": {"WARN": 3}},
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    return path


def test_load_history(tmp_path):
    path = write_history(tmp_path)
    entries = load_history(path)
    assert len(entries) == 4


def test_build_report_flags_provider(tmp_path):
    path = write_history(tmp_path)
    entries = load_history(path)
    report = build_report(entries, window=2, compare_window=2, min_delta=1)
    assert "YAHOO" in report
    assert "SOXX" in report


def test_compute_trend_returns_deltas(tmp_path):
    path = write_history(tmp_path)
    entries = load_history(path)
    deltas = compute_trend(entries, window=2, compare_window=2)
    assert deltas["YAHOO"]["CRIT"] == -3
    assert deltas["SOXX"]["WARN"] == 4
