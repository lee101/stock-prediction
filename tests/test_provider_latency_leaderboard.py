from __future__ import annotations

import json
from pathlib import Path

from scripts.provider_latency_leaderboard import build_leaderboard, load_history


def write_history(tmp_path: Path) -> Path:
    path = tmp_path / "history.jsonl"
    entries = [
        {
            "timestamp": "2025-10-24T20:00:00+00:00",
            "provider_severity": {"YAHOO": {"CRIT": 2, "WARN": 1}},
            "severity_totals": {"CRIT": 2, "WARN": 1},
        },
        {
            "timestamp": "2025-10-24T21:00:00+00:00",
            "provider_severity": {"YAHOO": {"CRIT": 1}, "SOXX": {"WARN": 2}},
            "severity_totals": {"CRIT": 1, "WARN": 2},
        },
        {
            "timestamp": "2025-10-24T22:00:00+00:00",
            "provider_severity": {"SOXX": {"WARN": 1}},
            "severity_totals": {"WARN": 1},
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    return path


def test_load_history(tmp_path):
    path = write_history(tmp_path)
    entries = load_history(path)
    assert len(entries) == 3


def test_build_leaderboard(tmp_path):
    path = write_history(tmp_path)
    entries = load_history(path)
    leaderboard = build_leaderboard(entries, window=2, compare_window=1)
    assert "YAHOO" in leaderboard
    assert "SOXX" in leaderboard
    assert "ΔTotal" in leaderboard
