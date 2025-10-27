from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.provider_latency_trend_gate import main as gate_main


def write_history(tmp_path: Path, crit_delta: int, warn_delta: int) -> Path:
    path = tmp_path / "history.jsonl"
    entries = [
        {
            "timestamp": "2025-10-24",
            "provider_severity": {"YAHOO": {"CRIT": 1, "WARN": 1}},
        },
        {
            "timestamp": "2025-10-25",
            "provider_severity": {"YAHOO": {"CRIT": 1 + crit_delta, "WARN": 1 + warn_delta}},
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    return path


def test_trend_gate_pass(tmp_path, monkeypatch):
    history = write_history(tmp_path, crit_delta=0, warn_delta=0)
    argv = [
        "trend_gate.py",
        "--history",
        str(history),
        "--window",
        "1",
        "--compare-window",
        "1",
        "--crit-limit",
        "2",
        "--warn-limit",
        "2",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    gate_main()


def test_trend_gate_fail(tmp_path, monkeypatch):
    history = write_history(tmp_path, crit_delta=3, warn_delta=0)
    argv = [
        "trend_gate.py",
        "--history",
        str(history),
        "--window",
        "1",
        "--compare-window",
        "1",
        "--crit-limit",
        "2",
        "--warn-limit",
        "2",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        gate_main()
    assert excinfo.value.code == 2
