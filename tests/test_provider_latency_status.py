from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.provider_latency_status import evaluate, main as status_main


def test_evaluate_thresholds():
    snapshot = {
        "yahoo": {"avg_ms": 320.0, "delta_avg_ms": 35.0, "p95_ms": 340.0},
        "stooq": {"avg_ms": 310.0, "delta_avg_ms": 5.0, "p95_ms": 320.0},
    }
    status, details = evaluate(snapshot, warn_threshold=20.0, crit_threshold=40.0)
    assert status == "WARN"
    assert details["yahoo"]["severity"] == "warn"
    assert details["stooq"]["severity"] == "ok"


def test_main_outputs_json(tmp_path, monkeypatch):
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        json.dumps({"yahoo": {"avg_ms": 320.0, "delta_avg_ms": 45.0, "p95_ms": 350.0}}),
        encoding="utf-8",
    )
    argv = [
        "provider_latency_status.py",
        "--snapshot",
        str(snapshot_path),
        "--json",
        "--warn",
        "20",
        "--crit",
        "40",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        status_main()
    assert excinfo.value.code == 2
