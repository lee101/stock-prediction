from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import scripts.write_latency_step_summary as summary


def test_write_summary(tmp_path, monkeypatch):
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        json.dumps({"yahoo": {"avg_ms": 320.0, "delta_avg_ms": 5.0, "p95_ms": 340.0}}),
        encoding="utf-8",
    )
    digest_path = tmp_path / "digest.md"
    digest_path.write_text("Latency Alert Digest\n- alert", encoding="utf-8")

    summary_path = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))
    argv = [
        "write_latency_step_summary.py",
        "--snapshot",
        str(snapshot_path),
        "--digest",
        str(digest_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    summary.main()
    content = summary_path.read_text(encoding="utf-8")
    assert "Latency Health" in content
    assert "yahoo" in content
