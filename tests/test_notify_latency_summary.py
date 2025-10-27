from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.notify_latency_summary as summary


def test_main_posts_summary(tmp_path, monkeypatch):
    digest = tmp_path / "digest.md"
    digest.write_text("Latency Alert Digest\n- alert", encoding="utf-8")
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"yahoo": {"avg_ms": 320.0, "delta_avg_ms": 5.0, "p95_ms": 340.0}}), encoding="utf-8")
    leaderboard = tmp_path / "leaderboard.md"
    leaderboard.write_text(
        "| Provider | INFO | WARN | CRIT | Total |\n|----------|------|------|------|-------|\n| YAHOO | 0 | 1 | 2 | 3 |\n",
        encoding="utf-8",
    )
    weekly = tmp_path / "weekly.md"
    weekly.write_text(
        "| Provider | CRIT Δ | WARN Δ |\n|----------|---------|---------|\n| YAHOO | +2 | +1 |\n",
        encoding="utf-8",
    )

    captured = {}

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=0):  # noqa: ANN001
        captured["body"] = request.data
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr(summary.urllib.request, "urlopen", fake_urlopen)
    argv = [
        "notify_latency_summary.py",
        "--digest",
        str(digest),
        "--snapshot",
        str(snapshot),
        "--leaderboard",
        str(leaderboard),
        "--weekly-report",
        str(weekly),
        "--webhook",
        "https://example.com/hook",
        "--format",
        "slack",
        "--image-url",
        "https://img",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    summary.main()
    payload = json.loads(captured["body"].decode("utf-8"))
    assert "Latency Alert Digest" in payload["text"]
    assert payload["attachments"][0]["image_url"] == "https://img"
    assert "Top latency offenders" in payload["text"]
    assert "Weekly trend highlights" in payload["text"]
