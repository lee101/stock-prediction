from __future__ import annotations

import sys
from pathlib import Path

import json

import scripts.notify_latency_alert as notify


def test_alert_appends_log(tmp_path, monkeypatch):
    log_path = tmp_path / "alerts.log"
    argv = [
        "notify_latency_alert.py",
        "--message",
        "Rolling latency shift +50.0 ms",
        "--log",
        str(log_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    notify.main()
    content = log_path.read_text(encoding="utf-8")
    assert "Rolling latency shift" in content

def test_alert_posts_webhook(tmp_path, monkeypatch):
    log_path = tmp_path / "alerts.log"
    captured = {}

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=0):  # noqa: ANN001
        captured["request"] = request
        captured["timeout"] = timeout
        captured["body"] = request.data
        return DummyResponse()

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)
    argv = [
        "notify_latency_alert.py",
        "--message",
        "Rolling latency shift +50.0 ms",
        "--log",
        str(log_path),
        "--webhook",
        "https://example.com/hook",
        "--format",
        "slack",
        "--channel",
        "#ops",
        "--log-link",
        "https://logs",
        "--plot-link",
        "https://plot",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    notify.main()
    assert captured["request"].full_url == "https://example.com/hook"
    payload = json.loads(captured["body"].decode("utf-8"))
    assert payload["channel"] == "#ops"
    assert payload["username"] == "LatencyBot"
    assert "https://logs" in payload["text"]
    assert "https://plot" in payload["text"]
