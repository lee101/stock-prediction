from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import binanceneural.binance_watchers as module


def test_spawn_watcher_reuses_active_watcher_within_price_tolerance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(module, "WATCHERS_DIR", tmp_path)
    monkeypatch.setattr(module, "_is_pid_alive", lambda pid: True)

    existing_path = module.watcher_config_path("BTCUSD", "buy", "entry", suffix="100")
    existing_path.write_text(
        json.dumps(
            {
                "active": True,
                "pid": 1234,
                "symbol": "BTCUSD",
                "side": "buy",
                "mode": "entry",
                "limit_price": 100.0,
                "target_qty": 1.0,
                "expiry_at": (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    stop_calls: list[Path] = []
    monkeypatch.setattr(module, "stop_existing_watcher", lambda path, reason="": stop_calls.append(path))
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not spawn")))

    path = module.spawn_watcher(
        module.WatcherPlan(
            symbol="BTCUSD",
            side="buy",
            mode="entry",
            limit_price=100.05,
            target_qty=1.0,
            expiry_minutes=30,
            poll_seconds=10,
            price_tolerance=0.001,
        )
    )

    assert path == existing_path
    assert stop_calls == []


def test_spawn_watcher_writes_log_paths_and_uses_file_handles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(module, "WATCHERS_DIR", tmp_path)
    monkeypatch.setattr(module, "_is_pid_alive", lambda pid: False)

    captured: dict[str, object] = {}

    def _fake_popen(*args, **kwargs):
        captured["stdout_name"] = getattr(kwargs.get("stdout"), "name", None)
        captured["stderr_name"] = getattr(kwargs.get("stderr"), "name", None)
        return SimpleNamespace(pid=4321)

    monkeypatch.setattr(module.subprocess, "Popen", _fake_popen)

    path = module.spawn_watcher(
        module.WatcherPlan(
            symbol="ETHUSD",
            side="sell",
            mode="exit",
            limit_price=2500.0,
            target_qty=0.25,
            expiry_minutes=60,
            poll_seconds=15,
            price_tolerance=0.002,
        )
    )

    assert path is not None
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["stdout_log_path"].endswith(".stdout.log")
    assert metadata["stderr_log_path"].endswith(".stderr.log")
    assert captured["stdout_name"] == metadata["stdout_log_path"]
    assert captured["stderr_name"] == metadata["stderr_log_path"]
