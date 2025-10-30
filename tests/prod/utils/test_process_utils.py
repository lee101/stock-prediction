import json
from types import SimpleNamespace

import pytest

from src import process_utils


@pytest.fixture
def tmp_watchers_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(process_utils, "MAXDIFF_WATCHERS_DIR", tmp_path)
    return tmp_path


def test_spawn_open_replaces_existing_watcher(tmp_watchers_dir, monkeypatch):
    symbol = "AAPL"
    side = "buy"
    limit_price = 98.5
    target_qty = 3.0

    suffix = process_utils._format_float(limit_price, 4)
    config_path = process_utils._watcher_config_path(symbol, side, "entry", suffix=suffix)
    existing_pid = 12345
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"pid": existing_pid, "active": True, "state": "launched"})
    )

    killed = []

    def fake_kill(pid, sig):
        killed.append((pid, sig))

    monkeypatch.setattr(process_utils.os, "kill", fake_kill)

    dummy_process = SimpleNamespace(pid=67890)
    monkeypatch.setattr(process_utils.subprocess, "Popen", lambda *a, **k: dummy_process)

    process_utils.spawn_open_position_at_maxdiff_takeprofit(symbol, side, limit_price, target_qty)

    assert killed == [(existing_pid, process_utils.signal.SIGTERM)]

    metadata = json.loads(config_path.read_text())
    assert metadata["pid"] == dummy_process.pid
    assert metadata["state"] == "launched"
    assert metadata["poll_seconds"] == process_utils.MAXDIFF_ENTRY_DEFAULT_POLL_SECONDS


def test_spawn_close_replaces_existing_watcher(tmp_watchers_dir, monkeypatch):
    symbol = "AAPL"
    side = "buy"
    takeprofit_price = 132.0

    suffix = process_utils._format_float(takeprofit_price, 4)
    config_path = process_utils._watcher_config_path(symbol, side, "exit", suffix=suffix)
    existing_pid = 54321
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"pid": existing_pid, "active": True, "state": "launched"})
    )

    killed = []

    def fake_kill(pid, sig):
        killed.append((pid, sig))

    monkeypatch.setattr(process_utils.os, "kill", fake_kill)

    dummy_process = SimpleNamespace(pid=98765)
    monkeypatch.setattr(process_utils.subprocess, "Popen", lambda *a, **k: dummy_process)

    process_utils.spawn_close_position_at_maxdiff_takeprofit(symbol, side, takeprofit_price)

    assert killed == [(existing_pid, process_utils.signal.SIGTERM)]

    metadata = json.loads(config_path.read_text())
    assert metadata["pid"] == dummy_process.pid
    assert metadata["state"] == "launched"
    assert metadata["poll_seconds"] == process_utils.MAXDIFF_EXIT_DEFAULT_POLL_SECONDS
    assert metadata["price_tolerance"] == pytest.approx(
        process_utils.MAXDIFF_EXIT_DEFAULT_PRICE_TOLERANCE
    )
