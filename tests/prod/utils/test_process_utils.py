import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from src import process_utils


@pytest.fixture
def tmp_watchers_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(process_utils, "MAXDIFF_WATCHERS_DIR", tmp_path)
    return tmp_path


def test_spawn_open_replaces_existing_watcher(tmp_watchers_dir, monkeypatch):
    symbol = "AAPL_PRUNE"
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
        if sig == 0:  # Process alive check
            return
        killed.append((pid, sig))

    monkeypatch.setattr(process_utils.os, "kill", fake_kill)
    # Mock data freshness to always return True
    monkeypatch.setattr(process_utils, "_is_data_bar_fresh", lambda symbol, current_time=None: True)

    dummy_process = SimpleNamespace(pid=67890)
    monkeypatch.setattr(process_utils.subprocess, "Popen", lambda *a, **k: dummy_process)

    process_utils.spawn_open_position_at_maxdiff_takeprofit(symbol, side, limit_price, target_qty)

    assert killed == [(existing_pid, process_utils.signal.SIGTERM)]

    metadata = json.loads(config_path.read_text())
    assert metadata["pid"] == dummy_process.pid
    assert metadata["state"] == "launched"
    assert metadata["poll_seconds"] == process_utils.MAXDIFF_ENTRY_DEFAULT_POLL_SECONDS
    assert metadata["force_immediate"] is False
    assert "priority_rank" not in metadata


def test_spawn_open_force_immediate_sets_metadata(tmp_watchers_dir, monkeypatch):
    symbol = "MSFT"
    side = "sell"
    limit_price = 142.25
    target_qty = 5.0

    suffix = process_utils._format_float(limit_price, 4)
    config_path = process_utils._watcher_config_path(symbol, side, "entry", suffix=suffix)

    commands = []

    def fake_popen(command, *args, **kwargs):
        commands.append(command)
        return SimpleNamespace(pid=11111)

    monkeypatch.setattr(process_utils.subprocess, "Popen", fake_popen)

    process_utils.spawn_open_position_at_maxdiff_takeprofit(
        symbol,
        side,
        limit_price,
        target_qty,
        force_immediate=True,
        priority_rank=2,
    )

    metadata = json.loads(config_path.read_text())
    assert metadata["force_immediate"] is True
    assert metadata["priority_rank"] == 2
    assert any("--force-immediate" in cmd for cmd in commands)
    assert any("--priority-rank=2" in cmd for cmd in commands)


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
        if sig == 0:  # Process alive check
            return
        killed.append((pid, sig))

    monkeypatch.setattr(process_utils.os, "kill", fake_kill)
    # Mock data freshness to always return True
    monkeypatch.setattr(process_utils, "_is_data_bar_fresh", lambda symbol, current_time=None: True)

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


def test_is_pid_alive_returns_false_for_invalid_pid():
    assert process_utils._is_pid_alive(None) is False
    assert process_utils._is_pid_alive(0) is False
    assert process_utils._is_pid_alive(-1) is False


def test_watcher_matches_params_inactive_watcher():
    metadata = {"active": False, "limit_price": 100.0}
    assert process_utils._watcher_matches_params(metadata, limit_price=100.0) is False


def test_watcher_matches_params_no_pid():
    metadata = {"active": True, "limit_price": 100.0}
    assert process_utils._watcher_matches_params(metadata, limit_price=100.0) is False


def test_watcher_matches_params_dead_process(monkeypatch):
    def fake_kill(pid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr(process_utils.os, "kill", fake_kill)
    metadata = {"active": True, "pid": 99999, "limit_price": 100.0}
    assert process_utils._watcher_matches_params(metadata, limit_price=100.0) is False


def test_watcher_matches_params_expired_watcher(monkeypatch):
    def fake_kill(pid, sig):
        pass  # Simulate alive process

    monkeypatch.setattr(process_utils.os, "kill", fake_kill)
    expired = datetime.now(timezone.utc) - timedelta(hours=1)
    metadata = {
        "active": True,
        "pid": 12345,
        "limit_price": 100.0,
        "expiry_at": expired.isoformat(),
    }
    assert process_utils._watcher_matches_params(metadata, limit_price=100.0) is False


def test_watcher_matches_params_different_price(monkeypatch):
    def fake_kill(pid, sig):
        pass  # Simulate alive process

    monkeypatch.setattr(process_utils.os, "kill", fake_kill)
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    metadata = {
        "active": True,
        "pid": 12345,
        "limit_price": 100.0,
        "expiry_at": future.isoformat(),
    }
    assert process_utils._watcher_matches_params(metadata, limit_price=101.0) is False


def test_watcher_matches_params_different_strategy(monkeypatch):
    def fake_kill(pid, sig):
        pass  # Simulate alive process

    monkeypatch.setattr(process_utils.os, "kill", fake_kill)
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    metadata = {
        "active": True,
        "pid": 12345,
        "limit_price": 100.0,
        "entry_strategy": "maxdiff",
        "expiry_at": future.isoformat(),
    }
    assert (
        process_utils._watcher_matches_params(
            metadata, limit_price=100.0, entry_strategy="maxdiffalwayson"
        )
        is False
    )


def test_watcher_matches_params_success(monkeypatch):
    def fake_kill(pid, sig):
        pass  # Simulate alive process

    monkeypatch.setattr(process_utils.os, "kill", fake_kill)
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    metadata = {
        "active": True,
        "pid": 12345,
        "limit_price": 100.0,
        "target_qty": 5.0,
        "tolerance_pct": 0.0066,
        "entry_strategy": "maxdiffalwayson",
        "expiry_at": future.isoformat(),
    }
    assert (
        process_utils._watcher_matches_params(
            metadata,
            limit_price=100.0,
            target_qty=5.0,
            tolerance_pct=0.0066,
            entry_strategy="maxdiffalwayson",
        )
        is True
    )


def test_spawn_open_skips_identical_watcher(tmp_watchers_dir, monkeypatch):
    """Test that spawning identical watcher does not kill existing process."""
    symbol = "AAPL"
    side = "buy"
    limit_price = 98.5
    target_qty = 3.0
    entry_strategy = "maxdiffalwayson"

    suffix = process_utils._format_float(limit_price, 4)
    config_path = process_utils._watcher_config_path(symbol, side, "entry", suffix=suffix)
    existing_pid = 12345
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Create existing watcher with matching parameters
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    existing_metadata = {
        "pid": existing_pid,
        "active": True,
        "state": "launched",
        "limit_price": float(limit_price),
        "target_qty": float(target_qty),
        "tolerance_pct": 0.0066,
        "entry_strategy": entry_strategy,
        "expiry_at": future.isoformat(),
    }
    config_path.write_text(json.dumps(existing_metadata))

    killed = []

    def fake_kill(pid, sig):
        if sig == 0:  # os.kill(pid, 0) is used to check if process is alive
            return  # Simulate alive process
        killed.append((pid, sig))

    monkeypatch.setattr(process_utils.os, "kill", fake_kill)

    spawned = []

    def fake_popen(*args, **kwargs):
        spawned.append((args, kwargs))
        return SimpleNamespace(pid=67890)

    monkeypatch.setattr(process_utils.subprocess, "Popen", fake_popen)

    # Call spawn with identical parameters
    process_utils.spawn_open_position_at_maxdiff_takeprofit(
        symbol, side, limit_price, target_qty, entry_strategy=entry_strategy
    )

    # Should NOT kill existing process
    assert killed == []
    # Should NOT spawn new process
    assert spawned == []


def test_spawn_open_replaces_different_strategy(tmp_watchers_dir, monkeypatch):
    """Test that spawning with different strategy replaces existing watcher."""
    symbol = "MSFT"  # Use different symbol to avoid debounce cache
    side = "buy"
    limit_price = 350.25
    target_qty = 2.5

    suffix = process_utils._format_float(limit_price, 4)
    config_path = process_utils._watcher_config_path(symbol, side, "entry", suffix=suffix)
    existing_pid = 12345
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Create existing watcher with different strategy
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    existing_metadata = {
        "pid": existing_pid,
        "active": True,
        "state": "launched",
        "limit_price": float(limit_price),
        "target_qty": float(target_qty),
        "tolerance_pct": 0.0066,
        "entry_strategy": "maxdiff",
        "expiry_at": future.isoformat(),
    }
    config_path.write_text(json.dumps(existing_metadata))

    killed = []

    def fake_kill(pid, sig):
        if sig == 0:  # Check if alive
            return
        killed.append((pid, sig))

    monkeypatch.setattr(process_utils.os, "kill", fake_kill)
    # Mock data freshness to always return True
    monkeypatch.setattr(process_utils, "_is_data_bar_fresh", lambda symbol, current_time=None: True)

    dummy_process = SimpleNamespace(pid=67890)
    monkeypatch.setattr(process_utils.subprocess, "Popen", lambda *a, **k: dummy_process)

    # Call spawn with different strategy
    process_utils.spawn_open_position_at_maxdiff_takeprofit(
        symbol, side, limit_price, target_qty, entry_strategy="maxdiffalwayson"
    )

    # Should kill existing process
    assert killed == [(existing_pid, process_utils.signal.SIGTERM)]

    # Should have new metadata with new strategy
    metadata = json.loads(config_path.read_text())
    assert metadata["pid"] == dummy_process.pid
    assert metadata["entry_strategy"] == "maxdiffalwayson"


def test_spawn_open_prunes_conflicting_watchers(tmp_watchers_dir, monkeypatch):
    """Spawning a new watcher should terminate older ones for the same strategy."""
    symbol = "AAPL_PRUNE"
    side = "buy"
    entry_strategy = "maxdiffalwayson"
    new_limit = 98.5
    target_qty = 3.0

    # Existing watcher with same strategy but outdated limit
    legacy_limit = 101.25
    legacy_suffix = process_utils._format_float(legacy_limit, 4)
    legacy_path = process_utils._watcher_config_path(symbol, side, "entry", suffix=legacy_suffix)
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_metadata = {
        "pid": 11111,
        "active": True,
        "state": "waiting_for_trigger",
        "mode": "entry",
        "limit_price": float(legacy_limit),
        "target_qty": target_qty,
        "tolerance_pct": 0.0066,
        "entry_strategy": entry_strategy,
        "expiry_at": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
    }
    legacy_path.write_text(json.dumps(legacy_metadata))

    # Legacy watcher without entry_strategy should also be pruned for maxdiff family
    old_limit = 100.75
    old_suffix = process_utils._format_float(old_limit, 4)
    old_path = process_utils._watcher_config_path(symbol, side, "entry", suffix=old_suffix)
    old_metadata = {
        "pid": 22222,
        "active": True,
        "state": "waiting_for_trigger",
        "mode": "entry",
        "limit_price": float(old_limit),
        "target_qty": target_qty,
        "tolerance_pct": 0.0066,
        "expiry_at": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
    }
    old_path.write_text(json.dumps(old_metadata))

    # Watcher for a different strategy should remain untouched
    other_limit = 97.75
    other_suffix = process_utils._format_float(other_limit, 4)
    other_path = process_utils._watcher_config_path(symbol, side, "entry", suffix=other_suffix)
    other_metadata = {
        "pid": 33333,
        "active": True,
        "state": "waiting_for_trigger",
        "mode": "entry",
        "limit_price": float(other_limit),
        "target_qty": target_qty,
        "tolerance_pct": 0.0066,
        "entry_strategy": "maxdiff",
        "expiry_at": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
    }
    other_path.write_text(json.dumps(other_metadata))

    killed = []

    def fake_kill(pid, sig):
        if sig == 0:
            return
        killed.append((pid, sig))

    monkeypatch.setattr(process_utils.os, "kill", fake_kill)
    monkeypatch.setattr(process_utils, "_is_data_bar_fresh", lambda symbol, current_time=None: True)

    dummy_process = SimpleNamespace(pid=44444)
    monkeypatch.setattr(process_utils.subprocess, "Popen", lambda *a, **k: dummy_process)

    process_utils.spawn_open_position_at_maxdiff_takeprofit(
        symbol,
        side,
        new_limit,
        target_qty,
        entry_strategy=entry_strategy,
    )

    expected_pid_set = {11111, 22222}
    killed_pid_set = {pid for pid, sig in killed if sig == process_utils.signal.SIGTERM}
    assert killed_pid_set == expected_pid_set

    legacy_metadata_after = json.loads(legacy_path.read_text())
    assert legacy_metadata_after["active"] is False
    assert legacy_metadata_after["state"] == "superseded_entry_watcher"

    old_metadata_after = json.loads(old_path.read_text())
    assert old_metadata_after["active"] is False
    assert old_metadata_after["state"] == "superseded_entry_watcher"

    other_metadata_after = json.loads(other_path.read_text())
    assert other_metadata_after["active"] is True
    assert other_metadata_after["entry_strategy"] == "maxdiff"


def test_spawn_open_prunes_conflicts_with_matching_watcher(tmp_watchers_dir, monkeypatch):
    """Even if identical watcher exists, stale ones should be terminated."""
    symbol = "AAPL_PRUNE_MATCH"
    side = "buy"
    entry_strategy = "maxdiffalwayson"
    limit_price = 98.5
    target_qty = 3.5

    suffix = process_utils._format_float(limit_price, 4)
    config_path = process_utils._watcher_config_path(symbol, side, "entry", suffix=suffix)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    future_ts = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
    matching_metadata = {
        "pid": 55555,
        "active": True,
        "state": "waiting_for_trigger",
        "mode": "entry",
        "symbol": symbol,
        "side": side,
        "limit_price": float(limit_price),
        "target_qty": float(target_qty),
        "tolerance_pct": 0.0066,
        "entry_strategy": entry_strategy,
        "expiry_at": future_ts,
    }
    config_path.write_text(json.dumps(matching_metadata))

    stale_limit = 101.0
    stale_suffix = process_utils._format_float(stale_limit, 4)
    stale_path = process_utils._watcher_config_path(symbol, side, "entry", suffix=stale_suffix)
    stale_metadata = dict(matching_metadata)
    stale_metadata.update(
        {
            "pid": 66666,
            "limit_price": float(stale_limit),
        }
    )
    stale_path.write_text(json.dumps(stale_metadata))

    legacy_limit = 100.25
    legacy_suffix = process_utils._format_float(legacy_limit, 4)
    legacy_path = process_utils._watcher_config_path(symbol, side, "entry", suffix=legacy_suffix)
    legacy_metadata = {
        "pid": 77777,
        "active": True,
        "state": "waiting_for_trigger",
        "mode": "entry",
        "symbol": symbol,
        "side": side,
        "limit_price": float(legacy_limit),
        "target_qty": float(target_qty),
        "tolerance_pct": 0.0066,
        "expiry_at": future_ts,
    }
    legacy_path.write_text(json.dumps(legacy_metadata))

    killed = []

    def fake_kill(pid, sig):
        if sig == 0:
            return
        killed.append((pid, sig))

    monkeypatch.setattr(process_utils.os, "kill", fake_kill)

    spawned = []

    def fake_popen(*args, **kwargs):
        spawned.append((args, kwargs))
        return SimpleNamespace(pid=88888)

    monkeypatch.setattr(process_utils.subprocess, "Popen", fake_popen)

    process_utils.spawn_open_position_at_maxdiff_takeprofit(
        symbol,
        side,
        limit_price,
        target_qty,
        entry_strategy=entry_strategy,
    )

    # Should not spawn a new process because existing watcher matches.
    assert spawned == []

    expected_pid_set = {66666, 77777}
    killed_pid_set = {pid for pid, sig in killed if sig == process_utils.signal.SIGTERM}
    assert killed_pid_set == expected_pid_set

    matching_after = json.loads(config_path.read_text())
    assert matching_after["pid"] == matching_metadata["pid"]
    assert matching_after["active"] is True

    stale_after = json.loads(stale_path.read_text())
    assert stale_after["state"] == "superseded_entry_watcher"
    assert stale_after["active"] is False

    legacy_after = json.loads(legacy_path.read_text())
    assert legacy_after["state"] == "superseded_entry_watcher"
    assert legacy_after["active"] is False


def test_spawn_close_skips_identical_watcher(tmp_watchers_dir, monkeypatch):
    """Test that spawning identical exit watcher does not kill existing process."""
    symbol = "AAPL"
    side = "buy"
    takeprofit_price = 132.0
    entry_strategy = "maxdiffalwayson"

    suffix = process_utils._format_float(takeprofit_price, 4)
    config_path = process_utils._watcher_config_path(symbol, side, "exit", suffix=suffix)
    existing_pid = 54321
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Create existing watcher with matching parameters
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    existing_metadata = {
        "pid": existing_pid,
        "active": True,
        "state": "launched",
        "takeprofit_price": float(takeprofit_price),
        "price_tolerance": process_utils.MAXDIFF_EXIT_DEFAULT_PRICE_TOLERANCE,
        "entry_strategy": entry_strategy,
        "expiry_at": future.isoformat(),
    }
    config_path.write_text(json.dumps(existing_metadata))

    killed = []

    def fake_kill(pid, sig):
        if sig == 0:
            return
        killed.append((pid, sig))

    monkeypatch.setattr(process_utils.os, "kill", fake_kill)

    spawned = []

    def fake_popen(*args, **kwargs):
        spawned.append((args, kwargs))
        return SimpleNamespace(pid=98765)

    monkeypatch.setattr(process_utils.subprocess, "Popen", fake_popen)

    # Call spawn with identical parameters
    process_utils.spawn_close_position_at_maxdiff_takeprofit(
        symbol, side, takeprofit_price, entry_strategy=entry_strategy
    )

    # Should NOT kill existing process
    assert killed == []
    # Should NOT spawn new process
    assert spawned == []


def test_spawn_open_stores_strategy_in_metadata(tmp_watchers_dir, monkeypatch):
    """Test that entry_strategy is stored in watcher metadata."""
    symbol = "TSLA"  # Use different symbol to avoid debounce cache
    side = "buy"
    limit_price = 250.75
    target_qty = 5.0
    entry_strategy = "maxdiffalwayson"

    def fake_kill(pid, sig):
        if sig == 0:
            raise ProcessLookupError()  # No existing process
        pass

    monkeypatch.setattr(process_utils.os, "kill", fake_kill)
    # Mock data freshness to always return True
    monkeypatch.setattr(process_utils, "_is_data_bar_fresh", lambda symbol, current_time=None: True)
    dummy_process = SimpleNamespace(pid=67890)
    monkeypatch.setattr(process_utils.subprocess, "Popen", lambda *a, **k: dummy_process)

    process_utils.spawn_open_position_at_maxdiff_takeprofit(
        symbol, side, limit_price, target_qty, entry_strategy=entry_strategy
    )

    suffix = process_utils._format_float(limit_price, 4)
    config_path = process_utils._watcher_config_path(symbol, side, "entry", suffix=suffix)
    metadata = json.loads(config_path.read_text())

    assert metadata["entry_strategy"] == entry_strategy
    assert metadata["symbol"] == symbol
    assert metadata["side"] == side
    assert metadata["limit_price"] == limit_price
    assert metadata["target_qty"] == target_qty


def test_spawn_close_stores_strategy_in_metadata(tmp_watchers_dir, monkeypatch):
    """Test that entry_strategy is stored in exit watcher metadata."""
    symbol = "AAPL"
    side = "buy"
    takeprofit_price = 132.0
    entry_strategy = "maxdiff"

    monkeypatch.setattr(process_utils.os, "kill", lambda pid, sig: None)
    # Mock data freshness to always return True
    monkeypatch.setattr(process_utils, "_is_data_bar_fresh", lambda symbol, current_time=None: True)
    dummy_process = SimpleNamespace(pid=98765)
    monkeypatch.setattr(process_utils.subprocess, "Popen", lambda *a, **k: dummy_process)

    process_utils.spawn_close_position_at_maxdiff_takeprofit(
        symbol, side, takeprofit_price, entry_strategy=entry_strategy
    )

    suffix = process_utils._format_float(takeprofit_price, 4)
    config_path = process_utils._watcher_config_path(symbol, side, "exit", suffix=suffix)
    metadata = json.loads(config_path.read_text())

    assert metadata["entry_strategy"] == entry_strategy
    assert metadata["symbol"] == symbol
    assert metadata["side"] == side
    assert metadata["takeprofit_price"] == takeprofit_price


# Market timing tests


def test_calculate_next_crypto_bar_time():
    """Test that crypto bar time calculation returns next UTC midnight."""
    # Test at 14:30 UTC
    current = datetime(2025, 11, 1, 14, 30, 0, tzinfo=timezone.utc)
    next_bar = process_utils._calculate_next_crypto_bar_time(current)

    expected = datetime(2025, 11, 2, 0, 0, 0, tzinfo=timezone.utc)
    assert next_bar == expected


def test_calculate_next_crypto_bar_time_near_midnight():
    """Test crypto bar time when very close to midnight."""
    # Test at 23:59 UTC
    current = datetime(2025, 11, 1, 23, 59, 0, tzinfo=timezone.utc)
    next_bar = process_utils._calculate_next_crypto_bar_time(current)

    expected = datetime(2025, 11, 2, 0, 0, 0, tzinfo=timezone.utc)
    assert next_bar == expected


def test_calculate_next_nyse_close_during_trading():
    """Test NYSE close calculation during trading hours."""
    # Friday 2025-10-31 at 10:00 AM ET (14:00 UTC)
    current = datetime(2025, 10, 31, 14, 0, 0, tzinfo=timezone.utc)
    next_close = process_utils._calculate_next_nyse_close(current)

    # Should be same day at 4 PM ET (20:00 UTC)
    expected_et = datetime(2025, 10, 31, 16, 0, 0, tzinfo=ZoneInfo("America/New_York"))
    expected_utc = expected_et.astimezone(timezone.utc)
    assert next_close == expected_utc


def test_calculate_next_nyse_close_after_market():
    """Test NYSE close calculation after market hours."""
    # Friday 2025-10-31 at 5:00 PM ET (21:00 UTC)
    current = datetime(2025, 10, 31, 21, 0, 0, tzinfo=timezone.utc)
    next_close = process_utils._calculate_next_nyse_close(current)

    # Should be Monday at 4 PM ET (skip weekend)
    # November 3, 2025 is a Monday
    expected_et = datetime(2025, 11, 3, 16, 0, 0, tzinfo=ZoneInfo("America/New_York"))
    expected_utc = expected_et.astimezone(timezone.utc)
    assert next_close == expected_utc


def test_calculate_next_nyse_close_on_weekend():
    """Test NYSE close calculation on Saturday."""
    # Saturday 2025-11-01 at 10:00 AM ET
    current = datetime(2025, 11, 1, 14, 0, 0, tzinfo=timezone.utc)
    next_close = process_utils._calculate_next_nyse_close(current)

    # Should be Monday at 4 PM ET
    expected_et = datetime(2025, 11, 3, 16, 0, 0, tzinfo=ZoneInfo("America/New_York"))
    expected_utc = expected_et.astimezone(timezone.utc)
    assert next_close == expected_utc


def test_calculate_market_aware_expiry_crypto():
    """Test market-aware expiry for crypto."""
    # 2:00 AM UTC on Nov 1
    current = datetime(2025, 11, 1, 2, 0, 0, tzinfo=timezone.utc)
    expiry = process_utils._calculate_market_aware_expiry("BTCUSD", current)

    # Should expire at next UTC midnight (Nov 2 00:00)
    expected = datetime(2025, 11, 2, 0, 0, 0, tzinfo=timezone.utc)
    assert expiry == expected


def test_calculate_market_aware_expiry_stock():
    """Test market-aware expiry for stock."""
    # Friday 10:00 AM ET
    current = datetime(2025, 10, 31, 14, 0, 0, tzinfo=timezone.utc)
    expiry = process_utils._calculate_market_aware_expiry("AAPL", current)

    # Should expire at NYSE close (4 PM ET same day)
    expected_et = datetime(2025, 10, 31, 16, 0, 0, tzinfo=ZoneInfo("America/New_York"))
    expected_utc = expected_et.astimezone(timezone.utc)
    assert expiry == expected_utc


def test_calculate_market_aware_expiry_respects_min_duration():
    """Test that expiry respects minimum duration even if market closes sooner."""
    # Very close to market close: 3:50 PM ET
    current = datetime(2025, 10, 31, 19, 50, 0, tzinfo=timezone.utc)
    expiry = process_utils._calculate_market_aware_expiry("AAPL", current, min_duration_minutes=120)

    # Market closes in 10 minutes, but min is 120 minutes
    # So should expire at current + 120 minutes
    expected = current + timedelta(minutes=120)
    assert expiry == expected


def test_is_data_bar_fresh_crypto_safe_window():
    """Test data freshness check for crypto in safe window."""
    # 00:10 UTC - safe (5+ minutes after midnight)
    current = datetime(2025, 11, 1, 0, 10, 0, tzinfo=timezone.utc)
    assert process_utils._is_data_bar_fresh("BTCUSD", current) is True

    # 14:00 UTC - safe (during day)
    current = datetime(2025, 11, 1, 14, 0, 0, tzinfo=timezone.utc)
    assert process_utils._is_data_bar_fresh("BTCUSD", current) is True


def test_is_data_bar_fresh_crypto_too_early():
    """Test data freshness check for crypto too soon after midnight."""
    # 00:02 UTC - too early (less than 5 minutes after midnight)
    current = datetime(2025, 11, 1, 0, 2, 0, tzinfo=timezone.utc)
    assert process_utils._is_data_bar_fresh("BTCUSD", current) is False


def test_is_data_bar_fresh_stock_safe_window():
    """Test data freshness check for stock during trading hours."""
    # Friday 10:00 AM ET (14:00 UTC) - safe
    current = datetime(2025, 10, 31, 14, 0, 0, tzinfo=timezone.utc)
    assert process_utils._is_data_bar_fresh("AAPL", current) is True


def test_is_data_bar_fresh_stock_too_early():
    """Test data freshness check for stock too soon after market open."""
    # Friday 9:32 AM ET (13:32 UTC) - too early (less than 5 min after open)
    current = datetime(2025, 10, 31, 13, 32, 0, tzinfo=timezone.utc)
    assert process_utils._is_data_bar_fresh("AAPL", current) is False


def test_is_data_bar_fresh_stock_weekend():
    """Test data freshness check for stock on weekend."""
    # Saturday
    current = datetime(2025, 11, 1, 14, 0, 0, tzinfo=timezone.utc)
    assert process_utils._is_data_bar_fresh("AAPL", current) is False


def test_spawn_open_proceeds_when_data_not_fresh(tmp_watchers_dir, monkeypatch):
    """Test that spawn proceeds even if data bar is not fresh (with warning)."""
    symbol = "BTCUSD"
    side = "buy"
    limit_price = 100000.0
    target_qty = 1.0

    # Set time to 00:02 UTC (too soon after midnight)
    current_time = datetime(2025, 11, 1, 0, 2, 0, tzinfo=timezone.utc)

    # Mock datetime.now to return our test time
    original_datetime = datetime

    class MockDatetime:
        @classmethod
        def now(cls, tz=None):
            return current_time

        def __getattr__(self, name):
            return getattr(original_datetime, name)

    monkeypatch.setattr(process_utils, "datetime", type("datetime", (), {
        "now": lambda tz=None: current_time,
        "fromisoformat": original_datetime.fromisoformat,
    }))

    spawned = []

    def fake_popen(*args, **kwargs):
        spawned.append((args, kwargs))
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(process_utils.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(process_utils.os, "kill", lambda pid, sig: None)

    # Should spawn even though data is not fresh
    process_utils.spawn_open_position_at_maxdiff_takeprofit(
        symbol, side, limit_price, target_qty
    )

    # Should have spawned (key change: now proceeds instead of blocking)
    assert len(spawned) == 1

    # Verify watcher metadata was created
    suffix = process_utils._format_float(limit_price, 8)
    config_path = process_utils._watcher_config_path(symbol, side, "entry", suffix=suffix)
    assert config_path.exists()
    metadata = json.loads(config_path.read_text())
    assert metadata["symbol"] == symbol
    assert metadata["limit_price"] == limit_price
