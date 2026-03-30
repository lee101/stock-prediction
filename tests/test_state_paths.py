from __future__ import annotations

from pathlib import Path

from unified_orchestrator.state_paths import (
    cycle_event_log_path,
    fill_events_file_path,
    resolve_state_dir,
    stock_event_log_path,
)


def test_resolve_state_dir_uses_runtime_env_without_module_reload(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("STATE_DIR", str(tmp_path))

    assert resolve_state_dir() == tmp_path
    assert fill_events_file_path() == tmp_path / "fill_events.jsonl"
    assert stock_event_log_path() == tmp_path / "stock_event_log.jsonl"
    assert cycle_event_log_path() == tmp_path / "orchestrator_cycle_events.jsonl"


def test_resolve_state_dir_keeps_relative_paths_under_repo(monkeypatch):
    monkeypatch.setenv("STATE_DIR", ".cache/custom_state")

    resolved = resolve_state_dir()

    assert resolved.name == "custom_state"
    assert resolved.is_absolute()
    assert fill_events_file_path().parent == resolved
