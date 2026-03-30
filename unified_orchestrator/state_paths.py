from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def resolve_state_dir(state_dir: str | Path | None = None) -> Path:
    """Resolve the strategy state directory from an explicit path or environment."""

    raw_path = state_dir if state_dir is not None else os.environ.get("STATE_DIR", "strategy_state")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO / path
    return path


def fill_events_file_path(state_dir: str | Path | None = None) -> Path:
    return resolve_state_dir(state_dir) / "fill_events.jsonl"


def stock_event_log_path(state_dir: str | Path | None = None) -> Path:
    return resolve_state_dir(state_dir) / "stock_event_log.jsonl"


def cycle_event_log_path(state_dir: str | Path | None = None) -> Path:
    return resolve_state_dir(state_dir) / "orchestrator_cycle_events.jsonl"
