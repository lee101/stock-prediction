from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict

STATE_DIRNAME = "strategy_state"


def is_paper_mode() -> bool:
    """Check if running in paper trading mode based on PAPER env var."""
    return os.getenv("PAPER", "1") not in {"0", "false", "FALSE"}


def get_paper_suffix() -> str:
    """Return '_paper' or '_live' suffix based on PAPER mode."""
    return "_paper" if is_paper_mode() else "_live"


@lru_cache(maxsize=16)
def _state_dir_for(override: str | None) -> Path:
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[1] / STATE_DIRNAME


def get_state_dir() -> Path:
    """Location for persistent trading state artifacts.

    Honours the `STATE_DIR` environment variable when set so tests and
    deployments can isolate state without mutating the repo checkout.
    """
    return _state_dir_for(os.getenv("STATE_DIR") or None)


# Backwards-compatible access for tests/utilities which previously relied on
# `functools.lru_cache` being applied directly to `get_state_dir`.
get_state_dir.cache_clear = _state_dir_for.cache_clear  # type: ignore[attr-defined]


def resolve_state_suffix(raw_suffix: str | None = None) -> str:
    """Normalise the trade state suffix used for FlatShelf files."""
    suffix = (raw_suffix if raw_suffix is not None else os.getenv("TRADE_STATE_SUFFIX", "")).strip()
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    return suffix


def get_state_file(name: str, suffix: str | None = None, extension: str = ".json") -> Path:
    """Return the fully-qualified path for a named state file."""
    resolved_suffix = resolve_state_suffix(suffix)
    filename = f"{name}{resolved_suffix}{extension}"
    return get_state_dir() / filename


def get_default_state_paths(suffix: str | None = None) -> Dict[str, Path]:
    """Convenience helper yielding the canonical state file layout."""
    return {
        "trade_outcomes": get_state_file("trade_outcomes", suffix),
        "trade_learning": get_state_file("trade_learning", suffix),
        "active_trades": get_state_file("active_trades", suffix),
        "trade_history": get_state_file("trade_history", suffix),
    }


def ensure_state_dir() -> None:
    """Create the state directory if missing."""
    get_state_dir().mkdir(parents=True, exist_ok=True)
