from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict

STATE_DIRNAME = "strategy_state"


@lru_cache(maxsize=1)
def get_state_dir() -> Path:
    """Location for persistent trading state artifacts."""
    return Path(__file__).resolve().parents[1] / STATE_DIRNAME


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
