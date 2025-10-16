from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from jsonshelve import FlatShelf

from stock.state import get_default_state_paths, resolve_state_suffix

STATE_KEY_SEPARATOR = "|"


class StateLoadError(RuntimeError):
    """Raised when persisted trading state cannot be loaded."""


def _load_flatshelf(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        shelf = FlatShelf(str(path))
        shelf.load()
        return dict(shelf.data)
    except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover - rare but critical
        raise StateLoadError(f"Failed reading state file '{path}': {exc}") from exc


def _parse_state_key(key: str) -> Tuple[str, str]:
    if STATE_KEY_SEPARATOR in key:
        symbol, side = key.split(STATE_KEY_SEPARATOR, 1)
        return symbol, side
    return key, "buy"


def load_all_state(suffix: str | None = None) -> Dict[str, Dict[str, Any]]:
    paths = get_default_state_paths(suffix)
    return {name: _load_flatshelf(path) for name, path in paths.items()}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso_to_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class ProbeStatus:
    symbol: str
    side: str
    pending_probe: bool
    probe_active: bool
    last_pnl: Optional[float]
    last_reason: Optional[str]
    last_closed_at: Optional[datetime]
    active_mode: Optional[str]
    active_qty: Optional[float]
    active_opened_at: Optional[datetime]
    learning_updated_at: Optional[datetime]


def collect_probe_statuses(suffix: str | None = None) -> List[ProbeStatus]:
    state_suffix = resolve_state_suffix(suffix)
    state = load_all_state(state_suffix)
    learning = state.get("trade_learning", {})
    outcomes = state.get("trade_outcomes", {})
    active = state.get("active_trades", {})

    keys: Iterable[str] = set(learning) | set(outcomes) | set(active)
    statuses: List[ProbeStatus] = []

    for key in sorted(keys):
        symbol, side = _parse_state_key(key)
        learning_state = learning.get(key, {})
        outcome_state = outcomes.get(key, {})
        active_state = active.get(key, {})

        statuses.append(
            ProbeStatus(
                symbol=symbol,
                side=side,
                pending_probe=bool(learning_state.get("pending_probe")),
                probe_active=bool(learning_state.get("probe_active")),
                last_pnl=_safe_float(outcome_state.get("pnl")),
                last_reason=outcome_state.get("reason"),
                last_closed_at=_iso_to_datetime(outcome_state.get("closed_at")),
                active_mode=active_state.get("mode"),
                active_qty=_safe_float(active_state.get("qty")),
                active_opened_at=_iso_to_datetime(active_state.get("opened_at")),
                learning_updated_at=_iso_to_datetime(learning_state.get("updated_at")),
            )
        )

    return statuses


def render_ascii_line(values: List[float], width: int = 60) -> List[str]:
    """Render a simple ASCII bar chart for CLI display."""
    if not values:
        return []

    if len(values) > width:
        step = len(values) / width
        downsampled = []
        idx = 0.0
        while len(downsampled) < width and int(idx) < len(values):
            downsampled.append(values[int(idx)])
            idx += step
        values = downsampled

    min_val = min(values)
    max_val = max(values)
    if min_val == max_val:
        return ["#" * len(values)]

    palette = " .:-=+*#%@"
    divisor = max_val - min_val
    line = []
    for value in values:
        normalized = 0.0 if divisor == 0 else (value - min_val) / divisor
        index = min(len(palette) - 1, max(0, int(normalized * (len(palette) - 1))))
        line.append(palette[index])
    return ["".join(line)]
