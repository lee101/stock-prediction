"""Per-signal position cap tracker.

Prevents position stacking by freezing the maximum allowed position for each
symbol/side when the first trading signal fires.  Subsequent watcher respawns
or hourly cycles that compute a *higher* target will be capped to the frozen
value.  The cap expires after ``max_age_hours`` so the model can set a fresh
cap on the next signal cycle.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CAPS_DIR = Path(__file__).resolve().parent.parent / "data" / "position_caps"


def _cap_path(symbol: str, side: str) -> Path:
    """Return the JSON path for a given symbol/side cap."""
    _CAPS_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{symbol.upper().replace('/', '')}_{side.lower()}"
    return _CAPS_DIR / f"{key}.json"


def set_position_cap(
    symbol: str,
    side: str,
    max_qty: float,
    buy_signal_qty: float,
) -> None:
    """Store the max allowed position for this signal cycle."""
    path = _cap_path(symbol, side)
    data = {
        "symbol": symbol.upper(),
        "side": side.lower(),
        "max_qty": float(max_qty),
        "buy_signal_qty": float(buy_signal_qty),
        "set_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(data, indent=2))
    logger.debug(
        "Position cap set for %s %s: max_qty=%.6f signal_qty=%.6f",
        symbol, side, max_qty, buy_signal_qty,
    )


def get_position_cap(
    symbol: str,
    side: str,
    max_age_hours: float = 2.0,
) -> Optional[float]:
    """Get the frozen cap.  Returns ``None`` if expired or not set."""
    path = _cap_path(symbol, side)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    set_at_str = data.get("set_at")
    if not set_at_str:
        return None
    try:
        set_at = datetime.fromisoformat(set_at_str)
    except (ValueError, TypeError):
        return None
    if set_at.tzinfo is None:
        set_at = set_at.replace(tzinfo=timezone.utc)

    age_hours = (datetime.now(timezone.utc) - set_at).total_seconds() / 3600.0
    if age_hours > max_age_hours:
        return None

    max_qty = data.get("max_qty")
    if max_qty is None or float(max_qty) <= 0:
        return None
    return float(max_qty)


def clear_position_cap(symbol: str) -> None:
    """Clear caps for all sides of a symbol."""
    for side in ("buy", "sell"):
        path = _cap_path(symbol, side)
        if path.exists():
            try:
                path.unlink()
                logger.debug("Cleared position cap for %s %s", symbol, side)
            except OSError:
                pass
