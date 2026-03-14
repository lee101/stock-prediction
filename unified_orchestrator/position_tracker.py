"""Crypto position entry-time and peak-price tracker.

Persists when each crypto position was entered so the orchestrator can enforce
a max-hold-hours force exit — mirroring the backtest simulator's behavior.
Also tracks peak price since entry for trailing stop exits.

The JSON state file records the UTC ISO timestamp when a position was first
detected.  Each cycle:
  1. ``update_entry_times`` compares the current Alpaca positions against the
     previous snapshot to detect new fills.
  2. ``get_force_exit_symbols`` returns any symbol whose held time exceeds the
     configured maximum.
  3. ``get_trailing_stop_symbols`` returns any symbol whose current price has
     dropped more than trail_pct% from its peak since entry.

State files:
  strategy_state/crypto_entries.json — {symbol: ISO timestamp}
  strategy_state/crypto_peaks.json   — {symbol: peak_price_float}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

ENTRY_TIMES_FILE = Path("strategy_state/crypto_entries.json")
PEAK_PRICES_FILE = Path("strategy_state/crypto_peaks.json")


def load_entry_times() -> dict[str, str]:
    """Load persisted entry times (symbol -> ISO datetime string)."""
    if ENTRY_TIMES_FILE.exists():
        try:
            return json.loads(ENTRY_TIMES_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_entry_times(entry_times: dict[str, str]) -> None:
    """Persist entry times to disk atomically."""
    ENTRY_TIMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ENTRY_TIMES_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(entry_times, indent=2))
    tmp.replace(ENTRY_TIMES_FILE)


def update_entry_times(
    entry_times: dict[str, str],
    current_positions: dict,  # symbol -> Position
    min_qty: float = 0.0001,
) -> dict[str, str]:
    """Reconcile tracked entry times against the current broker positions.

    - New positions (held now but not tracked) → record entry time as now.
    - Positions that no longer exist → remove from tracker.
    - Existing positions → keep their original entry time.

    Returns the updated entry_times dict (does NOT save — caller must call
    save_entry_times).
    """
    now_str = datetime.now(timezone.utc).isoformat()
    updated: dict[str, str] = {}

    for sym, pos in current_positions.items():
        if getattr(pos, "qty", 0) < min_qty:
            continue  # dust — ignore
        if sym in entry_times:
            updated[sym] = entry_times[sym]  # keep original entry time
        else:
            # New fill detected between last cycle and now
            updated[sym] = now_str
            logger.info(f"  PositionTracker: {sym} new fill detected — entry time = {now_str}")

    # Symbols that dropped out of positions are implicitly removed (not in updated)
    removed = set(entry_times) - set(updated)
    if removed:
        logger.info(f"  PositionTracker: cleared exited positions: {removed}")

    return updated


def get_force_exit_symbols(
    entry_times: dict[str, str],
    max_hold_hours: int,
    now: datetime | None = None,
) -> list[str]:
    """Return symbols whose hold time exceeds max_hold_hours.

    Args:
        entry_times: {symbol: ISO entry timestamp}
        max_hold_hours: Force-exit threshold (matches backtest max_hold_hours).
        now: Override current time (useful for testing).

    Returns:
        List of symbols that should be force-exited at near-market price.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    force_exit: list[str] = []
    for sym, entry_str in entry_times.items():
        try:
            entry_dt = datetime.fromisoformat(entry_str)
            held_hours = (now - entry_dt).total_seconds() / 3600.0
            if held_hours >= max_hold_hours:
                logger.info(
                    f"  PositionTracker: {sym} held {held_hours:.1f}h >= {max_hold_hours}h "
                    f"— FORCE EXIT"
                )
                force_exit.append(sym)
        except Exception as e:
            logger.debug(f"  PositionTracker: bad entry time for {sym}: {e}")

    return force_exit


# ---------------------------------------------------------------------------
# Peak price tracking (for trailing stop)
# ---------------------------------------------------------------------------

def load_peak_prices() -> dict[str, float]:
    """Load persisted peak prices (symbol -> peak price float)."""
    if PEAK_PRICES_FILE.exists():
        try:
            return json.loads(PEAK_PRICES_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_peak_prices(peaks: dict[str, float]) -> None:
    """Persist peak prices to disk atomically."""
    PEAK_PRICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PEAK_PRICES_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(peaks, indent=2))
    tmp.replace(PEAK_PRICES_FILE)


def update_peak_prices(
    peaks: dict[str, float],
    current_positions: dict,  # symbol -> Position (with .current_price)
    min_qty: float = 0.0001,
) -> dict[str, float]:
    """Update peak prices for currently held positions.

    - New positions: initialize peak at current_price.
    - Existing positions: update peak if current_price > stored peak.
    - Exited positions: remove from tracker.
    """
    updated: dict[str, float] = {}
    for sym, pos in current_positions.items():
        if getattr(pos, "qty", 0) < min_qty:
            continue
        current = float(getattr(pos, "current_price", 0))
        if current <= 0:
            continue
        old_peak = peaks.get(sym, 0.0)
        new_peak = max(old_peak, current)
        if old_peak == 0:
            logger.info(f"  PeakTracker: {sym} new position — peak initialized at ${current:.2f}")
        elif new_peak > old_peak:
            logger.info(f"  PeakTracker: {sym} new peak ${new_peak:.2f} (was ${old_peak:.2f})")
        updated[sym] = new_peak
    return updated


def get_trailing_stop_symbols(
    peaks: dict[str, float],
    current_positions: dict,  # symbol -> Position (with .current_price)
    trail_pct: float = 0.3,
    min_qty: float = 0.0001,
) -> list[str]:
    """Return symbols whose current price has dropped trail_pct% from their peak.

    Args:
        peaks: {symbol: peak_price} from update_peak_prices.
        current_positions: current broker positions.
        trail_pct: trailing stop percentage (0.3 = 0.3% from peak).

    Returns:
        List of symbols that should be exited via trailing stop.
    """
    trail_exit: list[str] = []
    for sym, peak in peaks.items():
        pos = current_positions.get(sym)
        if pos is None or getattr(pos, "qty", 0) < min_qty:
            continue
        current = float(getattr(pos, "current_price", 0))
        if current <= 0 or peak <= 0:
            continue
        drop_pct = (peak - current) / peak * 100
        if drop_pct >= trail_pct:
            logger.info(
                f"  PeakTracker: {sym} trailing stop triggered — "
                f"current ${current:.2f} is {drop_pct:.2f}% below peak ${peak:.2f} "
                f"(threshold {trail_pct}%)"
            )
            trail_exit.append(sym)
    return trail_exit
