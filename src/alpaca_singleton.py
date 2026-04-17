"""Process-wide singleton guard for LIVE Alpaca trading.

Fires at import-time inside ``alpaca_wrapper`` so that every process which
goes anywhere near a live Alpaca write API has to go through this gate. The
guard:

1. **Refuses to start a 2nd live process on the same account.** Uses the
   existing ``src.alpaca_account_lock`` fcntl-based writer lock. Any
   attempt to import ``alpaca_wrapper`` in LIVE mode while another live
   process holds the lock raises ``SystemExit(42)`` with a loud message
   naming the holder PID / hostname / start time. Paper mode skips the
   guard entirely — any number of paper processes may run.

2. **Blocks death-spiral sells.** Before any order is submitted, callers
   pass through ``guard_sell_against_death_spiral(symbol, side, price)``.
   If ``side==sell`` and the price is more than
   ``DEATH_SPIRAL_TOLERANCE_BPS`` below the most recent recorded buy for
   the same symbol, the order is REFUSED. Buy-prices are tracked on disk
   so the guard survives process restarts.

Both gates are defeatable by an explicit environment variable override
(``ALPACA_SINGLETON_OVERRIDE=1`` / ``ALPACA_DEATH_SPIRAL_OVERRIDE=1``) —
defeats are logged loudly on every call so they show up in any
post-mortem grep.

The module is intentionally tiny: no trading client dependencies, so it
can be imported from anywhere (tests, CI, alpaca_wrapper) without network
or heavyweight deps.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Optional

from src.alpaca_account_lock import (
    AlpacaAccountLock,
    acquire_alpaca_account_lock,
    lock_path_for_account,
    normalize_alpaca_account_name,
)


# Default tolerance: refuse a sell if it's priced more than this many bps
# BELOW the most recent buy for the symbol. 50 bps = 0.5% of buy price —
# generous enough to accommodate spread/slippage but tight enough to catch
# runaway loops that keep slashing the ask.
DEFAULT_DEATH_SPIRAL_TOLERANCE_BPS = 50.0

# Buy-price memory window. Older entries are pruned so a week-old buy
# doesn't block a legitimate market-driven sell.
DEFAULT_BUY_MEMORY_SECONDS = 60 * 60 * 24 * 3  # 3 days
DEFAULT_ALPACA_ACCOUNT_NAME = "alpaca_live_writer"
SINGLETON_OVERRIDE_ENV_VAR = "ALPACA_SINGLETON_OVERRIDE"
DEATH_SPIRAL_OVERRIDE_ENV_VAR = "ALPACA_DEATH_SPIRAL_OVERRIDE"


# ---------------------------------------------------------------------------
# State paths + constants
# ---------------------------------------------------------------------------


def _state_dir() -> Path:
    """Resolve a process-stable state directory for the guard."""
    from unified_orchestrator.state_paths import resolve_state_dir
    return resolve_state_dir(None) / "alpaca_singleton"


def _buy_memory_path(account_name: str) -> Path:
    safe = normalize_alpaca_account_name(account_name)
    return _state_dir() / f"{safe}_buys.json"


def _buy_memory_file_lock_path(account_name: str) -> Path:
    safe = normalize_alpaca_account_name(account_name)
    return _state_dir() / f"{safe}_buys.lock"


def _env_flag(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _current_account_name() -> str:
    with _STATE_MU:
        state = _STATE
    if state is None:
        return DEFAULT_ALPACA_ACCOUNT_NAME
    return state.account_name


def _current_buy_memory_seconds() -> int:
    with _STATE_MU:
        state = _STATE
    if state is None:
        return DEFAULT_BUY_MEMORY_SECONDS
    return state.buy_memory_seconds


# ---------------------------------------------------------------------------
# Singleton writer lock
# ---------------------------------------------------------------------------


@dataclass
class SingletonState:
    account_name: str
    lock: Optional[AlpacaAccountLock] = None
    death_spiral_guard_enabled: bool = True
    death_spiral_tolerance_bps: float = DEFAULT_DEATH_SPIRAL_TOLERANCE_BPS
    buy_memory_seconds: int = DEFAULT_BUY_MEMORY_SECONDS
    _mu: RLock = field(default_factory=RLock)


_STATE: Optional[SingletonState] = None
_STATE_MU = RLock()
_BUY_MEMORY_LOCKS: dict[str, RLock] = {}
_BUY_MEMORY_LOCKS_MU = RLock()


def _write_marker(where: str, msg: str) -> None:
    """Loud marker into stderr so any override / failure shows up in logs."""
    print(f"[alpaca_singleton] {where}: {msg}", file=sys.stderr, flush=True)


def _buy_memory_lock(account_name: str) -> RLock:
    safe_account = normalize_alpaca_account_name(account_name)
    with _BUY_MEMORY_LOCKS_MU:
        existing = _BUY_MEMORY_LOCKS.get(safe_account)
        if existing is not None:
            return existing
        created = RLock()
        _BUY_MEMORY_LOCKS[safe_account] = created
        return created


@contextmanager
def _buy_memory_guard(account_name: str):
    lock = _buy_memory_lock(account_name)
    lock_path = _buy_memory_file_lock_path(account_name)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def enforce_live_singleton(
    *,
    service_name: str,
    account_name: str = DEFAULT_ALPACA_ACCOUNT_NAME,
) -> Optional[AlpacaAccountLock]:
    """Acquire the live-writer lock. Paper mode is a no-op.

    Call this once at startup (``alpaca_wrapper`` does it automatically at
    import time). Returns the ``AlpacaAccountLock`` handle on success, or
    ``None`` on paper. Raises ``SystemExit(42)`` on contention.

    The override ``ALPACA_SINGLETON_OVERRIDE=1`` disables the lock but
    logs every time it does so, so a catastrophic override leaves a loud
    trail in journald.
    """
    global _STATE

    # Paper mode: no singleton required — unlimited paper instances OK.
    try:
        from env_real import PAPER
    except Exception:
        PAPER = True
    if PAPER:
        _write_marker("enforce_live_singleton",
                      f"paper mode, no lock (service={service_name})")
        with _STATE_MU:
            if _STATE is None:
                _STATE = SingletonState(account_name=account_name, lock=None)
        return None

    if _env_flag(SINGLETON_OVERRIDE_ENV_VAR):
        _write_marker("enforce_live_singleton",
                      f"OVERRIDE ACTIVE — running live without singleton lock "
                      f"(service={service_name}, account={account_name})")
        with _STATE_MU:
            if _STATE is None:
                _STATE = SingletonState(account_name=account_name, lock=None)
        return None

    try:
        lock = acquire_alpaca_account_lock(
            service_name=service_name, account_name=account_name,
        )
    except RuntimeError as exc:
        # Lock held by another live process. Fail CLOSED hard — SystemExit
        # (not RuntimeError) so the server can't catch-and-continue past it.
        _write_marker(
            "enforce_live_singleton",
            f"REFUSING TO START — live singleton already held: {exc}",
        )
        raise SystemExit(42) from exc

    _write_marker(
        "enforce_live_singleton",
        f"lock acquired: service={service_name} account={account_name} "
        f"path={lock_path_for_account(account_name)}",
    )
    with _STATE_MU:
        _STATE = SingletonState(account_name=account_name, lock=lock)
    return lock


# ---------------------------------------------------------------------------
# Death-spiral guard
# ---------------------------------------------------------------------------


def _load_buys(account_name: str) -> dict:
    path = _buy_memory_path(account_name)
    if not path.exists():
        return {}
    try:
        raw = path.read_text()
    except Exception as exc:
        _write_marker(
            "_load_buys",
            f"failed to read buy memory: account={account_name} path={path} error={exc}",
        )
        return {}
    try:
        data = json.loads(raw)
    except Exception as exc:
        quarantine_path = _quarantine_buy_memory_file(path)
        _write_marker(
            "_load_buys",
            "CORRUPT BUY MEMORY ignored: "
            f"account={account_name} path={path} quarantine={quarantine_path} error={exc}",
        )
        return {}
    if not isinstance(data, dict):
        quarantine_path = _quarantine_buy_memory_file(path)
        _write_marker(
            "_load_buys",
            "INVALID BUY MEMORY ignored: "
            f"account={account_name} path={path} quarantine={quarantine_path} "
            f"top_level_type={type(data).__name__}",
        )
        return {}
    return data


def _quarantine_buy_memory_file(path: Path) -> Path | None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine_path = path.with_name(f"{path.stem}.corrupt-{timestamp}{path.suffix}")
    try:
        shutil.move(str(path), str(quarantine_path))
    except Exception as exc:
        _write_marker(
            "_quarantine_buy_memory_file",
            f"failed to quarantine buy memory: path={path} quarantine={quarantine_path} error={exc}",
        )
        return None
    return quarantine_path


def _save_buys(account_name: str, data: dict) -> None:
    path = _buy_memory_path(account_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{path.stem}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(data, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _prune_buys(data: dict, max_age_seconds: int) -> dict:
    now = time.time()
    out: dict = {}
    for sym, rec in data.items():
        if not isinstance(rec, dict):
            continue
        ts = float(rec.get("ts", 0))
        if now - ts <= max_age_seconds:
            out[sym] = rec
    return out


def record_buy_price(symbol: str, price: float) -> None:
    """Remember that we bought `symbol` at `price` so the death-spiral guard
    can refuse sells that are far below it. Safe to call from paper mode."""
    if not symbol or price <= 0 or not _finite(price):
        return
    account_name = _current_account_name()
    with _buy_memory_guard(account_name):
        data = _load_buys(account_name)
        data = _prune_buys(data, _current_buy_memory_seconds())
        data[str(symbol).upper()] = {
            "price": float(price),
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat(),
        }
        _save_buys(account_name, data)


def _finite(x: float) -> bool:
    try:
        return x == x and x not in (float("inf"), float("-inf"))
    except Exception:
        return False


def guard_sell_against_death_spiral(
    symbol: str,
    side: str,
    price: float,
    *,
    tolerance_bps: Optional[float] = None,
) -> None:
    """Raise RuntimeError if the sell would put us below the remembered buy.

    Raises:
      RuntimeError — refused sell (sellers in a death spiral). Callers
      should let this propagate and crash the daemon; the systemd unit
      will mark the service as failed, which is the desired behaviour
      (loud manual re-enable only).
    """
    if not side:
        return
    side_norm = str(side).strip().lower()
    if side_norm not in ("sell", "short"):
        return
    if not symbol or not _finite(price) or price <= 0:
        return

    with _STATE_MU:
        state = _STATE
    account_name = _current_account_name()
    tol_bps = (
        float(tolerance_bps)
        if tolerance_bps is not None
        else (state.death_spiral_tolerance_bps if state is not None
              else DEFAULT_DEATH_SPIRAL_TOLERANCE_BPS)
    )

    if _env_flag(DEATH_SPIRAL_OVERRIDE_ENV_VAR):
        _write_marker(
            "guard_sell_against_death_spiral",
            f"OVERRIDE ACTIVE — allowing sell of {symbol} at {price} "
            f"without price check",
        )
        return

    with _buy_memory_guard(account_name):
        data = _load_buys(account_name)
        pruned = _prune_buys(data, _current_buy_memory_seconds())
        if pruned != data:
            _save_buys(account_name, pruned)
        data = pruned
        rec = data.get(str(symbol).upper())
        if rec is None:
            # No recent buy on record — nothing to compare against. Allow
            # (this is the common case for fresh positions or hand-deposited
            # shares; the guard's job is to prevent *round-trip* death spirals).
            return

        buy_price = float(rec.get("price", 0.0))
        if buy_price <= 0:
            return

        floor = buy_price * (1.0 - float(tol_bps) / 10000.0)
        if price < floor:
            _write_marker(
                "guard_sell_against_death_spiral",
                f"REFUSING SELL of {symbol} at {price:.4f}: "
                f"last buy at {buy_price:.4f}, floor {floor:.4f} "
                f"(tolerance {tol_bps} bps). Set ALPACA_DEATH_SPIRAL_OVERRIDE=1 "
                f"to bypass. This refusal raises RuntimeError to stop the loop.",
            )
            raise RuntimeError(
                f"alpaca_singleton: refusing death-spiral SELL of {symbol} "
                f"at {price} (last buy {buy_price}, floor {floor}, "
                f"tolerance {tol_bps} bps)"
            )


# ---------------------------------------------------------------------------
# Convenience: clear buy memory (used by tests + operators after a manual
# reset).
# ---------------------------------------------------------------------------


def forget_buy(symbol: str) -> None:
    account_name = _current_account_name()
    with _buy_memory_guard(account_name):
        data = _load_buys(account_name)
        data.pop(str(symbol).upper(), None)
        _save_buys(account_name, data)


def forget_all_buys() -> None:
    account_name = _current_account_name()
    with _buy_memory_guard(account_name):
        _save_buys(account_name, {})


__all__ = [
    "enforce_live_singleton",
    "record_buy_price",
    "guard_sell_against_death_spiral",
    "forget_buy",
    "forget_all_buys",
    "DEFAULT_DEATH_SPIRAL_TOLERANCE_BPS",
    "DEFAULT_BUY_MEMORY_SECONDS",
]
