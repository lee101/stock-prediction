from __future__ import annotations

import atexit
import json
import os
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from unified_orchestrator.state_paths import resolve_state_dir

try:
    import fcntl
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("alpaca_account_lock requires fcntl support") from exc


@dataclass
class AlpacaAccountLock:
    service_name: str
    account_name: str
    path: Path
    handle: object
    registry_key: str
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            try:
                self.handle.close()
            except Exception:
                pass
            held = _HELD_LOCKS.get(self.registry_key)
            if held is self:
                _HELD_LOCKS.pop(self.registry_key, None)


def _lock_payload(service_name: str, account_name: str) -> dict[str, object]:
    return {
        "service_name": service_name,
        "account_name": account_name,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cmdline": list(sys.argv),
    }


def _env_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def require_explicit_live_trading_enable(
    service_name: str,
    *,
    env_var: str = "ALLOW_ALPACA_LIVE_TRADING",
) -> None:
    """Fail closed unless live Alpaca trading is explicitly enabled."""

    if _env_truthy(os.getenv(env_var)):
        return
    raise RuntimeError(
        f"Refusing live Alpaca trading for {service_name}: {env_var}=1 is required. "
        "Repo is currently in paper-first safety mode."
    )


def _lock_dir(state_dir: str | Path | None = None) -> Path:
    return resolve_state_dir(state_dir) / "account_locks"


def lock_path_for_account(account_name: str, *, state_dir: str | Path | None = None) -> Path:
    safe_account = str(account_name).strip().lower().replace("/", "_").replace(" ", "_")
    return _lock_dir(state_dir) / f"{safe_account}.lock"


# Per-process idempotency: once this process has acquired a given account
# lock, subsequent calls for the same path return the existing handle. This
# lets alpaca_wrapper acquire at import time AND trade_daily_stock_prod
# call acquire_alpaca_account_lock later without the second call racing
# itself on the same fcntl file descriptor.
_HELD_LOCKS: dict[str, "AlpacaAccountLock"] = {}


def _active_held_lock(registry_key: str) -> AlpacaAccountLock | None:
    existing = _HELD_LOCKS.get(registry_key)
    if existing is None:
        return None
    if existing.released:
        _HELD_LOCKS.pop(registry_key, None)
        return None
    return existing


def acquire_alpaca_account_lock(
    service_name: str,
    *,
    account_name: str = "alpaca_live_writer",
    state_dir: str | Path | None = None,
) -> AlpacaAccountLock:
    """Acquire a non-blocking exclusive lock for an Alpaca account writer.

    The lock is process-scoped and released automatically on process exit.
    Idempotent: if this process already holds the same lock, the existing
    ``AlpacaAccountLock`` is returned and no new fd is opened.
    """

    lock_path = lock_path_for_account(account_name, state_dir=state_dir)
    key = str(lock_path.resolve())
    existing = _active_held_lock(key)
    if existing is not None:
        if existing.service_name == service_name:
            return existing
        raise RuntimeError(
            "Alpaca account writer lock is already held in-process: "
            f"account={account_name} path={lock_path} holder_service={existing.service_name} "
            f"holder_pid={os.getpid()} holder_host={socket.gethostname()}"
        )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        holder_raw = handle.read().strip()
        try:
            holder = json.loads(holder_raw) if holder_raw else {}
        except Exception:
            holder = {"raw": holder_raw}
        holder_service = holder.get("service_name", "unknown")
        holder_pid = holder.get("pid", "unknown")
        holder_host = holder.get("hostname", "unknown")
        holder_started = holder.get("started_at", "unknown")
        handle.close()
        raise RuntimeError(
            "Alpaca account writer lock is already held: "
            f"account={account_name} path={lock_path} holder_service={holder_service} "
            f"holder_pid={holder_pid} holder_host={holder_host} holder_started_at={holder_started}"
        ) from exc

    payload = _lock_payload(service_name, account_name)
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(payload, sort_keys=True))
    handle.flush()
    os.fsync(handle.fileno())

    lock = AlpacaAccountLock(
        service_name=service_name,
        account_name=account_name,
        path=lock_path,
        handle=handle,
        registry_key=key,
    )
    _HELD_LOCKS[key] = lock
    atexit.register(lock.release)
    return lock
