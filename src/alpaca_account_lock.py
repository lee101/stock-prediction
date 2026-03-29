"""Alpaca account locking and live-trading guardrails.

Provides:
- require_explicit_live_trading_enable: env-var gated live-trading guardrail.
- acquire_alpaca_account_lock: acquire a per-account advisory file lock so that
  only one writer bot can trade a given Alpaca account at a time.
"""

from __future__ import annotations

import fcntl
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}

# Directory under which lock files are stored.
_LOCKS_DIR = Path(os.environ.get("ALPACA_LOCKS_DIR", "/tmp/.alpaca_locks"))


def _is_truthy_env(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY_ENV_VALUES


def require_explicit_live_trading_enable(bot_name: str, env_var: str = "ALPACA_ENABLE_LIVE_TRADING") -> None:
    """Guardrail: raise SystemExit unless live trading is explicitly enabled.

    Args:
        bot_name: Human-readable name of the bot (used in the error message).
        env_var:  Environment variable that must be set to a truthy value.
    """
    if _is_truthy_env(env_var, default=False):
        return
    raise SystemExit(
        f"Alpaca live trading for '{bot_name}' is disabled by default. "
        f"To enable intentionally: set {env_var}=1 and re-run with --live."
    )


class AlpacaAccountLock:
    """Advisory file lock that prevents concurrent writes to a single Alpaca account.

    Attributes:
        path: Path of the lock file that is currently held.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None

    # ------------------------------------------------------------------
    # Context-manager protocol (optional – can also be used standalone)
    # ------------------------------------------------------------------

    def __enter__(self) -> "AlpacaAccountLock":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Explicit release
    # ------------------------------------------------------------------

    def release(self) -> None:
        if self._handle is not None:
            try:
                fcntl.flock(self._handle, fcntl.LOCK_UN)
                self._handle.close()
            except Exception:
                pass
            finally:
                self._handle = None


def acquire_alpaca_account_lock(
    bot_name: str,
    *,
    account_name: str = "default",
    locks_dir: Path | None = None,
) -> AlpacaAccountLock:
    """Acquire an exclusive advisory file lock for the given Alpaca account.

    Only one process/bot may hold the lock for a given *account_name* at any
    time.  The lock is released when the returned :class:`AlpacaAccountLock`
    object is garbage-collected, its :meth:`~AlpacaAccountLock.release` method
    is called, or it is used as a context manager.

    Args:
        bot_name:    Human-readable identifier for the bot acquiring the lock.
        account_name: Logical name of the Alpaca account (e.g. ``"alpaca_live_writer"``).
        locks_dir:   Optional override for the directory that holds lock files.

    Returns:
        An :class:`AlpacaAccountLock` whose ``.path`` attribute is the path of
        the acquired lock file.

    Raises:
        RuntimeError: If the lock cannot be acquired (e.g. because another
            process already holds it).
    """
    base_dir: Path = locks_dir or _LOCKS_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    safe_account = "".join(c if c.isalnum() else "_" for c in account_name)
    lock_path = base_dir / f"{safe_account}.lock"

    lock = AlpacaAccountLock(lock_path)
    handle = lock_path.open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"Could not acquire Alpaca account lock for '{account_name}' "
            f"(bot='{bot_name}'). Another process may already hold it: {lock_path}"
        ) from exc

    handle.write(f"bot={bot_name}\n")
    handle.flush()
    lock._handle = handle
    logger.info("Acquired Alpaca account lock: bot=%s account=%s path=%s", bot_name, account_name, lock_path)
    return lock
