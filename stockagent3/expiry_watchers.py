"""Expiry watcher helpers for stockagent3 trade plans."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from loguru import logger

from src.symbol_utils import is_crypto_symbol
from stock.state import get_state_dir, resolve_state_suffix, get_paper_suffix

UTC = ZoneInfo("UTC")
NEW_YORK = ZoneInfo("America/New_York")

STATE_SUFFIX = resolve_state_suffix()
PAPER_SUFFIX = get_paper_suffix()
EXPIRY_WATCHERS_DIR = get_state_dir() / f"expiry_watchers{PAPER_SUFFIX}{STATE_SUFFIX or ''}"
EXPIRY_WATCHERS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ExpiryWatcherConfig:
    symbol: str
    side: str
    expiry_at: datetime
    strategy: str
    reason: str


def _sanitize(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _watcher_config_path(symbol: str, side: str, strategy: str) -> Path:
    safe_symbol = _sanitize(symbol)
    safe_side = _sanitize(side)
    safe_strategy = _sanitize(strategy or "stockagent3")
    return EXPIRY_WATCHERS_DIR / f"{safe_symbol}_{safe_side}_expiry_{safe_strategy}.json"


def _persist_metadata(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        temp_path.replace(path)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to persist expiry metadata %s: %s", path, exc)


def _load_metadata(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to read expiry metadata %s: %s", path, exc)
        return None


def _is_pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def stop_expiry_watchers(
    *,
    symbol: str,
    side: Optional[str] = None,
    strategy: str = "stockagent3",
    reason: str = "reset",
) -> None:
    symbol = symbol.upper()
    safe_symbol = _sanitize(symbol)
    pattern = f"{safe_symbol}_*_expiry_{_sanitize(strategy)}.json" if side is None else f"{safe_symbol}_{_sanitize(side)}_expiry_{_sanitize(strategy)}.json"
    for path in EXPIRY_WATCHERS_DIR.glob(pattern):
        metadata = _load_metadata(path) or {}
        pid_raw = metadata.get("pid")
        pid = None
        try:
            pid = int(pid_raw)
        except (TypeError, ValueError):
            pid = None
        if pid and _is_pid_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
                logger.info("Terminated expiry watcher %s (pid=%s)", path.name, pid)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to terminate expiry watcher %s: %s", path.name, exc)
        if metadata:
            metadata["active"] = False
            metadata["state"] = reason
            metadata["terminated_at"] = datetime.now(timezone.utc).isoformat()
            _persist_metadata(path, metadata)


def list_expiry_watchers(strategy: str = "stockagent3") -> list[dict]:
    safe_strategy = _sanitize(strategy or "stockagent3")
    watchers: list[dict] = []
    for path in EXPIRY_WATCHERS_DIR.glob(f"*_expiry_{safe_strategy}.json"):
        metadata = _load_metadata(path)
        if metadata:
            watchers.append(metadata)
    return watchers


def compute_expiry_at(symbol: str, hold_days: int, now: Optional[datetime] = None) -> datetime:
    hold_days = max(1, min(8, int(hold_days)))
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if is_crypto_symbol(symbol):
        return now_utc + timedelta(days=hold_days)

    target = now_utc + timedelta(days=hold_days)
    target_et = target.astimezone(NEW_YORK)
    expiry_et = target_et.replace(hour=16, minute=0, second=0, microsecond=0)
    if expiry_et <= target_et:
        expiry_et += timedelta(days=1)
    while expiry_et.weekday() >= 5:
        expiry_et += timedelta(days=1)
        expiry_et = expiry_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return expiry_et.astimezone(timezone.utc)


def spawn_expiry_watcher(config: ExpiryWatcherConfig) -> None:
    symbol = config.symbol.upper()
    side = config.side.lower()
    strategy = config.strategy or "stockagent3"
    expiry_at = config.expiry_at.astimezone(timezone.utc)

    stop_expiry_watchers(symbol=symbol, side=side, strategy=strategy, reason="superseded")
    config_path = _watcher_config_path(symbol, side, strategy)

    metadata = {
        "config_version": 1,
        "mode": "expiry",
        "symbol": symbol,
        "side": side,
        "strategy": strategy,
        "expiry_at": expiry_at.isoformat(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "reason": config.reason,
        "active": True,
        "state": "pending_launch",
        "config_path": str(config_path),
    }
    _persist_metadata(config_path, metadata)

    python_bin = os.environ.get("PYTHON", None) or os.sys.executable or "python"
    command = (
        f"{python_bin} scripts/stockagent3_expiry_watcher.py"
        f" --symbol={symbol}"
        f" --side={side}"
        f" --expiry-at={expiry_at.isoformat()}"
        f" --strategy={strategy}"
        f" --config-path={config_path}"
        f" --reason={config.reason}"
    )

    logger.info("Running expiry watcher command %s", command)
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except Exception as exc:
        metadata["state"] = "launch_failed"
        metadata["active"] = False
        metadata["error"] = str(exc)
        metadata["last_update"] = datetime.now(timezone.utc).isoformat()
        _persist_metadata(config_path, metadata)
        raise
    else:
        metadata["pid"] = process.pid
        metadata["state"] = "launched"
        metadata["last_update"] = datetime.now(timezone.utc).isoformat()
        _persist_metadata(config_path, metadata)


__all__ = [
    "ExpiryWatcherConfig",
    "compute_expiry_at",
    "spawn_expiry_watcher",
    "stop_expiry_watchers",
    "list_expiry_watchers",
    "EXPIRY_WATCHERS_DIR",
]
