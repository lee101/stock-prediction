from __future__ import annotations

import json
import os
import signal
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

from stock.state import get_state_dir, resolve_state_suffix, get_paper_suffix
from src.binan import binance_wrapper
from src.stock_utils import binance_remap_symbols


STATE_SUFFIX = resolve_state_suffix()
PAPER_SUFFIX = get_paper_suffix()
WATCHERS_DIR = get_state_dir() / f"binanceneural_watchers{PAPER_SUFFIX}{STATE_SUFFIX or ''}"
WATCHERS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class WatcherPlan:
    symbol: str
    side: str
    mode: str
    limit_price: float
    target_qty: float
    expiry_minutes: int
    poll_seconds: int
    price_tolerance: float = 0.0
    dry_run: bool = False


def _sanitize(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def watcher_config_path(symbol: str, side: str, mode: str, *, suffix: Optional[str] = None) -> Path:
    safe_symbol = _sanitize(symbol)
    safe_side = _sanitize(side)
    base_name = f"{safe_symbol}_{safe_side}_{mode}"
    if suffix:
        base_name = f"{base_name}_{_sanitize(suffix)}"
    return WATCHERS_DIR / f"{base_name}.json"


def _load_metadata(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to read watcher metadata %s: %s", path, exc)
        return None


def _persist_metadata(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        temp_path.replace(path)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to persist watcher metadata %s: %s", path, exc)


def _is_pid_alive(pid: Optional[int]) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


def _cancel_order(metadata: dict) -> None:
    order_id = metadata.get("order_id")
    symbol = metadata.get("binance_symbol")
    if not order_id or not symbol:
        return
    try:
        client = binance_wrapper.get_client()
        client.cancel_order(symbol=symbol, orderId=order_id)
        logger.info("Canceled Binance order %s for %s", order_id, symbol)
    except Exception as exc:
        logger.warning("Failed to cancel Binance order %s for %s: %s", order_id, symbol, exc)


def stop_existing_watcher(path: Path, *, reason: str = "replaced") -> None:
    metadata = _load_metadata(path)
    if not metadata:
        return
    pid = metadata.get("pid")
    if isinstance(pid, int) and pid > 0:
        try:
            os.killpg(pid, signal.SIGTERM)
            logger.info("Terminated watcher process group %s (%s)", pid, path.name)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to terminate watcher %s pgid=%s: %s", path, pid, exc)
    _cancel_order(metadata)
    metadata["active"] = False
    metadata["state"] = "stopped"
    metadata["stop_reason"] = reason
    metadata["last_update"] = datetime.now(timezone.utc).isoformat()
    _persist_metadata(path, metadata)


def _float_eq(a: float, b: float, tol: float = 1e-8) -> bool:
    return abs(float(a) - float(b)) <= tol


def _matches_plan(metadata: dict, plan: WatcherPlan) -> bool:
    if not metadata or not metadata.get("active"):
        return False
    if not _is_pid_alive(metadata.get("pid")):
        return False
    if metadata.get("symbol") != plan.symbol:
        return False
    if metadata.get("side") != plan.side:
        return False
    if metadata.get("mode") != plan.mode:
        return False
    if not _float_eq(metadata.get("limit_price", -1.0), plan.limit_price):
        return False
    if not _float_eq(metadata.get("target_qty", -1.0), plan.target_qty):
        return False
    expiry_at = metadata.get("expiry_at")
    if expiry_at:
        try:
            expiry_dt = datetime.fromisoformat(expiry_at.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) >= expiry_dt:
                return False
        except Exception:
            return False
    return True


def spawn_watcher(plan: WatcherPlan) -> Optional[Path]:
    precision = 8
    price_suffix = f"{plan.limit_price:.{precision}f}".rstrip("0").rstrip(".")
    config_path = watcher_config_path(plan.symbol, plan.side, plan.mode, suffix=price_suffix)

    pattern = f"{_sanitize(plan.symbol)}_{_sanitize(plan.side)}_{plan.mode}_*.json"
    for path in WATCHERS_DIR.glob(pattern):
        if path == config_path:
            continue
        metadata = _load_metadata(path)
        if metadata and metadata.get("active"):
            stop_existing_watcher(path, reason="superseded")

    existing = _load_metadata(config_path)
    if existing and _matches_plan(existing, plan):
        logger.debug("Watcher already active for %s %s %s", plan.symbol, plan.side, plan.mode)
        return config_path

    if existing:
        stop_existing_watcher(config_path, reason="refresh")

    started_at = datetime.now(timezone.utc)
    expiry_minutes = max(1, int(plan.expiry_minutes))
    expiry_at = started_at + timedelta(minutes=expiry_minutes)
    binance_symbol = binance_remap_symbols(plan.symbol)

    metadata = {
        "config_version": 1,
        "mode": plan.mode,
        "symbol": plan.symbol,
        "side": plan.side,
        "limit_price": float(plan.limit_price),
        "target_qty": float(plan.target_qty),
        "expiry_minutes": expiry_minutes,
        "expiry_at": expiry_at.isoformat(),
        "started_at": started_at.isoformat(),
        "state": "pending_launch",
        "active": True,
        "config_path": str(config_path),
        "poll_seconds": int(plan.poll_seconds),
        "price_tolerance": float(plan.price_tolerance),
        "binance_symbol": binance_symbol,
        "dry_run": bool(plan.dry_run),
    }
    _persist_metadata(config_path, metadata)

    python_bin = os.environ.get("PYTHON_BIN") or os.sys.executable or "python"
    command = [
        python_bin,
        "-m",
        "binanceneural.binance_watcher_cli",
        "watch",
        plan.symbol,
        "--side",
        plan.side,
        "--limit-price",
        str(plan.limit_price),
        "--target-qty",
        str(plan.target_qty),
        "--mode",
        plan.mode,
        "--expiry-minutes",
        str(expiry_minutes),
        "--poll-seconds",
        str(int(plan.poll_seconds)),
        "--config-path",
        str(config_path),
        "--price-tolerance",
        str(plan.price_tolerance),
    ]
    if plan.dry_run:
        command.append("--dry-run")

    logger.info("Launching Binance watcher: %s", " ".join(command))
    try:
        process = subprocess.Popen(
            command,
            env=os.environ.copy(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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

    return config_path


__all__ = ["WatcherPlan", "spawn_watcher", "stop_existing_watcher", "watcher_config_path"]
