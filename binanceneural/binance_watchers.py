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
DEFAULT_PLAN_PRICE_REL_TOL = 0.0005  # 5 bps
DEFAULT_PLAN_QTY_REL_TOL = 0.02  # 2%


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
    margin: bool = False
    side_effect_type: str = "NO_SIDE_EFFECT"


def _sanitize(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def watcher_config_path(symbol: str, side: str, mode: str, *, suffix: Optional[str] = None) -> Path:
    safe_symbol = _sanitize(symbol)
    safe_side = _sanitize(side)
    base_name = f"{safe_symbol}_{safe_side}_{mode}"
    if suffix:
        base_name = f"{base_name}_{_sanitize(suffix)}"
    return WATCHERS_DIR / f"{base_name}.json"


def watcher_log_paths(config_path: Path) -> tuple[Path, Path]:
    stem = config_path.with_suffix("")
    return stem.with_suffix(".stdout.log"), stem.with_suffix(".stderr.log")


def _load_metadata(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to read watcher metadata {}: {}", path, exc)
        return None


def _persist_metadata(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        temp_path.replace(path)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to persist watcher metadata {}: {}", path, exc)


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
        if metadata.get("margin"):
            from src.binan.binance_margin import cancel_margin_order
            cancel_margin_order(symbol, order_id=order_id)
        else:
            client = binance_wrapper.get_client()
            client.cancel_order(symbol=symbol, orderId=order_id)
        logger.info("Canceled Binance order {} for {}", order_id, symbol)
    except Exception as exc:
        logger.warning("Failed to cancel Binance order {} for {}: {}", order_id, symbol, exc)


def stop_existing_watcher(path: Path, *, reason: str = "replaced") -> None:
    metadata = _load_metadata(path)
    if not metadata:
        return
    pid = metadata.get("pid")
    if isinstance(pid, int) and pid > 0:
        try:
            os.killpg(pid, signal.SIGTERM)
            logger.info("Terminated watcher process group {} ({})", pid, path.name)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to terminate watcher {} pgid={}: {}", path, pid, exc)
    _cancel_order(metadata)
    metadata["active"] = False
    metadata["state"] = "stopped"
    metadata["stop_reason"] = reason
    metadata["last_update"] = datetime.now(timezone.utc).isoformat()
    _persist_metadata(path, metadata)


def _float_eq(a: float, b: float, tol: float = 1e-8) -> bool:
    return abs(float(a) - float(b)) <= tol


def _price_within_tolerance(existing_price: object, new_price: float, tolerance: float) -> bool:
    try:
        existing = float(existing_price)
        proposed = float(new_price)
    except (TypeError, ValueError):
        return False
    if tolerance <= 0:
        return _float_eq(existing, proposed)
    baseline = max(abs(existing), abs(proposed), 1e-12)
    return abs(existing - proposed) / baseline <= float(tolerance)


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
    if not _price_within_tolerance(metadata.get("limit_price", -1.0), plan.limit_price, plan.price_tolerance):
        return False
    metadata_side_effect = str(metadata.get("side_effect_type", "NO_SIDE_EFFECT") or "NO_SIDE_EFFECT")
    if metadata_side_effect != str(plan.side_effect_type or "NO_SIDE_EFFECT"):
        return False
    price_rel_tol = max(
        DEFAULT_PLAN_PRICE_REL_TOL,
        float(metadata.get("price_tolerance", 0.0) or 0.0),
        float(plan.price_tolerance),
    )
    if not _float_rel_eq(metadata.get("limit_price", -1.0), plan.limit_price, rel_tol=price_rel_tol):
        return False
    if not _float_qty_eq(metadata.get("target_qty", -1.0), plan.target_qty, rel_tol=DEFAULT_PLAN_QTY_REL_TOL):
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
    active_paths: list[Path] = []
    for path in WATCHERS_DIR.glob(pattern):
        metadata = _load_metadata(path)
        if metadata and _matches_plan(metadata, plan):
            logger.debug("Watcher already active for %s %s %s", plan.symbol, plan.side, plan.mode)
            return path

    for path in WATCHERS_DIR.glob(pattern):
        if path == config_path:
            continue
        stop_existing_watcher(path, reason="superseded")

    existing = _load_metadata(config_path)
    if existing:
        stop_existing_watcher(config_path, reason="refresh")

    started_at = datetime.now(timezone.utc)
    expiry_minutes = max(1, int(plan.expiry_minutes))
    expiry_at = started_at + timedelta(minutes=expiry_minutes)
    binance_symbol = binance_remap_symbols(plan.symbol)
    stdout_log_path, stderr_log_path = watcher_log_paths(config_path)

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
        "margin": bool(plan.margin),
        "side_effect_type": plan.side_effect_type,
        "stdout_log_path": str(stdout_log_path),
        "stderr_log_path": str(stderr_log_path),
    }
    _persist_metadata(config_path, metadata)

    python_bin = os.environ.get("PYTHON_BIN") or os.sys.executable or "python"
    command = [
        python_bin,
        "-m",
        "binanceneural.binance_watcher_cli",
        plan.symbol,  # No "watch" subcommand - it's the default
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
    if plan.margin:
        command.extend(["--margin", "--side-effect-type", plan.side_effect_type])

    logger.info("Launching Binance watcher: {}", " ".join(command))
    try:
        stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_log_path.open("ab") as stdout_handle, stderr_log_path.open("ab") as stderr_handle:
            process = subprocess.Popen(
                command,
                env=os.environ.copy(),
                stdout=stdout_handle,
                stderr=stderr_handle,
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


def cancel_entry_watchers(*, exclude_symbol: Optional[str] = None) -> None:
    for path in WATCHERS_DIR.glob("*.json"):
        metadata = _load_metadata(path)
        if not metadata or not metadata.get("active"):
            continue
        if metadata.get("mode") != "entry":
            continue
        if exclude_symbol and metadata.get("symbol") == exclude_symbol:
            continue
        stop_existing_watcher(path, reason="symbol_switched")


__all__ = [
    "WatcherPlan",
    "spawn_watcher",
    "stop_existing_watcher",
    "watcher_config_path",
    "watcher_log_paths",
    "cancel_entry_watchers",
]
