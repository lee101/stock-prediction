import json
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from shlex import quote
from typing import Optional

from loguru import logger

from src.fixtures import crypto_symbols
from src.utils import debounce
from stock.state import get_state_dir, resolve_state_suffix

cwd = Path.cwd()
STATE_SUFFIX = resolve_state_suffix()
MAXDIFF_WATCHERS_DIR = get_state_dir() / f"maxdiff_watchers{STATE_SUFFIX or ''}"
MAXDIFF_WATCHERS_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _watcher_config_path(symbol: str, side: str, mode: str) -> Path:
    safe_symbol = _sanitize(symbol)
    safe_side = _sanitize(side)
    return MAXDIFF_WATCHERS_DIR / f"{safe_symbol}_{safe_side}_{mode}.json"


def _persist_watcher_metadata(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        temp_path.replace(path)
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.warning("Failed to persist watcher metadata %s: %s", path, exc)


@debounce(
    60 * 10, key_func=lambda symbol: symbol
)  # 10 minutes to not call too much for the same symbol
def backout_near_market(symbol):
    command = (
        f"PYTHONPATH={cwd} python scripts/alpaca_cli.py backout_near_market {symbol}"
    )
    logger.info(f"Running command {command}")
    # Run process in background without waiting
    subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


@debounce(60 * 10, key_func=lambda symbol, side, target_qty=None: f"{symbol}_{side}_{target_qty}")
def ramp_into_position(
    symbol: str,
    side: str = "buy",
    target_qty: Optional[float] = None,
    maxdiff_overflow: bool = False,
    risk_threshold: Optional[float] = None,
):
    """Ramp into a position over time using the alpaca CLI.

    Args:
        symbol: The trading symbol
        side: 'buy' or 'sell'
        target_qty: Optional target quantity
        maxdiff_overflow: If True, this is a maxdiff overflow trade that should check leverage
        risk_threshold: Optional risk threshold to check against (will be fetched if not provided)
    """
    command = f"PYTHONPATH={cwd} python scripts/alpaca_cli.py ramp_into_position {symbol} --side={side}"
    if target_qty is not None:
        command += f" --target-qty={target_qty}"
    if maxdiff_overflow:
        command += " --maxdiff-overflow"
    if risk_threshold is not None:
        command += f" --risk-threshold={risk_threshold}"
    logger.info(f"Running command {command}")
    # Run process in background without waiting
    subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


@debounce(60 * 10, key_func=lambda symbol, takeprofit_price: f"{symbol}_{takeprofit_price}")  # only once in 10 minutes
def spawn_close_position_at_takeprofit(symbol: str, takeprofit_price: float):
    command = f"PYTHONPATH={cwd} python scripts/alpaca_cli.py close_position_at_takeprofit {symbol} --takeprofit_price={takeprofit_price}"
    logger.info(f"Running command {command}")
    # Run process in background without waiting
    subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def _format_float(value: float, precision: int = 6) -> str:
    return f"{value:.{precision}f}"


@debounce(
    60 * 10,
    key_func=lambda symbol, side, limit_price, target_qty, tolerance_pct=0.0066, expiry_minutes=1440: (
        f"{symbol}_{side}_{limit_price}_{target_qty}_{tolerance_pct}_{expiry_minutes}"
    ),
)
def spawn_open_position_at_maxdiff_takeprofit(
    symbol: str,
    side: str,
    limit_price: float,
    target_qty: float,
    tolerance_pct: float = 0.0066,
    expiry_minutes: int = 60 * 24,
):
    """
    Spawn a watchdog process that attempts to open a maxdiff position when price approaches the target.

    The spawned process:
        * waits until the live price is within ``tolerance_pct`` of ``limit_price``
        * checks buying power to avoid using margin/leverage
        * keeps the qualifying limit order alive for up to ``expiry_minutes`` minutes
    """
    precision = 8 if symbol in crypto_symbols else 4
    started_at = datetime.now(timezone.utc)
    expiry_minutes_int = int(max(1, expiry_minutes))
    expiry_at = started_at + timedelta(minutes=expiry_minutes_int)
    config_path = _watcher_config_path(symbol, side, "entry")
    metadata = {
        "config_version": 1,
        "mode": "entry",
        "symbol": symbol,
        "side": side,
        "limit_price": float(limit_price),
        "target_qty": float(target_qty),
        "tolerance_pct": float(tolerance_pct),
        "precision": precision,
        "expiry_minutes": expiry_minutes_int,
        "expiry_at": expiry_at.isoformat(),
        "started_at": started_at.isoformat(),
        "state": "pending_launch",
        "active": True,
        "config_path": str(config_path),
    }
    _persist_watcher_metadata(config_path, metadata)
    command = (
        f"PYTHONPATH={cwd} python scripts/maxdiff_cli.py open-position {symbol}"
        f" --side={side}"
        f" --limit-price={_format_float(limit_price, precision)}"
        f" --target-qty={_format_float(target_qty, 8)}"
        f" --tolerance-pct={_format_float(tolerance_pct, 4)}"
        f" --expiry-minutes={expiry_minutes_int}"
        f" --config-path={quote(str(config_path))}"
    )
    if symbol in crypto_symbols:
        command += " --asset-class=crypto"
    logger.info(f"Running command {command}")
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except Exception as exc:
        metadata["state"] = "launch_failed"
        metadata["active"] = False
        metadata["error"] = str(exc)
        metadata["last_update"] = datetime.now(timezone.utc).isoformat()
        _persist_watcher_metadata(config_path, metadata)
        raise
    else:
        metadata["pid"] = process.pid
        metadata["state"] = "launched"
        metadata["last_update"] = datetime.now(timezone.utc).isoformat()
        _persist_watcher_metadata(config_path, metadata)


@debounce(
    60 * 10,
    key_func=lambda symbol, side, takeprofit_price, expiry_minutes=1440: (
        f"{symbol}_{side}_{takeprofit_price}_{expiry_minutes}"
    ),
)
def spawn_close_position_at_maxdiff_takeprofit(
    symbol: str,
    side: str,
    takeprofit_price: float,
    expiry_minutes: int = 60 * 24,
):
    """
    Spawn a watchdog process that continually re-arms maxdiff take-profit exits over ``expiry_minutes``.
    """
    precision = 8 if symbol in crypto_symbols else 4
    started_at = datetime.now(timezone.utc)
    expiry_minutes_int = int(max(1, expiry_minutes))
    expiry_at = started_at + timedelta(minutes=expiry_minutes_int)
    config_path = _watcher_config_path(symbol, side, "exit")
    exit_side = "sell" if side.lower().startswith("b") else "buy"
    metadata = {
        "config_version": 1,
        "mode": "exit",
        "symbol": symbol,
        "side": side,
        "exit_side": exit_side,
        "takeprofit_price": float(takeprofit_price),
        "price_tolerance": 0.001,
        "precision": precision,
        "expiry_minutes": expiry_minutes_int,
        "expiry_at": expiry_at.isoformat(),
        "started_at": started_at.isoformat(),
        "state": "pending_launch",
        "active": True,
        "config_path": str(config_path),
    }
    _persist_watcher_metadata(config_path, metadata)
    command = (
        f"PYTHONPATH={cwd} python scripts/maxdiff_cli.py close-position {symbol}"
        f" --side={side}"
        f" --takeprofit-price={_format_float(takeprofit_price, precision)}"
        f" --expiry-minutes={expiry_minutes_int}"
        f" --config-path={quote(str(config_path))}"
    )
    if symbol in crypto_symbols:
        command += " --asset-class=crypto"
    logger.info(f"Running command {command}")
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except Exception as exc:
        metadata["state"] = "launch_failed"
        metadata["active"] = False
        metadata["error"] = str(exc)
        metadata["last_update"] = datetime.now(timezone.utc).isoformat()
        _persist_watcher_metadata(config_path, metadata)
        raise
    else:
        metadata["pid"] = process.pid
        metadata["state"] = "launched"
        metadata["last_update"] = datetime.now(timezone.utc).isoformat()
        _persist_watcher_metadata(config_path, metadata)
