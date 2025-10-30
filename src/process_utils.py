import json
import os
import signal
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

_DEFAULT_ENTRY_DEBOUNCE = int(os.getenv("MAXDIFF_ENTRY_SPAWN_DEBOUNCE_SECONDS", "120"))
_DEFAULT_EXIT_DEBOUNCE = int(os.getenv("MAXDIFF_EXIT_SPAWN_DEBOUNCE_SECONDS", "120"))
_DEFAULT_TAKEPROFIT_DEBOUNCE = int(os.getenv("TAKEPROFIT_SPAWN_DEBOUNCE_SECONDS", "120"))

MAXDIFF_ENTRY_SPAWN_DEBOUNCE_SECONDS = max(5, _DEFAULT_ENTRY_DEBOUNCE)
MAXDIFF_EXIT_SPAWN_DEBOUNCE_SECONDS = max(5, _DEFAULT_EXIT_DEBOUNCE)
TAKEPROFIT_SPAWN_DEBOUNCE_SECONDS = max(5, _DEFAULT_TAKEPROFIT_DEBOUNCE)

MAXDIFF_ENTRY_DEFAULT_POLL_SECONDS = max(5, int(os.getenv("MAXDIFF_ENTRY_POLL_SECONDS", "12")))
MAXDIFF_EXIT_DEFAULT_POLL_SECONDS = max(5, int(os.getenv("MAXDIFF_EXIT_POLL_SECONDS", "12")))
MAXDIFF_EXIT_DEFAULT_PRICE_TOLERANCE = float(os.getenv("MAXDIFF_EXIT_PRICE_TOLERANCE", "0.001"))


def _sanitize(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _watcher_config_path(symbol: str, side: str, mode: str, *, suffix: Optional[str] = None) -> Path:
    safe_symbol = _sanitize(symbol)
    safe_side = _sanitize(side)
    base_name = f"{safe_symbol}_{safe_side}_{mode}"
    if suffix:
        base_name = f"{base_name}_{_sanitize(suffix)}"
    return MAXDIFF_WATCHERS_DIR / f"{base_name}.json"


def _persist_watcher_metadata(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        temp_path.replace(path)
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.warning("Failed to persist watcher metadata %s: %s", path, exc)


def _load_watcher_metadata(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.warning("Failed to read watcher metadata %s: %s", path, exc)
        return None


def _stop_existing_watcher(config_path: Path, *, reason: str) -> None:
    metadata = _load_watcher_metadata(config_path)
    if not metadata:
        return

    pid_raw = metadata.get("pid")
    try:
        pid = int(pid_raw)
    except (TypeError, ValueError):
        pid = None

    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info("Terminated prior maxdiff watcher at %s (pid=%s)", config_path.name, pid)
        except ProcessLookupError:
            logger.debug("Watcher pid %s already exited for %s", pid, config_path)
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.warning("Failed to terminate watcher %s pid=%s: %s", config_path, pid, exc)

    if metadata.get("active") or pid:
        metadata["active"] = False
        metadata["state"] = reason
        metadata["terminated_at"] = datetime.now(timezone.utc).isoformat()
        _persist_watcher_metadata(config_path, metadata)


def _backout_key(symbol: str, **kwargs) -> str:
    extras = []
    for key in (
        "start_offset_minutes",
        "ramp_minutes",
        "market_after_minutes",
        "sleep_seconds",
        "market_close_buffer_minutes",
        "market_close_force_minutes",
    ):
        value = kwargs.get(key)
        if value is not None:
            extras.append(f"{key}={value}")
    suffix = "|".join(extras)
    return f"{symbol}|{suffix}" if suffix else symbol


@debounce(
    60 * 10, key_func=_backout_key
)  # 10 minutes to not call too much for the same symbol
def backout_near_market(
    symbol: str,
    *,
    start_offset_minutes: Optional[int] = None,
    ramp_minutes: Optional[int] = None,
    market_after_minutes: Optional[int] = None,
    sleep_seconds: Optional[int] = None,
    market_close_buffer_minutes: Optional[int] = None,
    market_close_force_minutes: Optional[int] = None,
):
    command = (
        f"PYTHONPATH={cwd} python scripts/alpaca_cli.py backout_near_market {symbol}"
    )
    option_map = {
        "start_offset_minutes": "--start-offset-minutes",
        "ramp_minutes": "--ramp-minutes",
        "market_after_minutes": "--market-after-minutes",
        "sleep_seconds": "--sleep-seconds",
        "market_close_buffer_minutes": "--market-close-buffer-minutes",
        "market_close_force_minutes": "--market-close-force-minutes",
    }
    options = []
    local_values = {
        "start_offset_minutes": start_offset_minutes,
        "ramp_minutes": ramp_minutes,
        "market_after_minutes": market_after_minutes,
        "sleep_seconds": sleep_seconds,
        "market_close_buffer_minutes": market_close_buffer_minutes,
        "market_close_force_minutes": market_close_force_minutes,
    }
    for key, flag in option_map.items():
        value = local_values.get(key)
        if value is None:
            continue
        options.append(f"{flag}={value}")
    if options:
        command = f"{command} {' '.join(options)}"
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


@debounce(TAKEPROFIT_SPAWN_DEBOUNCE_SECONDS, key_func=lambda symbol, takeprofit_price: f"{symbol}_{takeprofit_price}")
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
    MAXDIFF_ENTRY_SPAWN_DEBOUNCE_SECONDS,
    key_func=lambda symbol, side, limit_price, target_qty, tolerance_pct=0.0066, expiry_minutes=1440, poll_seconds=MAXDIFF_ENTRY_DEFAULT_POLL_SECONDS: (
        f"{symbol}_{side}_{limit_price}_{target_qty}_{tolerance_pct}_{expiry_minutes}_{poll_seconds}"
    ),
)
def spawn_open_position_at_maxdiff_takeprofit(
    symbol: str,
    side: str,
    limit_price: float,
    target_qty: float,
    tolerance_pct: float = 0.0066,
    expiry_minutes: int = 60 * 24,
    poll_seconds: int = MAXDIFF_ENTRY_DEFAULT_POLL_SECONDS,
):
    """
    Spawn a watchdog process that attempts to open a maxdiff position when price approaches the target.

    The spawned process:
        * waits until the live price is within ``tolerance_pct`` of ``limit_price``
        * checks buying power to avoid using margin/leverage
        * keeps the qualifying limit order alive for up to ``expiry_minutes`` minutes
    """
    precision = 8 if symbol in crypto_symbols else 4
    poll_seconds_int = max(1, int(poll_seconds))
    price_suffix = _format_float(limit_price, precision)
    config_path = _watcher_config_path(symbol, side, "entry", suffix=price_suffix)
    started_at = datetime.now(timezone.utc)
    expiry_minutes_int = int(max(1, expiry_minutes))
    expiry_at = started_at + timedelta(minutes=expiry_minutes_int)
    _stop_existing_watcher(config_path, reason="replaced_entry_watcher")
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
        "poll_seconds": poll_seconds_int,
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
        f" --poll-seconds={poll_seconds_int}"
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
    MAXDIFF_EXIT_SPAWN_DEBOUNCE_SECONDS,
    key_func=lambda symbol, side, takeprofit_price, expiry_minutes=1440, poll_seconds=MAXDIFF_EXIT_DEFAULT_POLL_SECONDS, price_tolerance=MAXDIFF_EXIT_DEFAULT_PRICE_TOLERANCE: (
        f"{symbol}_{side}_{takeprofit_price}_{expiry_minutes}_{poll_seconds}_{price_tolerance}"
    ),
)
def spawn_close_position_at_maxdiff_takeprofit(
    symbol: str,
    side: str,
    takeprofit_price: float,
    expiry_minutes: int = 60 * 24,
    poll_seconds: int = MAXDIFF_EXIT_DEFAULT_POLL_SECONDS,
    price_tolerance: float = MAXDIFF_EXIT_DEFAULT_PRICE_TOLERANCE,
):
    """
    Spawn a watchdog process that continually re-arms maxdiff take-profit exits over ``expiry_minutes``.
    """
    precision = 8 if symbol in crypto_symbols else 4
    poll_seconds_int = max(1, int(poll_seconds))
    price_tolerance_val = float(price_tolerance)
    started_at = datetime.now(timezone.utc)
    expiry_minutes_int = int(max(1, expiry_minutes))
    expiry_at = started_at + timedelta(minutes=expiry_minutes_int)
    price_suffix = _format_float(takeprofit_price, precision)
    config_path = _watcher_config_path(symbol, side, "exit", suffix=price_suffix)
    exit_side = "sell" if side.lower().startswith("b") else "buy"
    _stop_existing_watcher(config_path, reason="replaced_exit_watcher")
    metadata = {
        "config_version": 1,
        "mode": "exit",
        "symbol": symbol,
        "side": side,
        "exit_side": exit_side,
        "takeprofit_price": float(takeprofit_price),
        "price_tolerance": price_tolerance_val,
        "precision": precision,
        "expiry_minutes": expiry_minutes_int,
        "expiry_at": expiry_at.isoformat(),
        "started_at": started_at.isoformat(),
        "state": "pending_launch",
        "active": True,
        "config_path": str(config_path),
        "poll_seconds": poll_seconds_int,
    }
    _persist_watcher_metadata(config_path, metadata)
    command = (
        f"PYTHONPATH={cwd} python scripts/maxdiff_cli.py close-position {symbol}"
        f" --side={side}"
        f" --takeprofit-price={_format_float(takeprofit_price, precision)}"
        f" --expiry-minutes={expiry_minutes_int}"
        f" --config-path={quote(str(config_path))}"
        f" --poll-seconds={poll_seconds_int}"
        f" --price-tolerance={_format_float(price_tolerance_val, 6)}"
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
