import json
import os
import sys
import signal
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shlex import quote
from typing import Optional
from zoneinfo import ZoneInfo

from loguru import logger
from stock.state import get_state_dir, resolve_state_suffix, get_paper_suffix

from src.fixtures import crypto_symbols
from src.utils import debounce
from src.work_stealing_config import (
    CRYPTO_SYMBOLS,
    get_entry_tolerance_for_symbol,
    is_crypto_out_of_hours,
    should_force_immediate_crypto,
)

cwd = Path.cwd()
STATE_SUFFIX = resolve_state_suffix()
PAPER_SUFFIX = get_paper_suffix()
MAXDIFF_WATCHERS_DIR = get_state_dir() / f"maxdiff_watchers{PAPER_SUFFIX}{STATE_SUFFIX or ''}"
MAXDIFF_WATCHERS_DIR.mkdir(parents=True, exist_ok=True)

_DEFAULT_ENTRY_DEBOUNCE = int(os.getenv("MAXDIFF_ENTRY_SPAWN_DEBOUNCE_SECONDS", "120"))
_DEFAULT_EXIT_DEBOUNCE = int(os.getenv("MAXDIFF_EXIT_SPAWN_DEBOUNCE_SECONDS", "120"))
_DEFAULT_TAKEPROFIT_DEBOUNCE = int(os.getenv("TAKEPROFIT_SPAWN_DEBOUNCE_SECONDS", "120"))

MAXDIFF_ENTRY_SPAWN_DEBOUNCE_SECONDS = max(5, _DEFAULT_ENTRY_DEBOUNCE)
MAXDIFF_EXIT_SPAWN_DEBOUNCE_SECONDS = max(5, _DEFAULT_EXIT_DEBOUNCE)
TAKEPROFIT_SPAWN_DEBOUNCE_SECONDS = max(5, _DEFAULT_TAKEPROFIT_DEBOUNCE)

MAXDIFF_ENTRY_DEFAULT_POLL_SECONDS = max(5, int(os.getenv("MAXDIFF_ENTRY_POLL_SECONDS", "45")))
MAXDIFF_EXIT_DEFAULT_POLL_SECONDS = max(5, int(os.getenv("MAXDIFF_EXIT_POLL_SECONDS", "45")))
MAXDIFF_EXIT_DEFAULT_PRICE_TOLERANCE = float(os.getenv("MAXDIFF_EXIT_PRICE_TOLERANCE", "0.001"))

# Timezone constants
UTC = ZoneInfo("UTC")
NEW_YORK = ZoneInfo("America/New_York")

# Strategy families that rely on staged entry watchers (MaxDiff variants).
MAXDIFF_STRATEGY_NAMES = {"maxdiff", "maxdiffalwayson", "pctdiff", "highlow"}


def _calculate_next_crypto_bar_time(current_time: Optional[datetime] = None) -> datetime:
    """Calculate the next crypto watcher expiry time.

    Crypto watchers expire at the next analysis run (22:00 EST) to ensure 24/7 coverage.
    This gives 24+ hour watcher lifetime, preventing gaps between analysis runs.

    For crypto (24/7 trading), always ensures at least 24 hours of coverage by using
    tomorrow's 22:00 EST if today's 22:00 EST is less than 24 hours away.
    """
    now = current_time or datetime.now(timezone.utc)
    # Ensure timezone aware
    now_utc = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    now_et = now_utc.astimezone(NEW_YORK)

    # Next analysis run is at 22:00 EST (initial analysis window)
    next_analysis = now_et.replace(hour=22, minute=0, second=0, microsecond=0)

    # For crypto (24/7 trading), always ensure at least 24 hours of coverage
    # If today's 22:00 is in the past or less than 24 hours away, use tomorrow's 22:00
    if next_analysis <= now_et:
        # If 22:00 already passed today, use tomorrow's 22:00
        next_analysis += timedelta(days=1)
    elif next_analysis <= now_et + timedelta(hours=24):
        # If today's 22:00 is less than 24 hours away, use tomorrow's 22:00 for full day coverage
        next_analysis += timedelta(days=1)

    return next_analysis.astimezone(timezone.utc)


def _calculate_next_nyse_close(current_time: Optional[datetime] = None) -> datetime:
    """Calculate the next NYSE market close time (4:00 PM ET).

    Returns the next market close at 16:00 ET, skipping weekends.
    """
    now = current_time or datetime.now(timezone.utc)
    now_utc = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    now_et = now_utc.astimezone(NEW_YORK)

    # Start with today's market close
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    # If we're past today's close or it's weekend, find next trading day close
    while market_close <= now_et or market_close.weekday() >= 5:
        market_close += timedelta(days=1)
        market_close = market_close.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_close.astimezone(timezone.utc)


def _calculate_market_aware_expiry(
    symbol: str,
    current_time: Optional[datetime] = None,
    min_duration_minutes: int = 60,
) -> datetime:
    """Calculate market-aware expiry time for watchers.

    For crypto: Expires at next analysis run (22:00 EST) for 24/7 coverage
    For stocks: Expires at next NYSE market close

    Args:
        symbol: Trading symbol
        current_time: Current time (defaults to now)
        min_duration_minutes: Minimum watcher lifetime in minutes

    Returns:
        Expiry datetime aligned with market timing
    """
    now = current_time or datetime.now(timezone.utc)
    now_utc = now if now.tzinfo else now.replace(tzinfo=timezone.utc)

    # Calculate minimum expiry time
    min_expiry = now_utc + timedelta(minutes=min_duration_minutes)

    if symbol in crypto_symbols:
        # Crypto: expire at next analysis run (22:00 EST) for 24+ hour coverage
        market_expiry = _calculate_next_crypto_bar_time(now_utc)
    else:
        # Stocks: expire at next NYSE close
        market_expiry = _calculate_next_nyse_close(now_utc)

    # Use whichever is later to ensure minimum duration
    return max(min_expiry, market_expiry)


def _is_data_bar_fresh(symbol: str, current_time: Optional[datetime] = None) -> bool:
    """Check if we're in a safe window after a new data bar should be available.

    For crypto: Safe after 00:05 UTC (5 minutes after midnight)
    For stocks: Safe after 09:35 ET (5 minutes after market open)

    This prevents spawning watchers before new forecast data is ready.
    """
    now = current_time or datetime.now(timezone.utc)
    now_utc = now if now.tzinfo else now.replace(tzinfo=timezone.utc)

    if symbol in crypto_symbols:
        # Crypto: check if we're at least 5 minutes past UTC midnight
        current_hour = now_utc.hour
        current_minute = now_utc.minute

        # Safe window: 00:05 UTC to 23:59 UTC
        if current_hour == 0 and current_minute < 5:
            return False  # Too early after midnight
        return True
    else:
        # Stocks: check if we're at least 5 minutes past NYSE market open
        now_et = now_utc.astimezone(NEW_YORK)

        # Skip weekends
        if now_et.weekday() >= 5:
            return False

        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        safe_time = now_et.replace(hour=9, minute=35, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

        # Safe window: 09:35 ET to 16:00 ET on trading days
        return safe_time <= now_et <= market_close


def _sanitize(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _watcher_config_path(symbol: str, side: str, mode: str, *, suffix: Optional[str] = None) -> Path:
    safe_symbol = _sanitize(symbol)
    safe_side = _sanitize(side)
    base_name = f"{safe_symbol}_{safe_side}_{mode}"
    if suffix:
        base_name = f"{base_name}_{_sanitize(suffix)}"
    return MAXDIFF_WATCHERS_DIR / f"{base_name}.json"


def stop_all_entry_watchers(symbol: str, *, reason: str = "reset") -> None:
    """Stop all entry watchers (buy and sell) for a symbol."""
    pattern = f"{_sanitize(symbol)}_*_entry_*.json"
    for path in MAXDIFF_WATCHERS_DIR.glob(pattern):
        try:
            _stop_existing_watcher(path, reason=reason)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed stopping watcher %s: %s", path.name, exc)


def enforce_min_spread(buy_price: float, sell_price: float, min_spread_pct: float = 0.0003) -> tuple[float, float]:
    """Return (buy, sell) with at least min_spread_pct between them."""
    buy = max(buy_price, 1e-6)
    min_sell = buy * (1.0 + min_spread_pct)
    sell = max(sell_price, min_sell)
    return buy, sell


def _persist_watcher_metadata(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        temp_path.replace(path)
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.warning(f"Failed to persist watcher metadata {path}: {exc}")


def _load_watcher_metadata(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.warning(f"Failed to read watcher metadata {path}: {exc}")
        return None


def _is_pid_alive(pid: Optional[int]) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    return True


def _watcher_matches_params(metadata: dict, **expected_params) -> bool:
    """Check if existing watcher metadata matches expected parameters."""
    if not metadata or not metadata.get("active"):
        return False

    # Check if process is still alive
    if not _is_pid_alive(metadata.get("pid")):
        return False

    # Check if watcher has expired
    expiry_at_str = metadata.get("expiry_at")
    if expiry_at_str:
        try:
            expiry_at = datetime.fromisoformat(expiry_at_str.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) >= expiry_at:
                return False
        except (ValueError, TypeError):
            pass

    # Compare relevant parameters
    for key, expected_value in expected_params.items():
        # If expected param is not None but missing from metadata, it's a mismatch
        if expected_value is not None and key not in metadata:
            return False

        if key not in metadata:
            continue

        actual_value = metadata[key]

        # For numeric values, allow small tolerance
        if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
            if abs(float(expected_value) - float(actual_value)) > 1e-6:
                return False
        elif actual_value != expected_value:
            return False

    return True


def _stop_existing_watcher(config_path: Path, *, reason: str) -> bool:
    """Stop an existing watcher. Returns True if a running process was terminated."""
    metadata = _load_watcher_metadata(config_path)
    if not metadata:
        return False

    pid_raw = metadata.get("pid")
    try:
        pid = int(pid_raw)
    except (TypeError, ValueError):
        pid = None

    terminated = False
    if pid:
        try:
            # Use killpg to kill entire process group since watchers are spawned
            # with shell=True + start_new_session=True. The PID stored is the shell
            # wrapper, and os.kill() only kills the shell leaving the Python child
            # orphaned. killpg() kills the entire group including the Python child.
            os.killpg(pid, signal.SIGTERM)
            logger.info(f"Terminated prior maxdiff watcher process group at {config_path.name} (pgid={pid})")
            terminated = True
        except ProcessLookupError:
            pass  # Already exited, nothing to do
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.warning(f"Failed to terminate watcher {config_path} pgid={pid}: {exc}")

    if metadata.get("active") or pid:
        metadata["active"] = False
        metadata["state"] = reason
        metadata["terminated_at"] = datetime.now(timezone.utc).isoformat()
        _persist_watcher_metadata(config_path, metadata)

    return terminated


def _stop_conflicting_entry_watchers(
    symbol: str,
    side: str,
    *,
    entry_strategy: Optional[str],
    new_limit_price: float,
    skip_path: Path,
) -> None:
    """Terminate other entry watchers for the same strategy that use outdated limits."""
    if not entry_strategy:
        return

    safe_symbol = _sanitize(symbol)
    safe_side = _sanitize(side)
    prefix = f"{safe_symbol}_{safe_side}_entry"

    for path in MAXDIFF_WATCHERS_DIR.glob(f"{prefix}_*.json"):
        if path == skip_path:
            continue

        metadata = _load_watcher_metadata(path)
        if not metadata:
            continue

        if metadata.get("mode") != "entry":
            continue

        existing_strategy = metadata.get("entry_strategy")
        if existing_strategy and existing_strategy != entry_strategy:
            logger.info(
                f"Terminating {symbol} {side} entry watcher at {path.name} "
                f"due to strategy change {existing_strategy}->{entry_strategy}"
            )
            _stop_existing_watcher(path, reason="strategy_changed_entry_watcher")
            continue

        if not existing_strategy:
            logger.info(
                f"Terminating legacy {symbol} {side} entry watcher at {path.name} "
                f"for strategy {entry_strategy}"
            )
            _stop_existing_watcher(path, reason="legacy_strategy_entry_watcher")
            continue

        existing_limit = metadata.get("limit_price")
        if existing_limit is None:
            continue

        try:
            limit_delta = abs(float(existing_limit) - float(new_limit_price))
        except (TypeError, ValueError):
            limit_delta = float("inf")

        if limit_delta <= 1e-6:
            continue

        if _stop_existing_watcher(path, reason="superseded_entry_watcher"):
            logger.info(
                f"Terminated conflicting {symbol} {side} entry watcher at {path.name} "
                f"(limit {float(existing_limit):.8f}) in favor of {float(new_limit_price):.8f}"
            )


def _stop_conflicting_exit_watchers(
    symbol: str,
    side: str,
    *,
    entry_strategy: Optional[str],
    new_takeprofit_price: float,
    skip_path: Path,
) -> None:
    """Terminate other exit watchers for the same strategy that use outdated take-profit prices."""
    if not entry_strategy:
        return

    safe_symbol = _sanitize(symbol)
    safe_side = _sanitize(side)
    prefix = f"{safe_symbol}_{safe_side}_exit"

    for path in MAXDIFF_WATCHERS_DIR.glob(f"{prefix}_*.json"):
        if path == skip_path:
            continue

        metadata = _load_watcher_metadata(path)
        if not metadata:
            continue

        if metadata.get("mode") != "exit":
            continue

        existing_strategy = metadata.get("entry_strategy")
        if existing_strategy and existing_strategy != entry_strategy:
            if _stop_existing_watcher(path, reason="strategy_changed_exit_watcher"):
                logger.info(
                    f"Terminated {symbol} {side} exit watcher at {path.name} "
                    f"due to strategy change {existing_strategy}->{entry_strategy}"
                )
            continue

        if not existing_strategy:
            if _stop_existing_watcher(path, reason="legacy_strategy_exit_watcher"):
                logger.info(
                    f"Terminated legacy {symbol} {side} exit watcher at {path.name} "
                    f"for strategy {entry_strategy}"
                )
            continue

        existing_tp = metadata.get("takeprofit_price")
        if existing_tp is None:
            continue

        try:
            tp_delta = abs(float(existing_tp) - float(new_takeprofit_price))
        except (TypeError, ValueError):
            tp_delta = float("inf")

        if tp_delta <= 1e-6:
            continue

        if _stop_existing_watcher(path, reason="superseded_exit_watcher"):
            logger.info(
                f"Terminated conflicting {symbol} {side} exit watcher at {path.name} "
                f"(takeprofit {float(existing_tp):.8f}) in favor of {float(new_takeprofit_price):.8f}"
            )


def _get_inherited_env():
    """Get environment with PYTHONPATH set, preserving critical variables like PAPER."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)

    # Ensure Alpaca credentials are explicitly passed
    # Import here to get current values loaded in parent process
    from env_real import (
        ALP_ENDPOINT,
        ALP_KEY_ID,
        ALP_KEY_ID_PROD,
        ALP_SECRET_KEY,
        ALP_SECRET_KEY_PROD,
        PAPER,
    )

    # Explicitly set all credentials in child environment
    # This ensures child processes use the same credentials as parent
    env["ALP_KEY_ID"] = ALP_KEY_ID
    env["ALP_SECRET_KEY"] = ALP_SECRET_KEY
    env["ALP_KEY_ID_PROD"] = ALP_KEY_ID_PROD
    env["ALP_SECRET_KEY_PROD"] = ALP_SECRET_KEY_PROD
    env["ALP_ENDPOINT"] = ALP_ENDPOINT
    env["PAPER"] = "1" if PAPER else "0"

    # Verify credentials are set (not placeholders)
    active_key = ALP_KEY_ID if PAPER else ALP_KEY_ID_PROD
    active_secret = ALP_SECRET_KEY if PAPER else ALP_SECRET_KEY_PROD

    has_valid_credentials = (
        active_key
        and active_secret
        and "placeholder" not in active_key.lower()
        and "placeholder" not in active_secret.lower()
    )

    if not has_valid_credentials:
        logger.warning(
            f"Spawning subprocess with PAPER={env['PAPER']} but credentials appear invalid! "
            f"Child process may get 401 errors. Check that credentials are exported in environment."
        )
    else:
        logger.debug(
            f"Spawning subprocess with PAPER={env['PAPER']}, "
            f"using {'paper' if PAPER else 'prod'} credentials (validated)"
        )

    return env


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


@debounce(60 * 10, key_func=_backout_key)  # 10 minutes to not call too much for the same symbol
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
    command = f"python scripts/alpaca_cli.py backout_near_market {symbol}"
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
        env=_get_inherited_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


@debounce(
    60 * 10,
    key_func=lambda symbol,
    side,
    target_qty=None,
    maxdiff_overflow=False,
    risk_threshold=None: f"{symbol}_{side}_{target_qty}_{maxdiff_overflow}",
)
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
    command = f"python scripts/alpaca_cli.py ramp_into_position {symbol} --side={side}"
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
        env=_get_inherited_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


@debounce(TAKEPROFIT_SPAWN_DEBOUNCE_SECONDS, key_func=lambda symbol, takeprofit_price: f"{symbol}_{takeprofit_price}")
def spawn_close_position_at_takeprofit(symbol: str, takeprofit_price: float):
    command = (
        f"python scripts/alpaca_cli.py close_position_at_takeprofit {symbol} --takeprofit_price={takeprofit_price}"
    )
    logger.info(f"Running command {command}")
    # Run process in background without waiting
    subprocess.Popen(
        command,
        shell=True,
        env=_get_inherited_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def _format_float(value: float, precision: int = 6) -> str:
    return f"{value:.{precision}f}"


@debounce(
    MAXDIFF_ENTRY_SPAWN_DEBOUNCE_SECONDS,
    key_func=lambda symbol,
    side,
    limit_price,
    target_qty,
    tolerance_pct=None,
    expiry_minutes=1440,
    poll_seconds=MAXDIFF_ENTRY_DEFAULT_POLL_SECONDS,
    entry_strategy=None,
    force_immediate=False,
    priority_rank=None,
    crypto_rank=None: (
        f"{symbol}_{side}_{limit_price}_{target_qty}_{tolerance_pct or 'auto'}_{expiry_minutes}_{poll_seconds}_{entry_strategy or ''}_{int(bool(force_immediate))}_{priority_rank if priority_rank is not None else 'none'}_{crypto_rank or 'none'}"
    ),
)
def spawn_open_position_at_maxdiff_takeprofit(
    symbol: str,
    side: str,
    limit_price: float,
    target_qty: float,
    tolerance_pct: Optional[float] = None,
    expiry_minutes: int = 60 * 24,
    poll_seconds: int = MAXDIFF_ENTRY_DEFAULT_POLL_SECONDS,
    entry_strategy: Optional[str] = None,
    *,
    force_immediate: bool = False,
    priority_rank: Optional[int] = None,
    crypto_rank: Optional[int] = None,
):
    """
    Spawn a watchdog process that attempts to open a maxdiff position when price approaches the target.

    The spawned process:
        * waits until the live price is within ``tolerance_pct`` of ``limit_price``
        * checks buying power to avoid using margin/leverage
        * keeps the qualifying limit order alive for up to ``expiry_minutes`` minutes

    Args:
        symbol: Trading symbol
        side: 'buy' or 'sell'
        limit_price: Target limit price
        target_qty: Quantity to accumulate
        tolerance_pct: Price tolerance (auto-calculated for crypto if None)
        expiry_minutes: Watcher lifetime in minutes
        poll_seconds: Polling interval
        entry_strategy: Strategy name
        force_immediate: Ignore tolerance, enter immediately
        priority_rank: Priority for always-on strategies
        crypto_rank: Crypto rank for out-of-hours tolerance (1=best)
    """
    # Auto-calculate tolerance for crypto based on rank and market hours
    if tolerance_pct is None:
        is_top_crypto = crypto_rank == 1 if crypto_rank is not None else False
        tolerance_pct = get_entry_tolerance_for_symbol(symbol, is_top_crypto)
        logger.debug(
            f"{symbol}: Auto-calculated tolerance={tolerance_pct:.4f} "
            f"(crypto_rank={crypto_rank}, out_of_hours={is_crypto_out_of_hours()})"
        )

    # Override force_immediate for top crypto out-of-hours
    if not force_immediate and symbol in CRYPTO_SYMBOLS and crypto_rank is not None:
        if should_force_immediate_crypto(crypto_rank):
            force_immediate = True
            logger.info(f"{symbol}: Auto-enabled force_immediate (crypto rank {crypto_rank} during out-of-hours)")
    precision = 8 if symbol in crypto_symbols else 4
    poll_seconds_int = max(1, int(poll_seconds))
    price_suffix = _format_float(limit_price, precision)
    config_path = _watcher_config_path(symbol, side, "entry", suffix=price_suffix)
    started_at = datetime.now(timezone.utc)

    # Use market-aware expiry if no explicit duration provided
    if expiry_minutes == 60 * 24:  # Default value
        expiry_at = _calculate_market_aware_expiry(symbol, started_at)
        expiry_minutes_int = int((expiry_at - started_at).total_seconds() / 60)
    else:
        expiry_minutes_int = int(max(1, expiry_minutes))
        expiry_at = started_at + timedelta(minutes=expiry_minutes_int)

    # Warn if data bar might not be fresh, but proceed anyway
    if not _is_data_bar_fresh(symbol, started_at):
        logger.warning(
            f"Spawning {symbol} {side} entry watcher @ {limit_price:.4f} shortly after data bar refresh - forecast may be based on previous bar"
        )

    # Check if existing watcher matches desired parameters
    _stop_conflicting_entry_watchers(
        symbol,
        side,
        entry_strategy=entry_strategy,
        new_limit_price=float(limit_price),
        skip_path=config_path,
    )

    # Always restart watchers to ensure fresh code unless an identical watcher is already active
    existing_metadata = _load_watcher_metadata(config_path)
    if _watcher_matches_params(
        existing_metadata,
        limit_price=float(limit_price),
        target_qty=float(target_qty),
        tolerance_pct=float(tolerance_pct),
        entry_strategy=entry_strategy,
    ):
        logger.debug(
            "Skipping spawn for %s %s entry watcher @ %.4f - existing watcher matches parameters",
            symbol,
            side,
            limit_price,
        )
        return

    if existing_metadata:
        logger.debug(
            f"Restarting {symbol} {side} entry watcher @ {limit_price:.4f} (fresh code/params)"
        )
    _stop_existing_watcher(config_path, reason="replaced_entry_watcher")
    priority_value: Optional[int]
    if priority_rank is None:
        priority_value = None
    else:
        try:
            priority_value = int(priority_rank)
        except (TypeError, ValueError):
            priority_value = None

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
        "entry_strategy": entry_strategy,
        "force_immediate": bool(force_immediate),
    }
    if priority_value is not None:
        metadata["priority_rank"] = priority_value
    _persist_watcher_metadata(config_path, metadata)
    python_bin = sys.executable or "python"
    command = (
        f"{python_bin} scripts/maxdiff_cli.py open-position {symbol}"
        f" --side={side}"
        f" --limit-price={_format_float(limit_price, precision)}"
        f" --target-qty={_format_float(target_qty, 8)}"
        f" --tolerance-pct={_format_float(tolerance_pct, 4)}"
        f" --expiry-minutes={expiry_minutes_int}"
        f" --config-path={quote(str(config_path))}"
        f" --poll-seconds={poll_seconds_int}"
    )
    if force_immediate:
        command += " --force-immediate"
    if priority_value is not None:
        command += f" --priority-rank={priority_value}"
    if symbol in crypto_symbols or symbol.upper().endswith("USD"):
        command += " --asset-class=crypto"
    logger.info(f"Running command {command}")
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            env=_get_inherited_env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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
    key_func=lambda symbol,
    side,
    takeprofit_price,
    expiry_minutes=1440,
    poll_seconds=MAXDIFF_EXIT_DEFAULT_POLL_SECONDS,
    price_tolerance=MAXDIFF_EXIT_DEFAULT_PRICE_TOLERANCE,
    entry_strategy=None,
    target_qty=None: (
        f"{symbol}_{side}_{takeprofit_price}_{expiry_minutes}_{poll_seconds}_{price_tolerance}_{entry_strategy or ''}_{target_qty if target_qty is not None else 'full'}"
    ),
)
def spawn_close_position_at_maxdiff_takeprofit(
    symbol: str,
    side: str,
    takeprofit_price: float,
    expiry_minutes: int = 60 * 24,
    poll_seconds: int = MAXDIFF_EXIT_DEFAULT_POLL_SECONDS,
    price_tolerance: float = MAXDIFF_EXIT_DEFAULT_PRICE_TOLERANCE,
    entry_strategy: Optional[str] = None,
    target_qty: Optional[float] = None,
):
    """
    Spawn a watchdog process that continually re-arms maxdiff take-profit exits over ``expiry_minutes``.
    """
    precision = 8 if symbol in crypto_symbols else 4
    poll_seconds_int = max(1, int(poll_seconds))
    price_tolerance_val = float(price_tolerance)
    try:
        target_qty_val = float(target_qty) if target_qty is not None else None
    except (TypeError, ValueError):
        target_qty_val = None
    if target_qty_val is not None and target_qty_val <= 0:
        target_qty_val = None
    started_at = datetime.now(timezone.utc)

    # Use market-aware expiry if no explicit duration provided
    if expiry_minutes == 60 * 24:  # Default value
        expiry_at = _calculate_market_aware_expiry(symbol, started_at)
        expiry_minutes_int = int((expiry_at - started_at).total_seconds() / 60)
    else:
        expiry_minutes_int = int(max(1, expiry_minutes))
        expiry_at = started_at + timedelta(minutes=expiry_minutes_int)

    price_suffix = _format_float(takeprofit_price, precision)
    config_path = _watcher_config_path(symbol, side, "exit", suffix=price_suffix)
    exit_side = "sell" if side.lower().startswith("b") else "buy"

    # Warn if data bar might not be fresh, but proceed anyway
    if not _is_data_bar_fresh(symbol, started_at):
        logger.warning(
            f"Spawning {symbol} {side} exit watcher @ {takeprofit_price:.4f} shortly after data bar refresh - forecast may be based on previous bar"
        )

    # Stop conflicting exit watchers with different take-profit prices
    _stop_conflicting_exit_watchers(
        symbol,
        side,
        entry_strategy=entry_strategy,
        new_takeprofit_price=float(takeprofit_price),
        skip_path=config_path,
    )

    # Restart exit watchers only when parameters change
    existing_metadata = _load_watcher_metadata(config_path)
    if _watcher_matches_params(
        existing_metadata,
        takeprofit_price=float(takeprofit_price),
        price_tolerance=price_tolerance_val,
        entry_strategy=entry_strategy,
        target_qty=target_qty_val,
    ):
        logger.debug(
            "Skipping spawn for %s %s exit watcher @ %.4f - existing watcher matches parameters",
            symbol,
            side,
            takeprofit_price,
        )
        return

    if existing_metadata:
        logger.debug(
            f"Restarting {symbol} {side} exit watcher @ {takeprofit_price:.4f} (fresh code/params)"
        )
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
        "entry_strategy": entry_strategy,
    }
    if target_qty_val is not None:
        metadata["target_qty"] = target_qty_val
    _persist_watcher_metadata(config_path, metadata)
    python_bin = sys.executable or "python"
    command = (
        f"{python_bin} scripts/maxdiff_cli.py close-position {symbol}"
        f" --side={side}"
        f" --takeprofit-price={_format_float(takeprofit_price, precision)}"
        f" --expiry-minutes={expiry_minutes_int}"
        f" --config-path={quote(str(config_path))}"
        f" --poll-seconds={poll_seconds_int}"
        f" --price-tolerance={_format_float(price_tolerance_val, 6)}"
    )
    if target_qty_val is not None:
        command += f" --target-qty={_format_float(target_qty_val, 8)}"
    if symbol in crypto_symbols:
        command += " --asset-class=crypto"
    logger.info(f"Running command {command}")
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            env=_get_inherited_env(),
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
