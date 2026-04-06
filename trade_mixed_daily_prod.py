#!/usr/bin/env python3
"""
Production daily mixed stock+crypto trading bot.

Uses the C env binding directly for inference — guaranteed to match training.
Exports fresh daily bars to MKTD binary, runs policy forward pass through
the C env observation pipeline, decodes action to trade signal.

Usage:
    # Generate today's signal
    python -u trade_mixed_daily_prod.py --once

    # Run as daily daemon (executes at midnight UTC)
    python -u trade_mixed_daily_prod.py --daemon

    # Dry run (print signal without executing)
    python -u trade_mixed_daily_prod.py --once --dry-run
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import json
import math
import re
import shlex
import struct
import sys
import tempfile
import threading
import time
import weakref
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, TypedDict, cast

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.local_data_health import format_local_data_health_lines

DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "PLTR", "NET",
    "JPM", "V", "SPY", "QQQ", "NFLX", "AMD",
    "BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD", "DOGEUSD", "LINKUSD", "AAVEUSD",
]

DEFAULT_CHECKPOINT = "pufferlib_market/checkpoints/mixed23_fresh_targeted/reg_combo_2/best.pt"
DATA_DIR = "trainingdata/train"
TEMP_BIN_PREFIX = "mixed23_daily_live_"
DEFAULT_LOOKBACK_DAYS = 120
DEFAULT_WARMUP_BUFFER_DAYS = 30
DEFAULT_HIDDEN_SIZE = 1024
DEFAULT_ALLOCATION_PCT = 10.0
DEFAULT_DAEMON_HOUR_UTC = 0
DEFAULT_DAEMON_MINUTE_UTC = 5
DEFAULT_INFERENCE_MAX_STEPS = 90
DEFAULT_FILL_SLIPPAGE_BPS = 5.0
DEFAULT_FEE_RATE = 0.0
DEFAULT_MAX_LEVERAGE = 1.0
DEFAULT_PERIODS_PER_YEAR = 365.0
DEFAULT_ACTION_MAX_OFFSET_BPS = 0.0
MIXED_ENV_PORTFOLIO_FEATURES = 5
MIXED_ENV_SEED = 42
MIXED_ENV_ACTION_ALLOCATION_BINS = 1
MIXED_ENV_ACTION_LEVEL_BINS = 1
MAX_MIXED_SYMBOL_LENGTH = 20
MAX_MIXED_SYMBOL_COUNT = 256
MAX_SYMBOLS_FILE_BYTES = 64 * 1024
SAFE_MIXED_SYMBOL_RE = re.compile(
    rf"^[A-Z0-9](?:[A-Z0-9.-]{{0,{MAX_MIXED_SYMBOL_LENGTH - 2}}}[A-Z0-9])?$"
)
MKTD_MAGIC = b"MKTD"
MKTD_VERSION = 2
MKTD_HEADER_RESERVED_BYTES = 40
MKTD_HEADER_FORMAT = f"<4sIIIII{MKTD_HEADER_RESERVED_BYTES}s"
MKTD_HEADER_SIZE = struct.calcsize(MKTD_HEADER_FORMAT)
MKTD_SYMBOL_BYTES = 16
MKTD_SYMBOL_PAYLOAD_BYTES = MKTD_SYMBOL_BYTES - 1
MKTD_FLOAT32_SIZE = np.dtype(np.float32).itemsize
MKTD_REQUIRED_PRICE_FEATURES = 5
MIXED_DAILY_SUPPORTED_CHECKPOINT_ARCH = "mlp"
MIXED_DAILY_ALTERNATIVE_SIGNAL_LIMIT = 5
MixedDailySymbolSource = Literal["default", "cli", "file"]
MixedDailyRunMode = Literal["once", "dry_run", "daemon", "config_only"]
MixedDailyPreviewRunMode = Literal["once", "dry_run", "daemon", "check_config"]
MixedDailySymbolHealthStatus = Literal["usable", "stale", "missing", "invalid"]


@dataclass(frozen=True)
class MixedDailySymbolDetail:
    asset_class: Literal["stock", "crypto"]
    status: MixedDailySymbolHealthStatus
    local_data_date: str | None
    reason: str | None


@dataclass(frozen=True)
class MixedDailyRuntimeConfig:
    checkpoint: str
    checkpoint_exists: bool
    checkpoint_arch: str | None
    checkpoint_hidden_size: int | None
    checkpoint_obs_size: int | None
    checkpoint_num_actions: int | None
    checkpoint_matches_runtime: bool | None
    checkpoint_inspection_error: str | None
    expected_obs_size: int | None
    expected_num_actions: int | None
    data_dir: str
    data_dir_exists: bool
    symbols_file: str | None
    symbol_source: MixedDailySymbolSource
    symbols: list[str]
    symbol_count: int
    stock_symbol_count: int
    crypto_symbol_count: int
    usable_symbols: list[str]
    usable_symbol_count: int
    missing_symbol_data: list[str]
    invalid_symbol_data: dict[str, str]
    latest_local_data_date: str | None
    oldest_local_data_date: str | None
    stale_symbol_data: dict[str, str]
    stale_symbol_count: int
    symbol_details: dict[str, MixedDailySymbolDetail]
    removed_duplicate_symbols: list[str]
    ignored_symbol_inputs: list[str]
    hidden_size: int
    requested_hidden_size: int
    allocation_pct: float
    lookback_days: int
    warmup_buffer_days: int
    daemon_hour_utc: int
    daemon_minute_utc: int
    max_episode_steps: int
    fill_slippage_bps: float
    fee_rate: float
    max_leverage: float
    periods_per_year: float
    action_max_offset_bps: float
    allow_unsafe_checkpoint_loading: bool
    once: bool
    daemon: bool
    dry_run: bool
    check_config: bool
    print_config: bool
    configured_run_mode: MixedDailyRunMode
    suggested_run_mode: MixedDailyRunMode
    daemon_schedule_utc: str
    summary: str
    check_command_preview: str
    run_command_preview: str
    safe_command_preview: str
    next_steps: list[str]
    ready: bool
    errors: list[str]
    warnings: list[str]


SignalDirection = Literal["LONG", "SHORT", "FLAT"]
ExecutableDirection = Literal["LONG", "SHORT"]
TradeExecutionStatus = Literal["submitted", "skipped", "rejected"]
MixedDailyExecutionAttemptKind = Literal["primary", "alternative"]


class ExecutableSignal(TypedDict):
    direction: ExecutableDirection
    symbol: str
    confidence: float


class InferenceSignal(TypedDict):
    direction: SignalDirection
    symbol: str | None
    confidence: float
    value: float
    action: int
    timestamp: str
    all_probs: list[float]
    sym_names: list[str]


class _BrokerAccount(Protocol):
    portfolio_value: object
    buying_power: object


class _BrokerPosition(Protocol):
    qty: object


class _BrokerQuote(Protocol):
    ask_price: object
    bid_price: object


class _BrokerOrderSide(Protocol):
    value: str


class _BrokerSubmittedOrder(Protocol):
    id: object
    status: object
    side: object
    qty: object
    symbol: object
    type: object


_BarsResponse: TypeAlias = Mapping[str, Sequence[object]]
_BarsRequestFactory: TypeAlias = Callable[..., object]
_LimitOrderRequestFactory: TypeAlias = Callable[..., object]
_LatestDataFn: TypeAlias = Callable[[str], _BrokerQuote]
_MidpointLimitPriceFn: TypeAlias = Callable[[str, str, float], object]
_TimeInForceFn: TypeAlias = Callable[[float, str], object]


class _TradingClientProtocol(Protocol):
    def get_account(self) -> _BrokerAccount: ...

    def get_open_position(self, symbol: str) -> _BrokerPosition: ...

    def submit_order(self, order: object) -> _BrokerSubmittedOrder: ...


class _StockDataClientProtocol(Protocol):
    def get_stock_bars(self, request: object) -> _BarsResponse: ...


class _CryptoDataClientProtocol(Protocol):
    def get_crypto_bars(self, request: object) -> _BarsResponse: ...


@dataclass(frozen=True)
class TradeExecutionResult:
    submitted: bool
    status: TradeExecutionStatus
    reason: str
    symbol: str
    alpaca_symbol: str
    direction: ExecutableDirection
    confidence: float | None
    is_crypto: bool
    trade_value: float | None = None
    price: float | None = None
    qty: float | None = None
    limit_price: float | None = None
    order_id: str | None = None
    order_status: str | None = None
    log_write_error: str | None = None


@dataclass(frozen=True)
class _MixedDailyBrokerApi:
    trading_client: _TradingClientProtocol
    order_type_limit: object
    order_side_buy: _BrokerOrderSide
    order_side_sell: _BrokerOrderSide
    limit_order_request: _LimitOrderRequestFactory
    latest_data: _LatestDataFn
    midpoint_limit_price: _MidpointLimitPriceFn
    time_in_force_for_qty: _TimeInForceFn
    data_client: _StockDataClientProtocol
    crypto_client: _CryptoDataClientProtocol | None
    stock_bars_request: _BarsRequestFactory | None
    crypto_bars_request: _BarsRequestFactory | None
    timeframe_hour: object


@dataclass(frozen=True)
class _SignalExecutionCandidate:
    signal: ExecutableSignal
    attempt_kind: MixedDailyExecutionAttemptKind
    attempt_index: int


@dataclass(frozen=True)
class _SymbolPriceFrameCacheEntry:
    size: int
    mtime_ns: int
    frame: pd.DataFrame | None
    error: str | None = None


@dataclass
class _SymbolPriceFramePathLock:
    lock: threading.Lock


@dataclass(frozen=True)
class _MktdLayout:
    num_symbols: int
    num_timesteps: int
    features_per_sym: int
    price_features: int

    @property
    def symbol_table_size(self) -> int:
        return self.num_symbols * MKTD_SYMBOL_BYTES

    @property
    def features_section_bytes(self) -> int:
        return self.num_timesteps * self.num_symbols * self.features_per_sym * MKTD_FLOAT32_SIZE

    @property
    def price_row_bytes(self) -> int:
        return self.num_symbols * self.price_features * MKTD_FLOAT32_SIZE

    @property
    def last_price_offset(self) -> int:
        return (
            MKTD_HEADER_SIZE
            + self.symbol_table_size
            + self.features_section_bytes
            + (self.num_timesteps - 1) * self.price_row_bytes
        )


@dataclass(frozen=True)
class _MktdContext:
    layout: _MktdLayout
    symbol_names: list[str]
    last_prices: np.ndarray

    @property
    def num_symbols(self) -> int:
        return self.layout.num_symbols

    @property
    def num_timesteps(self) -> int:
        return self.layout.num_timesteps

    @property
    def features_per_sym(self) -> int:
        return self.layout.features_per_sym


@dataclass(frozen=True)
class _MixedDailyRuntimeOptions:
    hidden_size: int
    allocation_pct: float
    lookback_days: int
    warmup_buffer_days: int
    daemon_hour_utc: int
    daemon_minute_utc: int
    max_episode_steps: int
    fill_slippage_bps: float
    fee_rate: float
    max_leverage: float
    periods_per_year: float
    action_max_offset_bps: float
    allow_unsafe_checkpoint_loading: bool


@dataclass(frozen=True)
class _CheckpointPolicyShape:
    arch: str
    hidden_size: int
    obs_size: int
    num_actions: int | None


@dataclass(frozen=True)
class _CheckpointPayloadCacheEntry:
    size: int
    mtime_ns: int
    payload: dict[str, object]


@dataclass(frozen=True)
class _InferencePolicyCacheEntry:
    size: int
    mtime_ns: int
    policy: torch.nn.Module


_SYMBOL_PRICE_FRAME_CACHE: dict[Path, _SymbolPriceFrameCacheEntry] = {}
_SYMBOL_PRICE_FRAME_CACHE_LOCK = threading.Lock()
_SYMBOL_PRICE_FRAME_PATH_LOCKS: weakref.WeakValueDictionary[Path, _SymbolPriceFramePathLock] = (
    weakref.WeakValueDictionary()
)
_CHECKPOINT_CACHE_LOCK = threading.Lock()
_CHECKPOINT_PAYLOAD_CACHE_MAX_ENTRIES = 4
_INFERENCE_POLICY_CACHE_MAX_ENTRIES = 2
_CHECKPOINT_PAYLOAD_CACHE: OrderedDict[tuple[Path, bool], _CheckpointPayloadCacheEntry] = OrderedDict()
_INFERENCE_POLICY_CACHE: OrderedDict[
    tuple[Path, bool, int, int, int, str],
    _InferencePolicyCacheEntry,
] = OrderedDict()
_SYMBOL_PRICE_FRAME_COLUMNS = {"timestamp", "date", "open", "high", "low", "close", "volume"}


def _prune_ordered_cache(cache: OrderedDict[object, object], *, max_entries: int) -> None:
    while len(cache) > max_entries:
        cache.popitem(last=False)


def _build_executable_signal(direction: ExecutableDirection, symbol: str, confidence: float) -> ExecutableSignal:
    return {
        "direction": direction,
        "symbol": symbol,
        "confidence": confidence,
    }


def _as_executable_signal(signal: InferenceSignal) -> ExecutableSignal | None:
    symbol = signal["symbol"]
    direction = signal["direction"]
    if symbol is None or direction == "FLAT":
        return None
    return _build_executable_signal(cast(ExecutableDirection, direction), symbol, signal["confidence"])


def _decode_inference_action_signal(signal: InferenceSignal, action: int) -> ExecutableSignal | None:
    probs = signal["all_probs"]
    symbol_names = signal["sym_names"]
    symbol_count = len(symbol_names)
    if action <= 0 or action >= len(probs):
        return None
    if action <= symbol_count:
        return _build_executable_signal("LONG", symbol_names[action - 1], float(probs[action]))
    short_index = action - symbol_count - 1
    if short_index < 0 or short_index >= symbol_count:
        return None
    return _build_executable_signal("SHORT", symbol_names[short_index], float(probs[action]))


def _build_signal_execution_candidates(
    signal: InferenceSignal,
    *,
    max_alternatives: int = MIXED_DAILY_ALTERNATIVE_SIGNAL_LIMIT,
) -> list[_SignalExecutionCandidate]:
    primary = _as_executable_signal(signal)
    if primary is None:
        return []

    candidates = [_SignalExecutionCandidate(signal=primary, attempt_kind="primary", attempt_index=0)]
    probs = np.asarray(signal["all_probs"], dtype=np.float64)
    if probs.ndim != 1 or probs.size == 0 or max_alternatives <= 0:
        return candidates

    skipped_actions = {0, int(signal["action"])}
    alternative_count = 0
    for action_idx in np.argsort(probs)[::-1]:
        action = int(action_idx)
        if action in skipped_actions:
            continue
        skipped_actions.add(action)
        alternative_signal = _decode_inference_action_signal(signal, action)
        if alternative_signal is None:
            continue
        alternative_count += 1
        candidates.append(
            _SignalExecutionCandidate(
                signal=alternative_signal,
                attempt_kind="alternative",
                attempt_index=alternative_count,
            )
        )
        if alternative_count >= max_alternatives:
            break
    return candidates


def _normalize_mixed_symbol(raw_symbol: object) -> str:
    symbol = str(raw_symbol).strip().upper()
    if not symbol:
        raise ValueError("symbol is required")
    if ".." in symbol or "/" in symbol or "\\" in symbol:
        raise ValueError(f"Unsupported symbol: {raw_symbol}")
    if not SAFE_MIXED_SYMBOL_RE.fullmatch(symbol):
        raise ValueError(f"Unsupported symbol: {raw_symbol}")
    return symbol


def _normalize_symbols(raw_symbols: list[str] | None) -> tuple[list[str], list[str], list[str]]:
    if raw_symbols is None:
        raw_symbols = list(DEFAULT_SYMBOLS)
    normalized: list[str] = []
    removed_duplicates: list[str] = []
    ignored: list[str] = []
    seen: set[str] = set()
    removed_seen: set[str] = set()
    for raw in raw_symbols:
        text = str(raw or "").strip().upper()
        if not text:
            ignored.append(str(raw))
            continue
        symbol = _normalize_mixed_symbol(text)
        if symbol in seen:
            if symbol not in removed_seen:
                removed_duplicates.append(symbol)
                removed_seen.add(symbol)
            continue
        seen.add(symbol)
        normalized.append(symbol)
        if len(normalized) > MAX_MIXED_SYMBOL_COUNT:
            raise ValueError(
                "too many symbols after normalization: "
                f"{len(normalized)} > {MAX_MIXED_SYMBOL_COUNT}; "
                "reduce --symbols or --symbols-file"
            )
    return normalized, removed_duplicates, ignored


def _load_symbols_file(path: str | Path) -> list[str]:
    path_obj = Path(path)
    size = path_obj.stat().st_size
    if size > MAX_SYMBOLS_FILE_BYTES:
        raise ValueError(
            f"symbols_file exceeds {MAX_SYMBOLS_FILE_BYTES} bytes: {path_obj}"
        )
    values: list[str] = []
    for raw_line in path_obj.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.replace(",", " ").split():
            values.append(_normalize_mixed_symbol(token))
    if not values:
        raise ValueError(f"No valid symbols found in {path}")
    return values


def _resolve_symbol_data_path(data_dir: str | Path, symbol: str) -> Path | None:
    base = Path(data_dir)
    normalized_symbol = _normalize_mixed_symbol(symbol)
    for subdir in ["", "crypto", "stocks"]:
        path = base / subdir / f"{normalized_symbol}.csv" if subdir else base / f"{normalized_symbol}.csv"
        if path.exists():
            return path
    return None


def _cache_symbol_price_frame_error(
    cache_key: Path,
    *,
    size: int,
    mtime_ns: int,
    message: str,
) -> ValueError:
    _SYMBOL_PRICE_FRAME_CACHE[cache_key] = _SymbolPriceFrameCacheEntry(
        size=size,
        mtime_ns=mtime_ns,
        frame=None,
        error=message,
    )
    return ValueError(message)


def _lookup_symbol_price_frame_cache(
    cache_key: Path,
    *,
    size: int,
    mtime_ns: int,
) -> _SymbolPriceFrameCacheEntry | None:
    cached = _SYMBOL_PRICE_FRAME_CACHE.get(cache_key)
    if cached is None:
        return None
    if cached.size != size or cached.mtime_ns != mtime_ns:
        return None
    return cached


def _get_symbol_price_frame_path_lock(cache_key: Path) -> _SymbolPriceFramePathLock:
    with _SYMBOL_PRICE_FRAME_CACHE_LOCK:
        path_lock = _SYMBOL_PRICE_FRAME_PATH_LOCKS.get(cache_key)
        if path_lock is None:
            path_lock = _SymbolPriceFramePathLock(lock=threading.Lock())
            _SYMBOL_PRICE_FRAME_PATH_LOCKS[cache_key] = path_lock
        return path_lock


def _load_symbol_price_frame(path: Path, symbol: str) -> pd.DataFrame:
    normalized_symbol = _normalize_mixed_symbol(symbol)
    cache_key = path.expanduser().resolve(strict=False)
    try:
        stat_result = cache_key.stat()
    except OSError as exc:
        raise ValueError(f"{normalized_symbol}: unable to stat {path}: {exc}") from exc

    with _SYMBOL_PRICE_FRAME_CACHE_LOCK:
        cached = _lookup_symbol_price_frame_cache(
            cache_key,
            size=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
        )
    if cached is not None:
        if cached.error is not None:
            raise ValueError(cached.error)
        assert cached.frame is not None
        return cached.frame

    path_lock = _get_symbol_price_frame_path_lock(cache_key)
    with path_lock.lock:
        with _SYMBOL_PRICE_FRAME_CACHE_LOCK:
            cached = _lookup_symbol_price_frame_cache(
                cache_key,
                size=stat_result.st_size,
                mtime_ns=stat_result.st_mtime_ns,
            )
        if cached is not None:
            if cached.error is not None:
                raise ValueError(cached.error)
            assert cached.frame is not None
            return cached.frame

        try:
            frame = pd.read_csv(
                path,
                usecols=lambda column: str(column).lower() in _SYMBOL_PRICE_FRAME_COLUMNS,
            )
        except Exception as exc:  # pragma: no cover - pandas error types vary by version
            with _SYMBOL_PRICE_FRAME_CACHE_LOCK:
                raise _cache_symbol_price_frame_error(
                    cache_key,
                    size=stat_result.st_size,
                    mtime_ns=stat_result.st_mtime_ns,
                    message=f"{normalized_symbol}: unable to read {path}: {exc}",
                ) from exc

        frame.columns = [str(column).lower() for column in frame.columns]
        ts_col = "timestamp" if "timestamp" in frame.columns else "date" if "date" in frame.columns else None
        if ts_col is None:
            with _SYMBOL_PRICE_FRAME_CACHE_LOCK:
                raise _cache_symbol_price_frame_error(
                    cache_key,
                    size=stat_result.st_size,
                    mtime_ns=stat_result.st_mtime_ns,
                    message=f"{normalized_symbol}: {path} is missing timestamp/date column",
                )

        required = ["open", "high", "low", "close", "volume"]
        missing_columns = [column for column in required if column not in frame.columns]
        if missing_columns:
            with _SYMBOL_PRICE_FRAME_CACHE_LOCK:
                raise _cache_symbol_price_frame_error(
                    cache_key,
                    size=stat_result.st_size,
                    mtime_ns=stat_result.st_mtime_ns,
                    message=f"{normalized_symbol}: {path} is missing required columns: {', '.join(missing_columns)}",
                )

        try:
            frame["timestamp"] = pd.to_datetime(frame[ts_col], utc=True)
        except Exception as exc:  # pragma: no cover - pandas error types vary by version
            with _SYMBOL_PRICE_FRAME_CACHE_LOCK:
                raise _cache_symbol_price_frame_error(
                    cache_key,
                    size=stat_result.st_size,
                    mtime_ns=stat_result.st_mtime_ns,
                    message=f"{normalized_symbol}: {path} has invalid timestamps: {exc}",
                ) from exc

        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        if frame.empty:
            with _SYMBOL_PRICE_FRAME_CACHE_LOCK:
                raise _cache_symbol_price_frame_error(
                    cache_key,
                    size=stat_result.st_size,
                    mtime_ns=stat_result.st_mtime_ns,
                    message=f"{normalized_symbol}: {path} has no valid timestamped rows",
                )

        try:
            parsed_frame = frame.set_index("timestamp")[required].astype(float)
        except Exception as exc:  # pragma: no cover - pandas error types vary by version
            with _SYMBOL_PRICE_FRAME_CACHE_LOCK:
                raise _cache_symbol_price_frame_error(
                    cache_key,
                    size=stat_result.st_size,
                    mtime_ns=stat_result.st_mtime_ns,
                    message=f"{normalized_symbol}: {path} contains non-numeric OHLCV values: {exc}",
                ) from exc

        with _SYMBOL_PRICE_FRAME_CACHE_LOCK:
            _SYMBOL_PRICE_FRAME_CACHE[cache_key] = _SymbolPriceFrameCacheEntry(
                size=stat_result.st_size,
                mtime_ns=stat_result.st_mtime_ns,
                frame=parsed_frame,
            )
        return parsed_frame


def _is_crypto_symbol(symbol: str) -> bool:
    return symbol.endswith("USD")


def _configured_run_mode(args: argparse.Namespace) -> MixedDailyRunMode:
    if args.daemon:
        return "daemon"
    if args.dry_run:
        return "dry_run"
    if args.once:
        return "once"
    return "config_only"


def _suggested_run_mode(configured_run_mode: MixedDailyRunMode) -> MixedDailyRunMode:
    if configured_run_mode == "config_only":
        return "once"
    return configured_run_mode


def _format_daemon_schedule(hour: int, minute: int) -> str:
    return f"{hour:02d}:{minute:02d} UTC"


def _coerce_runtime_options(args: argparse.Namespace) -> _MixedDailyRuntimeOptions:
    return _MixedDailyRuntimeOptions(
        hidden_size=int(args.hidden_size),
        allocation_pct=float(args.allocation_pct),
        lookback_days=int(args.lookback_days),
        warmup_buffer_days=int(args.warmup_buffer_days),
        daemon_hour_utc=int(args.daemon_hour_utc),
        daemon_minute_utc=int(args.daemon_minute_utc),
        max_episode_steps=int(args.max_episode_steps),
        fill_slippage_bps=float(args.fill_slippage_bps),
        fee_rate=float(args.fee_rate),
        max_leverage=float(args.max_leverage),
        periods_per_year=float(args.periods_per_year),
        action_max_offset_bps=float(args.action_max_offset_bps),
        allow_unsafe_checkpoint_loading=bool(args.allow_unsafe_checkpoint_loading),
    )


def _build_preview_command(
    *,
    checkpoint: str,
    data_dir: str,
    symbols_file: str | None,
    symbol_source: MixedDailySymbolSource,
    symbols: list[str],
    runtime_options: _MixedDailyRuntimeOptions,
    run_mode: MixedDailyPreviewRunMode,
) -> str:
    command = [
        "python",
        "-u",
        "trade_mixed_daily_prod.py",
    ]
    if run_mode == "check_config":
        command.append("--check-config")
    elif run_mode == "daemon":
        command.append("--daemon")
    elif run_mode == "dry_run":
        command.extend(["--once", "--dry-run"])
    else:
        command.append("--once")
    command.extend(["--checkpoint", checkpoint, "--data-dir", data_dir])
    if runtime_options.allow_unsafe_checkpoint_loading:
        command.append("--allow-unsafe-checkpoint-loading")
    if symbol_source == "file" and symbols_file:
        command.extend(["--symbols-file", symbols_file])
    elif symbols:
        command.extend(["--symbols", *symbols])
    command.extend(
        [
            "--hidden-size",
            str(runtime_options.hidden_size),
            "--allocation-pct",
            str(runtime_options.allocation_pct),
            "--lookback-days",
            str(runtime_options.lookback_days),
            "--warmup-buffer-days",
            str(runtime_options.warmup_buffer_days),
            "--daemon-hour-utc",
            str(runtime_options.daemon_hour_utc),
            "--daemon-minute-utc",
            str(runtime_options.daemon_minute_utc),
            "--max-episode-steps",
            str(runtime_options.max_episode_steps),
            "--fill-slippage-bps",
            str(runtime_options.fill_slippage_bps),
            "--fee-rate",
            str(runtime_options.fee_rate),
            "--max-leverage",
            str(runtime_options.max_leverage),
            "--periods-per-year",
            str(runtime_options.periods_per_year),
            "--action-max-offset-bps",
            str(runtime_options.action_max_offset_bps),
        ]
    )
    return shlex.join(command)


def _build_next_steps(
    *,
    checkpoint_exists: bool,
    checkpoint: str,
    checkpoint_arch: str | None,
    checkpoint_num_actions: int | None,
    checkpoint_obs_size: int | None,
    checkpoint_inspection_error: str | None,
    expected_num_actions: int | None,
    expected_obs_size: int | None,
    data_dir_exists: bool,
    data_dir: str,
    missing_symbol_data: list[str],
    invalid_symbol_data: dict[str, str],
    stale_symbol_data: Mapping[str, str],
    symbol_source: MixedDailySymbolSource,
    symbols_file: str | None,
    ready: bool,
    check_command_preview: str,
    safe_command_preview: str,
    run_command_preview: str,
) -> list[str]:
    steps: list[str] = []
    if not checkpoint_exists:
        steps.append(f"Set --checkpoint to a trained mixed-daily policy file: {checkpoint}")
    else:
        steps.extend(
            _build_checkpoint_compatibility_next_steps(
                checkpoint=checkpoint,
                checkpoint_arch=checkpoint_arch,
                checkpoint_num_actions=checkpoint_num_actions,
                checkpoint_obs_size=checkpoint_obs_size,
                checkpoint_inspection_error=checkpoint_inspection_error,
                expected_num_actions=expected_num_actions,
                expected_obs_size=expected_obs_size,
            )
        )
    if not data_dir_exists:
        steps.append(f"Set --data-dir to a directory containing per-symbol OHLCV CSV files: {data_dir}")
    if symbol_source == "file" and symbols_file and not Path(symbols_file).exists():
        steps.append(f"Fix --symbols-file or create it with one symbol per line: {symbols_file}")
    if missing_symbol_data:
        preview = ", ".join(missing_symbol_data[:5])
        suffix = " ..." if len(missing_symbol_data) > 5 else ""
        steps.append(f"Add local CSVs for missing symbols under {data_dir}: {preview}{suffix}")
    if invalid_symbol_data:
        preview = ", ".join(sorted(invalid_symbol_data)[:5])
        suffix = " ..." if len(invalid_symbol_data) > 5 else ""
        steps.append(f"Repair unreadable local CSVs for: {preview}{suffix}")
    if stale_symbol_data:
        preview = ", ".join(sorted(stale_symbol_data)[:5])
        suffix = " ..." if len(stale_symbol_data) > 5 else ""
        steps.append(f"Refresh stale local CSVs for: {preview}{suffix}")
    if ready:
        steps.append(f"Dry run first: {safe_command_preview}")
        steps.append(f"Then run live: {run_command_preview}")
    else:
        steps.append(f"Re-check setup after fixes: {check_command_preview}")
    return steps


def _infer_symbol_count_from_num_actions(num_actions: int | None) -> int | None:
    if num_actions is None or num_actions < 1:
        return None
    action_count_without_flat = num_actions - 1
    if action_count_without_flat < 0 or action_count_without_flat % 2 != 0:
        return None
    symbol_count = action_count_without_flat // 2
    return symbol_count if symbol_count > 0 else None


def _build_checkpoint_compatibility_next_steps(
    *,
    checkpoint: str,
    checkpoint_arch: str | None,
    checkpoint_num_actions: int | None,
    checkpoint_obs_size: int | None,
    checkpoint_inspection_error: str | None,
    expected_num_actions: int | None,
    expected_obs_size: int | None,
) -> list[str]:
    steps: list[str] = []
    if checkpoint_inspection_error:
        steps.append(f"Replace or repair the checkpoint file: {checkpoint}")
        return steps

    if checkpoint_arch is not None and checkpoint_arch != MIXED_DAILY_SUPPORTED_CHECKPOINT_ARCH:
        steps.append(
            "Use a supported mixed-daily checkpoint architecture "
            f"({MIXED_DAILY_SUPPORTED_CHECKPOINT_ARCH!r}) instead of {checkpoint_arch!r}: {checkpoint}"
        )

    shape_mismatch = (
        (checkpoint_obs_size is not None and expected_obs_size is not None and checkpoint_obs_size != expected_obs_size)
        or (
            checkpoint_num_actions is not None
            and expected_num_actions is not None
            and checkpoint_num_actions != expected_num_actions
        )
    )
    if shape_mismatch:
        checkpoint_symbol_count = _infer_symbol_count_from_num_actions(checkpoint_num_actions)
        runtime_symbol_count = _infer_symbol_count_from_num_actions(expected_num_actions)
        if checkpoint_symbol_count is not None and runtime_symbol_count is not None:
            steps.append(
                "Use a checkpoint trained for this symbol set, or change --symbols / --symbols-file to match the "
                f"checkpoint (checkpoint: {checkpoint_symbol_count} symbols, runtime: {runtime_symbol_count})"
            )
        else:
            steps.append(
                "Use a checkpoint whose policy shape matches this runtime, or change --symbols / --symbols-file "
                "to match the checkpoint"
            )
    return steps


def _build_symbol_details(
    *,
    symbols: Sequence[str],
    latest_local_data_date: str | None,
    symbol_local_data_dates: Mapping[str, str],
    missing_symbol_data: Sequence[str],
    invalid_symbol_data: Mapping[str, str],
    stale_symbol_data: Mapping[str, str],
) -> dict[str, MixedDailySymbolDetail]:
    details: dict[str, MixedDailySymbolDetail] = {}
    missing_symbols = set(missing_symbol_data)
    for symbol in symbols:
        asset_class: Literal["stock", "crypto"] = "crypto" if _is_crypto_symbol(symbol) else "stock"
        if symbol in invalid_symbol_data:
            details[symbol] = MixedDailySymbolDetail(
                asset_class=asset_class,
                status="invalid",
                local_data_date=None,
                reason=invalid_symbol_data[symbol],
            )
            continue
        if symbol in missing_symbols:
            details[symbol] = MixedDailySymbolDetail(
                asset_class=asset_class,
                status="missing",
                local_data_date=None,
                reason="missing local CSV data",
            )
            continue

        local_data_date = symbol_local_data_dates.get(symbol)
        if symbol in stale_symbol_data:
            stale_reason = "local CSV data lags freshest symbol date"
            if latest_local_data_date is not None:
                stale_reason += f" {latest_local_data_date}"
            details[symbol] = MixedDailySymbolDetail(
                asset_class=asset_class,
                status="stale",
                local_data_date=local_data_date,
                reason=stale_reason,
            )
            continue

        details[symbol] = MixedDailySymbolDetail(
            asset_class=asset_class,
            status="usable",
            local_data_date=local_data_date,
            reason=None,
        )
    return details


def _infer_checkpoint_num_actions(model_state: Mapping[str, object]) -> int | None:
    final_actor_layer: tuple[int, int] | None = None
    for key, value in model_state.items():
        key_str = str(key)
        if not key_str.startswith("actor.") or not key_str.endswith(".weight"):
            continue
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            continue
        parts = key_str.split(".")
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        layer_index = int(parts[1])
        if final_actor_layer is None or layer_index > final_actor_layer[0]:
            final_actor_layer = (layer_index, int(value.shape[0]))
    return None if final_actor_layer is None else final_actor_layer[1]


def _inspect_checkpoint_policy_shape(
    checkpoint: str | Path,
    *,
    allow_unsafe_checkpoint_loading: bool = False,
) -> _CheckpointPolicyShape:
    payload = _load_checkpoint_payload(
        checkpoint,
        torch.device("cpu"),
        allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
    )
    model_state = payload.get("model")
    checkpoint_path = Path(checkpoint).expanduser()
    if not isinstance(model_state, Mapping):
        raise ValueError(f"{checkpoint_path}: checkpoint is missing a model state dict")

    encoder_weight = model_state.get("encoder.0.weight")
    input_proj_weight = model_state.get("input_proj.weight")
    if isinstance(encoder_weight, torch.Tensor) and encoder_weight.ndim == 2:
        arch = "mlp"
        hidden_size = int(encoder_weight.shape[0])
        obs_size = int(encoder_weight.shape[1])
    elif isinstance(input_proj_weight, torch.Tensor) and input_proj_weight.ndim == 2:
        arch = "resmlp"
        hidden_size = int(input_proj_weight.shape[0])
        obs_size = int(input_proj_weight.shape[1])
    else:
        raise ValueError(f"{checkpoint_path}: could not infer checkpoint policy input size from model weights")

    return _CheckpointPolicyShape(
        arch=arch,
        hidden_size=hidden_size,
        obs_size=obs_size,
        num_actions=_infer_checkpoint_num_actions(model_state),
    )


def _expected_obs_size_for_symbols(symbol_count: int) -> int | None:
    if symbol_count <= 0:
        return None
    from pufferlib_market.export_data_daily import FEATURES_PER_SYM

    return symbol_count * FEATURES_PER_SYM + MIXED_ENV_PORTFOLIO_FEATURES + symbol_count


def _expected_num_actions_for_symbols(symbol_count: int) -> int | None:
    if symbol_count <= 0:
        return None
    return 1 + 2 * symbol_count


def _build_runtime_config(args: argparse.Namespace) -> MixedDailyRuntimeConfig:
    checkpoint_path = Path(args.checkpoint).expanduser()
    data_dir_path = Path(args.data_dir).expanduser()
    symbols_file_path = str(Path(args.symbols_file).expanduser()) if args.symbols_file else None
    configured_run_mode: MixedDailyRunMode = _configured_run_mode(args)
    suggested_run_mode: MixedDailyRunMode = _suggested_run_mode(configured_run_mode)
    runtime_options = _coerce_runtime_options(args)
    daemon_schedule_utc = _format_daemon_schedule(
        runtime_options.daemon_hour_utc,
        runtime_options.daemon_minute_utc,
    )
    errors: list[str] = []
    warnings: list[str] = []
    symbol_source: MixedDailySymbolSource = "default"
    symbol_inputs: list[str] | None = args.symbols
    removed_duplicates: list[str] = []
    ignored_symbol_inputs: list[str] = []
    symbols: list[str] = []
    requested_hidden_size = runtime_options.hidden_size
    effective_hidden_size = requested_hidden_size
    checkpoint_arch: str | None = None
    checkpoint_hidden_size: int | None = None
    checkpoint_obs_size: int | None = None
    checkpoint_num_actions: int | None = None
    checkpoint_matches_runtime: bool | None = None
    checkpoint_inspection_error: str | None = None

    if args.symbols_file:
        symbol_source = "file"
        if args.symbols:
            warnings.append("ignoring inline symbols because --symbols-file was provided")
        try:
            symbol_inputs = _load_symbols_file(args.symbols_file)
        except (FileNotFoundError, OSError) as exc:
            errors.append(f"symbols_file could not be read: {exc}")
            symbol_inputs = []
        except ValueError as exc:
            errors.append(str(exc))
            symbol_inputs = []
    elif args.symbols is not None:
        symbol_source = "cli"

    try:
        symbols, removed_duplicates, ignored_symbol_inputs = _normalize_symbols(symbol_inputs)
    except ValueError as exc:
        errors.append(str(exc))
        symbols = []

    if not checkpoint_path.exists():
        errors.append(f"checkpoint not found: {checkpoint_path}")
    if not data_dir_path.exists():
        errors.append(f"data_dir not found: {data_dir_path}")
    if runtime_options.hidden_size <= 0:
        errors.append(f"hidden_size must be > 0 (got {runtime_options.hidden_size})")
    if runtime_options.allocation_pct <= 0:
        errors.append(f"allocation_pct must be > 0 (got {runtime_options.allocation_pct})")
    if runtime_options.lookback_days <= 0:
        errors.append(f"lookback_days must be > 0 (got {runtime_options.lookback_days})")
    if runtime_options.warmup_buffer_days < 0:
        errors.append(f"warmup_buffer_days must be >= 0 (got {runtime_options.warmup_buffer_days})")
    if runtime_options.max_episode_steps <= 0:
        errors.append(f"max_episode_steps must be > 0 (got {runtime_options.max_episode_steps})")
    if runtime_options.fill_slippage_bps < 0:
        errors.append(f"fill_slippage_bps must be >= 0 (got {runtime_options.fill_slippage_bps})")
    if runtime_options.fee_rate < 0:
        errors.append(f"fee_rate must be >= 0 (got {runtime_options.fee_rate})")
    if runtime_options.max_leverage <= 0:
        errors.append(f"max_leverage must be > 0 (got {runtime_options.max_leverage})")
    if runtime_options.periods_per_year <= 0:
        errors.append(f"periods_per_year must be > 0 (got {runtime_options.periods_per_year})")
    if runtime_options.action_max_offset_bps < 0:
        errors.append(f"action_max_offset_bps must be >= 0 (got {runtime_options.action_max_offset_bps})")
    if runtime_options.allow_unsafe_checkpoint_loading:
        warnings.append("unsafe checkpoint loading enabled; only use trusted checkpoint files")
    if not 0 <= runtime_options.daemon_hour_utc <= 23:
        errors.append(f"daemon_hour_utc must be between 0 and 23 (got {runtime_options.daemon_hour_utc})")
    if not 0 <= runtime_options.daemon_minute_utc <= 59:
        errors.append(f"daemon_minute_utc must be between 0 and 59 (got {runtime_options.daemon_minute_utc})")
    if not symbols:
        errors.append("at least one non-empty symbol is required")
    usable_symbol_frames: dict[str, pd.DataFrame] = {}
    symbol_local_data_dates: dict[str, str] = {}
    missing_symbol_data: list[str] = []
    invalid_symbol_data: dict[str, str] = {}
    for symbol in symbols:
        path = _resolve_symbol_data_path(data_dir_path, symbol)
        if path is None:
            missing_symbol_data.append(symbol)
            continue
        try:
            frame = _load_symbol_price_frame(path, symbol)
        except ValueError as exc:
            invalid_symbol_data[symbol] = str(exc)
        else:
            usable_symbol_frames[symbol] = frame
            symbol_local_data_dates[symbol] = frame.index.max().date().isoformat()
    if missing_symbol_data:
        warnings.append(f"missing local CSV data for {len(missing_symbol_data)} symbols")
    if invalid_symbol_data:
        warnings.append(f"found unreadable local CSV data for {len(invalid_symbol_data)} symbols")
    if symbols and len(missing_symbol_data) + len(invalid_symbol_data) == len(symbols):
        errors.append("no usable local CSV data found for any requested symbol")
    usable_symbols = list(usable_symbol_frames)
    usable_symbol_count = len(usable_symbols)
    latest_local_data_date: str | None = None
    oldest_local_data_date: str | None = None
    stale_symbol_data: dict[str, str] = {}
    if usable_symbol_frames:
        latest_timestamps = {symbol: frame.index.max() for symbol, frame in usable_symbol_frames.items()}
        newest_timestamp = max(latest_timestamps.values())
        oldest_timestamp = min(latest_timestamps.values())
        latest_local_data_date = newest_timestamp.date().isoformat()
        oldest_local_data_date = oldest_timestamp.date().isoformat()
        stale_symbol_data = {
            symbol: timestamp.date().isoformat()
            for symbol, timestamp in latest_timestamps.items()
            if timestamp < newest_timestamp
        }
    if stale_symbol_data and latest_local_data_date is not None:
        warnings.append(
            f"local CSV data for {len(stale_symbol_data)} symbols lags freshest date {latest_local_data_date}"
        )
    symbol_details = _build_symbol_details(
        symbols=symbols,
        latest_local_data_date=latest_local_data_date,
        symbol_local_data_dates=symbol_local_data_dates,
        missing_symbol_data=missing_symbol_data,
        invalid_symbol_data=invalid_symbol_data,
        stale_symbol_data=stale_symbol_data,
    )
    if removed_duplicates:
        warnings.append(f"removed {len(removed_duplicates)} duplicate symbols")
    if ignored_symbol_inputs:
        warnings.append(f"ignored {len(ignored_symbol_inputs)} blank symbol inputs")

    crypto_symbol_count = sum(1 for symbol in symbols if _is_crypto_symbol(symbol))
    stock_symbol_count = len(symbols) - crypto_symbol_count
    expected_obs_size = _expected_obs_size_for_symbols(len(symbols))
    expected_num_actions = _expected_num_actions_for_symbols(len(symbols))

    if checkpoint_path.exists():
        try:
            checkpoint_shape = _inspect_checkpoint_policy_shape(
                checkpoint_path,
                allow_unsafe_checkpoint_loading=runtime_options.allow_unsafe_checkpoint_loading,
            )
        except ValueError as exc:
            checkpoint_inspection_error = str(exc)
            errors.append(checkpoint_inspection_error)
        else:
            checkpoint_arch = checkpoint_shape.arch
            checkpoint_hidden_size = checkpoint_shape.hidden_size
            checkpoint_obs_size = checkpoint_shape.obs_size
            checkpoint_num_actions = checkpoint_shape.num_actions
            effective_hidden_size = checkpoint_hidden_size
            if checkpoint_hidden_size != requested_hidden_size:
                warnings.append(
                    f"checkpoint hidden_size={checkpoint_hidden_size} overrides requested hidden_size={requested_hidden_size}"
                )
            checkpoint_issues: list[str] = []
            if checkpoint_arch != MIXED_DAILY_SUPPORTED_CHECKPOINT_ARCH:
                checkpoint_issues.append(
                    f"checkpoint architecture {checkpoint_arch!r} is not supported by this mixed-daily runtime "
                    f"(expected {MIXED_DAILY_SUPPORTED_CHECKPOINT_ARCH!r})"
                )
            if expected_obs_size is not None and checkpoint_obs_size != expected_obs_size:
                checkpoint_issues.append(
                    f"checkpoint obs_size={checkpoint_obs_size} does not match current runtime obs_size={expected_obs_size}"
                )
            if (
                expected_num_actions is not None
                and checkpoint_num_actions is not None
                and checkpoint_num_actions != expected_num_actions
            ):
                checkpoint_issues.append(
                    "checkpoint num_actions="
                    f"{checkpoint_num_actions} does not match current runtime num_actions={expected_num_actions}"
                )
            checkpoint_matches_runtime = not checkpoint_issues
            errors.extend(checkpoint_issues)

    summary = (
        f"{'ready' if not errors else 'not ready'}: {configured_run_mode} config for {len(symbols)} symbols "
        f"({stock_symbol_count} stock, {crypto_symbol_count} crypto) from {symbol_source}; "
        f"usable local data for {usable_symbol_count}/{len(symbols)} symbols"
    )
    if latest_local_data_date is not None:
        summary += f" through {latest_local_data_date}"
        if stale_symbol_data:
            summary += f" ({len(stale_symbol_data)} stale)"
    check_command_preview = _build_preview_command(
        checkpoint=str(checkpoint_path),
        data_dir=str(data_dir_path),
        symbols_file=symbols_file_path,
        symbol_source=symbol_source,
        symbols=symbols,
        runtime_options=replace(runtime_options, hidden_size=effective_hidden_size),
        run_mode="check_config",
    )
    run_command_preview = _build_preview_command(
        checkpoint=str(checkpoint_path),
        data_dir=str(data_dir_path),
        symbols_file=symbols_file_path,
        symbol_source=symbol_source,
        symbols=symbols,
        runtime_options=replace(runtime_options, hidden_size=effective_hidden_size),
        run_mode=suggested_run_mode,
    )
    safe_command_preview = _build_preview_command(
        checkpoint=str(checkpoint_path),
        data_dir=str(data_dir_path),
        symbols_file=symbols_file_path,
        symbol_source=symbol_source,
        symbols=symbols,
        runtime_options=replace(runtime_options, hidden_size=effective_hidden_size),
        run_mode="dry_run",
    )
    ready = not errors
    next_steps = _build_next_steps(
        checkpoint_exists=checkpoint_path.exists(),
        checkpoint=str(checkpoint_path),
        checkpoint_arch=checkpoint_arch,
        checkpoint_num_actions=checkpoint_num_actions,
        checkpoint_obs_size=checkpoint_obs_size,
        checkpoint_inspection_error=checkpoint_inspection_error,
        expected_num_actions=expected_num_actions,
        expected_obs_size=expected_obs_size,
        data_dir_exists=data_dir_path.exists(),
        data_dir=str(data_dir_path),
        missing_symbol_data=missing_symbol_data,
        invalid_symbol_data=invalid_symbol_data,
        stale_symbol_data=stale_symbol_data,
        symbol_source=symbol_source,
        symbols_file=symbols_file_path,
        ready=ready,
        check_command_preview=check_command_preview,
        safe_command_preview=safe_command_preview,
        run_command_preview=run_command_preview,
    )

    return MixedDailyRuntimeConfig(
        checkpoint=str(checkpoint_path),
        checkpoint_exists=checkpoint_path.exists(),
        checkpoint_arch=checkpoint_arch,
        checkpoint_hidden_size=checkpoint_hidden_size,
        checkpoint_obs_size=checkpoint_obs_size,
        checkpoint_num_actions=checkpoint_num_actions,
        checkpoint_matches_runtime=checkpoint_matches_runtime,
        checkpoint_inspection_error=checkpoint_inspection_error,
        expected_obs_size=expected_obs_size,
        expected_num_actions=expected_num_actions,
        data_dir=str(data_dir_path),
        data_dir_exists=data_dir_path.exists(),
        symbols_file=symbols_file_path,
        symbol_source=symbol_source,
        symbols=symbols,
        symbol_count=len(symbols),
        stock_symbol_count=stock_symbol_count,
        crypto_symbol_count=crypto_symbol_count,
        usable_symbols=usable_symbols,
        usable_symbol_count=usable_symbol_count,
        missing_symbol_data=missing_symbol_data,
        invalid_symbol_data=invalid_symbol_data,
        latest_local_data_date=latest_local_data_date,
        oldest_local_data_date=oldest_local_data_date,
        stale_symbol_data=stale_symbol_data,
        stale_symbol_count=len(stale_symbol_data),
        symbol_details=symbol_details,
        removed_duplicate_symbols=removed_duplicates,
        ignored_symbol_inputs=ignored_symbol_inputs,
        hidden_size=effective_hidden_size,
        requested_hidden_size=requested_hidden_size,
        allocation_pct=runtime_options.allocation_pct,
        lookback_days=runtime_options.lookback_days,
        warmup_buffer_days=runtime_options.warmup_buffer_days,
        daemon_hour_utc=runtime_options.daemon_hour_utc,
        daemon_minute_utc=runtime_options.daemon_minute_utc,
        max_episode_steps=runtime_options.max_episode_steps,
        fill_slippage_bps=runtime_options.fill_slippage_bps,
        fee_rate=runtime_options.fee_rate,
        max_leverage=runtime_options.max_leverage,
        periods_per_year=runtime_options.periods_per_year,
        action_max_offset_bps=runtime_options.action_max_offset_bps,
        allow_unsafe_checkpoint_loading=runtime_options.allow_unsafe_checkpoint_loading,
        once=bool(args.once),
        daemon=bool(args.daemon),
        dry_run=bool(args.dry_run),
        check_config=bool(args.check_config),
        print_config=bool(args.print_config),
        configured_run_mode=configured_run_mode,
        suggested_run_mode=suggested_run_mode,
        daemon_schedule_utc=daemon_schedule_utc,
        summary=summary,
        check_command_preview=check_command_preview,
        run_command_preview=run_command_preview,
        safe_command_preview=safe_command_preview,
        next_steps=next_steps,
        ready=ready,
        errors=errors,
        warnings=warnings,
    )


def _load_checkpoint_payload(
    checkpoint: str | Path,
    device: torch.device,
    *,
    allow_unsafe_checkpoint_loading: bool = False,
) -> dict[str, object]:
    checkpoint_path = Path(checkpoint).expanduser()
    del device  # checkpoint payloads are cached on CPU and moved into modules as needed
    resolved_path = checkpoint_path.resolve(strict=False)
    try:
        stat_result = resolved_path.stat()
    except OSError as exc:
        raise ValueError(f"{checkpoint_path}: unable to stat checkpoint: {exc}") from exc

    cache_key = (resolved_path, allow_unsafe_checkpoint_loading)
    with _CHECKPOINT_CACHE_LOCK:
        cached = _CHECKPOINT_PAYLOAD_CACHE.get(cache_key)
        if cached is not None and cached.size == stat_result.st_size and cached.mtime_ns == stat_result.st_mtime_ns:
            _CHECKPOINT_PAYLOAD_CACHE.move_to_end(cache_key)
            return cached.payload

    load_kwargs = {"map_location": torch.device("cpu")}
    if allow_unsafe_checkpoint_loading:
        payload = torch.load(checkpoint_path, weights_only=False, **load_kwargs)
    else:
        try:
            payload = torch.load(checkpoint_path, weights_only=True, **load_kwargs)
        except Exception as exc:
            raise ValueError(
                f"{checkpoint_path}: safe checkpoint loading failed: {exc}. "
                "Re-run with --allow-unsafe-checkpoint-loading only if you trust this file."
            ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{checkpoint_path}: expected checkpoint dict, got {type(payload).__name__}")
    model_state = payload.get("model")
    if not isinstance(model_state, dict):
        raise ValueError(f"{checkpoint_path}: checkpoint is missing a model state dict")
    with _CHECKPOINT_CACHE_LOCK:
        _CHECKPOINT_PAYLOAD_CACHE[cache_key] = _CheckpointPayloadCacheEntry(
            size=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
            payload=payload,
        )
        _CHECKPOINT_PAYLOAD_CACHE.move_to_end(cache_key)
        _prune_ordered_cache(_CHECKPOINT_PAYLOAD_CACHE, max_entries=_CHECKPOINT_PAYLOAD_CACHE_MAX_ENTRIES)
    return payload


def _load_inference_policy(
    checkpoint: str | Path,
    *,
    obs_size: int,
    num_actions: int,
    hidden_size: int,
    device: torch.device,
    allow_unsafe_checkpoint_loading: bool = False,
) -> torch.nn.Module:
    from pufferlib_market.train import TradingPolicy

    checkpoint_path = Path(checkpoint).expanduser()
    resolved_path = checkpoint_path.resolve(strict=False)
    try:
        stat_result = resolved_path.stat()
    except OSError as exc:
        raise ValueError(f"{checkpoint_path}: unable to stat checkpoint: {exc}") from exc

    cache_key = (
        resolved_path,
        allow_unsafe_checkpoint_loading,
        obs_size,
        num_actions,
        hidden_size,
        str(device),
    )
    with _CHECKPOINT_CACHE_LOCK:
        cached = _INFERENCE_POLICY_CACHE.get(cache_key)
        if cached is not None and cached.size == stat_result.st_size and cached.mtime_ns == stat_result.st_mtime_ns:
            _INFERENCE_POLICY_CACHE.move_to_end(cache_key)
            return cached.policy

    ckpt = _load_checkpoint_payload(
        checkpoint,
        device,
        allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
    )
    policy = TradingPolicy(obs_size, num_actions, hidden_size)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    policy.to(device)

    with _CHECKPOINT_CACHE_LOCK:
        _INFERENCE_POLICY_CACHE[cache_key] = _InferencePolicyCacheEntry(
            size=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
            policy=policy,
        )
        _INFERENCE_POLICY_CACHE.move_to_end(cache_key)
        _prune_ordered_cache(_INFERENCE_POLICY_CACHE, max_entries=_INFERENCE_POLICY_CACHE_MAX_ENTRIES)
    return policy


def _make_temp_data_bin_path() -> Path:
    with tempfile.NamedTemporaryFile(prefix=TEMP_BIN_PREFIX, suffix=".bin", delete=False) as handle:
        path = Path(handle.name)
    path.unlink(missing_ok=True)
    return path


def _cleanup_temp_data_bin(path: str | Path) -> None:
    Path(path).unlink(missing_ok=True)


def _pack_mktd_header(layout: _MktdLayout) -> bytes:
    return struct.pack(
        MKTD_HEADER_FORMAT,
        MKTD_MAGIC,
        MKTD_VERSION,
        layout.num_symbols,
        layout.num_timesteps,
        layout.features_per_sym,
        layout.price_features,
        b"\x00" * MKTD_HEADER_RESERVED_BYTES,
    )


def _emit_runtime_event(
    event: str,
    *,
    config: MixedDailyRuntimeConfig,
    mode: str,
    stage: str,
    **extra: object,
) -> None:
    payload: dict[str, object] = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "stage": stage,
        "checkpoint": config.checkpoint,
        "checkpoint_arch": config.checkpoint_arch,
        "checkpoint_hidden_size": config.checkpoint_hidden_size,
        "checkpoint_obs_size": config.checkpoint_obs_size,
        "checkpoint_num_actions": config.checkpoint_num_actions,
        "checkpoint_matches_runtime": config.checkpoint_matches_runtime,
        "expected_obs_size": config.expected_obs_size,
        "expected_num_actions": config.expected_num_actions,
        "data_dir": config.data_dir,
        "allow_unsafe_checkpoint_loading": config.allow_unsafe_checkpoint_loading,
        "symbol_count": config.symbol_count,
        "usable_symbol_count": config.usable_symbol_count,
        "stale_symbol_count": config.stale_symbol_count,
        "latest_local_data_date": config.latest_local_data_date,
        "oldest_local_data_date": config.oldest_local_data_date,
        "summary": config.summary,
        "run_command_preview": config.run_command_preview,
        "safe_command_preview": config.safe_command_preview,
    }
    payload.update(extra)
    print(json.dumps(payload, sort_keys=True))


def _emit_signal_event(
    *,
    config: MixedDailyRuntimeConfig,
    mode: MixedDailyRunMode,
    signal: InferenceSignal,
) -> None:
    _emit_runtime_event(
        "mixed_daily_signal",
        config=config,
        mode=mode,
        stage="decision",
        signal_direction=signal["direction"],
        signal_symbol=signal["symbol"],
        signal_confidence=signal["confidence"],
        signal_value=signal["value"],
        signal_action=signal["action"],
        signal_timestamp=signal["timestamp"],
    )


def _emit_trade_execution_event(
    *,
    config: MixedDailyRuntimeConfig,
    mode: MixedDailyRunMode,
    signal: ExecutableSignal,
    result: TradeExecutionResult,
    attempt_kind: str,
    attempt_index: int,
) -> None:
    _emit_runtime_event(
        "mixed_daily_trade_execution",
        config=config,
        mode=mode,
        stage="execution",
        attempt_kind=attempt_kind,
        attempt_index=attempt_index,
        signal_direction=signal["direction"],
        signal_symbol=signal["symbol"],
        signal_confidence=signal["confidence"],
        execution_submitted=result.submitted,
        execution_status=result.status,
        execution_reason=result.reason,
        alpaca_symbol=result.alpaca_symbol,
        is_crypto=result.is_crypto,
        trade_value=result.trade_value,
        price=result.price,
        qty=result.qty,
        limit_price=result.limit_price,
        broker_order_id=result.order_id,
        broker_order_status=result.order_status,
        log_write_error=result.log_write_error,
    )


def _format_runtime_preflight_failure(config: MixedDailyRuntimeConfig) -> str:
    lines = [f"Configuration not ready: {config.summary}"]
    if config.errors:
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in config.errors)
    if config.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in config.warnings)
    lines.extend(_format_checkpoint_compatibility_lines(config))
    if config.symbol_details:
        lines.extend(
            format_local_data_health_lines(
                symbol_details=config.symbol_details,
                usable_symbol_count=config.usable_symbol_count,
                latest_local_data_date=config.latest_local_data_date,
            )
        )
    lines.append(f"Check config: {config.check_command_preview}")
    if config.next_steps:
        lines.append("Next steps:")
        lines.extend(f"- {step}" for step in config.next_steps)
    return "\n".join(lines)


def _format_runtime_preflight_ready(config: MixedDailyRuntimeConfig) -> str:
    summary = config.summary
    if summary.startswith("ready: "):
        summary = summary[len("ready: ") :]

    lines = [f"Configuration ready: {summary}"]
    if config.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in config.warnings)
    lines.extend(_format_checkpoint_compatibility_lines(config))
    if config.symbol_details:
        lines.extend(
            format_local_data_health_lines(
                symbol_details=config.symbol_details,
                usable_symbol_count=config.usable_symbol_count,
                latest_local_data_date=config.latest_local_data_date,
            )
        )

    suggested_run_label = {
        "once": "one-off run",
        "daemon": "daemon run",
        "dry_run": "dry run",
        "check_config": "config check",
    }.get(config.suggested_run_mode, config.suggested_run_mode.replace("_", " "))
    lines.append("Suggested commands:")
    lines.append(f"- dry run: {config.safe_command_preview}")
    lines.append(f"- {suggested_run_label}: {config.run_command_preview}")

    command_steps = {
        f"Dry run first: {config.safe_command_preview}",
        f"Then run live: {config.run_command_preview}",
    }
    other_steps = [step for step in config.next_steps if step not in command_steps]
    if other_steps:
        lines.append("Additional next steps:")
        lines.extend(f"- {step}" for step in other_steps)
    return "\n".join(lines)


def _format_checkpoint_compatibility_lines(config: MixedDailyRuntimeConfig) -> list[str]:
    hidden_size_override = (
        config.checkpoint_hidden_size is not None
        and config.checkpoint_hidden_size != config.requested_hidden_size
    )
    should_render = (
        config.checkpoint_inspection_error is not None
        or config.checkpoint_matches_runtime is False
        or hidden_size_override
    )
    if not should_render:
        return []

    lines = ["Checkpoint compatibility:"]
    if config.checkpoint_inspection_error:
        lines.append(f"- inspection error: {config.checkpoint_inspection_error}")
        return lines

    if config.checkpoint_arch is not None:
        if config.checkpoint_arch == MIXED_DAILY_SUPPORTED_CHECKPOINT_ARCH:
            lines.append(f"- architecture: {config.checkpoint_arch}")
        else:
            lines.append(
                "- architecture: "
                f"checkpoint {config.checkpoint_arch}, runtime {MIXED_DAILY_SUPPORTED_CHECKPOINT_ARCH}"
            )

    if config.checkpoint_hidden_size is not None:
        if hidden_size_override:
            lines.append(
                "- hidden size: "
                f"checkpoint {config.checkpoint_hidden_size}, "
                f"requested {config.requested_hidden_size}, "
                f"runtime {config.hidden_size}"
            )
        else:
            lines.append(f"- hidden size: {config.hidden_size}")

    if config.checkpoint_obs_size is not None or config.expected_obs_size is not None:
        checkpoint_obs_size = (
            str(config.checkpoint_obs_size)
            if config.checkpoint_obs_size is not None
            else "unknown"
        )
        expected_obs_size = (
            str(config.expected_obs_size)
            if config.expected_obs_size is not None
            else "unknown"
        )
        lines.append(
            f"- obs size: checkpoint {checkpoint_obs_size}, runtime {expected_obs_size}"
        )

    if config.checkpoint_num_actions is not None or config.expected_num_actions is not None:
        checkpoint_num_actions = (
            str(config.checkpoint_num_actions)
            if config.checkpoint_num_actions is not None
            else "unknown"
        )
        expected_num_actions = (
            str(config.expected_num_actions)
            if config.expected_num_actions is not None
            else "unknown"
        )
        lines.append(
            "- action count: "
            f"checkpoint {checkpoint_num_actions}, runtime {expected_num_actions}"
        )

    return lines


def _format_runtime_execution_failure(
    config: MixedDailyRuntimeConfig,
    *,
    stage: str,
    error: BaseException,
) -> str:
    stage_label = stage.replace("_", " ")
    lines = [
        f"Mixed daily run failed during {stage_label}.",
        f"Error: {type(error).__name__}: {error}",
        f"Check config: {config.check_command_preview}",
    ]
    error_notes = _exception_notes(error)
    if error_notes:
        lines.append("Additional context:")
        lines.extend(f"- {note}" for note in error_notes)
    if config.next_steps:
        lines.append("Next steps:")
        lines.extend(f"- {step}" for step in config.next_steps)
    return "\n".join(lines)


def _exception_notes(error: BaseException) -> list[str]:
    notes = getattr(error, "__notes__", None)
    if not notes:
        return []
    return [str(note) for note in notes if str(note).strip()]


def _coerce_finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _load_mixed_daily_broker_api() -> _MixedDailyBrokerApi:
    from importlib import import_module

    import alpaca_wrapper as aw
    requests_module = import_module("alpaca.data.requests")
    timeframe_module = import_module("alpaca.data.timeframe")
    timeframe = getattr(timeframe_module, "TimeFrame")

    return _MixedDailyBrokerApi(
        trading_client=cast(_TradingClientProtocol, aw.TradingClient(aw.ALP_KEY_ID, aw.ALP_SECRET_KEY, paper=True)),
        order_type_limit=aw.OrderType.LIMIT,
        order_side_buy=cast(_BrokerOrderSide, aw.OrderSide.BUY),
        order_side_sell=cast(_BrokerOrderSide, aw.OrderSide.SELL),
        limit_order_request=cast(_LimitOrderRequestFactory, aw.LimitOrderRequest),
        latest_data=cast(_LatestDataFn, aw.latest_data),
        midpoint_limit_price=cast(_MidpointLimitPriceFn, aw._midpoint_limit_price),
        time_in_force_for_qty=cast(_TimeInForceFn, aw._get_time_in_force_for_qty),
        data_client=cast(_StockDataClientProtocol, aw.data_client),
        crypto_client=cast(_CryptoDataClientProtocol | None, getattr(aw, "crypto_client", None)),
        stock_bars_request=cast(_BarsRequestFactory | None, getattr(requests_module, "StockBarsRequest", None)),
        crypto_bars_request=cast(_BarsRequestFactory | None, getattr(requests_module, "CryptoBarsRequest", None)),
        timeframe_hour=getattr(timeframe, "Hour"),
    )


def _run_export_and_inference_cycle(config: MixedDailyRuntimeConfig) -> InferenceSignal:
    temp_data_bin = _make_temp_data_bin_path()
    primary_error: BaseException | None = None
    try:
        print(f"Exporting daily data for {len(config.symbols)} symbols...")
        data_bin, _, _ = export_live_binary(
            config.symbols,
            config.data_dir,
            str(temp_data_bin),
            lookback_days=config.lookback_days,
            warmup_buffer_days=config.warmup_buffer_days,
        )
        return run_inference(
            config.checkpoint,
            data_bin,
            config.hidden_size,
            max_episode_steps=config.max_episode_steps,
            fill_slippage_bps=config.fill_slippage_bps,
            fee_rate=config.fee_rate,
            max_leverage=config.max_leverage,
            periods_per_year=config.periods_per_year,
            action_max_offset_bps=config.action_max_offset_bps,
            allow_unsafe_checkpoint_loading=config.allow_unsafe_checkpoint_loading,
        )
    except Exception as exc:
        primary_error = exc
        raise
    finally:
        try:
            _cleanup_temp_data_bin(temp_data_bin)
        except OSError as cleanup_exc:
            message = f"Temporary data bin cleanup failed for {temp_data_bin}: {cleanup_exc}"
            if primary_error is not None:
                primary_error.add_note(message)
            else:
                print(f"WARNING: {message}", file=sys.stderr)


def export_live_binary(
    symbols: list[str],
    data_dir: str,
    output: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    warmup_buffer_days: int = DEFAULT_WARMUP_BUFFER_DAYS,
):
    """Export latest daily bars to MKTD binary for C env inference."""
    normalized_symbols = [_normalize_mixed_symbol(symbol) for symbol in symbols]
    from pufferlib_market.export_data_daily import compute_daily_features, FEATURES_PER_SYM, PRICE_FEATURES

    all_prices: dict[str, pd.DataFrame] = {}
    invalid_symbol_data: dict[str, str] = {}
    missing_symbols: list[str] = []
    for sym in normalized_symbols:
        p = _resolve_symbol_data_path(data_dir, sym)
        if p is None:
            missing_symbols.append(sym)
            continue
        try:
            all_prices[sym] = _load_symbol_price_frame(p, sym)
        except ValueError as exc:
            invalid_symbol_data[sym] = str(exc)

    if missing_symbols:
        print(f"WARNING: Missing data for {set(missing_symbols)}")
    if invalid_symbol_data:
        print(f"WARNING: Unusable local CSV data for {sorted(invalid_symbol_data)}")

    if not all_prices:
        details: list[str] = []
        if missing_symbols:
            details.append(f"missing data for {sorted(missing_symbols)}")
        if invalid_symbol_data:
            detail_text = "; ".join(invalid_symbol_data[symbol] for symbol in sorted(invalid_symbol_data))
            details.append(detail_text)
        detail_suffix = f": {'; '.join(details)}" if details else ""
        raise ValueError(f"No usable local CSV data found for requested symbols{detail_suffix}")

    # Use last lookback_days
    latest = max(df.index.max() for df in all_prices.values())
    start = latest - pd.Timedelta(days=lookback_days + warmup_buffer_days)
    full_index = pd.date_range(start.floor("D"), latest.floor("D"), freq="D", tz="UTC")

    layout = _MktdLayout(
        num_symbols=len(normalized_symbols),
        num_timesteps=len(full_index),
        features_per_sym=FEATURES_PER_SYM,
        price_features=PRICE_FEATURES,
    )

    feature_arr = np.zeros((layout.num_timesteps, layout.num_symbols, layout.features_per_sym), dtype=np.float32)
    price_arr = np.zeros((layout.num_timesteps, layout.num_symbols, layout.price_features), dtype=np.float32)
    mask_arr = np.ones((layout.num_timesteps, layout.num_symbols), dtype=np.uint8)

    for si, sym in enumerate(normalized_symbols):
        if sym not in all_prices:
            mask_arr[:, si] = 0
            continue
        df = all_prices[sym]
        mask = full_index.isin(df.index).astype(np.uint8)
        reindexed_df = df.reindex(full_index, method="ffill")
        reindexed_df["volume"] = reindexed_df["volume"].where(mask.astype(bool), 0.0)
        reindexed_df = reindexed_df.bfill().fillna(0.0)
        feats = compute_daily_features(reindexed_df)
        feature_arr[:, si, :] = feats.values.astype(np.float32, copy=False)
        price_arr[:, si, :] = reindexed_df[["open", "high", "low", "close", "volume"]].values.astype(
            np.float32, copy=False
        )
        mask_arr[:, si] = mask

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        f.write(_pack_mktd_header(layout))
        for sym in normalized_symbols:
            raw = sym.encode("ascii", errors="ignore")[:MKTD_SYMBOL_PAYLOAD_BYTES]
            f.write(raw.ljust(MKTD_SYMBOL_BYTES, b"\x00"))
        f.write(feature_arr.tobytes(order="C"))
        f.write(price_arr.tobytes(order="C"))
        f.write(mask_arr.tobytes(order="C"))

    print(f"Exported {output}: {layout.num_symbols} symbols, {layout.num_timesteps} days")
    print(f"  Latest date: {full_index[-1].strftime('%Y-%m-%d')}")
    return output, layout.num_symbols, layout.num_timesteps


def _read_mktd_context(data_bin: str | Path) -> _MktdContext:
    data_path = Path(data_bin)
    with data_path.open("rb") as handle:
        header = handle.read(MKTD_HEADER_SIZE)
        if len(header) < MKTD_HEADER_SIZE:
            raise ValueError(
                f"{data_path}: invalid MKTD header: expected {MKTD_HEADER_SIZE} bytes, found {len(header)}"
            )
        try:
            magic, version, num_symbols, num_timesteps, features_per_sym, price_features, _reserved = struct.unpack(
                MKTD_HEADER_FORMAT,
                header,
            )
        except struct.error as exc:
            raise ValueError(f"{data_path}: invalid MKTD header: {exc}") from exc
        if magic != MKTD_MAGIC:
            raise ValueError(f"{data_path}: invalid MKTD magic {magic!r}")
        if version != MKTD_VERSION:
            raise ValueError(f"{data_path}: unsupported MKTD version {version}")
        if num_symbols <= 0 or num_timesteps <= 0:
            raise ValueError(
                f"{data_path}: invalid MKTD dimensions: symbols={num_symbols}, timesteps={num_timesteps}"
            )
        if features_per_sym <= 0 or price_features < MKTD_REQUIRED_PRICE_FEATURES:
            raise ValueError(
                f"{data_path}: invalid MKTD feature layout: features_per_sym={features_per_sym}, "
                f"price_features={price_features}"
            )
        layout = _MktdLayout(
            num_symbols=num_symbols,
            num_timesteps=num_timesteps,
            features_per_sym=features_per_sym,
            price_features=price_features,
        )

        symbol_table = handle.read(layout.symbol_table_size)
        if len(symbol_table) < layout.symbol_table_size:
            raise ValueError(
                f"{data_path}: truncated MKTD symbol table: expected {layout.symbol_table_size} bytes, "
                f"found {len(symbol_table)}"
            )
        sym_names = [
            symbol_table[offset : offset + MKTD_SYMBOL_BYTES].decode("ascii", errors="ignore").rstrip("\x00")
            for offset in range(0, layout.symbol_table_size, MKTD_SYMBOL_BYTES)
        ]

        handle.seek(layout.last_price_offset)
        last_price_bytes = handle.read(layout.price_row_bytes)
        if len(last_price_bytes) < layout.price_row_bytes:
            raise ValueError(
                f"{data_path}: truncated MKTD price section: expected {layout.price_row_bytes} bytes, "
                f"found {len(last_price_bytes)}"
            )
        last_prices = (
            np.frombuffer(last_price_bytes, dtype=np.float32)
            .reshape(layout.num_symbols, layout.price_features)[:, :MKTD_REQUIRED_PRICE_FEATURES]
        )

    return _MktdContext(layout=layout, symbol_names=sym_names, last_prices=last_prices)


def run_inference(
    checkpoint: str,
    data_bin: str,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    *,
    max_episode_steps: int = DEFAULT_INFERENCE_MAX_STEPS,
    fill_slippage_bps: float = DEFAULT_FILL_SLIPPAGE_BPS,
    fee_rate: float = DEFAULT_FEE_RATE,
    max_leverage: float = DEFAULT_MAX_LEVERAGE,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
    action_max_offset_bps: float = DEFAULT_ACTION_MAX_OFFSET_BPS,
    allow_unsafe_checkpoint_loading: bool = False,
) -> InferenceSignal:
    """Run single-step C env inference to get today's signal."""
    import pufferlib_market.binding as binding

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mktd_context = _read_mktd_context(data_bin)
    num_symbols = mktd_context.num_symbols
    features_per_sym = mktd_context.features_per_sym
    sym_names = mktd_context.symbol_names
    last_prices = mktd_context.last_prices

    obs_size = num_symbols * features_per_sym + MIXED_ENV_PORTFOLIO_FEATURES + num_symbols
    num_actions = 1 + 2 * num_symbols

    policy = _load_inference_policy(
        checkpoint,
        obs_size=obs_size,
        num_actions=num_actions,
        hidden_size=hidden_size,
        device=device,
        allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
    )

    # Use C env for proper observation construction
    binding.shared(data_path=str(Path(data_bin).resolve()))

    # Run a single episode starting from the END of data (most recent)
    num_envs = 1
    obs_buf = np.zeros((num_envs, obs_size), dtype=np.float32)
    act_buf = np.zeros((num_envs,), dtype=np.int32)
    rew_buf = np.zeros((num_envs,), dtype=np.float32)
    term_buf = np.zeros((num_envs,), dtype=np.uint8)
    trunc_buf = np.zeros((num_envs,), dtype=np.uint8)

    vec_handle = binding.vec_init(
        obs_buf, act_buf, rew_buf, term_buf, trunc_buf,
        num_envs, MIXED_ENV_SEED,
        max_steps=max_episode_steps,
        fee_rate=fee_rate,
        max_leverage=max_leverage,
        periods_per_year=periods_per_year,
        action_allocation_bins=MIXED_ENV_ACTION_ALLOCATION_BINS,
        action_level_bins=MIXED_ENV_ACTION_LEVEL_BINS,
        action_max_offset_bps=action_max_offset_bps,
        fill_slippage_bps=fill_slippage_bps,
    )

    try:
        # Reset to get initial observation (starts near end of data)
        binding.vec_reset(vec_handle, MIXED_ENV_SEED)

        # Get policy action for current state
        obs_tensor = torch.from_numpy(obs_buf).to(device)
        with torch.inference_mode():
            encoded_obs = policy.encoder(obs_tensor)
            logits = policy.actor(encoded_obs)
            value = policy.critic(encoded_obs)

        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        action = int(logits.argmax().item())
        confidence = float(probs[action])

        # Decode action
        if action == 0:
            direction = "FLAT"
            symbol = None
        elif action <= num_symbols:
            direction = "LONG"
            symbol = sym_names[action - 1]
        else:
            direction = "SHORT"
            symbol = sym_names[action - num_symbols - 1]
    finally:
        binding.vec_close(vec_handle)

    # Build signal report
    print(f"\n{'='*60}")
    print(f"DAILY SIGNAL — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")
    print(f"  Direction:  {direction}")
    print(f"  Symbol:     {symbol or 'N/A (flat)'}")
    print(f"  Confidence: {confidence:.1%}")
    print(f"  Value est:  {float(value.item()):.4f}")

    if symbol:
        sym_idx = sym_names.index(symbol)
        price = last_prices[sym_idx]
        print(f"  Price:      O={price[0]:.2f} H={price[1]:.2f} L={price[2]:.2f} C={price[3]:.2f}")

    # Top 5 actions by probability
    print(f"\n  Top 5 actions:")
    top5 = np.argsort(probs)[-5:][::-1]
    for idx in top5:
        if idx == 0:
            label = "FLAT"
        elif idx <= num_symbols:
            label = f"LONG  {sym_names[idx-1]}"
        else:
            label = f"SHORT {sym_names[idx-num_symbols-1]}"
        print(f"    {label:20s} prob={probs[idx]:.3f}")

    return {
        "direction": direction,
        "symbol": symbol,
        "confidence": confidence,
        "value": float(value.item()),
        "action": action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_probs": probs.tolist(),
        "sym_names": sym_names,
    }


STRATEGY_TAG = "mixed23_daily_rl"  # tag for position tracking

CRYPTO_ALPACA = {"BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD", "DOGEUSD", "LINKUSD", "AAVEUSD", "UNIUSD"}


def execute_signal_result(
    signal: ExecutableSignal,
    allocation_pct: float = DEFAULT_ALLOCATION_PCT,
) -> TradeExecutionResult:
    """Execute the RL signal on Alpaca and return a structured result."""
    from loguru import logger

    symbol = _normalize_mixed_symbol(signal["symbol"])
    direction = signal["direction"]
    confidence = _coerce_finite_float(signal["confidence"])
    is_crypto = symbol in CRYPTO_ALPACA
    alpaca_symbol = symbol.replace("USD", "/USD") if is_crypto else symbol

    def build_result(
        status: TradeExecutionStatus,
        reason: str,
        **kwargs: object,
    ) -> TradeExecutionResult:
        return TradeExecutionResult(
            submitted=status == "submitted",
            status=status,
            reason=reason,
            symbol=symbol,
            alpaca_symbol=alpaca_symbol,
            direction=direction,
            confidence=confidence,
            is_crypto=is_crypto,
            **kwargs,
        )

    try:
        broker = _load_mixed_daily_broker_api()
    except Exception as exc:
        logger.error(f"Broker API unavailable for {alpaca_symbol}: {exc}")
        return build_result("rejected", "broker_api_unavailable")

    try:
        account = broker.trading_client.get_account()
    except Exception as exc:
        logger.error(f"Account fetch failed for {alpaca_symbol}: {exc}")
        return build_result("rejected", "account_fetch_failed")

    portfolio_value = _coerce_finite_float(getattr(account, "portfolio_value", None))
    buying_power = _coerce_finite_float(getattr(account, "buying_power", None))

    if portfolio_value is None or portfolio_value <= 0.0:
        logger.error(f"Invalid account portfolio value for {symbol}: {getattr(account, 'portfolio_value', None)!r}")
        return build_result("rejected", "invalid_portfolio_value")
    if buying_power is None or buying_power <= 0.0:
        logger.error(f"Invalid account buying power for {symbol}: {getattr(account, 'buying_power', None)!r}")
        return build_result("rejected", "invalid_buying_power")

    if direction not in {"LONG", "SHORT"}:
        logger.error(f"Unsupported trade direction for {symbol}: {direction!r}")
        return build_result("rejected", "invalid_direction")
    if confidence is None or confidence < 0.0:
        logger.error(f"Invalid confidence for {symbol}: {signal['confidence']!r}")
        return build_result("rejected", "invalid_confidence")

    logger.info(f"Account: portfolio=${portfolio_value:,.2f}, buying_power=${buying_power:,.2f}")
    logger.info(f"Signal: {direction} {symbol} ({alpaca_symbol}) conf={confidence:.1%}")

    # Calculate allocation
    trade_value = portfolio_value * (allocation_pct / 100.0)
    trade_value = min(trade_value, buying_power * 0.9)  # don't use 100% of BP
    if not math.isfinite(trade_value) or trade_value <= 0.0:
        logger.error(
            f"Invalid trade value for {alpaca_symbol}: trade_value={trade_value!r}, "
            f"portfolio_value={portfolio_value!r}, buying_power={buying_power!r}, allocation_pct={allocation_pct!r}"
        )
        return build_result("rejected", "invalid_trade_value", trade_value=trade_value)

    if trade_value < 1.0:
        logger.warning(f"Trade value too small: ${trade_value:.2f}")
        return build_result("skipped", "trade_value_too_small", trade_value=trade_value)

    # Get current price
    price = 0.0
    try:
        if is_crypto:
            if broker.crypto_client is None or broker.crypto_bars_request is None:
                logger.error(f"Crypto market data support unavailable for {alpaca_symbol}")
                return build_result("rejected", "crypto_market_data_unavailable", trade_value=trade_value)
            bars = broker.crypto_client.get_crypto_bars(
                broker.crypto_bars_request(symbol_or_symbols=alpaca_symbol, timeframe=broker.timeframe_hour, limit=1)
            )
            data = bars[alpaca_symbol]
            if data and len(data) > 0:
                maybe_price = _coerce_finite_float(getattr(data[0], "close", None))
                if maybe_price is not None:
                    price = maybe_price
        else:
            if broker.stock_bars_request is None:
                logger.error(f"Stock market data support unavailable for {alpaca_symbol}")
                return build_result("rejected", "stock_market_data_unavailable", trade_value=trade_value)
            bars = broker.data_client.get_stock_bars(
                broker.stock_bars_request(symbol_or_symbols=alpaca_symbol, timeframe=broker.timeframe_hour, limit=1)
            )
            data = bars[alpaca_symbol]
            if data and len(data) > 0:
                maybe_price = _coerce_finite_float(getattr(data[0], "close", None))
                if maybe_price is not None:
                    price = maybe_price
    except Exception as e:
        logger.warning(f"Price fetch error for {alpaca_symbol}: {e}")

    if not math.isfinite(price) or price <= 0:
        logger.error(f"Could not get price for {alpaca_symbol}")
        return build_result("rejected", "price_unavailable", trade_value=trade_value, price=price)

    # Calculate quantity
    if is_crypto:
        qty = round(trade_value / price, 8)
    else:
        qty = int(trade_value / price)  # whole shares for stocks
        if qty < 1:
            qty = round(trade_value / price, 4)  # fractional shares

    qty_float = _coerce_finite_float(qty)
    if qty_float is None or qty_float <= 0.0:
        logger.warning(f"Quantity too small for {alpaca_symbol}")
        return build_result("skipped", "quantity_too_small", trade_value=trade_value, price=price, qty=qty_float)
    qty = qty_float

    # Place order — crypto can't be shorted on Alpaca, skip those
    if direction == "SHORT" and is_crypto:
        # Check if we hold this crypto to sell
        try:
            pos = broker.trading_client.get_open_position(alpaca_symbol)
            position_qty = _coerce_finite_float(getattr(pos, "qty", None))
            if position_qty is not None and position_qty > 0.0:
                qty = min(qty, position_qty)
                logger.info(f"Selling existing {alpaca_symbol} position: {qty}")
            else:
                logger.info(f"SKIP: Can't short {alpaca_symbol} on Alpaca (no position to sell)")
                return build_result(
                    "skipped",
                    "crypto_short_without_position",
                    trade_value=trade_value,
                    price=price,
                    qty=qty,
                )
        except Exception:
            logger.info(f"SKIP: Can't short {alpaca_symbol} on Alpaca (no position)")
            return build_result(
                "skipped",
                "crypto_short_without_position",
                trade_value=trade_value,
                price=price,
                qty=qty,
            )

    side = broker.order_side_buy if direction == "LONG" else broker.order_side_sell
    side_value = side.value.lower()
    limit_reference = price
    try:
        quote = broker.latest_data(symbol)
        ask_price = _coerce_finite_float(getattr(quote, "ask_price", None)) or 0.0
        bid_price = _coerce_finite_float(getattr(quote, "bid_price", None)) or 0.0
        if ask_price > 0.0 and bid_price > 0.0:
            limit_reference = (ask_price + bid_price) / 2.0
    except Exception as e:
        logger.warning(f"Quote fetch error for passive limit on {symbol}: {e}")

    try:
        limit_price = _coerce_finite_float(broker.midpoint_limit_price(symbol, side_value, limit_reference))
    except Exception as exc:
        logger.error(f"Limit price calculation failed for {alpaca_symbol}: {exc}")
        return build_result(
            "rejected",
            "limit_price_calculation_failed",
            trade_value=trade_value,
            price=price,
            qty=qty,
        )
    if limit_price is None or limit_price <= 0.0:
        logger.error(f"Invalid limit price for {alpaca_symbol}: {limit_price!r}")
        return build_result(
            "rejected",
            "invalid_limit_price",
            trade_value=trade_value,
            price=price,
            qty=qty,
            limit_price=limit_price,
        )
    limit_price = round(limit_price, 6 if is_crypto else 2)
    try:
        tif = broker.time_in_force_for_qty(qty, symbol)
    except Exception as exc:
        logger.error(f"Time-in-force selection failed for {alpaca_symbol}: {exc}")
        return build_result(
            "rejected",
            "time_in_force_unavailable",
            trade_value=trade_value,
            price=price,
            qty=qty,
            limit_price=limit_price,
        )
    logger.info(
        f"Placing {side.value} midpoint limit order: {qty} {alpaca_symbol} @ ${limit_price:.6f} (${trade_value:,.2f})"
    )

    try:
        order = broker.trading_client.submit_order(
            broker.limit_order_request(
                symbol=alpaca_symbol,
                qty=qty,
                side=side,
                type=broker.order_type_limit,
                time_in_force=tif,
                limit_price=limit_price,
            )
        )
    except Exception as e:
        logger.error(f"Order failed: {e}")
        return build_result(
            "rejected",
            "order_submit_failed",
            trade_value=trade_value,
            price=price,
            qty=qty,
            limit_price=limit_price,
        )

    logger.info(f"Order submitted: {order.id} status={order.status}")
    logger.info(f"  {order.side} {order.qty} {order.symbol} type={order.type}")

    # Logging is best-effort; a local audit-write failure should not make the caller
    # believe the broker-side order was rejected and potentially retry it.
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy": STRATEGY_TAG,
        "symbol": symbol,
        "alpaca_symbol": alpaca_symbol,
        "direction": direction,
        "confidence": confidence,
        "qty": float(qty),
        "price_approx": price,
        "trade_value": trade_value,
        "order_id": str(order.id),
        "status": str(order.status),
    }
    log_path = Path("strategy_state/mixed23_daily_trades.jsonl")
    log_write_error: str | None = None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except (OSError, TypeError, ValueError) as exc:
        log_write_error = str(exc)
        logger.warning(f"Trade submitted but could not be logged to {log_path}: {exc}")
    else:
        logger.info(f"Trade logged to {log_path}")
    return build_result(
        "submitted",
        "order_submitted",
        trade_value=trade_value,
        price=price,
        qty=qty,
        limit_price=limit_price,
        order_id=str(order.id),
        order_status=str(order.status),
        log_write_error=log_write_error,
    )


def execute_signal(signal: ExecutableSignal, allocation_pct: float = DEFAULT_ALLOCATION_PCT) -> bool:
    """Execute the RL signal on Alpaca. Returns True if order placed."""
    return execute_signal_result(signal, allocation_pct).submitted


def _execute_signal_attempt(
    *,
    config: MixedDailyRuntimeConfig,
    mode: MixedDailyRunMode,
    signal: ExecutableSignal,
    attempt_kind: str,
    attempt_index: int,
) -> TradeExecutionResult:
    try:
        result = execute_signal_result(signal, config.allocation_pct)
    except Exception as exc:
        error_notes = _exception_notes(exc)
        _emit_runtime_event(
            "mixed_daily_runtime_error",
            config=config,
            mode=mode,
            stage="trade_execution",
            error=str(exc),
            error_type=type(exc).__name__,
            error_notes=error_notes,
            attempt_kind=attempt_kind,
            attempt_index=attempt_index,
            signal_direction=signal["direction"],
            signal_symbol=signal["symbol"],
            signal_confidence=signal["confidence"],
        )
        raise
    _emit_trade_execution_event(
        config=config,
        mode=mode,
        signal=signal,
        result=result,
        attempt_kind=attempt_kind,
        attempt_index=attempt_index,
    )
    return result


def _execute_inference_signal(
    *,
    config: MixedDailyRuntimeConfig,
    mode: MixedDailyRunMode,
    signal: InferenceSignal,
) -> TradeExecutionResult | None:
    candidates = _build_signal_execution_candidates(signal)
    if not candidates:
        return None

    primary = candidates[0]
    print(f"\n  EXECUTING: {primary.signal['direction']} {primary.signal['symbol']}")
    result = _execute_signal_attempt(
        config=config,
        mode=mode,
        signal=primary.signal,
        attempt_kind=primary.attempt_kind,
        attempt_index=primary.attempt_index,
    )
    if result.submitted or len(candidates) == 1:
        return result

    print("  Primary signal skipped, trying alternatives...")
    for candidate in candidates[1:]:
        print(
            f"  Trying: {candidate.signal['direction']} {candidate.signal['symbol']} "
            f"(conf={candidate.signal['confidence']:.3f})"
        )
        result = _execute_signal_attempt(
            config=config,
            mode=mode,
            signal=candidate.signal,
            attempt_kind=candidate.attempt_kind,
            attempt_index=candidate.attempt_index,
        )
        if result.submitted:
            return result
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Production daily mixed trading bot")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--symbols-file", default=None)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--once", action="store_true", help="Generate one signal and exit")
    parser.add_argument("--daemon", action="store_true", help="Run daily at midnight UTC")
    parser.add_argument("--dry-run", action="store_true", help="Print signal without executing")
    parser.add_argument("--check-config", action="store_true", help="Print a setup readiness report and exit")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved runtime config and exit")
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--allocation-pct", type=float, default=DEFAULT_ALLOCATION_PCT,
                        help="Percentage of portfolio to allocate per trade (default 10%%)")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--warmup-buffer-days", type=int, default=DEFAULT_WARMUP_BUFFER_DAYS)
    parser.add_argument("--daemon-hour-utc", type=int, default=DEFAULT_DAEMON_HOUR_UTC)
    parser.add_argument("--daemon-minute-utc", type=int, default=DEFAULT_DAEMON_MINUTE_UTC)
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_INFERENCE_MAX_STEPS)
    parser.add_argument("--fill-slippage-bps", type=float, default=DEFAULT_FILL_SLIPPAGE_BPS)
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    parser.add_argument("--max-leverage", type=float, default=DEFAULT_MAX_LEVERAGE)
    parser.add_argument("--periods-per-year", type=float, default=DEFAULT_PERIODS_PER_YEAR)
    parser.add_argument("--action-max-offset-bps", type=float, default=DEFAULT_ACTION_MAX_OFFSET_BPS)
    parser.add_argument(
        "--allow-unsafe-checkpoint-loading",
        action="store_true",
        help="Allow legacy pickle checkpoint loading. Only use this with trusted checkpoint files.",
    )
    parser.add_argument(
        "--check-config-text",
        action="store_true",
        help="When combined with --check-config, also print a human-readable setup summary to stderr.",
    )
    args = parser.parse_args(argv)

    config = _build_runtime_config(args)

    if config.print_config or config.check_config:
        print(json.dumps(asdict(config), indent=2, sort_keys=True))
        if args.check_config and args.check_config_text:
            rendered = (
                _format_runtime_preflight_ready(config)
                if config.ready
                else _format_runtime_preflight_failure(config)
            )
            print(rendered, file=sys.stderr)
        return 0 if config.ready else 1
    if not config.ready:
        print(_format_runtime_preflight_failure(config), file=sys.stderr)
        return 1

    symbols = config.symbols

    if args.once or args.dry_run:
        mode = "dry_run" if args.dry_run else "once"
        _emit_runtime_event("mixed_daily_runtime", config=config, mode=mode, stage="startup")
        try:
            signal = _run_export_and_inference_cycle(config)
        except Exception as exc:
            error_notes = _exception_notes(exc)
            _emit_runtime_event(
                "mixed_daily_runtime_error",
                config=config,
                mode=mode,
                stage="export_or_inference",
                error=str(exc),
                error_type=type(exc).__name__,
                error_notes=error_notes,
            )
            print(
                _format_runtime_execution_failure(
                    config,
                    stage="export_or_inference",
                    error=exc,
                ),
                file=sys.stderr,
            )
            return 1
        _emit_signal_event(config=config, mode=cast(MixedDailyRunMode, mode), signal=signal)
        if not args.dry_run:
            try:
                _execute_inference_signal(
                    config=config,
                    mode=cast(MixedDailyRunMode, mode),
                    signal=signal,
                )
            except Exception as exc:
                print(
                    _format_runtime_execution_failure(
                        config,
                        stage="trade_execution",
                        error=exc,
                    ),
                    file=sys.stderr,
                )
                return 1
        elif args.dry_run:
            print(f"\n  [DRY RUN] Would {signal['direction']} {signal['symbol'] or 'stay flat'}")

    elif args.daemon:
        _emit_runtime_event("mixed_daily_runtime", config=config, mode="daemon", stage="startup")
        print(f"Starting daily daemon for {len(symbols)} symbols...")
        print(f"Allocation: {config.allocation_pct}% of portfolio per trade")
        while True:
            now = datetime.now(timezone.utc)
            next_run = now.replace(
                hour=config.daemon_hour_utc,
                minute=config.daemon_minute_utc,
                second=0,
                microsecond=0,
            )
            if next_run <= now:
                next_run += pd.Timedelta(days=1)
            wait_s = (next_run - now).total_seconds()
            print(f"\nNext run: {next_run.isoformat()} (in {wait_s/3600:.1f}h)")
            time.sleep(wait_s)

            try:
                print(f"\n{'='*60}")
                print(f"DAILY RUN — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
                print(f"{'='*60}")
                signal = _run_export_and_inference_cycle(config)
                _emit_signal_event(config=config, mode="daemon", signal=signal)
                if _execute_inference_signal(
                    config=config,
                    mode="daemon",
                    signal=signal,
                ) is None:
                    print(f"\n  SIGNAL: FLAT — closing any open positions from this strategy")
            except Exception as e:
                _emit_runtime_event(
                    "mixed_daily_runtime_error",
                    config=config,
                    mode="daemon",
                    stage="daemon_iteration",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
