#!/usr/bin/env python3
"""
Production daily stock RL trading bot.

This trader runs a long-only daily PPO policy on U.S. equities. It uses the
previous completed trading day's bar set for inference and places a single
market order shortly after the regular session opens.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import shlex
import sys
import tempfile
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, time as dt_time, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
from typing import Iterable, Iterator, Literal, Mapping, Optional, Sequence, TypeAlias, TypedDict, cast
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    import fcntl as _fcntl
except Exception:  # pragma: no cover - non-POSIX fallback
    _fcntl = None

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.inference_daily import DailyPPOTrader
from pufferlib_market.inference import TradingSignal
from pufferlib_market.checkpoint_loader import load_checkpoint_payload
from src.alpaca_account_lock import acquire_alpaca_account_lock, require_explicit_live_trading_enable
from src.market_regime import regime_filter_reason
from src.daily_stock_feature_schema import (
    DailyStockFeatureSchema,
    build_daily_feature_history_for_schema,
    compute_daily_feature_vector_for_schema,
    daily_feature_dimension,
    resolve_daily_feature_schema,
)
from src.daily_stock_defaults import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_DIR,
    DEFAULT_EXTRA_CHECKPOINTS,
    DEFAULT_MIN_OPEN_CONFIDENCE,
    DEFAULT_MIN_OPEN_VALUE_ESTIMATE,
    DEFAULT_SYMBOLS,
)
from src.local_data_health import (
    LocalDataHealthStatus,
    LocalDataStatusCounts,
    format_local_data_health_lines,
    local_data_status_counts,
)
from src.shared_path_guard import shared_path_guard
from src import stock_symbol_inputs
from src.trading_server.client import (
    InMemoryTradingServerClient,
    TradingServerBaseUrlDetails,
    TradingServerAccountSnapshot,
    TradingServerClient,
    TradingServerClientLike,
    TradingServerPositionPayload,
    TradingServerQuotePayload,
    describe_trading_server_base_url,
    is_secure_or_loopback_trading_server_url,
)
from src.trading_server.settings import TRADING_SERVER_BASE_URL_ENV, resolve_trading_server_base_url
from unified_orchestrator.jsonl_utils import append_jsonl_row

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("daily_stock_rl")

# Backwards-compatible module-level helpers.
# Keep these names patchable for existing tests and callers; the schema-aware
# wrappers below decide when to use the legacy prod feature vector instead.
def compute_daily_feature_history(price_df: pd.DataFrame, *, schema: str = "rsi_v5") -> pd.DataFrame:
    return build_daily_feature_history_for_schema(price_df, schema=schema)


def compute_daily_features(price_df: pd.DataFrame, *, schema: str = "rsi_v5") -> np.ndarray:
    return compute_daily_feature_vector_for_schema(price_df, schema=schema)

EASTERN = ZoneInfo("America/New_York")
RUN_AFTER_OPEN_ET = dt_time(hour=9, minute=35)

DEFAULT_ALLOCATION_PCT = 12.5  # Calibrated 2026-03-31: was 25%, 0.5x scale improves val_p10 by +1.9%
DEFAULT_SORTINO_SERVER_CHECKPOINT = "pufferlib_market/checkpoints/stocks12_v2_sweep/stock_trade_pen_05_s123/best.pt"
DEFAULT_SERVER_PAPER_ACCOUNT = "paper_sortino_daily"
DEFAULT_SERVER_PAPER_BOT_ID = "daily_stock_sortino_v1"
DEFAULT_BACKTEST_SERVER_ACCOUNT = "paper_backtest_daily_sortino"
DEFAULT_BACKTEST_SERVER_BOT_ID = "daily_stock_backtest_v1"
DEFAULT_BACKTEST_SERVER_SESSION_ID = "daily-stock-backtest"
DEFAULT_BACKTEST_WRITER_TTL_SECONDS = 3600
DEFAULT_BACKTEST_STARTING_CASH = 10_000.0
DEFAULT_BACKTEST_BUYING_POWER_MULTIPLIER = 1.0
DEFAULT_DAILY_FRAME_MIN_DAYS = 120
DEFAULT_ALPACA_DAILY_HISTORY_LOOKBACK_DAYS = 420
DEFAULT_BAR_FRESHNESS_MAX_AGE_DAYS = 5
DEFAULT_DAEMON_RETRY_SLEEP_SECONDS = 300.0
DEFAULT_MULTI_POSITION = 0
DEFAULT_MULTI_POSITION_MIN_PROB_RATIO = 0.3
DEFAULT_SYMBOL_PREVIEW_LIMIT = 5
BUYING_POWER_USAGE_CAP = 0.95
SERVER_MARKETABLE_LIMIT_BUFFER_BPS = 100.0
# Calibrated execution offsets (2026-03-31 sweep over 726 combos, 788 windows)
# Best: entry=+5bps, exit=+25bps, scale=0.5x → val_p10=-0.4% vs baseline -2.3%
CALIBRATED_ENTRY_OFFSET_BPS = 5.0   # Buy limit at open * 1.0005
CALIBRATED_EXIT_OFFSET_BPS = 25.0   # Sell limit at open * 1.0025
DEFAULT_ALLOCATION_SIZING_MODE = "static"
STATE_PATH = REPO / "strategy_state/daily_stock_rl_state.json"
SIGNAL_LOG_PATH = REPO / "strategy_state/daily_stock_rl_signals.jsonl"
RUN_EVENT_LOG_PATH = REPO / "strategy_state/daily_stock_rl_run_events.jsonl"
STATE_LOCK_SUFFIX = ".lock"

StockDataSource: TypeAlias = Literal["alpaca", "local"]
StockExecutionBackend: TypeAlias = Literal["alpaca", "trading_server"]
StockRunMode: TypeAlias = Literal["once", "daemon", "backtest"]
StockAccountMode: TypeAlias = Literal["paper", "live"]
StockAllocationSizingMode: TypeAlias = Literal["static", "confidence_scaled"]
TradingServerUrlSource: TypeAlias = Literal["cli", "env", "default"]


class LocalDataDirResolutionSource(StrEnum):
    REQUESTED = "requested"
    NESTED_TRAIN = "nested_train"


StockExecutionStatus: TypeAlias = Literal[
    "local_only",
    "submitted",
    "dry_run_would_execute",
    "skipped_market_closed",
    "skipped_stale_bars",
    "no_action_flat_signal",
    "blocked_open_gate",
    "no_action_executor_declined",
]
StockExecutionSkipReason: TypeAlias = Literal[
    "local_data_mode",
    "market_closed",
    "stale_bars",
    "flat_signal",
    "open_gate",
    "executor_declined",
]
StockLocalDataStatus = LocalDataHealthStatus


class RuntimeLogPayload(TypedDict, total=False):
    account_mode: StockAccountMode
    run_mode: StockRunMode
    dry_run: bool
    data_source: StockDataSource
    execution_backend: StockExecutionBackend
    symbol_count: int
    symbols: list[str]
    symbol_source: str
    symbol_source_label: str
    symbol_preview: list[str]
    symbol_preview_text: str
    checkpoint: str
    ensemble_size: int
    command_preview: str
    data_dir: str
    requested_local_data_dir: str
    resolved_local_data_dir: str
    resolved_local_data_dir_source: LocalDataDirResolutionSource
    server_account: str
    server_bot_id: str
    configured_server_url: str | None
    server_url: str
    resolved_server_url: str
    server_url_source: TradingServerUrlSource
    server_url_transport: str
    server_url_scope: str
    server_url_security: str
    checkpoint_feature_schema: DailyStockFeatureSchema
    checkpoint_feature_dimension: int
    primary_checkpoint_arch: str
    extra_checkpoint_classes: list[str]
    usable_symbol_count: int
    latest_local_data_date: str
    stale_symbol_count: int
    missing_symbol_count: int
    invalid_symbol_count: int
    preflight_warnings: list[str]


class CheckpointLoadPrimarySummary(TypedDict, total=False):
    checkpoint: str
    device: str
    arch: str
    hidden_size: int
    num_actions: int
    num_symbols: int
    long_only: bool
    action_allocation_bins: int
    action_level_bins: int
    action_max_offset_bps: float
    max_steps: int
    symbols: list[str]
    feature_schema: DailyStockFeatureSchema
    feature_dimension: int


CheckpointLoadExtraSummary = TypedDict(
    "CheckpointLoadExtraSummary",
    {
        "path": str,
        "class": str,
    },
)


class CheckpointLoadDiagnostics(TypedDict, total=False):
    ok: bool | None
    error: str | None
    primary: CheckpointLoadPrimarySummary | None
    extras: list[CheckpointLoadExtraSummary]
    skipped: bool


class DaemonWarningPayload(TypedDict, total=False):
    event: str
    timestamp: str
    checkpoint: str
    stage: str
    error_type: str
    error: str
    account_mode: str
    execution_backend: str
    sleep_seconds: float
    state_path: str
    retry_with_paper_clock: bool
    clock_source: str
    server_session_id: str


class SignalPayload(TypedDict):
    run_id: str
    timestamp: str
    checkpoint: str
    action: str
    symbol: str | None
    direction: str | None
    confidence: float
    value_estimate: float
    allocation_fraction: float
    quotes: dict[str, float]


class PortfolioSignalPayload(TypedDict):
    action: str
    symbol: str | None
    direction: str | None
    confidence: float
    value_estimate: float
    allocation_fraction: float


class ExecutionObservabilityFields(TypedDict):
    execution_submitted: bool
    execution_would_submit: bool
    execution_status: StockExecutionStatus
    execution_skip_reason: StockExecutionSkipReason | None


class RunSummaryPayload(TypedDict):
    event: str
    run_id: object
    timestamp: object
    checkpoint: object
    action: object
    symbol: object
    direction: object
    confidence: object
    value_estimate: object
    bar_data_source: object
    quote_data_source: object
    latest_bar_timestamp: object
    bars_fresh: object
    market_open: object
    dry_run: object
    execution_backend: object
    execution_status: object
    execution_skip_reason: object
    execution_submitted: object
    execution_would_submit: object
    allow_open: object
    allow_open_reason: object
    state_advanced: object
    signal_log_written: object
    signal_log_write_error: object
    run_event_log_written: object
    run_event_log_write_error: object


class RunFailurePayload(TypedDict):
    event: str
    run_id: str
    timestamp: str
    checkpoint: str
    stage: str
    error_type: str
    error: str
    data_source: StockDataSource
    execution_backend: StockExecutionBackend
    account_mode: StockAccountMode
    dry_run: bool
    state_path: str
    symbols: list[str]
    requested_local_data_dir: str | None
    resolved_local_data_dir: str | None
    resolved_local_data_dir_source: LocalDataDirResolutionSource | None
    server_account: str | None
    server_bot_id: str | None
    server_url: str | None
    observability: dict[str, object]


class StockLocalSymbolDetail(TypedDict):
    status: StockLocalDataStatus
    file_path: str
    local_data_date: str | None
    row_count: int | None
    reason: str | None


class LocalDataDirContext(TypedDict):
    requested_local_data_dir: str
    resolved_local_data_dir: str
    resolved_local_data_dir_source: LocalDataDirResolutionSource


@dataclass(frozen=True)
class LocalDailySymbolInspection:
    row_count: int
    latest_timestamp: pd.Timestamp


@dataclass
class StrategyState:
    active_symbol: Optional[str] = None
    active_qty: float = 0.0
    entry_price: float = 0.0
    entry_date: Optional[str] = None
    last_run_date: Optional[str] = None
    last_signal_action: Optional[str] = None
    last_signal_timestamp: Optional[str] = None
    last_order_id: Optional[str] = None
    pending_close_symbol: Optional[str] = None
    pending_close_order_id: Optional[str] = None


@dataclass(frozen=True)
class PortfolioContext:
    cash: float = DEFAULT_BACKTEST_STARTING_CASH
    current_symbol: Optional[str] = None
    position_qty: float = 0.0
    entry_price: float = 0.0
    hold_days: int = 0


@dataclass
class _DailyTraderCacheEntry:
    size: int
    mtime_ns: int
    trader: DailyPPOTrader


@dataclass
class _BarePolicyCacheEntry:
    size: int
    mtime_ns: int
    policy: torch.nn.Module


@dataclass
class _LocalDailySymbolInspectionCacheEntry:
    size: int
    mtime_ns: int
    inspection: LocalDailySymbolInspection


@dataclass(frozen=True)
class ServerPositionView:
    symbol: str
    qty: float
    side: str = "long"
    avg_entry_price: float = 0.0
    current_price: float = 0.0


@dataclass(frozen=True)
class PortfolioRebalanceTarget:
    symbol: str
    existing_qty: float
    target_qty: float


@dataclass(frozen=True)
class BacktestVariantSpec:
    name: str
    allocation_pct: float
    allocation_sizing_mode: StockAllocationSizingMode = DEFAULT_ALLOCATION_SIZING_MODE
    multi_position: int = DEFAULT_MULTI_POSITION
    multi_position_min_prob_ratio: float = DEFAULT_MULTI_POSITION_MIN_PROB_RATIO
    buying_power_multiplier: float = DEFAULT_BACKTEST_BUYING_POWER_MULTIPLIER


@dataclass
class PreparedDailyBacktestData:
    feature_schema: DailyStockFeatureSchema
    indexed: dict[str, pd.DataFrame]
    trader_template: DailyPPOTrader
    extra_policies: list[torch.nn.Module]
    symbols: tuple[str, ...]
    feature_cube: np.ndarray
    close_matrix: np.ndarray
    timestamps: tuple[datetime, ...]
    start: int
    min_len: int


_DAILY_TRADER_CACHE_MAX_ENTRIES = 8
_BARE_POLICY_CACHE_MAX_ENTRIES = 64
_LOCAL_DAILY_SYMBOL_INSPECTION_CACHE_MAX_ENTRIES = 512
_LOCAL_DAILY_SYMBOL_INSPECTION_MAX_ATTEMPTS = 2
_LOCAL_DATA_NESTED_TRAIN_DIRNAME = "train"
_DAILY_TRADER_CACHE: "OrderedDict[tuple[object, ...], _DailyTraderCacheEntry]" = OrderedDict()
_BARE_POLICY_CACHE: "OrderedDict[tuple[object, ...], _BarePolicyCacheEntry]" = OrderedDict()
_LOCAL_DAILY_SYMBOL_INSPECTION_CACHE: "OrderedDict[Path, _LocalDailySymbolInspectionCacheEntry]" = OrderedDict()
_DAILY_TRADER_CACHE_INFLIGHT: dict[tuple[object, ...], Event] = {}
_BARE_POLICY_CACHE_INFLIGHT: dict[tuple[object, ...], Event] = {}
_LOCAL_DAILY_SYMBOL_INSPECTION_CACHE_INFLIGHT: dict[Path, Event] = {}
_DAILY_TRADER_CACHE_LOCK = Lock()
_BARE_POLICY_CACHE_LOCK = Lock()
_LOCAL_DAILY_SYMBOL_INSPECTION_CACHE_LOCK = Lock()
_PORTFOLIO_REBALANCE_QTY_EPSILON = 1e-4


@dataclass(frozen=True)
class CliRuntimeConfig:
    paper: bool
    symbols: list[str]
    checkpoint: str
    extra_checkpoints: Optional[list[str]]
    data_dir: str
    data_source: StockDataSource
    allocation_pct: float
    execution_backend: StockExecutionBackend
    server_account: str
    server_bot_id: str
    server_url: Optional[str]
    dry_run: bool
    backtest: bool
    backtest_days: int
    backtest_starting_cash: float
    daemon: bool
    compare_server_parity: bool
    allocation_sizing_mode: StockAllocationSizingMode = DEFAULT_ALLOCATION_SIZING_MODE
    multi_position: int = DEFAULT_MULTI_POSITION
    multi_position_min_prob_ratio: float = DEFAULT_MULTI_POSITION_MIN_PROB_RATIO
    backtest_buying_power_multiplier: float = DEFAULT_BACKTEST_BUYING_POWER_MULTIPLIER
    backtest_entry_offset_bps: float = 0.0
    backtest_exit_offset_bps: float = 0.0
    symbols_file: Optional[str] = None
    min_open_confidence: float = DEFAULT_MIN_OPEN_CONFIDENCE
    min_open_value_estimate: float = DEFAULT_MIN_OPEN_VALUE_ESTIMATE
    print_payload: bool = False
    allow_unsafe_checkpoint_loading: bool = False
    meta_selector: bool = False
    meta_top_k: int = 1
    meta_lookback: int = 3
    removed_duplicate_symbols: list[str] = field(default_factory=list)
    ignored_symbol_inputs: list[str] = field(default_factory=list)

    @property
    def ensemble_enabled(self) -> bool:
        return self.extra_checkpoints is not None

    @property
    def ensemble_size(self) -> int:
        return 1 + len(self.extra_checkpoints or [])

    @property
    def account_mode(self) -> StockAccountMode:
        return "paper" if self.paper else "live"

    @property
    def run_mode(self) -> StockRunMode:
        return "backtest" if self.backtest else ("daemon" if self.daemon else "once")

    @property
    def checkpoint_paths(self) -> list[str]:
        return [self.checkpoint, *(self.extra_checkpoints or [])]

    @property
    def missing_checkpoint_paths(self) -> list[str]:
        return [path for path in self.checkpoint_paths if not Path(path).exists()]

    @property
    def checkpoints_exist(self) -> bool:
        return not self.missing_checkpoint_paths

    @property
    def resolved_server_url(self) -> str:
        return resolve_trading_server_base_url(self.server_url)

    @property
    def server_url_source(self) -> TradingServerUrlSource:
        if self.server_url:
            return "cli"
        if os.getenv(TRADING_SERVER_BASE_URL_ENV):
            return "env"
        return "default"

    @property
    def resolved_server_url_details(self) -> TradingServerBaseUrlDetails:
        return describe_trading_server_base_url(self.resolved_server_url)

    @property
    def portfolio_mode(self) -> bool:
        return self.multi_position > 1

    @property
    def position_capacity(self) -> int:
        return self.multi_position if self.portfolio_mode else 1

    @property
    def strategy_mode(self) -> str:
        if self.portfolio_mode:
            return f"portfolio (up to {self.multi_position} positions)"
        return "single-position"

    @property
    def symbol_source(self) -> str:
        return "symbols_file" if self.symbols_file else "cli"

    @property
    def symbol_source_label(self) -> str:
        if self.symbols_file:
            return f"symbols file: {self.symbols_file}"
        return "--symbols"

    @property
    def symbol_preview(self) -> list[str]:
        return list(self.symbols[:DEFAULT_SYMBOL_PREVIEW_LIMIT])

    @property
    def symbol_preview_text(self) -> str:
        preview = ", ".join(self.symbol_preview)
        remaining = len(self.symbols) - len(self.symbol_preview)
        if remaining > 0:
            return f"{preview} (+{remaining} more)"
        return preview

    def to_runtime_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["symbol_count"] = len(self.symbols)
        payload["ensemble_enabled"] = self.ensemble_enabled
        payload["ensemble_size"] = self.ensemble_size
        payload["account_mode"] = self.account_mode
        payload["run_mode"] = self.run_mode
        payload["symbol_source"] = self.symbol_source
        payload["symbol_source_label"] = self.symbol_source_label
        payload["symbol_preview"] = self.symbol_preview
        payload["symbol_preview_text"] = self.symbol_preview_text
        payload["portfolio_mode"] = self.portfolio_mode
        payload["position_capacity"] = self.position_capacity
        payload["strategy_mode"] = self.strategy_mode
        payload["summary"] = self.summary
        payload["check_command_preview"] = self.command_preview(check_config=True)
        payload["check_text_command_preview"] = self.command_preview(
            check_config=True,
            check_config_text=True,
        )
        payload["run_command_preview"] = self.command_preview()
        payload["safe_command_preview"] = self.command_preview(force_dry_run=True)
        payload["checkpoints_exist"] = self.checkpoints_exist
        payload["missing_checkpoints"] = self.missing_checkpoint_paths
        payload["daily_frame_min_days"] = DEFAULT_DAILY_FRAME_MIN_DAYS
        payload["alpaca_daily_history_lookback_days"] = DEFAULT_ALPACA_DAILY_HISTORY_LOOKBACK_DAYS
        payload["bar_freshness_max_age_days"] = DEFAULT_BAR_FRESHNESS_MAX_AGE_DAYS
        if self.execution_backend == "trading_server":
            payload.update(_trading_server_runtime_fields(self))
        return payload

    @property
    def summary(self) -> str:
        checkpoint_label = "checkpoint" if self.ensemble_size == 1 else "checkpoints"
        symbol_label = "symbol" if len(self.symbols) == 1 else "symbols"
        data_label = "local data" if self.data_source == "local" else "alpaca data"
        return (
            f"{self.account_mode} {self.run_mode} via {self.execution_backend} "
            f"using {data_label} on {len(self.symbols)} {symbol_label} "
            f"with {self.ensemble_size} {checkpoint_label} "
            f"as {self.strategy_mode}"
        )

    def command_preview(
        self,
        *,
        force_dry_run: bool | None = None,
        check_config: bool = False,
        check_config_text: bool = False,
    ) -> str:
        args = ["python", Path(__file__).name]
        if check_config or check_config_text:
            args.append("--check-config")
        if check_config_text:
            args.append("--check-config-text")

        if self.run_mode == "daemon":
            args.append("--daemon")
        elif self.run_mode == "backtest":
            args.append("--backtest")
        else:
            args.append("--once")

        args.append("--paper" if self.paper else "--live")
        use_dry_run = False if check_config else (force_dry_run if force_dry_run is not None else self.dry_run)
        if use_dry_run and not self.backtest:
            args.append("--dry-run")

        args.extend(["--checkpoint", self.checkpoint])
        if self.allow_unsafe_checkpoint_loading:
            args.append("--allow-unsafe-checkpoint-loading")
        if self.extra_checkpoints is None:
            args.append("--no-ensemble")
        elif self.extra_checkpoints != _resolved_default_extra_checkpoints():
            args.append("--extra-checkpoints")
            args.extend(self.extra_checkpoints)

        args.extend(["--data-source", self.data_source])
        args.extend(["--data-dir", self.data_dir])
        args.extend(["--allocation-pct", f"{self.allocation_pct:g}"])
        if self.allocation_sizing_mode != DEFAULT_ALLOCATION_SIZING_MODE:
            args.extend(["--allocation-sizing-mode", self.allocation_sizing_mode])
        if self.multi_position > 0:
            args.extend(["--multi-position", str(self.multi_position)])
            if self.multi_position_min_prob_ratio != DEFAULT_MULTI_POSITION_MIN_PROB_RATIO:
                args.extend(
                    [
                        "--multi-position-min-prob-ratio",
                        f"{self.multi_position_min_prob_ratio:g}",
                    ]
                )
        if self.min_open_confidence != DEFAULT_MIN_OPEN_CONFIDENCE:
            args.extend(["--min-open-confidence", f"{self.min_open_confidence:g}"])
        if self.min_open_value_estimate != DEFAULT_MIN_OPEN_VALUE_ESTIMATE:
            args.extend(["--min-open-value-estimate", f"{self.min_open_value_estimate:g}"])
        args.extend(["--execution-backend", self.execution_backend])
        args.extend(["--server-account", self.server_account])
        args.extend(["--server-bot-id", self.server_bot_id])
        if self.server_url:
            args.extend(["--server-url", self.server_url])
        if self.backtest:
            args.extend(["--backtest-days", str(self.backtest_days)])
            args.extend(["--backtest-starting-cash", f"{self.backtest_starting_cash:g}"])
            if self.backtest_buying_power_multiplier != DEFAULT_BACKTEST_BUYING_POWER_MULTIPLIER:
                args.extend(
                    [
                        "--backtest-buying-power-multiplier",
                        f"{self.backtest_buying_power_multiplier:g}",
                    ]
                )
            if self.backtest_entry_offset_bps != 0.0:
                args.extend(["--backtest-entry-offset-bps", f"{self.backtest_entry_offset_bps:g}"])
            if self.backtest_exit_offset_bps != 0.0:
                args.extend(["--backtest-exit-offset-bps", f"{self.backtest_exit_offset_bps:g}"])
        if self.compare_server_parity:
            args.append("--compare-server-parity")
        if self.print_payload:
            args.append("--print-payload")
        if self.symbols:
            if self.symbols_file:
                args.extend(["--symbols-file", self.symbols_file])
            else:
                args.append("--symbols")
                args.extend(self.symbols)

        return " ".join(shlex.quote(arg) for arg in args)


def _trading_server_runtime_fields(config: CliRuntimeConfig) -> dict[str, object]:
    details = config.resolved_server_url_details
    return {
        "configured_server_url": config.server_url,
        "server_url": config.resolved_server_url,
        "resolved_server_url": config.resolved_server_url,
        "server_url_source": config.server_url_source,
        "server_url_transport": details["transport"],
        "server_url_scope": details["scope"],
        "server_url_security": details["security"],
    }


def _trading_server_url_source_label(source: object) -> str:
    normalized = str(source or "").strip().lower()
    if normalized == "cli":
        return "--server-url"
    if normalized == "env":
        return TRADING_SERVER_BASE_URL_ENV
    return "built-in default"


def _trading_server_url_update_hint(source: object) -> str:
    normalized = str(source or "").strip().lower()
    if normalized == "cli":
        return "Update --server-url"
    if normalized == "env":
        return f"Update {TRADING_SERVER_BASE_URL_ENV}"
    return f"Pass --server-url or set {TRADING_SERVER_BASE_URL_ENV}"


def _trading_server_url_security_label(security: object) -> str:
    normalized = str(security or "").strip().lower()
    if normalized == "https":
        return "https"
    if normalized == "loopback_http":
        return "loopback http"
    if normalized == "insecure_remote_http":
        return "remote http rejected in live mode"
    return "invalid url"


def _runtime_log_payload(
    config: CliRuntimeConfig,
    *,
    preflight_payload: Mapping[str, object] | None = None,
) -> RuntimeLogPayload:
    payload: RuntimeLogPayload = {
        "account_mode": config.account_mode,
        "run_mode": config.run_mode,
        "dry_run": config.dry_run,
        "data_source": config.data_source,
        "execution_backend": config.execution_backend,
        "symbol_count": len(config.symbols),
        "symbols": list(config.symbols),
        "symbol_source": config.symbol_source,
        "symbol_source_label": config.symbol_source_label,
        "symbol_preview": config.symbol_preview,
        "symbol_preview_text": config.symbol_preview_text,
        "checkpoint": config.checkpoint,
        "ensemble_size": config.ensemble_size,
        "command_preview": config.command_preview(),
    }
    if config.backtest or config.data_source == "local":
        payload["data_dir"] = config.data_dir
        local_data_dir_context_payload = (
            _coerce_local_data_dir_context(preflight_payload)
            if preflight_payload is not None
            else None
        )
        if local_data_dir_context_payload is None:
            local_data_dir_context_payload = _local_data_dir_context(config.data_dir, config.symbols)
        payload.update(cast(RuntimeLogPayload, local_data_dir_context_payload))
    if config.execution_backend == "trading_server":
        payload["server_account"] = config.server_account
        payload["server_bot_id"] = config.server_bot_id
        payload.update(cast(RuntimeLogPayload, _trading_server_runtime_fields(config)))
    if preflight_payload is not None:
        local_data_status_counts = preflight_payload.get("local_data_status_counts")
        if isinstance(local_data_status_counts, Mapping):
            usable_symbol_count = local_data_status_counts.get("usable")
            if isinstance(usable_symbol_count, int):
                payload["usable_symbol_count"] = usable_symbol_count
            stale_symbol_count = local_data_status_counts.get("stale")
            if isinstance(stale_symbol_count, int):
                payload["stale_symbol_count"] = stale_symbol_count
            missing_symbol_count = local_data_status_counts.get("missing")
            if isinstance(missing_symbol_count, int):
                payload["missing_symbol_count"] = missing_symbol_count
            invalid_symbol_count = local_data_status_counts.get("invalid")
            if isinstance(invalid_symbol_count, int):
                payload["invalid_symbol_count"] = invalid_symbol_count
        checkpoint_load = preflight_payload.get("checkpoint_load")
        if isinstance(checkpoint_load, Mapping):
            primary = checkpoint_load.get("primary")
            if isinstance(primary, Mapping):
                feature_schema = primary.get("feature_schema")
                if isinstance(feature_schema, str):
                    payload["checkpoint_feature_schema"] = cast(DailyStockFeatureSchema, feature_schema)
                feature_dimension = primary.get("feature_dimension")
                if isinstance(feature_dimension, int):
                    payload["checkpoint_feature_dimension"] = feature_dimension
                primary_checkpoint_arch = primary.get("arch")
                if isinstance(primary_checkpoint_arch, str) and primary_checkpoint_arch.strip():
                    payload["primary_checkpoint_arch"] = primary_checkpoint_arch
            extras = checkpoint_load.get("extras")
            if isinstance(extras, list):
                extra_checkpoint_classes = [
                    str(extra.get("class"))
                    for extra in extras
                    if isinstance(extra, Mapping) and str(extra.get("class") or "").strip()
                ]
                if extra_checkpoint_classes:
                    payload["extra_checkpoint_classes"] = extra_checkpoint_classes
        usable_symbol_count = preflight_payload.get("usable_symbol_count")
        if isinstance(usable_symbol_count, int) and "usable_symbol_count" not in payload:
            payload["usable_symbol_count"] = usable_symbol_count
        latest_local_data_date = preflight_payload.get("latest_local_data_date")
        if isinstance(latest_local_data_date, str) and latest_local_data_date.strip():
            payload["latest_local_data_date"] = latest_local_data_date
        resolved_local_data_dir = preflight_payload.get("resolved_local_data_dir")
        if (
            isinstance(resolved_local_data_dir, str)
            and resolved_local_data_dir.strip()
            and "resolved_local_data_dir" not in payload
        ):
            payload["resolved_local_data_dir"] = resolved_local_data_dir
        resolved_local_data_dir_source = _coerce_local_data_dir_resolution_source(
            preflight_payload.get("resolved_local_data_dir_source")
        )
        if resolved_local_data_dir_source is not None and "resolved_local_data_dir_source" not in payload:
            payload["resolved_local_data_dir_source"] = resolved_local_data_dir_source
        stale_symbol_data = preflight_payload.get("stale_symbol_data")
        if isinstance(stale_symbol_data, Mapping) and "stale_symbol_count" not in payload:
            payload["stale_symbol_count"] = len(stale_symbol_data)
        warnings = preflight_payload.get("warnings")
        if isinstance(warnings, list):
            preflight_warnings = [str(warning) for warning in warnings if str(warning).strip()]
            if preflight_warnings:
                payload["preflight_warnings"] = preflight_warnings
    return payload


def _log_runtime_start(
    config: CliRuntimeConfig,
    *,
    preflight_payload: Mapping[str, object] | None = None,
) -> None:
    logger.info(
        "Runtime config: %s",
        json.dumps(_runtime_log_payload(config, preflight_payload=preflight_payload), sort_keys=True),
    )


def _normalize_stock_symbol_list(symbols: Iterable[object]) -> list[str]:
    return stock_symbol_inputs.normalize_stock_symbol_list(symbols)


def _normalize_symbols(raw_symbols: Sequence[object]) -> tuple[list[str], list[str], list[str]]:
    return stock_symbol_inputs.normalize_symbols(raw_symbols)


def _normalize_stock_symbol(raw_symbol: object) -> str:
    return stock_symbol_inputs.normalize_stock_symbol(raw_symbol)


def _load_symbols_file(path: str | Path) -> list[str]:
    return stock_symbol_inputs.load_symbols_file(path)


def _resolved_default_extra_checkpoints() -> list[str]:
    return [
        str((REPO / path).resolve()) if not Path(path).is_absolute() else path
        for path in DEFAULT_EXTRA_CHECKPOINTS
    ]


def _resolve_local_data_base_with_source(
    data_dir: str | Path,
    symbols: Iterable[str],
) -> tuple[Path, LocalDataDirResolutionSource]:
    base = (REPO / Path(data_dir).expanduser()).resolve()
    normalized_symbols = _normalize_stock_symbol_list(symbols)
    nested_train = base / _LOCAL_DATA_NESTED_TRAIN_DIRNAME
    if base.name != _LOCAL_DATA_NESTED_TRAIN_DIRNAME and nested_train.exists():
        if all((nested_train / f"{symbol}.csv").exists() for symbol in normalized_symbols):
            return nested_train, LocalDataDirResolutionSource.NESTED_TRAIN
    return base, LocalDataDirResolutionSource.REQUESTED


def _resolve_local_data_base(data_dir: str | Path, symbols: Iterable[str]) -> Path:
    resolved, _source = _resolve_local_data_base_with_source(data_dir, symbols)
    return resolved


def _local_data_dir_context(
    data_dir: str | Path,
    symbols: Iterable[str],
) -> LocalDataDirContext:
    resolved_local_data_dir, resolved_local_data_dir_source = _resolve_local_data_base_with_source(
        data_dir,
        symbols,
    )
    return {
        "requested_local_data_dir": str((REPO / Path(data_dir).expanduser()).resolve()),
        "resolved_local_data_dir": str(resolved_local_data_dir),
        "resolved_local_data_dir_source": resolved_local_data_dir_source,
    }


def _coerce_local_data_dir_context(payload: Mapping[str, object]) -> LocalDataDirContext | None:
    requested_local_data_dir = payload.get("requested_local_data_dir")
    resolved_local_data_dir = payload.get("resolved_local_data_dir")
    resolved_local_data_dir_source = _coerce_local_data_dir_resolution_source(
        payload.get("resolved_local_data_dir_source")
    )
    if (
        isinstance(requested_local_data_dir, str)
        and requested_local_data_dir.strip()
        and isinstance(resolved_local_data_dir, str)
        and resolved_local_data_dir.strip()
        and resolved_local_data_dir_source is not None
    ):
        return {
            "requested_local_data_dir": requested_local_data_dir,
            "resolved_local_data_dir": resolved_local_data_dir,
            "resolved_local_data_dir_source": resolved_local_data_dir_source,
        }
    return None


def _local_data_dir_exists(resolved_local_data_dir: Path | None) -> bool:
    return resolved_local_data_dir is not None and resolved_local_data_dir.exists()


def _coerce_local_data_dir_resolution_source(
    value: object,
) -> LocalDataDirResolutionSource | None:
    if isinstance(value, LocalDataDirResolutionSource):
        return value
    if isinstance(value, str):
        try:
            return LocalDataDirResolutionSource(value)
        except ValueError:
            return None
    return None


def _resolve_daily_frame_columns(columns: Iterable[object]) -> tuple[str, list[str]]:
    lower_map = {str(col).lower(): str(col) for col in columns}
    ts_col = lower_map.get("timestamp") or lower_map.get("date")
    required = ["open", "high", "low", "close", "volume"]
    missing = [name for name in required if name not in lower_map]
    if ts_col is None or missing:
        raise ValueError(f"Daily frame missing columns: timestamp/date + {required}")
    return ts_col, [lower_map[name] for name in required]


def _parse_daily_timestamps(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce", format="mixed")


def _normalize_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ts_col, required_columns = _resolve_daily_frame_columns(frame.columns)

    normalized = frame.rename(columns={src: src.lower() for src in frame.columns}).copy()
    normalized["timestamp"] = _parse_daily_timestamps(normalized[ts_col.lower()])
    normalized = normalized[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    normalized = normalized.dropna(subset=["timestamp"]).sort_values("timestamp")
    normalized = normalized.drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)
    for column in (name.lower() for name in required_columns):
        normalized[column] = normalized[column].astype(float)
    return normalized


def _inspect_local_daily_symbol_file(path: Path) -> LocalDailySymbolInspection:
    resolved_path = path.resolve(strict=False)
    for attempt in range(_LOCAL_DAILY_SYMBOL_INSPECTION_MAX_ATTEMPTS):
        stat_result = resolved_path.stat()
        while True:
            with _LOCAL_DAILY_SYMBOL_INSPECTION_CACHE_LOCK:
                cached = _LOCAL_DAILY_SYMBOL_INSPECTION_CACHE.get(resolved_path)
                if cached is not None and cached.size == stat_result.st_size and cached.mtime_ns == stat_result.st_mtime_ns:
                    _LOCAL_DAILY_SYMBOL_INSPECTION_CACHE.move_to_end(resolved_path)
                    return cached.inspection
                inflight = _LOCAL_DAILY_SYMBOL_INSPECTION_CACHE_INFLIGHT.get(resolved_path)
                if inflight is None:
                    inflight = Event()
                    _LOCAL_DAILY_SYMBOL_INSPECTION_CACHE_INFLIGHT[resolved_path] = inflight
                    break
            inflight.wait()
            stat_result = resolved_path.stat()
        try:
            ts_col, required_columns = _resolve_daily_frame_columns(pd.read_csv(path, nrows=0).columns)
            selected_columns = [ts_col, *required_columns]
            frame = pd.read_csv(path, usecols=selected_columns)
            normalized_names = {str(col).lower(): str(col) for col in frame.columns}
            timestamps = _parse_daily_timestamps(frame[normalized_names[ts_col.lower()]])
            valid_mask = timestamps.notna()
            for column in required_columns:
                numeric = pd.to_numeric(frame[normalized_names[column.lower()]], errors="coerce")
                valid_mask &= numeric.notna()
            valid_timestamps = pd.DatetimeIndex(timestamps.loc[valid_mask]).sort_values()
            if len(valid_timestamps) == 0:
                raise ValueError("Daily frame has no valid timestamped OHLCV rows")
            valid_timestamps = valid_timestamps.drop_duplicates(keep="last")
            inspection = LocalDailySymbolInspection(
                row_count=len(valid_timestamps),
                latest_timestamp=pd.Timestamp(valid_timestamps[-1]),
            )
            final_stat_result = resolved_path.stat()
            if (
                final_stat_result.st_size != stat_result.st_size
                or final_stat_result.st_mtime_ns != stat_result.st_mtime_ns
            ):
                if attempt + 1 < _LOCAL_DAILY_SYMBOL_INSPECTION_MAX_ATTEMPTS:
                    with _LOCAL_DAILY_SYMBOL_INSPECTION_CACHE_LOCK:
                        inflight = _LOCAL_DAILY_SYMBOL_INSPECTION_CACHE_INFLIGHT.pop(resolved_path, None)
                        if inflight is not None:
                            inflight.set()
                    continue
                raise RuntimeError(f"Local daily CSV changed during inspection: {resolved_path}")
            with _LOCAL_DAILY_SYMBOL_INSPECTION_CACHE_LOCK:
                _LOCAL_DAILY_SYMBOL_INSPECTION_CACHE[resolved_path] = _LocalDailySymbolInspectionCacheEntry(
                    size=final_stat_result.st_size,
                    mtime_ns=final_stat_result.st_mtime_ns,
                    inspection=inspection,
                )
                _LOCAL_DAILY_SYMBOL_INSPECTION_CACHE.move_to_end(resolved_path)
                _prune_ordered_cache(
                    _LOCAL_DAILY_SYMBOL_INSPECTION_CACHE,
                    max_entries=_LOCAL_DAILY_SYMBOL_INSPECTION_CACHE_MAX_ENTRIES,
                )
                inflight = _LOCAL_DAILY_SYMBOL_INSPECTION_CACHE_INFLIGHT.pop(resolved_path, None)
                if inflight is not None:
                    inflight.set()
            return inspection
        except Exception:
            with _LOCAL_DAILY_SYMBOL_INSPECTION_CACHE_LOCK:
                inflight = _LOCAL_DAILY_SYMBOL_INSPECTION_CACHE_INFLIGHT.pop(resolved_path, None)
                if inflight is not None:
                    inflight.set()
            raise
    raise AssertionError("unreachable")


def _align_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if not frames:
        raise ValueError("No daily frames to align")

    common_index: Optional[pd.DatetimeIndex] = None
    aligned: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        indexed = _normalize_daily_frame(frame).set_index("timestamp").sort_index()
        common_index = indexed.index if common_index is None else common_index.intersection(indexed.index)
        aligned[symbol] = indexed

    if common_index is None or len(common_index) == 0:
        raise ValueError("No common daily timestamps across symbols")

    result: dict[str, pd.DataFrame] = {}
    for symbol, indexed in aligned.items():
        trimmed = indexed.loc[common_index].copy()
        trimmed.index.name = "timestamp"
        result[symbol] = trimmed.reset_index()
    return result


def load_local_daily_frames(
    symbols: Iterable[str],
    *,
    data_dir: str = DEFAULT_DATA_DIR,
    min_days: int = DEFAULT_DAILY_FRAME_MIN_DAYS,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    normalized_symbols = _normalize_stock_symbol_list(symbols)
    base = _resolve_local_data_base(data_dir, normalized_symbols)
    for symbol in normalized_symbols:
        path = base / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing local daily data for {symbol}: {path}")
        frame = _normalize_daily_frame(pd.read_csv(path))
        if len(frame) < min_days:
            raise ValueError(f"{symbol}: only {len(frame)} rows in {path}, need at least {min_days}")
        frames[symbol] = frame
    return _align_frames(frames)


def _drop_incomplete_session(frame: pd.DataFrame, *, now: datetime) -> pd.DataFrame:
    if frame.empty:
        return frame
    latest_ts = pd.Timestamp(frame["timestamp"].iloc[-1])
    latest_et = latest_ts.tz_convert(EASTERN)
    now_et = now.astimezone(EASTERN)
    if latest_et.date() >= now_et.date():
        return frame.iloc[:-1].reset_index(drop=True)
    return frame


def build_trading_client(*, paper: bool):
    from alpaca.trading.client import TradingClient
    key_id, secret = _resolve_alpaca_credentials(paper=paper)

    return TradingClient(key_id, secret, paper=paper)


def build_data_client(*, paper: bool):
    from alpaca.data import StockHistoricalDataClient
    key_id, secret = _resolve_alpaca_credentials(paper=paper)

    return StockHistoricalDataClient(key_id, secret)


def _frames_from_alpaca_bars(
    *,
    symbols: Sequence[str],
    bars_df: pd.DataFrame | None,
    now: datetime,
) -> dict[str, pd.DataFrame]:
    if bars_df is None or len(bars_df) == 0:
        return {}

    ordered_symbols = _normalize_stock_symbol_list(symbols)
    frames: dict[str, pd.DataFrame] = {}
    if isinstance(bars_df.index, pd.MultiIndex):
        grouped = bars_df.reset_index().groupby("symbol", sort=False)
        for symbol, group in grouped:
            normalized = _normalize_daily_frame(group)
            frames[str(symbol).upper()] = _drop_incomplete_session(normalized, now=now)
        return frames

    flat = bars_df.reset_index()
    if "symbol" in flat.columns:
        grouped = flat.groupby("symbol", sort=False)
        for symbol, group in grouped:
            normalized = _normalize_daily_frame(group)
            frames[str(symbol).upper()] = _drop_incomplete_session(normalized, now=now)
        return frames

    if len(ordered_symbols) != 1:
        raise RuntimeError("Expected Alpaca bars dataframe to include symbol information for multi-symbol requests")
    normalized = _normalize_daily_frame(flat)
    frames[ordered_symbols[0]] = _drop_incomplete_session(normalized, now=now)
    return frames


def _request_alpaca_daily_frames(
    *,
    client,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    feed,
    now: datetime,
) -> dict[str, pd.DataFrame]:
    from alpaca.data import StockBarsRequest, TimeFrame

    request = StockBarsRequest(
        symbol_or_symbols=list(symbols),
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment="raw",
        feed=feed,
    )
    bars = client.get_stock_bars(request)
    bars_df = getattr(bars, "df", None)
    return _frames_from_alpaca_bars(symbols=symbols, bars_df=bars_df, now=now)


def load_alpaca_daily_frames(
    symbols: Iterable[str],
    *,
    paper: bool,
    min_days: int = DEFAULT_DAILY_FRAME_MIN_DAYS,
    history_days: int = DEFAULT_ALPACA_DAILY_HISTORY_LOOKBACK_DAYS,
    now: Optional[datetime] = None,
    data_client=None,
) -> dict[str, pd.DataFrame]:
    from alpaca.data.enums import DataFeed

    now = now or datetime.now(timezone.utc)
    client = data_client or build_data_client(paper=paper)
    ordered_symbols = _normalize_stock_symbol_list(symbols)
    start = now - timedelta(days=history_days)

    frames: dict[str, pd.DataFrame] = {}
    insufficient_counts: dict[str, int] = {}
    fetch_errors: list[str] = []

    def _accept_resolved(candidates: dict[str, pd.DataFrame]) -> None:
        for symbol, frame in candidates.items():
            rows = len(frame)
            if rows < min_days:
                insufficient_counts[symbol] = max(rows, insufficient_counts.get(symbol, 0))
                continue
            frames[symbol] = frame
            insufficient_counts.pop(symbol, None)

    def _remaining_symbols() -> list[str]:
        return [symbol for symbol in ordered_symbols if symbol not in frames]

    for feed in (DataFeed.IEX, DataFeed.SIP):
        remaining = _remaining_symbols()
        if not remaining:
            break
        try:
            _accept_resolved(
                _request_alpaca_daily_frames(
                    client=client,
                    symbols=remaining,
                    start=start,
                    end=now,
                    feed=feed,
                    now=now,
                )
            )
        except Exception as exc:
            fetch_errors.append(f"{getattr(feed, 'value', feed)} batch: {exc}")

        for symbol in list(_remaining_symbols()):
            try:
                _accept_resolved(
                    _request_alpaca_daily_frames(
                        client=client,
                        symbols=[symbol],
                        start=start,
                        end=now,
                        feed=feed,
                        now=now,
                    )
                )
            except Exception as exc:
                fetch_errors.append(f"{getattr(feed, 'value', feed)} {symbol}: {exc}")

    if not frames:
        details = f" ({'; '.join(fetch_errors)})" if fetch_errors else ""
        raise RuntimeError(f"No Alpaca daily bars returned{details}")

    missing = _remaining_symbols()
    if missing:
        missing_with_counts = [
            f"{symbol} ({insufficient_counts[symbol]} bars)"
            if symbol in insufficient_counts
            else symbol
            for symbol in missing
        ]
        if any(symbol in insufficient_counts for symbol in missing):
            raise ValueError(
                f"Missing enough Alpaca daily bars for: {', '.join(missing_with_counts)}; need at least {min_days}"
            )
        raise RuntimeError(f"Missing Alpaca daily bars for: {', '.join(missing_with_counts)}")
    return _align_frames({symbol: frames[symbol] for symbol in ordered_symbols})


def load_state(path: Path = STATE_PATH) -> StrategyState:
    try:
        with _state_file_guard(path, write=False):
            if not path.exists():
                return StrategyState()
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError(f"Expected JSON object, got {type(payload).__name__}")
            return StrategyState(
                active_symbol=payload.get("active_symbol"),
                active_qty=float(payload.get("active_qty", 0.0) or 0.0),
                entry_price=float(payload.get("entry_price", 0.0) or 0.0),
                entry_date=payload.get("entry_date"),
                last_run_date=payload.get("last_run_date"),
                last_signal_action=payload.get("last_signal_action"),
                last_signal_timestamp=payload.get("last_signal_timestamp"),
                last_order_id=payload.get("last_order_id"),
            )
    except Exception as exc:
        raise RuntimeError(f"Unreadable strategy state file {path}: {exc}") from exc


def _state_lock_path(path: Path) -> Path:
    return path.parent / f".{path.name}{STATE_LOCK_SUFFIX}"


def _ensure_non_symlink_artifact_path(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise OSError(f"Unsafe {label} path is symlinked: {path}")


def _ensure_non_symlink_artifact_parent_dirs(path: Path, *, label: str) -> None:
    for parent in path.parents:
        if parent.is_symlink():
            raise OSError(f"Unsafe {label} parent directory is symlinked: {parent}")


def _jsonl_log_lock_path(path: Path) -> Path:
    return path.parent / f".{path.name}{STATE_LOCK_SUFFIX}"


@contextmanager
def _state_file_guard(path: Path, *, write: bool) -> Iterator[None]:
    lock_path = _state_lock_path(path)
    _ensure_non_symlink_artifact_path(path, label="strategy state file")
    _ensure_non_symlink_artifact_parent_dirs(path, label="strategy state file")
    _ensure_non_symlink_artifact_path(lock_path, label="strategy state lock file")
    _ensure_non_symlink_artifact_parent_dirs(lock_path, label="strategy state lock file")
    guard = shared_path_guard(lock_path)
    acquire = guard.acquire_write if write else guard.acquire_read
    release = guard.release_write if write else guard.release_read
    handle = None
    acquire()
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+", encoding="utf-8")
        if _fcntl is not None:
            lock_mode = _fcntl.LOCK_EX if write else _fcntl.LOCK_SH
            _fcntl.flock(handle.fileno(), lock_mode)
        yield
    finally:
        try:
            if _fcntl is not None and handle is not None:
                _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)
        finally:
            if handle is not None:
                handle.close()
            release()


@contextmanager
def _jsonl_log_guard(path: Path, *, label: str) -> Iterator[None]:
    lock_path = _jsonl_log_lock_path(path)
    _ensure_non_symlink_artifact_path(path, label=label)
    _ensure_non_symlink_artifact_parent_dirs(path, label=label)
    _ensure_non_symlink_artifact_path(lock_path, label=f"{label} lock file")
    _ensure_non_symlink_artifact_parent_dirs(lock_path, label=f"{label} lock file")
    guard = shared_path_guard(lock_path)
    handle = None
    guard.acquire_write()
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+", encoding="utf-8")
        if _fcntl is not None:
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX)
        yield
    finally:
        try:
            if _fcntl is not None and handle is not None:
                _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)
        finally:
            if handle is not None:
                handle.close()
            guard.release_write()


def save_state(state: StrategyState, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(state), indent=2, sort_keys=True)
    temp_path: Path | None = None
    with _state_file_guard(path, write=True):
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    if temp_path is not None and temp_path.exists():
        temp_path.unlink(missing_ok=True)


def append_signal_log(payload: dict, path: Path = SIGNAL_LOG_PATH) -> None:
    with _jsonl_log_guard(path, label="signal log"):
        append_jsonl_row(path, payload, sort_keys=True)


def _append_jsonl_log_best_effort(
    *,
    payload: dict,
    path: Path,
    writer,
    log_label: str,
) -> str | None:
    try:
        writer(payload, path=path)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("%s write failed for %s: %s", log_label, path, exc)
        return str(exc)
    return None


def _append_signal_log_best_effort(payload: dict, path: Path = SIGNAL_LOG_PATH) -> str | None:
    return _append_jsonl_log_best_effort(
        payload=payload,
        path=path,
        writer=append_signal_log,
        log_label="Signal log",
    )


def append_run_event_log(payload: dict, path: Path = RUN_EVENT_LOG_PATH) -> None:
    with _jsonl_log_guard(path, label="run event log"):
        append_jsonl_row(path, payload, sort_keys=True)


def _append_run_event_log_best_effort(payload: dict, path: Path = RUN_EVENT_LOG_PATH) -> str | None:
    return _append_jsonl_log_best_effort(
        payload=payload,
        path=path,
        writer=append_run_event_log,
        log_label="Run event log",
    )


def latest_close_prices(frames: dict[str, pd.DataFrame]) -> dict[str, float]:
    return {symbol: float(frame["close"].iloc[-1]) for symbol, frame in frames.items()}


def load_latest_quotes_with_source(
    symbols: Iterable[str],
    *,
    paper: bool,
    fallback_prices: dict[str, float],
    data_client=None,
) -> tuple[dict[str, float], str, dict[str, str]]:
    from alpaca.data import StockLatestQuoteRequest
    from alpaca.data.enums import DataFeed

    normalized_symbols = _normalize_stock_symbol_list(symbols)
    client = data_client or build_data_client(paper=paper)
    request = StockLatestQuoteRequest(symbol_or_symbols=normalized_symbols, feed=DataFeed.IEX)
    try:
        quotes = client.get_stock_latest_quote(request)
    except Exception as exc:
        logger.warning("Falling back to previous close prices for quotes: %s", exc)
        prices = {symbol: float(fallback_prices[symbol]) for symbol in normalized_symbols}
        sources = {symbol: "close_fallback" for symbol in normalized_symbols}
        return prices, "close_fallback", sources

    prices: dict[str, float] = {}
    quote_source_by_symbol: dict[str, str] = {}
    for symbol in normalized_symbols:
        quote = quotes.get(symbol)
        ask = float(getattr(quote, "ask_price", 0.0) or 0.0) if quote is not None else 0.0
        bid = float(getattr(quote, "bid_price", 0.0) or 0.0) if quote is not None else 0.0
        if ask > 0.0 and bid > 0.0:
            prices[symbol] = (ask + bid) / 2.0
        else:
            prices[symbol] = ask or bid or float(fallback_prices[symbol])
        quote_source_by_symbol[symbol] = "alpaca" if (ask > 0.0 or bid > 0.0) else "close_fallback"

    overall_source = (
        "alpaca"
        if all(source == "alpaca" for source in quote_source_by_symbol.values())
        else "mixed_fallback"
    )
    return prices, overall_source, quote_source_by_symbol


def load_latest_quotes(
    symbols: Iterable[str],
    *,
    paper: bool,
    fallback_prices: dict[str, float],
    data_client=None,
) -> dict[str, float]:
    prices, _, _ = load_latest_quotes_with_source(
        symbols,
        paper=paper,
        fallback_prices=fallback_prices,
        data_client=data_client,
    )
    return prices


def load_inference_frames(
    symbols: Iterable[str],
    *,
    paper: bool,
    data_dir: str,
    now: datetime,
    data_client=None,
) -> tuple[dict[str, pd.DataFrame], str]:
    try:
        frames = load_alpaca_daily_frames(
            symbols,
            paper=paper,
            min_days=DEFAULT_DAILY_FRAME_MIN_DAYS,
            history_days=DEFAULT_ALPACA_DAILY_HISTORY_LOOKBACK_DAYS,
            data_client=data_client,
            now=now,
        )
        return frames, "alpaca"
    except Exception as exc:
        logger.warning("Falling back to local daily CSVs for inference bars: %s", exc)
        frames = load_local_daily_frames(
            symbols,
            data_dir=data_dir,
            min_days=DEFAULT_DAILY_FRAME_MIN_DAYS,
        )
        return frames, "local_fallback"


def _prune_ordered_cache(cache: OrderedDict[object, object], *, max_entries: int) -> None:
    while len(cache) > max_entries:
        cache.popitem(last=False)


def _clone_daily_trader_template(template: DailyPPOTrader) -> DailyPPOTrader:
    """Return a fresh runtime-state clone that reuses the loaded policy module."""
    trader = copy.copy(template)
    trader.SYMBOLS = list(template.SYMBOLS)
    trader.current_position = None
    trader.cash = 10000.0
    trader.position_qty = 0.0
    trader.entry_price = 0.0
    trader.hold_hours = 0
    trader.step = 0
    if hasattr(trader, "hold_days"):
        trader.hold_days = 0
    return trader


def _load_cached_daily_trader(
    checkpoint_path: str,
    *,
    device: str = "cpu",
    long_only: bool = False,
    symbols: Optional[Sequence[str]] = None,
    allow_unsafe_checkpoint_loading: bool = False,
) -> DailyPPOTrader:
    normalized_symbols = tuple(str(symbol).upper() for symbol in (symbols or []))
    checkpoint = Path(checkpoint_path).expanduser()
    resolved_path = checkpoint.resolve(strict=False)
    try:
        stat_result = resolved_path.stat()
    except OSError:
        stat_result = None
    cache_key = (
        resolved_path,
        str(device),
        bool(long_only),
        normalized_symbols,
        bool(allow_unsafe_checkpoint_loading),
    )
    if stat_result is not None:
        while True:
            with _DAILY_TRADER_CACHE_LOCK:
                cached = _DAILY_TRADER_CACHE.get(cache_key)
                if cached is not None and cached.size == stat_result.st_size and cached.mtime_ns == stat_result.st_mtime_ns:
                    _DAILY_TRADER_CACHE.move_to_end(cache_key)
                    return _clone_daily_trader_template(cached.trader)
                inflight = _DAILY_TRADER_CACHE_INFLIGHT.get(cache_key)
                if inflight is None:
                    inflight = Event()
                    _DAILY_TRADER_CACHE_INFLIGHT[cache_key] = inflight
                    break
            inflight.wait()

    try:
        trader = DailyPPOTrader(
            checkpoint_path,
            device=device,
            long_only=long_only,
            symbols=list(normalized_symbols) if normalized_symbols else None,
            allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
        )
        if stat_result is not None:
            template = _clone_daily_trader_template(trader)
            with _DAILY_TRADER_CACHE_LOCK:
                _DAILY_TRADER_CACHE[cache_key] = _DailyTraderCacheEntry(
                    size=stat_result.st_size,
                    mtime_ns=stat_result.st_mtime_ns,
                    trader=template,
                )
                _DAILY_TRADER_CACHE.move_to_end(cache_key)
                _prune_ordered_cache(_DAILY_TRADER_CACHE, max_entries=_DAILY_TRADER_CACHE_MAX_ENTRIES)
                inflight = _DAILY_TRADER_CACHE_INFLIGHT.pop(cache_key, None)
                if inflight is not None:
                    inflight.set()
        return trader
    except Exception:
        if stat_result is not None:
            with _DAILY_TRADER_CACHE_LOCK:
                inflight = _DAILY_TRADER_CACHE_INFLIGHT.pop(cache_key, None)
                if inflight is not None:
                    inflight.set()
        raise


def _load_bare_policy(
    checkpoint_path: str,
    obs_size: int,
    num_actions: int,
    device: str,
    *,
    allow_unsafe_checkpoint_loading: bool = False,
):
    """Load a policy nn.Module from a checkpoint without full PPOTrader overhead."""
    checkpoint = Path(checkpoint_path).expanduser()
    resolved_path = checkpoint.resolve(strict=False)
    try:
        stat_result = resolved_path.stat()
    except OSError:
        stat_result = None
    cache_key = (
        resolved_path,
        int(obs_size),
        int(num_actions),
        str(device),
        bool(allow_unsafe_checkpoint_loading),
    )
    if stat_result is not None:
        while True:
            with _BARE_POLICY_CACHE_LOCK:
                cached = _BARE_POLICY_CACHE.get(cache_key)
                if cached is not None and cached.size == stat_result.st_size and cached.mtime_ns == stat_result.st_mtime_ns:
                    _BARE_POLICY_CACHE.move_to_end(cache_key)
                    # Bare policies are eval-only (no mutable state changed during inference_mode forward)
                    return cached.policy
                inflight = _BARE_POLICY_CACHE_INFLIGHT.get(cache_key)
                if inflight is None:
                    inflight = Event()
                    _BARE_POLICY_CACHE_INFLIGHT[cache_key] = inflight
                    break
            inflight.wait()

    try:
        ckpt = load_checkpoint_payload(
            checkpoint_path,
            map_location=device,
            allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
        )
        # Support multiple checkpoint formats: direct state_dict, {"model": sd}, {"model_state_dict": sd}
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        encoder_key = [k for k in state_dict if "encoder" in k and "weight" in k]
        if encoder_key:
            hidden = state_dict[encoder_key[0]].shape[0]
            if isinstance(ckpt, dict) and "use_encoder_norm" in ckpt:
                has_encoder_norm = bool(ckpt["use_encoder_norm"])
            else:
                has_encoder_norm = any("encoder_norm" in k for k in state_dict)
            activation = "relu"
            if isinstance(ckpt, dict):
                raw_arch = str(ckpt.get("arch") or "").strip().lower()
                raw_activation = str(ckpt.get("activation") or "").strip().lower()
                if raw_arch == "mlp_relu_sq" or raw_activation == "relu_sq":
                    activation = "relu_sq"
            from pufferlib_market.train import TradingPolicy
            policy = TradingPolicy(
                obs_size,
                num_actions,
                hidden,
                activation=activation,
                use_encoder_norm=has_encoder_norm,
            )
        else:
            input_proj_key = [k for k in state_dict if "input_proj" in k and "weight" in k]
            hidden = state_dict[input_proj_key[0]].shape[0] if input_proj_key else 256
            from pufferlib_market.inference import Policy
            policy = Policy(obs_size, num_actions, hidden, 3)
        policy.load_state_dict(state_dict, strict=False)
        policy.to(torch.device(device))
        policy.eval()
        if stat_result is not None:
            with _BARE_POLICY_CACHE_LOCK:
                _BARE_POLICY_CACHE[cache_key] = _BarePolicyCacheEntry(
                    size=stat_result.st_size,
                    mtime_ns=stat_result.st_mtime_ns,
                    policy=policy,
                )
                _BARE_POLICY_CACHE.move_to_end(cache_key)
                _prune_ordered_cache(_BARE_POLICY_CACHE, max_entries=_BARE_POLICY_CACHE_MAX_ENTRIES)
                inflight = _BARE_POLICY_CACHE_INFLIGHT.pop(cache_key, None)
                if inflight is not None:
                    inflight.set()
        return policy
    except Exception:
        if stat_result is not None:
            with _BARE_POLICY_CACHE_LOCK:
                inflight = _BARE_POLICY_CACHE_INFLIGHT.pop(cache_key, None)
                if inflight is not None:
                    inflight.set()
        raise


def _ensemble_softmax_signal(
    primary: DailyPPOTrader,
    extra_policies: list,
    features: np.ndarray,
    prices: dict,
):
    """Softmax-average probabilities across primary + extra policies, return TradingSignal."""
    obs = primary.build_observation(features, prices)
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(primary.device)
    all_probs = []
    all_values: list[float] = []
    with torch.inference_mode():
        logits, value = primary.policy(obs_t)
        logits = primary.apply_action_constraints(logits)
        all_probs.append(F.softmax(logits, dim=-1))
        all_values.append(float(value.item()))
        for pol in extra_policies:
            logits_i, value_i = pol(obs_t)
            logits_i = primary.apply_action_constraints(logits_i)
            all_probs.append(F.softmax(logits_i, dim=-1))
            all_values.append(float(value_i.item()))
    avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
    action = int(avg_probs.argmax(dim=-1).item())
    confidence = float(avg_probs[0, action].item())
    value_est = float(sum(all_values) / max(len(all_values), 1))
    return primary._decode_action(action, confidence, value_est)


_META_SELECTOR_INSTANCE: "MetaSelector | None" = None


def _get_or_create_meta_selector(
    checkpoint_paths: list[str],
    symbols: list[str],
    top_k: int = 1,
    lookback: int = 3,
    device: str = "cpu",
) -> "MetaSelector":
    global _META_SELECTOR_INSTANCE
    if _META_SELECTOR_INSTANCE is not None:
        return _META_SELECTOR_INSTANCE
    from src.meta_selector import MetaSelector
    state_path = STATE_PATH / "meta_selector_state.json"
    sel = MetaSelector(
        checkpoint_paths,
        symbols,
        top_k=top_k,
        lookback=lookback,
        device=device,
        state_path=state_path,
    )
    _META_SELECTOR_INSTANCE = sel
    logger.info("MetaSelector created: %d models, top_k=%d, lookback=%d", len(sel.names), top_k, lookback)
    return sel


def _meta_selector_signal(
    checkpoint_paths: list[str],
    symbols: list[str],
    features: np.ndarray,
    prices: dict[str, float],
    frames: dict[str, pd.DataFrame],
    top_k: int = 1,
    lookback: int = 3,
    device: str = "cpu",
) -> TradingSignal:
    """Build a TradingSignal using MetaSelector (momentum-based model selection)."""
    sel = _get_or_create_meta_selector(checkpoint_paths, symbols, top_k, lookback, device)
    if sel._day_count == 0 and frames:
        sel.warmup_from_frames(frames, min_days=10)
    meta_sig = sel.get_meta_signal(features, prices)
    if not meta_sig.selected_symbols or all(s is None for s in meta_sig.selected_symbols):
        return TradingSignal(
            action="flat", symbol=None, direction=None,
            confidence=0.0, value_estimate=0.0,
            allocation_pct=0.0, level_offset_bps=0.0,
        )
    sym = meta_sig.selected_symbols[0]
    conf = meta_sig.confidences[0]
    logger.info("Meta-selector: model=%s sym=%s conf=%.3f returns=%s",
                meta_sig.selected_models[0], sym, conf,
                {n: f"{r:+.2%}" for n, r in meta_sig.model_returns.items()})
    return TradingSignal(
        action="long" if sym else "flat",
        symbol=sym,
        direction="long" if sym else None,
        confidence=conf,
        value_estimate=0.0,
        allocation_pct=1.0,
        level_offset_bps=0.0,
    )


def _ensemble_top_k_signals(
    primary: DailyPPOTrader,
    extra_policies: list,
    features: np.ndarray,
    prices: dict,
    k: int = 4,
    min_prob_ratio: float = 0.3,
):
    """Return top K long signals from ensemble softmax, with proportional allocation.

    Args:
        k: Max number of simultaneous positions
        min_prob_ratio: Only include signals with prob >= min_prob_ratio * top_prob
    """
    obs = primary.build_observation(features, prices)
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(primary.device)
    all_probs = []
    all_values: list[float] = []
    with torch.inference_mode():
        logits, value = primary.policy(obs_t)
        logits = primary.apply_action_constraints(logits)
        all_probs.append(F.softmax(logits, dim=-1))
        all_values.append(float(value.item()))
        for pol in extra_policies:
            logits_i, value_i = pol(obs_t)
            logits_i = primary.apply_action_constraints(logits_i)
            all_probs.append(F.softmax(logits_i, dim=-1))
            all_values.append(float(value_i.item()))
    avg_probs = torch.stack(all_probs, dim=0).mean(dim=0).squeeze(0)  # [num_actions]
    value_est = float(sum(all_values) / max(len(all_values), 1))

    # Collect per-symbol long probabilities (sum across allocation/level bins)
    num_symbols = primary.num_symbols
    per_sym = max(1, primary.per_symbol_actions)
    symbol_probs: list[tuple[int, float]] = []
    for sym_idx in range(num_symbols):
        start = 1 + sym_idx * per_sym
        end = start + per_sym
        sym_prob = float(avg_probs[start:end].sum().item())
        symbol_probs.append((sym_idx, sym_prob))

    # Sort by probability descending
    symbol_probs.sort(key=lambda x: -x[1])
    flat_prob = float(avg_probs[0].item())

    signals = []
    top_prob = symbol_probs[0][1] if symbol_probs else 0.0
    prob_threshold = max(flat_prob, top_prob * min_prob_ratio)

    for sym_idx, sym_prob in symbol_probs[:k]:
        if sym_prob < prob_threshold:
            break
        symbol = primary.SYMBOLS[sym_idx]
        signals.append(TradingSignal(
            action=f"long_{symbol}",
            symbol=symbol,
            direction="long",
            confidence=sym_prob,
            value_estimate=value_est,
            allocation_pct=sym_prob,  # Will be normalized later
            level_offset_bps=0.0,
        ))

    # Normalize allocations to sum to 1.0
    if signals:
        total_prob = sum(s.allocation_pct for s in signals)
        for i, s in enumerate(signals):
            signals[i] = TradingSignal(
                action=s.action,
                symbol=s.symbol,
                direction=s.direction,
                confidence=s.confidence,
                value_estimate=s.value_estimate,
                allocation_pct=s.allocation_pct / total_prob,
                level_offset_bps=s.level_offset_bps,
            )

    return signals


def _build_daily_feature_cube(
    indexed: dict[str, pd.DataFrame],
    *,
    symbols: Sequence[str],
    feature_schema: DailyStockFeatureSchema,
) -> np.ndarray:
    """Materialize aligned daily features once as [time, symbol, feature]."""
    feature_dimension = daily_feature_dimension(feature_schema)
    feature_blocks = [
        _daily_feature_history_for_schema(indexed[symbol], feature_schema=feature_schema).to_numpy(dtype=np.float32, copy=False)
        for symbol in symbols
    ]
    if not feature_blocks:
        return np.zeros((0, 0, feature_dimension), dtype=np.float32)
    return np.stack(feature_blocks, axis=1)


def _build_daily_close_matrix(
    indexed: dict[str, pd.DataFrame],
    *,
    symbols: Sequence[str],
) -> np.ndarray:
    close_columns = [
        indexed[symbol]["close"].to_numpy(dtype=np.float64, copy=False)
        for symbol in symbols
    ]
    if not close_columns:
        return np.zeros((0, 0), dtype=np.float64)
    return np.stack(close_columns, axis=1)


def _build_daily_timestamps(
    indexed: dict[str, pd.DataFrame],
    *,
    symbols: Sequence[str],
) -> tuple[datetime, ...]:
    if not symbols:
        return ()
    reference_index = indexed[symbols[0]].index
    timestamps: list[datetime] = []
    for raw_timestamp in reference_index:
        current_now = pd.Timestamp(raw_timestamp).to_pydatetime()
        if current_now.tzinfo is None:
            current_now = current_now.replace(tzinfo=timezone.utc)
        timestamps.append(current_now)
    return tuple(timestamps)


def _daily_feature_history_for_schema(
    price_df: pd.DataFrame,
    *,
    feature_schema: DailyStockFeatureSchema,
) -> pd.DataFrame:
    if feature_schema == "legacy_prod":
        return build_daily_feature_history_for_schema(price_df, schema="legacy_prod")
    try:
        return compute_daily_feature_history(price_df, schema="rsi_v5")
    except TypeError:
        return compute_daily_feature_history(price_df)


def _daily_feature_vector_for_schema(
    price_df: pd.DataFrame,
    *,
    feature_schema: DailyStockFeatureSchema,
) -> np.ndarray:
    if feature_schema == "legacy_prod":
        return compute_daily_feature_vector_for_schema(price_df, schema="legacy_prod")
    try:
        return compute_daily_features(price_df, schema="rsi_v5")
    except TypeError:
        return compute_daily_features(price_df)


def _prepare_daily_backtest_data(
    *,
    checkpoint: str,
    symbols: Iterable[str],
    data_dir: str,
    days: int,
    extra_checkpoints: Optional[list[str]] = None,
    allow_unsafe_checkpoint_loading: bool = False,
) -> PreparedDailyBacktestData:
    frames = load_local_daily_frames(
        symbols,
        data_dir=data_dir,
        min_days=days + DEFAULT_DAILY_FRAME_MIN_DAYS,
    )
    indexed = {
        symbol: frame.set_index("timestamp")[["open", "high", "low", "close", "volume"]].copy()
        for symbol, frame in frames.items()
    }
    min_len = min(len(frame) for frame in indexed.values())
    start = min_len - days
    if start < 1:
        raise ValueError(f"Need at least {days + 1} aligned days for backtest")

    trader_template = _load_cached_daily_trader(
        checkpoint,
        device="cpu",
        long_only=True,
        symbols=list(indexed.keys()),
        allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
    )
    extra_policies = [
        _load_bare_policy(
            str((REPO / path).resolve()) if not Path(path).is_absolute() else path,
            trader_template.obs_size,
            trader_template.num_actions,
            "cpu",
            allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
        )
        for path in (extra_checkpoints or [])
    ]
    feature_schema = resolve_daily_feature_schema(
        checkpoint,
        extra_checkpoints=extra_checkpoints,
    )
    symbols_order = tuple(str(symbol).upper() for symbol in trader_template.SYMBOLS)
    feature_cube = _build_daily_feature_cube(
        indexed,
        symbols=symbols_order,
        feature_schema=feature_schema,
    )
    close_matrix = _build_daily_close_matrix(
        indexed,
        symbols=symbols_order,
    )
    timestamps = _build_daily_timestamps(
        indexed,
        symbols=symbols_order,
    )
    return PreparedDailyBacktestData(
        feature_schema=feature_schema,
        indexed=indexed,
        trader_template=trader_template,
        extra_policies=extra_policies,
        symbols=symbols_order,
        feature_cube=feature_cube,
        close_matrix=close_matrix,
        timestamps=timestamps,
        start=start,
        min_len=min_len,
    )


def _build_multi_position_signals(
    checkpoint: str,
    frames: dict[str, pd.DataFrame],
    *,
    quotes: dict[str, float],
    portfolio_value: float,
    multi_position: int,
    multi_position_min_prob_ratio: float,
    device: str = "cpu",
    extra_checkpoints: Optional[list[str]] = None,
    allow_unsafe_checkpoint_loading: bool = False,
) -> list[TradingSignal]:
    aligned = _align_frames(frames)
    indexed = {
        symbol: frame.set_index("timestamp")[["open", "high", "low", "close", "volume"]].copy()
        for symbol, frame in aligned.items()
    }
    trader = _load_cached_daily_trader(
        checkpoint,
        device=device,
        long_only=True,
        symbols=list(indexed.keys()),
        allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
    )
    _apply_portfolio_context_to_trader(
        trader,
        portfolio=PortfolioContext(cash=max(0.0, float(portfolio_value))),
    )
    feature_schema = resolve_daily_feature_schema(
        checkpoint,
        extra_checkpoints=extra_checkpoints,
    )
    feature_dimension = daily_feature_dimension(feature_schema)
    features = np.zeros((trader.num_symbols, feature_dimension), dtype=np.float32)
    for index, symbol in enumerate(trader.SYMBOLS):
        features[index] = _daily_feature_vector_for_schema(indexed[symbol], feature_schema=feature_schema)
    prices = {
        symbol: float(quotes.get(symbol, frame["close"].iloc[-1]) or frame["close"].iloc[-1])
        for symbol, frame in aligned.items()
    }
    extra_policies = [
        _load_bare_policy(
            str((REPO / path).resolve()) if not Path(path).is_absolute() else path,
            trader.obs_size,
            trader.num_actions,
            device,
            allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
        )
        for path in (extra_checkpoints or [])
    ]
    return _ensemble_top_k_signals(
        trader,
        extra_policies,
        features,
        prices,
        k=multi_position,
        min_prob_ratio=multi_position_min_prob_ratio,
    )


def _trader_signal_from_features(
    trader: DailyPPOTrader,
    *,
    features: np.ndarray,
    prices: dict[str, float],
    indexed: dict[str, pd.DataFrame],
    idx: int,
) -> TradingSignal:
    """Prefer direct feature inference, but preserve compatibility with simpler test doubles."""
    get_signal = getattr(trader, "get_signal", None)
    if callable(get_signal):
        return cast(TradingSignal, get_signal(features, prices))
    return cast(
        TradingSignal,
        trader.get_daily_signal(
            {symbol: frame.iloc[: idx + 1] for symbol, frame in indexed.items()},
            prices,
        ),
    )


def _apply_portfolio_context_to_trader(
    trader: DailyPPOTrader,
    *,
    portfolio: PortfolioContext,
) -> None:
    trader.cash = float(portfolio.cash)
    trader.position_qty = float(portfolio.position_qty)
    trader.entry_price = float(portfolio.entry_price)
    trader.hold_days = int(max(0, portfolio.hold_days))
    trader.hold_hours = trader.hold_days
    trader.step = min(trader.hold_days, trader.max_steps)
    trader.current_position = None
    if portfolio.current_symbol:
        symbol_upper = portfolio.current_symbol.upper()
        if symbol_upper in trader.SYMBOLS and trader.position_qty > 0:
            trader.current_position = trader.SYMBOLS.index(symbol_upper)


def _open_gate_reasons(
    signal,
    *,
    signal_quote_source: str,
    quote_data_source: str,
    min_open_confidence: float,
    min_open_value_estimate: float,
) -> list[str]:
    reasons: list[str] = []
    if not signal.symbol or signal.direction != "long":
        return reasons
    if signal_quote_source != "alpaca":
        reasons.append(
            f"quote_source={signal_quote_source}, overall_quote_source={quote_data_source}"
        )
    confidence = float(getattr(signal, "confidence", 0.0) or 0.0)
    if confidence < float(min_open_confidence):
        reasons.append(
            f"confidence={confidence:.4f} < min_open_confidence={float(min_open_confidence):.4f}"
        )
    value_estimate = float(getattr(signal, "value_estimate", 0.0) or 0.0)
    if value_estimate < float(min_open_value_estimate):
        reasons.append(
            "value_estimate="
            f"{value_estimate:.4f} < min_open_value_estimate={float(min_open_value_estimate):.4f}"
        )
    return reasons


def build_signal(
    checkpoint: str,
    frames: dict[str, pd.DataFrame],
    *,
    device: str = "cpu",
    portfolio: PortfolioContext = PortfolioContext(),
    extra_checkpoints: Optional[list] = None,
    allow_unsafe_checkpoint_loading: bool = False,
    meta_selector: bool = False,
    meta_top_k: int = 1,
    meta_lookback: int = 3,
):
    aligned = _align_frames(frames)
    indexed = {
        symbol: frame.set_index("timestamp")[["open", "high", "low", "close", "volume"]].copy()
        for symbol, frame in aligned.items()
    }
    prices = {symbol: float(frame["close"].iloc[-1]) for symbol, frame in aligned.items()}
    trader = _load_cached_daily_trader(
        checkpoint,
        device=device,
        long_only=True,
        symbols=list(indexed.keys()),
        allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
    )
    _apply_portfolio_context_to_trader(trader, portfolio=portfolio)
    feature_schema = resolve_daily_feature_schema(
        checkpoint,
        extra_checkpoints=extra_checkpoints,
    )
    feature_dimension = daily_feature_dimension(feature_schema)
    features = np.zeros((trader.num_symbols, feature_dimension), dtype=np.float32)
    for i, sym in enumerate(trader.SYMBOLS):
        if sym in indexed:
            features[i] = _daily_feature_vector_for_schema(indexed[sym], feature_schema=feature_schema)

    if meta_selector and extra_checkpoints:
        all_paths = [checkpoint] + list(extra_checkpoints)
        signal = _meta_selector_signal(
            all_paths, list(indexed.keys()), features, prices, indexed,
            top_k=meta_top_k, lookback=meta_lookback, device=device,
        )
        logger.info("Meta-selector signal (%d models, top_k=%d, lookback=%d)", len(all_paths), meta_top_k, meta_lookback)
    elif extra_checkpoints:
        extra_policies = [
            _load_bare_policy(
                str((REPO / p).resolve()) if not Path(p).is_absolute() else p,
                trader.obs_size,
                trader.num_actions,
                device,
                allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
            )
            for p in extra_checkpoints
        ]
        signal = _ensemble_softmax_signal(trader, extra_policies, features, prices)
        logger.info("Ensemble signal (%d policies, softmax_avg)", 1 + len(extra_policies))
    else:
        signal = _trader_signal_from_features(
            trader,
            features=features,
            prices=prices,
            indexed=indexed,
            idx=max(len(frame) for frame in indexed.values()) - 1,
        )

    if signal.direction == "short":
        logger.warning("Checkpoint produced short signal on long-only path; flattening")
        signal = signal.__class__(
            action="flat",
            symbol=None,
            direction=None,
            confidence=float(signal.confidence),
            value_estimate=float(signal.value_estimate),
            allocation_pct=0.0,
            level_offset_bps=0.0,
    )
    return signal, prices


def latest_bar_timestamp(frames: dict[str, pd.DataFrame]) -> pd.Timestamp:
    if not frames:
        raise ValueError("No frames available")
    latest = pd.Timestamp(next(iter(frames.values()))["timestamp"].iloc[-1])
    return latest.tz_localize("UTC") if latest.tzinfo is None else latest.tz_convert("UTC")


def bars_are_fresh(
    *,
    latest_bar: pd.Timestamp,
    now: datetime,
    max_age_days: int = DEFAULT_BAR_FRESHNESS_MAX_AGE_DAYS,
) -> bool:
    latest_bar = latest_bar.tz_localize("UTC") if latest_bar.tzinfo is None else latest_bar.tz_convert("UTC")
    age_days = (now.date() - latest_bar.date()).days
    return age_days <= max(0, int(max_age_days))


def _signed_position_qty(position) -> float:
    qty = float(getattr(position, "qty", 0.0) or 0.0)
    side = str(getattr(position, "side", "long") or "long").lower()
    if side == "short" and qty > 0:
        return -qty
    return qty


def positions_by_symbol(client, symbols: Iterable[str]) -> dict[str, object]:
    target = {str(symbol).upper() for symbol in symbols}
    positions: dict[str, object] = {}
    for position in client.get_all_positions():
        symbol = str(getattr(position, "symbol", "")).upper()
        if symbol in target:
            positions[symbol] = position
    return positions


def _market_order_side_for_qty(qty: float) -> str:
    return "sell" if qty > 0 else "buy"


def submit_market_order(client, *, symbol: str, qty: float, side: str):
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    if qty <= 0:
        raise ValueError("qty must be positive")
    side_value = OrderSide.BUY if side == "buy" else OrderSide.SELL
    request = MarketOrderRequest(
        symbol=symbol,
        qty=round(float(qty), 4),
        side=side_value,
        time_in_force=TimeInForce.DAY,
    )
    return client.submit_order(request)


def submit_limit_order(
    client,
    *,
    symbol: str,
    qty: float,
    side: str,
    limit_price: float,
) -> object:
    """Submit a DAY limit order via Alpaca."""
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest

    if qty <= 0:
        raise ValueError("qty must be positive")
    side_value = OrderSide.BUY if side == "buy" else OrderSide.SELL
    request = LimitOrderRequest(
        symbol=symbol,
        qty=round(float(qty), 4),
        side=side_value,
        time_in_force=TimeInForce.DAY,
        limit_price=round(float(limit_price), 2),
    )
    return client.submit_order(request)


def compute_target_qty(*, account, price: float, allocation_pct: float) -> float:
    portfolio_value = account_portfolio_value(account)
    buying_power = float(getattr(account, "buying_power", 0.0) or 0.0)
    return compute_target_qty_from_values(
        portfolio_value=portfolio_value,
        buying_power=buying_power,
        price=price,
        allocation_pct=allocation_pct,
    )


def account_portfolio_value(account) -> float:
    return float(
        getattr(account, "portfolio_value", 0.0)
        or getattr(account, "equity", 0.0)
        or getattr(account, "cash", 0.0)
        or getattr(account, "buying_power", 0.0)
        or 0.0
    )


def compute_target_qty_from_values(
    *,
    portfolio_value: float,
    buying_power: float,
    price: float,
    allocation_pct: float,
) -> float:
    if price <= 0 or portfolio_value <= 0 or buying_power <= 0:
        return 0.0
    target_notional = portfolio_value * max(0.0, allocation_pct) / 100.0
    target_notional = min(target_notional, buying_power * BUYING_POWER_USAGE_CAP)
    if target_notional <= 0:
        return 0.0
    return round(target_notional / price, 4)


def _raw_portfolio_target_qty(
    *,
    portfolio_value: float,
    price: float,
    total_allocation_pct: float,
    allocation_fraction: float,
) -> float:
    if portfolio_value <= 0.0 or price <= 0.0:
        return 0.0
    target_notional = portfolio_value * max(0.0, float(total_allocation_pct)) / 100.0
    target_notional *= max(0.0, float(allocation_fraction))
    if target_notional <= 0.0:
        return 0.0
    return round(target_notional / price, 4)


def _portfolio_rebalance_targets(
    *,
    desired_allocations: Mapping[str, float],
    existing_qty_by_symbol: Mapping[str, float],
    buy_prices: Mapping[str, float],
    sell_prices: Mapping[str, float],
    portfolio_value: float,
    buying_power: float,
    total_allocation_pct: float,
) -> dict[str, PortfolioRebalanceTarget]:
    """Compute rebalance targets while sharing one buying-power budget."""
    raw_targets: dict[str, float] = {}
    available_buy_notional = max(0.0, float(buying_power)) * BUYING_POWER_USAGE_CAP
    all_symbols = set(existing_qty_by_symbol) | set(desired_allocations)
    for symbol in all_symbols:
        existing_qty = max(0.0, float(existing_qty_by_symbol.get(symbol, 0.0) or 0.0))
        allocation_fraction = max(0.0, float(desired_allocations.get(symbol, 0.0) or 0.0))
        raw_target = _raw_portfolio_target_qty(
            portfolio_value=portfolio_value,
            price=float(buy_prices.get(symbol, 0.0) or 0.0),
            total_allocation_pct=total_allocation_pct,
            allocation_fraction=allocation_fraction,
        )
        raw_targets[symbol] = raw_target
        sell_delta = max(0.0, existing_qty - raw_target)
        sell_price = float(sell_prices.get(symbol, 0.0) or 0.0)
        if sell_delta > 0.0 and sell_price > 0.0:
            available_buy_notional += sell_delta * sell_price

    required_buy_notional = 0.0
    for symbol, allocation_fraction in desired_allocations.items():
        if float(allocation_fraction or 0.0) <= 0.0:
            continue
        existing_qty = max(0.0, float(existing_qty_by_symbol.get(symbol, 0.0) or 0.0))
        raw_target = raw_targets.get(symbol, 0.0)
        buy_delta = max(0.0, raw_target - existing_qty)
        buy_price = float(buy_prices.get(symbol, 0.0) or 0.0)
        if buy_delta > 0.0 and buy_price > 0.0:
            required_buy_notional += buy_delta * buy_price

    buy_scale = min(1.0, available_buy_notional / required_buy_notional) if required_buy_notional > 0.0 else 1.0

    planned: dict[str, PortfolioRebalanceTarget] = {}
    for symbol, allocation_fraction in desired_allocations.items():
        if float(allocation_fraction or 0.0) <= 0.0:
            continue
        existing_qty = max(0.0, float(existing_qty_by_symbol.get(symbol, 0.0) or 0.0))
        raw_target = raw_targets.get(symbol, 0.0)
        if raw_target <= existing_qty:
            target_qty = raw_target
        else:
            target_qty = existing_qty + ((raw_target - existing_qty) * buy_scale)
        target_qty = round(max(0.0, target_qty), 4)
        planned[symbol] = PortfolioRebalanceTarget(
            symbol=symbol,
            existing_qty=existing_qty,
            target_qty=target_qty,
        )
    return planned


def effective_signal_allocation_pct(signal, *, base_allocation_pct: float) -> float:
    return resolved_signal_allocation_pct(
        signal,
        base_allocation_pct=base_allocation_pct,
        sizing_mode=DEFAULT_ALLOCATION_SIZING_MODE,
        min_open_confidence=DEFAULT_MIN_OPEN_CONFIDENCE,
    )


def _coerce_signal_allocation_fraction(
    raw_fraction: object,
    *,
    default: float | None,
) -> float | None:
    if raw_fraction is None:
        return default
    try:
        fraction = float(raw_fraction)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(fraction):
        return default
    return min(max(fraction, 0.0), 1.0)


def _signal_allocation_fraction(signal) -> float:
    fraction = _coerce_signal_allocation_fraction(
        getattr(signal, "allocation_pct", None),
        default=1.0,
    )
    return 1.0 if fraction is None else fraction


def _portfolio_signal_allocation_fraction(signal) -> float | None:
    fraction = _coerce_signal_allocation_fraction(
        getattr(signal, "allocation_pct", None),
        default=None,
    )
    if fraction is None:
        return None
    if fraction <= 0.0:
        return None
    return fraction


def _signal_confidence_fraction(
    signal,
    *,
    min_open_confidence: float,
) -> float:
    raw_confidence = getattr(signal, "confidence", None)
    if raw_confidence is None:
        return 1.0
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(confidence):
        return 1.0
    confidence_floor = min(max(float(min_open_confidence), 0.0), 1.0)
    return min(max(confidence, confidence_floor), 1.0)


def resolved_signal_allocation_pct(
    signal,
    *,
    base_allocation_pct: float,
    sizing_mode: StockAllocationSizingMode,
    min_open_confidence: float,
) -> float:
    effective_pct = max(0.0, float(base_allocation_pct))
    effective_pct *= _signal_allocation_fraction(signal)
    if sizing_mode == "confidence_scaled":
        effective_pct *= _signal_confidence_fraction(
            signal,
            min_open_confidence=min_open_confidence,
        )
    return effective_pct


def build_server_client(
    *,
    account: str,
    bot_id: str,
    paper: bool,
    base_url: str | None = None,
    session_id: str | None = None,
):
    return TradingServerClient(
        base_url=base_url,
        account=account,
        bot_id=bot_id,
        session_id=session_id,
        execution_mode="paper" if paper else "live",
    )


def _server_position_to_object(symbol: str, payload: TradingServerPositionPayload) -> ServerPositionView:
    return ServerPositionView(
        symbol=symbol,
        qty=float(payload.get("qty", 0.0) or 0.0),
        avg_entry_price=float(payload.get("avg_entry_price", 0.0) or 0.0),
        current_price=float(payload.get("current_price", 0.0) or 0.0),
    )


def server_positions_by_symbol(
    snapshot: TradingServerAccountSnapshot,
    symbols: Iterable[str],
) -> dict[str, ServerPositionView]:
    target = {str(symbol).upper() for symbol in symbols}
    positions = snapshot["positions"]
    out: dict[str, ServerPositionView] = {}
    for symbol, payload in positions.items():
        if str(symbol).upper() not in target:
            continue
        out[str(symbol).upper()] = _server_position_to_object(str(symbol).upper(), payload)
    return out


def server_portfolio_context(
    *,
    snapshot: TradingServerAccountSnapshot,
    state: StrategyState,
    quotes: dict[str, float],
    now: Optional[datetime] = None,
) -> PortfolioContext:
    now = now or datetime.now(timezone.utc)
    cash = float(snapshot["cash"] or 0.0)
    positions = snapshot["positions"]
    symbol = (state.active_symbol or "").upper()
    payload = positions.get(symbol)
    if not symbol or payload is None:
        return PortfolioContext(cash=cash)

    hold_days = 0
    opened_at = payload.get("opened_at")
    try:
        opened_date = datetime.fromisoformat(str(opened_at)).astimezone(EASTERN).date() if opened_at else None
    except Exception:
        opened_date = None
    if opened_date is not None:
        hold_days = max(0, (now.astimezone(EASTERN).date() - opened_date).days)

    entry_price = float(payload.get("avg_entry_price", 0.0) or 0.0)
    return PortfolioContext(
        cash=cash,
        current_symbol=symbol,
        position_qty=float(payload.get("qty", 0.0) or 0.0),
        entry_price=entry_price,
        hold_days=hold_days,
    )

def server_equity(snapshot: TradingServerAccountSnapshot, quotes: dict[str, float]) -> float:
    cash = float(snapshot["cash"] or 0.0)
    positions = snapshot["positions"]
    equity = cash
    for symbol, payload in positions.items():
        equity += float(payload.get("qty", 0.0) or 0.0) * float(quotes.get(str(symbol).upper(), 0.0) or 0.0)
    return equity


def compute_target_qty_from_server_snapshot(
    *,
    snapshot: TradingServerAccountSnapshot,
    quotes: dict[str, float],
    price: float,
    allocation_pct: float,
) -> float:
    return compute_target_qty_from_values(
        portfolio_value=server_equity(snapshot, quotes),
        buying_power=float(snapshot.get("buying_power", snapshot["cash"]) or 0.0),
        price=price,
        allocation_pct=allocation_pct,
    )


def _marketable_limit_price(price: float, side: str, *, buffer_bps: float = SERVER_MARKETABLE_LIMIT_BUFFER_BPS) -> float:
    side = str(side).strip().lower()
    scale = 1.0 + (float(buffer_bps) / 10_000.0)
    if side == "buy":
        return round(float(price) * scale, 4)
    return round(float(price) / scale, 4)


def _is_server_loss_guard_rejection(exc: Exception) -> bool:
    text = str(exc)
    return "below safety floor" in text and "allow_loss_exit=true" in text


def _submit_server_limit_order_with_loss_guard(
    server_client: TradingServerClientLike,
    *,
    symbol: str,
    qty: float,
    side: str,
    limit_price: float,
    metadata: dict[str, object],
) -> dict[str, object] | None:
    try:
        return server_client.submit_limit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            limit_price=limit_price,
            metadata=metadata,
        )
    except Exception as exc:
        if side == "sell" and _is_server_loss_guard_rejection(exc):
            logger.warning(
                "Server rejected ordinary sell for %s at %.4f due to loss guard; keeping position: %s",
                symbol,
                limit_price,
                exc,
            )
            return None
        raise


def execute_signal_with_trading_server(
    signal,
    *,
    server_client: TradingServerClientLike,
    quotes: dict[str, float],
    state: StrategyState,
    symbols: Iterable[str],
    allocation_pct: float,
    dry_run: bool,
    allocation_sizing_mode: StockAllocationSizingMode = DEFAULT_ALLOCATION_SIZING_MODE,
    now: Optional[datetime] = None,
    allow_open: bool = True,
    allow_open_reason: str | None = None,
    min_open_confidence: float = DEFAULT_MIN_OPEN_CONFIDENCE,
) -> bool:
    now = now or datetime.now(timezone.utc)
    symbol_set = [str(symbol).upper() for symbol in symbols]
    snapshot = server_client.get_account()
    live_positions = server_positions_by_symbol(snapshot, symbol_set)
    managed_symbol = state.active_symbol.upper() if state.active_symbol else None
    desired_symbol = signal.symbol.upper() if signal.symbol and signal.direction == "long" else None

    unmanaged = sorted(symbol for symbol in live_positions if symbol != managed_symbol)
    if unmanaged:
        logger.warning(
            "Found unmanaged server position(s) in the strategy universe: %s. Refusing to place new orders.",
            ", ".join(unmanaged),
        )
        return False

    managed_position = live_positions.get(managed_symbol) if managed_symbol else None
    if desired_symbol and managed_symbol == desired_symbol and managed_position is not None:
        logger.info("Holding existing managed server position in %s", desired_symbol)
        return False

    if managed_position is not None and managed_symbol is not None:
        qty = abs(_signed_position_qty(managed_position))
        logger.info("Closing managed server position: %s qty=%.4f", managed_symbol, qty)
        if not dry_run:
            server_client.refresh_prices(symbols=[managed_symbol])
            order = _submit_server_limit_order_with_loss_guard(
                server_client,
                symbol=managed_symbol,
                qty=qty,
                side="sell",
                limit_price=float(quotes[managed_symbol]),
                metadata={"strategy": "daily_stock_rl", "intent": "close_managed"},
            )
            if order is None:
                return False
            state.last_order_id = str(order.get("order", {}).get("id", ""))
            state.pending_close_symbol = managed_symbol
            state.pending_close_order_id = state.last_order_id
            state.active_symbol = None
            state.active_qty = 0.0
            state.entry_price = 0.0
            state.entry_date = None
            snapshot = server_client.get_account()

    if desired_symbol is None:
        logger.info("Signal is flat; no new server position opened")
        return managed_position is not None

    if not allow_open:
        suffix = f" ({allow_open_reason})" if allow_open_reason else ""
        logger.warning("Skipping new server position open for %s because execution safety gate is active%s", desired_symbol, suffix)
        return managed_position is not None

    price = float(quotes.get(desired_symbol, 0.0) or 0.0)
    effective_allocation_pct = resolved_signal_allocation_pct(
        signal,
        base_allocation_pct=allocation_pct,
        sizing_mode=allocation_sizing_mode,
        min_open_confidence=min_open_confidence,
    )
    qty = compute_target_qty_from_server_snapshot(
        snapshot=snapshot,
        quotes=quotes,
        price=price,
        allocation_pct=effective_allocation_pct,
    )
    if qty <= 0:
        logger.warning("Computed zero-sized server order for %s at %.4f", desired_symbol, price)
        return False

    logger.info(
        "Opening managed server position: %s qty=%.4f @ %.4f (alloc=%.1f%%, signal_frac=%.2f)",
        desired_symbol,
        qty,
        price,
        effective_allocation_pct,
        float(getattr(signal, "allocation_pct", 1.0) or 0.0),
    )
    if not dry_run:
        server_client.refresh_prices(symbols=[desired_symbol])
        order = server_client.submit_limit_order(
            symbol=desired_symbol,
            qty=qty,
            side="buy",
            limit_price=_marketable_limit_price(price, "buy"),
            metadata={"strategy": "daily_stock_rl", "intent": "open_managed"},
        )
        state.last_order_id = str(order.get("order", {}).get("id", ""))
        state.active_symbol = desired_symbol
        state.active_qty = qty
        state.entry_price = price
        state.entry_date = now.astimezone(EASTERN).date().isoformat()
    return True


def execute_signal(
    signal,
    *,
    client,
    paper: bool = False,
    quotes: dict[str, float],
    state: StrategyState,
    symbols: Iterable[str],
    allocation_pct: float,
    dry_run: bool,
    allocation_sizing_mode: StockAllocationSizingMode = DEFAULT_ALLOCATION_SIZING_MODE,
    now: Optional[datetime] = None,
    allow_open: bool = True,
    allow_open_reason: str | None = None,
    min_open_confidence: float = DEFAULT_MIN_OPEN_CONFIDENCE,
) -> bool:
    now = now or datetime.now(timezone.utc)
    symbol_set = [str(symbol).upper() for symbol in symbols]
    live_positions = positions_by_symbol(client, symbol_set)
    managed_symbol = state.active_symbol.upper() if state.active_symbol else None
    desired_symbol = signal.symbol.upper() if signal.symbol and signal.direction == "long" else None

    unmanaged = sorted(symbol for symbol in live_positions if symbol != managed_symbol)
    if unmanaged:
        logger.warning(
            "Found unmanaged position(s) in the strategy universe: %s. Refusing to place new orders.",
            ", ".join(unmanaged),
        )
        return False

    managed_position = live_positions.get(managed_symbol) if managed_symbol else None
    if desired_symbol and managed_symbol == desired_symbol and managed_position is not None:
        logger.info("Holding existing managed position in %s", desired_symbol)
        return False

    if managed_position is not None and managed_symbol is not None:
        qty = _signed_position_qty(managed_position)
        close_side = _market_order_side_for_qty(qty)
        sell_price = float(quotes.get(managed_symbol, 0.0) or 0.0)
        limit_sell_price = (
            sell_price
            if paper
            else sell_price * (1.0 + CALIBRATED_EXIT_OFFSET_BPS / 10_000.0)
        ) if sell_price > 0 else 0
        logger.info(
            "Closing managed position: %s qty=%.4f side=%s limit=%.2f (%s)",
            managed_symbol,
            abs(qty),
            close_side,
            limit_sell_price,
            "paper_midpoint" if paper else f"exit_offset=+{CALIBRATED_EXIT_OFFSET_BPS:.0f}bps",
        )
        if not dry_run:
            if limit_sell_price > 0:
                order = submit_limit_order(
                    client, symbol=managed_symbol, qty=abs(qty),
                    side=close_side, limit_price=limit_sell_price,
                )
            elif paper:
                logger.warning(
                    "Skipping paper close for %s because no defensible limit price was available",
                    managed_symbol,
                )
                return False
            else:
                order = submit_market_order(client, symbol=managed_symbol, qty=abs(qty), side=close_side)
            state.last_order_id = str(getattr(order, "id", ""))
            state.pending_close_symbol = managed_symbol
            state.pending_close_order_id = state.last_order_id
            state.active_symbol = None
            state.active_qty = 0.0
            state.entry_price = 0.0
            state.entry_date = None

    if desired_symbol is None:
        logger.info("Signal is flat; no new position opened")
        return managed_position is not None

    if state.pending_close_symbol:
        logger.warning(
            "Pending close for %s still outstanding (order %s); skipping new open",
            state.pending_close_symbol, state.pending_close_order_id,
        )
        return managed_position is not None

    if not allow_open:
        suffix = f" ({allow_open_reason})" if allow_open_reason else ""
        logger.warning("Skipping new position open for %s because execution safety gate is active%s", desired_symbol, suffix)
        return managed_position is not None

    account = client.get_account()
    price = float(quotes.get(desired_symbol, 0.0) or 0.0)
    limit_buy_price = (
        price
        if paper
        else price * (1.0 + CALIBRATED_ENTRY_OFFSET_BPS / 10_000.0)
    ) if price > 0 else 0
    effective_allocation_pct = resolved_signal_allocation_pct(
        signal,
        base_allocation_pct=allocation_pct,
        sizing_mode=allocation_sizing_mode,
        min_open_confidence=min_open_confidence,
    )
    qty = compute_target_qty(
        account=account,
        price=limit_buy_price or price,
        allocation_pct=effective_allocation_pct,
    )
    if qty <= 0:
        logger.warning("Computed zero-sized order for %s at %.4f", desired_symbol, price)
        return False

    logger.info(
        "Opening managed position: %s qty=%.4f @ %.4f limit=%.2f (alloc=%.1f%%, signal_frac=%.2f, %s)",
        desired_symbol,
        qty,
        price,
        limit_buy_price,
        effective_allocation_pct,
        float(getattr(signal, "allocation_pct", 1.0) or 0.0),
        "paper_midpoint" if paper else f"entry_offset=+{CALIBRATED_ENTRY_OFFSET_BPS:.0f}bps",
    )
    if not dry_run:
        if limit_buy_price > 0:
            order = submit_limit_order(client, symbol=desired_symbol, qty=qty, side="buy", limit_price=limit_buy_price)
        elif paper:
            logger.warning(
                "Skipping paper open for %s because no defensible limit price was available",
                desired_symbol,
            )
            return False
        else:
            order = submit_market_order(client, symbol=desired_symbol, qty=qty, side="buy")
        state.last_order_id = str(getattr(order, "id", ""))
        state.active_symbol = desired_symbol
        state.active_qty = qty
        state.entry_price = price
        state.entry_date = now.astimezone(EASTERN).date().isoformat()
    return True


def build_portfolio_context(
    *,
    state: StrategyState,
    live_positions: dict[str, object],
    account,
    now: Optional[datetime] = None,
) -> PortfolioContext:
    now = now or datetime.now(timezone.utc)
    cash = float(
        getattr(account, "cash", 0.0)
        or getattr(account, "buying_power", 0.0)
        or getattr(account, "portfolio_value", DEFAULT_BACKTEST_STARTING_CASH)
        or DEFAULT_BACKTEST_STARTING_CASH
    )
    if not state.active_symbol:
        return PortfolioContext(cash=cash)

    symbol = state.active_symbol.upper()
    position = live_positions.get(symbol)
    if position is None:
        return PortfolioContext(cash=cash)

    hold_days = 0
    if state.entry_date:
        try:
            entry_day = datetime.fromisoformat(state.entry_date).date()
            hold_days = max(0, (now.astimezone(EASTERN).date() - entry_day).days)
        except ValueError:
            hold_days = 0

    entry_price = float(state.entry_price or getattr(position, "avg_entry_price", 0.0) or 0.0)
    return PortfolioContext(
        cash=cash,
        current_symbol=symbol,
        position_qty=abs(_signed_position_qty(position)),
        entry_price=entry_price,
        hold_days=hold_days,
    )


def execute_multi_position_signals(
    signals: list,
    *,
    client,
    paper: bool = False,
    quotes: dict[str, float],
    symbols: Iterable[str],
    total_allocation_pct: float,
    dry_run: bool,
    now: Optional[datetime] = None,
) -> dict[str, float]:
    """Execute a portfolio of top-K signals, managing multiple positions.

    Returns dict of symbol -> qty for currently held positions.
    """
    now = now or datetime.now(timezone.utc)
    symbol_set = [str(s).upper() for s in symbols]
    live_positions = positions_by_symbol(client, symbol_set)
    existing_qty_by_symbol = {
        sym: abs(_signed_position_qty(pos))
        for sym, pos in live_positions.items()
    }

    # Desired portfolio: symbol -> target_allocation_fraction
    desired: dict[str, float] = {}
    for sig in signals:
        if sig.symbol and sig.direction == "long":
            sym = sig.symbol.upper()
            alloc_frac = _portfolio_signal_allocation_fraction(sig)
            if alloc_frac is None:
                logger.warning(
                    "Skipping malformed portfolio signal allocation for %s: %r",
                    sym,
                    getattr(sig, "allocation_pct", None),
                )
                continue
            desired[sym] = alloc_frac

    # Get account for buying power
    account = client.get_account()
    portfolio_value = account_portfolio_value(account)
    buying_power = float(getattr(account, "buying_power", 0) or 0)
    buy_prices = {
        sym: (
            float(quotes.get(sym, 0.0) or 0.0)
            if paper
            else float(quotes.get(sym, 0.0) or 0.0) * (1.0 + CALIBRATED_ENTRY_OFFSET_BPS / 10_000.0)
        )
        for sym in set(existing_qty_by_symbol) | set(desired)
    }
    sell_prices = {
        sym: (
            float(quotes.get(sym, 0.0) or 0.0)
            if paper
            else float(quotes.get(sym, 0.0) or 0.0) * (1.0 + CALIBRATED_EXIT_OFFSET_BPS / 10_000.0)
        )
        for sym in set(existing_qty_by_symbol) | set(desired)
    }
    target_plan = _portfolio_rebalance_targets(
        desired_allocations=desired,
        existing_qty_by_symbol=existing_qty_by_symbol,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        portfolio_value=portfolio_value,
        buying_power=buying_power,
        total_allocation_pct=total_allocation_pct,
    )

    orders_placed = 0
    held = {}
    for sym, existing_qty in existing_qty_by_symbol.items():
        if sym in target_plan:
            continue
        sell_price = sell_prices.get(sym, 0.0)
        if existing_qty <= 0.0 or sell_price <= 0.0:
            continue
        logger.info("Multi-pos: closing %s qty=%.4f limit=%.2f", sym, existing_qty, sell_price)
        if not dry_run:
            submit_limit_order(client, symbol=sym, qty=existing_qty, side="sell", limit_price=sell_price)
            orders_placed += 1

    for sym, alloc_frac in desired.items():
        plan = target_plan[sym]
        existing_qty = plan.existing_qty
        target_qty = plan.target_qty
        buy_price = buy_prices.get(sym, 0.0)
        sell_price = sell_prices.get(sym, 0.0)
        delta_qty = round(target_qty - existing_qty, 4)
        if target_qty <= 0.0:
            continue
        held[sym] = target_qty
        if abs(delta_qty) <= _PORTFOLIO_REBALANCE_QTY_EPSILON:
            logger.info("Multi-pos: holding %s qty=%.4f (target=%.4f)", sym, existing_qty, target_qty)
            continue
        if delta_qty < 0.0:
            logger.info(
                "Multi-pos: trimming %s qty=%.4f -> %.4f limit=%.2f",
                sym,
                existing_qty,
                target_qty,
                sell_price,
            )
            if not dry_run and sell_price > 0.0:
                submit_limit_order(client, symbol=sym, qty=abs(delta_qty), side="sell", limit_price=sell_price)
                orders_placed += 1
            continue
        logger.info(
            "Multi-pos: buying %s qty=%.4f -> %.4f limit=%.2f (alloc=%.1f%%)",
            sym,
            existing_qty,
            target_qty,
            buy_price,
            alloc_frac * total_allocation_pct,
        )
        if not dry_run and buy_price > 0.0:
            submit_limit_order(client, symbol=sym, qty=delta_qty, side="buy", limit_price=buy_price)
            orders_placed += 1

    logger.info("Multi-pos: %d orders placed, %d positions targeted", orders_placed, len(held))
    return held


def execute_multi_position_signals_with_trading_server(
    signals: list,
    *,
    server_client: TradingServerClientLike,
    quotes: dict[str, float],
    symbols: Iterable[str],
    total_allocation_pct: float,
    dry_run: bool,
) -> dict[str, float]:
    """Execute a portfolio of top-K signals via the trading server."""
    symbol_set = [str(symbol).upper() for symbol in symbols]
    snapshot = server_client.get_account()
    live_positions = server_positions_by_symbol(snapshot, symbol_set)
    existing_qty_by_symbol = {
        symbol: abs(_signed_position_qty(position))
        for symbol, position in live_positions.items()
    }

    desired: dict[str, float] = {}
    for sig in signals:
        if sig.symbol and sig.direction == "long":
            symbol = str(sig.symbol).upper()
            alloc_frac = _portfolio_signal_allocation_fraction(sig)
            if alloc_frac is None:
                logger.warning(
                    "Skipping malformed server portfolio signal allocation for %s: %r",
                    symbol,
                    getattr(sig, "allocation_pct", None),
                )
                continue
            desired[symbol] = alloc_frac

    equity = server_equity(snapshot, quotes)
    buy_prices = {
        symbol: float(quotes.get(symbol, 0.0) or 0.0)
        for symbol in set(existing_qty_by_symbol) | set(desired)
    }
    sell_prices = dict(buy_prices)
    target_plan = _portfolio_rebalance_targets(
        desired_allocations=desired,
        existing_qty_by_symbol=existing_qty_by_symbol,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        portfolio_value=equity,
        buying_power=float(snapshot.get("buying_power", snapshot["cash"]) or 0.0),
        total_allocation_pct=total_allocation_pct,
    )
    sell_orders: list[tuple[str, float, float, str]] = []
    buy_orders: list[tuple[str, float, float, str]] = []
    orders_placed = 0
    held: dict[str, float] = {}
    actual_held: dict[str, float] = dict(existing_qty_by_symbol)
    for sym, existing_qty in existing_qty_by_symbol.items():
        if sym in target_plan:
            continue
        sell_price = sell_prices.get(sym, 0.0)
        if existing_qty <= 0.0 or sell_price <= 0.0:
            continue
        logger.info("Server multi-pos: closing %s qty=%.4f", sym, existing_qty)
        if not dry_run:
            sell_orders.append((sym, existing_qty, sell_price, "close_portfolio_position"))

    for sym, alloc_frac in desired.items():
        plan = target_plan[sym]
        existing_qty = plan.existing_qty
        target_qty = plan.target_qty
        price = buy_prices.get(sym, 0.0)
        if target_qty <= 0.0:
            continue
        held[sym] = target_qty
        delta_qty = round(target_qty - existing_qty, 4)
        if abs(delta_qty) <= _PORTFOLIO_REBALANCE_QTY_EPSILON:
            logger.info(
                "Server multi-pos: holding %s qty=%.4f (target=%.4f)",
                sym,
                existing_qty,
                target_qty,
            )
            continue
        if delta_qty < 0.0:
            logger.info("Server multi-pos: trimming %s qty=%.4f -> %.4f", sym, existing_qty, target_qty)
            if not dry_run and price > 0.0:
                sell_orders.append((sym, abs(delta_qty), price, "rebalance_portfolio_position"))
            continue
        logger.info(
            "Server multi-pos: buying %s qty=%.4f -> %.4f @ %.4f (alloc=%.1f%%)",
            sym,
            existing_qty,
            target_qty,
            price,
            float(alloc_frac) * float(total_allocation_pct),
        )
        if not dry_run and price > 0.0:
            buy_orders.append((sym, delta_qty, price, "open_portfolio_position"))
    if not dry_run and (sell_orders or buy_orders):
        refresh_symbols = sorted({symbol for symbol, *_rest in sell_orders + buy_orders})
        server_client.refresh_prices(symbols=refresh_symbols)
        blocked_sell_symbols: list[str] = []
        for sym, qty, sell_price, intent in sell_orders:
            order = _submit_server_limit_order_with_loss_guard(
                server_client,
                symbol=sym,
                qty=qty,
                side="sell",
                limit_price=sell_price,
                metadata={"strategy": "daily_stock_rl", "intent": intent},
            )
            if order is None:
                blocked_sell_symbols.append(sym)
                actual_held[sym] = existing_qty_by_symbol.get(sym, 0.0)
                continue
            remaining_qty = max(0.0, actual_held.get(sym, 0.0) - qty)
            if remaining_qty <= _PORTFOLIO_REBALANCE_QTY_EPSILON:
                actual_held.pop(sym, None)
            else:
                actual_held[sym] = remaining_qty
            orders_placed += 1
        if blocked_sell_symbols:
            logger.warning(
                "Server multi-pos: skipped %d buy order(s) because loss guard kept existing positions in %s",
                len(buy_orders),
                ", ".join(sorted(blocked_sell_symbols)),
            )
            return {sym: qty for sym, qty in actual_held.items() if qty > _PORTFOLIO_REBALANCE_QTY_EPSILON}
        for sym, qty, price, intent in buy_orders:
            server_client.submit_limit_order(
                symbol=sym,
                qty=qty,
                side="buy",
                limit_price=_marketable_limit_price(price, "buy"),
                metadata={"strategy": "daily_stock_rl", "intent": intent},
            )
            actual_held[sym] = actual_held.get(sym, 0.0) + qty
            orders_placed += 1

    logger.info(
        "Server multi-pos: %d orders placed, %d positions targeted",
        orders_placed,
        len(held),
    )
    if dry_run:
        return held
    return {sym: qty for sym, qty in actual_held.items() if qty > _PORTFOLIO_REBALANCE_QTY_EPSILON}


def reconcile_pending_close(
    *,
    state: StrategyState,
    live_positions: dict[str, object],
) -> None:
    """Clear pending_close_symbol once the close order has actually filled."""
    if not state.pending_close_symbol:
        return
    sym = state.pending_close_symbol.upper()
    if sym not in live_positions:
        logger.info(
            "Pending close for %s confirmed filled (order %s); clearing pending state",
            sym, state.pending_close_order_id,
        )
        state.pending_close_symbol = None
        state.pending_close_order_id = None
    else:
        logger.warning(
            "Pending close for %s NOT yet filled (order %s); position still open on broker",
            sym, state.pending_close_order_id,
        )
        # Re-adopt the position so we don't lose track of it
        position = live_positions[sym]
        qty = abs(_signed_position_qty(position))
        if qty > 0 and not state.active_symbol:
            state.active_symbol = sym
            state.active_qty = qty
            state.entry_price = float(getattr(position, "avg_entry_price", 0.0) or 0.0)
            state.pending_close_symbol = None
            state.pending_close_order_id = None
            logger.warning("Re-adopted unfilled close position %s back into active state", sym)


def adopt_existing_position(
    *,
    state: StrategyState,
    live_positions: dict[str, object],
    now: Optional[datetime] = None,
) -> bool:
    now = now or datetime.now(timezone.utc)
    if state.active_symbol or len(live_positions) != 1:
        return False
    symbol, position = next(iter(live_positions.items()))
    qty = abs(_signed_position_qty(position))
    if qty <= 0:
        return False
    state.active_symbol = symbol
    state.active_qty = qty
    state.entry_price = float(getattr(position, "avg_entry_price", 0.0) or 0.0)
    state.entry_date = now.astimezone(EASTERN).date().isoformat()
    logger.warning("Adopting pre-existing %s position into daily stock RL state", symbol)
    return True


def should_run_today(*, now: datetime, is_market_open: bool, last_run_date: Optional[str]) -> bool:
    if not is_market_open:
        return False
    now_et = now.astimezone(EASTERN)
    if now_et.weekday() >= 5:
        return False
    if last_run_date == now_et.date().isoformat():
        return False
    return now_et.timetz().replace(tzinfo=None) >= RUN_AFTER_OPEN_ET


def seconds_until_next_check(*, now: datetime, is_market_open: bool, next_open: Optional[datetime]) -> float:
    now_et = now.astimezone(EASTERN)
    if is_market_open and now_et.timetz().replace(tzinfo=None) < RUN_AFTER_OPEN_ET:
        target_et = datetime.combine(now_et.date(), RUN_AFTER_OPEN_ET, tzinfo=EASTERN)
        return max(30.0, (target_et - now_et).total_seconds())
    if next_open is not None:
        target_ts = pd.Timestamp(next_open)
        if target_ts.tzinfo is None:
            target_ts = target_ts.tz_localize("UTC")
        else:
            target_ts = target_ts.tz_convert("UTC")
        target = target_ts.to_pydatetime()
        target += timedelta(minutes=5)
        return max(60.0, (target - now).total_seconds())
    return 300.0 if is_market_open else 900.0


def _signal_payload(signal, *, checkpoint: str, quotes: dict[str, float], now: datetime, run_id: str) -> SignalPayload:
    return {
        "run_id": run_id,
        "timestamp": now.isoformat(),
        "checkpoint": checkpoint,
        "action": signal.action,
        "symbol": signal.symbol,
        "direction": signal.direction,
        "confidence": float(signal.confidence),
        "value_estimate": float(signal.value_estimate),
        "allocation_fraction": float(getattr(signal, "allocation_pct", 1.0) or 0.0),
        "quotes": {symbol: float(price) for symbol, price in quotes.items()},
    }


def _portfolio_signal_payload(signal) -> PortfolioSignalPayload:
    return {
        "action": signal.action,
        "symbol": signal.symbol,
        "direction": signal.direction,
        "confidence": float(signal.confidence),
        "value_estimate": float(signal.value_estimate),
        "allocation_fraction": float(getattr(signal, "allocation_pct", 1.0) or 0.0),
    }


def _execution_observability_fields(
    *,
    data_source: StockDataSource,
    dry_run: bool,
    market_open: bool | None,
    bars_fresh: bool,
    signal,
    allow_open: bool,
    executed: bool,
) -> ExecutionObservabilityFields:
    execution_submitted = bool(data_source == "alpaca" and not dry_run and executed)
    execution_would_submit = bool(data_source == "alpaca" and dry_run and executed)
    skip_reason: StockExecutionSkipReason | None = None

    if data_source != "alpaca":
        status: StockExecutionStatus = "local_only"
        skip_reason = "local_data_mode"
    elif execution_submitted:
        status = "submitted"
    elif execution_would_submit:
        status = "dry_run_would_execute"
    elif not dry_run and market_open is False:
        status = "skipped_market_closed"
        skip_reason = "market_closed"
    elif not dry_run and not bars_fresh:
        status = "skipped_stale_bars"
        skip_reason = "stale_bars"
    elif signal.symbol is None or signal.direction != "long":
        status = "no_action_flat_signal"
        skip_reason = "flat_signal"
    elif not allow_open:
        status = "blocked_open_gate"
        skip_reason = "open_gate"
    else:
        status = "no_action_executor_declined"
        skip_reason = "executor_declined"

    return {
        "execution_submitted": execution_submitted,
        "execution_would_submit": execution_would_submit,
        "execution_status": status,
        "execution_skip_reason": skip_reason,
    }


def _run_summary_payload(payload: dict[str, object]) -> RunSummaryPayload:
    return {
        "event": "daily_stock_run_once",
        "run_id": payload.get("run_id"),
        "timestamp": payload.get("timestamp"),
        "checkpoint": payload.get("checkpoint"),
        "action": payload.get("action"),
        "symbol": payload.get("symbol"),
        "direction": payload.get("direction"),
        "confidence": payload.get("confidence"),
        "value_estimate": payload.get("value_estimate"),
        "bar_data_source": payload.get("bar_data_source"),
        "quote_data_source": payload.get("quote_data_source"),
        "latest_bar_timestamp": payload.get("latest_bar_timestamp"),
        "bars_fresh": payload.get("bars_fresh"),
        "market_open": payload.get("market_open"),
        "dry_run": payload.get("dry_run"),
        "execution_backend": payload.get("execution_backend"),
        "allow_open": payload.get("allow_open"),
        "allow_open_reason": payload.get("allow_open_reason"),
        "execution_status": payload.get("execution_status"),
        "execution_skip_reason": payload.get("execution_skip_reason"),
        "execution_submitted": payload.get("execution_submitted"),
        "execution_would_submit": payload.get("execution_would_submit"),
        "state_advanced": payload.get("state_advanced"),
        "signal_log_written": payload.get("signal_log_written"),
        "signal_log_write_error": payload.get("signal_log_write_error"),
        "run_event_log_written": payload.get("run_event_log_written"),
        "run_event_log_write_error": payload.get("run_event_log_write_error"),
    }


def _log_run_summary(payload: dict[str, object]) -> None:
    logger.info("Run summary: %s", json.dumps(_run_summary_payload(payload), sort_keys=True))


def _compact_run_failure_observability(
    observability: dict[str, object] | None,
) -> dict[str, object]:
    if not observability:
        return {}
    compact: dict[str, object] = {}
    for key, value in observability.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, tuple, dict)) and not value:
            continue
        compact[key] = value
    return compact


def _format_run_failure_context_note(observability: dict[str, object] | None) -> str | None:
    compact = _compact_run_failure_observability(observability)
    if not compact:
        return None

    parts: list[str] = []
    for key in (
        "state_active_symbol",
        "live_position_symbols",
        "bar_data_source",
        "quote_data_source",
        "quote_fallback_symbols",
        "latest_bar_timestamp",
        "market_open",
        "signal_action",
        "signal_symbol",
        "signal_confidence",
        "allow_open",
        "allow_open_reason",
    ):
        if key not in compact:
            continue
        parts.append(f"{key}={compact[key]}")
    if not parts:
        return None
    return "run_once context: " + "; ".join(parts)


def _run_failure_payload(
    *,
    run_id: str,
    now: datetime,
    checkpoint: str,
    stage: str,
    exc: Exception,
    data_source: StockDataSource,
    execution_backend: StockExecutionBackend,
    paper: bool,
    dry_run: bool,
    state_path: Path,
    symbols: Sequence[str],
    data_dir: str | None,
    server_account: str | None,
    server_bot_id: str | None,
    server_url: str | None,
    observability: dict[str, object] | None = None,
) -> RunFailurePayload:
    requested_local_data_dir: str | None = None
    resolved_local_data_dir: str | None = None
    resolved_local_data_dir_source: LocalDataDirResolutionSource | None = None
    if data_source == "local" and data_dir is not None:
        local_data_dir_context = _local_data_dir_context(data_dir, symbols)
        requested_local_data_dir = local_data_dir_context["requested_local_data_dir"]
        resolved_local_data_dir = local_data_dir_context["resolved_local_data_dir"]
        resolved_local_data_dir_source = local_data_dir_context["resolved_local_data_dir_source"]
    return {
        "event": "daily_stock_run_once_failed",
        "run_id": run_id,
        "timestamp": now.isoformat(),
        "checkpoint": checkpoint,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "data_source": data_source,
        "execution_backend": execution_backend,
        "account_mode": "paper" if paper else "live",
        "dry_run": bool(dry_run),
        "state_path": str(state_path),
        "symbols": [str(symbol).upper() for symbol in symbols],
        "requested_local_data_dir": requested_local_data_dir,
        "resolved_local_data_dir": resolved_local_data_dir,
        "resolved_local_data_dir_source": resolved_local_data_dir_source,
        "server_account": str(server_account).strip() or None,
        "server_bot_id": str(server_bot_id).strip() or None,
        "server_url": (str(server_url).strip().rstrip("/") or None) if server_url is not None else None,
        "observability": _compact_run_failure_observability(observability),
    }


def _log_run_failure(payload: RunFailurePayload) -> None:
    logger.error("Run failure: %s", json.dumps(payload, sort_keys=True))


def _daemon_warning_payload(
    *,
    now: datetime,
    checkpoint: str,
    stage: str,
    exc: Exception,
    paper: bool,
    execution_backend: str,
    sleep_seconds: float | None = None,
    state_path: Path | None = None,
    retry_with_paper_clock: bool | None = None,
    clock_source: str | None = None,
    server_session_id: str | None = None,
) -> DaemonWarningPayload:
    payload: DaemonWarningPayload = {
        "event": "daily_stock_daemon_warning",
        "timestamp": now.isoformat(),
        "checkpoint": checkpoint,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "account_mode": "paper" if paper else "live",
        "execution_backend": execution_backend,
    }
    if sleep_seconds is not None:
        payload["sleep_seconds"] = float(sleep_seconds)
    if state_path is not None:
        payload["state_path"] = str(state_path)
    if retry_with_paper_clock is not None:
        payload["retry_with_paper_clock"] = bool(retry_with_paper_clock)
    if clock_source is not None:
        payload["clock_source"] = clock_source
    if server_session_id:
        payload["server_session_id"] = server_session_id
    return payload


def _log_daemon_warning(payload: DaemonWarningPayload) -> None:
    logger.warning("Daemon warning: %s", json.dumps(payload, sort_keys=True))


def _log_daemon_warning_and_sleep(
    *,
    checkpoint: str,
    stage: str,
    exc: Exception,
    paper: bool,
    execution_backend: str,
    sleep_seconds: float,
    state_path: Path | None = None,
    retry_with_paper_clock: bool | None = None,
    clock_source: str | None = None,
    server_session_id: str | None = None,
) -> None:
    _log_daemon_warning(
        _daemon_warning_payload(
            now=datetime.now(timezone.utc),
            checkpoint=checkpoint,
            stage=stage,
            exc=exc,
            paper=paper,
            execution_backend=execution_backend,
            sleep_seconds=sleep_seconds,
            state_path=state_path,
            retry_with_paper_clock=retry_with_paper_clock,
            clock_source=clock_source,
            server_session_id=server_session_id,
        )
    )
    time.sleep(sleep_seconds)


def run_backtest(
    *,
    checkpoint: str,
    symbols: Iterable[str],
    data_dir: str,
    days: int,
    allocation_pct: float = 100.0,
    allocation_sizing_mode: StockAllocationSizingMode = DEFAULT_ALLOCATION_SIZING_MODE,
    starting_cash: float = DEFAULT_BACKTEST_STARTING_CASH,
    multi_position: int = DEFAULT_MULTI_POSITION,
    multi_position_min_prob_ratio: float = DEFAULT_MULTI_POSITION_MIN_PROB_RATIO,
    extra_checkpoints: Optional[list[str]] = None,
    buying_power_multiplier: float = DEFAULT_BACKTEST_BUYING_POWER_MULTIPLIER,
    allow_unsafe_checkpoint_loading: bool = False,
    entry_offset_bps: float = 0.0,
    exit_offset_bps: float = 0.0,
    min_open_confidence: float = DEFAULT_MIN_OPEN_CONFIDENCE,
    min_open_value_estimate: float = DEFAULT_MIN_OPEN_VALUE_ESTIMATE,
) -> dict[str, float]:
    if starting_cash <= 0:
        raise ValueError("starting_cash must be positive")
    if buying_power_multiplier <= 0:
        raise ValueError("buying_power_multiplier must be positive")
    min_open_confidence = float(min_open_confidence)
    min_open_value_estimate = float(min_open_value_estimate)

    def _signal_passes_open_gate(sig) -> bool:
        confidence = float(getattr(sig, "confidence", 0.0) or 0.0)
        if confidence < min_open_confidence:
            return False
        value_estimate = float(getattr(sig, "value_estimate", 0.0) or 0.0)
        if value_estimate < min_open_value_estimate:
            return False
        return True

    gate_blocked_opens = 0
    feature_schema = resolve_daily_feature_schema(
        checkpoint,
        extra_checkpoints=extra_checkpoints,
    )
    frames = load_local_daily_frames(
        symbols,
        data_dir=data_dir,
        min_days=days + DEFAULT_DAILY_FRAME_MIN_DAYS,
    )
    indexed = {
        symbol: frame.set_index("timestamp")[["open", "high", "low", "close", "volume"]].copy()
        for symbol, frame in frames.items()
    }
    min_len = min(len(frame) for frame in indexed.values())
    start = min_len - days
    if start < 1:
        raise ValueError(f"Need at least {days + 1} aligned days for backtest")

    trader = _load_cached_daily_trader(
        checkpoint,
        device="cpu",
        long_only=True,
        symbols=list(indexed.keys()),
        allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
    )
    extra_policies = [
        _load_bare_policy(
            str((REPO / path).resolve()) if not Path(path).is_absolute() else path,
            trader.obs_size,
            trader.num_actions,
            "cpu",
            allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
        )
        for path in (extra_checkpoints or [])
    ]
    feature_cube = _build_daily_feature_cube(
        indexed,
        symbols=trader.SYMBOLS,
        feature_schema=feature_schema,
    )

    cash = float(starting_cash)
    position: Optional[tuple[str, float, float]] = None
    portfolio_positions: dict[str, tuple[float, float]] = {}
    equity_curve: list[float] = []
    trades = 0

    for idx in range(start, min_len):
        prices = {symbol: float(frame["close"].iloc[idx]) for symbol, frame in indexed.items()}
        if multi_position > 1:
            equity = cash + sum(qty * prices[symbol] for symbol, (qty, _entry_price) in portfolio_positions.items())
            trader.cash = equity
            trader.current_position = None
            trader.position_qty = 0.0
            trader.entry_price = 0.0
        else:
            trader.cash = cash
            trader.current_position = None
            trader.position_qty = 0.0
            trader.entry_price = 0.0
            if position is not None:
                pos_symbol, qty, entry_price = position
                trader.current_position = trader.SYMBOLS.index(pos_symbol)
                trader.position_qty = qty
                trader.entry_price = entry_price
            equity = cash
            if position is not None:
                pos_symbol, qty, _ = position
                equity += qty * prices[pos_symbol]

        features = feature_cube[idx]
        if extra_policies or multi_position > 1:
            if multi_position > 1:
                signals = _ensemble_top_k_signals(
                    trader,
                    extra_policies,
                    features,
                    prices,
                    k=multi_position,
                    min_prob_ratio=multi_position_min_prob_ratio,
                )
            else:
                signal = _ensemble_softmax_signal(trader, extra_policies, features, prices)
        else:
            signal = _trader_signal_from_features(
                trader,
                features=features,
                prices=prices,
                indexed=indexed,
                idx=idx,
            )
        equity_curve.append(equity)

        if multi_position > 1:
            desired: dict[str, float] = {}
            for sig in signals:
                if not sig.symbol or sig.direction != "long":
                    continue
                if not _signal_passes_open_gate(sig):
                    gate_blocked_opens += 1
                    continue
                alloc_frac = _portfolio_signal_allocation_fraction(sig)
                if alloc_frac is None:
                    continue
                desired[str(sig.symbol).upper()] = alloc_frac
            max_total_allocation_pct = max(0.0, 100.0 * BUYING_POWER_USAGE_CAP * float(buying_power_multiplier))
            total_allocation_pct = min(max(0.0, float(allocation_pct)), max_total_allocation_pct)
            target_plan = _portfolio_rebalance_targets(
                desired_allocations=desired,
                existing_qty_by_symbol={
                    symbol: qty for symbol, (qty, _entry_price) in portfolio_positions.items()
                },
                buy_prices={
                    symbol: prices.get(symbol, 0.0) * (1.0 + entry_offset_bps / 10_000.0)
                    for symbol in set(portfolio_positions) | set(desired)
                },
                sell_prices={
                    symbol: prices.get(symbol, 0.0) * (1.0 + exit_offset_bps / 10_000.0)
                    for symbol in set(portfolio_positions) | set(desired)
                },
                portfolio_value=equity,
                buying_power=max(0.0, cash * float(buying_power_multiplier)),
                total_allocation_pct=total_allocation_pct,
            )
            for pos_symbol, (qty, entry_price) in list(portfolio_positions.items()):
                plan = target_plan.get(pos_symbol)
                sell_price = prices.get(pos_symbol, 0.0) * (1.0 + exit_offset_bps / 10_000.0)
                target_qty = 0.0 if plan is None else plan.target_qty
                delta_qty = round(target_qty - qty, 4)
                if delta_qty >= -_PORTFOLIO_REBALANCE_QTY_EPSILON:
                    continue
                cash += abs(delta_qty) * sell_price
                trades += 1
                if target_qty <= _PORTFOLIO_REBALANCE_QTY_EPSILON:
                    del portfolio_positions[pos_symbol]
                else:
                    portfolio_positions[pos_symbol] = (target_qty, entry_price)
            for pos_symbol, plan in target_plan.items():
                target_qty = plan.target_qty
                if target_qty <= _PORTFOLIO_REBALANCE_QTY_EPSILON:
                    continue
                existing_qty, existing_entry_price = portfolio_positions.get(pos_symbol, (0.0, 0.0))
                delta_qty = round(target_qty - existing_qty, 4)
                if delta_qty <= _PORTFOLIO_REBALANCE_QTY_EPSILON:
                    continue
                buy_price = prices[pos_symbol] * (1.0 + entry_offset_bps / 10_000.0)
                if buy_price <= 0.0:
                    continue
                cash -= delta_qty * buy_price
                trades += 1
                if existing_qty > 0.0:
                    entry_price = ((existing_qty * existing_entry_price) + (delta_qty * buy_price)) / target_qty
                else:
                    entry_price = buy_price
                portfolio_positions[pos_symbol] = (target_qty, entry_price)
            trader.step_day()
            continue

        if position is not None and (signal.symbol != position[0] or signal.direction != "long"):
            pos_symbol, qty, _ = position
            sell_price = prices[pos_symbol] * (1.0 + exit_offset_bps / 10_000.0)
            cash += qty * sell_price
            position = None
            trades += 1
            trader.update_state(0, 0.0, "")

        if position is None and signal.symbol and signal.direction == "long":
            if not _signal_passes_open_gate(signal):
                gate_blocked_opens += 1
                trader.step_day()
                continue
            effective_allocation_pct = resolved_signal_allocation_pct(
                signal,
                base_allocation_pct=allocation_pct,
                sizing_mode=allocation_sizing_mode,
                min_open_confidence=DEFAULT_MIN_OPEN_CONFIDENCE,
            )
            buy_price = prices[signal.symbol] * (1.0 + entry_offset_bps / 10_000.0)
            qty = compute_target_qty_from_values(
                portfolio_value=equity,
                buying_power=equity * float(buying_power_multiplier),
                price=buy_price,
                allocation_pct=effective_allocation_pct,
            )
            if qty <= 0:
                trader.step_day()
                continue
            cash -= qty * buy_price
            position = (signal.symbol, qty, buy_price)
            trader.update_state(trader.SYMBOLS.index(signal.symbol) + 1, prices[signal.symbol], signal.symbol, qty=qty)

        trader.step_day()

    if position is not None:
        equity_curve.append(cash + position[1] * indexed[position[0]]["close"].iloc[min_len - 1])
    elif portfolio_positions:
        equity_curve.append(
            cash
            + sum(
                qty * indexed[symbol]["close"].iloc[min_len - 1]
                for symbol, (qty, _entry_price) in portfolio_positions.items()
            )
        )
    else:
        equity_curve.append(cash)

    curve = np.asarray(equity_curve, dtype=np.float64)
    total_return = float(curve[-1] / curve[0] - 1.0)
    daily_returns = np.diff(curve) / np.clip(curve[:-1], 1e-8, None)
    downside = daily_returns[daily_returns < 0.0]
    downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 1e-8
    sortino = float(np.mean(daily_returns) / downside_dev * np.sqrt(252.0)) if len(daily_returns) else 0.0
    max_dd = float(np.min(curve / np.maximum.accumulate(curve) - 1.0))
    if total_return <= -1.0:
        annualized = -1.0
    else:
        annualized = float((1.0 + total_return) ** (252.0 / max(1, days)) - 1.0)

    results = {
        "total_return": total_return,
        "annualized_return": annualized,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "trades": float(trades),
        "gate_blocked_opens": float(gate_blocked_opens),
    }
    logger.info("Backtest results: %s", json.dumps(results, sort_keys=True))
    return results


def run_once(
    *,
    checkpoint: str,
    symbols: Iterable[str],
    paper: bool,
    allocation_pct: float,
    dry_run: bool,
    data_source: StockDataSource,
    data_dir: str,
    allocation_sizing_mode: StockAllocationSizingMode = DEFAULT_ALLOCATION_SIZING_MODE,
    state_path: Path = STATE_PATH,
    extra_checkpoints: Optional[list] = None,
    execution_backend: StockExecutionBackend = "alpaca",
    server_account: str = DEFAULT_SERVER_PAPER_ACCOUNT,
    server_bot_id: str = DEFAULT_SERVER_PAPER_BOT_ID,
    server_url: str | None = None,
    server_session_id: str | None = None,
    multi_position: int = DEFAULT_MULTI_POSITION,
    multi_position_min_prob_ratio: float = DEFAULT_MULTI_POSITION_MIN_PROB_RATIO,
    min_open_confidence: float = DEFAULT_MIN_OPEN_CONFIDENCE,
    min_open_value_estimate: float = DEFAULT_MIN_OPEN_VALUE_ESTIMATE,
    allow_unsafe_checkpoint_loading: bool = False,
    meta_selector: bool = False,
    meta_top_k: int = 1,
    meta_lookback: int = 3,
) -> dict:
    now = datetime.now(timezone.utc)
    run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:12]}"
    symbol_list = [str(symbol).upper() for symbol in symbols]
    failure_stage = "load_state"
    failure_observability: dict[str, object] = {
        "run_id": run_id,
        "symbol_count": len(symbol_list),
        "execution_backend": execution_backend,
        "data_source": data_source,
        "account_mode": "paper" if paper else "live",
        "dry_run": bool(dry_run),
    }
    try:
        state = load_state(state_path)
        failure_observability["state_active_symbol"] = state.active_symbol
        failure_observability["state_last_run_date"] = state.last_run_date
        failure_observability["state_pending_close_symbol"] = state.pending_close_symbol
        quote_data_source = "local"
        quote_source_by_symbol: dict[str, str] = {}
        latest_bar = None
        market_open = None
        portfolio_equity = float(DEFAULT_BACKTEST_STARTING_CASH)
        if data_source == "alpaca":
            failure_stage = "build_data_client"
            data_client = build_data_client(paper=paper)
            failure_stage = "build_clock_client"
            clock_client = build_trading_client(paper=paper)
            failure_stage = "load_inference_frames"
            frames, bar_data_source = load_inference_frames(
                symbol_list,
                paper=paper,
                data_dir=data_dir,
                now=now,
                data_client=data_client,
            )
            close_prices = latest_close_prices(frames)
            latest_bar = latest_bar_timestamp(frames)
            failure_observability["bar_data_source"] = bar_data_source
            failure_observability["latest_bar_timestamp"] = (
                latest_bar.isoformat() if latest_bar is not None else None
            )
            failure_stage = "load_latest_quotes"
            quotes, quote_data_source, quote_source_by_symbol = load_latest_quotes_with_source(
                symbol_list,
                paper=paper,
                fallback_prices=close_prices,
                data_client=data_client,
            )
            failure_observability["quote_data_source"] = quote_data_source
            if quote_source_by_symbol:
                failure_observability["quote_source_by_symbol"] = dict(quote_source_by_symbol)
                fallback_symbols = sorted(
                    symbol for symbol, source in quote_source_by_symbol.items() if source != "alpaca"
                )
                if fallback_symbols:
                    failure_observability["quote_fallback_symbols"] = fallback_symbols
            if execution_backend == "trading_server":
                failure_stage = "build_trading_server_client"
                client = build_server_client(
                    account=server_account,
                    bot_id=server_bot_id,
                    paper=paper,
                    base_url=server_url,
                    session_id=server_session_id,
                )
                if not dry_run:
                    failure_stage = "claim_trading_server_writer"
                    client.claim_writer()
                failure_stage = "get_trading_server_account"
                snapshot = client.get_account()
                live_positions = server_positions_by_symbol(snapshot, symbol_list)
                if not dry_run and multi_position <= 1:
                    reconcile_pending_close(state=state, live_positions=live_positions)
                    adopt_existing_position(state=state, live_positions=live_positions, now=now)
                failure_observability["live_position_symbols"] = sorted(live_positions.keys())
                portfolio_equity = server_equity(snapshot, quotes)
                if multi_position > 1:
                    portfolio = PortfolioContext(cash=portfolio_equity)
                else:
                    portfolio = server_portfolio_context(
                        snapshot=snapshot,
                        state=state,
                        quotes=quotes,
                        now=now,
                    )
            else:
                failure_stage = "build_alpaca_trading_client"
                client = build_trading_client(paper=paper)
                failure_stage = "get_alpaca_positions"
                live_positions = positions_by_symbol(client, symbol_list)
                if not dry_run and multi_position <= 1:
                    reconcile_pending_close(state=state, live_positions=live_positions)
                    adopt_existing_position(state=state, live_positions=live_positions, now=now)
                failure_observability["live_position_symbols"] = sorted(live_positions.keys())
                failure_stage = "get_alpaca_account"
                account = client.get_account()
                portfolio_equity = account_portfolio_value(account) or float(DEFAULT_BACKTEST_STARTING_CASH)
                if multi_position > 1:
                    portfolio = PortfolioContext(cash=portfolio_equity)
                else:
                    portfolio = build_portfolio_context(
                        state=state,
                        live_positions=live_positions,
                        account=account,
                        now=now,
                    )
            try:
                failure_stage = "get_market_clock"
                market_open = bool(getattr(clock_client.get_clock(), "is_open", False))
            except Exception as exc:
                logger.warning("Could not read Alpaca market clock: %s", exc)
                market_open = False
            failure_observability["market_open"] = market_open
        else:
            failure_stage = "load_local_daily_frames"
            frames = load_local_daily_frames(
                symbol_list,
                data_dir=data_dir,
                min_days=DEFAULT_DAILY_FRAME_MIN_DAYS,
            )
            quotes = latest_close_prices(frames)
            portfolio = PortfolioContext()
            bar_data_source = "local"
            latest_bar = latest_bar_timestamp(frames)
            portfolio_equity = float(portfolio.cash)
            failure_observability["bar_data_source"] = bar_data_source
            failure_observability["quote_data_source"] = "local_close"
            failure_observability["latest_bar_timestamp"] = (
                latest_bar.isoformat() if latest_bar is not None else None
            )

        failure_stage = "build_signal"
        close_prices = latest_close_prices(frames)
        portfolio_signals: list[TradingSignal] = []
        if multi_position > 1:
            portfolio_quotes = {
                symbol: float(quotes.get(symbol, close_prices.get(symbol, 0.0)) or close_prices.get(symbol, 0.0))
                for symbol in symbol_list
            }
            portfolio_signals = _build_multi_position_signals(
                checkpoint,
                frames,
                quotes=portfolio_quotes,
                portfolio_value=portfolio_equity,
                multi_position=multi_position,
                multi_position_min_prob_ratio=multi_position_min_prob_ratio,
                extra_checkpoints=extra_checkpoints,
                allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
            )
            signal = (
                portfolio_signals[0]
                if portfolio_signals
                else TradingSignal(
                    action="flat",
                    symbol=None,
                    direction=None,
                    confidence=0.0,
                    value_estimate=0.0,
                    allocation_pct=0.0,
                    level_offset_bps=0.0,
                )
            )
            failure_observability["portfolio_signal_symbols"] = [
                str(sig.symbol).upper()
                for sig in portfolio_signals
                if sig.symbol
            ]
        else:
            signal, close_prices = build_signal(
                checkpoint,
                frames,
                portfolio=portfolio,
                extra_checkpoints=extra_checkpoints,
                allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
                meta_selector=meta_selector,
                meta_top_k=meta_top_k,
                meta_lookback=meta_lookback,
            )
        failure_observability["signal_action"] = signal.action
        failure_observability["signal_symbol"] = signal.symbol
        failure_observability["signal_direction"] = signal.direction
        failure_observability["signal_confidence"] = float(signal.confidence)
        failure_observability["signal_value_estimate"] = float(signal.value_estimate)
        bars_fresh = (
            bars_are_fresh(
                latest_bar=latest_bar,
                now=now,
                max_age_days=DEFAULT_BAR_FRESHNESS_MAX_AGE_DAYS,
            )
            if latest_bar is not None
            else False
        )

        payload = _signal_payload(signal, checkpoint=checkpoint, quotes=quotes, now=now, run_id=run_id)
        payload["close_prices"] = close_prices
        payload["bar_data_source"] = bar_data_source
        payload["quote_data_source"] = quote_data_source
        payload["quote_source_by_symbol"] = dict(quote_source_by_symbol)
        payload["latest_bar_timestamp"] = latest_bar.isoformat() if latest_bar is not None else None
        payload["bars_fresh"] = bars_fresh
        payload["market_open"] = market_open
        payload["dry_run"] = bool(dry_run)
        payload["execution_backend"] = execution_backend
        payload["allocation_sizing_mode"] = allocation_sizing_mode
        payload["min_open_confidence"] = float(min_open_confidence)
        payload["min_open_value_estimate"] = float(min_open_value_estimate)
        payload["portfolio_mode"] = bool(multi_position > 1)
        if multi_position > 1:
            payload["portfolio_signal_count"] = len(portfolio_signals)
            payload["portfolio_signals"] = [
                _portfolio_signal_payload(portfolio_signal)
                for portfolio_signal in portfolio_signals
            ]

        logger.info("%s", "=" * 60)
        logger.info("DAILY STOCK RL SIGNAL (%s, run_id=%s)", now.strftime("%Y-%m-%d %H:%M UTC"), run_id)
        logger.info("%s", "=" * 60)
        logger.info("Action:     %s", signal.action)
        logger.info("Symbol:     %s", signal.symbol or "N/A")
        logger.info("Direction:  %s", signal.direction or "N/A")
        logger.info("Confidence: %.1f%%", float(signal.confidence) * 100.0)
        logger.info("Value est:  %.4f", float(signal.value_estimate))
        if multi_position > 1:
            rendered_portfolio = ", ".join(
                f"{str(sig.symbol).upper()}:{float(sig.allocation_pct) * 100.0:.1f}%"
                for sig in portfolio_signals
                if sig.symbol
            ) or "flat"
            logger.info("Portfolio:  %s", rendered_portfolio)
        logger.info("Bars:       %s latest=%s fresh=%s", bar_data_source, latest_bar.isoformat() if latest_bar is not None else "n/a", bars_fresh)
        logger.info("Quotes:     %s", quote_data_source)
        if quote_source_by_symbol:
            fallback_symbols = sorted(symbol for symbol, source in quote_source_by_symbol.items() if source != "alpaca")
            if fallback_symbols:
                logger.info("Quote fallbacks: %s", ", ".join(fallback_symbols))

        executed = False
        allow_open = True
        allow_open_reason: str | None = None
        allow_open_reasons: list[str] = []
        blocked_portfolio_signals: list[dict[str, object]] = []

        # Regime filter: skip opening new positions in bear markets (SPY < 20-day MA).
        # Does NOT force-close existing positions — only blocks new entries.
        _regime_ok, _regime_reason = regime_filter_reason(data_dir=data_dir)
        if not _regime_ok:
            allow_open = False
            allow_open_reason = _regime_reason
            logger.warning("Regime filter: %s — skipping new position opens", _regime_reason)

        if data_source == "alpaca":
            if multi_position > 1:
                desired_portfolio_signals = [
                    portfolio_signal
                    for portfolio_signal in portfolio_signals
                    if portfolio_signal.symbol and portfolio_signal.direction == "long"
                ]
                executable_signals: list[TradingSignal] = []
                for portfolio_signal in desired_portfolio_signals:
                    signal_quote_source = quote_source_by_symbol.get(
                        str(portfolio_signal.symbol).upper(),
                        quote_data_source,
                    )
                    reasons = _open_gate_reasons(
                        portfolio_signal,
                        signal_quote_source=signal_quote_source,
                        quote_data_source=quote_data_source,
                        min_open_confidence=min_open_confidence,
                        min_open_value_estimate=min_open_value_estimate,
                    )
                    if reasons:
                        blocked_portfolio_signals.append(
                            {
                                "symbol": str(portfolio_signal.symbol).upper(),
                                "reasons": reasons,
                            }
                        )
                        continue
                    executable_signals.append(portfolio_signal)
                allow_open = bool(executable_signals) or not desired_portfolio_signals
                if blocked_portfolio_signals and not allow_open:
                    allow_open_reasons = [
                        f"{blocked['symbol']}: {'; '.join(cast(list[str], blocked['reasons']))}"
                        for blocked in blocked_portfolio_signals
                    ]
                    allow_open_reason = " | ".join(allow_open_reasons)
                    logger.warning("Execution safety gate active for portfolio: %s", allow_open_reason)
                if blocked_portfolio_signals:
                    payload["blocked_portfolio_signals"] = blocked_portfolio_signals
                    failure_observability["blocked_portfolio_signal_symbols"] = [
                        str(blocked["symbol"])
                        for blocked in blocked_portfolio_signals
                    ]
            else:
                signal_quote_source = quote_source_by_symbol.get(signal.symbol, quote_data_source) if signal.symbol else quote_data_source
                allow_open_reasons = _open_gate_reasons(
                    signal,
                    signal_quote_source=signal_quote_source,
                    quote_data_source=quote_data_source,
                    min_open_confidence=min_open_confidence,
                    min_open_value_estimate=min_open_value_estimate,
                )
                allow_open = not allow_open_reasons
                if signal.symbol and not allow_open:
                    allow_open_reason = "; ".join(allow_open_reasons)
                    logger.warning(
                        "Execution safety gate active for %s: %s",
                        signal.symbol,
                        allow_open_reason,
                    )
            failure_observability["allow_open"] = allow_open
            failure_observability["allow_open_reason"] = allow_open_reason
            if not dry_run and not bool(market_open):
                logger.warning("Market is closed; skipping order placement")
            elif not dry_run and not bars_fresh:
                logger.warning("Latest inference bar is stale; skipping order placement")
            else:
                if execution_backend == "trading_server":
                    if multi_position > 1:
                        failure_stage = "execute_multi_position_signals_with_trading_server"
                        held_positions = execute_multi_position_signals_with_trading_server(
                            executable_signals,
                            server_client=client,
                            quotes=quotes,
                            symbols=symbol_list,
                            total_allocation_pct=allocation_pct,
                            dry_run=dry_run,
                        )
                        desired_symbols = {
                            str(portfolio_signal.symbol).upper()
                            for portfolio_signal in executable_signals
                            if portfolio_signal.symbol
                        }
                        live_position_symbols = set(live_positions.keys())
                        executed = bool(held_positions) or bool(live_position_symbols - desired_symbols)
                        payload["held_positions"] = held_positions
                    else:
                        failure_stage = "execute_signal_with_trading_server"
                        executed = execute_signal_with_trading_server(
                            signal,
                            server_client=client,
                            quotes=quotes,
                            state=state,
                            symbols=symbol_list,
                            allocation_pct=allocation_pct,
                            allocation_sizing_mode=allocation_sizing_mode,
                            dry_run=dry_run,
                            now=now,
                            allow_open=allow_open,
                            allow_open_reason=allow_open_reason,
                            min_open_confidence=min_open_confidence,
                        )
                    failure_stage = "get_trading_server_account_snapshot"
                    payload["server_account"] = server_account
                    payload["server_bot_id"] = server_bot_id
                    payload["server_snapshot"] = client.get_account()
                else:
                    if multi_position > 1:
                        failure_stage = "execute_multi_position_signals"
                        held_positions = execute_multi_position_signals(
                            executable_signals,
                            client=client,
                            paper=paper,
                            quotes=quotes,
                            symbols=symbol_list,
                            total_allocation_pct=allocation_pct,
                            dry_run=dry_run,
                            now=now,
                        )
                        desired_symbols = {
                            str(portfolio_signal.symbol).upper()
                            for portfolio_signal in executable_signals
                            if portfolio_signal.symbol
                        }
                        live_position_symbols = set(live_positions.keys())
                        executed = bool(held_positions) or bool(live_position_symbols - desired_symbols)
                        payload["held_positions"] = held_positions
                    else:
                        failure_stage = "execute_signal"
                        executed = execute_signal(
                            signal,
                            client=client,
                            paper=paper,
                            quotes=quotes,
                            state=state,
                            symbols=symbol_list,
                            allocation_pct=allocation_pct,
                            allocation_sizing_mode=allocation_sizing_mode,
                            dry_run=dry_run,
                            now=now,
                            allow_open=allow_open,
                            allow_open_reason=allow_open_reason,
                            min_open_confidence=min_open_confidence,
                        )
        else:
            logger.info("Local data mode selected; skipping execution")

        payload["allow_open"] = allow_open
        payload["allow_open_reason"] = allow_open_reason
        payload["allow_open_reasons"] = allow_open_reasons
        should_advance_state = (
            not dry_run
            and (data_source != "alpaca" or bool(market_open))
        )
        payload.update(
            _execution_observability_fields(
                data_source=data_source,
                dry_run=dry_run,
                market_open=market_open,
                bars_fresh=bars_fresh,
                signal=signal,
                allow_open=allow_open,
                executed=executed,
            )
        )
        payload["state_advanced"] = should_advance_state
        if should_advance_state:
            failure_stage = "save_state"
            state.last_run_date = now.astimezone(EASTERN).date().isoformat()
            state.last_signal_action = "portfolio" if multi_position > 1 else signal.action
            state.last_signal_timestamp = now.isoformat()
            if multi_position > 1:
                state.active_symbol = None
                state.active_qty = 0.0
                state.entry_price = 0.0
                state.entry_date = None
                state.last_order_id = None
                state.pending_close_symbol = None
                state.pending_close_order_id = None
            save_state(state, path=state_path)
        payload["executed"] = executed
        signal_log_write_error = _append_signal_log_best_effort(payload)
        payload["signal_log_written"] = signal_log_write_error is None
        payload["signal_log_write_error"] = signal_log_write_error
        run_event_log_write_error = _append_run_event_log_best_effort(_run_summary_payload(payload))
        payload["run_event_log_written"] = run_event_log_write_error is None
        payload["run_event_log_write_error"] = run_event_log_write_error
        failure_stage = "log_run_summary"
        _log_run_summary(payload)
        logger.info("%s", "=" * 60)
        return payload
    except Exception as exc:
        stage_note = f"run_once stage: {failure_stage}"
        if hasattr(exc, "add_note") and stage_note not in _exception_notes(exc):
            exc.add_note(stage_note)
        context_note = _format_run_failure_context_note(failure_observability)
        if hasattr(exc, "add_note") and context_note and context_note not in _exception_notes(exc):
            exc.add_note(context_note)
        failure_payload = _run_failure_payload(
            run_id=run_id,
            now=now,
            checkpoint=checkpoint,
            stage=failure_stage,
            exc=exc,
            data_source=data_source,
            execution_backend=execution_backend,
            paper=paper,
            dry_run=dry_run,
            state_path=state_path,
            symbols=symbol_list,
            data_dir=data_dir,
            server_account=server_account if execution_backend == "trading_server" else None,
            server_bot_id=server_bot_id if execution_backend == "trading_server" else None,
            server_url=resolve_trading_server_base_url(server_url)
            if execution_backend == "trading_server"
            else None,
            observability=failure_observability,
        )
        _log_run_failure(failure_payload)
        _append_run_event_log_best_effort(failure_payload)
        raise


def run_daemon(
    *,
    checkpoint: str,
    symbols: Iterable[str],
    paper: bool,
    allocation_pct: float,
    allocation_sizing_mode: StockAllocationSizingMode,
    dry_run: bool,
    data_dir: str,
    extra_checkpoints: Optional[list] = None,
    execution_backend: str = "alpaca",
    server_account: str = DEFAULT_SERVER_PAPER_ACCOUNT,
    server_bot_id: str = DEFAULT_SERVER_PAPER_BOT_ID,
    server_url: str | None = None,
    multi_position: int = DEFAULT_MULTI_POSITION,
    multi_position_min_prob_ratio: float = DEFAULT_MULTI_POSITION_MIN_PROB_RATIO,
    min_open_confidence: float = DEFAULT_MIN_OPEN_CONFIDENCE,
    min_open_value_estimate: float = DEFAULT_MIN_OPEN_VALUE_ESTIMATE,
    allow_unsafe_checkpoint_loading: bool = False,
    meta_selector: bool = False,
    meta_top_k: int = 1,
    meta_lookback: int = 3,
) -> None:
    logger.info("Starting daily stock RL daemon")
    server_session_id = f"daily-rl-trader-{execution_backend}-{os.getpid()}"
    state_path = STATE_PATH
    while True:
        try:
            state = load_state(state_path)
        except Exception as exc:
            logger.warning("State load failed (%s); sleeping 5min", exc)
            _log_daemon_warning_and_sleep(
                checkpoint=checkpoint,
                stage="load_state",
                exc=exc,
                paper=paper,
                execution_backend=execution_backend,
                sleep_seconds=DEFAULT_DAEMON_RETRY_SLEEP_SECONDS,
                state_path=state_path,
                server_session_id=server_session_id,
            )
            continue
        # Use paper API for clock check (market hours same regardless of paper/live).
        # Fall back to paper=True if live keys are invalid (401) — service stays alive.
        clock_client = build_trading_client(paper=paper)
        try:
            clock = clock_client.get_clock()
        except Exception as _clock_err:
            if not paper:
                logger.warning("Live clock check failed (%s); retrying with paper API", _clock_err)
                _log_daemon_warning(
                    _daemon_warning_payload(
                        now=datetime.now(timezone.utc),
                        checkpoint=checkpoint,
                        stage="clock_check_live",
                        exc=_clock_err,
                        paper=paper,
                        execution_backend=execution_backend,
                        retry_with_paper_clock=True,
                        clock_source="live",
                        server_session_id=server_session_id,
                    )
                )
                try:
                    clock = build_trading_client(paper=True).get_clock()
                except Exception as _paper_err:
                    logger.warning("Paper clock check also failed (%s); sleeping 5min", _paper_err)
                    _log_daemon_warning_and_sleep(
                        checkpoint=checkpoint,
                        stage="clock_check_paper_fallback",
                        exc=_paper_err,
                        paper=paper,
                        execution_backend=execution_backend,
                        sleep_seconds=DEFAULT_DAEMON_RETRY_SLEEP_SECONDS,
                        clock_source="paper_fallback",
                        server_session_id=server_session_id,
                    )
                    continue
            else:
                logger.warning("Clock check failed (%s); sleeping 5min", _clock_err)
                _log_daemon_warning_and_sleep(
                    checkpoint=checkpoint,
                    stage="clock_check_paper",
                    exc=_clock_err,
                    paper=paper,
                    execution_backend=execution_backend,
                    sleep_seconds=DEFAULT_DAEMON_RETRY_SLEEP_SECONDS,
                    clock_source="paper",
                    server_session_id=server_session_id,
                )
                continue
        now = datetime.now(timezone.utc)
        if should_run_today(
            now=now,
            is_market_open=bool(getattr(clock, "is_open", False)),
            last_run_date=state.last_run_date,
        ):
            try:
                run_once(
                    checkpoint=checkpoint,
                    symbols=symbols,
                    paper=paper,
                    allocation_pct=allocation_pct,
                    allocation_sizing_mode=allocation_sizing_mode,
                    dry_run=dry_run,
                    data_source="alpaca",
                    data_dir=data_dir,
                    extra_checkpoints=extra_checkpoints,
                    execution_backend=execution_backend,
                    server_account=server_account,
                    server_bot_id=server_bot_id,
                    server_url=server_url,
                    server_session_id=server_session_id,
                    multi_position=multi_position,
                    multi_position_min_prob_ratio=multi_position_min_prob_ratio,
                    min_open_confidence=min_open_confidence,
                    min_open_value_estimate=min_open_value_estimate,
                    allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
                    meta_selector=meta_selector,
                    meta_top_k=meta_top_k,
                    meta_lookback=meta_lookback,
                )
            except Exception as exc:
                logger.exception("Daily stock RL cycle failed: %s", exc)
            time.sleep(60.0)
            continue

        sleep_seconds = seconds_until_next_check(
            now=now,
            is_market_open=bool(getattr(clock, "is_open", False)),
            next_open=getattr(clock, "next_open", None),
        )
        logger.info("Sleeping %.1f minutes", sleep_seconds / 60.0)
        time.sleep(sleep_seconds)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production daily stock RL trader")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--extra-checkpoints", nargs="*", default=None,
                        help="Additional checkpoints for ensemble (default: DEFAULT_EXTRA_CHECKPOINTS)")
    parser.add_argument("--ensemble-dir", default=None,
                        help="Directory of .pt checkpoints (auto-discovers all, overrides --extra-checkpoints)")
    parser.add_argument("--no-ensemble", action="store_true",
                        help="Disable ensemble, use --checkpoint alone")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--data-source", choices=["alpaca", "local"], default="alpaca")
    parser.add_argument("--allocation-pct", type=float, default=DEFAULT_ALLOCATION_PCT)
    parser.add_argument(
        "--allocation-sizing-mode",
        choices=["static", "confidence_scaled"],
        default=DEFAULT_ALLOCATION_SIZING_MODE,
        help="Map the base allocation to order size either directly or scaled by model confidence.",
    )
    parser.add_argument("--multi-position", type=int, default=DEFAULT_MULTI_POSITION,
                        help="Hold up to N simultaneous positions (0=single position mode)")
    parser.add_argument("--multi-position-min-prob-ratio", type=float, default=DEFAULT_MULTI_POSITION_MIN_PROB_RATIO,
                        help="Min probability ratio vs top signal to include in portfolio")
    run_mode_group = parser.add_mutually_exclusive_group()
    run_mode_group.add_argument("--once", action="store_true", help="Run one inference cycle")
    run_mode_group.add_argument("--daemon", action="store_true", help="Run as a daemon around market open")
    parser.add_argument("--dry-run", action="store_true", help="Print signals without placing orders")
    account_group = parser.add_mutually_exclusive_group()
    account_group.add_argument("--paper", action="store_true", help="Use the Alpaca paper account (default)")
    account_group.add_argument("--live", action="store_true", help="Use the Alpaca live account")
    run_mode_group.add_argument("--backtest", action="store_true", help="Run a local historical backtest")
    parser.add_argument("--backtest-days", type=int, default=60)
    parser.add_argument("--backtest-starting-cash", type=float, default=DEFAULT_BACKTEST_STARTING_CASH)
    parser.add_argument(
        "--backtest-buying-power-multiplier",
        type=float,
        default=DEFAULT_BACKTEST_BUYING_POWER_MULTIPLIER,
        help="Sizing-only backtest buying power multiplier (1.0=cash-only, 2.0=2x margin).",
    )
    parser.add_argument(
        "--backtest-entry-offset-bps",
        type=float,
        default=0.0,
        help=(
            "Entry price offset used only for local backtests. "
            "Positive values buy above the next open; negative values require a dip below it."
        ),
    )
    parser.add_argument(
        "--backtest-exit-offset-bps",
        type=float,
        default=0.0,
        help=(
            "Exit price offset used only for local backtests. "
            "Positive values sell above the next open when reached."
        ),
    )
    parser.add_argument("--compare-server-parity", action="store_true",
                        help="Compare the legacy daily backtest against the trading-server paper replay")
    parser.add_argument("--execution-backend", choices=["alpaca", "trading_server"], default="alpaca")
    parser.add_argument("--server-account", default=DEFAULT_SERVER_PAPER_ACCOUNT)
    parser.add_argument("--server-bot-id", default=DEFAULT_SERVER_PAPER_BOT_ID)
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--symbols-file", default=None,
                        help="Optional newline/comma separated symbol file. Overrides --symbols.")
    parser.add_argument("--check-config", action="store_true",
                        help="Run a preflight config check, print a readiness report, and exit")
    parser.add_argument(
        "--check-config-text",
        action="store_true",
        help="When combined with --check-config, also print a human-readable readiness summary to stderr",
    )
    parser.add_argument(
        "--check-config-summary",
        action="store_true",
        help="When combined with --check-config, also print a compact one-line readiness summary to stderr",
    )
    parser.add_argument("--print-config", action="store_true",
                        help="Print the fully resolved runtime configuration and exit")
    parser.add_argument("--print-payload", action="store_true",
                        help="Print the one-shot inference payload as JSON after a successful run")
    parser.add_argument("--min-open-confidence", type=float, default=DEFAULT_MIN_OPEN_CONFIDENCE,
                        help="Minimum ensemble confidence required before opening a new position")
    parser.add_argument("--min-open-value-estimate", type=float, default=DEFAULT_MIN_OPEN_VALUE_ESTIMATE,
                        help="Minimum critic value estimate required before opening a new position")
    parser.add_argument(
        "--allow-unsafe-checkpoint-loading",
        action="store_true",
        help="Allow legacy pickle checkpoint loading. Only use this with trusted checkpoint files.",
    )
    parser.add_argument(
        "--meta-selector",
        action="store_true",
        help="Use meta-selector (per-model momentum selection) instead of softmax ensemble",
    )
    parser.add_argument("--meta-top-k", type=int, default=1,
                        help="Number of top models to follow in meta-selector mode")
    parser.add_argument("--meta-lookback", type=int, default=3,
                        help="Lookback days for meta-selector momentum ranking")
    args = parser.parse_args(argv)
    if args.check_config_text and not args.check_config:
        parser.error("--check-config-text requires --check-config")
    if args.check_config_summary and not args.check_config:
        parser.error("--check-config-summary requires --check-config")
    return args


def _resolve_runtime_config(args: argparse.Namespace) -> CliRuntimeConfig:
    paper = not bool(args.live)
    symbol_inputs = _load_symbols_file(args.symbols_file) if args.symbols_file else (args.symbols or DEFAULT_SYMBOLS)
    symbols, removed_duplicate_symbols, ignored_symbol_inputs = _normalize_symbols(
        symbol_inputs
    )
    checkpoint = (
        str((REPO / args.checkpoint).resolve())
        if not Path(args.checkpoint).is_absolute()
        else args.checkpoint
    )

    def _resolve(path: str) -> str:
        return str((REPO / path).resolve()) if not Path(path).is_absolute() else path

    if args.no_ensemble:
        extra_checkpoints: Optional[list[str]] = None
    elif args.ensemble_dir is not None:
        ens_dir = Path(_resolve(args.ensemble_dir))
        all_pts = sorted(ens_dir.glob("*.pt"))
        extra_checkpoints = [str(p) for p in all_pts]
        checkpoint = extra_checkpoints[0] if extra_checkpoints else checkpoint
        extra_checkpoints = extra_checkpoints[1:]
    elif args.extra_checkpoints is not None:
        extra_checkpoints = [_resolve(path) for path in args.extra_checkpoints]
    else:
        extra_checkpoints = _resolved_default_extra_checkpoints()

    return CliRuntimeConfig(
        paper=paper,
        symbols=symbols,
        symbols_file=args.symbols_file,
        removed_duplicate_symbols=removed_duplicate_symbols,
        ignored_symbol_inputs=ignored_symbol_inputs,
        checkpoint=checkpoint,
        extra_checkpoints=extra_checkpoints,
        data_dir=args.data_dir,
        data_source=cast(StockDataSource, args.data_source),
        allocation_pct=float(args.allocation_pct),
        allocation_sizing_mode=cast(StockAllocationSizingMode, args.allocation_sizing_mode),
        multi_position=int(args.multi_position),
        multi_position_min_prob_ratio=float(args.multi_position_min_prob_ratio),
        execution_backend=cast(StockExecutionBackend, args.execution_backend),
        server_account=args.server_account,
        server_bot_id=args.server_bot_id,
        server_url=args.server_url,
        dry_run=bool(args.dry_run),
        backtest=bool(args.backtest),
        backtest_days=int(args.backtest_days),
        backtest_starting_cash=float(args.backtest_starting_cash),
        backtest_buying_power_multiplier=float(args.backtest_buying_power_multiplier),
        backtest_entry_offset_bps=float(args.backtest_entry_offset_bps),
        backtest_exit_offset_bps=float(args.backtest_exit_offset_bps),
        daemon=bool(args.daemon),
        compare_server_parity=bool(args.compare_server_parity),
        min_open_confidence=float(args.min_open_confidence),
        min_open_value_estimate=float(args.min_open_value_estimate),
        print_payload=bool(args.print_payload),
        allow_unsafe_checkpoint_loading=bool(args.allow_unsafe_checkpoint_loading),
        meta_selector=bool(args.meta_selector),
        meta_top_k=int(args.meta_top_k),
        meta_lookback=int(args.meta_lookback),
    )


def _runtime_config_payload(config: CliRuntimeConfig) -> dict[str, object]:
    return config.to_runtime_payload()


def _missing_checkpoint_paths(config: CliRuntimeConfig) -> list[str]:
    return config.missing_checkpoint_paths


def _checkpoint_load_diagnostics(config: CliRuntimeConfig) -> CheckpointLoadDiagnostics:
    if config.missing_checkpoint_paths:
        return {
            "ok": False,
            "error": "checkpoint files are missing",
            "primary": None,
            "extras": [],
        }
    try:
        feature_schema = resolve_daily_feature_schema(
            config.checkpoint,
            extra_checkpoints=config.extra_checkpoints,
        )
        trader = _load_cached_daily_trader(
            config.checkpoint,
            device="cpu",
            long_only=True,
            symbols=list(config.symbols),
            allow_unsafe_checkpoint_loading=config.allow_unsafe_checkpoint_loading,
        )
        primary_summary: CheckpointLoadPrimarySummary = cast(
            CheckpointLoadPrimarySummary,
            trader.summary_dict(),
        )
        primary_summary["feature_schema"] = feature_schema
        primary_summary["feature_dimension"] = daily_feature_dimension(feature_schema)
        extras: list[CheckpointLoadExtraSummary] = []
        for path in config.extra_checkpoints or []:
            policy = _load_bare_policy(
                path,
                trader.obs_size,
                trader.num_actions,
                "cpu",
                allow_unsafe_checkpoint_loading=config.allow_unsafe_checkpoint_loading,
            )
            extras.append(
                {
                    "path": path,
                    "class": type(policy).__name__,
                }
            )
        return {
            "ok": True,
            "error": None,
            "primary": primary_summary,
            "extras": extras,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "primary": None,
            "extras": [],
        }


def _resolve_alpaca_credentials(*, paper: bool) -> tuple[str, str]:
    import importlib

    try:
        env_real = importlib.import_module("env_real")
    except Exception as exc:
        raise RuntimeError(f"Unable to load env_real for Alpaca credentials: {exc}") from exc
    paper_key_id = getattr(env_real, "ALP_KEY_ID_PAPER", getattr(env_real, "ALP_KEY_ID", ""))
    paper_secret = getattr(env_real, "ALP_SECRET_KEY_PAPER", getattr(env_real, "ALP_SECRET_KEY", ""))
    live_key_id = getattr(env_real, "ALP_KEY_ID_PROD", "")
    live_secret = getattr(env_real, "ALP_SECRET_KEY_PROD", "")

    if paper:
        return str(paper_key_id), str(paper_secret)
    return str(live_key_id), str(live_secret)


def _looks_placeholder(value: str | None) -> bool:
    normalized = str(value or "").strip()
    return not normalized or "placeholder" in normalized.lower()


def _resolved_trading_server_url(config: CliRuntimeConfig) -> str:
    return config.resolved_server_url


def _preflight_next_steps(payload: dict[str, object]) -> list[str]:
    steps: list[str] = []
    check_command = str(payload.get("check_command_preview") or "").strip()
    check_text_command = str(payload.get("check_text_command_preview") or "").strip()
    safe_command = str(payload.get("safe_command_preview") or "").strip()
    run_command = str(payload.get("run_command_preview") or "").strip()

    if bool(payload.get("ready")):
        if safe_command:
            steps.append(f"Run a safe dry run: {safe_command}")
        if run_command and run_command != safe_command:
            steps.append(f"Start the configured runtime: {run_command}")
        if check_text_command:
            steps.append(f"Re-run text preflight later: {check_text_command}")
        elif check_command:
            steps.append(f"Re-run preflight later: {check_command}")
        return steps

    if payload.get("missing_checkpoints"):
        steps.append("Provide the missing checkpoint file(s) or update --checkpoint/--extra-checkpoints.")
    if payload.get("missing_local_symbol_files"):
        steps.append("Add the missing local daily CSV files under the resolved data directory.")
    stale_symbol_data = payload.get("stale_symbol_data")
    if isinstance(stale_symbol_data, dict) and stale_symbol_data:
        steps.append(
            "Refresh stale local daily CSVs for: "
            + ", ".join(str(symbol) for symbol in stale_symbol_data)
            + "."
        )
    symbol_details = payload.get("symbol_details")
    if isinstance(symbol_details, dict):
        invalid_symbols = [
            str(symbol)
            for symbol, detail in symbol_details.items()
            if isinstance(detail, dict) and detail.get("status") == "invalid"
        ]
        if invalid_symbols:
            steps.append("Repair unreadable local daily CSVs for: " + ", ".join(invalid_symbols) + ".")
    if bool(payload.get("alpaca_required")) and not bool(payload.get("alpaca_credentials_configured")):
        steps.append("Configure valid Alpaca credentials for the selected account mode.")
    if bool(payload.get("trading_server_required")):
        server_step_count = len(steps)
        server_url_source = payload.get("server_url_source")
        server_url_update_hint = _trading_server_url_update_hint(server_url_source)
        resolved_server_url = str(
            payload.get("resolved_server_url") or payload.get("server_url") or ""
        ).strip()
        server_url_security = str(payload.get("server_url_security") or "").strip().lower()
        if server_url_security == "invalid":
            steps.append(
                f"{server_url_update_hint} to a valid http:// or https:// trading server URL "
                f"(current: {resolved_server_url or '<unset>'})."
            )
        elif server_url_security == "insecure_remote_http":
            steps.append(
                f"{server_url_update_hint} to an https:// trading server URL for live remote "
                f"execution (current: {resolved_server_url})."
            )
        if str(payload.get("account_mode") or "").strip().lower() == "live":
            if str(payload.get("server_url_source") or "").strip().lower() == "default":
                steps.append(
                    "Pass --server-url explicitly or set TRADING_SERVER_URL before live "
                    "trading_server runs if the server is not local."
                )
            if str(payload.get("server_account") or "").strip() == DEFAULT_SERVER_PAPER_ACCOUNT:
                steps.append(
                    "Pass --server-account explicitly for the live trading_server account "
                    "instead of the paper default."
                )
        if not str(payload.get("server_account") or "").strip():
            steps.append("Set --server-account to the trading_server account before retrying.")
        if not str(payload.get("server_bot_id") or "").strip():
            steps.append("Set --server-bot-id to the trading_server bot id before retrying.")
        if len(steps) == server_step_count:
            steps.append("Verify --server-url, --server-account, and --server-bot-id before retrying.")
    if check_command:
        steps.append(f"Re-run preflight after fixes: {check_text_command or check_command}")
    return steps


def _build_local_symbol_details(
    *,
    symbols: Sequence[str],
    resolved_local_data_dir: Path | None,
) -> tuple[
    dict[str, StockLocalSymbolDetail],
    list[str],
    str | None,
    str | None,
]:
    details: dict[str, StockLocalSymbolDetail] = {}
    usable_symbols: list[str] = []
    usable_timestamps: dict[str, pd.Timestamp] = {}

    if resolved_local_data_dir is None:
        return details, usable_symbols, None, None

    for symbol in symbols:
        path = (resolved_local_data_dir / f"{symbol}.csv").resolve()
        if not path.exists():
            details[symbol] = {
                "status": LocalDataHealthStatus.MISSING,
                "file_path": str(path),
                "local_data_date": None,
                "row_count": None,
                "reason": "missing local daily CSV",
            }
            continue
        try:
            inspection = _inspect_local_daily_symbol_file(path)
        except Exception as exc:
            details[symbol] = {
                "status": LocalDataHealthStatus.INVALID,
                "file_path": str(path),
                "local_data_date": None,
                "row_count": None,
                "reason": str(exc),
            }
            continue

        latest_timestamp = inspection.latest_timestamp
        latest_date = latest_timestamp.date().isoformat()
        usable_symbols.append(symbol)
        usable_timestamps[symbol] = latest_timestamp
        details[symbol] = {
            "status": LocalDataHealthStatus.USABLE,
            "file_path": str(path),
            "local_data_date": latest_date,
            "row_count": inspection.row_count,
            "reason": None,
        }

    latest_local_data_date: str | None = None
    oldest_local_data_date: str | None = None
    if usable_timestamps:
        newest_timestamp = max(usable_timestamps.values())
        oldest_timestamp = min(usable_timestamps.values())
        latest_local_data_date = newest_timestamp.date().isoformat()
        oldest_local_data_date = oldest_timestamp.date().isoformat()
        for symbol, timestamp in usable_timestamps.items():
            if timestamp < newest_timestamp:
                details[symbol] = {
                    **details[symbol],
                    "status": LocalDataHealthStatus.STALE,
                    "reason": f"local data lags freshest date {latest_local_data_date}",
                }

    return details, usable_symbols, latest_local_data_date, oldest_local_data_date


def _preflight_config_payload(config: CliRuntimeConfig) -> dict[str, object]:
    payload = _runtime_config_payload(config)
    errors: list[str] = []
    warnings: list[str] = []

    if config.removed_duplicate_symbols:
        warnings.append(
            "Removed duplicate symbol input(s): "
            + ", ".join(config.removed_duplicate_symbols)
        )
    if config.allow_unsafe_checkpoint_loading:
        warnings.append("unsafe checkpoint loading enabled; only use trusted checkpoint files")
    if config.ignored_symbol_inputs:
        warnings.append(
            "Ignored blank symbol input(s): "
            + ", ".join(config.ignored_symbol_inputs)
        )

    if config.backtest_days < 1:
        errors.append("--backtest-days must be at least 1")
    if config.backtest_starting_cash <= 0:
        errors.append("--backtest-starting-cash must be positive")
    if not math.isfinite(float(config.allocation_pct)):
        errors.append("--allocation-pct must be finite")
    elif float(config.allocation_pct) < 0.0:
        errors.append("--allocation-pct must be >= 0")
    if config.backtest_buying_power_multiplier <= 0:
        errors.append("--backtest-buying-power-multiplier must be positive")
    if not math.isfinite(float(config.backtest_entry_offset_bps)):
        errors.append("--backtest-entry-offset-bps must be finite")
    if not math.isfinite(float(config.backtest_exit_offset_bps)):
        errors.append("--backtest-exit-offset-bps must be finite")
    if config.multi_position < 0:
        errors.append("--multi-position must be >= 0")
    if config.multi_position == 1:
        errors.append("--multi-position must be 0 or at least 2")
    if not 0.0 <= float(config.multi_position_min_prob_ratio) <= 1.0:
        errors.append("--multi-position-min-prob-ratio must be between 0 and 1")
    if not 0.0 <= float(config.min_open_confidence) <= 1.0:
        errors.append("--min-open-confidence must be between 0 and 1")
    if config.compare_server_parity and not config.backtest:
        errors.append("--compare-server-parity requires --backtest")
    if config.compare_server_parity and (
        float(config.backtest_entry_offset_bps) != 0.0
        or float(config.backtest_exit_offset_bps) != 0.0
    ):
        errors.append("--compare-server-parity does not support custom backtest entry/exit offsets")
    if config.backtest and not config.paper:
        errors.append("--backtest is local-only; omit --live")
    if payload["missing_checkpoints"]:
        errors.append(
            "Missing checkpoint file(s): "
            + ", ".join(str(path) for path in payload["missing_checkpoints"])
        )
        payload["checkpoint_load"] = {
            "ok": False,
            "error": "checkpoint files are missing",
            "primary": None,
            "extras": [],
        }
    elif errors:
        payload["checkpoint_load"] = {
            "ok": None,
            "error": None,
            "primary": None,
            "extras": [],
            "skipped": True,
        }
    else:
        checkpoint_load = _checkpoint_load_diagnostics(config)
        payload["checkpoint_load"] = checkpoint_load
        if not checkpoint_load["ok"]:
            errors.append(f"Checkpoint load failed: {checkpoint_load['error']}")

    local_data_required = bool(config.backtest or config.data_source == "local")
    missing_local_symbol_files: list[str] = []
    local_data_dir_context: LocalDataDirContext | None = None
    resolved_local_data_dir: Path | None = None
    resolved_local_data_dir_exists = False
    symbol_details: dict[str, StockLocalSymbolDetail] = {}
    usable_symbols: list[str] = []
    usable_symbol_count = 0
    latest_local_data_date: str | None = None
    oldest_local_data_date: str | None = None
    stale_symbol_data: dict[str, str] = {}
    status_counts: LocalDataStatusCounts = {
        "usable": 0,
        "stale": 0,
        "missing": 0,
        "invalid": 0,
    }
    if local_data_required:
        local_data_dir_context = _local_data_dir_context(config.data_dir, config.symbols)
        resolved_local_data_dir = Path(local_data_dir_context["resolved_local_data_dir"])
        resolved_local_data_dir_exists = _local_data_dir_exists(resolved_local_data_dir)
        if not resolved_local_data_dir_exists:
            errors.append(f"Local data directory does not exist: {resolved_local_data_dir}")
            (
                symbol_details,
                usable_symbols,
                latest_local_data_date,
                oldest_local_data_date,
            ) = _build_local_symbol_details(
                symbols=config.symbols,
                resolved_local_data_dir=resolved_local_data_dir,
            )
            usable_symbol_count = len(usable_symbols)
            status_counts = local_data_status_counts(symbol_details)
            missing_local_symbol_files = [
                detail["file_path"]
                for detail in symbol_details.values()
                if detail["status"] == "missing"
            ]
        else:
            (
                symbol_details,
                usable_symbols,
                latest_local_data_date,
                oldest_local_data_date,
            ) = _build_local_symbol_details(
                symbols=config.symbols,
                resolved_local_data_dir=resolved_local_data_dir,
            )
            usable_symbol_count = len(usable_symbols)
            stale_symbol_data = {
                symbol: detail["local_data_date"]
                for symbol, detail in symbol_details.items()
                if detail["status"] == "stale" and detail["local_data_date"] is not None
            }
            status_counts = local_data_status_counts(symbol_details)
            missing_local_symbol_files = [
                detail["file_path"]
                for detail in symbol_details.values()
                if detail["status"] == "missing"
            ]
            if missing_local_symbol_files:
                errors.append(
                    "Missing local daily data file(s): "
                    + ", ".join(missing_local_symbol_files)
                )
            invalid_local_symbol_files = [
                detail["file_path"]
                for detail in symbol_details.values()
                if detail["status"] == "invalid"
            ]
            if invalid_local_symbol_files:
                errors.append(
                    "Unreadable local daily data file(s): "
                    + ", ".join(invalid_local_symbol_files)
                )
            if stale_symbol_data and latest_local_data_date is not None:
                stale_symbol_label = (
                    "symbol" if len(stale_symbol_data) == 1 else "symbols"
                )
                warnings.append(
                    "local daily CSV data for "
                    f"{len(stale_symbol_data)} {stale_symbol_label} "
                    f"lags freshest date {latest_local_data_date}"
                )
    payload["local_data_required"] = local_data_required
    payload["data_dir_exists"] = resolved_local_data_dir_exists
    if local_data_dir_context is not None:
        payload["requested_local_data_dir"] = local_data_dir_context["requested_local_data_dir"]
        payload["resolved_local_data_dir"] = local_data_dir_context["resolved_local_data_dir"]
        payload["resolved_local_data_dir_source"] = local_data_dir_context["resolved_local_data_dir_source"]
    else:
        payload["resolved_local_data_dir"] = None
        payload["resolved_local_data_dir_source"] = None
    payload["missing_local_symbol_files"] = missing_local_symbol_files
    payload["usable_symbols"] = usable_symbols
    payload["usable_symbol_count"] = usable_symbol_count
    payload["latest_local_data_date"] = latest_local_data_date
    payload["oldest_local_data_date"] = oldest_local_data_date
    payload["stale_symbol_data"] = stale_symbol_data
    payload["local_data_status_counts"] = status_counts
    payload["symbol_details"] = symbol_details

    alpaca_required = bool(not config.backtest and config.data_source == "alpaca")
    alpaca_missing_env: list[str] = []
    alpaca_credentials_error: str | None = None
    if alpaca_required:
        mode_label = "paper" if config.paper else "live"
        try:
            key_id, secret = _resolve_alpaca_credentials(paper=config.paper)
        except RuntimeError as exc:
            alpaca_credentials_error = str(exc)
            errors.append(alpaca_credentials_error)
        else:
            if _looks_placeholder(key_id):
                alpaca_missing_env.append("ALP_KEY_ID_PAPER" if config.paper else "ALP_KEY_ID_PROD")
            if _looks_placeholder(secret):
                alpaca_missing_env.append("ALP_SECRET_KEY_PAPER" if config.paper else "ALP_SECRET_KEY_PROD")
            if alpaca_missing_env:
                errors.append(
                    f"Missing Alpaca credential env var(s) for {mode_label} mode: "
                    + ", ".join(alpaca_missing_env)
                )
    payload["alpaca_required"] = alpaca_required
    payload["alpaca_missing_env"] = alpaca_missing_env
    payload["alpaca_credentials_error"] = alpaca_credentials_error
    payload["alpaca_credentials_configured"] = (
        not alpaca_required or (not alpaca_missing_env and alpaca_credentials_error is None)
    )

    trading_server_required = bool(not config.backtest and config.execution_backend == "trading_server")
    resolved_server_url = _resolved_trading_server_url(config)
    server_url_details = config.resolved_server_url_details
    payload["trading_server_required"] = trading_server_required
    payload["resolved_server_url"] = resolved_server_url
    if trading_server_required:
        if server_url_details["security"] == "invalid":
            errors.append(f"Invalid trading server URL: {resolved_server_url}")
        elif not config.paper and not is_secure_or_loopback_trading_server_url(resolved_server_url):
            errors.append(
                "Live trading_server requires an https URL unless it targets loopback: "
                + resolved_server_url
            )
        if not str(config.server_account).strip():
            errors.append("--server-account must not be empty when using trading_server")
        if not str(config.server_bot_id).strip():
            errors.append("--server-bot-id must not be empty when using trading_server")
        if not config.paper and config.server_url_source == "default":
            warnings.append(
                "--live with trading_server is using the built-in loopback default server URL; "
                "pass --server-url explicitly or set TRADING_SERVER_URL if the server is remote"
            )
        if not config.paper and config.server_account == DEFAULT_SERVER_PAPER_ACCOUNT:
            warnings.append(
                "--live with trading_server is still using the paper default server account; "
                "pass --server-account explicitly"
            )

    payload["ready"] = not errors
    payload["errors"] = errors
    payload["warnings"] = warnings
    payload["next_steps"] = _preflight_next_steps(payload)
    return payload

def _format_checkpoint_load_lines(payload: Mapping[str, object]) -> list[str]:
    checkpoint_load = payload.get("checkpoint_load")
    if not isinstance(checkpoint_load, Mapping):
        return []

    lines = [
        "Checkpoint load:",
    ]
    primary = checkpoint_load.get("primary")
    if isinstance(primary, Mapping):
        details: list[str] = []
        arch = str(primary.get("arch") or "").strip()
        if arch:
            details.append(f"arch={arch}")
        feature_schema = str(primary.get("feature_schema") or "").strip()
        if feature_schema:
            details.append(f"schema={feature_schema}")
        feature_dimension = primary.get("feature_dimension")
        if isinstance(feature_dimension, int):
            details.append(f"feature_dim={feature_dimension}")
        if details:
            lines.append("- primary: " + ", ".join(details))
    extras = checkpoint_load.get("extras")
    if isinstance(extras, list):
        extra_classes = [
            str(extra.get("class"))
            for extra in extras
            if isinstance(extra, Mapping) and str(extra.get("class") or "").strip()
        ]
        if extra_classes:
            lines.append("- ensemble extras: " + ", ".join(extra_classes))
    error = checkpoint_load.get("error")
    if isinstance(error, str) and error.strip() and len(lines) == 1:
        lines.append(f"- error: {error}")
    if len(lines) == 1:
        return []
    return lines


def _compact_preflight_summary_field(value: object) -> str:
    return " ".join(str(value).split())


def _format_runtime_preflight_summary(payload: Mapping[str, object]) -> str:
    ready = bool(payload.get("ready"))
    parts = ["READY" if ready else "NOT READY"]

    strategy_mode = _compact_preflight_summary_field(payload.get("strategy_mode") or "")
    if strategy_mode:
        parts.append(strategy_mode)

    symbol_count = int(payload.get("symbol_count") or 0)
    usable_symbol_count = payload.get("usable_symbol_count")
    if bool(payload.get("local_data_required")) and isinstance(usable_symbol_count, int):
        parts.append(f"symbols={usable_symbol_count}/{symbol_count} usable")
    elif symbol_count > 0:
        parts.append(f"symbols={symbol_count}")

    checkpoint_load = payload.get("checkpoint_load")
    if isinstance(checkpoint_load, Mapping):
        primary = checkpoint_load.get("primary")
        if isinstance(primary, Mapping):
            checkpoint_parts: list[str] = []
            arch = _compact_preflight_summary_field(primary.get("arch") or "")
            if arch:
                checkpoint_parts.append(arch)
            feature_schema = _compact_preflight_summary_field(primary.get("feature_schema") or "")
            if feature_schema:
                checkpoint_parts.append(f"schema={feature_schema}")
            feature_dimension = primary.get("feature_dimension")
            if isinstance(feature_dimension, int):
                checkpoint_parts.append(f"dim={feature_dimension}")
            if checkpoint_parts:
                parts.append("checkpoint=" + " ".join(checkpoint_parts))
        elif _compact_preflight_summary_field(checkpoint_load.get("error") or ""):
            parts.append("checkpoint=error")

    latest_local_data_date = _compact_preflight_summary_field(payload.get("latest_local_data_date") or "")
    if latest_local_data_date:
        parts.append(f"latest_local_data={latest_local_data_date}")

    local_data_status_counts = payload.get("local_data_status_counts")
    if isinstance(local_data_status_counts, Mapping):
        issue_parts: list[str] = []
        for status in ("stale", "missing", "invalid"):
            count = local_data_status_counts.get(status)
            if isinstance(count, int) and count > 0:
                issue_parts.append(f"{status}:{count}")
        if issue_parts:
            parts.append("local_issues=" + ",".join(issue_parts))

    raw_warnings = payload.get("warnings")
    warnings = (
        [
            _compact_preflight_summary_field(item)
            for item in raw_warnings
            if _compact_preflight_summary_field(item)
        ]
        if isinstance(raw_warnings, list)
        else []
    )
    raw_errors = payload.get("errors")
    errors = (
        [
            _compact_preflight_summary_field(item)
            for item in raw_errors
            if _compact_preflight_summary_field(item)
        ]
        if isinstance(raw_errors, list)
        else []
    )
    if warnings:
        parts.append(f"warnings={len(warnings)}")
    if errors:
        parts.append(f"errors={len(errors)}")
        parts.append(f"first_error={errors[0]}")

    return " | ".join(parts)


def _format_runtime_preflight_failure(payload: dict[str, object]) -> str:
    lines = [
        "Daily stock RL setup is not ready.",
        str(payload.get("summary") or ""),
    ]
    strategy_mode = str(payload.get("strategy_mode") or "").strip()
    position_capacity = int(payload.get("position_capacity") or 0)
    if strategy_mode:
        lines.append(f"Strategy mode: {strategy_mode}")
    if position_capacity > 0:
        lines.append(f"Position capacity: {position_capacity}")
    symbol_source_label = str(payload.get("symbol_source_label") or "").strip()
    symbol_preview_text = str(payload.get("symbol_preview_text") or "").strip()
    if symbol_source_label:
        lines.append(f"Symbol source: {symbol_source_label}")
    if symbol_preview_text:
        lines.append(f"Symbols: {symbol_preview_text}")
    lines.extend(_format_local_data_resolution_lines(payload))
    errors = [str(item) for item in payload.get("errors", [])]
    warnings = [str(item) for item in payload.get("warnings", [])]
    next_steps = [str(item) for item in payload.get("next_steps", [])]
    if errors:
        lines.append("Errors:")
        lines.extend(f"- {item}" for item in errors)
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {item}" for item in warnings)
    lines.extend(_format_checkpoint_load_lines(payload))
    symbol_details = payload.get("symbol_details")
    if isinstance(symbol_details, dict) and symbol_details:
        latest_local_data_date = str(payload.get("latest_local_data_date") or "").strip() or None
        lines.extend(
            format_local_data_health_lines(
                symbol_details=symbol_details,
                usable_symbol_count=int(payload.get("usable_symbol_count") or 0),
                latest_local_data_date=latest_local_data_date,
            )
        )
    if bool(payload.get("trading_server_required")):
        resolved_server_url = str(
            payload.get("resolved_server_url") or payload.get("server_url") or ""
        ).strip()
        lines.append("Trading server config:")
        if resolved_server_url:
            lines.append(f"- resolved server URL: {resolved_server_url}")
        lines.append(
            "- server URL source: "
            + _trading_server_url_source_label(payload.get("server_url_source"))
        )
        lines.append(
            "- server URL security: "
            + _trading_server_url_security_label(payload.get("server_url_security"))
        )
    if next_steps:
        lines.append("Next steps:")
        lines.extend(f"- {item}" for item in next_steps)
    return "\n".join(line for line in lines if line)


def _format_runtime_preflight_ready(payload: dict[str, object]) -> str:
    lines = [
        "Daily stock RL setup is ready.",
        str(payload.get("summary") or ""),
    ]
    strategy_mode = str(payload.get("strategy_mode") or "").strip()
    position_capacity = int(payload.get("position_capacity") or 0)
    if strategy_mode:
        lines.append(f"Strategy mode: {strategy_mode}")
    if position_capacity > 0:
        lines.append(f"Position capacity: {position_capacity}")
    symbol_source_label = str(payload.get("symbol_source_label") or "").strip()
    symbol_preview_text = str(payload.get("symbol_preview_text") or "").strip()
    if symbol_source_label:
        lines.append(f"Symbol source: {symbol_source_label}")
    if symbol_preview_text:
        lines.append(f"Symbols: {symbol_preview_text}")
    lines.extend(_format_local_data_resolution_lines(payload))
    warnings = [str(item) for item in payload.get("warnings", [])]
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {item}" for item in warnings)
    lines.extend(_format_checkpoint_load_lines(payload))
    symbol_details = payload.get("symbol_details")
    if isinstance(symbol_details, dict) and symbol_details:
        latest_local_data_date = str(payload.get("latest_local_data_date") or "").strip() or None
        lines.extend(
            format_local_data_health_lines(
                symbol_details=symbol_details,
                usable_symbol_count=int(payload.get("usable_symbol_count") or 0),
                latest_local_data_date=latest_local_data_date,
            )
        )
    if bool(payload.get("trading_server_required")):
        resolved_server_url = str(
            payload.get("resolved_server_url") or payload.get("server_url") or ""
        ).strip()
        lines.append("Trading server config:")
        if resolved_server_url:
            lines.append(f"- resolved server URL: {resolved_server_url}")
        lines.append(
            "- server URL source: "
            + _trading_server_url_source_label(payload.get("server_url_source"))
        )
        lines.append(
            "- server URL security: "
            + _trading_server_url_security_label(payload.get("server_url_security"))
        )

    safe_command = str(payload.get("safe_command_preview") or "").strip()
    run_command = str(payload.get("run_command_preview") or "").strip()
    check_command = str(payload.get("check_command_preview") or "").strip()
    check_text_command = str(payload.get("check_text_command_preview") or "").strip()
    lines.append("Suggested commands:")
    if safe_command:
        lines.append(f"- dry run: {safe_command}")
    if run_command and run_command != safe_command:
        lines.append(f"- configured runtime: {run_command}")

    next_steps = [str(item) for item in payload.get("next_steps", [])]
    command_steps = {
        f"Run a safe dry run: {safe_command}" if safe_command else "",
        f"Start the configured runtime: {run_command}" if run_command else "",
        f"Re-run preflight later: {check_command}" if check_command else "",
        f"Re-run text preflight later: {check_text_command}" if check_text_command else "",
    }
    command_steps.discard("")
    additional_steps = [item for item in next_steps if item not in command_steps]
    if additional_steps:
        lines.append("Additional next steps:")
        lines.extend(f"- {item}" for item in additional_steps)
    return "\n".join(line for line in lines if line)


def _format_local_data_resolution_lines(payload: Mapping[str, object]) -> list[str]:
    if not bool(payload.get("local_data_required")):
        return []
    requested_data_dir = str(payload.get("data_dir") or "").strip()
    local_data_dir_context = _coerce_local_data_dir_context(payload)
    if local_data_dir_context is None:
        requested_local_data_dir: str | None = None
        resolved_local_data_dir = str(payload.get("resolved_local_data_dir") or "").strip()
        resolved_local_data_dir_source = _coerce_local_data_dir_resolution_source(
            payload.get("resolved_local_data_dir_source")
        )
    else:
        requested_local_data_dir = local_data_dir_context["requested_local_data_dir"]
        resolved_local_data_dir = local_data_dir_context["resolved_local_data_dir"]
        resolved_local_data_dir_source = local_data_dir_context["resolved_local_data_dir_source"]
    if not resolved_local_data_dir:
        return []
    lines: list[str] = []
    if requested_local_data_dir:
        if requested_local_data_dir != resolved_local_data_dir:
            lines.append(f"Requested local data dir: {requested_local_data_dir}")
    elif requested_data_dir:
        requested_path = (REPO / Path(requested_data_dir).expanduser()).resolve()
        if str(requested_path) != resolved_local_data_dir:
            lines.append(f"Requested local data dir: {requested_path}")
    if bool(payload.get("data_dir_exists")):
        lines.append(f"Resolved local data dir: {resolved_local_data_dir}")
    else:
        lines.append(f"Resolved local data dir: {resolved_local_data_dir} (missing)")
    if resolved_local_data_dir_source == LocalDataDirResolutionSource.NESTED_TRAIN:
        lines.append("Local data dir source: auto-selected nested train/ directory")
    elif resolved_local_data_dir_source == LocalDataDirResolutionSource.REQUESTED:
        lines.append("Local data dir source: requested directory")
    return lines


def _exception_notes(exc: BaseException) -> list[str]:
    notes = getattr(exc, "__notes__", None)
    if not notes:
        return []
    return [str(note) for note in notes if str(note).strip()]


def _format_run_once_failure_message(config: CliRuntimeConfig, exc: BaseException) -> str:
    stage_label: str | None = None
    context_label: str | None = None
    for note in _exception_notes(exc):
        if note.startswith("run_once stage: "):
            stage_label = note.removeprefix("run_once stage: ").replace("_", " ")
        elif note.startswith("run_once context: ") and context_label is None:
            context_label = note.removeprefix("run_once context: ")

    if stage_label:
        lines = [f"Daily stock RL run failed during {stage_label}."]
    else:
        lines = ["Daily stock RL run failed."]
    lines.append(f"Error: {type(exc).__name__}: {exc}")
    if config.backtest or config.data_source == "local":
        try:
            local_data_dir_context = _local_data_dir_context(config.data_dir, config.symbols)
        except Exception as local_data_context_exc:
            lines.append(
                "Local data dir context unavailable: "
                + f"{type(local_data_context_exc).__name__}: {local_data_context_exc}"
            )
        else:
            lines.extend(
                _format_local_data_resolution_lines(
                    {
                        "local_data_required": True,
                        "data_dir": config.data_dir,
                        "data_dir_exists": _local_data_dir_exists(
                            Path(local_data_dir_context["resolved_local_data_dir"])
                        ),
                        "resolved_local_data_dir": local_data_dir_context["resolved_local_data_dir"],
                        "resolved_local_data_dir_source": local_data_dir_context["resolved_local_data_dir_source"],
                    }
                )
            )
    if context_label:
        lines.append(f"Latest context: {context_label}")
    lines.append(
        "Check config: "
        + config.command_preview(check_config=True, check_config_text=True)
    )

    safe_command = config.command_preview(force_dry_run=True)
    run_command = config.command_preview()
    if safe_command and safe_command != run_command:
        lines.append(f"Try a safe dry run: {safe_command}")

    extra_notes = [
        note
        for note in _exception_notes(exc)
        if not note.startswith("run_once stage: ")
        and not note.startswith("run_once context: ")
    ]
    if extra_notes:
        lines.append("Additional context:")
        lines.extend(f"- {note}" for note in extra_notes)
    return "\n".join(lines)


def run_backtest_via_trading_server(
    *,
    checkpoint: str,
    symbols: Iterable[str],
    data_dir: str,
    days: int,
    allocation_pct: float = 100.0,
    allocation_sizing_mode: StockAllocationSizingMode = DEFAULT_ALLOCATION_SIZING_MODE,
    starting_cash: float = DEFAULT_BACKTEST_STARTING_CASH,
    buying_power_multiplier: float = DEFAULT_BACKTEST_BUYING_POWER_MULTIPLIER,
    multi_position: int = DEFAULT_MULTI_POSITION,
    multi_position_min_prob_ratio: float = DEFAULT_MULTI_POSITION_MIN_PROB_RATIO,
    account: str = DEFAULT_BACKTEST_SERVER_ACCOUNT,
    bot_id: str = DEFAULT_BACKTEST_SERVER_BOT_ID,
    extra_checkpoints: Optional[list[str]] = None,
    allow_unsafe_checkpoint_loading: bool = False,
) -> dict[str, float]:
    if starting_cash <= 0:
        raise ValueError("starting_cash must be positive")
    if buying_power_multiplier <= 0:
        raise ValueError("buying_power_multiplier must be positive")
    prepared = _prepare_daily_backtest_data(
        checkpoint=checkpoint,
        symbols=symbols,
        data_dir=data_dir,
        days=days,
        extra_checkpoints=extra_checkpoints,
        allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
    )
    return _run_backtest_via_trading_server_with_prepared_data(
        prepared=prepared,
        symbols=symbols,
        days=days,
        allocation_pct=allocation_pct,
        allocation_sizing_mode=allocation_sizing_mode,
        starting_cash=starting_cash,
        buying_power_multiplier=buying_power_multiplier,
        multi_position=multi_position,
        multi_position_min_prob_ratio=multi_position_min_prob_ratio,
        account=account,
        bot_id=bot_id,
        checkpoint=checkpoint,
        extra_checkpoints=extra_checkpoints,
        allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
    )


def _run_backtest_via_trading_server_with_prepared_data(
    *,
    prepared: PreparedDailyBacktestData,
    symbols: Iterable[str],
    days: int,
    allocation_pct: float = 100.0,
    allocation_sizing_mode: StockAllocationSizingMode = DEFAULT_ALLOCATION_SIZING_MODE,
    starting_cash: float = DEFAULT_BACKTEST_STARTING_CASH,
    buying_power_multiplier: float = DEFAULT_BACKTEST_BUYING_POWER_MULTIPLIER,
    multi_position: int = DEFAULT_MULTI_POSITION,
    multi_position_min_prob_ratio: float = DEFAULT_MULTI_POSITION_MIN_PROB_RATIO,
    account: str = DEFAULT_BACKTEST_SERVER_ACCOUNT,
    bot_id: str = DEFAULT_BACKTEST_SERVER_BOT_ID,
    checkpoint: str = "",
    extra_checkpoints: Optional[list[str]] = None,
    allow_unsafe_checkpoint_loading: bool = False,
) -> dict[str, float]:
    if starting_cash <= 0:
        raise ValueError("starting_cash must be positive")
    if buying_power_multiplier <= 0:
        raise ValueError("buying_power_multiplier must be positive")
    from src.trading_server.server import TradingServerEngine

    indexed = prepared.indexed
    min_len = prepared.min_len
    start = prepared.start
    trader = _clone_daily_trader_template(prepared.trader_template)
    extra_policies = prepared.extra_policies
    symbols_order = prepared.symbols
    feature_cube = prepared.feature_cube
    close_matrix = prepared.close_matrix
    timestamps = prepared.timestamps

    current_state = StrategyState()
    current_quotes: dict[str, TradingServerQuotePayload] = {}
    current_now = datetime.now(timezone.utc)

    def _quote_provider(symbol: str) -> TradingServerQuotePayload | None:
        return current_quotes.get(str(symbol).upper())

    with TemporaryDirectory(prefix="daily_stock_server_bt_") as tmpdir:
        tmp_root = Path(tmpdir)
        registry_path = tmp_root / "registry.json"
        registry_path.write_text(
            json.dumps(
                {
                    "accounts": {
                        account: {
                            "mode": "paper",
                            "allowed_bot_id": bot_id,
                            "starting_cash": float(starting_cash),
                            "paper_buying_power_multiplier": float(buying_power_multiplier),
                            "symbols": [str(symbol).upper() for symbol in symbols],
                            "sell_loss_cooldown_seconds": 0,
                            "min_sell_markup_pct": 0.0,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        engine = TradingServerEngine(
            registry_path=registry_path,
            state_dir=tmp_root / "state",
            quote_provider=_quote_provider,
            now_fn=lambda: current_now,
        )
        server_client = InMemoryTradingServerClient(
            engine=engine,
            account=account,
            bot_id=bot_id,
            execution_mode="paper",
            session_id=DEFAULT_BACKTEST_SERVER_SESSION_ID,
        )
        server_client.claim_writer(ttl_seconds=DEFAULT_BACKTEST_WRITER_TTL_SECONDS)

        equity_curve: list[float] = []
        closes_last: dict[str, float] = {}
        for idx in range(start, min_len):
            current_now = timestamps[idx]
            prices = {
                symbol: float(price)
                for symbol, price in zip(symbols_order, close_matrix[idx], strict=True)
            }
            closes_last = prices
            current_quotes = {
                symbol: {
                    "symbol": symbol,
                    "bid_price": price,
                    "ask_price": price,
                    "last_price": price,
                    "as_of": current_now.isoformat(),
                }
                for symbol, price in prices.items()
            }
            server_client.refresh_prices(symbols=prices.keys())
            snapshot = server_client.get_account()
            equity = server_equity(snapshot, prices)
            equity_curve.append(equity)
            features = feature_cube[idx]

            if multi_position > 1:
                trader.cash = equity
                trader.current_position = None
                trader.position_qty = 0.0
                trader.entry_price = 0.0
                signals = _ensemble_top_k_signals(
                    trader,
                    extra_policies,
                    features,
                    prices,
                    k=multi_position,
                    min_prob_ratio=multi_position_min_prob_ratio,
                )
                max_total_allocation_pct = max(
                    0.0,
                    100.0 * BUYING_POWER_USAGE_CAP * float(buying_power_multiplier),
                )
                total_allocation_pct = min(max(0.0, float(allocation_pct)), max_total_allocation_pct)
                execute_multi_position_signals_with_trading_server(
                    signals,
                    server_client=server_client,
                    quotes=prices,
                    symbols=symbols,
                    total_allocation_pct=total_allocation_pct,
                    dry_run=False,
                )
                trader.step_day()
                continue

            portfolio = server_portfolio_context(
                snapshot=snapshot,
                state=current_state,
                quotes=prices,
                now=current_now,
            )
            _apply_portfolio_context_to_trader(trader, portfolio=portfolio)
            if extra_policies:
                signal = _ensemble_softmax_signal(trader, extra_policies, features, prices)
            elif callable(getattr(trader, "get_signal", None)):
                signal = cast(TradingSignal, trader.get_signal(features, prices))
            else:
                signal = build_signal(
                    checkpoint,
                    {symbol: frame.iloc[: idx + 1].reset_index() for symbol, frame in indexed.items()},
                    portfolio=portfolio,
                    extra_checkpoints=extra_checkpoints,
                    allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
                )[0]
            execute_signal_with_trading_server(
                signal,
                server_client=server_client,
                quotes=prices,
                state=current_state,
                symbols=symbols,
                allocation_pct=allocation_pct,
                allocation_sizing_mode=allocation_sizing_mode,
                dry_run=False,
                now=current_now,
                min_open_confidence=DEFAULT_MIN_OPEN_CONFIDENCE,
            )

        final_snapshot = server_client.get_account()
        order_payload = server_client.get_orders(include_history=True)
        order_history = order_payload.get("order_history", [])
        final_equity = server_equity(final_snapshot, closes_last)
        curve = np.asarray(equity_curve + [final_equity], dtype=np.float64)
        total_return = float(curve[-1] / curve[0] - 1.0)
        daily_returns = np.diff(curve) / np.clip(curve[:-1], 1e-8, None)
        downside = daily_returns[daily_returns < 0.0]
        downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 1e-8
        sortino = float(np.mean(daily_returns) / downside_dev * np.sqrt(252.0)) if len(daily_returns) else 0.0
        max_dd = float(np.min(curve / np.maximum.accumulate(curve) - 1.0))
        annualized = float((1.0 + total_return) ** (252.0 / max(1, days)) - 1.0)
        if multi_position > 1:
            # Legacy multi-position backtests count each rebalance order as a trade.
            trade_count = float(len(order_history))
        else:
            # Legacy single-position backtests count realized exits, not opens.
            trade_count = float(
                sum(1 for order in order_history if str(order.get("side", "")).lower() == "sell")
            )
        results = {
            "total_return": total_return,
            "annualized_return": annualized,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "trades": trade_count,
            "orders": float(len(order_history)),
        }
        logger.info("Trading-server paper backtest: %s", json.dumps(results, sort_keys=True))
        return results


def run_backtest_variant_matrix_via_trading_server(
    *,
    checkpoint: str,
    symbols: Iterable[str],
    data_dir: str,
    days: int,
    variants: Sequence[BacktestVariantSpec],
    starting_cash: float = DEFAULT_BACKTEST_STARTING_CASH,
    extra_checkpoints: Optional[list[str]] = None,
    allow_unsafe_checkpoint_loading: bool = False,
    account_prefix: str = DEFAULT_BACKTEST_SERVER_ACCOUNT,
    bot_id_prefix: str = DEFAULT_BACKTEST_SERVER_BOT_ID,
) -> list[dict[str, float | int | str]]:
    prepared = _prepare_daily_backtest_data(
        checkpoint=checkpoint,
        symbols=symbols,
        data_dir=data_dir,
        days=days,
        extra_checkpoints=extra_checkpoints,
        allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
    )
    return _run_backtest_variant_matrix_via_trading_server_with_prepared_data(
        prepared=prepared,
        symbols=symbols,
        days=days,
        variants=variants,
        starting_cash=starting_cash,
        extra_checkpoints=extra_checkpoints,
        allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
        account_prefix=account_prefix,
        bot_id_prefix=bot_id_prefix,
        checkpoint=checkpoint,
    )


def _run_backtest_variant_matrix_via_trading_server_with_prepared_data(
    *,
    prepared: PreparedDailyBacktestData,
    symbols: Iterable[str],
    days: int,
    variants: Sequence[BacktestVariantSpec],
    starting_cash: float = DEFAULT_BACKTEST_STARTING_CASH,
    extra_checkpoints: Optional[list[str]] = None,
    allow_unsafe_checkpoint_loading: bool = False,
    account_prefix: str = DEFAULT_BACKTEST_SERVER_ACCOUNT,
    bot_id_prefix: str = DEFAULT_BACKTEST_SERVER_BOT_ID,
    checkpoint: str = "",
) -> list[dict[str, float | int | str]]:
    results: list[dict[str, float | int | str]] = []
    for idx, variant in enumerate(variants):
        metrics = _run_backtest_via_trading_server_with_prepared_data(
            prepared=prepared,
            symbols=symbols,
            days=days,
            allocation_pct=variant.allocation_pct,
            allocation_sizing_mode=variant.allocation_sizing_mode,
            starting_cash=starting_cash,
            buying_power_multiplier=variant.buying_power_multiplier,
            multi_position=variant.multi_position,
            multi_position_min_prob_ratio=variant.multi_position_min_prob_ratio,
            account=f"{account_prefix}_{idx}",
            bot_id=f"{bot_id_prefix}_{idx}",
            checkpoint=checkpoint,
            extra_checkpoints=extra_checkpoints,
            allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
        )
        results.append(
            {
                "name": variant.name,
                "allocation_pct": float(variant.allocation_pct),
                "allocation_sizing_mode": str(variant.allocation_sizing_mode),
                "multi_position": int(variant.multi_position),
                "multi_position_min_prob_ratio": float(variant.multi_position_min_prob_ratio),
                "buying_power_multiplier": float(variant.buying_power_multiplier),
                **metrics,
            }
        )
    return results


def run_backtest_multi_window_variant_matrix_via_trading_server(
    *,
    checkpoint: str,
    symbols: Iterable[str],
    data_dir: str,
    days_list: Sequence[int],
    variants: Sequence[BacktestVariantSpec],
    starting_cash: float = DEFAULT_BACKTEST_STARTING_CASH,
    extra_checkpoints: Optional[list[str]] = None,
    allow_unsafe_checkpoint_loading: bool = False,
    account_prefix: str = DEFAULT_BACKTEST_SERVER_ACCOUNT,
    bot_id_prefix: str = DEFAULT_BACKTEST_SERVER_BOT_ID,
) -> list[dict[str, object]]:
    resolved_days = list(dict.fromkeys(int(day) for day in days_list if int(day) > 0))
    if not resolved_days:
        raise ValueError("days_list must contain at least one positive backtest window")
    prepared = _prepare_daily_backtest_data(
        checkpoint=checkpoint,
        symbols=symbols,
        data_dir=data_dir,
        days=max(resolved_days),
        extra_checkpoints=extra_checkpoints,
        allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
    )
    return [
        {
            "days": int(days),
            "results": _run_backtest_variant_matrix_via_trading_server_with_prepared_data(
                prepared=prepared,
                symbols=symbols,
                days=int(days),
                variants=variants,
                starting_cash=starting_cash,
                extra_checkpoints=extra_checkpoints,
                allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
                account_prefix=f"{account_prefix}_{int(days)}",
                bot_id_prefix=f"{bot_id_prefix}_{int(days)}",
                checkpoint=checkpoint,
            ),
        }
        for days in resolved_days
    ]


def compare_backtest_to_trading_server(
    *,
    checkpoint: str,
    symbols: Iterable[str],
    data_dir: str,
    days: int,
    allocation_pct: float = 100.0,
    allocation_sizing_mode: StockAllocationSizingMode = DEFAULT_ALLOCATION_SIZING_MODE,
    starting_cash: float = DEFAULT_BACKTEST_STARTING_CASH,
    multi_position: int = DEFAULT_MULTI_POSITION,
    multi_position_min_prob_ratio: float = DEFAULT_MULTI_POSITION_MIN_PROB_RATIO,
    extra_checkpoints: Optional[list[str]] = None,
    buying_power_multiplier: float = DEFAULT_BACKTEST_BUYING_POWER_MULTIPLIER,
    allow_unsafe_checkpoint_loading: bool = False,
) -> dict[str, object]:
    legacy = run_backtest(
        checkpoint=checkpoint,
        symbols=symbols,
        data_dir=data_dir,
        days=days,
        allocation_pct=allocation_pct,
        allocation_sizing_mode=allocation_sizing_mode,
        starting_cash=starting_cash,
        multi_position=multi_position,
        multi_position_min_prob_ratio=multi_position_min_prob_ratio,
        extra_checkpoints=extra_checkpoints,
        buying_power_multiplier=buying_power_multiplier,
        allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
    )
    server = run_backtest_via_trading_server(
        checkpoint=checkpoint,
        symbols=symbols,
        data_dir=data_dir,
        days=days,
        allocation_pct=allocation_pct,
        allocation_sizing_mode=allocation_sizing_mode,
        starting_cash=starting_cash,
        multi_position=multi_position,
        multi_position_min_prob_ratio=multi_position_min_prob_ratio,
        extra_checkpoints=extra_checkpoints,
        buying_power_multiplier=buying_power_multiplier,
        allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
    )
    deltas = {
        key: float(server.get(key, 0.0) - legacy.get(key, 0.0))
        for key in ("total_return", "annualized_return", "sortino", "max_drawdown", "trades")
    }
    result = {"legacy": legacy, "server": server, "delta": deltas}
    logger.info("Legacy/server parity summary: %s", json.dumps(result, sort_keys=True))
    return result


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    config = _resolve_runtime_config(args)
    if args.check_config:
        payload = _preflight_config_payload(config)
        print(json.dumps(payload, indent=2, sort_keys=True))
        if args.check_config_summary:
            print(_format_runtime_preflight_summary(payload), file=sys.stderr)
        if args.check_config_text:
            rendered = (
                _format_runtime_preflight_ready(payload)
                if payload["ready"]
                else _format_runtime_preflight_failure(payload)
            )
            print(rendered, file=sys.stderr)
        if not payload["ready"]:
            raise SystemExit(1)
        return
    if args.print_config:
        print(json.dumps(_runtime_config_payload(config), indent=2, sort_keys=True))
        return
    payload = _preflight_config_payload(config)
    if not payload["ready"]:
        print(_format_runtime_preflight_failure(payload), file=sys.stderr)
        raise SystemExit(1)
    _log_runtime_start(config, preflight_payload=payload)

    if config.backtest:
        if config.compare_server_parity:
            compare_backtest_to_trading_server(
                checkpoint=config.checkpoint,
                symbols=config.symbols,
                data_dir=config.data_dir,
                days=config.backtest_days,
                allocation_pct=config.allocation_pct,
                allocation_sizing_mode=config.allocation_sizing_mode,
                starting_cash=config.backtest_starting_cash,
                multi_position=config.multi_position,
                multi_position_min_prob_ratio=config.multi_position_min_prob_ratio,
                extra_checkpoints=config.extra_checkpoints,
                buying_power_multiplier=config.backtest_buying_power_multiplier,
                allow_unsafe_checkpoint_loading=config.allow_unsafe_checkpoint_loading,
            )
            return
        run_backtest(
            checkpoint=config.checkpoint,
            symbols=config.symbols,
            data_dir=config.data_dir,
            days=config.backtest_days,
            allocation_pct=config.allocation_pct,
            allocation_sizing_mode=config.allocation_sizing_mode,
            starting_cash=config.backtest_starting_cash,
            multi_position=config.multi_position,
            multi_position_min_prob_ratio=config.multi_position_min_prob_ratio,
            extra_checkpoints=config.extra_checkpoints,
            buying_power_multiplier=config.backtest_buying_power_multiplier,
            entry_offset_bps=config.backtest_entry_offset_bps,
            exit_offset_bps=config.backtest_exit_offset_bps,
            min_open_confidence=config.min_open_confidence,
            min_open_value_estimate=config.min_open_value_estimate,
            allow_unsafe_checkpoint_loading=config.allow_unsafe_checkpoint_loading,
        )
        return

    account_lock = None
    if not config.paper:
        require_explicit_live_trading_enable("daily-rl-trader")
    if not config.paper and not config.dry_run:
        account_lock = acquire_alpaca_account_lock(
            "daily-rl-trader",
            account_name="alpaca_live_writer",
        )
        logger.info("Acquired Alpaca live writer lock: %s", account_lock.path)

    if config.daemon:
        run_daemon(
            checkpoint=config.checkpoint,
            symbols=config.symbols,
            paper=config.paper,
            allocation_pct=config.allocation_pct,
            allocation_sizing_mode=config.allocation_sizing_mode,
            dry_run=config.dry_run,
            data_dir=config.data_dir,
            extra_checkpoints=config.extra_checkpoints,
            execution_backend=config.execution_backend,
            server_account=config.server_account,
            server_bot_id=config.server_bot_id,
            server_url=config.server_url,
            multi_position=config.multi_position,
            multi_position_min_prob_ratio=config.multi_position_min_prob_ratio,
            min_open_confidence=config.min_open_confidence,
            min_open_value_estimate=config.min_open_value_estimate,
            allow_unsafe_checkpoint_loading=config.allow_unsafe_checkpoint_loading,
            meta_selector=config.meta_selector,
            meta_top_k=config.meta_top_k,
            meta_lookback=config.meta_lookback,
        )
        return

    try:
        payload = run_once(
            checkpoint=config.checkpoint,
            symbols=config.symbols,
            paper=config.paper,
            allocation_pct=config.allocation_pct,
            allocation_sizing_mode=config.allocation_sizing_mode,
            dry_run=config.dry_run,
            data_source=config.data_source,
            data_dir=config.data_dir,
            extra_checkpoints=config.extra_checkpoints,
            execution_backend=config.execution_backend,
            server_account=config.server_account,
            server_bot_id=config.server_bot_id,
            server_url=config.server_url,
            multi_position=config.multi_position,
            multi_position_min_prob_ratio=config.multi_position_min_prob_ratio,
            min_open_confidence=config.min_open_confidence,
            min_open_value_estimate=config.min_open_value_estimate,
            allow_unsafe_checkpoint_loading=config.allow_unsafe_checkpoint_loading,
            meta_selector=config.meta_selector,
            meta_top_k=config.meta_top_k,
            meta_lookback=config.meta_lookback,
        )
    except Exception as exc:
        print(_format_run_once_failure_message(config, exc), file=sys.stderr)
        raise SystemExit(1) from exc
    if config.print_payload:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
