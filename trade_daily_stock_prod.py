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
import re
import shlex
import sys
import tempfile
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Iterable, Literal, Optional, Sequence, TypeAlias, TypedDict, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.inference_daily import DailyPPOTrader, compute_daily_features
from pufferlib_market.export_data_daily import compute_daily_features as compute_daily_feature_history
from pufferlib_market.checkpoint_loader import load_checkpoint_payload
from src.alpaca_account_lock import acquire_alpaca_account_lock, require_explicit_live_trading_enable
from src.local_data_health import format_local_data_health_lines
from src.shared_path_guard import shared_path_guard
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

EASTERN = ZoneInfo("America/New_York")
RUN_AFTER_OPEN_ET = dt_time(hour=9, minute=35)

DEFAULT_SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOG",
    "META",
    "TSLA",
    "SPY",
    "QQQ",
    "PLTR",
    "JPM",
    "V",
    "AMZN",
]
DEFAULT_CHECKPOINT = "pufferlib_market/prod_ensemble/tp10.pt"
# 32-model ensemble stored in prod_ensemble/ (protected from *_screen/ deletion pattern)
# Members: tp10+s15+s36+gamma_995+muon_wd_005+h1024_a40+s1731+gamma995_s2006+s1401+s1726+s1523+s2617+s2033+s2495+s1835+s2827+s2722+s3668+s3411+s4011+s4777+s4080+s4533+s4813+s5045+s5337+s5199+s5019+s6808+s3456+s7159+s6758
# Updated 2026-03-31 — all checkpoints are screen-phase (≤3M steps) or exact-match recoveries
# s2827 added 2026-03-28: +16% delta vs 15-model
# s2722 added 2026-03-29: +6% delta vs 16-model
# s3668 added 2026-03-29: +1.1% delta vs 17-model
# s3411 added 2026-03-29: +1.8% delta vs 18-model — 19-model: 0/111 neg, p10=44.1%
# s4011 added 2026-03-29: +4.4% delta vs 19-model — 20-model: 0/111 neg, p10=48.5%
# s4777 added 2026-03-29: +0.2% delta vs 20-model — 21-model: 0/111 neg, p10=48.7%
# s4080 added 2026-03-29: +0.1% delta vs 21-model — 22-model: 0/111 neg, p10=48.8%
# s4533 added 2026-03-29: +4.2% delta vs 22-model — 23-model: 0/111 neg, p10=52.9%
# s4813 added 2026-03-29: +4.4% delta vs 23-model — 24-model: 0/111 neg, p10=57.4%
# s5045 added 2026-03-29: +1.2% delta vs 24-model — 25-model: 0/111 neg, p10=58.6%
# s5337 added 2026-03-29: +1.8% delta vs 25-model — 26-model: 0/111 neg, p10=60.3%
# s5199 added 2026-03-29: +2.2% delta vs 26-model — 27-model: 0/111 neg, p10=62.6%
# s5019 added 2026-03-29: +0.9% delta vs 27-model — 28-model: 0/111 neg, p10=63.5%
# s6808 added 2026-03-30: +0.7% delta vs 28-model — 29-model: 0/111 neg, p10=64.1%
# s3456 added 2026-03-31: +0.5% delta vs 29-model — 30-model: 0/111 neg, p10=64.6%
# s7159 added 2026-03-31: +0.7% delta vs 30-model — 31-model: 0/111 neg, p10=65.3%
# s6758 added 2026-03-31: +1.0% delta vs 31-model — 32-model: 0/111 neg, med=73.4%, p10=66.2%
# (15-model was: 0/111 neg, med=50.9%, p10=19.2%)
# ENCODER_NORM NOTE: models use encoder_norm; production inference.py applies it correctly
# 33-model bar: 33-model exhaustive p10 >= 66.2% @fill_bps=5 (encoder_norm-correct methodology)
# NOTE: s4009 REJECTED (batch misidentification — actual delta=-25.1%)
# REJECTED: s2655, s2206, resmlp_a40, s28, tp03, s241, s541, s310, stock_ent_05
# REJECTED (high in-sample return = aggressive overfit): s2793, s2815, s2099, s2118, s2247, s2695
# REJECTED against 16-model: s2433/s2831/s2275 (correlated w/ s2827), s2137, s2276, s2279, s2435, s2575
# REJECTED against 17/18/19/20/21/22/23-model: 100+ seeds tested (see batch_new_*.log)
DEFAULT_EXTRA_CHECKPOINTS = [
    "pufferlib_market/prod_ensemble/s15.pt",
    "pufferlib_market/prod_ensemble/s36.pt",
    "pufferlib_market/prod_ensemble/gamma_995.pt",
    "pufferlib_market/prod_ensemble/muon_wd_005.pt",
    "pufferlib_market/prod_ensemble/h1024_a40.pt",
    "pufferlib_market/prod_ensemble/s1731.pt",
    "pufferlib_market/prod_ensemble/gamma995_s2006.pt",
    "pufferlib_market/prod_ensemble/s1401.pt",
    "pufferlib_market/prod_ensemble/s1726.pt",
    "pufferlib_market/prod_ensemble/s1523.pt",
    "pufferlib_market/prod_ensemble/s2617.pt",
    "pufferlib_market/prod_ensemble/s2033.pt",
    "pufferlib_market/prod_ensemble/s2495.pt",
    "pufferlib_market/prod_ensemble/s1835.pt",
    "pufferlib_market/prod_ensemble/s2827.pt",
    "pufferlib_market/prod_ensemble/s2722.pt",
    "pufferlib_market/prod_ensemble/s3668.pt",
    "pufferlib_market/prod_ensemble/s3411.pt",
    "pufferlib_market/prod_ensemble/s4011.pt",
    "pufferlib_market/prod_ensemble/s4777.pt",
    "pufferlib_market/prod_ensemble/s4080.pt",
    "pufferlib_market/prod_ensemble/s4533.pt",
    "pufferlib_market/prod_ensemble/s4813.pt",
    "pufferlib_market/prod_ensemble/s5045.pt",
    "pufferlib_market/prod_ensemble/s5337.pt",
    "pufferlib_market/prod_ensemble/s5199.pt",
    "pufferlib_market/prod_ensemble/s5019.pt",
    "pufferlib_market/prod_ensemble/s6808.pt",
    "pufferlib_market/prod_ensemble/s3456.pt",
    "pufferlib_market/prod_ensemble/s7159.pt",
    "pufferlib_market/prod_ensemble/s6758.pt",
]
DEFAULT_DATA_DIR = "trainingdata"
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
BUYING_POWER_USAGE_CAP = 0.95
MAX_STOCK_SYMBOL_LENGTH = 20
SAFE_STOCK_SYMBOL_RE = re.compile(
    rf"^[A-Z0-9](?:[A-Z0-9.-]{{0,{MAX_STOCK_SYMBOL_LENGTH - 2}}}[A-Z0-9])?$"
)
SERVER_MARKETABLE_LIMIT_BUFFER_BPS = 100.0
# Calibrated execution offsets (2026-03-31 sweep over 726 combos, 788 windows)
# Best: entry=+5bps, exit=+25bps, scale=0.5x → val_p10=-0.4% vs baseline -2.3%
CALIBRATED_ENTRY_OFFSET_BPS = 5.0   # Buy limit at open * 1.0005
CALIBRATED_EXIT_OFFSET_BPS = 25.0   # Sell limit at open * 1.0025
DEFAULT_MIN_OPEN_CONFIDENCE = 0.20
DEFAULT_MIN_OPEN_VALUE_ESTIMATE = 0.0
STATE_PATH = REPO / "strategy_state/daily_stock_rl_state.json"
SIGNAL_LOG_PATH = REPO / "strategy_state/daily_stock_rl_signals.jsonl"

StockDataSource: TypeAlias = Literal["alpaca", "local"]
StockExecutionBackend: TypeAlias = Literal["alpaca", "trading_server"]
StockRunMode: TypeAlias = Literal["once", "daemon", "backtest"]
StockAccountMode: TypeAlias = Literal["paper", "live"]
TradingServerUrlSource: TypeAlias = Literal["cli", "env", "default"]
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
StockLocalDataStatus: TypeAlias = Literal["usable", "stale", "missing", "invalid"]


class RuntimeLogPayload(TypedDict, total=False):
    account_mode: StockAccountMode
    run_mode: StockRunMode
    dry_run: bool
    data_source: StockDataSource
    execution_backend: StockExecutionBackend
    symbol_count: int
    symbols: list[str]
    checkpoint: str
    ensemble_size: int
    command_preview: str
    data_dir: str
    server_account: str
    server_bot_id: str
    configured_server_url: str | None
    server_url: str
    resolved_server_url: str
    server_url_source: TradingServerUrlSource
    server_url_transport: str
    server_url_scope: str
    server_url_security: str


class SignalPayload(TypedDict):
    timestamp: str
    checkpoint: str
    action: str
    symbol: str | None
    direction: str | None
    confidence: float
    value_estimate: float
    allocation_fraction: float
    quotes: dict[str, float]


class ExecutionObservabilityFields(TypedDict):
    execution_submitted: bool
    execution_would_submit: bool
    execution_status: StockExecutionStatus
    execution_skip_reason: StockExecutionSkipReason | None


class RunSummaryPayload(TypedDict):
    event: str
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


class RunFailurePayload(TypedDict):
    event: str
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
    server_account: str | None
    server_bot_id: str | None
    server_url: str | None


class StockLocalSymbolDetail(TypedDict):
    status: StockLocalDataStatus
    file_path: str
    local_data_date: str | None
    row_count: int | None
    reason: str | None


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


@dataclass(frozen=True)
class ServerPositionView:
    symbol: str
    qty: float
    side: str = "long"
    avg_entry_price: float = 0.0
    current_price: float = 0.0


_DAILY_TRADER_CACHE_MAX_ENTRIES = 8
_BARE_POLICY_CACHE_MAX_ENTRIES = 64
_DAILY_TRADER_CACHE: "OrderedDict[tuple[object, ...], _DailyTraderCacheEntry]" = OrderedDict()
_BARE_POLICY_CACHE: "OrderedDict[tuple[object, ...], _BarePolicyCacheEntry]" = OrderedDict()
_DAILY_TRADER_CACHE_LOCK = Lock()
_BARE_POLICY_CACHE_LOCK = Lock()


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
    backtest_buying_power_multiplier: float = DEFAULT_BACKTEST_BUYING_POWER_MULTIPLIER
    symbols_file: Optional[str] = None
    min_open_confidence: float = DEFAULT_MIN_OPEN_CONFIDENCE
    min_open_value_estimate: float = DEFAULT_MIN_OPEN_VALUE_ESTIMATE
    print_payload: bool = False
    allow_unsafe_checkpoint_loading: bool = False
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

    def to_runtime_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["symbol_count"] = len(self.symbols)
        payload["ensemble_enabled"] = self.ensemble_enabled
        payload["ensemble_size"] = self.ensemble_size
        payload["account_mode"] = self.account_mode
        payload["run_mode"] = self.run_mode
        payload["summary"] = self.summary
        payload["check_command_preview"] = self.command_preview(check_config=True)
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
            f"with {self.ensemble_size} {checkpoint_label}"
        )

    def command_preview(
        self,
        *,
        force_dry_run: bool | None = None,
        check_config: bool = False,
    ) -> str:
        args = ["python", Path(__file__).name]
        if check_config:
            args.append("--check-config")

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
        elif self.extra_checkpoints != DEFAULT_EXTRA_CHECKPOINTS:
            args.append("--extra-checkpoints")
            args.extend(self.extra_checkpoints)

        args.extend(["--data-source", self.data_source])
        args.extend(["--data-dir", self.data_dir])
        args.extend(["--allocation-pct", f"{self.allocation_pct:g}"])
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


def _runtime_log_payload(config: CliRuntimeConfig) -> RuntimeLogPayload:
    payload: RuntimeLogPayload = {
        "account_mode": config.account_mode,
        "run_mode": config.run_mode,
        "dry_run": config.dry_run,
        "data_source": config.data_source,
        "execution_backend": config.execution_backend,
        "symbol_count": len(config.symbols),
        "symbols": list(config.symbols),
        "checkpoint": config.checkpoint,
        "ensemble_size": config.ensemble_size,
        "command_preview": config.command_preview(),
    }
    if config.backtest or config.data_source == "local":
        payload["data_dir"] = config.data_dir
    if config.execution_backend == "trading_server":
        payload["server_account"] = config.server_account
        payload["server_bot_id"] = config.server_bot_id
        payload.update(cast(RuntimeLogPayload, _trading_server_runtime_fields(config)))
    return payload


def _log_runtime_start(config: CliRuntimeConfig) -> None:
    logger.info("Runtime config: %s", json.dumps(_runtime_log_payload(config), sort_keys=True))


def _normalize_symbols(raw_symbols: Sequence[object]) -> tuple[list[str], list[str], list[str]]:
    normalized: list[str] = []
    removed_duplicate_symbols: list[str] = []
    ignored_symbol_inputs: list[str] = []
    seen: set[str] = set()
    removed_seen: set[str] = set()

    for raw_symbol in raw_symbols:
        stripped = str(raw_symbol).strip()
        if not stripped:
            ignored_symbol_inputs.append("<blank>")
            continue
        symbol = _normalize_stock_symbol(stripped)
        if symbol in seen:
            if symbol not in removed_seen:
                removed_duplicate_symbols.append(symbol)
                removed_seen.add(symbol)
            continue
        normalized.append(symbol)
        seen.add(symbol)

    if not normalized:
        raise ValueError("No valid symbols configured after normalization")

    return normalized, removed_duplicate_symbols, ignored_symbol_inputs


def _normalize_stock_symbol(raw_symbol: object) -> str:
    symbol = str(raw_symbol).strip().upper()
    if not symbol:
        raise ValueError("symbol is required")
    if ".." in symbol or "/" in symbol or "\\" in symbol:
        raise ValueError(f"Unsupported symbol: {raw_symbol}")
    if not SAFE_STOCK_SYMBOL_RE.fullmatch(symbol):
        raise ValueError(f"Unsupported symbol: {raw_symbol}")
    return symbol


def _normalize_stock_symbol_list(symbols: Iterable[object]) -> list[str]:
    return [_normalize_stock_symbol(symbol) for symbol in symbols]


def _load_symbols_file(path: str | Path) -> list[str]:
    values: list[str] = []
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.replace(",", " ").split():
            symbol = _normalize_stock_symbol(token)
            values.append(symbol)
    if not values:
        raise ValueError(f"No valid symbols found in {path}")
    return values


def _resolve_local_data_base(data_dir: str | Path, symbols: Iterable[str]) -> Path:
    base = (REPO / Path(data_dir).expanduser()).resolve()
    normalized_symbols = _normalize_stock_symbol_list(symbols)
    nested_train = base / "train"
    if base.name != "train" and nested_train.exists():
        if all((nested_train / f"{symbol}.csv").exists() for symbol in normalized_symbols):
            return nested_train
    return base


def _normalize_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    lower_map = {str(col).lower(): col for col in frame.columns}
    ts_col = lower_map.get("timestamp") or lower_map.get("date")
    required = ["open", "high", "low", "close", "volume"]
    missing = [name for name in required if name not in lower_map]
    if ts_col is None or missing:
        raise ValueError(f"Daily frame missing columns: timestamp/date + {required}")

    normalized = frame.rename(columns={src: src.lower() for src in frame.columns}).copy()
    normalized["timestamp"] = pd.to_datetime(normalized[ts_col.lower()], utc=True)
    normalized = normalized[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    normalized = normalized.dropna(subset=["timestamp"]).sort_values("timestamp")
    normalized = normalized.drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)
    for column in required:
        normalized[column] = normalized[column].astype(float)
    return normalized


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
    guard = shared_path_guard(path)
    guard.acquire_read()
    try:
        if not path.exists():
            return StrategyState()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not parse %s: %s", path, exc)
            return StrategyState()
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
    finally:
        guard.release_read()


def save_state(state: StrategyState, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(state), indent=2, sort_keys=True)
    guard = shared_path_guard(path)
    temp_path: Path | None = None
    guard.acquire_write()
    try:
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
    finally:
        guard.release_write()
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def append_signal_log(payload: dict, path: Path = SIGNAL_LOG_PATH) -> None:
    append_jsonl_row(path, payload, sort_keys=True)


def _append_signal_log_best_effort(payload: dict, path: Path = SIGNAL_LOG_PATH) -> str | None:
    try:
        append_signal_log(payload, path=path)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Signal log write failed for %s: %s", path, exc)
        return str(exc)
    return None


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
        with _DAILY_TRADER_CACHE_LOCK:
            cached = _DAILY_TRADER_CACHE.get(cache_key)
            if cached is not None and cached.size == stat_result.st_size and cached.mtime_ns == stat_result.st_mtime_ns:
                _DAILY_TRADER_CACHE.move_to_end(cache_key)
                return copy.deepcopy(cached.trader)

    trader = DailyPPOTrader(
        checkpoint_path,
        device=device,
        long_only=long_only,
        symbols=list(normalized_symbols) if normalized_symbols else None,
        allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
    )
    if stat_result is not None:
        with _DAILY_TRADER_CACHE_LOCK:
            _DAILY_TRADER_CACHE[cache_key] = _DailyTraderCacheEntry(
                size=stat_result.st_size,
                mtime_ns=stat_result.st_mtime_ns,
                trader=copy.deepcopy(trader),
            )
            _DAILY_TRADER_CACHE.move_to_end(cache_key)
            _prune_ordered_cache(_DAILY_TRADER_CACHE, max_entries=_DAILY_TRADER_CACHE_MAX_ENTRIES)
    return trader


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
        with _BARE_POLICY_CACHE_LOCK:
            cached = _BARE_POLICY_CACHE.get(cache_key)
            if cached is not None and cached.size == stat_result.st_size and cached.mtime_ns == stat_result.st_mtime_ns:
                _BARE_POLICY_CACHE.move_to_end(cache_key)
                # Bare policies are eval-only (no mutable state changed during inference_mode forward)
                return cached.policy

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
                policy=copy.deepcopy(policy),
            )
            _BARE_POLICY_CACHE.move_to_end(cache_key)
            _prune_ordered_cache(_BARE_POLICY_CACHE, max_entries=_BARE_POLICY_CACHE_MAX_ENTRIES)
    return policy


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

    if extra_checkpoints:
        features = np.zeros((trader.num_symbols, 16), dtype=np.float32)
        for i, sym in enumerate(trader.SYMBOLS):
            if sym in indexed:
                features[i] = compute_daily_features(indexed[sym])
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
        signal = trader.get_daily_signal(indexed, prices)

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
    portfolio_value = float(getattr(account, "portfolio_value", 0.0) or 0.0)
    buying_power = float(getattr(account, "buying_power", 0.0) or 0.0)
    return compute_target_qty_from_values(
        portfolio_value=portfolio_value,
        buying_power=buying_power,
        price=price,
        allocation_pct=allocation_pct,
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


def effective_signal_allocation_pct(signal, *, base_allocation_pct: float) -> float:
    effective_pct = max(0.0, float(base_allocation_pct))
    raw_fraction = getattr(signal, "allocation_pct", None)
    if raw_fraction is None:
        return effective_pct
    try:
        fraction = float(raw_fraction)
    except (TypeError, ValueError):
        return effective_pct
    if not math.isfinite(fraction):
        return effective_pct
    return effective_pct * min(max(fraction, 0.0), 1.0)


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


def execute_signal_with_trading_server(
    signal,
    *,
    server_client: TradingServerClientLike,
    quotes: dict[str, float],
    state: StrategyState,
    symbols: Iterable[str],
    allocation_pct: float,
    dry_run: bool,
    now: Optional[datetime] = None,
    allow_open: bool = True,
    allow_open_reason: str | None = None,
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
            order = server_client.submit_limit_order(
                symbol=managed_symbol,
                qty=qty,
                side="sell",
                limit_price=_marketable_limit_price(float(quotes[managed_symbol]), "sell"),
                allow_loss_exit=True,
                force_exit_reason="daily strategy rotation",
                metadata={"strategy": "daily_stock_rl", "intent": "close_managed"},
            )
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
    effective_allocation_pct = effective_signal_allocation_pct(signal, base_allocation_pct=allocation_pct)
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
    now: Optional[datetime] = None,
    allow_open: bool = True,
    allow_open_reason: str | None = None,
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
    effective_allocation_pct = effective_signal_allocation_pct(signal, base_allocation_pct=allocation_pct)
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

    # Desired portfolio: symbol -> target_allocation_fraction
    desired = {}
    for sig in signals:
        if sig.symbol and sig.direction == "long":
            sym = sig.symbol.upper()
            desired[sym] = float(sig.allocation_pct)

    # Close positions not in desired set
    orders_placed = 0
    for sym, pos in list(live_positions.items()):
        if sym not in desired:
            qty = abs(_signed_position_qty(pos))
            if qty > 0:
                sell_price = float(quotes.get(sym, 0.0) or 0.0)
                limit_sell_price = (
                    sell_price if paper
                    else sell_price * (1.0 + CALIBRATED_EXIT_OFFSET_BPS / 10_000.0)
                ) if sell_price > 0 else 0
                logger.info("Multi-pos: closing %s qty=%.4f limit=%.2f", sym, qty, limit_sell_price)
                if not dry_run and limit_sell_price > 0:
                    submit_limit_order(client, symbol=sym, qty=qty, side="sell", limit_price=limit_sell_price)
                    orders_placed += 1

    # Get account for buying power
    account = client.get_account()
    equity = float(getattr(account, "equity", 0) or 0)
    buying_power = float(getattr(account, "buying_power", 0) or 0)

    # Open/adjust positions in desired set
    held = {}
    for sym, alloc_frac in desired.items():
        existing = live_positions.get(sym)
        existing_qty = abs(_signed_position_qty(existing)) if existing else 0.0
        price = float(quotes.get(sym, 0.0) or 0.0)
        if price <= 0:
            continue

        buy_price = price if paper else price * (1.0 + CALIBRATED_ENTRY_OFFSET_BPS / 10_000.0)
        target_value = equity * (total_allocation_pct / 100.0) * alloc_frac
        target_qty = int(target_value / buy_price) if buy_price > 0 else 0

        if existing_qty > 0:
            held[sym] = existing_qty
            logger.info("Multi-pos: holding %s qty=%.4f (target=%.4f)", sym, existing_qty, target_qty)
        elif target_qty > 0:
            logger.info("Multi-pos: opening %s qty=%d limit=%.2f (alloc=%.1f%%)",
                        sym, target_qty, buy_price, alloc_frac * total_allocation_pct)
            if not dry_run:
                submit_limit_order(client, symbol=sym, qty=target_qty, side="buy", limit_price=buy_price)
                orders_placed += 1
            held[sym] = target_qty

    logger.info("Multi-pos: %d orders placed, %d positions targeted", orders_placed, len(held))
    return held


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


def _signal_payload(signal, *, checkpoint: str, quotes: dict[str, float]) -> SignalPayload:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "action": signal.action,
        "symbol": signal.symbol,
        "direction": signal.direction,
        "confidence": float(signal.confidence),
        "value_estimate": float(signal.value_estimate),
        "allocation_fraction": float(getattr(signal, "allocation_pct", 1.0) or 0.0),
        "quotes": {symbol: float(price) for symbol, price in quotes.items()},
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
    }


def _log_run_summary(payload: dict[str, object]) -> None:
    logger.info("Run summary: %s", json.dumps(_run_summary_payload(payload), sort_keys=True))


def _run_failure_payload(
    *,
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
    server_account: str | None,
    server_bot_id: str | None,
    server_url: str | None,
) -> RunFailurePayload:
    return {
        "event": "daily_stock_run_once_failed",
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
        "server_account": str(server_account).strip() or None,
        "server_bot_id": str(server_bot_id).strip() or None,
        "server_url": (str(server_url).strip().rstrip("/") or None) if server_url is not None else None,
    }


def _log_run_failure(payload: RunFailurePayload) -> None:
    logger.error("Run failure: %s", json.dumps(payload, sort_keys=True))


def run_backtest(
    *,
    checkpoint: str,
    symbols: Iterable[str],
    data_dir: str,
    days: int,
    allocation_pct: float = 100.0,
    starting_cash: float = DEFAULT_BACKTEST_STARTING_CASH,
    extra_checkpoints: Optional[list[str]] = None,
    buying_power_multiplier: float = DEFAULT_BACKTEST_BUYING_POWER_MULTIPLIER,
    allow_unsafe_checkpoint_loading: bool = False,
    entry_offset_bps: float = 0.0,
    exit_offset_bps: float = 0.0,
) -> dict[str, float]:
    if starting_cash <= 0:
        raise ValueError("starting_cash must be positive")
    if buying_power_multiplier <= 0:
        raise ValueError("buying_power_multiplier must be positive")
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
    feature_history = (
        {
            symbol: compute_daily_feature_history(frame)
            for symbol, frame in indexed.items()
        }
        if extra_policies
        else {}
    )

    cash = float(starting_cash)
    position: Optional[tuple[str, float, float]] = None
    equity_curve: list[float] = []
    trades = 0

    for idx in range(start, min_len):
        prices = {symbol: float(frame["close"].iloc[idx]) for symbol, frame in indexed.items()}
        trader.cash = cash
        trader.current_position = None
        trader.position_qty = 0.0
        trader.entry_price = 0.0
        if position is not None:
            pos_symbol, qty, entry_price = position
            trader.current_position = trader.SYMBOLS.index(pos_symbol)
            trader.position_qty = qty
            trader.entry_price = entry_price

        if extra_policies:
            features = np.zeros((trader.num_symbols, 16), dtype=np.float32)
            for feature_idx, symbol in enumerate(trader.SYMBOLS):
                features[feature_idx] = feature_history[symbol].iloc[idx].to_numpy(dtype=np.float32, copy=False)
            signal = _ensemble_softmax_signal(trader, extra_policies, features, prices)
        else:
            signal = trader.get_daily_signal(
                {symbol: frame.iloc[: idx + 1] for symbol, frame in indexed.items()},
                prices,
            )

        equity = cash
        if position is not None:
            pos_symbol, qty, _ = position
            equity += qty * prices[pos_symbol]
        equity_curve.append(equity)

        if position is not None and (signal.symbol != position[0] or signal.direction != "long"):
            pos_symbol, qty, _ = position
            sell_price = prices[pos_symbol] * (1.0 + exit_offset_bps / 10_000.0)
            cash += qty * sell_price
            position = None
            trades += 1
            trader.update_state(0, 0.0, "")

        if position is None and signal.symbol and signal.direction == "long":
            effective_allocation_pct = effective_signal_allocation_pct(
                signal,
                base_allocation_pct=allocation_pct,
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
            trader.update_state(trader.SYMBOLS.index(signal.symbol) + 1, prices[signal.symbol], signal.symbol)

        trader.step_day()

    if position is not None:
        equity_curve.append(cash + position[1] * indexed[position[0]]["close"].iloc[min_len - 1])
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
    state_path: Path = STATE_PATH,
    extra_checkpoints: Optional[list] = None,
    execution_backend: StockExecutionBackend = "alpaca",
    server_account: str = DEFAULT_SERVER_PAPER_ACCOUNT,
    server_bot_id: str = DEFAULT_SERVER_PAPER_BOT_ID,
    server_url: str | None = None,
    server_session_id: str | None = None,
    min_open_confidence: float = DEFAULT_MIN_OPEN_CONFIDENCE,
    min_open_value_estimate: float = DEFAULT_MIN_OPEN_VALUE_ESTIMATE,
    allow_unsafe_checkpoint_loading: bool = False,
) -> dict:
    now = datetime.now(timezone.utc)
    symbol_list = [str(symbol).upper() for symbol in symbols]
    failure_stage = "load_state"
    try:
        state = load_state(state_path)
        quote_data_source = "local"
        quote_source_by_symbol: dict[str, str] = {}
        latest_bar = None
        market_open = None
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
            failure_stage = "load_latest_quotes"
            quotes, quote_data_source, quote_source_by_symbol = load_latest_quotes_with_source(
                symbol_list,
                paper=paper,
                fallback_prices=close_prices,
                data_client=data_client,
            )
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
                if not dry_run:
                    reconcile_pending_close(state=state, live_positions=live_positions)
                    adopt_existing_position(state=state, live_positions=live_positions, now=now)
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
                if not dry_run:
                    reconcile_pending_close(state=state, live_positions=live_positions)
                    adopt_existing_position(state=state, live_positions=live_positions, now=now)
                failure_stage = "get_alpaca_account"
                portfolio = build_portfolio_context(
                    state=state,
                    live_positions=live_positions,
                    account=client.get_account(),
                    now=now,
                )
            try:
                failure_stage = "get_market_clock"
                market_open = bool(getattr(clock_client.get_clock(), "is_open", False))
            except Exception as exc:
                logger.warning("Could not read Alpaca market clock: %s", exc)
                market_open = False
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

        failure_stage = "build_signal"
        signal, close_prices = build_signal(
            checkpoint,
            frames,
            portfolio=portfolio,
            extra_checkpoints=extra_checkpoints,
            allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
        )
        bars_fresh = (
            bars_are_fresh(
                latest_bar=latest_bar,
                now=now,
                max_age_days=DEFAULT_BAR_FRESHNESS_MAX_AGE_DAYS,
            )
            if latest_bar is not None
            else False
        )

        payload = _signal_payload(signal, checkpoint=checkpoint, quotes=quotes)
        payload["close_prices"] = close_prices
        payload["bar_data_source"] = bar_data_source
        payload["quote_data_source"] = quote_data_source
        payload["quote_source_by_symbol"] = dict(quote_source_by_symbol)
        payload["latest_bar_timestamp"] = latest_bar.isoformat() if latest_bar is not None else None
        payload["bars_fresh"] = bars_fresh
        payload["market_open"] = market_open
        payload["dry_run"] = bool(dry_run)
        payload["execution_backend"] = execution_backend
        payload["min_open_confidence"] = float(min_open_confidence)
        payload["min_open_value_estimate"] = float(min_open_value_estimate)

        logger.info("%s", "=" * 60)
        logger.info("DAILY STOCK RL SIGNAL (%s)", now.strftime("%Y-%m-%d %H:%M UTC"))
        logger.info("%s", "=" * 60)
        logger.info("Action:     %s", signal.action)
        logger.info("Symbol:     %s", signal.symbol or "N/A")
        logger.info("Direction:  %s", signal.direction or "N/A")
        logger.info("Confidence: %.1f%%", float(signal.confidence) * 100.0)
        logger.info("Value est:  %.4f", float(signal.value_estimate))
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
        if data_source == "alpaca":
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
            if not dry_run and not bool(market_open):
                logger.warning("Market is closed; skipping order placement")
            elif not dry_run and not bars_fresh:
                logger.warning("Latest inference bar is stale; skipping order placement")
            else:
                if execution_backend == "trading_server":
                    failure_stage = "execute_signal_with_trading_server"
                    executed = execute_signal_with_trading_server(
                        signal,
                        server_client=client,
                        quotes=quotes,
                        state=state,
                        symbols=symbol_list,
                        allocation_pct=allocation_pct,
                        dry_run=dry_run,
                        now=now,
                        allow_open=allow_open,
                        allow_open_reason=allow_open_reason,
                    )
                    failure_stage = "get_trading_server_account_snapshot"
                    payload["server_account"] = server_account
                    payload["server_bot_id"] = server_bot_id
                    payload["server_snapshot"] = client.get_account()
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
                        dry_run=dry_run,
                        now=now,
                        allow_open=allow_open,
                        allow_open_reason=allow_open_reason,
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
            state.last_signal_action = signal.action
            state.last_signal_timestamp = now.isoformat()
            save_state(state, path=state_path)
        payload["executed"] = executed
        signal_log_write_error = _append_signal_log_best_effort(payload)
        payload["signal_log_written"] = signal_log_write_error is None
        payload["signal_log_write_error"] = signal_log_write_error
        failure_stage = "log_run_summary"
        _log_run_summary(payload)
        logger.info("%s", "=" * 60)
        return payload
    except Exception as exc:
        stage_note = f"run_once stage: {failure_stage}"
        if hasattr(exc, "add_note") and stage_note not in _exception_notes(exc):
            exc.add_note(stage_note)
        _log_run_failure(
            _run_failure_payload(
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
                server_account=server_account if execution_backend == "trading_server" else None,
                server_bot_id=server_bot_id if execution_backend == "trading_server" else None,
                server_url=resolve_trading_server_base_url(server_url)
                if execution_backend == "trading_server"
                else None,
            )
        )
        raise


def run_daemon(
    *,
    checkpoint: str,
    symbols: Iterable[str],
    paper: bool,
    allocation_pct: float,
    dry_run: bool,
    data_dir: str,
    extra_checkpoints: Optional[list] = None,
    execution_backend: str = "alpaca",
    server_account: str = DEFAULT_SERVER_PAPER_ACCOUNT,
    server_bot_id: str = DEFAULT_SERVER_PAPER_BOT_ID,
    server_url: str | None = None,
    min_open_confidence: float = DEFAULT_MIN_OPEN_CONFIDENCE,
    min_open_value_estimate: float = DEFAULT_MIN_OPEN_VALUE_ESTIMATE,
    allow_unsafe_checkpoint_loading: bool = False,
) -> None:
    logger.info("Starting daily stock RL daemon")
    server_session_id = f"daily-rl-trader-{execution_backend}-{os.getpid()}"
    while True:
        state = load_state()
        # Use paper API for clock check (market hours same regardless of paper/live).
        # Fall back to paper=True if live keys are invalid (401) — service stays alive.
        clock_client = build_trading_client(paper=paper)
        try:
            clock = clock_client.get_clock()
        except Exception as _clock_err:
            if not paper:
                logger.warning("Live clock check failed (%s); retrying with paper API", _clock_err)
                try:
                    clock = build_trading_client(paper=True).get_clock()
                except Exception as _paper_err:
                    logger.warning("Paper clock check also failed (%s); sleeping 5min", _paper_err)
                    time.sleep(300.0)
                    continue
            else:
                logger.warning("Clock check failed (%s); sleeping 5min", _clock_err)
                time.sleep(300.0)
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
                    dry_run=dry_run,
                    data_source="alpaca",
                    data_dir=data_dir,
                    extra_checkpoints=extra_checkpoints,
                    execution_backend=execution_backend,
                    server_account=server_account,
                    server_bot_id=server_bot_id,
                server_url=server_url,
                server_session_id=server_session_id,
                min_open_confidence=min_open_confidence,
                min_open_value_estimate=min_open_value_estimate,
                allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
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
    parser.add_argument("--no-ensemble", action="store_true",
                        help="Disable ensemble, use --checkpoint alone")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--data-source", choices=["alpaca", "local"], default="alpaca")
    parser.add_argument("--allocation-pct", type=float, default=DEFAULT_ALLOCATION_PCT)
    parser.add_argument("--multi-position", type=int, default=0,
                        help="Hold up to N simultaneous positions (0=single position mode)")
    parser.add_argument("--multi-position-min-prob-ratio", type=float, default=0.3,
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
    return parser.parse_args(argv)


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
    elif args.extra_checkpoints is not None:
        extra_checkpoints = [_resolve(path) for path in args.extra_checkpoints]
    else:
        extra_checkpoints = [_resolve(path) for path in DEFAULT_EXTRA_CHECKPOINTS]

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
        execution_backend=cast(StockExecutionBackend, args.execution_backend),
        server_account=args.server_account,
        server_bot_id=args.server_bot_id,
        server_url=args.server_url,
        dry_run=bool(args.dry_run),
        backtest=bool(args.backtest),
        backtest_days=int(args.backtest_days),
        backtest_starting_cash=float(args.backtest_starting_cash),
        backtest_buying_power_multiplier=float(args.backtest_buying_power_multiplier),
        daemon=bool(args.daemon),
        compare_server_parity=bool(args.compare_server_parity),
        min_open_confidence=float(args.min_open_confidence),
        min_open_value_estimate=float(args.min_open_value_estimate),
        print_payload=bool(args.print_payload),
        allow_unsafe_checkpoint_loading=bool(args.allow_unsafe_checkpoint_loading),
    )


def _runtime_config_payload(config: CliRuntimeConfig) -> dict[str, object]:
    return config.to_runtime_payload()


def _missing_checkpoint_paths(config: CliRuntimeConfig) -> list[str]:
    return config.missing_checkpoint_paths


def _checkpoint_load_diagnostics(config: CliRuntimeConfig) -> dict[str, object]:
    if config.missing_checkpoint_paths:
        return {
            "ok": False,
            "error": "checkpoint files are missing",
            "primary": None,
            "extras": [],
        }
    try:
        trader = _load_cached_daily_trader(
            config.checkpoint,
            device="cpu",
            long_only=True,
            symbols=list(config.symbols),
            allow_unsafe_checkpoint_loading=config.allow_unsafe_checkpoint_loading,
        )
        extras: list[dict[str, object]] = []
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
            "primary": trader.summary_dict(),
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
    safe_command = str(payload.get("safe_command_preview") or "").strip()
    run_command = str(payload.get("run_command_preview") or "").strip()

    if bool(payload.get("ready")):
        if safe_command:
            steps.append(f"Run a safe dry run: {safe_command}")
        if run_command and run_command != safe_command:
            steps.append(f"Start the configured runtime: {run_command}")
        if check_command:
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
        steps.append(f"Re-run preflight after fixes: {check_command}")
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
                "status": "missing",
                "file_path": str(path),
                "local_data_date": None,
                "row_count": None,
                "reason": "missing local daily CSV",
            }
            continue
        try:
            frame = _normalize_daily_frame(pd.read_csv(path))
        except Exception as exc:
            details[symbol] = {
                "status": "invalid",
                "file_path": str(path),
                "local_data_date": None,
                "row_count": None,
                "reason": str(exc),
            }
            continue

        latest_timestamp = pd.Timestamp(frame["timestamp"].iloc[-1])
        latest_date = latest_timestamp.date().isoformat()
        usable_symbols.append(symbol)
        usable_timestamps[symbol] = latest_timestamp
        details[symbol] = {
            "status": "usable",
            "file_path": str(path),
            "local_data_date": latest_date,
            "row_count": int(len(frame)),
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
                    "status": "stale",
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
    if config.backtest_buying_power_multiplier <= 0:
        errors.append("--backtest-buying-power-multiplier must be positive")
    if not 0.0 <= float(config.min_open_confidence) <= 1.0:
        errors.append("--min-open-confidence must be between 0 and 1")
    if config.compare_server_parity and not config.backtest:
        errors.append("--compare-server-parity requires --backtest")
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

    data_dir = Path(config.data_dir).expanduser()
    local_data_required = bool(config.backtest or config.data_source == "local")
    missing_local_symbol_files: list[str] = []
    resolved_local_data_dir: Path | None = None
    symbol_details: dict[str, StockLocalSymbolDetail] = {}
    usable_symbols: list[str] = []
    usable_symbol_count = 0
    latest_local_data_date: str | None = None
    oldest_local_data_date: str | None = None
    stale_symbol_data: dict[str, str] = {}
    if local_data_required:
        if not data_dir.exists():
            errors.append(f"Local data directory does not exist: {data_dir}")
        else:
            resolved_local_data_dir = _resolve_local_data_base(config.data_dir, config.symbols)
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
    payload["data_dir_exists"] = data_dir.exists()
    payload["resolved_local_data_dir"] = (
        str(resolved_local_data_dir) if resolved_local_data_dir is not None else None
    )
    payload["missing_local_symbol_files"] = missing_local_symbol_files
    payload["usable_symbols"] = usable_symbols
    payload["usable_symbol_count"] = usable_symbol_count
    payload["latest_local_data_date"] = latest_local_data_date
    payload["oldest_local_data_date"] = oldest_local_data_date
    payload["stale_symbol_data"] = stale_symbol_data
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


def _format_runtime_preflight_failure(payload: dict[str, object]) -> str:
    lines = [
        "Daily stock RL setup is not ready.",
        str(payload.get("summary") or ""),
    ]
    errors = [str(item) for item in payload.get("errors", [])]
    warnings = [str(item) for item in payload.get("warnings", [])]
    next_steps = [str(item) for item in payload.get("next_steps", [])]
    if errors:
        lines.append("Errors:")
        lines.extend(f"- {item}" for item in errors)
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {item}" for item in warnings)
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
    warnings = [str(item) for item in payload.get("warnings", [])]
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {item}" for item in warnings)
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
    }
    command_steps.discard("")
    additional_steps = [item for item in next_steps if item not in command_steps]
    if additional_steps:
        lines.append("Additional next steps:")
        lines.extend(f"- {item}" for item in additional_steps)
    return "\n".join(line for line in lines if line)


def _exception_notes(exc: BaseException) -> list[str]:
    notes = getattr(exc, "__notes__", None)
    if not notes:
        return []
    return [str(note) for note in notes if str(note).strip()]


def _format_run_once_failure_message(config: CliRuntimeConfig, exc: BaseException) -> str:
    stage_label: str | None = None
    for note in _exception_notes(exc):
        if note.startswith("run_once stage: "):
            stage_label = note.removeprefix("run_once stage: ").replace("_", " ")
            break

    if stage_label:
        lines = [f"Daily stock RL run failed during {stage_label}."]
    else:
        lines = ["Daily stock RL run failed."]
    lines.append(f"Error: {type(exc).__name__}: {exc}")
    lines.append(f"Check config: {config.command_preview(check_config=True)}")

    safe_command = config.command_preview(force_dry_run=True)
    run_command = config.command_preview()
    if safe_command and safe_command != run_command:
        lines.append(f"Try a safe dry run: {safe_command}")

    extra_notes = [
        note for note in _exception_notes(exc) if not note.startswith("run_once stage: ")
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
    starting_cash: float = DEFAULT_BACKTEST_STARTING_CASH,
    buying_power_multiplier: float = DEFAULT_BACKTEST_BUYING_POWER_MULTIPLIER,
    account: str = DEFAULT_BACKTEST_SERVER_ACCOUNT,
    bot_id: str = DEFAULT_BACKTEST_SERVER_BOT_ID,
    extra_checkpoints: Optional[list[str]] = None,
    allow_unsafe_checkpoint_loading: bool = False,
) -> dict[str, float]:
    if starting_cash <= 0:
        raise ValueError("starting_cash must be positive")
    if buying_power_multiplier <= 0:
        raise ValueError("buying_power_multiplier must be positive")
    from src.trading_server.server import TradingServerEngine

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
            current_now = pd.Timestamp(next(iter(indexed.values())).index[idx]).to_pydatetime()
            if current_now.tzinfo is None:
                current_now = current_now.replace(tzinfo=timezone.utc)
            prices = {symbol: float(frame["close"].iloc[idx]) for symbol, frame in indexed.items()}
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
            portfolio = server_portfolio_context(
                snapshot=snapshot,
                state=current_state,
                quotes=prices,
                now=current_now,
            )
            signal = build_signal(
                checkpoint,
                {symbol: frame.iloc[: idx + 1].reset_index() for symbol, frame in indexed.items()},
                portfolio=portfolio,
                extra_checkpoints=extra_checkpoints,
                allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
            )[0]
            equity_curve.append(server_equity(snapshot, prices))
            execute_signal_with_trading_server(
                signal,
                server_client=server_client,
                quotes=prices,
                state=current_state,
                symbols=symbols,
                allocation_pct=allocation_pct,
                dry_run=False,
                now=current_now,
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
        trade_count = sum(1 for order in order_history if str(order.get("side", "")).lower() == "sell")
        results = {
            "total_return": total_return,
            "annualized_return": annualized,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "trades": float(trade_count),
            "orders": float(len(order_history)),
        }
        logger.info("Trading-server paper backtest: %s", json.dumps(results, sort_keys=True))
        return results


def compare_backtest_to_trading_server(
    *,
    checkpoint: str,
    symbols: Iterable[str],
    data_dir: str,
    days: int,
    allocation_pct: float = 100.0,
    starting_cash: float = DEFAULT_BACKTEST_STARTING_CASH,
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
        starting_cash=starting_cash,
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
        starting_cash=starting_cash,
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
    _log_runtime_start(config)

    if config.backtest:
        if config.compare_server_parity:
            compare_backtest_to_trading_server(
                checkpoint=config.checkpoint,
                symbols=config.symbols,
                data_dir=config.data_dir,
                days=config.backtest_days,
                allocation_pct=config.allocation_pct,
                starting_cash=config.backtest_starting_cash,
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
            starting_cash=config.backtest_starting_cash,
            extra_checkpoints=config.extra_checkpoints,
            buying_power_multiplier=config.backtest_buying_power_multiplier,
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
            dry_run=config.dry_run,
            data_dir=config.data_dir,
            extra_checkpoints=config.extra_checkpoints,
            execution_backend=config.execution_backend,
            server_account=config.server_account,
            server_bot_id=config.server_bot_id,
            server_url=config.server_url,
            min_open_confidence=config.min_open_confidence,
            min_open_value_estimate=config.min_open_value_estimate,
            allow_unsafe_checkpoint_loading=config.allow_unsafe_checkpoint_loading,
        )
        return

    try:
        payload = run_once(
            checkpoint=config.checkpoint,
            symbols=config.symbols,
            paper=config.paper,
            allocation_pct=config.allocation_pct,
            dry_run=config.dry_run,
            data_source=config.data_source,
            data_dir=config.data_dir,
            extra_checkpoints=config.extra_checkpoints,
            execution_backend=config.execution_backend,
            server_account=config.server_account,
            server_bot_id=config.server_bot_id,
            server_url=config.server_url,
            min_open_confidence=config.min_open_confidence,
            min_open_value_estimate=config.min_open_value_estimate,
            allow_unsafe_checkpoint_loading=config.allow_unsafe_checkpoint_loading,
        )
    except Exception as exc:
        print(_format_run_once_failure_message(config, exc), file=sys.stderr)
        raise SystemExit(1) from exc
    if config.print_payload:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
