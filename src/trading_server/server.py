from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import uuid
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, NotRequired, TypedDict, cast

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator

from src.background_refresher import BackgroundRefreshHandle, BackgroundRefreshRegistry
from src.file_backed_state_cache import FileBackedStateCache
from src.json_types import JsonObject
from src import order_validation as _order_validation
from src.order_validation import (
    normalize_order_metadata,
    normalize_positive_finite_float,
    safe_json_float,
)
from src.server_http_auth import (
    bearer_auth_matches,
    classify_bearer_auth_failure,
    normalize_auth_token,
)
from src.shared_path_guard import ReaderWriterGuard, shared_path_guard
from unified_orchestrator.jsonl_utils import append_jsonl_row
from unified_orchestrator.state_paths import resolve_state_dir
from . import settings as _settings

TradingServerSettings = _settings.TradingServerSettings
DEFAULT_REGISTRY_PATH = _settings.DEFAULT_REGISTRY_PATH
DEFAULT_QUOTE_STALE_SECONDS = _settings.DEFAULT_QUOTE_STALE_SECONDS
DEFAULT_WRITER_TTL_SECONDS = _settings.DEFAULT_WRITER_TTL_SECONDS
DEFAULT_BACKGROUND_POLL_SECONDS = _settings.DEFAULT_BACKGROUND_POLL_SECONDS
DEFAULT_QUOTE_FETCH_WORKERS = _settings.DEFAULT_QUOTE_FETCH_WORKERS
DEFAULT_MAX_ORDER_HISTORY = _settings.DEFAULT_MAX_ORDER_HISTORY
DEFAULT_SHARED_QUOTE_CACHE_SIZE = _settings.DEFAULT_SHARED_QUOTE_CACHE_SIZE
MIN_WRITER_TTL_SECONDS = _settings.MIN_WRITER_TTL_SECONDS
MAX_WRITER_TTL_SECONDS = _settings.MAX_WRITER_TTL_SECONDS
MAX_ACCOUNT_NAME_LENGTH = _settings.MAX_ACCOUNT_NAME_LENGTH
MAX_SYMBOL_LENGTH = _settings.MAX_SYMBOL_LENGTH
MAX_IDENTIFIER_LENGTH = 128
MAX_ORDER_METADATA_ITEMS = _order_validation.MAX_ORDER_METADATA_ITEMS
MAX_ORDER_METADATA_BYTES = _order_validation.MAX_ORDER_METADATA_BYTES

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover
    _fcntl = None

_SAFE_ACCOUNT_NAME_RE = re.compile(
    rf"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{{0,{MAX_ACCOUNT_NAME_LENGTH - 2}}}[A-Za-z0-9])?$"
)
_SAFE_SYMBOL_RE = re.compile(rf"^[A-Z0-9.]{{1,{MAX_SYMBOL_LENGTH}}}$")
_SAFE_IDENTIFIER_RE = re.compile(
    rf"^[A-Za-z0-9][A-Za-z0-9_.:-]{{0,{MAX_IDENTIFIER_LENGTH - 1}}}$"
)

TradingMode = Literal["paper", "live"]
OrderSide = Literal["buy", "sell"]
OrderStatus = Literal["open", "filled", "submitted"]
ExecutionMode = Literal["paper", "live"]
BrokerResponse = JsonObject
logger = logging.getLogger(__name__)


def _serialize_account_state(state: object) -> str:
    # Account state is rewritten frequently; keep on-disk payloads compact.
    return json.dumps(state, separators=(",", ":"), ensure_ascii=False)


def _shutdown_quote_fetch_executor(executor: ThreadPoolExecutor | None) -> None:
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=False)


class WriterClaim(TypedDict):
    bot_id: str
    session_id: str
    claimed_at: str | None
    expires_at: str | None
    ttl_seconds: int


class PublicWriterClaim(TypedDict):
    claimed_at: str | None
    expires_at: str | None
    active: bool


class QuotePayload(TypedDict):
    symbol: str
    bid_price: float
    ask_price: float
    last_price: float
    as_of: str | None


class PositionState(TypedDict, total=False):
    symbol: str
    qty: float
    avg_entry_price: float
    opened_at: str | None
    last_buy_at: str | None
    realized_pnl: float


class OrderRecord(TypedDict):
    id: str
    account: str
    bot_id: str
    session_id: str
    symbol: str
    side: OrderSide
    qty: float
    limit_price: float
    status: OrderStatus
    execution_mode: ExecutionMode
    allow_loss_exit: bool
    force_exit_reason: str | None
    metadata: JsonObject
    created_at: str | None
    filled_at: str | None
    fill_price: float | None
    broker_response: NotRequired[BrokerResponse]


class AccountConfig(TypedDict):
    name: str
    mode: TradingMode
    allowed_bot_id: str
    starting_cash: float
    paper_buying_power_multiplier: float
    base_currency: str
    sell_loss_cooldown_seconds: int
    min_sell_markup_pct: float
    symbols: list[str]


class AccountState(TypedDict):
    account: str
    mode: TradingMode
    base_currency: str
    cash: float
    realized_pnl: float
    positions: dict[str, PositionState]
    open_orders: list[OrderRecord]
    order_history: list[OrderRecord]
    price_cache: dict[str, QuotePayload]
    writer_claim: WriterClaim | None
    created_at: str | None
    updated_at: str | None


class SubmitOrderResult(TypedDict):
    order: OrderRecord
    quote: QuotePayload | None
    filled: bool


class ConfiguredAccountSummary(TypedDict):
    account: str
    mode: TradingMode
    symbols: list[str]


class _InFlightQuoteFetch:
    __slots__ = ("event", "quote", "unavailable_reason")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.quote: QuotePayload | None = None
        self.unavailable_reason: str | None = None


_AccountStateGuard = ReaderWriterGuard


def _quote_unavailable_reason(exc: BaseException | None = None) -> str:
    if exc is None:
        return "no quote returned"
    detail = str(exc).strip()
    if detail:
        return f"{type(exc).__name__}: {detail}"
    return type(exc).__name__


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(ts: datetime | None) -> str | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


def _parse_ts(raw: object) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo is not None else raw.replace(tzinfo=timezone.utc)
    text = str(raw).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _coerce_float(value: object, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return parsed


def _normalize_account_name(name: str) -> str:
    value = str(name or "").strip()
    if not value:
        raise ValueError("account is required")
    if not _SAFE_ACCOUNT_NAME_RE.fullmatch(value):
        raise ValueError(f"unsupported account name: {name}")
    return value


def _normalize_symbol(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if not raw:
        raise ValueError("symbol is required")
    if ".." in raw or "\\" in raw or raw.count("/") > 1:
        raise ValueError(f"unsupported symbol: {symbol}")
    value = raw.replace("/", "").replace("-", "")
    if not _SAFE_SYMBOL_RE.fullmatch(value):
        raise ValueError(f"unsupported symbol: {symbol}")
    return value


def _normalize_trading_mode(mode: object) -> TradingMode:
    value = str(mode or "").strip().lower()
    if value not in {"paper", "live"}:
        raise ValueError(f"unsupported mode={mode}")
    return cast(TradingMode, value)


def _normalize_side(side: object) -> OrderSide:
    value = str(side or "").strip().lower()
    if value not in {"buy", "sell"}:
        raise ValueError(f"unsupported side: {side}")
    return cast(OrderSide, value)


def _normalize_identifier(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    if not _SAFE_IDENTIFIER_RE.fullmatch(text):
        raise ValueError(f"unsupported {field_name}: {value}")
    return text


def _normalize_bot_id(bot_id: object) -> str:
    return _normalize_identifier(bot_id, field_name="bot_id")


def _normalize_session_id(session_id: object) -> str:
    return _normalize_identifier(session_id, field_name="session_id")


def _safe_metadata_for_rejection(metadata: object) -> JsonObject:
    try:
        return normalize_order_metadata(metadata)
    except ValueError as exc:
        return {"_metadata_error": str(exc)}


def _normalize_account_name_or_400(name: str) -> str:
    try:
        return _normalize_account_name(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _normalize_symbol_or_400(symbol: str) -> str:
    try:
        return _normalize_symbol(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class WriterLeaseRequest(BaseModel):
    account: str
    bot_id: str
    session_id: str | None = None
    ttl_seconds: int = Field(
        default_factory=lambda: TradingServerSettings.from_env().writer_ttl_seconds,
        ge=MIN_WRITER_TTL_SECONDS,
        le=MAX_WRITER_TTL_SECONDS,
    )

    @field_validator("bot_id")
    @classmethod
    def _validate_bot_id(cls, value: str) -> str:
        return _normalize_bot_id(value)

    @field_validator("session_id")
    @classmethod
    def _validate_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_session_id(value)


class OrderRequest(BaseModel):
    account: str
    bot_id: str
    session_id: str
    symbol: str
    side: OrderSide
    qty: float = Field(gt=0)
    limit_price: float = Field(gt=0)
    execution_mode: ExecutionMode = "paper"
    allow_loss_exit: bool = False
    force_exit_reason: str | None = None
    live_ack: str | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("bot_id")
    @classmethod
    def _validate_bot_id(cls, value: str) -> str:
        return _normalize_bot_id(value)

    @field_validator("session_id")
    @classmethod
    def _validate_session_id(cls, value: str) -> str:
        return _normalize_session_id(value)

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, value: JsonObject) -> JsonObject:
        return normalize_order_metadata(value)


OrderRequest.model_rebuild()


class RefreshPricesRequest(BaseModel):
    account: str | None = None
    symbols: list[str] = Field(default_factory=list)


class TradingServerEngine:
    def __init__(
        self,
        *,
        registry_path: str | Path | None = None,
        state_dir: str | Path | None = None,
        quote_provider: Callable[[str], QuotePayload | None] | None = None,
        live_executor: Callable[[OrderRecord], BrokerResponse | None] | None = None,
        now_fn: Callable[[], datetime] | None = None,
        quote_stale_seconds: int | None = None,
        quote_fetch_workers: int | None = None,
        max_order_history: int | None = None,
        shared_quote_cache_size: int | None = None,
        auth_token: str | None = None,
    ) -> None:
        self.settings_resolution = _settings.resolve_settings_resolution(
            registry_path=registry_path,
            quote_stale_seconds=quote_stale_seconds,
            writer_ttl_seconds=None,
            background_poll_seconds=None,
            quote_fetch_workers=quote_fetch_workers,
            max_order_history=max_order_history,
            shared_quote_cache_size=shared_quote_cache_size,
            auth_token=auth_token,
        )
        self.settings = self.settings_resolution.settings()
        self.registry_path = self.settings.registry_path
        self._registry_path_detail = self.settings_resolution.registry_path.detail
        self.state_root = resolve_state_dir(state_dir) / "trading_server"
        self.accounts_root = self.state_root / "accounts"
        self.event_root = self.state_root / "events"
        self.locks_root = self.state_root / "locks"
        self.accounts_root.mkdir(parents=True, exist_ok=True)
        self.event_root.mkdir(parents=True, exist_ok=True)
        self.locks_root.mkdir(parents=True, exist_ok=True)
        self.quote_provider = quote_provider or self._default_quote_provider
        self.live_executor = live_executor or self._default_live_executor
        self.now_fn = now_fn or _utc_now
        self.quote_stale_seconds = self.settings.quote_stale_seconds
        self.quote_fetch_workers = self.settings.quote_fetch_workers
        self.max_order_history = self.settings.max_order_history
        self.shared_quote_cache_size = self.settings.shared_quote_cache_size
        self.auth_token = normalize_auth_token(self.settings.auth_token)
        self._lock = threading.RLock()
        self._quote_fetch_executor_lock = threading.Lock()
        self._quote_fetch_executor: ThreadPoolExecutor | None = None
        self._quote_fetch_executor_finalizer: weakref.finalize | None = None
        self._shared_quote_cache: OrderedDict[str, QuotePayload] = OrderedDict()
        self._inflight_quote_fetches: dict[str, _InFlightQuoteFetch] = {}
        self._state_cache = FileBackedStateCache[AccountState]()
        self._registry = self._load_registry()

    def close(self, *, wait: bool = True) -> None:
        with self._quote_fetch_executor_lock:
            executor = self._quote_fetch_executor
            self._quote_fetch_executor = None
            finalizer = self._quote_fetch_executor_finalizer
            self._quote_fetch_executor_finalizer = None
        if finalizer is not None and finalizer.alive:
            finalizer.detach()
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=False)

    def _get_quote_fetch_executor(self) -> ThreadPoolExecutor:
        executor = self._quote_fetch_executor
        if executor is not None:
            return executor
        with self._quote_fetch_executor_lock:
            executor = self._quote_fetch_executor
            if executor is None:
                executor = ThreadPoolExecutor(
                    max_workers=self.quote_fetch_workers,
                    thread_name_prefix="trading-server-quotes",
                )
                self._quote_fetch_executor = executor
                self._quote_fetch_executor_finalizer = weakref.finalize(
                    self,
                    _shutdown_quote_fetch_executor,
                    executor,
                )
        return executor

    def _load_registry(self) -> dict[str, AccountConfig]:
        if not self.registry_path.exists():
            raise RuntimeError(
                f"Trading server registry missing: {self.registry_path} "
                f"(resolved from {self._registry_path_detail})"
            )
        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        accounts_raw = payload.get("accounts")
        if not isinstance(accounts_raw, dict) or not accounts_raw:
            raise RuntimeError(
                f"Trading server registry has no accounts: {self.registry_path} "
                f"(resolved from {self._registry_path_detail})"
            )

        accounts: dict[str, AccountConfig] = {}
        for account_name, config in accounts_raw.items():
            name = _normalize_account_name(account_name)
            if not isinstance(config, dict):
                raise RuntimeError(f"Trading server account config for {name} must be an object")
            try:
                mode = _normalize_trading_mode(config.get("mode", "paper"))
            except ValueError as exc:
                raise RuntimeError(f"Trading server account {name} has {exc}") from exc
            try:
                bot_id = _normalize_bot_id(config.get("allowed_bot_id", ""))
            except ValueError as exc:
                raise RuntimeError(f"Trading server account {name} has {exc}") from exc
            accounts[name] = {
                "name": name,
                "mode": mode,
                "allowed_bot_id": bot_id,
                "starting_cash": float(config.get("starting_cash", 100_000.0)),
                "paper_buying_power_multiplier": max(1.0, float(config.get("paper_buying_power_multiplier", 1.0))),
                "base_currency": str(config.get("base_currency", "USD")).strip().upper() or "USD",
                "sell_loss_cooldown_seconds": int(config.get("sell_loss_cooldown_seconds", 20 * 60)),
                "min_sell_markup_pct": float(config.get("min_sell_markup_pct", 0.001)),
                "symbols": [_normalize_symbol(symbol) for symbol in config.get("symbols", [])],
            }
        return accounts

    def configured_accounts(self) -> list[ConfiguredAccountSummary]:
        return [
            {
                "account": config["name"],
                "mode": config["mode"],
                "symbols": list(config["symbols"]),
            }
            for config in self._registry.values()
        ]

    def runtime_settings(self) -> dict[str, dict[str, object]]:
        return self.settings_resolution.as_dict()

    def _public_writer_claim(self, claim: WriterClaim | None) -> PublicWriterClaim | None:
        if not isinstance(claim, dict):
            return None
        return {
            "claimed_at": claim.get("claimed_at"),
            "expires_at": claim.get("expires_at"),
            "active": self._claim_is_active(claim, now=self.now_fn()),
        }

    def _public_order_payload(self, order: OrderRecord) -> dict[str, Any]:
        public: dict[str, Any] = {}
        for key in (
            "id",
            "symbol",
            "side",
            "qty",
            "limit_price",
            "status",
            "execution_mode",
            "allow_loss_exit",
            "force_exit_reason",
            "metadata",
            "created_at",
            "updated_at",
            "filled_at",
            "fill_price",
            "broker_response",
        ):
            if key in order:
                public[key] = order[key]
        return public

    def _config_for_account(self, account: str) -> AccountConfig:
        name = _normalize_account_name_or_400(account)
        config = self._registry.get(name)
        if config is None:
            raise HTTPException(status_code=404, detail=f"unknown trading server account: {name}")
        return config

    def _account_path(self, account: str) -> Path:
        return self.accounts_root / f"{_normalize_account_name(account)}.json"

    def _account_lock_path(self, account: str) -> Path:
        return self.locks_root / f"{_normalize_account_name(account)}.lock"

    def _fills_path(self, account: str) -> Path:
        return self.event_root / f"{_normalize_account_name(account)}.fills.jsonl"

    def _rejections_path(self, account: str) -> Path:
        return self.event_root / f"{_normalize_account_name(account)}.rejections.jsonl"

    def _audit_path(self, account: str) -> Path:
        return self.event_root / f"{_normalize_account_name(account)}.audit.jsonl"

    def _append_audit_event(self, account: str, event_type: str, **payload: object) -> None:
        try:
            append_jsonl_row(
                self._audit_path(account),
                {
                    "timestamp": _isoformat(self.now_fn()),
                    "account": account,
                    "event_type": event_type,
                    **payload,
                },
                sort_keys=True,
                default=str,
            )
        except Exception:
            return

    @contextmanager
    def _account_state_guard(self, account: str, *, write: bool = True) -> Iterator[None]:
        lock_path = self._account_lock_path(account)
        thread_guard: ReaderWriterGuard = shared_path_guard(lock_path)
        acquire = thread_guard.acquire_write if write else thread_guard.acquire_read
        release = thread_guard.release_write if write else thread_guard.release_read
        acquire()
        handle = None
        try:
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

    def _default_state(self, config: AccountConfig) -> AccountState:
        now = self.now_fn()
        return {
            "account": config["name"],
            "mode": config["mode"],
            "base_currency": config["base_currency"],
            "cash": float(config["starting_cash"]),
            "realized_pnl": 0.0,
            "positions": {},
            "open_orders": [],
            "order_history": [],
            "price_cache": {},
            "writer_claim": None,
            "created_at": _isoformat(now),
            "updated_at": _isoformat(now),
        }

    def _load_state_unlocked(self, account: str, config: AccountConfig) -> AccountState:
        path = self._account_path(account)
        cached_state, stat_key = self._state_cache.load(account, path)
        if cached_state is not None:
            return cached_state
        if stat_key is None:
            state = self._default_state(config)
            return self._state_cache.store(account, state, stat_key=None)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"Corrupt trading server account state: {path}")
        payload.setdefault("account", config["name"])
        try:
            payload["mode"] = _normalize_trading_mode(payload.get("mode", config["mode"]))
        except ValueError as exc:
            raise RuntimeError(f"Corrupt trading server account state: {path}: {exc}") from exc
        payload.setdefault("base_currency", config["base_currency"])
        payload.setdefault("cash", float(config["starting_cash"]))
        payload.setdefault("realized_pnl", 0.0)
        payload.setdefault("positions", {})
        payload.setdefault("open_orders", [])
        payload.setdefault("order_history", [])
        payload.setdefault("price_cache", {})
        payload.setdefault("writer_claim", None)
        state = cast(AccountState, payload)
        self._prune_order_history_unlocked(state)
        return self._state_cache.store(account, state, stat_key=stat_key)

    def _save_state_unlocked(self, state: AccountState) -> None:
        self._prune_order_history_unlocked(state)
        state["updated_at"] = _isoformat(self.now_fn())
        path = self._account_path(state["account"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(_serialize_account_state(state), encoding="utf-8")
        temp_path.replace(path)
        self._state_cache.store_for_path(state["account"], state, path)

    def _prune_order_history_unlocked(self, state: AccountState) -> None:
        history = state.setdefault("order_history", [])
        excess = len(history) - self.max_order_history
        if excess > 0:
            del history[:excess]

    def _claim_is_active(self, claim: WriterClaim | None, *, now: datetime) -> bool:
        if not claim:
            return False
        expires_at = _parse_ts(claim.get("expires_at"))
        if expires_at is None:
            return False
        return expires_at > now

    def _require_allowed_bot(self, config: AccountConfig, bot_id: str) -> str:
        try:
            normalized_bot_id = _normalize_bot_id(bot_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if normalized_bot_id != config["allowed_bot_id"]:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"bot_id {normalized_bot_id!r} is not allowed for account {config['name']}. "
                    "Update the registry and restart the trading server to change bot ownership."
                ),
            )
        return normalized_bot_id

    def claim_writer(self, request: WriterLeaseRequest) -> dict[str, Any]:
        config = self._config_for_account(request.account)
        account = config["name"]
        try:
            normalized_bot_id = self._require_allowed_bot(config, request.bot_id)
        except HTTPException as exc:
            self._append_audit_event(
                account,
                "writer_claim_rejected",
                bot_id=request.bot_id,
                session_id=request.session_id,
                detail=str(exc.detail),
            )
            raise
        now = self.now_fn()
        try:
            session_id = _normalize_session_id(request.session_id or uuid.uuid4())
        except ValueError as exc:
            self._append_audit_event(
                account,
                "writer_claim_rejected",
                bot_id=normalized_bot_id,
                detail=str(exc),
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        with self._account_state_guard(account):
            with self._lock:
                state = self._load_state_unlocked(account, config)
                existing = state.get("writer_claim")
                if self._claim_is_active(existing, now=now):
                    if str(existing.get("session_id")) != session_id:
                        detail = (
                            f"writer lease already held for account {account} by session "
                            f"{existing.get('session_id')}"
                        )
                        self._append_audit_event(
                            account,
                            "writer_claim_rejected",
                            bot_id=request.bot_id,
                            session_id=session_id,
                            detail=detail,
                        )
                        raise HTTPException(
                            status_code=409,
                            detail=detail,
                        )

                expires_at = now + timedelta(seconds=int(request.ttl_seconds))
                state["writer_claim"] = cast(
                    WriterClaim,
                    {
                        "bot_id": config["allowed_bot_id"],
                        "session_id": session_id,
                        "claimed_at": _isoformat(now),
                        "expires_at": _isoformat(expires_at),
                        "ttl_seconds": int(request.ttl_seconds),
                    },
                )
                self._save_state_unlocked(state)
                result = {
                    "account": account,
                    "bot_id": config["allowed_bot_id"],
                    "session_id": session_id,
                    "expires_at": _isoformat(expires_at),
                    "mode": config["mode"],
                }
                self._append_audit_event(
                    account,
                    "writer_claimed",
                    bot_id=config["allowed_bot_id"],
                    session_id=session_id,
                    expires_at=_isoformat(expires_at),
                    mode=config["mode"],
                )
                return result

    def heartbeat_writer(self, request: WriterLeaseRequest) -> dict[str, Any]:
        config = self._config_for_account(request.account)
        account = config["name"]
        try:
            normalized_bot_id = self._require_allowed_bot(config, request.bot_id)
        except HTTPException as exc:
            self._append_audit_event(
                account,
                "writer_heartbeat_rejected",
                bot_id=request.bot_id,
                session_id=request.session_id,
                detail=str(exc.detail),
            )
            raise
        now = self.now_fn()
        if request.session_id is None:
            self._append_audit_event(
                account,
                "writer_heartbeat_rejected",
                bot_id=normalized_bot_id,
                detail="session_id is required for heartbeat",
            )
            raise HTTPException(status_code=400, detail="session_id is required for heartbeat")
        try:
            session_id = _normalize_session_id(request.session_id)
        except ValueError as exc:
            self._append_audit_event(
                account,
                "writer_heartbeat_rejected",
                bot_id=normalized_bot_id,
                detail=str(exc),
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        with self._account_state_guard(account):
            with self._lock:
                state = self._load_state_unlocked(account, config)
                claim = state.get("writer_claim")
                if not self._claim_is_active(claim, now=now):
                    detail = f"writer lease is not active for account {account}"
                    self._append_audit_event(
                        account,
                        "writer_heartbeat_rejected",
                        bot_id=request.bot_id,
                        session_id=session_id,
                        detail=detail,
                    )
                    raise HTTPException(status_code=409, detail=f"writer lease is not active for account {account}")
                if str(claim.get("session_id")) != session_id:
                    detail = f"session_id does not own account {account}"
                    self._append_audit_event(
                        account,
                        "writer_heartbeat_rejected",
                        bot_id=request.bot_id,
                        session_id=session_id,
                        detail=detail,
                    )
                    raise HTTPException(status_code=409, detail=f"session_id does not own account {account}")

                expires_at = now + timedelta(seconds=int(request.ttl_seconds))
                claim["expires_at"] = _isoformat(expires_at)
                claim["ttl_seconds"] = int(request.ttl_seconds)
                state["writer_claim"] = claim
                self._save_state_unlocked(state)
                result = {
                    "account": account,
                    "session_id": session_id,
                    "expires_at": _isoformat(expires_at),
                }
                self._append_audit_event(
                    account,
                    "writer_heartbeat",
                    bot_id=config["allowed_bot_id"],
                    session_id=session_id,
                    expires_at=_isoformat(expires_at),
                )
                return result

    def _require_writer_claim(
        self,
        *,
        state: AccountState,
        config: AccountConfig,
        bot_id: str,
        session_id: str,
    ) -> None:
        self._require_allowed_bot(config, bot_id)
        try:
            normalized_session_id = _normalize_session_id(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        now = self.now_fn()
        claim = state.get("writer_claim")
        if not self._claim_is_active(claim, now=now):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"no active writer lease for account {config['name']}. "
                    "Claim the writer lease before submitting orders."
                ),
            )
        if str(claim.get("session_id")) != normalized_session_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"session_id {normalized_session_id!r} does not own account {config['name']}; "
                    f"active session is {claim.get('session_id')!r}"
                ),
            )

    def _write_rejection(
        self,
        *,
        account: str,
        request: OrderRequest,
        reason: str,
        detail: str,
    ) -> None:
        try:
            symbol = _normalize_symbol(request.symbol)
        except ValueError:
            symbol = str(request.symbol)
        try:
            side = _normalize_side(request.side)
        except ValueError:
            side = str(request.side)
        append_jsonl_row(
            self._rejections_path(account),
            {
                "rejected_at": _isoformat(self.now_fn()),
                "account": account,
                "bot_id": request.bot_id,
                "session_id": request.session_id,
                "symbol": symbol,
                "side": side,
                "qty": safe_json_float(request.qty),
                "limit_price": safe_json_float(request.limit_price),
                "reason": reason,
                "detail": detail,
                "metadata": _safe_metadata_for_rejection(getattr(request, "metadata", {})),
            },
            sort_keys=True,
        )

    def _quote_from_cache_unlocked(self, state: AccountState, symbol: str) -> QuotePayload | None:
        for quote in (state.get("price_cache", {}).get(symbol),):
            if not isinstance(quote, dict):
                continue
            as_of = _parse_ts(quote.get("as_of"))
            if as_of is None:
                continue
            age = (self.now_fn() - as_of).total_seconds()
            if age > self.quote_stale_seconds:
                continue
            return quote
        return self._shared_quote_from_cache_unlocked(symbol)

    def _shared_quote_from_cache_unlocked(self, symbol: str) -> QuotePayload | None:
        shared_quote = self._shared_quote_cache.get(symbol)
        if not isinstance(shared_quote, dict):
            self._shared_quote_cache.pop(symbol, None)
            return None
        as_of = _parse_ts(shared_quote.get("as_of"))
        if as_of is None:
            self._shared_quote_cache.pop(symbol, None)
            return None
        age = (self.now_fn() - as_of).total_seconds()
        if age > self.quote_stale_seconds:
            self._shared_quote_cache.pop(symbol, None)
            return None
        self._shared_quote_cache.move_to_end(symbol)
        return shared_quote

    def _remember_shared_quote_unlocked(self, symbol: str, quote: QuotePayload) -> None:
        self._shared_quote_cache.pop(symbol, None)
        self._shared_quote_cache[symbol] = quote
        while len(self._shared_quote_cache) > self.shared_quote_cache_size:
            self._shared_quote_cache.popitem(last=False)

    def _begin_inflight_quote_fetch_unlocked(
        self,
        symbol: str,
        *,
        force_refresh: bool,
    ) -> tuple[QuotePayload | None, _InFlightQuoteFetch, bool]:
        if not force_refresh:
            cached_quote = self._shared_quote_from_cache_unlocked(symbol)
            if cached_quote is not None:
                completed = _InFlightQuoteFetch()
                completed.quote = cached_quote
                completed.event.set()
                return cached_quote, completed, False
        inflight = self._inflight_quote_fetches.get(symbol)
        if inflight is not None:
            return None, inflight, False
        inflight = _InFlightQuoteFetch()
        self._inflight_quote_fetches[symbol] = inflight
        return None, inflight, True

    def _finish_inflight_quote_fetch_unlocked(
        self,
        symbol: str,
        inflight: _InFlightQuoteFetch,
        *,
        quote: QuotePayload | None,
        unavailable_reason: str | None,
    ) -> None:
        inflight.quote = quote
        inflight.unavailable_reason = unavailable_reason
        if quote is not None:
            self._remember_shared_quote_unlocked(symbol, quote)
        inflight.event.set()
        if self._inflight_quote_fetches.get(symbol) is inflight:
            self._inflight_quote_fetches.pop(symbol, None)

    def _default_quote_provider(self, symbol: str) -> QuotePayload | None:
        try:
            import alpaca_wrapper

            raw = alpaca_wrapper.latest_data(symbol)
        except Exception:
            return None
        bid = _coerce_float(getattr(raw, "bid_price", 0.0))
        ask = _coerce_float(getattr(raw, "ask_price", 0.0))
        last = _coerce_float(getattr(raw, "price", 0.0))
        midpoint = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
        if last <= 0:
            last = midpoint
        if bid <= 0:
            bid = last
        if ask <= 0:
            ask = last
        if max(bid, ask, last) <= 0:
            return None
        return {
            "symbol": _normalize_symbol(symbol),
            "bid_price": bid,
            "ask_price": ask,
            "last_price": last,
            "as_of": _isoformat(self.now_fn()),
        }

    def _normalize_quote_payload(self, quote: dict[str, Any] | QuotePayload, symbol: str) -> QuotePayload:
        normalized = {
            "symbol": _normalize_symbol(quote.get("symbol", symbol)),
            "bid_price": max(_coerce_float(quote.get("bid_price")), 0.0),
            "ask_price": max(_coerce_float(quote.get("ask_price")), 0.0),
            "last_price": max(_coerce_float(quote.get("last_price")), 0.0),
            "as_of": _isoformat(_parse_ts(quote.get("as_of")) or self.now_fn()),
        }
        if normalized["bid_price"] <= 0:
            normalized["bid_price"] = normalized["last_price"]
        if normalized["ask_price"] <= 0:
            normalized["ask_price"] = normalized["last_price"]
        return normalized

    def _fetch_quote(self, symbol: str) -> QuotePayload | None:
        quote = self.quote_provider(symbol)
        if quote is None:
            return None
        return self._normalize_quote_payload(quote, symbol)

    def _fetch_quote_with_coalescing(
        self,
        symbol: str,
        *,
        force_refresh: bool = False,
    ) -> tuple[QuotePayload | None, str | None]:
        with self._lock:
            cached_quote, inflight, is_leader = self._begin_inflight_quote_fetch_unlocked(
                symbol,
                force_refresh=force_refresh,
            )
        if cached_quote is not None:
            return cached_quote, None
        if not is_leader:
            inflight.event.wait()
            return inflight.quote, inflight.unavailable_reason

        quote: QuotePayload | None = None
        unavailable_reason: str | None = None
        try:
            quote = self._fetch_quote(symbol)
            if quote is None:
                unavailable_reason = _quote_unavailable_reason()
        except Exception as exc:
            unavailable_reason = _quote_unavailable_reason(exc)
        with self._lock:
            self._finish_inflight_quote_fetch_unlocked(
                symbol,
                inflight,
                quote=quote,
                unavailable_reason=unavailable_reason,
            )
        return quote, unavailable_reason

    def _fetch_quotes_for_symbols(
        self,
        symbols: Iterable[str],
        *,
        force_refresh_symbols: Iterable[str] | None = None,
    ) -> tuple[dict[str, QuotePayload], dict[str, str]]:
        unique_symbols = sorted({str(symbol).strip() for symbol in symbols if str(symbol).strip()})
        if not unique_symbols:
            return {}, {}
        forced_symbols = {
            str(symbol).strip()
            for symbol in (force_refresh_symbols or [])
            if str(symbol).strip()
        }
        if self.quote_fetch_workers <= 1 or len(unique_symbols) <= 1:
            fetched: dict[str, QuotePayload] = {}
            unavailable_reasons: dict[str, str] = {}
            for symbol in unique_symbols:
                quote, unavailable_reason = self._fetch_quote_with_coalescing(
                    symbol,
                    force_refresh=symbol in forced_symbols,
                )
                if quote is not None:
                    fetched[symbol] = quote
                    continue
                unavailable_reasons[symbol] = unavailable_reason or _quote_unavailable_reason()
            return fetched, unavailable_reasons

        fetched: dict[str, QuotePayload] = {}
        unavailable_reasons: dict[str, str] = {}
        executor = self._get_quote_fetch_executor()
        futures = {
            executor.submit(
                self._fetch_quote_with_coalescing,
                symbol,
                force_refresh=symbol in forced_symbols,
            ): symbol
            for symbol in unique_symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                quote, unavailable_reason = future.result()
            except Exception as exc:
                unavailable_reasons[symbol] = _quote_unavailable_reason(exc)
                continue
            if quote is not None:
                fetched[symbol] = quote
                continue
            unavailable_reasons[symbol] = unavailable_reason or _quote_unavailable_reason()
        return fetched, unavailable_reasons

    def _store_quote_unlocked(
        self,
        state: AccountState,
        symbol: str,
        quote: dict[str, Any] | QuotePayload,
        *,
        persist: bool = True,
    ) -> QuotePayload:
        normalized = self._normalize_quote_payload(quote, symbol)
        self._remember_shared_quote_unlocked(symbol, normalized)
        if persist:
            state.setdefault("price_cache", {})[symbol] = normalized
        return normalized

    def _refresh_quote_unlocked(
        self,
        state: AccountState,
        symbol: str,
        *,
        persist: bool = True,
    ) -> QuotePayload | None:
        normalized = self._fetch_quote(symbol)
        if normalized is None:
            return None
        return self._store_quote_unlocked(state, symbol, normalized, persist=persist)

    def _get_or_refresh_quote_unlocked(self, state: AccountState, symbol: str) -> QuotePayload:
        quote = self._quote_from_cache_unlocked(state, symbol)
        if quote is not None:
            return quote
        refreshed = self._refresh_quote_unlocked(state, symbol)
        if refreshed is None:
            raise HTTPException(status_code=503, detail=f"quote unavailable for {symbol}")
        return refreshed

    def _position_for_symbol(self, state: AccountState, symbol: str) -> PositionState | None:
        position = state.get("positions", {}).get(symbol)
        if not isinstance(position, dict):
            return None
        return position

    def _sell_floor_for_position(
        self,
        *,
        position: PositionState,
        config: AccountConfig,
        now: datetime,
    ) -> float:
        avg_entry = max(_coerce_float(position.get("avg_entry_price")), 0.0)
        if avg_entry <= 0:
            return 0.0
        cooldown_seconds = max(int(config["sell_loss_cooldown_seconds"]), 0)
        min_markup_pct = max(float(config["min_sell_markup_pct"]), 0.0)
        last_buy_at = _parse_ts(position.get("last_buy_at")) or _parse_ts(position.get("opened_at"))
        within_cooldown = False
        if last_buy_at is not None and cooldown_seconds > 0:
            within_cooldown = (now - last_buy_at).total_seconds() < cooldown_seconds
        if within_cooldown:
            return avg_entry * (1.0 + min_markup_pct)
        return avg_entry

    def _mark_price_for_symbol_unlocked(self, state: AccountState, symbol: str, position: PositionState) -> float:
        quote = self._quote_from_cache_unlocked(state, symbol)
        if quote is not None:
            for key in ("last_price", "bid_price", "ask_price"):
                price = _coerce_float(quote.get(key))
                if price > 0.0:
                    return price
        avg_entry = _coerce_float(position.get("avg_entry_price"))
        return max(avg_entry, 0.0)

    def _gross_exposure_unlocked(self, state: AccountState) -> float:
        gross = 0.0
        for symbol, position in state.get("positions", {}).items():
            if not isinstance(position, dict):
                continue
            qty = abs(_coerce_float(position.get("qty")))
            if qty <= 0.0:
                continue
            gross += qty * self._mark_price_for_symbol_unlocked(state, str(symbol), position)
        return gross

    def _account_equity_unlocked(self, state: AccountState) -> float:
        equity = _coerce_float(state.get("cash"))
        for symbol, position in state.get("positions", {}).items():
            if not isinstance(position, dict):
                continue
            qty = _coerce_float(position.get("qty"))
            if abs(qty) <= 0.0:
                continue
            equity += qty * self._mark_price_for_symbol_unlocked(state, str(symbol), position)
        return equity

    def _available_buying_power_unlocked(self, state: AccountState, config: AccountConfig) -> float:
        multiplier = 1.0
        if config["mode"] == "paper":
            multiplier = max(1.0, float(config.get("paper_buying_power_multiplier", 1.0)))
        equity = self._account_equity_unlocked(state)
        gross = self._gross_exposure_unlocked(state)
        return max(0.0, equity * multiplier - gross)

    def _validate_order_unlocked(
        self,
        *,
        request: OrderRequest,
        state: AccountState,
        config: AccountConfig,
    ) -> JsonObject:
        try:
            normalized_metadata = normalize_order_metadata(getattr(request, "metadata", {}))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        symbol = _normalize_symbol_or_400(request.symbol)
        side = _normalize_side(request.side)
        try:
            qty = normalize_positive_finite_float(request.qty, field_name="qty")
            limit_price = normalize_positive_finite_float(request.limit_price, field_name="limit_price")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        now = self.now_fn()

        if request.execution_mode != config["mode"]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"account {config['name']} is configured for mode={config['mode']}; "
                    f"request execution_mode={request.execution_mode} does not match"
                ),
            )

        if request.execution_mode == "live":
            if str(request.live_ack).strip().upper() != "LIVE":
                raise HTTPException(
                    status_code=400,
                    detail="live order rejected: live_ack must equal LIVE",
                )
            if str(os.getenv("ALLOW_ALPACA_LIVE_TRADING", "")).strip() not in {"1", "true", "TRUE"}:
                raise HTTPException(
                    status_code=403,
                    detail="live order rejected: ALLOW_ALPACA_LIVE_TRADING=1 is required",
                )

        if side == "buy":
            buying_power = self._available_buying_power_unlocked(state, config)
            notional = qty * limit_price
            if request.execution_mode == "paper" and buying_power + 1e-9 < notional:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"insufficient paper buying power for {symbol}: need {notional:.2f}, have {buying_power:.2f}"
                    ),
                )
            return normalized_metadata

        position = self._position_for_symbol(state, symbol)
        if position is None:
            raise HTTPException(status_code=400, detail=f"cannot sell {symbol}: no open position")
        current_qty = _coerce_float(position.get("qty"))
        if qty > current_qty + 1e-9:
            raise HTTPException(
                status_code=400,
                detail=f"cannot sell {qty:.8f} {symbol}: current position is {current_qty:.8f}",
            )
        if request.allow_loss_exit:
            if not str(request.force_exit_reason or "").strip():
                raise HTTPException(
                    status_code=400,
                    detail="allow_loss_exit requires a non-empty force_exit_reason",
                )
            return normalized_metadata

        sell_floor = self._sell_floor_for_position(position=position, config=config, now=now)
        if limit_price + 1e-9 < sell_floor:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"sell rejected for {symbol}: limit {limit_price:.4f} is below safety floor "
                    f"{sell_floor:.4f}. Loss-taking exits require allow_loss_exit=true and a force_exit_reason."
                ),
            )
        return normalized_metadata

    def _fill_price_for_order(self, order: OrderRecord, quote: QuotePayload) -> float | None:
        side = _normalize_side(order["side"])
        limit_price = float(order["limit_price"])
        bid = max(_coerce_float(quote.get("bid_price")), 0.0)
        ask = max(_coerce_float(quote.get("ask_price")), 0.0)
        last = max(_coerce_float(quote.get("last_price")), 0.0)
        if side == "buy":
            marketable = ask if ask > 0 else last
            if marketable > 0 and limit_price + 1e-9 >= marketable:
                return min(limit_price, marketable)
            return None
        marketable = bid if bid > 0 else last
        if marketable > 0 and limit_price <= marketable + 1e-9:
            return max(limit_price, marketable)
        return None

    def _record_fill_unlocked(
        self,
        *,
        state: AccountState,
        config: AccountConfig,
        order: OrderRecord,
        fill_price: float,
        filled_at: datetime,
    ) -> None:
        symbol = order["symbol"]
        side = order["side"]
        qty = float(order["qty"])
        positions = state.setdefault("positions", {})
        cash = _coerce_float(state.get("cash"))
        realized_pnl = _coerce_float(state.get("realized_pnl"))

        if side == "buy":
            position = dict(positions.get(symbol) or {})
            current_qty = _coerce_float(position.get("qty"))
            current_avg = _coerce_float(position.get("avg_entry_price"))
            new_qty = current_qty + qty
            weighted_cost = (current_qty * current_avg) + (qty * fill_price)
            new_avg = weighted_cost / new_qty if new_qty > 0 else 0.0
            positions[symbol] = {
                "symbol": symbol,
                "qty": new_qty,
                "avg_entry_price": new_avg,
                "opened_at": position.get("opened_at") or _isoformat(filled_at),
                "last_buy_at": _isoformat(filled_at),
                "realized_pnl": _coerce_float(position.get("realized_pnl")),
            }
            state["cash"] = cash - (qty * fill_price)
        else:
            position = dict(positions.get(symbol) or {})
            current_qty = _coerce_float(position.get("qty"))
            avg_entry = _coerce_float(position.get("avg_entry_price"))
            remaining_qty = current_qty - qty
            trade_realized = (fill_price - avg_entry) * qty
            state["cash"] = cash + (qty * fill_price)
            state["realized_pnl"] = realized_pnl + trade_realized
            position["realized_pnl"] = _coerce_float(position.get("realized_pnl")) + trade_realized
            if remaining_qty <= 1e-9:
                positions.pop(symbol, None)
            else:
                position["qty"] = remaining_qty
                positions[symbol] = position

        order["status"] = "filled"
        order["filled_at"] = _isoformat(filled_at)
        order["fill_price"] = fill_price
        state.setdefault("order_history", []).append(order)
        append_jsonl_row(
            self._fills_path(state["account"]),
            {
                "filled_at": _isoformat(filled_at),
                "account": state["account"],
                "mode": config["mode"],
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "limit_price": float(order["limit_price"]),
                "fill_price": float(fill_price),
                "bot_id": order["bot_id"],
                "session_id": order["session_id"],
                "metadata": order.get("metadata", {}),
            },
            sort_keys=True,
        )

    def _submit_paper_order_unlocked(
        self,
        *,
        state: AccountState,
        config: AccountConfig,
        request: OrderRequest,
        normalized_metadata: JsonObject,
    ) -> SubmitOrderResult:
        symbol = _normalize_symbol(request.symbol)
        order = cast(
            OrderRecord,
            {
                "id": str(uuid.uuid4()),
                "account": state["account"],
                "bot_id": request.bot_id,
                "session_id": request.session_id,
                "symbol": symbol,
                "side": _normalize_side(request.side),
                "qty": float(request.qty),
                "limit_price": float(request.limit_price),
                "status": "open",
                "execution_mode": "paper",
                "allow_loss_exit": bool(request.allow_loss_exit),
                "force_exit_reason": request.force_exit_reason,
                "metadata": normalized_metadata,
                "created_at": _isoformat(self.now_fn()),
                "filled_at": None,
                "fill_price": None,
            },
        )
        quote = self._get_or_refresh_quote_unlocked(state, symbol)
        fill_price = self._fill_price_for_order(order, quote)
        if fill_price is None:
            state.setdefault("open_orders", []).append(order)
            return {"order": order, "quote": quote, "filled": False}

        self._record_fill_unlocked(
            state=state,
            config=config,
            order=order,
            fill_price=fill_price,
            filled_at=self.now_fn(),
        )
        return {"order": order, "quote": quote, "filled": True}

    def _default_live_executor(self, order: OrderRecord) -> BrokerResponse | None:
        import alpaca_wrapper

        result = alpaca_wrapper.open_order_at_price_or_all(
            order["symbol"],
            order["qty"],
            order["side"],
            order["limit_price"],
        )
        if result is None:
            return None
        return {
            "broker_order_id": str(getattr(result, "id", "")),
            "status": str(getattr(result, "status", "accepted")),
        }

    def _submit_live_order_unlocked(
        self,
        *,
        state: AccountState,
        config: AccountConfig,
        request: OrderRequest,
        normalized_metadata: JsonObject,
    ) -> SubmitOrderResult:
        order = cast(
            OrderRecord,
            {
                "id": str(uuid.uuid4()),
                "account": state["account"],
                "bot_id": request.bot_id,
                "session_id": request.session_id,
                "symbol": _normalize_symbol(request.symbol),
                "side": _normalize_side(request.side),
                "qty": float(request.qty),
                "limit_price": float(request.limit_price),
                "status": "open",
                "execution_mode": "live",
                "allow_loss_exit": bool(request.allow_loss_exit),
                "force_exit_reason": request.force_exit_reason,
                "metadata": normalized_metadata,
                "created_at": _isoformat(self.now_fn()),
                "filled_at": None,
                "fill_price": None,
            },
        )
        try:
            broker_response = self.live_executor(order)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"live broker error for {order['symbol']}: {type(exc).__name__}: {exc}",
            ) from exc
        if broker_response is None:
            raise HTTPException(status_code=502, detail=f"live broker rejected order for {order['symbol']}")
        order["broker_response"] = broker_response
        quote = self._refresh_quote_unlocked(state, order["symbol"])
        if quote is not None:
            fill_price = self._fill_price_for_order(order, quote)
            if fill_price is not None:
                self._record_fill_unlocked(
                    state=state,
                    config=config,
                    order=order,
                    fill_price=fill_price,
                    filled_at=self.now_fn(),
                )
                return {"order": order, "filled": True, "quote": quote}
        state.setdefault("open_orders", []).append(order)
        return {"order": order, "filled": False, "quote": quote}

    def submit_order(self, request: OrderRequest) -> SubmitOrderResult:
        config = self._config_for_account(request.account)
        account = config["name"]
        with self._account_state_guard(account):
            with self._lock:
                state = self._load_state_unlocked(account, config)
                try:
                    self._require_writer_claim(
                        state=state,
                        config=config,
                        bot_id=request.bot_id,
                        session_id=request.session_id,
                    )
                    normalized_metadata = self._validate_order_unlocked(
                        request=request,
                        state=state,
                        config=config,
                    )
                except HTTPException as exc:
                    self._write_rejection(
                        account=account,
                        request=request,
                        reason="order_rejected",
                        detail=str(exc.detail),
                    )
                    self._append_audit_event(
                        account,
                        "order_rejected",
                        bot_id=request.bot_id,
                        session_id=request.session_id,
                        symbol=str(request.symbol),
                        side=str(request.side),
                        execution_mode=str(request.execution_mode),
                        detail=str(exc.detail),
                    )
                    raise

                try:
                    if request.execution_mode == "paper":
                        result = self._submit_paper_order_unlocked(
                            state=state,
                            config=config,
                            request=request,
                            normalized_metadata=normalized_metadata,
                        )
                    else:
                        result = self._submit_live_order_unlocked(
                            state=state,
                            config=config,
                            request=request,
                            normalized_metadata=normalized_metadata,
                        )
                except HTTPException as exc:
                    self._write_rejection(
                        account=account,
                        request=request,
                        reason="order_submit_failed",
                        detail=str(exc.detail),
                    )
                    self._append_audit_event(
                        account,
                        "order_submit_failed",
                        bot_id=request.bot_id,
                        session_id=request.session_id,
                        symbol=str(request.symbol),
                        side=str(request.side),
                        execution_mode=str(request.execution_mode),
                        qty=safe_json_float(request.qty),
                        limit_price=safe_json_float(request.limit_price),
                        allow_loss_exit=bool(request.allow_loss_exit),
                        force_exit_reason=request.force_exit_reason,
                        status_code=exc.status_code,
                        detail=str(exc.detail),
                    )
                    raise
                self._save_state_unlocked(state)
                order = result.get("order") or {}
                broker_response = order.get("broker_response") if isinstance(order.get("broker_response"), dict) else {}
                self._append_audit_event(
                    account,
                    "order_submitted",
                    bot_id=request.bot_id,
                    session_id=request.session_id,
                    order_id=order.get("id"),
                    symbol=order.get("symbol"),
                    side=order.get("side"),
                    execution_mode=order.get("execution_mode"),
                    status=order.get("status"),
                    qty=order.get("qty"),
                    limit_price=order.get("limit_price"),
                    allow_loss_exit=order.get("allow_loss_exit"),
                    force_exit_reason=order.get("force_exit_reason"),
                    filled=bool(result.get("filled")),
                    fill_price=order.get("fill_price"),
                    broker_order_id=broker_response.get("broker_order_id"),
                    broker_status=broker_response.get("status"),
                )
                return result

    def _attempt_open_order_fills_unlocked(
        self,
        *,
        state: AccountState,
        config: AccountConfig,
        symbols_filter: set[str] | None = None,
        quote_overrides: dict[str, QuotePayload] | None = None,
        refresh_missing_quotes: bool = True,
    ) -> list[OrderRecord]:
        open_orders = list(state.get("open_orders", []))
        if not open_orders:
            return []
        remaining_orders: list[OrderRecord] = []
        filled_orders: list[OrderRecord] = []
        for order in open_orders:
            symbol = _normalize_symbol(order["symbol"])
            if symbols_filter is not None and symbol not in symbols_filter:
                remaining_orders.append(order)
                continue
            quote = None
            if quote_overrides and symbol in quote_overrides:
                quote = self._store_quote_unlocked(state, symbol, quote_overrides[symbol])
            elif refresh_missing_quotes:
                quote = self._get_or_refresh_quote_unlocked(state, symbol)
            else:
                quote = self._quote_from_cache_unlocked(state, symbol)
            if quote is None:
                remaining_orders.append(order)
                continue
            fill_price = self._fill_price_for_order(order, quote)
            if fill_price is None:
                remaining_orders.append(order)
                continue
            self._record_fill_unlocked(
                state=state,
                config=config,
                order=order,
                fill_price=fill_price,
                filled_at=self.now_fn(),
            )
            filled_orders.append(order)
        state["open_orders"] = remaining_orders
        return filled_orders

    def refresh_prices(
        self,
        *,
        account: str | None = None,
        symbols: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        requested_symbols = {_normalize_symbol_or_400(symbol) for symbol in (symbols or []) if str(symbol).strip()}
        account_symbols: dict[str, set[str]] = {}
        open_order_symbols_to_refresh: set[str] = set()
        target_accounts = [self._config_for_account(account)["name"]] if account else list(self._registry.keys())
        for account_name in target_accounts:
            with self._account_state_guard(account_name, write=False):
                with self._lock:
                    config = self._config_for_account(account_name)
                    state = self._load_state_unlocked(account_name, config)
                    symbols_to_refresh: set[str] = set(requested_symbols)
                    open_order_symbols = {
                        _normalize_symbol(order["symbol"])
                        for order in state.get("open_orders", [])
                    }
                    if not symbols_to_refresh:
                        symbols_to_refresh.update(config["symbols"])
                        symbols_to_refresh.update(state.get("positions", {}).keys())
                        symbols_to_refresh.update(open_order_symbols)
                        symbols_to_refresh.update(state.get("price_cache", {}).keys())
                    account_symbols[account_name] = symbols_to_refresh
                    open_order_symbols_to_refresh.update(
                        symbol for symbol in open_order_symbols if symbol in symbols_to_refresh
                    )

        fetched_quotes, unavailable_reasons = self._fetch_quotes_for_symbols(
            {symbol for symbols_to_refresh in account_symbols.values() for symbol in symbols_to_refresh},
            force_refresh_symbols=open_order_symbols_to_refresh,
        )

        refreshed_accounts: list[dict[str, Any]] = []
        for account_name, symbols_to_refresh in account_symbols.items():
            with self._account_state_guard(account_name):
                with self._lock:
                    config = self._config_for_account(account_name)
                    state = self._load_state_unlocked(account_name, config)
                    refreshed_symbols: list[str] = []
                    for symbol in sorted(symbols_to_refresh):
                        quote = fetched_quotes.get(symbol)
                        if quote is not None:
                            self._store_quote_unlocked(state, symbol, quote, persist=False)
                            refreshed_symbols.append(symbol)
                    fills = self._attempt_open_order_fills_unlocked(
                        state=state,
                        config=config,
                        symbols_filter=symbols_to_refresh or None,
                        quote_overrides=fetched_quotes,
                        refresh_missing_quotes=False,
                    )
                    if fills:
                        self._save_state_unlocked(state)
                    unavailable_symbols = sorted(symbol for symbol in symbols_to_refresh if symbol not in fetched_quotes)
                    account_quote_error_symbols = sorted(
                        symbol
                        for symbol in unavailable_symbols
                        if unavailable_reasons.get(symbol) != "no quote returned"
                    )
                    account_unavailable_reasons = {
                        symbol: unavailable_reasons[symbol]
                        for symbol in unavailable_symbols
                        if symbol in unavailable_reasons
                    }
                    self._append_audit_event(
                        account_name,
                        "prices_refreshed",
                        refreshed_symbols=refreshed_symbols,
                        unavailable_symbols=unavailable_symbols,
                        quote_error_symbols=account_quote_error_symbols,
                        unavailable_reasons=account_unavailable_reasons,
                        filled_orders=[order["id"] for order in fills],
                    )
                    refreshed_accounts.append(
                        {
                            "account": account_name,
                            "refreshed_symbols": refreshed_symbols,
                            "unavailable_symbols": unavailable_symbols,
                            "quote_error_symbols": account_quote_error_symbols,
                            "unavailable_reasons": account_unavailable_reasons,
                            "filled_orders": [order["id"] for order in fills],
                        }
                    )
        return {"accounts": refreshed_accounts}

    def get_account_snapshot(self, account: str) -> dict[str, Any]:
        config = self._config_for_account(account)
        with self._account_state_guard(account, write=False):
            with self._lock:
                state = self._load_state_unlocked(account, config)
                equity = self._account_equity_unlocked(state)
                buying_power = self._available_buying_power_unlocked(state, config)
                return {
                    "account": account,
                    "mode": config["mode"],
                    "cash": _coerce_float(state.get("cash")),
                    "equity": equity,
                    "buying_power": buying_power,
                    "realized_pnl": _coerce_float(state.get("realized_pnl")),
                    "positions": state.get("positions", {}),
                    "open_orders": [self._public_order_payload(order) for order in state.get("open_orders", [])],
                    "writer_claim": self._public_writer_claim(state.get("writer_claim")),
                    "updated_at": state.get("updated_at"),
                }

    def get_orders(self, account: str, *, include_history: bool = False) -> dict[str, Any]:
        config = self._config_for_account(account)
        with self._account_state_guard(account, write=False):
            with self._lock:
                state = self._load_state_unlocked(account, config)
                payload = {
                    "account": account,
                    "open_orders": [self._public_order_payload(order) for order in state.get("open_orders", [])],
                }
                if include_history:
                    payload["order_history"] = [
                        self._public_order_payload(order) for order in state.get("order_history", [])
                    ]
                return payload


_BackgroundRefreshHandle = BackgroundRefreshHandle
_background_registry = BackgroundRefreshRegistry[TradingServerEngine](
    thread_name_prefix="trading-server-refresh",
    min_poll_seconds=1,
)
_background_lock = _background_registry.lock
_background_refreshers = _background_registry.refreshers


def ensure_background_refresh(engine: TradingServerEngine, *, poll_seconds: int | None = None) -> threading.Thread:
    return _background_registry.ensure(
        engine,
        refresh_fn=lambda current: current.refresh_prices(),
        default_poll_seconds=engine.settings.background_poll_seconds,
        poll_seconds=poll_seconds,
    )


def stop_background_refresh(engine: TradingServerEngine | None = None, timeout: float = 1.0) -> None:
    _background_registry.stop(engine, timeout=timeout)


def create_app(engine: TradingServerEngine | None = None) -> FastAPI:
    engine = engine or TradingServerEngine()
    poll_seconds = engine.settings.background_poll_seconds
    expected_auth_token = normalize_auth_token(engine.auth_token)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        ensure_background_refresh(engine, poll_seconds=poll_seconds)
        try:
            yield
        finally:
            try:
                stop_background_refresh(engine)
            finally:
                engine.close(wait=True)

    app = FastAPI(lifespan=_lifespan)

    def require_auth(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> None:
        if bearer_auth_matches(expected_token=expected_auth_token, authorization=authorization):
            return
        client_host = request.client.host if request.client is not None else "unknown"
        logger.warning(
            "Rejected unauthorized trading server request: method=%s path=%s client=%s auth=%s",
            request.method,
            request.url.path,
            client_host,
            classify_bearer_auth_failure(authorization),
        )
        raise HTTPException(
            status_code=401,
            detail="invalid auth token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.get("/api/v1/accounts", dependencies=[Depends(require_auth)])
    def list_accounts():
        return {"accounts": engine.configured_accounts()}

    @app.get("/api/v1/runtime-config", dependencies=[Depends(require_auth)])
    def runtime_config():
        return engine.runtime_settings()

    @app.post("/api/v1/writer/claim", dependencies=[Depends(require_auth)])
    def claim_writer(request: WriterLeaseRequest):
        return engine.claim_writer(request)

    @app.post("/api/v1/writer/heartbeat", dependencies=[Depends(require_auth)])
    def heartbeat_writer(request: WriterLeaseRequest):
        return engine.heartbeat_writer(request)

    @app.get("/api/v1/account/{account}", dependencies=[Depends(require_auth)])
    def get_account(account: str):
        return engine.get_account_snapshot(account)

    @app.get("/api/v1/orders/{account}", dependencies=[Depends(require_auth)])
    def get_orders(
        account: str,
        include_history: bool = Query(default=False),
    ):
        return engine.get_orders(account, include_history=bool(include_history))

    @app.post("/api/v1/orders", dependencies=[Depends(require_auth)])
    def submit_order(request: OrderRequest):
        return engine.submit_order(request)

    @app.post("/api/v1/prices/refresh", dependencies=[Depends(require_auth)])
    def refresh_prices(request: RefreshPricesRequest):
        return engine.refresh_prices(account=request.account, symbols=request.symbols)

    return app


try:
    app = create_app()
except RuntimeError:
    # Registry file may not exist in CI/test environments; tests construct their own engine.
    app = None  # type: ignore[assignment]
