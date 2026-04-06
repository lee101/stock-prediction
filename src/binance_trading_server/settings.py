from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.server_settings_utils import (
    IntResolution,
    PathResolution as RegistryPathResolution,
    SecretResolution,
    resolve_explicit_or_env_int_resolution,
    resolve_optional_secret,
    resolve_optional_secret_resolution,
    resolve_repo_relative_path,
)

REPO = Path(__file__).resolve().parents[2]

BINANCE_TRADING_SERVER_BASE_URL_ENV = "BINANCE_TRADING_SERVER_URL"
BINANCE_TS_REGISTRY_PATH_ENV = "BINANCE_TRADING_SERVER_REGISTRY_PATH"
BINANCE_TS_QUOTE_STALE_SECONDS_ENV = "BINANCE_TRADING_SERVER_QUOTE_STALE_SECONDS"
BINANCE_TS_WRITER_TTL_SECONDS_ENV = "BINANCE_TRADING_SERVER_WRITER_TTL_SECONDS"
BINANCE_TS_BACKGROUND_POLL_SECONDS_ENV = "BINANCE_TRADING_SERVER_BACKGROUND_POLL_SECONDS"
BINANCE_TS_QUOTE_FETCH_WORKERS_ENV = "BINANCE_TRADING_SERVER_QUOTE_FETCH_WORKERS"
BINANCE_TS_SIM_POLL_SECONDS_ENV = "BINANCE_TRADING_SERVER_SIM_POLL_SECONDS"
BINANCE_TS_MAX_ORDER_HISTORY_ENV = "BINANCE_TRADING_SERVER_MAX_ORDER_HISTORY"
BINANCE_TS_SHARED_QUOTE_CACHE_SIZE_ENV = "BINANCE_TRADING_SERVER_SHARED_QUOTE_CACHE_SIZE"
BINANCE_TS_AUTH_TOKEN_ENV = "BINANCE_TRADING_SERVER_AUTH_TOKEN"

DEFAULT_REGISTRY_PATH = REPO / "config" / "binance_trading_server" / "accounts.json"
DEFAULT_QUOTE_STALE_SECONDS = 30
DEFAULT_WRITER_TTL_SECONDS = 120
DEFAULT_BACKGROUND_POLL_SECONDS = 60
DEFAULT_QUOTE_FETCH_WORKERS = 4
DEFAULT_SIM_POLL_SECONDS = 60
DEFAULT_MAX_ORDER_HISTORY = 1000
DEFAULT_SHARED_QUOTE_CACHE_SIZE = 512
DEFAULT_PORT = 8060
DEFAULT_BINANCE_TRADING_SERVER_BASE_URL = f"http://127.0.0.1:{DEFAULT_PORT}"

MIN_QUOTE_STALE_SECONDS = 1
MIN_WRITER_TTL_SECONDS = 10
MAX_WRITER_TTL_SECONDS = 3600
MIN_BACKGROUND_POLL_SECONDS = 10
MIN_SIM_POLL_SECONDS = 10
MIN_QUOTE_FETCH_WORKERS = 1
MIN_MAX_ORDER_HISTORY = 1
MIN_SHARED_QUOTE_CACHE_SIZE = 1
MAX_ACCOUNT_NAME_LENGTH = 64
MAX_SYMBOL_LENGTH = 20


def resolve_binance_trading_server_base_url(base_url: str | None = None) -> str:
    raw_url = (
        base_url
        if base_url is not None
        else os.getenv(BINANCE_TRADING_SERVER_BASE_URL_ENV, DEFAULT_BINANCE_TRADING_SERVER_BASE_URL)
    )
    return str(raw_url).strip().rstrip("/")


def resolve_registry_path(path: str | Path | None = None) -> Path:
    return resolve_registry_path_resolution(path).path


def resolve_registry_path_resolution(path: str | Path | None = None) -> RegistryPathResolution:
    return resolve_repo_relative_path(
        path,
        repo_root=REPO,
        env_name=BINANCE_TS_REGISTRY_PATH_ENV,
        default_path=DEFAULT_REGISTRY_PATH,
        explicit_label="registry_path",
    )


def resolve_auth_token(auth_token: str | None = None) -> str | None:
    return resolve_optional_secret(
        auth_token,
        env_name=BINANCE_TS_AUTH_TOKEN_ENV,
        explicit_label="auth_token",
    )


@dataclass(frozen=True)
class BinanceTradingServerSettings:
    registry_path: Path
    quote_stale_seconds: int
    writer_ttl_seconds: int
    background_poll_seconds: int
    quote_fetch_workers: int
    sim_poll_seconds: int
    max_order_history: int
    shared_quote_cache_size: int
    auth_token: str | None

    @classmethod
    def from_env(
        cls,
        *,
        registry_path: str | Path | None = None,
        quote_stale_seconds: int | None = None,
        writer_ttl_seconds: int | None = None,
        background_poll_seconds: int | None = None,
        quote_fetch_workers: int | None = None,
        sim_poll_seconds: int | None = None,
        max_order_history: int | None = None,
        shared_quote_cache_size: int | None = None,
        auth_token: str | None = None,
    ) -> BinanceTradingServerSettings:
        return resolve_settings_resolution(
            registry_path=registry_path,
            quote_stale_seconds=quote_stale_seconds,
            writer_ttl_seconds=writer_ttl_seconds,
            background_poll_seconds=background_poll_seconds,
            quote_fetch_workers=quote_fetch_workers,
            sim_poll_seconds=sim_poll_seconds,
            max_order_history=max_order_history,
            shared_quote_cache_size=shared_quote_cache_size,
            auth_token=auth_token,
        ).settings()


@dataclass(frozen=True)
class BinanceTradingServerSettingsResolution:
    registry_path: RegistryPathResolution
    quote_stale_seconds: IntResolution
    writer_ttl_seconds: IntResolution
    background_poll_seconds: IntResolution
    quote_fetch_workers: IntResolution
    sim_poll_seconds: IntResolution
    max_order_history: IntResolution
    shared_quote_cache_size: IntResolution
    auth_token: SecretResolution

    def settings(self) -> BinanceTradingServerSettings:
        return BinanceTradingServerSettings(
            registry_path=self.registry_path.path,
            quote_stale_seconds=self.quote_stale_seconds.value,
            writer_ttl_seconds=self.writer_ttl_seconds.value,
            background_poll_seconds=self.background_poll_seconds.value,
            quote_fetch_workers=self.quote_fetch_workers.value,
            sim_poll_seconds=self.sim_poll_seconds.value,
            max_order_history=self.max_order_history.value,
            shared_quote_cache_size=self.shared_quote_cache_size.value,
            auth_token=self.auth_token.secret,
        )

    def as_dict(self) -> dict[str, dict[str, object]]:
        return {
            "registry_path": self.registry_path.as_dict(),
            "quote_stale_seconds": self.quote_stale_seconds.as_dict(),
            "writer_ttl_seconds": self.writer_ttl_seconds.as_dict(),
            "background_poll_seconds": self.background_poll_seconds.as_dict(),
            "quote_fetch_workers": self.quote_fetch_workers.as_dict(),
            "sim_poll_seconds": self.sim_poll_seconds.as_dict(),
            "max_order_history": self.max_order_history.as_dict(),
            "shared_quote_cache_size": self.shared_quote_cache_size.as_dict(),
            "auth_token": self.auth_token.as_dict(),
        }


def resolve_settings_resolution(
    *,
    registry_path: str | Path | None = None,
    quote_stale_seconds: int | None = None,
    writer_ttl_seconds: int | None = None,
    background_poll_seconds: int | None = None,
    quote_fetch_workers: int | None = None,
    sim_poll_seconds: int | None = None,
    max_order_history: int | None = None,
    shared_quote_cache_size: int | None = None,
    auth_token: str | None = None,
) -> BinanceTradingServerSettingsResolution:
    return BinanceTradingServerSettingsResolution(
        registry_path=resolve_registry_path_resolution(registry_path),
        quote_stale_seconds=resolve_explicit_or_env_int_resolution(
            quote_stale_seconds,
            env_name=BINANCE_TS_QUOTE_STALE_SECONDS_ENV,
            default=DEFAULT_QUOTE_STALE_SECONDS,
            minimum=MIN_QUOTE_STALE_SECONDS,
            explicit_label="quote_stale_seconds",
        ),
        writer_ttl_seconds=resolve_explicit_or_env_int_resolution(
            writer_ttl_seconds,
            env_name=BINANCE_TS_WRITER_TTL_SECONDS_ENV,
            default=DEFAULT_WRITER_TTL_SECONDS,
            minimum=MIN_WRITER_TTL_SECONDS,
            maximum=MAX_WRITER_TTL_SECONDS,
            explicit_label="writer_ttl_seconds",
        ),
        background_poll_seconds=resolve_explicit_or_env_int_resolution(
            background_poll_seconds,
            env_name=BINANCE_TS_BACKGROUND_POLL_SECONDS_ENV,
            default=DEFAULT_BACKGROUND_POLL_SECONDS,
            minimum=MIN_BACKGROUND_POLL_SECONDS,
            explicit_label="background_poll_seconds",
        ),
        quote_fetch_workers=resolve_explicit_or_env_int_resolution(
            quote_fetch_workers,
            env_name=BINANCE_TS_QUOTE_FETCH_WORKERS_ENV,
            default=DEFAULT_QUOTE_FETCH_WORKERS,
            minimum=MIN_QUOTE_FETCH_WORKERS,
            explicit_label="quote_fetch_workers",
        ),
        sim_poll_seconds=resolve_explicit_or_env_int_resolution(
            sim_poll_seconds,
            env_name=BINANCE_TS_SIM_POLL_SECONDS_ENV,
            default=DEFAULT_SIM_POLL_SECONDS,
            minimum=MIN_SIM_POLL_SECONDS,
            explicit_label="sim_poll_seconds",
        ),
        max_order_history=resolve_explicit_or_env_int_resolution(
            max_order_history,
            env_name=BINANCE_TS_MAX_ORDER_HISTORY_ENV,
            default=DEFAULT_MAX_ORDER_HISTORY,
            minimum=MIN_MAX_ORDER_HISTORY,
            explicit_label="max_order_history",
        ),
        shared_quote_cache_size=resolve_explicit_or_env_int_resolution(
            shared_quote_cache_size,
            env_name=BINANCE_TS_SHARED_QUOTE_CACHE_SIZE_ENV,
            default=DEFAULT_SHARED_QUOTE_CACHE_SIZE,
            minimum=MIN_SHARED_QUOTE_CACHE_SIZE,
            explicit_label="shared_quote_cache_size",
        ),
        auth_token=resolve_optional_secret_resolution(
            auth_token,
            env_name=BINANCE_TS_AUTH_TOKEN_ENV,
            explicit_label="auth_token",
        ),
    )
