from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRADING_SERVER_REGISTRY_PATH_ENV = "TRADING_SERVER_REGISTRY_PATH"
TRADING_SERVER_QUOTE_STALE_SECONDS_ENV = "TRADING_SERVER_QUOTE_STALE_SECONDS"
TRADING_SERVER_WRITER_TTL_SECONDS_ENV = "TRADING_SERVER_WRITER_TTL_SECONDS"
TRADING_SERVER_BACKGROUND_POLL_SECONDS_ENV = "TRADING_SERVER_BACKGROUND_POLL_SECONDS"
TRADING_SERVER_QUOTE_FETCH_WORKERS_ENV = "TRADING_SERVER_QUOTE_FETCH_WORKERS"
TRADING_SERVER_MAX_ORDER_HISTORY_ENV = "TRADING_SERVER_MAX_ORDER_HISTORY"
DEFAULT_REGISTRY_PATH = REPO / "config" / "trading_server" / "accounts.json"
DEFAULT_QUOTE_STALE_SECONDS = 90
DEFAULT_WRITER_TTL_SECONDS = 120
DEFAULT_BACKGROUND_POLL_SECONDS = 60
DEFAULT_QUOTE_FETCH_WORKERS = 4
DEFAULT_MAX_ORDER_HISTORY = 1000
MIN_QUOTE_STALE_SECONDS = 1
MIN_WRITER_TTL_SECONDS = 10
MAX_WRITER_TTL_SECONDS = 3600
MIN_BACKGROUND_POLL_SECONDS = 1
MIN_QUOTE_FETCH_WORKERS = 1
MIN_MAX_ORDER_HISTORY = 1
MAX_ACCOUNT_NAME_LENGTH = 64
MAX_SYMBOL_LENGTH = 20


def resolve_registry_path(path: str | Path | None = None) -> Path:
    raw_path = path if path is not None else os.getenv(TRADING_SERVER_REGISTRY_PATH_ENV)
    resolved = Path(raw_path).expanduser() if raw_path is not None else DEFAULT_REGISTRY_PATH
    if not resolved.is_absolute():
        resolved = REPO / resolved
    return resolved


def _clamp_int(value: int, *, minimum: int, maximum: int | None = None) -> int:
    clamped = max(int(value), minimum)
    if maximum is not None:
        clamped = min(clamped, maximum)
    return clamped


def resolve_env_int(name: str, default: int, *, minimum: int, maximum: int | None = None) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        parsed = int(str(raw_value).strip())
    except (TypeError, ValueError):
        return default
    return _clamp_int(parsed, minimum=minimum, maximum=maximum)


def resolve_explicit_or_env_int(
    explicit: int | None,
    *,
    env_name: str,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if explicit is None:
        return resolve_env_int(env_name, default, minimum=minimum, maximum=maximum)
    return _clamp_int(int(explicit), minimum=minimum, maximum=maximum)


@dataclass(frozen=True)
class TradingServerSettings:
    registry_path: Path
    quote_stale_seconds: int
    writer_ttl_seconds: int
    background_poll_seconds: int
    quote_fetch_workers: int
    max_order_history: int

    @classmethod
    def from_env(
        cls,
        *,
        registry_path: str | Path | None = None,
        quote_stale_seconds: int | None = None,
        writer_ttl_seconds: int | None = None,
        background_poll_seconds: int | None = None,
        quote_fetch_workers: int | None = None,
        max_order_history: int | None = None,
    ) -> "TradingServerSettings":
        return cls(
            registry_path=resolve_registry_path(registry_path),
            quote_stale_seconds=resolve_explicit_or_env_int(
                quote_stale_seconds,
                env_name=TRADING_SERVER_QUOTE_STALE_SECONDS_ENV,
                default=DEFAULT_QUOTE_STALE_SECONDS,
                minimum=MIN_QUOTE_STALE_SECONDS,
            ),
            writer_ttl_seconds=resolve_explicit_or_env_int(
                writer_ttl_seconds,
                env_name=TRADING_SERVER_WRITER_TTL_SECONDS_ENV,
                default=DEFAULT_WRITER_TTL_SECONDS,
                minimum=MIN_WRITER_TTL_SECONDS,
                maximum=MAX_WRITER_TTL_SECONDS,
            ),
            background_poll_seconds=resolve_explicit_or_env_int(
                background_poll_seconds,
                env_name=TRADING_SERVER_BACKGROUND_POLL_SECONDS_ENV,
                default=DEFAULT_BACKGROUND_POLL_SECONDS,
                minimum=MIN_BACKGROUND_POLL_SECONDS,
            ),
            quote_fetch_workers=resolve_explicit_or_env_int(
                quote_fetch_workers,
                env_name=TRADING_SERVER_QUOTE_FETCH_WORKERS_ENV,
                default=DEFAULT_QUOTE_FETCH_WORKERS,
                minimum=MIN_QUOTE_FETCH_WORKERS,
            ),
            max_order_history=resolve_explicit_or_env_int(
                max_order_history,
                env_name=TRADING_SERVER_MAX_ORDER_HISTORY_ENV,
                default=DEFAULT_MAX_ORDER_HISTORY,
                minimum=MIN_MAX_ORDER_HISTORY,
            ),
        )
