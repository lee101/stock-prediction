from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Union

import requests
from alpaca.trading.client import TradingClient

import env_real as alpaca_env


ALP_KEY_ID = alpaca_env.ALP_KEY_ID
ALP_SECRET_KEY = alpaca_env.ALP_SECRET_KEY
ALP_ENDPOINT = getattr(alpaca_env, "ALP_ENDPOINT", "https://paper-api.alpaca.markets")

PAPER_API_BASE = "https://paper-api.alpaca.markets"
LIVE_API_BASE = "https://api.alpaca.markets"
API_VERSION_PATH = "/v2"
DEFAULT_TIMEOUT_SECONDS = 10
DATA_API_BASE = "https://data.alpaca.markets"
DATA_OPTIONS_PATH = "/v1beta1/options"


def _normalize_endpoint(endpoint: Optional[str]) -> str:
    if not endpoint:
        return PAPER_API_BASE
    base = endpoint.strip()
    if base.endswith(API_VERSION_PATH):
        base = base[: -len(API_VERSION_PATH)]
    return base.rstrip("/")


def _determine_base_endpoint(paper_override: Optional[bool]) -> str:
    if paper_override is True:
        return PAPER_API_BASE
    if paper_override is False:
        return LIVE_API_BASE

    endpoint = _normalize_endpoint(ALP_ENDPOINT)
    paper_flag = getattr(alpaca_env, "PAPER", None)

    if isinstance(paper_flag, bool):
        return PAPER_API_BASE if paper_flag else endpoint or LIVE_API_BASE

    if "paper" in endpoint.lower():
        return PAPER_API_BASE if not endpoint.startswith("http") else endpoint

    if endpoint:
        return endpoint

    return PAPER_API_BASE


def _infer_paper_flag(paper_override: Optional[bool]) -> bool:
    if paper_override is not None:
        return bool(paper_override)

    paper_flag = getattr(alpaca_env, "PAPER", None)
    if isinstance(paper_flag, bool):
        return paper_flag

    endpoint = (_normalize_endpoint(ALP_ENDPOINT)).lower()
    if "paper" in endpoint:
        return True

    return True  # Default to paper for safety if nothing explicit provided.


def _options_base_url(paper_override: Optional[bool] = None) -> str:
    base = _determine_base_endpoint(paper_override)
    if not base.endswith(API_VERSION_PATH):
        return f"{base}{API_VERSION_PATH}/options"
    return f"{base}/options"


def _positions_base_url(paper_override: Optional[bool] = None) -> str:
    base = _determine_base_endpoint(paper_override)
    if not base.endswith(API_VERSION_PATH):
        return f"{base}{API_VERSION_PATH}/positions"
    return f"{base}/positions"


def _auth_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALP_KEY_ID,
        "APCA-API-SECRET-KEY": ALP_SECRET_KEY,
    }


def _coerce_symbol_list(symbols: Iterable[str], param_name: str) -> str:
    cleaned = [s.strip().upper() for s in symbols if s and s.strip()]
    if not cleaned:
        raise ValueError(f"{param_name} must not be empty")
    return ",".join(cleaned)


def _coerce_underlying_symbols(symbols: Iterable[str]) -> str:
    return _coerce_symbol_list(symbols, "underlying_symbols")


def _coerce_contract_symbols(symbols: Iterable[str]) -> str:
    return _coerce_symbol_list(symbols, "symbols")


def _format_datetime_param(value: Optional[Union[str, datetime]]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def create_options_trading_client(
    paper_override: Optional[bool] = None,
) -> TradingClient:
    paper_flag = _infer_paper_flag(paper_override)
    return TradingClient(ALP_KEY_ID, ALP_SECRET_KEY, paper=paper_flag)


def get_option_contracts(
    underlying_symbols: Iterable[str],
    *,
    limit: int = 100,
    expiration_date_lte: Optional[str] = None,
    session: Optional[requests.sessions.Session] = None,
    paper_override: Optional[bool] = None,
) -> Mapping[str, Any]:
    if limit <= 0:
        raise ValueError("limit must be positive")

    params: Dict[str, Any] = {
        "underlying_symbols": _coerce_underlying_symbols(underlying_symbols),
        "limit": limit,
    }
    if expiration_date_lte:
        params["expiration_date_lte"] = expiration_date_lte

    requester = session or requests
    url = f"{_options_base_url(paper_override)}/contracts"
    response = requester.get(
        url,
        params=params,
        headers=_auth_headers(),
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def submit_option_order(
    *,
    symbol: str,
    qty: int,
    side: str,
    order_type: str,
    time_in_force: str,
    limit_price: Optional[float] = None,
    paper_override: Optional[bool] = None,
    client: Optional[TradingClient] = None,
    extra_fields: Optional[Mapping[str, Any]] = None,
) -> Any:
    if order_type.lower() == "limit" and limit_price is None:
        raise ValueError("limit orders require a limit_price")

    trading_client = client or create_options_trading_client(paper_override=paper_override)

    order: Dict[str, Any] = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force,
        "asset_class": "option",
    }

    if limit_price is not None:
        order["limit_price"] = limit_price

    if extra_fields:
        order.update(extra_fields)

    return trading_client.submit_order(order_data=order)


def exercise_option_position(
    symbol: str,
    *,
    session: Optional[requests.sessions.Session] = None,
    paper_override: Optional[bool] = None,
) -> Mapping[str, Any]:
    requester = session or requests
    url = f"{_positions_base_url(paper_override)}/{symbol}/exercise"
    response = requester.post(
        url,
        headers=_auth_headers(),
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    return payload


def _data_api_url(path_suffix: str) -> str:
    return f"{DATA_API_BASE}{DATA_OPTIONS_PATH}{path_suffix}"


def _validate_sort(sort: Optional[str]) -> Optional[str]:
    if sort is None:
        return None
    normalized = sort.lower()
    if normalized not in {"asc", "desc"}:
        raise ValueError("sort must be 'asc' or 'desc'")
    return normalized


def _ensure_positive_limit(limit: int) -> None:
    if limit <= 0:
        raise ValueError("limit must be positive")


def get_option_bars(
    symbols: Iterable[str],
    *,
    timeframe: str,
    start: Optional[Union[str, datetime]] = None,
    end: Optional[Union[str, datetime]] = None,
    limit: int = 1000,
    page_token: Optional[str] = None,
    sort: str = "asc",
    session: Optional[requests.sessions.Session] = None,
) -> Mapping[str, Any]:
    if not timeframe or not str(timeframe).strip():
        raise ValueError("timeframe is required")

    _ensure_positive_limit(limit)
    normalized_sort = _validate_sort(sort) or "asc"

    params: Dict[str, Any] = {
        "symbols": _coerce_contract_symbols(symbols),
        "timeframe": timeframe,
        "limit": limit,
        "sort": normalized_sort,
    }
    start_param = _format_datetime_param(start)
    end_param = _format_datetime_param(end)
    if start_param:
        params["start"] = start_param
    if end_param:
        params["end"] = end_param
    if page_token:
        params["page_token"] = page_token

    requester = session or requests
    response = requester.get(
        _data_api_url("/bars"),
        params=params,
        headers=_auth_headers(),
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def get_option_chain(
    underlying_symbol: str,
    *,
    feed: Optional[str] = None,
    limit: int = 100,
    updated_since: Optional[Union[str, datetime]] = None,
    page_token: Optional[str] = None,
    option_type: Optional[str] = None,
    strike_price_gte: Optional[float] = None,
    strike_price_lte: Optional[float] = None,
    expiration_date: Optional[Union[str, datetime]] = None,
    expiration_date_gte: Optional[Union[str, datetime]] = None,
    expiration_date_lte: Optional[Union[str, datetime]] = None,
    root_symbol: Optional[str] = None,
    session: Optional[requests.sessions.Session] = None,
) -> Mapping[str, Any]:
    if not underlying_symbol or not underlying_symbol.strip():
        raise ValueError("underlying_symbol is required")

    _ensure_positive_limit(limit)

    params: Dict[str, Any] = {"limit": limit}

    if feed:
        params["feed"] = feed
    if updated_since:
        params["updated_since"] = _format_datetime_param(updated_since)
    if page_token:
        params["page_token"] = page_token
    if option_type:
        params["type"] = option_type
    if strike_price_gte is not None:
        params["strike_price_gte"] = strike_price_gte
    if strike_price_lte is not None:
        params["strike_price_lte"] = strike_price_lte
    if expiration_date:
        params["expiration_date"] = _format_datetime_param(expiration_date)
    if expiration_date_gte:
        params["expiration_date_gte"] = _format_datetime_param(expiration_date_gte)
    if expiration_date_lte:
        params["expiration_date_lte"] = _format_datetime_param(expiration_date_lte)
    if root_symbol:
        params["root_symbol"] = root_symbol

    requester = session or requests
    response = requester.get(
        _data_api_url(f"/snapshots/{underlying_symbol}"),
        params=params,
        headers=_auth_headers(),
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def get_option_snapshots(
    symbols: Iterable[str],
    *,
    feed: Optional[str] = None,
    updated_since: Optional[Union[str, datetime]] = None,
    limit: int = 100,
    page_token: Optional[str] = None,
    session: Optional[requests.sessions.Session] = None,
) -> Mapping[str, Any]:
    _ensure_positive_limit(limit)

    params: Dict[str, Any] = {
        "symbols": _coerce_contract_symbols(symbols),
        "limit": limit,
    }
    if feed:
        params["feed"] = feed
    if updated_since:
        params["updated_since"] = _format_datetime_param(updated_since)
    if page_token:
        params["page_token"] = page_token

    requester = session or requests
    response = requester.get(
        _data_api_url("/snapshots"),
        params=params,
        headers=_auth_headers(),
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def get_option_trades(
    symbols: Iterable[str],
    *,
    start: Optional[Union[str, datetime]] = None,
    end: Optional[Union[str, datetime]] = None,
    limit: int = 1000,
    page_token: Optional[str] = None,
    sort: str = "asc",
    session: Optional[requests.sessions.Session] = None,
) -> Mapping[str, Any]:
    _ensure_positive_limit(limit)
    normalized_sort = _validate_sort(sort) or "asc"

    params: Dict[str, Any] = {
        "symbols": _coerce_contract_symbols(symbols),
        "limit": limit,
        "sort": normalized_sort,
    }
    if start:
        params["start"] = _format_datetime_param(start)
    if end:
        params["end"] = _format_datetime_param(end)
    if page_token:
        params["page_token"] = page_token

    requester = session or requests
    response = requester.get(
        _data_api_url("/trades"),
        params=params,
        headers=_auth_headers(),
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def get_latest_option_trades(
    symbols: Iterable[str],
    *,
    feed: Optional[str] = None,
    session: Optional[requests.sessions.Session] = None,
) -> Mapping[str, Any]:
    params: Dict[str, Any] = {
        "symbols": _coerce_contract_symbols(symbols),
    }
    if feed:
        params["feed"] = feed

    requester = session or requests
    response = requester.get(
        _data_api_url("/trades/latest"),
        params=params,
        headers=_auth_headers(),
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def get_latest_option_quotes(
    symbols: Iterable[str],
    *,
    feed: Optional[str] = None,
    session: Optional[requests.sessions.Session] = None,
) -> Mapping[str, Any]:
    params: Dict[str, Any] = {
        "symbols": _coerce_contract_symbols(symbols),
    }
    if feed:
        params["feed"] = feed

    requester = session or requests
    response = requester.get(
        _data_api_url("/quotes/latest"),
        params=params,
        headers=_auth_headers(),
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()
