from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

import src.crypto_loop.crypto_alpaca_looper_api as looper_api


@dataclass
class _FakeResponse:
    status_code: int = 200
    text: str = "ok"

    def raise_for_status(self) -> None:
        return None


@dataclass
class _FakeSubmitOrder:
    symbol: str
    side: str
    limit_price: float
    qty: float


def test_crypto_looper_api_base_url_defaults_and_trims_env(monkeypatch) -> None:
    monkeypatch.delenv(looper_api._CRYPTO_LOOPER_API_BASE_URL_ENV_VAR, raising=False)
    assert looper_api._crypto_looper_api_base_url() == looper_api._DEFAULT_CRYPTO_LOOPER_API_BASE_URL

    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_BASE_URL_ENV_VAR, " http://example.test:7777/ ")
    assert looper_api._crypto_looper_api_base_url() == "http://example.test:7777"
    assert looper_api._crypto_looper_api_url("/api/v1/stock_orders") == "http://example.test:7777/api/v1/stock_orders"

    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_BASE_URL_ENV_VAR, "   ")
    assert looper_api._crypto_looper_api_base_url() == looper_api._DEFAULT_CRYPTO_LOOPER_API_BASE_URL

    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_BASE_URL_ENV_VAR, "localhost:5050/")
    assert looper_api._crypto_looper_api_base_url() == "http://localhost:5050"

    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_BASE_URL_ENV_VAR, "://broken")
    assert looper_api._crypto_looper_api_base_url() == looper_api._DEFAULT_CRYPTO_LOOPER_API_BASE_URL

    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_BASE_URL_ENV_VAR, "ftp://example.test")
    assert looper_api._crypto_looper_api_base_url() == looper_api._DEFAULT_CRYPTO_LOOPER_API_BASE_URL

    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_BASE_URL_ENV_VAR, "http://user:secret@example.test")
    assert looper_api._crypto_looper_api_base_url() == looper_api._DEFAULT_CRYPTO_LOOPER_API_BASE_URL


def test_crypto_looper_api_timeout_defaults_and_validates_env(monkeypatch) -> None:
    monkeypatch.delenv(looper_api._CRYPTO_LOOPER_API_TIMEOUT_SECONDS_ENV_VAR, raising=False)
    assert looper_api._crypto_looper_api_timeout_seconds() == looper_api._DEFAULT_CRYPTO_LOOPER_API_TIMEOUT_SECONDS

    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_TIMEOUT_SECONDS_ENV_VAR, " 3.5 ")
    assert looper_api._crypto_looper_api_timeout_seconds() == 3.5

    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_TIMEOUT_SECONDS_ENV_VAR, "0")
    assert looper_api._crypto_looper_api_timeout_seconds() == looper_api._DEFAULT_CRYPTO_LOOPER_API_TIMEOUT_SECONDS

    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_TIMEOUT_SECONDS_ENV_VAR, "nan")
    assert looper_api._crypto_looper_api_timeout_seconds() == looper_api._DEFAULT_CRYPTO_LOOPER_API_TIMEOUT_SECONDS

    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_TIMEOUT_SECONDS_ENV_VAR, "bogus")
    assert looper_api._crypto_looper_api_timeout_seconds() == looper_api._DEFAULT_CRYPTO_LOOPER_API_TIMEOUT_SECONDS


def test_resolve_crypto_looper_api_config_exposes_normalized_runtime_config(monkeypatch) -> None:
    monkeypatch.setattr(looper_api, "_CRYPTO_LOOPER_API_CONFIG_CACHE_KEY", None)
    monkeypatch.setattr(looper_api, "_CRYPTO_LOOPER_API_CONFIG_CACHE_VALUE", None)
    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_BASE_URL_ENV_VAR, "localhost:5050/")
    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_TIMEOUT_SECONDS_ENV_VAR, "2.5")

    config = looper_api.resolve_crypto_looper_api_config()

    assert config == looper_api.CryptoLooperApiConfig(
        base_url="http://localhost:5050",
        timeout_seconds=2.5,
    )


def test_resolve_crypto_looper_api_config_reuses_cached_config_for_unchanged_env(monkeypatch) -> None:
    monkeypatch.setattr(looper_api, "_CRYPTO_LOOPER_API_CONFIG_CACHE_KEY", None)
    monkeypatch.setattr(looper_api, "_CRYPTO_LOOPER_API_CONFIG_CACHE_VALUE", None)
    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_BASE_URL_ENV_VAR, "http://api.test:9999/")
    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_TIMEOUT_SECONDS_ENV_VAR, "2.5")

    first = looper_api.resolve_crypto_looper_api_config()

    monkeypatch.setattr(
        looper_api,
        "_crypto_looper_api_base_url",
        lambda: (_ for _ in ()).throw(AssertionError("cached config should be reused")),
    )
    monkeypatch.setattr(
        looper_api,
        "_crypto_looper_api_timeout_seconds",
        lambda: (_ for _ in ()).throw(AssertionError("cached config should be reused")),
    )

    second = looper_api.resolve_crypto_looper_api_config()

    assert first is second


def test_resolve_crypto_looper_api_config_invalidates_cache_when_env_changes(monkeypatch) -> None:
    monkeypatch.setattr(looper_api, "_CRYPTO_LOOPER_API_CONFIG_CACHE_KEY", None)
    monkeypatch.setattr(looper_api, "_CRYPTO_LOOPER_API_CONFIG_CACHE_VALUE", None)
    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_BASE_URL_ENV_VAR, "http://api.test:9999/")
    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_TIMEOUT_SECONDS_ENV_VAR, "2.5")

    first = looper_api.resolve_crypto_looper_api_config()

    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_TIMEOUT_SECONDS_ENV_VAR, "4.0")
    second = looper_api.resolve_crypto_looper_api_config()

    assert first == looper_api.CryptoLooperApiConfig(
        base_url="http://api.test:9999",
        timeout_seconds=2.5,
    )
    assert second == looper_api.CryptoLooperApiConfig(
        base_url="http://api.test:9999",
        timeout_seconds=4.0,
    )
    assert second is not first


def test_stock_orders_reuses_cached_config_across_requests(monkeypatch) -> None:
    monkeypatch.setattr(looper_api, "_CRYPTO_LOOPER_API_CONFIG_CACHE_KEY", None)
    monkeypatch.setattr(looper_api, "_CRYPTO_LOOPER_API_CONFIG_CACHE_VALUE", None)
    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_BASE_URL_ENV_VAR, "http://api.test:9999/")
    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_TIMEOUT_SECONDS_ENV_VAR, "2.5")

    base_url_calls = 0
    timeout_calls = 0
    captured: list[tuple[str, float]] = []

    def counting_base_url() -> str:
        nonlocal base_url_calls
        base_url_calls += 1
        return "http://api.test:9999"

    def counting_timeout() -> float:
        nonlocal timeout_calls
        timeout_calls += 1
        return 2.5

    def fake_get(url: str, *, timeout: float) -> _FakeResponse:
        captured.append((url, timeout))
        return _FakeResponse()

    monkeypatch.setattr(looper_api, "_crypto_looper_api_base_url", counting_base_url)
    monkeypatch.setattr(looper_api, "_crypto_looper_api_timeout_seconds", counting_timeout)
    monkeypatch.setattr(looper_api.requests, "get", fake_get)

    first = looper_api.stock_orders()
    second = looper_api.stock_orders()

    assert isinstance(first, _FakeResponse)
    assert isinstance(second, _FakeResponse)
    assert captured == [
        ("http://api.test:9999/api/v1/stock_orders", 2.5),
        ("http://api.test:9999/api/v1/stock_orders", 2.5),
    ]
    assert base_url_calls == 1
    assert timeout_calls == 1


def test_submit_order_accepts_explicit_submission_contract(monkeypatch) -> None:
    captured: list[tuple[object, object, object, object]] = []

    monkeypatch.setattr(
        looper_api,
        "stock_order",
        lambda symbol, side, price, qty: captured.append((symbol, side, price, qty)) or _FakeResponse(),
    )

    response = looper_api.submit_order(_FakeSubmitOrder("BTCUSD", "buy", 123.45, 0.01))

    assert isinstance(response, _FakeResponse)
    assert captured == [("BTCUSD", "buy", 123.45, 0.01)]


def test_submit_order_returns_none_for_invalid_submission_object(monkeypatch) -> None:
    error_messages: list[str] = []

    monkeypatch.setattr(looper_api.logger, "error", lambda message: error_messages.append(str(message)))
    monkeypatch.setattr(
        looper_api,
        "stock_order",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("stock_order should not be called")),
    )

    class _InvalidOrder:
        symbol = "BTCUSD"
        side = "buy"

    assert looper_api.submit_order(_InvalidOrder()) is None
    assert any("missing required attributes: limit_price, qty" in message for message in error_messages)


def test_stock_order_path_quotes_symbols_and_cancel_path_is_explicit() -> None:
    assert looper_api._stock_order_path() == looper_api._CRYPTO_LOOPER_API_STOCK_ORDER_PATH
    assert looper_api._stock_order_path("BTC/USD") == "/api/v1/stock_order/BTC%2FUSD"
    assert looper_api._CRYPTO_LOOPER_API_CANCEL_ALL_ORDERS_PATH == "/api/v1/stock_order/cancel_all"


def test_response_log_preview_applies_explicit_preview_limit() -> None:
    response = _FakeResponse(text="x" * (looper_api._CRYPTO_LOOPER_API_RESPONSE_LOG_PREVIEW_CHARS + 10))

    preview = looper_api._response_log_preview(response)

    assert len(preview) == looper_api._CRYPTO_LOOPER_API_RESPONSE_LOG_PREVIEW_CHARS
    assert looper_api._response_log_preview(_FakeResponse(text="")) == "N/A"
    assert looper_api._response_log_preview(None) == "N/A"


def test_stock_order_uses_configured_base_url_and_timeout(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, *, timeout: float, json: dict[str, Any]) -> _FakeResponse:
        captured["url"] = url
        captured["timeout"] = timeout
        captured["json"] = json
        return _FakeResponse()

    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_BASE_URL_ENV_VAR, "http://api.test:9999/")
    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_TIMEOUT_SECONDS_ENV_VAR, "2.25")
    monkeypatch.setattr(looper_api.requests, "post", fake_post)

    response = looper_api.stock_order("BTCUSD", "buy", 123.45, 0.01)

    assert isinstance(response, _FakeResponse)
    assert captured["url"] == "http://api.test:9999/api/v1/stock_order"
    assert captured["timeout"] == 2.25
    assert captured["json"] == {
        "symbol": "BTCUSD",
        "side": "buy",
        "price": "123.45",
        "qty": "0.01",
    }


def test_symbol_endpoints_quote_symbol_and_use_timeout(monkeypatch) -> None:
    captured: list[tuple[str, str, float]] = []

    def fake_get(url: str, *, timeout: float) -> _FakeResponse:
        captured.append(("get", url, timeout))
        return _FakeResponse()

    def fake_delete(url: str, *, timeout: float) -> _FakeResponse:
        captured.append(("delete", url, timeout))
        return _FakeResponse()

    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_BASE_URL_ENV_VAR, "http://api.test:9999")
    monkeypatch.setenv(looper_api._CRYPTO_LOOPER_API_TIMEOUT_SECONDS_ENV_VAR, "4.0")
    monkeypatch.setattr(looper_api.requests, "get", fake_get)
    monkeypatch.setattr(looper_api.requests, "delete", fake_delete)

    symbol = "BTC/USD"
    looper_api.get_stock_order(symbol)
    looper_api.delete_stock_order(symbol)
    looper_api.stock_orders()
    looper_api.delete_stock_orders()

    assert captured == [
        ("get", "http://api.test:9999/api/v1/stock_order/BTC%2FUSD", 4.0),
        ("delete", "http://api.test:9999/api/v1/stock_order/BTC%2FUSD", 4.0),
        ("get", "http://api.test:9999/api/v1/stock_orders", 4.0),
        ("delete", "http://api.test:9999/api/v1/stock_order/cancel_all", 4.0),
    ]


def test_stock_order_returns_none_on_request_exception(monkeypatch) -> None:
    def fake_post(url: str, *, timeout: float, json: dict[str, Any]) -> _FakeResponse:
        raise requests.exceptions.Timeout("timed out")

    monkeypatch.setattr(looper_api.requests, "post", fake_post)

    assert looper_api.stock_order("BTCUSD", "buy", 123.45, 0.01) is None


def test_get_orders_returns_empty_list_on_json_decode_error(monkeypatch) -> None:
    class _BadJsonResponse(_FakeResponse):
        text = "{not-json"

        def json(self) -> dict[str, Any]:
            raise requests.exceptions.JSONDecodeError("bad json", self.text, 0)

    monkeypatch.setattr(looper_api, "stock_orders", lambda: _BadJsonResponse())

    assert looper_api.get_orders() == []


def test_get_orders_returns_empty_list_on_unexpected_payload_shapes(monkeypatch) -> None:
    error_messages: list[str] = []

    monkeypatch.setattr(looper_api.logger, "error", lambda message: error_messages.append(str(message)))

    class _TopLevelListResponse(_FakeResponse):
        def json(self) -> list[str]:
            return ["not", "a", "dict"]

    class _DataListResponse(_FakeResponse):
        def json(self) -> dict[str, Any]:
            return {"data": ["still", "wrong"]}

    monkeypatch.setattr(looper_api, "stock_orders", lambda: _TopLevelListResponse())
    assert looper_api.get_orders() == []
    assert any("payload type: list" in message for message in error_messages)

    monkeypatch.setattr(looper_api, "stock_orders", lambda: _DataListResponse())
    assert looper_api.get_orders() == []
    assert any("top-level keys=['data']" in message for message in error_messages)


def test_parse_orders_response_payload_parses_and_summarizes(monkeypatch) -> None:
    error_messages: list[str] = []
    info_messages: list[str] = []

    monkeypatch.setattr(looper_api.logger, "error", lambda message: error_messages.append(str(message)))
    monkeypatch.setattr(looper_api.logger, "info", lambda message: info_messages.append(str(message)))

    orders = looper_api._parse_orders_response_payload(
        {
            "data": {
                "first": {
                    "symbol": "BTCUSD",
                    "side": "buy",
                    "price": "101.25",
                    "qty": "0.5",
                    "created_at": "2026-04-08T12:30:45.123456",
                },
                "broken": "not-a-dict",
            }
        }
    )

    assert len(orders) == 1
    assert orders[0].symbol == "BTCUSD"
    assert orders[0].created_at == looper_api.load_iso_format("2026-04-08T12:30:45.123456")
    assert any("Skipping malformed order payload for key 'broken': str" in message for message in error_messages)
    assert any("parsed=1 malformed=1 total=2" in message for message in info_messages)


def test_parse_order_payload_returns_typed_fake_order() -> None:
    order = looper_api._parse_order_payload(
        {
            "symbol": "BTCUSD",
            "side": "buy",
            "price": "101.25",
            "qty": "0.5",
            "created_at": "2026-04-08T12:30:45.123456",
        }
    )

    assert isinstance(order, looper_api.FakeOrder)
    assert order == looper_api.FakeOrder(
        symbol="BTCUSD",
        side="buy",
        limit_price="101.25",
        qty="0.5",
        created_at=looper_api.load_iso_format("2026-04-08T12:30:45.123456"),
    )


def test_get_orders_parses_valid_orders_and_tolerates_bad_created_at(monkeypatch) -> None:
    class _OrdersResponse(_FakeResponse):
        def json(self) -> dict[str, Any]:
            return {
                "data": {
                    "first": {
                        "symbol": "BTCUSD",
                        "side": "buy",
                        "price": "101.25",
                        "qty": "0.5",
                        "created_at": "2026-04-08T12:30:45.123456",
                    },
                    "second": {
                        "symbol": "ETHUSD",
                        "side": "sell",
                        "price": "202.50",
                        "qty": "1.0",
                        "created_at": "not-a-timestamp",
                    },
                }
            }

    monkeypatch.setattr(looper_api, "stock_orders", lambda: _OrdersResponse())

    orders = looper_api.get_orders()

    assert len(orders) == 2
    assert orders[0].symbol == "BTCUSD"
    assert orders[0].side == "buy"
    assert orders[0].limit_price == "101.25"
    assert orders[0].qty == "0.5"
    assert orders[0].created_at == looper_api.load_iso_format("2026-04-08T12:30:45.123456")
    assert orders[1].symbol == "ETHUSD"
    assert orders[1].created_at is None


def test_get_orders_skips_malformed_individual_order_payloads(monkeypatch) -> None:
    error_messages: list[str] = []
    info_messages: list[str] = []

    monkeypatch.setattr(looper_api.logger, "error", lambda message: error_messages.append(str(message)))
    monkeypatch.setattr(looper_api.logger, "info", lambda message: info_messages.append(str(message)))

    class _OrdersResponse(_FakeResponse):
        def json(self) -> dict[str, Any]:
            return {
                "data": {
                    "first": {
                        "symbol": "BTCUSD",
                        "side": "buy",
                        "price": "101.25",
                        "qty": "0.5",
                        "created_at": "2026-04-08T12:30:45.123456",
                    },
                    "broken": "not-a-dict",
                }
            }

    monkeypatch.setattr(looper_api, "stock_orders", lambda: _OrdersResponse())

    orders = looper_api.get_orders()

    assert len(orders) == 1
    assert orders[0].symbol == "BTCUSD"
    assert any("Skipping malformed order payload for key 'broken': str" in message for message in error_messages)
    assert any("parsed=1 malformed=1 total=2" in message for message in info_messages)
