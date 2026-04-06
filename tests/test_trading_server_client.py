from __future__ import annotations

import math

import pytest
import requests

from src.trading_server.client import (
    InMemoryTradingServerClient,
    TradingServerClient,
    describe_trading_server_base_url,
)
from src.trading_server.settings import (
    DEFAULT_TRADING_SERVER_BASE_URL,
    MAX_WRITER_TTL_SECONDS,
    TRADING_SERVER_BASE_URL_ENV,
    TradingServerSettings,
    resolve_trading_server_base_url,
)


def test_claim_writer_uses_shared_writer_ttl_default(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_SERVER_WRITER_TTL_SECONDS", "333")
    client = TradingServerClient(account="paper_test", bot_id="paper_test_v1", session_id="session-a")
    captured: dict[str, object] = {}

    def fake_post(path: str, payload: dict[str, object]) -> dict[str, object]:
        captured["path"] = path
        captured["payload"] = payload
        return {"ok": True}

    client._post = fake_post  # type: ignore[method-assign]

    result = client.claim_writer()

    assert result == {"ok": True}
    assert captured["path"] == "/api/v1/writer/claim"
    assert captured["payload"]["ttl_seconds"] == TradingServerSettings.from_env().writer_ttl_seconds  # type: ignore[index]


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (
            "https://server.internal:8050",
            {"transport": "https", "scope": "remote", "security": "https", "host": "server.internal"},
        ),
        (
            "http://127.0.0.1:8050",
            {"transport": "http", "scope": "loopback", "security": "loopback_http", "host": "127.0.0.1"},
        ),
        (
            "http://server.internal:8050",
            {"transport": "http", "scope": "remote", "security": "insecure_remote_http", "host": "server.internal"},
        ),
        (
            "server.internal:8050",
            {"transport": "unsupported", "scope": "invalid", "security": "invalid", "host": None},
        ),
    ],
)
def test_describe_trading_server_base_url_classifies_expected_shapes(
    base_url: str,
    expected: dict[str, str | None],
) -> None:
    assert describe_trading_server_base_url(base_url) == expected


def test_resolve_trading_server_base_url_uses_shared_default_env_and_cli(monkeypatch) -> None:
    monkeypatch.delenv(TRADING_SERVER_BASE_URL_ENV, raising=False)
    assert resolve_trading_server_base_url() == DEFAULT_TRADING_SERVER_BASE_URL

    monkeypatch.setenv(TRADING_SERVER_BASE_URL_ENV, " http://server.internal:8050/ ")
    assert resolve_trading_server_base_url() == "http://server.internal:8050"

    assert resolve_trading_server_base_url(" https://server.internal:9443/ ") == "https://server.internal:9443"


def test_heartbeat_writer_explicit_ttl_overrides_shared_default(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_SERVER_WRITER_TTL_SECONDS", "333")
    client = TradingServerClient(account="paper_test", bot_id="paper_test_v1", session_id="session-a")
    captured: dict[str, object] = {}

    def fake_post(path: str, payload: dict[str, object]) -> dict[str, object]:
        captured["path"] = path
        captured["payload"] = payload
        return {"ok": True}

    client._post = fake_post  # type: ignore[method-assign]

    result = client.heartbeat_writer(ttl_seconds=45)

    assert result == {"ok": True}
    assert captured["path"] == "/api/v1/writer/heartbeat"
    assert captured["payload"]["ttl_seconds"] == 45  # type: ignore[index]


def test_claim_writer_clamps_shared_writer_ttl_default_to_contract_max(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_SERVER_WRITER_TTL_SECONDS", "999999")
    client = TradingServerClient(account="paper_test", bot_id="paper_test_v1", session_id="session-a")
    captured: dict[str, object] = {}

    def fake_post(path: str, payload: dict[str, object]) -> dict[str, object]:
        captured["path"] = path
        captured["payload"] = payload
        return {"ok": True}

    client._post = fake_post  # type: ignore[method-assign]

    result = client.claim_writer()

    assert result == {"ok": True}
    assert captured["path"] == "/api/v1/writer/claim"
    assert captured["payload"]["ttl_seconds"] == TradingServerSettings.from_env().writer_ttl_seconds  # type: ignore[index]
    assert captured["payload"]["ttl_seconds"] == MAX_WRITER_TTL_SECONDS  # type: ignore[index]


def test_inmemory_client_claim_writer_uses_shared_writer_ttl_default(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_SERVER_WRITER_TTL_SECONDS", "333")
    captured: dict[str, object] = {}

    class _Engine:
        def claim_writer(self, request):
            captured["request"] = request
            return {"ok": True}

    client = InMemoryTradingServerClient(
        engine=_Engine(),
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
    )

    result = client.claim_writer()

    assert result == {"ok": True}
    request = captured["request"]
    assert request.ttl_seconds == TradingServerSettings.from_env().writer_ttl_seconds  # type: ignore[attr-defined]
    assert request.session_id == "session-a"  # type: ignore[attr-defined]


def test_inmemory_client_heartbeat_and_get_orders_delegate_to_engine() -> None:
    captured: dict[str, object] = {}

    class _Engine:
        def heartbeat_writer(self, request):
            captured["heartbeat_request"] = request
            return {"heartbeat": True}

        def get_orders(self, account: str, *, include_history: bool = False):
            captured["orders_args"] = (account, include_history)
            return {"account": account, "open_orders": [], "order_history": []}

    client = InMemoryTradingServerClient(
        engine=_Engine(),
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
    )

    heartbeat = client.heartbeat_writer(ttl_seconds=45)
    orders = client.get_orders(include_history=True)

    assert heartbeat == {"heartbeat": True}
    request = captured["heartbeat_request"]
    assert request.ttl_seconds == 45  # type: ignore[attr-defined]
    assert request.session_id == "session-a"  # type: ignore[attr-defined]
    assert orders == {"account": "paper_test", "open_orders": [], "order_history": []}
    assert captured["orders_args"] == ("paper_test", True)


def test_live_trading_server_client_rejects_insecure_remote_http_url() -> None:
    with pytest.raises(ValueError, match="requires an https URL unless the server targets loopback"):
        TradingServerClient(
            base_url="http://server.internal:8050",
            account="live_account",
            bot_id="live_bot",
            execution_mode="live",
        )


def test_live_trading_server_client_allows_loopback_http_url() -> None:
    client = TradingServerClient(
        base_url="http://127.0.0.1:8050",
        account="live_account",
        bot_id="live_bot",
        execution_mode="live",
    )

    assert client.base_url == "http://127.0.0.1:8050"


def test_trading_server_client_rejects_invalid_execution_mode() -> None:
    with pytest.raises(ValueError, match="unsupported trading server execution mode"):
        TradingServerClient(
            base_url="http://127.0.0.1:8050",
            account="paper_test",
            bot_id="paper_test_v1",
            execution_mode="sandbox",
        )


def test_get_account_http_error_includes_status_and_body() -> None:
    class _FakeSession:
        def get(self, url, params, timeout):
            return _ErrorResponse()

        def post(self, url, json, timeout):  # pragma: no cover - defensive
            raise AssertionError("unexpected POST")

    client = TradingServerClient(
        base_url="http://127.0.0.1:8050",
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
        session=_FakeSession(),
    )

    class _ErrorResponse:
        status_code = 503
        text = '{"error":"maintenance"}'

        def raise_for_status(self) -> None:
            raise requests.HTTPError("503 Server Error", response=self)

        def json(self) -> dict[str, object]:
            return {"error": "maintenance"}
    with pytest.raises(RuntimeError) as exc_info:
        client.get_account()

    message = str(exc_info.value)
    assert "trading server GET /api/v1/account/paper_test failed" in message
    assert "url=http://127.0.0.1:8050/api/v1/account/paper_test" in message
    assert "status=503" in message
    assert 'body={"error":"maintenance"}' in message


def test_claim_writer_invalid_json_includes_body_excerpt() -> None:
    class _FakeSession:
        def get(self, url, params, timeout):  # pragma: no cover - defensive
            raise AssertionError("unexpected GET")

        def post(self, url, json, timeout):
            return _InvalidJsonResponse()

    client = TradingServerClient(
        base_url="http://127.0.0.1:8050",
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
        session=_FakeSession(),
    )

    class _InvalidJsonResponse:
        status_code = 200
        text = "<html>oops</html>"

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            raise ValueError("not json")
    with pytest.raises(RuntimeError) as exc_info:
        client.claim_writer()

    message = str(exc_info.value)
    assert "trading server POST /api/v1/writer/claim returned invalid JSON" in message
    assert "url=http://127.0.0.1:8050/api/v1/writer/claim" in message
    assert "status=200" in message
    assert "body=<html>oops</html>" in message


def test_get_account_rejects_non_object_json_payload() -> None:
    class _FakeSession:
        def get(self, url, params, timeout):
            return _ArrayJsonResponse()

        def post(self, url, json, timeout):  # pragma: no cover - defensive
            raise AssertionError("unexpected POST")

    client = TradingServerClient(
        base_url="http://127.0.0.1:8050",
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
        session=_FakeSession(),
    )

    class _ArrayJsonResponse:
        status_code = 200
        text = '["oops"]'

        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[str]:
            return ["oops"]

    with pytest.raises(RuntimeError) as exc_info:
        client.get_account()

    message = str(exc_info.value)
    assert "trading server GET /api/v1/account/paper_test returned unexpected JSON payload" in message
    assert "url=http://127.0.0.1:8050/api/v1/account/paper_test" in message
    assert "status=200" in message
    assert "type=list" in message
    assert 'body=["oops"]' in message


def test_trading_server_client_reuses_requests_session() -> None:
    calls: list[tuple[str, str]] = []

    class _FakeSession:
        def get(self, url, params, timeout):
            calls.append(("GET", url))
            return _JsonResponse({"account": "paper_test"})

        def post(self, url, json, timeout):
            calls.append(("POST", url))
            return _JsonResponse({"ok": True})

    class _JsonResponse:
        status_code = 200
        text = "{}"

        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    session = _FakeSession()
    client = TradingServerClient(
        base_url="http://127.0.0.1:8050",
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
        session=session,  # type: ignore[arg-type]
    )

    assert client.get_account() == {"account": "paper_test"}
    assert client.claim_writer() == {"ok": True}
    assert client._session is session  # type: ignore[attr-defined]
    assert calls == [
        ("GET", "http://127.0.0.1:8050/api/v1/account/paper_test"),
        ("POST", "http://127.0.0.1:8050/api/v1/writer/claim"),
    ]


def test_trading_server_client_uses_bearer_auth_header_from_env(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_SERVER_AUTH_TOKEN", "client-secret")
    captured: dict[str, object] = {}

    class _FakeSession:
        def get(self, url, params, timeout, headers):
            captured["headers"] = headers
            return _JsonResponse({"account": "paper_test"})

        def post(self, url, json, timeout, headers):  # pragma: no cover - defensive
            raise AssertionError("unexpected POST")

    class _JsonResponse:
        status_code = 200
        text = "{}"

        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    client = TradingServerClient(
        base_url="http://127.0.0.1:8050",
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
        session=_FakeSession(),  # type: ignore[arg-type]
    )

    assert client.get_account() == {"account": "paper_test"}
    assert captured["headers"] == {"Authorization": "Bearer client-secret"}


def test_trading_server_client_context_manager_closes_owned_session(monkeypatch) -> None:
    events: list[str] = []

    class _FakeSession:
        def get(self, url, params, timeout):
            events.append("get")
            return _JsonResponse({"account": "paper_test"})

        def post(self, url, json, timeout):  # pragma: no cover - defensive
            raise AssertionError("unexpected POST")

        def close(self) -> None:
            events.append("close")

    class _JsonResponse:
        status_code = 200
        text = "{}"

        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    monkeypatch.setattr("src.trading_server.client.requests.Session", _FakeSession)

    with TradingServerClient(
        base_url="http://127.0.0.1:8050",
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
    ) as client:
        assert client.get_account() == {"account": "paper_test"}

    assert events == ["get", "close"]


def test_trading_server_client_context_manager_does_not_close_injected_session() -> None:
    events: list[str] = []

    class _FakeSession:
        def get(self, url, params, timeout):
            events.append("get")
            return _JsonResponse({"account": "paper_test"})

        def post(self, url, json, timeout):  # pragma: no cover - defensive
            raise AssertionError("unexpected POST")

        def close(self) -> None:
            events.append("close")

    class _JsonResponse:
        status_code = 200
        text = "{}"

        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    session = _FakeSession()
    with TradingServerClient(
        base_url="http://127.0.0.1:8050",
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
        session=session,  # type: ignore[arg-type]
    ) as client:
        assert client.get_account() == {"account": "paper_test"}

    assert events == ["get"]


def test_trading_server_client_refresh_prices_normalizes_symbol_payload() -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    class _FakeSession:
        def get(self, url, params, timeout):  # pragma: no cover - defensive
            raise AssertionError("unexpected GET")

        def post(self, url, json, timeout):
            calls.append(("POST", url, json))
            return _JsonResponse({"accounts": []})

    class _JsonResponse:
        status_code = 200
        text = "{}"

        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    client = TradingServerClient(
        base_url="http://127.0.0.1:8050",
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
        session=_FakeSession(),  # type: ignore[arg-type]
    )

    assert client.refresh_prices(symbols=[" aapl ", "", "MSFT", " "]) == {"accounts": []}
    assert calls == [
        (
            "POST",
            "http://127.0.0.1:8050/api/v1/prices/refresh",
            {"account": "paper_test", "symbols": ["AAPL", "MSFT"]},
        )
    ]


def test_trading_server_client_get_orders_sets_include_history_query_param() -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    class _FakeSession:
        def get(self, url, params, timeout):
            calls.append(("GET", url, params))
            return _JsonResponse({"account": "paper_test", "open_orders": []})

        def post(self, url, json, timeout):  # pragma: no cover - defensive
            raise AssertionError("unexpected POST")

    class _JsonResponse:
        status_code = 200
        text = "{}"

        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    client = TradingServerClient(
        base_url="http://127.0.0.1:8050",
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
        session=_FakeSession(),  # type: ignore[arg-type]
    )

    assert client.get_orders(include_history=True) == {"account": "paper_test", "open_orders": []}
    assert calls == [
        (
            "GET",
            "http://127.0.0.1:8050/api/v1/orders/paper_test",
            {"include_history": "true"},
        )
    ]


def test_trading_server_client_submit_limit_order_posts_expected_payload() -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    class _FakeSession:
        def get(self, url, params, timeout):  # pragma: no cover - defensive
            raise AssertionError("unexpected GET")

        def post(self, url, json, timeout):
            calls.append(("POST", url, json))
            return _JsonResponse({"order": {"id": "order-1"}, "quote": None, "filled": False})

    class _JsonResponse:
        status_code = 200
        text = "{}"

        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    client = TradingServerClient(
        base_url="http://127.0.0.1:8050",
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
        execution_mode="paper",
        session=_FakeSession(),  # type: ignore[arg-type]
    )

    nested_metadata = {
        "strategy": "daily",
        "context": {"window": 5, "tags": ["gap", "momentum"]},
    }

    result = client.submit_limit_order(
        symbol="AAPL",
        side="buy",
        qty=10,
        limit_price=123.45,
        allow_loss_exit=True,
        force_exit_reason="risk_off",
        live_ack="ack-1",
        metadata=nested_metadata,
    )

    assert result == {"order": {"id": "order-1"}, "quote": None, "filled": False}
    assert calls == [
        (
            "POST",
            "http://127.0.0.1:8050/api/v1/orders",
            {
                "account": "paper_test",
                "bot_id": "paper_test_v1",
                "session_id": "session-a",
                "symbol": "AAPL",
                "side": "buy",
                "qty": 10.0,
                "limit_price": 123.45,
                "execution_mode": "paper",
                "allow_loss_exit": True,
                "force_exit_reason": "risk_off",
                "live_ack": "ack-1",
                "metadata": nested_metadata,
            },
        )
    ]


def test_trading_server_client_submit_limit_order_normalizes_side_and_mode() -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    class _FakeSession:
        def get(self, url, params, timeout):  # pragma: no cover - defensive
            raise AssertionError("unexpected GET")

        def post(self, url, json, timeout):
            calls.append(("POST", url, json))
            return _JsonResponse({"order": {"id": "order-1"}, "quote": None, "filled": False})

    class _JsonResponse:
        status_code = 200
        text = "{}"

        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    client = TradingServerClient(
        base_url="http://127.0.0.1:8050",
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
        execution_mode="LIVE",
        session=_FakeSession(),  # type: ignore[arg-type]
    )

    client.submit_limit_order(
        symbol="AAPL",
        side="BUY",
        qty=1.0,
        limit_price=123.45,
    )

    assert calls[0][2]["side"] == "buy"
    assert calls[0][2]["execution_mode"] == "live"


def test_trading_server_client_submit_limit_order_rejects_invalid_side() -> None:
    client = TradingServerClient(
        base_url="http://127.0.0.1:8050",
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
    )

    with pytest.raises(ValueError, match="unsupported trading server order side"):
        client.submit_limit_order(
            symbol="AAPL",
            side="hold",
            qty=1.0,
            limit_price=123.45,
        )


def test_trading_server_client_submit_limit_order_rejects_non_finite_qty() -> None:
    client = TradingServerClient(
        base_url="http://127.0.0.1:8050",
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
    )

    with pytest.raises(ValueError, match="qty must be a positive finite number"):
        client.submit_limit_order(
            symbol="AAPL",
            side="buy",
            qty=math.inf,
            limit_price=123.45,
        )


def test_trading_server_client_submit_limit_order_rejects_non_finite_metadata() -> None:
    client = TradingServerClient(
        base_url="http://127.0.0.1:8050",
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
    )

    with pytest.raises(
        ValueError,
        match="trading server metadata must be JSON-serializable without NaN or Infinity values",
    ):
        client.submit_limit_order(
            symbol="AAPL",
            side="buy",
            qty=1.0,
            limit_price=123.45,
            metadata={"confidence": math.nan},
        )


def test_trading_server_client_submit_limit_order_rejects_oversized_metadata() -> None:
    client = TradingServerClient(
        base_url="http://127.0.0.1:8050",
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
    )

    with pytest.raises(ValueError, match="trading server metadata exceeds 4096 bytes when serialized"):
        client.submit_limit_order(
            symbol="AAPL",
            side="buy",
            qty=1.0,
            limit_price=123.45,
            metadata={"payload": "x" * 5000},
        )


def test_inmemory_client_submit_limit_order_rejects_metadata_with_too_many_entries() -> None:
    class _Engine:
        def submit_order(self, request):  # pragma: no cover - should not be called
            raise AssertionError("unexpected submit_order")

    client = InMemoryTradingServerClient(
        engine=_Engine(),
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
    )

    with pytest.raises(
        ValueError,
        match="trading server metadata may contain at most 32 entries",
    ):
        client.submit_limit_order(
            symbol="AAPL",
            side="buy",
            qty=1.0,
            limit_price=250.0,
            metadata={f"k{i}": i for i in range(33)},
        )


def test_inmemory_client_submit_limit_order_delegates_full_request() -> None:
    captured: dict[str, object] = {}
    nested_metadata = {
        "source": "test",
        "plan": {"tiers": [1, 2], "active": True},
    }

    class _Engine:
        def submit_order(self, request):
            captured["request"] = request
            return {"order": {"id": "order-1"}, "quote": None, "filled": False}

    client = InMemoryTradingServerClient(
        engine=_Engine(),
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
        execution_mode="live",
    )

    result = client.submit_limit_order(
        symbol="AAPL",
        side="sell",
        qty=5.0,
        limit_price=250.0,
        allow_loss_exit=True,
        force_exit_reason="exit_rule",
        live_ack="ack-2",
        metadata=nested_metadata,
    )

    request = captured["request"]
    assert result == {"order": {"id": "order-1"}, "quote": None, "filled": False}
    assert request.account == "paper_test"  # type: ignore[attr-defined]
    assert request.bot_id == "paper_test_v1"  # type: ignore[attr-defined]
    assert request.session_id == "session-a"  # type: ignore[attr-defined]
    assert request.symbol == "AAPL"  # type: ignore[attr-defined]
    assert request.side == "sell"  # type: ignore[attr-defined]
    assert request.qty == 5.0  # type: ignore[attr-defined]
    assert request.limit_price == 250.0  # type: ignore[attr-defined]
    assert request.execution_mode == "live"  # type: ignore[attr-defined]
    assert request.allow_loss_exit is True  # type: ignore[attr-defined]
    assert request.force_exit_reason == "exit_rule"  # type: ignore[attr-defined]
    assert request.live_ack == "ack-2"  # type: ignore[attr-defined]
    assert request.metadata == nested_metadata  # type: ignore[attr-defined]


def test_inmemory_client_submit_limit_order_normalizes_side_and_mode() -> None:
    captured: dict[str, object] = {}

    class _Engine:
        def submit_order(self, request):
            captured["request"] = request
            return {"order": {"id": "order-1"}, "quote": None, "filled": False}

    client = InMemoryTradingServerClient(
        engine=_Engine(),
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
        execution_mode="LIVE",
    )

    client.submit_limit_order(
        symbol="AAPL",
        side="SELL",
        qty=5.0,
        limit_price=250.0,
    )

    request = captured["request"]
    assert request.side == "sell"  # type: ignore[attr-defined]
    assert request.execution_mode == "live"  # type: ignore[attr-defined]


def test_inmemory_client_submit_limit_order_rejects_non_finite_metadata() -> None:
    class _Engine:
        def submit_order(self, request):  # pragma: no cover - should not be called
            raise AssertionError("unexpected submit_order")

    client = InMemoryTradingServerClient(
        engine=_Engine(),
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
    )

    with pytest.raises(
        ValueError,
        match="trading server metadata must be JSON-serializable without NaN or Infinity values",
    ):
        client.submit_limit_order(
            symbol="AAPL",
            side="buy",
            qty=1.0,
            limit_price=250.0,
            metadata={"confidence": math.nan},
        )


def test_inmemory_client_supports_context_manager_close_boundary() -> None:
    class _Engine:
        def get_account_snapshot(self, account: str):
            return {"account": account}

    with InMemoryTradingServerClient(
        engine=_Engine(),
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
    ) as client:
        assert client.get_account() == {"account": "paper_test"}

    assert client.close() is None
