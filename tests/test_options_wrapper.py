from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from options import alpaca_options_wrapper as options_wrapper


class DummyResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class DummySession:
    def __init__(self):
        self.calls = []
        self.response = DummyResponse({"option_contracts": []})

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(("GET", url, params, headers, timeout))
        return self.response

    def post(self, url, headers=None, timeout=None):
        self.calls.append(("POST", url, headers, timeout))
        return self.response


def test_create_trading_client_honors_paper_override(monkeypatch):
    trading_cls = MagicMock()
    fake_client = MagicMock()
    trading_cls.return_value = fake_client
    monkeypatch.setattr(options_wrapper, "TradingClient", trading_cls)

    client = options_wrapper.create_options_trading_client(paper_override=True)

    trading_cls.assert_called_once_with(
        options_wrapper.ALP_KEY_ID,
        options_wrapper.ALP_SECRET_KEY,
        paper=True,
    )
    assert client is fake_client


def test_get_option_contracts_builds_request(monkeypatch):
    session = DummySession()
    response_payload = {
        "option_contracts": [
            {"symbol": "AAPL240119C00100000", "tradable": True},
        ]
    }
    session.response = DummyResponse(response_payload)

    data = options_wrapper.get_option_contracts(
        ["AAPL"],
        limit=25,
        session=session,
    )

    assert data == response_payload
    assert len(session.calls) == 1
    method, url, params, headers, timeout = session.calls[0]
    assert method == "GET"
    assert "/v2/options/contracts" in url
    assert params["underlying_symbols"] == "AAPL"
    assert params["limit"] == 25
    assert "APCA-API-KEY-ID" in headers
    assert timeout == options_wrapper.DEFAULT_TIMEOUT_SECONDS


def test_submit_option_order_uses_trading_client(monkeypatch):
    fake_client = MagicMock()
    monkeypatch.setattr(
        options_wrapper,
        "create_options_trading_client",
        MagicMock(return_value=fake_client),
    )

    options_wrapper.submit_option_order(
        symbol="AAPL240119C00100000",
        qty=2,
        side="buy",
        order_type="market",
        time_in_force="day",
        paper_override=True,
    )

    assert fake_client.submit_order.call_count == 1
    kwargs = fake_client.submit_order.call_args.kwargs
    assert kwargs["order_data"]["symbol"] == "AAPL240119C00100000"
    assert kwargs["order_data"]["qty"] == 2
    assert kwargs["order_data"]["side"] == "buy"
    assert kwargs["order_data"]["type"] == "market"
    assert kwargs["order_data"]["time_in_force"] == "day"
    assert kwargs["order_data"]["asset_class"] == "option"


def test_submit_option_order_requires_limit_price_for_limit_orders(monkeypatch):
    fake_client = MagicMock()
    monkeypatch.setattr(
        options_wrapper,
        "create_options_trading_client",
        MagicMock(return_value=fake_client),
    )

    with pytest.raises(ValueError):
        options_wrapper.submit_option_order(
            symbol="AAPL240119C00100000",
            qty=1,
            side="buy",
            order_type="limit",
            time_in_force="day",
            paper_override=True,
            limit_price=None,
        )


def test_exercise_option_position_invokes_endpoint(monkeypatch):
    session = DummySession()
    options_wrapper.exercise_option_position(
        "AAPL240119C00100000",
        session=session,
    )

    assert len(session.calls) == 1
    method, url, headers, timeout = session.calls[0]
    assert method == "POST"
    assert "/v2/positions/AAPL240119C00100000/exercise" in url
    assert "APCA-API-KEY-ID" in headers
    assert timeout == options_wrapper.DEFAULT_TIMEOUT_SECONDS


def test_get_option_bars_builds_parameters():
    session = DummySession()
    start_ts = datetime(2025, 1, 2, 13, 0, tzinfo=timezone.utc)
    end_ts = datetime(2025, 1, 2, 14, 0, tzinfo=timezone.utc)
    session.response = DummyResponse({"bars": []})

    options_wrapper.get_option_bars(
        ["AAPL240119C00100000", "AAPL240119P00100000"],
        timeframe="5Min",
        start=start_ts,
        end=end_ts,
        limit=500,
        sort="desc",
        page_token="token123",
        session=session,
    )

    assert len(session.calls) == 1
    method, url, params, headers, timeout = session.calls[0]
    assert method == "GET"
    assert url.endswith("/v1beta1/options/bars")
    assert params["symbols"] == "AAPL240119C00100000,AAPL240119P00100000"
    assert params["timeframe"] == "5Min"
    assert params["start"] == start_ts.isoformat()
    assert params["end"] == end_ts.isoformat()
    assert params["limit"] == 500
    assert params["sort"] == "desc"
    assert params["page_token"] == "token123"
    assert "APCA-API-KEY-ID" in headers
    assert timeout == options_wrapper.DEFAULT_TIMEOUT_SECONDS


def test_get_option_chain_filters():
    session = DummySession()
    session.response = DummyResponse({"snapshots": []})

    options_wrapper.get_option_chain(
        "AAPL",
        feed="indicative",
        limit=50,
        updated_since="2025-01-01T00:00:00Z",
        option_type="call",
        strike_price_gte=100.0,
        strike_price_lte=120.0,
        expiration_date="2025-01-17",
        root_symbol="AAPL",
        session=session,
    )

    assert len(session.calls) == 1
    method, url, params, headers, timeout = session.calls[0]
    assert method == "GET"
    assert url.endswith("/v1beta1/options/snapshots/AAPL")
    assert params["feed"] == "indicative"
    assert params["limit"] == 50
    assert params["type"] == "call"
    assert params["strike_price_gte"] == 100.0
    assert params["strike_price_lte"] == 120.0
    assert params["expiration_date"] == "2025-01-17"
    assert params["root_symbol"] == "AAPL"
    assert "APCA-API-KEY-ID" in headers
    assert timeout == options_wrapper.DEFAULT_TIMEOUT_SECONDS


def test_get_option_snapshots_requires_symbols():
    session = DummySession()
    session.response = DummyResponse({"snapshots": []})

    data = options_wrapper.get_option_snapshots(
        ["AAPL240119C00100000"],
        feed="opra",
        updated_since=datetime(2025, 1, 1, tzinfo=timezone.utc),
        limit=25,
        session=session,
    )

    assert data == {"snapshots": []}
    assert len(session.calls) == 1
    method, url, params, headers, timeout = session.calls[0]
    assert method == "GET"
    assert url.endswith("/v1beta1/options/snapshots")
    assert params["symbols"] == "AAPL240119C00100000"
    assert params["limit"] == 25
    assert params["feed"] == "opra"
    assert "updated_since" in params
    assert "APCA-API-KEY-ID" in headers
    assert timeout == options_wrapper.DEFAULT_TIMEOUT_SECONDS


def test_get_option_trades_enforces_sort_and_pagination():
    session = DummySession()
    session.response = DummyResponse({"trades": []})

    options_wrapper.get_option_trades(
        ["AAPL240119C00100000"],
        start="2025-01-01T00:00:00Z",
        end="2025-01-02T00:00:00Z",
        limit=100,
        sort="asc",
        page_token="abc",
        session=session,
    )

    assert len(session.calls) == 1
    method, url, params, headers, timeout = session.calls[0]
    assert method == "GET"
    assert url.endswith("/v1beta1/options/trades")
    assert params["symbols"] == "AAPL240119C00100000"
    assert params["limit"] == 100
    assert params["sort"] == "asc"
    assert params["page_token"] == "abc"
    assert params["start"] == "2025-01-01T00:00:00Z"
    assert params["end"] == "2025-01-02T00:00:00Z"
    assert "APCA-API-KEY-ID" in headers
    assert timeout == options_wrapper.DEFAULT_TIMEOUT_SECONDS


def test_get_latest_option_trades_accepts_feed():
    session = DummySession()
    session.response = DummyResponse({"latest_trades": []})

    options_wrapper.get_latest_option_trades(
        ["AAPL240119C00100000", "AAPL240119P00100000"],
        feed="indicative",
        session=session,
    )

    assert len(session.calls) == 1
    method, url, params, headers, timeout = session.calls[0]
    assert method == "GET"
    assert url.endswith("/v1beta1/options/trades/latest")
    assert params["symbols"] == "AAPL240119C00100000,AAPL240119P00100000"
    assert params["feed"] == "indicative"
    assert "APCA-API-KEY-ID" in headers
    assert timeout == options_wrapper.DEFAULT_TIMEOUT_SECONDS


def test_get_option_bars_requires_positive_limit():
    with pytest.raises(ValueError):
        options_wrapper.get_option_bars(["AAPL240119C00100000"], timeframe="1Day", limit=0)


def test_get_latest_option_quotes():
    session = DummySession()
    session.response = DummyResponse({"quotes": {}})

    options_wrapper.get_latest_option_quotes(
        ["AAPL240119C00100000"],
        feed="indicative",
        session=session,
    )

    assert len(session.calls) == 1
    method, url, params, headers, timeout = session.calls[0]
    assert method == "GET"
    assert url.endswith("/v1beta1/options/quotes/latest")
    assert params["symbols"] == "AAPL240119C00100000"
    assert params["feed"] == "indicative"
    assert "APCA-API-KEY-ID" in headers
    assert timeout == options_wrapper.DEFAULT_TIMEOUT_SECONDS
