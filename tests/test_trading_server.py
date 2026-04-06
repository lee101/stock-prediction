from __future__ import annotations

import json
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.trading_server import server as trading_server_module
from src.trading_server.server import (
    TradingServerEngine,
    TradingServerSettings,
    create_app,
    ensure_background_refresh,
    stop_background_refresh,
)


def _write_registry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "accounts": {
                    "paper_test": {
                        "mode": "paper",
                        "allowed_bot_id": "paper_test_v1",
                        "starting_cash": 1000.0,
                        "sell_loss_cooldown_seconds": 1200,
                        "min_sell_markup_pct": 0.001,
                        "symbols": ["ETHUSD"],
                    },
                    "live_test": {
                        "mode": "live",
                        "allowed_bot_id": "live_test_v1",
                        "starting_cash": 0.0,
                        "sell_loss_cooldown_seconds": 1200,
                        "min_sell_markup_pct": 0.001,
                        "symbols": ["ETHUSD"],
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def _build_engine(
    tmp_path: Path,
    *,
    now: datetime | None = None,
    now_fn=None,
    quotes: list[dict] | None = None,
    live_executor=None,
    max_order_history: int | None = None,
    auth_token: str | None = None,
) -> TradingServerEngine:
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    quotes = list(quotes or [])
    current_now = now or datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)

    def quote_provider(_symbol: str):
        if quotes:
            return quotes.pop(0)
        return {
            "symbol": "ETHUSD",
            "bid_price": 2000.0,
            "ask_price": 2001.0,
            "last_price": 2000.5,
            "as_of": current_now.isoformat(),
        }

    return TradingServerEngine(
        registry_path=registry,
        state_dir=tmp_path / "state",
        quote_provider=quote_provider,
        live_executor=live_executor,
        now_fn=now_fn or (lambda: current_now),
        quote_stale_seconds=300,
        max_order_history=max_order_history,
        auth_token=auth_token,
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_engine_uses_registry_path_env_by_default(tmp_path, monkeypatch):
    registry = tmp_path / "custom-registry.json"
    _write_registry(registry)
    monkeypatch.setenv("TRADING_SERVER_REGISTRY_PATH", str(registry))

    engine = TradingServerEngine(state_dir=tmp_path / "state")

    assert engine.registry_path == registry


def test_missing_registry_error_reports_resolution_source(tmp_path, monkeypatch):
    missing_registry = tmp_path / "missing-trading-server-registry.json"
    monkeypatch.setenv("TRADING_SERVER_REGISTRY_PATH", str(missing_registry))

    with pytest.raises(RuntimeError) as exc_info:
        TradingServerEngine(state_dir=tmp_path / "state")

    message = str(exc_info.value)
    assert str(missing_registry) in message
    assert "TRADING_SERVER_REGISTRY_PATH=" in message


def test_engine_uses_runtime_env_quote_defaults(tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    monkeypatch.setenv("TRADING_SERVER_REGISTRY_PATH", str(registry))
    monkeypatch.setenv("TRADING_SERVER_QUOTE_STALE_SECONDS", "17")
    monkeypatch.setenv("TRADING_SERVER_QUOTE_FETCH_WORKERS", "9")
    monkeypatch.setenv("TRADING_SERVER_SHARED_QUOTE_CACHE_SIZE", "23")

    engine = TradingServerEngine(state_dir=tmp_path / "state")

    assert engine.quote_stale_seconds == 17
    assert engine.quote_fetch_workers == 9
    assert engine.shared_quote_cache_size == 23


def test_invalid_persisted_mode_is_rejected(tmp_path):
    engine = _build_engine(tmp_path)
    account_path = engine._account_path("paper_test")
    account_path.parent.mkdir(parents=True, exist_ok=True)
    account_path.write_text(
        json.dumps(
            {
                "account": "paper_test",
                "mode": "demo",
                "cash": 1000.0,
                "positions": {},
                "open_orders": [],
                "order_history": [],
                "price_cache": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unsupported mode=demo"):
        engine.get_account_snapshot("paper_test")


def test_account_state_cache_reuses_disk_load_until_file_changes(tmp_path, monkeypatch):
    engine = _build_engine(tmp_path)
    account = "paper_test"
    config = engine._config_for_account(account)
    account_path = engine._account_path(account)

    with engine._account_state_guard(account):
        state = engine._load_state_unlocked(account, config)
        engine._save_state_unlocked(state)
    engine._state_cache.clear()

    original_read_text = Path.read_text
    read_count = 0

    def counting_read_text(self: Path, *args, **kwargs):
        nonlocal read_count
        if self == account_path:
            read_count += 1
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)

    first = engine.get_account_snapshot(account)
    second = engine.get_account_snapshot(account)

    assert first["cash"] == second["cash"] == 1000.0
    assert read_count == 1

    updated = json.loads(original_read_text(account_path, encoding="utf-8"))
    updated["cash"] = 777.0
    account_path.write_text(json.dumps(updated), encoding="utf-8")

    third = engine.get_account_snapshot(account)

    assert third["cash"] == 777.0
    assert read_count == 2


def test_account_state_saves_use_compact_json(tmp_path):
    engine = _build_engine(tmp_path)
    account = "paper_test"
    config = engine._config_for_account(account)

    with engine._account_state_guard(account):
        state = engine._load_state_unlocked(account, config)
        engine._save_state_unlocked(state)

    payload = engine._account_path(account).read_text(encoding="utf-8")

    assert "\n" not in payload
    assert json.loads(payload)["account"] == account


def test_registry_rejects_unsafe_allowed_bot_id(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "accounts": {
                    "paper_test": {
                        "mode": "paper",
                        "allowed_bot_id": "paper test",
                        "symbols": ["ETHUSD"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unsupported bot_id"):
        TradingServerEngine(registry_path=registry, state_dir=tmp_path / "state")


def test_trading_server_settings_collect_runtime_defaults(tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    monkeypatch.setenv("TRADING_SERVER_REGISTRY_PATH", str(registry))
    monkeypatch.setenv("TRADING_SERVER_QUOTE_STALE_SECONDS", "17")
    monkeypatch.setenv("TRADING_SERVER_WRITER_TTL_SECONDS", "333")
    monkeypatch.setenv("TRADING_SERVER_BACKGROUND_POLL_SECONDS", "42")
    monkeypatch.setenv("TRADING_SERVER_QUOTE_FETCH_WORKERS", "9")
    monkeypatch.setenv("TRADING_SERVER_MAX_ORDER_HISTORY", "12")
    monkeypatch.setenv("TRADING_SERVER_SHARED_QUOTE_CACHE_SIZE", "18")

    settings = TradingServerSettings.from_env()

    assert settings.registry_path == registry
    assert settings.quote_stale_seconds == 17
    assert settings.writer_ttl_seconds == 333
    assert settings.background_poll_seconds == 42
    assert settings.quote_fetch_workers == 9
    assert settings.max_order_history == 12
    assert settings.shared_quote_cache_size == 18


def test_engine_explicit_config_overrides_env_defaults(tmp_path, monkeypatch):
    env_registry = tmp_path / "env-registry.json"
    explicit_registry = tmp_path / "explicit-registry.json"
    _write_registry(env_registry)
    _write_registry(explicit_registry)
    monkeypatch.setenv("TRADING_SERVER_REGISTRY_PATH", str(env_registry))
    monkeypatch.setenv("TRADING_SERVER_QUOTE_STALE_SECONDS", "17")
    monkeypatch.setenv("TRADING_SERVER_QUOTE_FETCH_WORKERS", "9")

    engine = TradingServerEngine(
        registry_path=explicit_registry,
        state_dir=tmp_path / "state",
        quote_stale_seconds=5,
        quote_fetch_workers=2,
        max_order_history=7,
        shared_quote_cache_size=11,
    )

    assert engine.registry_path == explicit_registry
    assert engine.quote_stale_seconds == 5
    assert engine.quote_fetch_workers == 2
    assert engine.max_order_history == 7
    assert engine.shared_quote_cache_size == 11
    assert engine.settings.registry_path == explicit_registry
    assert engine.settings.quote_stale_seconds == 5
    assert engine.settings.quote_fetch_workers == 2
    assert engine.settings.max_order_history == 7
    assert engine.settings.shared_quote_cache_size == 11


def test_invalid_runtime_env_values_fall_back_to_safe_defaults(tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    monkeypatch.setenv("TRADING_SERVER_REGISTRY_PATH", str(registry))
    monkeypatch.setenv("TRADING_SERVER_QUOTE_STALE_SECONDS", "not-a-number")
    monkeypatch.setenv("TRADING_SERVER_QUOTE_FETCH_WORKERS", "0")
    monkeypatch.setenv("TRADING_SERVER_WRITER_TTL_SECONDS", "oops")
    monkeypatch.setenv("TRADING_SERVER_BACKGROUND_POLL_SECONDS", "-5")
    monkeypatch.setenv("TRADING_SERVER_MAX_ORDER_HISTORY", "0")
    monkeypatch.setenv("TRADING_SERVER_SHARED_QUOTE_CACHE_SIZE", "0")

    engine = TradingServerEngine(state_dir=tmp_path / "state")
    request = trading_server_module.WriterLeaseRequest(
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
    )

    assert engine.quote_stale_seconds == trading_server_module.DEFAULT_QUOTE_STALE_SECONDS
    assert engine.quote_fetch_workers == 1
    assert engine.max_order_history == 1
    assert engine.shared_quote_cache_size == 1
    assert request.ttl_seconds == trading_server_module.DEFAULT_WRITER_TTL_SECONDS
    assert TradingServerSettings.from_env().background_poll_seconds == 1
    assert TradingServerSettings.from_env().max_order_history == 1
    assert TradingServerSettings.from_env().shared_quote_cache_size == 1


def test_account_state_guard_blocks_new_readers_while_writer_waits() -> None:
    guard = trading_server_module._AccountStateGuard()
    writer_acquired = threading.Event()
    allow_writer_release = threading.Event()
    second_reader_acquired = threading.Event()

    guard.acquire_read()

    def _writer() -> None:
        guard.acquire_write()
        writer_acquired.set()
        allow_writer_release.wait(timeout=1.0)
        guard.release_write()

    def _second_reader() -> None:
        guard.acquire_read()
        second_reader_acquired.set()
        guard.release_read()

    writer = threading.Thread(target=_writer)
    reader = threading.Thread(target=_second_reader)
    writer.start()

    deadline = time.time() + 1.0
    while guard._waiting_writers == 0 and time.time() < deadline:
        time.sleep(0.01)
    assert guard._waiting_writers == 1

    reader.start()
    assert not second_reader_acquired.wait(timeout=0.05)

    guard.release_read()
    assert writer_acquired.wait(timeout=1.0)
    assert not second_reader_acquired.is_set()

    allow_writer_release.set()
    writer.join(timeout=1.0)
    reader.join(timeout=1.0)

    assert not writer.is_alive()
    assert not reader.is_alive()
    assert second_reader_acquired.is_set()


def test_writer_lease_request_uses_runtime_env_default_ttl(monkeypatch):
    monkeypatch.setenv("TRADING_SERVER_WRITER_TTL_SECONDS", "333")

    request = trading_server_module.WriterLeaseRequest(
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
    )

    assert request.ttl_seconds == 333


def test_writer_lease_request_clamps_runtime_env_default_ttl_to_contract_max(monkeypatch):
    monkeypatch.setenv("TRADING_SERVER_WRITER_TTL_SECONDS", "999999")

    settings = TradingServerSettings.from_env()
    request = trading_server_module.WriterLeaseRequest(
        account="paper_test",
        bot_id="paper_test_v1",
        session_id="session-a",
    )

    assert settings.writer_ttl_seconds == trading_server_module.MAX_WRITER_TTL_SECONDS
    assert request.ttl_seconds == trading_server_module.MAX_WRITER_TTL_SECONDS


def test_create_app_uses_runtime_background_poll_env(tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    monkeypatch.setenv("TRADING_SERVER_REGISTRY_PATH", str(registry))
    monkeypatch.setenv("TRADING_SERVER_BACKGROUND_POLL_SECONDS", "42")
    calls: dict[str, object] = {}

    def fake_ensure(engine, *, poll_seconds=None):
        calls["engine"] = engine
        calls["poll_seconds"] = poll_seconds
        return threading.current_thread()

    def fake_stop(engine=None, timeout=1.0):
        calls["stopped_engine"] = engine
        calls["stopped_timeout"] = timeout

    monkeypatch.setattr(trading_server_module, "ensure_background_refresh", fake_ensure)
    monkeypatch.setattr(trading_server_module, "stop_background_refresh", fake_stop)

    app = trading_server_module.create_app()
    with TestClient(app):
        pass

    assert calls["poll_seconds"] == 42
    assert calls["engine"] is calls["stopped_engine"]


def test_create_app_uses_engine_settings_snapshot(tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    monkeypatch.setenv("TRADING_SERVER_REGISTRY_PATH", str(registry))
    monkeypatch.setenv("TRADING_SERVER_BACKGROUND_POLL_SECONDS", "42")
    engine = TradingServerEngine(state_dir=tmp_path / "state")
    monkeypatch.setenv("TRADING_SERVER_BACKGROUND_POLL_SECONDS", "99")
    calls: dict[str, object] = {}

    def fake_ensure(current_engine, *, poll_seconds=None):
        calls["engine"] = current_engine
        calls["poll_seconds"] = poll_seconds
        return threading.current_thread()

    def fake_stop(engine=None, timeout=1.0):
        calls["stopped_engine"] = engine

    monkeypatch.setattr(trading_server_module, "ensure_background_refresh", fake_ensure)
    monkeypatch.setattr(trading_server_module, "stop_background_refresh", fake_stop)

    app = trading_server_module.create_app(engine)
    with TestClient(app):
        pass

    assert calls["poll_seconds"] == 42
    assert calls["engine"] is engine
    assert calls["stopped_engine"] is engine


def test_runtime_config_endpoint_reports_effective_values_and_sources(tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    monkeypatch.setenv("TRADING_SERVER_REGISTRY_PATH", str(registry))
    monkeypatch.setenv("TRADING_SERVER_BACKGROUND_POLL_SECONDS", "oops")
    monkeypatch.setenv("TRADING_SERVER_QUOTE_FETCH_WORKERS", "0")

    engine = TradingServerEngine(
        state_dir=tmp_path / "state",
        quote_stale_seconds=5,
        shared_quote_cache_size=11,
    )

    response = TestClient(create_app(engine)).get("/api/v1/runtime-config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["registry_path"]["value"] == str(registry)
    assert payload["registry_path"]["source"] == "env"
    assert payload["quote_stale_seconds"]["value"] == 5
    assert payload["quote_stale_seconds"]["source"] == "explicit"
    assert payload["shared_quote_cache_size"]["value"] == 11
    assert payload["shared_quote_cache_size"]["source"] == "explicit"
    assert payload["quote_fetch_workers"]["value"] == 1
    assert payload["quote_fetch_workers"]["source"] == "env"
    assert "clamped to 1" in payload["quote_fetch_workers"]["detail"]
    assert payload["background_poll_seconds"]["value"] == trading_server_module.DEFAULT_BACKGROUND_POLL_SECONDS
    assert payload["background_poll_seconds"]["source"] == "env-invalid"
    assert "invalid" in payload["background_poll_seconds"]["detail"]
    assert payload["auth_token"]["value"] is False


def test_create_app_optionally_requires_bearer_auth(tmp_path):
    engine = _build_engine(tmp_path, auth_token="server-secret")
    app = create_app(engine)

    with TestClient(app) as client:
        unauthorized = client.get("/api/v1/accounts")
        wrong = client.get("/api/v1/accounts", headers={"Authorization": "Bearer wrong"})
        authorized = client.get("/api/v1/accounts", headers={"Authorization": "Bearer server-secret"})

    assert unauthorized.status_code == 401
    assert unauthorized.json()["detail"] == "invalid auth token"
    assert unauthorized.headers["www-authenticate"] == "Bearer"
    assert wrong.status_code == 401
    assert authorized.status_code == 200
    assert authorized.json()["accounts"]


def test_create_app_logs_sanitized_auth_failures(tmp_path, caplog):
    engine = _build_engine(tmp_path, auth_token="server-secret")
    app = create_app(engine)
    caplog.set_level(logging.WARNING, logger=trading_server_module.__name__)

    with TestClient(app) as client:
        client.get("/api/v1/accounts")
        client.get("/api/v1/accounts", headers={"Authorization": "Bearer wrong"})

    assert "Rejected unauthorized trading server request" in caplog.text
    assert "path=/api/v1/accounts" in caplog.text
    assert "auth=missing" in caplog.text
    assert "auth=mismatch" in caplog.text
    assert "server-secret" not in caplog.text
    assert "Bearer wrong" not in caplog.text


def test_writer_lease_blocks_second_session(tmp_path):
    engine = _build_engine(tmp_path)
    app = create_app(engine)
    with TestClient(app) as client:
        first = client.post(
            "/api/v1/writer/claim",
            json={
                "account": "paper_test",
                "bot_id": "paper_test_v1",
                "session_id": "session-a",
                "ttl_seconds": 120,
            },
        )
        assert first.status_code == 200

        second = client.post(
            "/api/v1/writer/claim",
            json={
                "account": "paper_test",
                "bot_id": "paper_test_v1",
                "session_id": "session-b",
                "ttl_seconds": 120,
            },
        )
        assert second.status_code == 409
        assert "writer lease already held" in second.json()["detail"]


def test_writer_claim_rejects_wrong_bot_id(tmp_path):
    engine = _build_engine(tmp_path)
    app = create_app(engine)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/writer/claim",
            json={
                "account": "paper_test",
                "bot_id": "wrong_bot",
                "session_id": "session-a",
                "ttl_seconds": 120,
            },
        )
    assert response.status_code == 403
    assert "not allowed" in response.json()["detail"]


def test_writer_claim_audit_log_records_success_and_conflict(tmp_path):
    engine = _build_engine(tmp_path)

    claim = engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "session-a", "ttl_seconds": 120})()
    )
    assert claim["session_id"] == "session-a"

    with pytest.raises(HTTPException):
        engine.claim_writer(
            type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "session-b", "ttl_seconds": 120})()
        )

    events = _read_jsonl(tmp_path / "state" / "trading_server" / "events" / "paper_test.audit.jsonl")
    assert [event["event_type"] for event in events] == ["writer_claimed", "writer_claim_rejected"]
    assert events[0]["session_id"] == "session-a"
    assert "writer lease already held" in events[1]["detail"]


def test_writer_claim_rejects_unsafe_account_name(tmp_path):
    engine = _build_engine(tmp_path)
    app = create_app(engine)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/writer/claim",
            json={
                "account": "../paper_test",
                "bot_id": "paper_test_v1",
                "session_id": "session-a",
                "ttl_seconds": 120,
            },
        )
    assert response.status_code == 400
    assert "unsupported account name" in response.json()["detail"]


def test_writer_claim_rejects_overlong_account_name(tmp_path):
    engine = _build_engine(tmp_path)
    app = create_app(engine)
    account_name = "a" * (trading_server_module.MAX_ACCOUNT_NAME_LENGTH + 1)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/writer/claim",
            json={
                "account": account_name,
                "bot_id": "paper_test_v1",
                "session_id": "session-a",
                "ttl_seconds": 120,
            },
        )
    assert response.status_code == 400
    assert "unsupported account name" in response.json()["detail"]


def test_writer_claim_rejects_unsafe_session_id_direct_engine_call(tmp_path):
    engine = _build_engine(tmp_path)

    with pytest.raises(HTTPException) as exc_info:
        engine.claim_writer(
            type(
                "Lease",
                (),
                {
                    "account": "paper_test",
                    "bot_id": "paper_test_v1",
                    "session_id": "bad/session",
                    "ttl_seconds": 120,
                },
            )()
        )

    assert exc_info.value.status_code == 400
    assert "unsupported session_id" in str(exc_info.value.detail)


def test_submit_order_rejects_unsafe_session_id_direct_engine_call(tmp_path):
    engine = _build_engine(tmp_path)
    engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )

    with pytest.raises(HTTPException) as exc_info:
        engine.submit_order(
            type(
                "Order",
                (),
                {
                    "account": "paper_test",
                    "bot_id": "paper_test_v1",
                    "session_id": "bad/session",
                    "symbol": "ETHUSD",
                    "side": "buy",
                    "qty": 1.0,
                    "limit_price": 100.0,
                    "execution_mode": "paper",
                    "allow_loss_exit": False,
                    "force_exit_reason": None,
                    "live_ack": None,
                    "metadata": {},
                },
            )()
        )

    assert exc_info.value.status_code == 400
    assert "unsupported session_id" in str(exc_info.value.detail)


def test_submit_order_rejects_oversized_metadata_direct_engine_call(tmp_path):
    engine = _build_engine(tmp_path)
    engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )
    oversized_metadata = {
        "payload": "x" * (trading_server_module.MAX_ORDER_METADATA_BYTES + 128)
    }
    encoded_size = len(
        json.dumps(
            oversized_metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    expected_error = (
        f"metadata exceeds {trading_server_module.MAX_ORDER_METADATA_BYTES} bytes "
        f"when serialized (got {encoded_size} bytes)"
    )

    with pytest.raises(HTTPException) as exc_info:
        engine.submit_order(
            type(
                "Order",
                (),
                {
                    "account": "paper_test",
                    "bot_id": "paper_test_v1",
                    "session_id": "owner",
                    "symbol": "ETHUSD",
                    "side": "buy",
                    "qty": 1.0,
                    "limit_price": 100.0,
                    "execution_mode": "paper",
                    "allow_loss_exit": False,
                    "force_exit_reason": None,
                    "live_ack": None,
                    "metadata": oversized_metadata,
                },
            )()
        )

    assert exc_info.value.status_code == 400
    assert str(exc_info.value.detail) == expected_error
    rejected = _read_jsonl(
        tmp_path / "state" / "trading_server" / "events" / "paper_test.rejections.jsonl"
    )
    assert "_metadata_error" in rejected[-1]["metadata"]
    assert rejected[-1]["metadata"]["_metadata_error"] == expected_error


def test_submit_order_api_rejects_oversized_metadata(tmp_path):
    engine = _build_engine(tmp_path)
    app = create_app(engine)
    oversized_metadata = {
        "payload": "x" * (trading_server_module.MAX_ORDER_METADATA_BYTES + 128)
    }

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/orders",
            json={
                "account": "paper_test",
                "bot_id": "paper_test_v1",
                "session_id": "owner",
                "symbol": "ETHUSD",
                "side": "buy",
                "qty": 1.0,
                "limit_price": 100.0,
                "execution_mode": "paper",
                "metadata": oversized_metadata,
            },
        )

    assert response.status_code == 422
    assert "metadata exceeds" in json.dumps(response.json())


def test_submit_order_rejects_metadata_with_too_many_entries_direct_engine_call(tmp_path):
    engine = _build_engine(tmp_path)
    engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )
    oversized_metadata = {
        f"k{i}": i for i in range(trading_server_module.MAX_ORDER_METADATA_ITEMS + 1)
    }

    with pytest.raises(HTTPException) as exc_info:
        engine.submit_order(
            type(
                "Order",
                (),
                {
                    "account": "paper_test",
                    "bot_id": "paper_test_v1",
                    "session_id": "owner",
                    "symbol": "ETHUSD",
                    "side": "buy",
                    "qty": 1.0,
                    "limit_price": 100.0,
                    "execution_mode": "paper",
                    "allow_loss_exit": False,
                    "force_exit_reason": None,
                    "live_ack": None,
                    "metadata": oversized_metadata,
                },
            )()
        )

    assert exc_info.value.status_code == 400
    assert (
        str(exc_info.value.detail)
        == "metadata may contain at most 32 entries (got 33)"
    )


def test_submit_order_rejects_non_finite_metadata_direct_engine_call(tmp_path):
    engine = _build_engine(tmp_path)
    engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )

    with pytest.raises(HTTPException) as exc_info:
        engine.submit_order(
            type(
                "Order",
                (),
                {
                    "account": "paper_test",
                    "bot_id": "paper_test_v1",
                    "session_id": "owner",
                    "symbol": "ETHUSD",
                    "side": "buy",
                    "qty": 1.0,
                    "limit_price": 100.0,
                    "execution_mode": "paper",
                    "allow_loss_exit": False,
                    "force_exit_reason": None,
                    "live_ack": None,
                    "metadata": {"confidence": math.nan},
                },
            )()
        )

    assert exc_info.value.status_code == 400
    assert str(exc_info.value.detail) == "metadata must be JSON-serializable without NaN or Infinity values"


def test_submit_order_rejects_non_finite_qty_and_sanitizes_rejection_log(tmp_path):
    engine = _build_engine(tmp_path)
    engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )

    with pytest.raises(HTTPException) as exc_info:
        engine.submit_order(
            type(
                "Order",
                (),
                {
                    "account": "paper_test",
                    "bot_id": "paper_test_v1",
                    "session_id": "owner",
                    "symbol": "ETHUSD",
                    "side": "buy",
                    "qty": math.inf,
                    "limit_price": 100.0,
                    "execution_mode": "paper",
                    "allow_loss_exit": False,
                    "force_exit_reason": None,
                    "live_ack": None,
                    "metadata": {},
                },
            )()
        )

    assert exc_info.value.status_code == 400
    assert str(exc_info.value.detail) == "qty must be a positive finite number"
    rejected = _read_jsonl(
        tmp_path / "state" / "trading_server" / "events" / "paper_test.rejections.jsonl"
    )
    assert rejected[-1]["qty"] is None
    assert rejected[-1]["limit_price"] == 100.0


def test_submit_order_rejects_without_active_writer_lease(tmp_path):
    engine = _build_engine(tmp_path)
    app = create_app(engine)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/orders",
            json={
                "account": "paper_test",
                "bot_id": "paper_test_v1",
                "session_id": "missing-lease",
                "symbol": "ETHUSD",
                "side": "buy",
                "qty": 0.1,
                "limit_price": 2000.0,
                "execution_mode": "paper",
            },
        )
    assert response.status_code == 409
    assert "Claim the writer lease" in response.json()["detail"]
    events = _read_jsonl(tmp_path / "state" / "trading_server" / "events" / "paper_test.audit.jsonl")
    rejected = [event for event in events if event["event_type"] == "order_rejected"]
    assert rejected
    assert "Claim the writer lease" in rejected[-1]["detail"]


def test_submit_order_rejects_unsafe_symbol(tmp_path):
    engine = _build_engine(tmp_path)
    claim = engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )
    response = TestClient(create_app(engine)).post(
        "/api/v1/orders",
        json={
            "account": "paper_test",
            "bot_id": "paper_test_v1",
            "session_id": claim["session_id"],
            "symbol": "../../ETHUSD",
            "side": "buy",
            "qty": 0.1,
            "limit_price": 2000.0,
            "execution_mode": "paper",
        },
    )
    assert response.status_code == 400
    assert "unsupported symbol" in response.json()["detail"]


def test_submit_order_rejects_overlong_symbol(tmp_path):
    engine = _build_engine(tmp_path)
    claim = engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )
    symbol = "A" * (trading_server_module.MAX_SYMBOL_LENGTH + 1)
    response = TestClient(create_app(engine)).post(
        "/api/v1/orders",
        json={
            "account": "paper_test",
            "bot_id": "paper_test_v1",
            "session_id": claim["session_id"],
            "symbol": symbol,
            "side": "buy",
            "qty": 0.1,
            "limit_price": 2000.0,
            "execution_mode": "paper",
        },
    )
    assert response.status_code == 400
    assert "unsupported symbol" in response.json()["detail"]


def test_expired_writer_lease_can_be_reclaimed(tmp_path):
    current_now = {"value": datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)}
    engine = _build_engine(tmp_path, now_fn=lambda: current_now["value"])

    first = engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "first", "ttl_seconds": 30})()
    )
    assert first["session_id"] == "first"

    current_now["value"] = current_now["value"] + timedelta(seconds=31)
    second = engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "second", "ttl_seconds": 30})()
    )
    assert second["session_id"] == "second"


def test_writer_heartbeat_persists_ttl_seconds_and_expiry(tmp_path):
    current_now = {"value": datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)}
    engine = _build_engine(tmp_path, now_fn=lambda: current_now["value"])

    claim = engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 30})()
    )
    initial_state = json.loads(engine._account_path("paper_test").read_text(encoding="utf-8"))

    assert initial_state["writer_claim"]["session_id"] == claim["session_id"]
    assert initial_state["writer_claim"]["ttl_seconds"] == 30
    assert initial_state["writer_claim"]["expires_at"] == (
        current_now["value"] + timedelta(seconds=30)
    ).isoformat()

    current_now["value"] = current_now["value"] + timedelta(seconds=5)
    heartbeat = engine.heartbeat_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 45})()
    )
    refreshed_state = json.loads(engine._account_path("paper_test").read_text(encoding="utf-8"))

    assert heartbeat["expires_at"] == (current_now["value"] + timedelta(seconds=45)).isoformat()
    assert refreshed_state["writer_claim"]["ttl_seconds"] == 45
    assert refreshed_state["writer_claim"]["expires_at"] == heartbeat["expires_at"]


def test_paper_buy_fills_and_updates_account(tmp_path):
    engine = _build_engine(
        tmp_path,
        quotes=[
            {
                "symbol": "ETHUSD",
                "bid_price": 1999.5,
                "ask_price": 2000.0,
                "last_price": 1999.8,
                "as_of": datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc).isoformat(),
            }
        ],
    )
    claim = engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )
    result = engine.submit_order(
        type(
            "Order",
            (),
            {
                "account": "paper_test",
                "bot_id": "paper_test_v1",
                "session_id": claim["session_id"],
                "symbol": "ETHUSD",
                "side": "buy",
                "qty": 0.4,
                "limit_price": 2000.0,
                "execution_mode": "paper",
                "allow_loss_exit": False,
                "force_exit_reason": None,
                "live_ack": None,
                "metadata": {},
            },
        )()
    )
    assert result["filled"] is True
    snapshot = engine.get_account_snapshot("paper_test")
    assert snapshot["positions"]["ETHUSD"]["qty"] == 0.4
    assert snapshot["positions"]["ETHUSD"]["avg_entry_price"] == 2000.0
    assert snapshot["cash"] == 200.0


def test_insufficient_paper_buying_power_rejected(tmp_path):
    engine = _build_engine(tmp_path)
    claim = engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )

    response = TestClient(create_app(engine)).post(
        "/api/v1/orders",
        json={
            "account": "paper_test",
            "bot_id": "paper_test_v1",
            "session_id": claim["session_id"],
            "symbol": "ETHUSD",
            "side": "buy",
            "qty": 1.0,
            "limit_price": 2000.0,
            "execution_mode": "paper",
        },
    )
    assert response.status_code == 400
    assert "insufficient paper buying power" in response.json()["detail"]


def test_sell_below_entry_inside_protection_window_is_rejected(tmp_path):
    now = datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)
    engine = _build_engine(
        tmp_path,
        now=now,
        quotes=[
            {
                "symbol": "ETHUSD",
                "bid_price": 2004.0,
                "ask_price": 2005.0,
                "last_price": 2004.5,
                "as_of": now.isoformat(),
            }
        ],
    )
    claim = engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )
    engine.submit_order(
        type(
            "Order",
            (),
            {
                "account": "paper_test",
                "bot_id": "paper_test_v1",
                "session_id": claim["session_id"],
                "symbol": "ETHUSD",
                "side": "buy",
                "qty": 0.25,
                "limit_price": 2005.0,
                "execution_mode": "paper",
                "allow_loss_exit": False,
                "force_exit_reason": None,
                "live_ack": None,
                "metadata": {},
            },
        )()
    )

    reject = TestClient(create_app(engine)).post(
        "/api/v1/orders",
        json={
            "account": "paper_test",
            "bot_id": "paper_test_v1",
            "session_id": claim["session_id"],
            "symbol": "ETHUSD",
            "side": "sell",
            "qty": 0.25,
            "limit_price": 2004.0,
            "execution_mode": "paper",
        },
    )
    assert reject.status_code == 400
    assert "below safety floor" in reject.json()["detail"]


def test_rejection_is_logged_to_jsonl(tmp_path):
    now = datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)
    engine = _build_engine(
        tmp_path,
        now=now,
        quotes=[
            {
                "symbol": "ETHUSD",
                "bid_price": 2004.0,
                "ask_price": 2005.0,
                "last_price": 2004.5,
                "as_of": now.isoformat(),
            }
        ],
    )
    claim = engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )
    engine.submit_order(
        type(
            "Order",
            (),
            {
                "account": "paper_test",
                "bot_id": "paper_test_v1",
                "session_id": claim["session_id"],
                "symbol": "ETHUSD",
                "side": "buy",
                "qty": 0.25,
                "limit_price": 2005.0,
                "execution_mode": "paper",
                "allow_loss_exit": False,
                "force_exit_reason": None,
                "live_ack": None,
                "metadata": {"source": "unit-test"},
            },
        )()
    )

    with pytest.raises(HTTPException):
        engine.submit_order(
            type(
                "Order",
                (),
                {
                    "account": "paper_test",
                    "bot_id": "paper_test_v1",
                    "session_id": claim["session_id"],
                    "symbol": "ETHUSD",
                    "side": "sell",
                    "qty": 0.25,
                    "limit_price": 2004.0,
                    "execution_mode": "paper",
                    "allow_loss_exit": False,
                    "force_exit_reason": None,
                    "live_ack": None,
                    "metadata": {"source": "unit-test"},
                },
            )()
        )

    rejection_path = tmp_path / "state" / "trading_server" / "events" / "paper_test.rejections.jsonl"
    lines = rejection_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["symbol"] == "ETHUSD"
    assert payload["reason"] == "order_rejected"
    assert "below safety floor" in payload["detail"]


def test_forced_loss_exit_is_allowed_when_reason_provided(tmp_path):
    now = datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)
    quotes = [
        {
            "symbol": "ETHUSD",
            "bid_price": 2004.0,
            "ask_price": 2005.0,
            "last_price": 2004.5,
            "as_of": now.isoformat(),
        },
        {
            "symbol": "ETHUSD",
            "bid_price": 1990.0,
            "ask_price": 1991.0,
            "last_price": 1990.5,
            "as_of": (now + timedelta(minutes=1)).isoformat(),
        },
    ]
    engine = _build_engine(tmp_path, now=now, quotes=quotes)
    claim = engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )
    engine.submit_order(
        type(
            "Order",
            (),
            {
                "account": "paper_test",
                "bot_id": "paper_test_v1",
                "session_id": claim["session_id"],
                "symbol": "ETHUSD",
                "side": "buy",
                "qty": 0.2,
                "limit_price": 2005.0,
                "execution_mode": "paper",
                "allow_loss_exit": False,
                "force_exit_reason": None,
                "live_ack": None,
                "metadata": {},
            },
        )()
    )

    result = engine.submit_order(
        type(
            "Order",
            (),
            {
                "account": "paper_test",
                "bot_id": "paper_test_v1",
                "session_id": claim["session_id"],
                "symbol": "ETHUSD",
                "side": "sell",
                "qty": 0.2,
                "limit_price": 1990.0,
                "execution_mode": "paper",
                "allow_loss_exit": True,
                "force_exit_reason": "manual risk liquidation",
                "live_ack": None,
                "metadata": {},
            },
        )()
    )
    assert result["filled"] is True
    snapshot = engine.get_account_snapshot("paper_test")
    assert snapshot["positions"] == {}
    assert snapshot["realized_pnl"] < 0


def test_forced_loss_exit_requires_reason(tmp_path):
    now = datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)
    engine = _build_engine(
        tmp_path,
        now=now,
        quotes=[
            {
                "symbol": "ETHUSD",
                "bid_price": 2004.0,
                "ask_price": 2005.0,
                "last_price": 2004.5,
                "as_of": now.isoformat(),
            }
        ],
    )
    claim = engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )
    engine.submit_order(
        type(
            "Order",
            (),
            {
                "account": "paper_test",
                "bot_id": "paper_test_v1",
                "session_id": claim["session_id"],
                "symbol": "ETHUSD",
                "side": "buy",
                "qty": 0.2,
                "limit_price": 2005.0,
                "execution_mode": "paper",
                "allow_loss_exit": False,
                "force_exit_reason": None,
                "live_ack": None,
                "metadata": {},
            },
        )()
    )

    response = TestClient(create_app(engine)).post(
        "/api/v1/orders",
        json={
            "account": "paper_test",
            "bot_id": "paper_test_v1",
            "session_id": claim["session_id"],
            "symbol": "ETHUSD",
            "side": "sell",
            "qty": 0.2,
            "limit_price": 1990.0,
            "execution_mode": "paper",
            "allow_loss_exit": True,
            "force_exit_reason": "",
        },
    )
    assert response.status_code == 400
    assert "requires a non-empty force_exit_reason" in response.json()["detail"]


def test_open_order_fills_on_price_refresh(tmp_path):
    now = datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)
    quotes = [
        {
            "symbol": "ETHUSD",
            "bid_price": 2004.0,
            "ask_price": 2006.0,
            "last_price": 2005.0,
            "as_of": now.isoformat(),
        },
        {
            "symbol": "ETHUSD",
            "bid_price": 1997.0,
            "ask_price": 1998.0,
            "last_price": 1997.5,
            "as_of": (now + timedelta(minutes=5)).isoformat(),
        },
    ]
    engine = _build_engine(tmp_path, now=now, quotes=quotes)
    claim = engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )
    result = engine.submit_order(
        type(
            "Order",
            (),
            {
                "account": "paper_test",
                "bot_id": "paper_test_v1",
                "session_id": claim["session_id"],
                "symbol": "ETHUSD",
                "side": "buy",
                "qty": 0.2,
                "limit_price": 1998.0,
                "execution_mode": "paper",
                "allow_loss_exit": False,
                "force_exit_reason": None,
                "live_ack": None,
                "metadata": {},
            },
        )()
    )
    assert result["filled"] is False
    open_orders = engine.get_orders("paper_test")["open_orders"]
    assert len(open_orders) == 1

    refreshed = engine.refresh_prices(account="paper_test")
    assert refreshed["accounts"][0]["filled_orders"]
    snapshot = engine.get_account_snapshot("paper_test")
    assert snapshot["positions"]["ETHUSD"]["qty"] == 0.2


def test_public_account_endpoints_redact_writer_credentials(tmp_path):
    engine = _build_engine(tmp_path)
    claim = engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )
    engine.submit_order(
        type(
            "Order",
            (),
            {
                "account": "paper_test",
                "bot_id": "paper_test_v1",
                "session_id": claim["session_id"],
                "symbol": "ETHUSD",
                "side": "buy",
                "qty": 0.1,
                "limit_price": 1998.0,
                "execution_mode": "paper",
                "allow_loss_exit": False,
                "force_exit_reason": None,
                "live_ack": None,
                "metadata": {"tag": "open"},
            },
        )()
    )
    engine.submit_order(
        type(
            "Order",
            (),
            {
                "account": "paper_test",
                "bot_id": "paper_test_v1",
                "session_id": claim["session_id"],
                "symbol": "ETHUSD",
                "side": "buy",
                "qty": 0.2,
                "limit_price": 2005.0,
                "execution_mode": "paper",
                "allow_loss_exit": False,
                "force_exit_reason": None,
                "live_ack": None,
                "metadata": {"tag": "entry"},
            },
        )()
    )

    app = create_app(engine)
    with TestClient(app) as client:
        accounts = client.get("/api/v1/accounts")
        snapshot = client.get("/api/v1/account/paper_test")
        orders = client.get("/api/v1/orders/paper_test", params={"include_history": "true"})

    assert accounts.status_code == 200
    assert "allowed_bot_id" not in accounts.json()["accounts"][0]

    snapshot_payload = snapshot.json()
    assert snapshot.status_code == 200
    assert "allowed_bot_id" not in snapshot_payload
    assert snapshot_payload["writer_claim"]["active"] is True
    assert "bot_id" not in snapshot_payload["writer_claim"]
    assert "session_id" not in snapshot_payload["writer_claim"]
    assert len(snapshot_payload["open_orders"]) == 1
    assert snapshot_payload["open_orders"][0]["metadata"] == {"tag": "open"}
    assert "bot_id" not in snapshot_payload["open_orders"][0]
    assert "session_id" not in snapshot_payload["open_orders"][0]

    orders_payload = orders.json()
    assert orders.status_code == 200
    assert len(orders_payload["open_orders"]) == 1
    assert orders_payload["open_orders"][0]["metadata"] == {"tag": "open"}
    assert "bot_id" not in orders_payload["open_orders"][0]
    assert "session_id" not in orders_payload["open_orders"][0]
    assert len(orders_payload["order_history"]) == 1
    history_entry = orders_payload["order_history"][0]
    assert history_entry["symbol"] == "ETHUSD"
    assert history_entry["metadata"] == {"tag": "entry"}
    assert "bot_id" not in history_entry
    assert "session_id" not in history_entry
    audit_events = _read_jsonl(tmp_path / "state" / "trading_server" / "events" / "paper_test.audit.jsonl")
    event_types = [event["event_type"] for event in audit_events]
    assert event_types.count("writer_claimed") == 1
    submitted = [event for event in audit_events if event["event_type"] == "order_submitted"]
    assert len(submitted) == 2
    assert submitted[0]["filled"] is False
    assert submitted[1]["filled"] is True


def test_live_order_requires_ack_and_env_gate(tmp_path, monkeypatch):
    broker_calls: list[dict] = []

    def fake_live_executor(order: dict) -> dict:
        broker_calls.append(order)
        return {"broker_order_id": "broker-1", "status": "accepted"}

    engine = _build_engine(tmp_path, live_executor=fake_live_executor)
    claim = engine.claim_writer(
        type("Lease", (), {"account": "live_test", "bot_id": "live_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )
    app = create_app(engine)
    with TestClient(app) as client:
        missing_ack = client.post(
            "/api/v1/orders",
            json={
                "account": "live_test",
                "bot_id": "live_test_v1",
                "session_id": claim["session_id"],
                "symbol": "ETHUSD",
                "side": "buy",
                "qty": 0.1,
                "limit_price": 2000.0,
                "execution_mode": "live",
            },
        )
        assert missing_ack.status_code == 400
        assert "live_ack must equal LIVE" in missing_ack.json()["detail"]

        monkeypatch.setenv("ALLOW_ALPACA_LIVE_TRADING", "1")
        accepted = client.post(
            "/api/v1/orders",
            json={
                "account": "live_test",
                "bot_id": "live_test_v1",
                "session_id": claim["session_id"],
                "symbol": "ETHUSD",
                "side": "buy",
                "qty": 0.1,
                "limit_price": 2000.0,
                "execution_mode": "live",
                "live_ack": "LIVE",
            },
        )
        history = client.get("/api/v1/orders/live_test?include_history=true")
    assert accepted.status_code == 200
    assert len(broker_calls) == 1
    assert broker_calls[0]["symbol"] == "ETHUSD"
    assert accepted.json()["order"]["broker_response"]["broker_order_id"] == "broker-1"
    assert accepted.json()["filled"] is False
    assert history.status_code == 200
    assert len(history.json()["open_orders"]) == 1
    assert history.json()["open_orders"][0]["broker_response"]["status"] == "accepted"
    assert history.json()["order_history"] == []
    audit_events = _read_jsonl(tmp_path / "state" / "trading_server" / "events" / "live_test.audit.jsonl")
    submitted = [event for event in audit_events if event["event_type"] == "order_submitted"]
    assert submitted[-1]["execution_mode"] == "live"
    assert submitted[-1]["qty"] == 0.1
    assert submitted[-1]["limit_price"] == 2000.0
    assert submitted[-1]["broker_order_id"] == "broker-1"
    assert submitted[-1]["broker_status"] == "accepted"


def test_live_broker_failure_is_audited_and_logged(tmp_path, monkeypatch):
    def failing_live_executor(_order: dict) -> dict:
        raise RuntimeError("alpaca offline")

    engine = _build_engine(tmp_path, live_executor=failing_live_executor)
    claim = engine.claim_writer(
        type("Lease", (), {"account": "live_test", "bot_id": "live_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )
    monkeypatch.setenv("ALLOW_ALPACA_LIVE_TRADING", "1")

    with pytest.raises(HTTPException) as exc_info:
        engine.submit_order(
            type(
                "Order",
                (),
                {
                    "account": "live_test",
                    "bot_id": "live_test_v1",
                    "session_id": claim["session_id"],
                    "symbol": "ETHUSD",
                    "side": "buy",
                    "qty": 0.1,
                    "limit_price": 2000.0,
                    "execution_mode": "live",
                    "allow_loss_exit": False,
                    "force_exit_reason": None,
                    "live_ack": "LIVE",
                    "metadata": {},
                },
            )()
        )

    assert exc_info.value.status_code == 502
    assert "RuntimeError: alpaca offline" in str(exc_info.value.detail)
    audit_events = _read_jsonl(tmp_path / "state" / "trading_server" / "events" / "live_test.audit.jsonl")
    assert audit_events[-1]["event_type"] == "order_submit_failed"
    assert audit_events[-1]["status_code"] == 502
    assert audit_events[-1]["qty"] == 0.1
    assert audit_events[-1]["limit_price"] == 2000.0
    assert "alpaca offline" in str(audit_events[-1]["detail"])
    rejected = _read_jsonl(tmp_path / "state" / "trading_server" / "events" / "live_test.rejections.jsonl")
    assert rejected[-1]["reason"] == "order_submit_failed"
    assert "alpaca offline" in str(rejected[-1]["detail"])


def test_live_order_updates_account_state_when_marketable(tmp_path, monkeypatch):
    now = datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)
    broker_calls: list[dict] = []

    def fake_live_executor(order: dict) -> dict:
        broker_calls.append(order)
        return {"broker_order_id": "broker-2", "status": "accepted"}

    engine = _build_engine(
        tmp_path,
        now=now,
        quotes=[
            {
                "symbol": "ETHUSD",
                "bid_price": 1999.5,
                "ask_price": 2000.0,
                "last_price": 1999.8,
                "as_of": now.isoformat(),
            }
        ],
        live_executor=fake_live_executor,
    )
    claim = engine.claim_writer(
        type("Lease", (), {"account": "live_test", "bot_id": "live_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )
    monkeypatch.setenv("ALLOW_ALPACA_LIVE_TRADING", "1")

    result = engine.submit_order(
        type(
            "Order",
            (),
            {
                "account": "live_test",
                "bot_id": "live_test_v1",
                "session_id": claim["session_id"],
                "symbol": "ETHUSD",
                "side": "buy",
                "qty": 0.1,
                "limit_price": 2000.0,
                "execution_mode": "live",
                "allow_loss_exit": False,
                "force_exit_reason": None,
                "live_ack": "LIVE",
                "metadata": {"source": "unit-test"},
            },
        )()
    )

    snapshot = engine.get_account_snapshot("live_test")
    orders = engine.get_orders("live_test", include_history=True)

    assert len(broker_calls) == 1
    assert result["filled"] is True
    assert result["order"]["status"] == "filled"
    assert result["order"]["broker_response"]["broker_order_id"] == "broker-2"
    assert snapshot["positions"]["ETHUSD"]["qty"] == 0.1
    assert orders["open_orders"] == []
    assert len(orders["order_history"]) == 1
    assert orders["order_history"][0]["fill_price"] == 2000.0


def test_order_history_is_capped_to_max_order_history(tmp_path):
    engine = _build_engine(tmp_path, max_order_history=2)
    claim = engine.claim_writer(
        trading_server_module.WriterLeaseRequest(
            account="paper_test",
            bot_id="paper_test_v1",
            session_id="session-a",
        )
    )

    for idx in range(3):
        result = engine.submit_order(
            trading_server_module.OrderRequest(
                account="paper_test",
                bot_id="paper_test_v1",
                session_id=claim["session_id"],
                symbol="ETHUSD",
                side="buy",
                qty=0.1,
                limit_price=2001.0 + idx,
                execution_mode="paper",
                metadata={"seq": idx},
            )
        )
        assert result["filled"] is True

    orders = engine.get_orders("paper_test", include_history=True)
    assert [entry["metadata"]["seq"] for entry in orders["order_history"]] == [1, 2]

    persisted = json.loads(engine._account_path("paper_test").read_text(encoding="utf-8"))
    assert [entry["metadata"]["seq"] for entry in persisted["order_history"]] == [1, 2]


def test_order_history_preserves_nested_metadata_round_trip(tmp_path):
    engine = _build_engine(tmp_path)
    claim = engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )
    nested_metadata = {
        "source": "unit-test",
        "context": {"levels": [1, 2], "flags": {"probe": True}},
    }

    result = engine.submit_order(
        trading_server_module.OrderRequest(
            account="paper_test",
            bot_id="paper_test_v1",
            session_id=claim["session_id"],
            symbol="ETHUSD",
            side="buy",
            qty=0.1,
            limit_price=2005.0,
            execution_mode="paper",
            metadata=nested_metadata,
        )
    )

    assert result["order"]["metadata"] == nested_metadata

    orders = engine.get_orders("paper_test", include_history=True)
    assert orders["order_history"][0]["metadata"] == nested_metadata

    persisted = json.loads(engine._account_path("paper_test").read_text(encoding="utf-8"))
    assert persisted["order_history"][0]["metadata"] == nested_metadata


def test_live_open_order_fills_on_price_refresh(tmp_path, monkeypatch):
    now = datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)

    engine = _build_engine(
        tmp_path,
        now=now,
        quotes=[
            {
                "symbol": "ETHUSD",
                "bid_price": 2004.0,
                "ask_price": 2006.0,
                "last_price": 2005.0,
                "as_of": now.isoformat(),
            },
            {
                "symbol": "ETHUSD",
                "bid_price": 1999.5,
                "ask_price": 2000.0,
                "last_price": 1999.8,
                "as_of": (now + timedelta(minutes=5)).isoformat(),
            },
        ],
        live_executor=lambda order: {"broker_order_id": "broker-3", "status": "accepted"},
    )
    claim = engine.claim_writer(
        type("Lease", (), {"account": "live_test", "bot_id": "live_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )
    monkeypatch.setenv("ALLOW_ALPACA_LIVE_TRADING", "1")

    submitted = engine.submit_order(
        type(
            "Order",
            (),
            {
                "account": "live_test",
                "bot_id": "live_test_v1",
                "session_id": claim["session_id"],
                "symbol": "ETHUSD",
                "side": "buy",
                "qty": 0.1,
                "limit_price": 2000.0,
                "execution_mode": "live",
                "allow_loss_exit": False,
                "force_exit_reason": None,
                "live_ack": "LIVE",
                "metadata": {},
            },
        )()
    )

    assert submitted["filled"] is False
    assert len(engine.get_orders("live_test")["open_orders"]) == 1

    refreshed = engine.refresh_prices(account="live_test")
    snapshot = engine.get_account_snapshot("live_test")
    orders = engine.get_orders("live_test", include_history=True)

    assert refreshed["accounts"][0]["filled_orders"]
    assert snapshot["positions"]["ETHUSD"]["qty"] == 0.1
    assert orders["open_orders"] == []
    assert len(orders["order_history"]) == 1


def test_background_refresh_is_scoped_per_engine(tmp_path):
    engine_a = _build_engine(tmp_path / "a")
    engine_b = _build_engine(tmp_path / "b")
    calls = {"a": 0, "b": 0}
    first_refresh_a = threading.Event()
    first_refresh_b = threading.Event()

    def refresh_a(*args, **kwargs):
        calls["a"] += 1
        first_refresh_a.set()
        return {"accounts": []}

    def refresh_b(*args, **kwargs):
        calls["b"] += 1
        first_refresh_b.set()
        return {"accounts": []}

    engine_a.refresh_prices = refresh_a
    engine_b.refresh_prices = refresh_b

    try:
        thread_a = ensure_background_refresh(engine_a, poll_seconds=1)
        thread_b = ensure_background_refresh(engine_b, poll_seconds=1)
        assert thread_a is not thread_b
        assert first_refresh_a.wait(timeout=1.0)
        assert first_refresh_b.wait(timeout=1.0)

        stop_background_refresh(engine_a, timeout=1.0)

        assert not thread_a.is_alive()
        assert thread_b.is_alive()
        assert calls["a"] >= 1
        assert calls["b"] >= 1
    finally:
        stop_background_refresh(engine_b, timeout=1.0)
        stop_background_refresh(timeout=1.0)


def test_background_refresh_uses_reference_count_for_shared_engine(tmp_path):
    engine = _build_engine(tmp_path)
    first_refresh = threading.Event()

    def refresh_prices(*args, **kwargs):
        first_refresh.set()
        return {"accounts": []}

    engine.refresh_prices = refresh_prices

    try:
        thread_one = ensure_background_refresh(engine, poll_seconds=1)
        thread_two = ensure_background_refresh(engine, poll_seconds=1)

        assert thread_one is thread_two
        assert first_refresh.wait(timeout=1.0)

        stop_background_refresh(engine, timeout=1.0)
        assert thread_one.is_alive()

        stop_background_refresh(engine, timeout=1.0)
        assert not thread_one.is_alive()
    finally:
        stop_background_refresh(timeout=1.0)


def test_background_refresh_reuses_stopping_thread_for_same_engine(tmp_path):
    engine = _build_engine(tmp_path)
    entered_refresh = threading.Event()
    release_refresh = threading.Event()
    refresh_calls = 0

    def refresh_prices(*args, **kwargs):
        nonlocal refresh_calls
        refresh_calls += 1
        entered_refresh.set()
        release_refresh.wait(timeout=1.0)
        return {"accounts": []}

    engine.refresh_prices = refresh_prices

    try:
        thread_one = ensure_background_refresh(engine, poll_seconds=1)
        assert entered_refresh.wait(timeout=1.0)

        stop_background_refresh(engine, timeout=0.01)
        assert thread_one.is_alive()

        thread_two = ensure_background_refresh(engine, poll_seconds=1)

        assert thread_two is thread_one
        assert refresh_calls == 1
    finally:
        release_refresh.set()
        stop_background_refresh(timeout=1.0)


def test_refresh_prices_dedupes_quote_fetches_across_accounts(tmp_path):
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    quote_calls: list[str] = []
    now = datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)

    def quote_provider(symbol: str):
        normalized = str(symbol).upper()
        quote_calls.append(normalized)
        return {
            "symbol": normalized,
            "bid_price": 2000.0,
            "ask_price": 2001.0,
            "last_price": 2000.5,
            "as_of": now.isoformat(),
        }

    engine = TradingServerEngine(
        registry_path=registry,
        state_dir=tmp_path / "state",
        quote_provider=quote_provider,
        now_fn=lambda: now,
        quote_stale_seconds=300,
    )

    refreshed = engine.refresh_prices()

    assert quote_calls == ["ETHUSD"]
    assert len(refreshed["accounts"]) == 2
    assert all(account["refreshed_symbols"] == ["ETHUSD"] for account in refreshed["accounts"])
    assert all(account["unavailable_symbols"] == [] for account in refreshed["accounts"])


def test_refresh_prices_coalesces_inflight_quote_fetches_across_requests(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "accounts": {
                    "paper_test": {
                        "mode": "paper",
                        "allowed_bot_id": "paper_test_v1",
                        "starting_cash": 1000.0,
                        "sell_loss_cooldown_seconds": 1200,
                        "min_sell_markup_pct": 0.001,
                        "symbols": ["ETHUSD"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    now = datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)
    call_lock = threading.Lock()
    release_fetch = threading.Event()
    first_fetch_started = threading.Event()
    unexpected_second_fetch = threading.Event()

    def quote_provider(symbol: str):
        normalized = str(symbol).upper()
        with call_lock:
            if first_fetch_started.is_set():
                unexpected_second_fetch.set()
            else:
                first_fetch_started.set()
        assert release_fetch.wait(timeout=1.0)
        return {
            "symbol": normalized,
            "bid_price": 2000.0,
            "ask_price": 2001.0,
            "last_price": 2000.5,
            "as_of": now.isoformat(),
        }

    engine = TradingServerEngine(
        registry_path=registry,
        state_dir=tmp_path / "state",
        quote_provider=quote_provider,
        now_fn=lambda: now,
        quote_stale_seconds=300,
        quote_fetch_workers=2,
    )

    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []
    second_finished = threading.Event()

    def run_refresh(mark_done: threading.Event | None = None) -> None:
        try:
            results.append(engine.refresh_prices(account="paper_test"))
        except BaseException as exc:  # pragma: no cover - test harness
            errors.append(exc)
        finally:
            if mark_done is not None:
                mark_done.set()

    first_thread = threading.Thread(target=run_refresh)
    second_thread = threading.Thread(target=run_refresh, kwargs={"mark_done": second_finished})
    first_thread.start()
    assert first_fetch_started.wait(timeout=1.0)

    second_thread.start()
    assert not second_finished.wait(timeout=0.05)
    assert not unexpected_second_fetch.wait(timeout=0.2)

    release_fetch.set()
    first_thread.join(timeout=1.0)
    second_thread.join(timeout=1.0)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert errors == []
    assert len(results) == 2
    assert all(result["accounts"][0]["refreshed_symbols"] == ["ETHUSD"] for result in results)


def test_refresh_prices_fetches_multiple_symbols_in_parallel(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "accounts": {
                    "paper_test": {
                        "mode": "paper",
                        "allowed_bot_id": "paper_test_v1",
                        "starting_cash": 1000.0,
                        "sell_loss_cooldown_seconds": 1200,
                        "min_sell_markup_pct": 0.001,
                        "symbols": ["ETHUSD", "BTCUSD"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    now = datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)
    started_lock = threading.Lock()
    started_count = 0
    both_started = threading.Event()

    def quote_provider(symbol: str):
        nonlocal started_count
        with started_lock:
            started_count += 1
            if started_count >= 2:
                both_started.set()
        assert both_started.wait(timeout=0.5)
        return {
            "symbol": str(symbol).upper(),
            "bid_price": 2000.0,
            "ask_price": 2001.0,
            "last_price": 2000.5,
            "as_of": now.isoformat(),
        }

    engine = TradingServerEngine(
        registry_path=registry,
        state_dir=tmp_path / "state",
        quote_provider=quote_provider,
        now_fn=lambda: now,
        quote_stale_seconds=300,
        quote_fetch_workers=2,
    )

    started_at = time.perf_counter()
    refreshed = engine.refresh_prices(account="paper_test")
    duration = time.perf_counter() - started_at

    assert both_started.is_set()
    assert duration < 0.4
    assert refreshed["accounts"][0]["refreshed_symbols"] == ["BTCUSD", "ETHUSD"]
    assert refreshed["accounts"][0]["unavailable_symbols"] == []


def test_refresh_prices_reuses_quote_fetch_executor_across_calls(tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "accounts": {
                    "paper_test": {
                        "mode": "paper",
                        "allowed_bot_id": "paper_test_v1",
                        "starting_cash": 1000.0,
                        "sell_loss_cooldown_seconds": 1200,
                        "min_sell_markup_pct": 0.001,
                        "symbols": ["ETHUSD", "BTCUSD"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    now = datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)
    created = 0

    class CountingExecutor(RealThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            nonlocal created
            created += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(trading_server_module, "ThreadPoolExecutor", CountingExecutor)

    engine = TradingServerEngine(
        registry_path=registry,
        state_dir=tmp_path / "state",
        quote_provider=lambda symbol: {
            "symbol": str(symbol).upper(),
            "bid_price": 2000.0,
            "ask_price": 2001.0,
            "last_price": 2000.5,
            "as_of": now.isoformat(),
        },
        now_fn=lambda: now,
        quote_stale_seconds=0,
        quote_fetch_workers=2,
    )
    try:
        first = engine.refresh_prices(account="paper_test")
        second = engine.refresh_prices(account="paper_test")
    finally:
        engine.close()

    assert created == 1
    assert first["accounts"][0]["refreshed_symbols"] == ["BTCUSD", "ETHUSD"]
    assert second["accounts"][0]["refreshed_symbols"] == ["BTCUSD", "ETHUSD"]


def test_refresh_prices_skips_unavailable_quotes_without_failing(tmp_path):
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    now = datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)
    quote_sequence = [
        {
            "symbol": "ETHUSD",
            "bid_price": 2004.0,
            "ask_price": 2006.0,
            "last_price": 2005.0,
            "as_of": now.isoformat(),
        },
        None,
    ]

    def quote_provider(_symbol: str):
        if quote_sequence:
            return quote_sequence.pop(0)
        return None

    engine = TradingServerEngine(
        registry_path=registry,
        state_dir=tmp_path / "state",
        quote_provider=quote_provider,
        now_fn=lambda: now,
        quote_stale_seconds=0,
    )
    claim = engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )
    result = engine.submit_order(
        type(
            "Order",
            (),
            {
                "account": "paper_test",
                "bot_id": "paper_test_v1",
                "session_id": claim["session_id"],
                "symbol": "ETHUSD",
                "side": "buy",
                "qty": 0.2,
                "limit_price": 1998.0,
                "execution_mode": "paper",
                "allow_loss_exit": False,
                "force_exit_reason": None,
                "live_ack": None,
                "metadata": {},
            },
        )()
    )
    assert result["filled"] is False

    refreshed = engine.refresh_prices(account="paper_test")

    assert refreshed["accounts"][0]["filled_orders"] == []
    assert refreshed["accounts"][0]["refreshed_symbols"] == []
    assert refreshed["accounts"][0]["unavailable_symbols"] == ["ETHUSD"]
    assert refreshed["accounts"][0]["quote_error_symbols"] == []
    assert refreshed["accounts"][0]["unavailable_reasons"] == {"ETHUSD": "no quote returned"}
    assert len(engine.get_orders("paper_test")["open_orders"]) == 1
    audit_events = _read_jsonl(tmp_path / "state" / "trading_server" / "events" / "paper_test.audit.jsonl")
    assert audit_events[-1]["event_type"] == "prices_refreshed"
    assert audit_events[-1]["unavailable_symbols"] == ["ETHUSD"]
    assert audit_events[-1]["quote_error_symbols"] == []
    assert audit_events[-1]["unavailable_reasons"] == {"ETHUSD": "no quote returned"}


def test_refresh_prices_isolates_quote_provider_failures(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "accounts": {
                    "paper_test": {
                        "mode": "paper",
                        "allowed_bot_id": "paper_test_v1",
                        "starting_cash": 1000.0,
                        "sell_loss_cooldown_seconds": 1200,
                        "min_sell_markup_pct": 0.001,
                        "symbols": ["ETHUSD", "BTCUSD"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    now = datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)

    def quote_provider(symbol: str):
        normalized = str(symbol).upper()
        if normalized == "BTCUSD":
            raise RuntimeError("provider boom")
        return {
            "symbol": normalized,
            "bid_price": 2000.0,
            "ask_price": 2001.0,
            "last_price": 2000.5,
            "as_of": now.isoformat(),
        }

    engine = TradingServerEngine(
        registry_path=registry,
        state_dir=tmp_path / "state",
        quote_provider=quote_provider,
        now_fn=lambda: now,
        quote_stale_seconds=300,
        quote_fetch_workers=2,
    )

    refreshed = engine.refresh_prices(account="paper_test")

    assert refreshed["accounts"][0]["refreshed_symbols"] == ["ETHUSD"]
    assert refreshed["accounts"][0]["unavailable_symbols"] == ["BTCUSD"]
    assert refreshed["accounts"][0]["quote_error_symbols"] == ["BTCUSD"]
    assert refreshed["accounts"][0]["unavailable_reasons"] == {
        "BTCUSD": "RuntimeError: provider boom"
    }


def test_refresh_prices_uses_shared_quote_cache_without_persisting_idle_state(tmp_path):
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    now = datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)
    quote_sequence = [
        {
            "symbol": "ETHUSD",
            "bid_price": 2000.0,
            "ask_price": 2001.0,
            "last_price": 2000.5,
            "as_of": now.isoformat(),
        }
    ]

    def quote_provider(_symbol: str):
        if quote_sequence:
            return quote_sequence.pop(0)
        return None

    engine = TradingServerEngine(
        registry_path=registry,
        state_dir=tmp_path / "state",
        quote_provider=quote_provider,
        now_fn=lambda: now,
        quote_stale_seconds=300,
    )
    account_path = engine._account_path("paper_test")

    refreshed = engine.refresh_prices(account="paper_test")

    assert refreshed["accounts"][0]["refreshed_symbols"] == ["ETHUSD"]
    assert not account_path.exists()

    claim = engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )
    result = engine.submit_order(
        type(
            "Order",
            (),
            {
                "account": "paper_test",
                "bot_id": "paper_test_v1",
                "session_id": claim["session_id"],
                "symbol": "ETHUSD",
                "side": "buy",
                "qty": 0.1,
                "limit_price": 2001.0,
                "execution_mode": "paper",
                "allow_loss_exit": False,
                "force_exit_reason": None,
                "live_ack": None,
                "metadata": {},
            },
        )()
    )

    assert result["filled"] is True


def test_refresh_prices_reuses_fresh_shared_quote_cache_before_refetching(tmp_path):
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    now = datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)
    quote_calls: list[str] = []

    def quote_provider(symbol: str):
        normalized = str(symbol).upper()
        quote_calls.append(normalized)
        return {
            "symbol": normalized,
            "bid_price": 2000.0,
            "ask_price": 2001.0,
            "last_price": 2000.5,
            "as_of": now.isoformat(),
        }

    engine = TradingServerEngine(
        registry_path=registry,
        state_dir=tmp_path / "state",
        quote_provider=quote_provider,
        now_fn=lambda: now,
        quote_stale_seconds=300,
    )

    first = engine.refresh_prices(account="paper_test", symbols=["ETHUSD"])
    second = engine.refresh_prices(account="paper_test", symbols=["ETHUSD"])

    assert first["accounts"][0]["refreshed_symbols"] == ["ETHUSD"]
    assert second["accounts"][0]["refreshed_symbols"] == ["ETHUSD"]
    assert quote_calls == ["ETHUSD"]


def test_refresh_prices_shared_quote_cache_evicts_oldest_symbol(tmp_path):
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    now = datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)

    def _quote(symbol: str, price: float) -> dict[str, object]:
        return {
            "symbol": symbol,
            "bid_price": price,
            "ask_price": price + 1.0,
            "last_price": price + 0.5,
            "as_of": now.isoformat(),
        }

    quotes = {
        "AAPL": _quote("AAPL", 190.0),
        "BTCUSD": _quote("BTCUSD", 60000.0),
        "ETHUSD": _quote("ETHUSD", 2000.0),
    }

    engine = TradingServerEngine(
        registry_path=registry,
        state_dir=tmp_path / "state",
        quote_provider=lambda symbol: quotes[trading_server_module._normalize_symbol(symbol)],
        now_fn=lambda: now,
        quote_stale_seconds=300,
        shared_quote_cache_size=2,
    )

    engine.refresh_prices(account="paper_test", symbols=["BTCUSD", "ETHUSD"])
    engine.refresh_prices(account="paper_test", symbols=["BTCUSD"])
    engine.refresh_prices(account="paper_test", symbols=["AAPL"])

    assert list(engine._shared_quote_cache) == ["BTCUSD", "AAPL"]


def test_claim_writer_serializes_across_engine_instances(tmp_path):
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    now = datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)

    engine_a = TradingServerEngine(
        registry_path=registry,
        state_dir=tmp_path / "state",
        now_fn=lambda: now,
    )
    engine_b = TradingServerEngine(
        registry_path=registry,
        state_dir=tmp_path / "state",
        now_fn=lambda: now,
    )

    entered_load = threading.Event()
    allow_first_to_continue = threading.Event()
    second_finished = threading.Event()
    results: dict[str, object] = {}

    original_load = engine_a._load_state_unlocked

    def slow_load(account: str, config: dict[str, object]):
        state = original_load(account, config)
        if not entered_load.is_set():
            entered_load.set()
            assert allow_first_to_continue.wait(timeout=1.0)
        return state

    engine_a._load_state_unlocked = slow_load  # type: ignore[method-assign]

    def run_first():
        results["first"] = engine_a.claim_writer(
            type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "first", "ttl_seconds": 120})()
        )

    def run_second():
        try:
            results["second"] = engine_b.claim_writer(
                type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "second", "ttl_seconds": 120})()
            )
        except Exception as exc:  # pragma: no cover - asserted below
            results["second_error"] = exc
        finally:
            second_finished.set()

    first_thread = threading.Thread(target=run_first, name="claim-first")
    second_thread = threading.Thread(target=run_second, name="claim-second")

    first_thread.start()
    assert entered_load.wait(timeout=1.0)

    second_thread.start()
    assert not second_finished.wait(timeout=0.1)

    allow_first_to_continue.set()
    first_thread.join(timeout=1.0)
    second_thread.join(timeout=1.0)

    assert results["first"]["session_id"] == "first"  # type: ignore[index]
    assert "second" not in results
    second_error = results.get("second_error")
    assert isinstance(second_error, HTTPException)
    assert second_error.status_code == 409
    assert "writer lease already held" in str(second_error.detail)


def test_submit_order_serializes_across_engine_instances(tmp_path):
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    now = datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)

    def quote_provider(_symbol: str):
        return {
            "symbol": "ETHUSD",
            "bid_price": 2100.0,
            "ask_price": 2101.0,
            "last_price": 2100.5,
            "as_of": now.isoformat(),
        }

    engine_a = TradingServerEngine(
        registry_path=registry,
        state_dir=tmp_path / "state",
        quote_provider=quote_provider,
        now_fn=lambda: now,
    )
    engine_b = TradingServerEngine(
        registry_path=registry,
        state_dir=tmp_path / "state",
        quote_provider=quote_provider,
        now_fn=lambda: now,
    )

    claim = engine_a.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 120})()
    )

    entered_save = threading.Event()
    allow_first_to_continue = threading.Event()
    second_finished = threading.Event()
    errors: list[Exception] = []

    original_save = engine_a._save_state_unlocked

    def slow_save(state: dict[str, object]) -> None:
        if state.get("account") == "paper_test" and not entered_save.is_set():
            entered_save.set()
            assert allow_first_to_continue.wait(timeout=1.0)
        original_save(state)

    engine_a._save_state_unlocked = slow_save  # type: ignore[method-assign]

    def _order(qty: float):
        return type(
            "Order",
            (),
            {
                "account": "paper_test",
                "bot_id": "paper_test_v1",
                "session_id": claim["session_id"],
                "symbol": "ETHUSD",
                "side": "buy",
                "qty": qty,
                "limit_price": 2000.0,
                "execution_mode": "paper",
                "allow_loss_exit": False,
                "force_exit_reason": None,
                "live_ack": None,
                "metadata": {},
            },
        )()

    def run_first():
        try:
            engine_a.submit_order(_order(0.1))
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def run_second():
        try:
            engine_b.submit_order(_order(0.2))
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            second_finished.set()

    first_thread = threading.Thread(target=run_first, name="submit-first")
    second_thread = threading.Thread(target=run_second, name="submit-second")

    first_thread.start()
    assert entered_save.wait(timeout=1.0)

    second_thread.start()
    assert not second_finished.wait(timeout=0.1)

    allow_first_to_continue.set()
    first_thread.join(timeout=1.0)
    second_thread.join(timeout=1.0)

    assert errors == []
    open_orders = engine_a.get_orders("paper_test")["open_orders"]
    assert len(open_orders) == 2
    assert sorted(order["qty"] for order in open_orders) == [0.1, 0.2]


def test_account_snapshot_allows_parallel_reads_across_engine_instances(tmp_path):
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    now = datetime(2026, 3, 29, 20, 0, 0, tzinfo=timezone.utc)

    engine_a = TradingServerEngine(
        registry_path=registry,
        state_dir=tmp_path / "state",
        now_fn=lambda: now,
    )
    engine_b = TradingServerEngine(
        registry_path=registry,
        state_dir=tmp_path / "state",
        now_fn=lambda: now,
    )

    entered_load = threading.Event()
    allow_first_to_continue = threading.Event()
    second_finished = threading.Event()
    results: dict[str, object] = {}

    original_load = engine_a._load_state_unlocked

    def slow_load(account: str, config: dict[str, object]):
        state = original_load(account, config)
        if not entered_load.is_set():
            entered_load.set()
            assert allow_first_to_continue.wait(timeout=1.0)
        return state

    engine_a._load_state_unlocked = slow_load  # type: ignore[method-assign]

    def run_first():
        results["first"] = engine_a.get_account_snapshot("paper_test")

    def run_second():
        try:
            results["second"] = engine_b.get_account_snapshot("paper_test")
        finally:
            second_finished.set()

    first_thread = threading.Thread(target=run_first, name="snapshot-first")
    second_thread = threading.Thread(target=run_second, name="snapshot-second")

    first_thread.start()
    assert entered_load.wait(timeout=1.0)

    second_thread.start()
    assert second_finished.wait(timeout=0.5)

    allow_first_to_continue.set()
    first_thread.join(timeout=1.0)
    second_thread.join(timeout=1.0)

    assert results["first"]["account"] == "paper_test"  # type: ignore[index]
    assert results["second"]["account"] == "paper_test"  # type: ignore[index]


# ---------------------------------------------------------------------------
# 1-minute polling defaults and quote staleness
# ---------------------------------------------------------------------------


def test_default_poll_and_staleness_constants():
    """Background poll=60s, quote staleness=90s for 1-minute accuracy."""
    assert trading_server_module.DEFAULT_BACKGROUND_POLL_SECONDS == 60
    assert trading_server_module.DEFAULT_QUOTE_STALE_SECONDS == 90
    settings = TradingServerSettings.from_env()
    assert settings.background_poll_seconds == 60
    assert settings.quote_stale_seconds == 90


def test_quote_expires_after_staleness_window(tmp_path):
    """Quote cached at t=0 is fresh at t=60s, stale at t=91s."""
    t0 = datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc)
    current_now = {"value": t0}
    fetch_count = {"n": 0}

    def quote_provider(_symbol: str):
        fetch_count["n"] += 1
        return {
            "symbol": "ETHUSD",
            "bid_price": 2000.0 + fetch_count["n"],
            "ask_price": 2001.0 + fetch_count["n"],
            "last_price": 2000.5 + fetch_count["n"],
            "as_of": current_now["value"].isoformat(),
        }

    registry = tmp_path / "registry.json"
    _write_registry(registry)
    engine = TradingServerEngine(
        registry_path=registry,
        state_dir=tmp_path / "state",
        quote_provider=quote_provider,
        now_fn=lambda: current_now["value"],
        quote_stale_seconds=90,
    )

    # First refresh at t=0
    engine.refresh_prices(account="paper_test")
    assert fetch_count["n"] == 1

    # At t=60s, quote is still fresh — order uses cache, no re-fetch
    current_now["value"] = t0 + timedelta(seconds=60)
    claim = engine.claim_writer(
        type("Lease", (), {"account": "paper_test", "bot_id": "paper_test_v1", "session_id": "owner", "ttl_seconds": 300})()
    )
    result = engine.submit_order(
        type(
            "Order",
            (),
            {
                "account": "paper_test",
                "bot_id": "paper_test_v1",
                "session_id": claim["session_id"],
                "symbol": "ETHUSD",
                "side": "buy",
                "qty": 0.1,
                "limit_price": 2005.0,
                "execution_mode": "paper",
                "allow_loss_exit": False,
                "force_exit_reason": None,
                "live_ack": None,
                "metadata": {},
            },
        )()
    )
    assert fetch_count["n"] == 1
    assert result["filled"] is True

    # At t=91s, quote is stale → forces a new fetch
    current_now["value"] = t0 + timedelta(seconds=91)
    engine.refresh_prices(account="paper_test")
    assert fetch_count["n"] == 2


def test_background_refresh_fires_multiple_cycles(tmp_path):
    """Background thread fires refresh immediately and again after poll interval."""
    engine = _build_engine(tmp_path)
    refresh_count = {"n": 0}
    two_refreshes = threading.Event()

    def fake_refresh(*args, **kwargs):
        refresh_count["n"] += 1
        if refresh_count["n"] >= 2:
            two_refreshes.set()
        return {"accounts": []}

    engine.refresh_prices = fake_refresh

    try:
        thread = ensure_background_refresh(engine, poll_seconds=1)
        assert two_refreshes.wait(timeout=3.0), f"only got {refresh_count['n']} refreshes"
        assert refresh_count["n"] >= 2
        assert thread.is_alive()
    finally:
        stop_background_refresh(engine, timeout=2.0)
