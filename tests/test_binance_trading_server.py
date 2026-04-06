from __future__ import annotations

import json
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.binance_trading_server import server as binance_server_module
from src.binance_trading_server.settings import BinanceTradingServerSettings
from src.binance_trading_server.server import (
    BinanceTradingServerEngine,
    OrderRequest,
    QuotePayload,
    WriterLeaseRequest,
    _normalize_symbol,
    create_app,
    ensure_background_refresh,
    stop_background_refresh,
)
from src.binance_trading_server.fee_schedule import (
    FDUSD_PAIRS,
    fee_fraction,
    get_fee_bps,
    margin_cost_per_hour,
)
from src.binance_trading_server.sell_guard import (
    SellGuardConfig,
    check_sell_guard,
)


def _make_registry(tmp: Path, accounts: dict[str, Any] | None = None) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    reg_path = tmp / "accounts.json"
    if accounts is None:
        accounts = {
            "test-paper": {
                "mode": "paper",
                "allowed_bot_id": "bot-1",
                "starting_cash": 10000.0,
                "base_currency": "USDT",
                "sell_loss_cooldown_seconds": 1800,
                "min_sell_markup_pct": 0.001,
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "margin_enabled": False,
            },
            "test-live": {
                "mode": "live",
                "allowed_bot_id": "bot-1",
                "starting_cash": 0,
                "base_currency": "USDT",
                "symbols": ["BTCUSDT"],
                "margin_enabled": True,
            },
        }
    reg_path.write_text(json.dumps({"accounts": accounts}))
    return reg_path


def _fake_quote(symbol: str, price: float = 100.0) -> QuotePayload:
    return {
        "symbol": symbol,
        "bid_price": price - 0.1,
        "ask_price": price + 0.1,
        "last_price": price,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class FakeClock:
    def __init__(self, start: datetime | None = None):
        self.now = start or datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def _make_engine(
    tmp_path: Path,
    accounts: dict[str, Any] | None = None,
    clock: FakeClock | None = None,
    *,
    max_order_history: int | None = None,
    auth_token: str | None = None,
) -> BinanceTradingServerEngine:
    reg = _make_registry(tmp_path / "config", accounts)
    clock = clock or FakeClock()
    return BinanceTradingServerEngine(
        registry_path=reg,
        state_dir=str(tmp_path / "state"),
        quote_provider=lambda sym: _fake_quote(sym, 100.0),
        now_fn=clock,
        max_order_history=max_order_history,
        auth_token=auth_token,
    )


def test_engine_uses_runtime_env_quote_defaults(tmp_path, monkeypatch):
    registry = _make_registry(tmp_path / "config")
    monkeypatch.setenv("BINANCE_TRADING_SERVER_REGISTRY_PATH", str(registry))
    monkeypatch.setenv("BINANCE_TRADING_SERVER_QUOTE_STALE_SECONDS", "17")
    monkeypatch.setenv("BINANCE_TRADING_SERVER_QUOTE_FETCH_WORKERS", "9")
    monkeypatch.setenv("BINANCE_TRADING_SERVER_MAX_ORDER_HISTORY", "12")
    monkeypatch.setenv("BINANCE_TRADING_SERVER_SHARED_QUOTE_CACHE_SIZE", "21")

    engine = BinanceTradingServerEngine(state_dir=str(tmp_path / "state"))

    assert engine.registry_path == registry
    assert engine.quote_stale_seconds == 17
    assert engine.quote_fetch_workers == 9
    assert engine.max_order_history == 12
    assert engine.shared_quote_cache_size == 21


def test_runtime_config_reports_auth_token_as_enabled_flag(tmp_path):
    engine = _make_engine(tmp_path, auth_token="binance-secret")

    response = TestClient(create_app(engine)).get(
        "/api/v1/runtime-config",
        headers={"Authorization": "Bearer binance-secret"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["auth_token"]["value"] is True
    assert payload["auth_token"]["source"] == "explicit"


def test_create_app_optionally_requires_bearer_auth(tmp_path):
    engine = _make_engine(tmp_path, auth_token="binance-secret")
    app = create_app(engine)

    with TestClient(app) as client:
        unauthorized = client.get("/api/v1/accounts")
        wrong = client.get("/api/v1/accounts", headers={"Authorization": "Bearer wrong"})
        authorized = client.get("/api/v1/accounts", headers={"Authorization": "Bearer binance-secret"})

    assert unauthorized.status_code == 401
    assert unauthorized.json()["detail"] == "invalid auth token"
    assert unauthorized.headers["www-authenticate"] == "Bearer"
    assert wrong.status_code == 401
    assert authorized.status_code == 200
    assert authorized.json()["accounts"]


def test_create_app_logs_sanitized_auth_failures(tmp_path, caplog):
    engine = _make_engine(tmp_path, auth_token="binance-secret")
    app = create_app(engine)
    caplog.set_level(logging.WARNING, logger=binance_server_module.__name__)

    with TestClient(app) as client:
        client.get("/api/v1/accounts")
        client.get("/api/v1/accounts", headers={"Authorization": "Bearer wrong"})

    assert "Rejected unauthorized binance trading server request" in caplog.text
    assert "path=/api/v1/accounts" in caplog.text
    assert "auth=missing" in caplog.text
    assert "auth=mismatch" in caplog.text
    assert "binance-secret" not in caplog.text
    assert "Bearer wrong" not in caplog.text


def test_missing_registry_error_reports_resolution_source(tmp_path, monkeypatch):
    missing_registry = tmp_path / "missing-binance-registry.json"
    monkeypatch.setenv("BINANCE_TRADING_SERVER_REGISTRY_PATH", str(missing_registry))

    with pytest.raises(RuntimeError) as exc_info:
        BinanceTradingServerEngine(state_dir=str(tmp_path / "state"))

    message = str(exc_info.value)
    assert str(missing_registry) in message
    assert "BINANCE_TRADING_SERVER_REGISTRY_PATH=" in message


def test_binance_trading_server_settings_collect_runtime_defaults(tmp_path, monkeypatch):
    registry = _make_registry(tmp_path / "config")
    monkeypatch.setenv("BINANCE_TRADING_SERVER_REGISTRY_PATH", str(registry))
    monkeypatch.setenv("BINANCE_TRADING_SERVER_QUOTE_STALE_SECONDS", "17")
    monkeypatch.setenv("BINANCE_TRADING_SERVER_QUOTE_FETCH_WORKERS", "9")
    monkeypatch.setenv("BINANCE_TRADING_SERVER_MAX_ORDER_HISTORY", "12")
    monkeypatch.setenv("BINANCE_TRADING_SERVER_SHARED_QUOTE_CACHE_SIZE", "21")

    settings = BinanceTradingServerSettings.from_env()

    assert settings.registry_path == registry
    assert settings.quote_stale_seconds == 17
    assert settings.quote_fetch_workers == 9
    assert settings.max_order_history == 12
    assert settings.shared_quote_cache_size == 21


def test_invalid_runtime_env_values_fall_back_to_safe_defaults(tmp_path, monkeypatch):
    registry = _make_registry(tmp_path / "config")
    monkeypatch.setenv("BINANCE_TRADING_SERVER_REGISTRY_PATH", str(registry))
    monkeypatch.setenv("BINANCE_TRADING_SERVER_QUOTE_STALE_SECONDS", "not-a-number")
    monkeypatch.setenv("BINANCE_TRADING_SERVER_QUOTE_FETCH_WORKERS", "0")
    monkeypatch.setenv("BINANCE_TRADING_SERVER_MAX_ORDER_HISTORY", "0")
    monkeypatch.setenv("BINANCE_TRADING_SERVER_SHARED_QUOTE_CACHE_SIZE", "0")

    engine = BinanceTradingServerEngine(state_dir=str(tmp_path / "state"))

    assert engine.quote_stale_seconds == binance_server_module.BinanceTradingServerSettings.from_env().quote_stale_seconds
    assert engine.quote_fetch_workers == 1
    assert engine.max_order_history == 1
    assert engine.shared_quote_cache_size == 1
    assert BinanceTradingServerSettings.from_env().shared_quote_cache_size == 1


def test_runtime_config_endpoint_reports_effective_values_and_sources(tmp_path, monkeypatch):
    registry = _make_registry(tmp_path / "config")
    monkeypatch.setenv("BINANCE_TRADING_SERVER_REGISTRY_PATH", str(registry))
    monkeypatch.setenv("BINANCE_TRADING_SERVER_BACKGROUND_POLL_SECONDS", "oops")
    monkeypatch.setenv("BINANCE_TRADING_SERVER_QUOTE_FETCH_WORKERS", "0")

    engine = BinanceTradingServerEngine(
        state_dir=str(tmp_path / "state"),
        registry_path=registry,
        quote_stale_seconds=5,
        shared_quote_cache_size=11,
    )

    response = TestClient(create_app(engine)).get("/api/v1/runtime-config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["registry_path"]["value"] == str(registry)
    assert payload["registry_path"]["source"] == "explicit"
    assert payload["quote_stale_seconds"]["value"] == 5
    assert payload["quote_stale_seconds"]["source"] == "explicit"
    assert payload["shared_quote_cache_size"]["value"] == 11
    assert payload["shared_quote_cache_size"]["source"] == "explicit"
    assert payload["quote_fetch_workers"]["value"] == 1
    assert payload["quote_fetch_workers"]["source"] == "env"
    assert "clamped to 1" in payload["quote_fetch_workers"]["detail"]
    assert payload["background_poll_seconds"]["value"] == BinanceTradingServerSettings.from_env(
        registry_path=registry
    ).background_poll_seconds
    assert payload["background_poll_seconds"]["source"] == "env-invalid"
    assert "invalid" in payload["background_poll_seconds"]["detail"]


def test_account_state_guard_blocks_new_readers_while_writer_waits() -> None:
    guard = binance_server_module._AccountStateGuard()
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


def test_background_refresh_is_scoped_per_engine(tmp_path):
    engine_a = _make_engine(tmp_path / "a")
    engine_b = _make_engine(tmp_path / "b")
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
    engine = _make_engine(tmp_path)
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


def test_stop_background_refresh_removes_dead_handle(tmp_path):
    engine = _make_engine(tmp_path)
    first_refresh = threading.Event()

    def refresh_prices(*args, **kwargs):
        first_refresh.set()
        return {"accounts": []}

    engine.refresh_prices = refresh_prices

    try:
        thread = ensure_background_refresh(engine, poll_seconds=1)
        assert first_refresh.wait(timeout=1.0)
        stop_background_refresh(engine, timeout=1.0)

        assert not thread.is_alive()
        with binance_server_module._background_lock:
            assert engine not in binance_server_module._background_refreshers
    finally:
        stop_background_refresh(timeout=1.0)


def test_account_state_cache_reuses_disk_load_until_file_changes(tmp_path, monkeypatch):
    engine = _make_engine(tmp_path)
    account = "test-paper"
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

    assert first["cash"] == second["cash"] == 10000.0
    assert read_count == 1

    updated = json.loads(original_read_text(account_path, encoding="utf-8"))
    updated["cash"] = 7777.0
    account_path.write_text(json.dumps(updated), encoding="utf-8")

    third = engine.get_account_snapshot(account)

    assert third["cash"] == 7777.0
    assert read_count == 2


def test_account_state_saves_use_compact_json(tmp_path):
    engine = _make_engine(tmp_path)
    account = "test-paper"
    config = engine._config_for_account(account)

    with engine._account_state_guard(account):
        state = engine._load_state_unlocked(account, config)
        engine._save_state_unlocked(state)

    payload = engine._account_path(account).read_text(encoding="utf-8")

    assert "\n" not in payload
    assert json.loads(payload)["account"] == account


class TestFeeSchedule:
    def test_fdusd_zero_fee(self):
        for pair in FDUSD_PAIRS:
            assert get_fee_bps(pair) == 0.0

    def test_usdt_10bps(self):
        assert get_fee_bps("BTCUSDT") == 10.0
        assert get_fee_bps("ETHUSDT") == 10.0

    def test_fee_fraction(self):
        assert fee_fraction("BTCFDUSD") == 0.0
        assert abs(fee_fraction("BTCUSDT") - 0.001) < 1e-9

    def test_margin_cost(self):
        cost = margin_cost_per_hour(10000.0)
        expected = 10000.0 * 0.0625 / 8760.0
        assert abs(cost - expected) < 1e-6

    def test_custom_fees(self):
        assert get_fee_bps("BTCUSDT", custom_fees={"BTCUSDT": 5.0}) == 5.0


class TestSellGuard:
    def test_block_sell_below_entry_within_cooldown(self):
        now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc)
        result = check_sell_guard(
            entry_price=100.0,
            limit_price=99.0,
            last_buy_at=now - timedelta(minutes=10),
            config=SellGuardConfig(cooldown_seconds=1800, min_markup_pct=0.001),
            now=now,
        )
        assert not result.allowed
        assert result.sell_floor == pytest.approx(100.1)

    def test_allow_sell_above_floor_within_cooldown(self):
        now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc)
        result = check_sell_guard(
            entry_price=100.0,
            limit_price=100.2,
            last_buy_at=now - timedelta(minutes=10),
            config=SellGuardConfig(cooldown_seconds=1800, min_markup_pct=0.001),
            now=now,
        )
        assert result.allowed

    def test_allow_sell_at_entry_after_cooldown(self):
        now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc)
        result = check_sell_guard(
            entry_price=100.0,
            limit_price=100.0,
            last_buy_at=now - timedelta(hours=1),
            config=SellGuardConfig(cooldown_seconds=1800),
            now=now,
        )
        assert result.allowed

    def test_block_sell_below_entry_after_cooldown(self):
        now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc)
        result = check_sell_guard(
            entry_price=100.0,
            limit_price=99.0,
            last_buy_at=now - timedelta(hours=1),
            config=SellGuardConfig(cooldown_seconds=1800),
            now=now,
        )
        assert not result.allowed

    def test_alert_mode_allows_but_warns(self):
        now = datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc)
        result = check_sell_guard(
            entry_price=100.0,
            limit_price=99.0,
            last_buy_at=now - timedelta(minutes=10),
            config=SellGuardConfig(cooldown_seconds=1800, mode="alert"),
            now=now,
        )
        assert result.allowed
        assert "ALERT" in result.reason

    def test_no_entry_price(self):
        result = check_sell_guard(
            entry_price=0.0, limit_price=50.0, last_buy_at=None,
            config=SellGuardConfig(),
        )
        assert result.allowed


class TestWriterLease:
    def test_registry_rejects_unsafe_allowed_bot_id(self, tmp_path):
        reg = _make_registry(
            tmp_path / "config",
            accounts={
                "test-paper": {
                    "mode": "paper",
                    "allowed_bot_id": "bot 1",
                    "symbols": ["BTCUSDT"],
                    "margin_enabled": False,
                }
            },
        )

        with pytest.raises(RuntimeError, match="unsupported bot_id"):
            BinanceTradingServerEngine(
                registry_path=reg,
                state_dir=str(tmp_path / "state"),
                quote_provider=lambda sym: _fake_quote(sym, 100.0),
                now_fn=FakeClock(),
            )

    def test_claim_and_heartbeat(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine.claim_writer(WriterLeaseRequest(account="test-paper", bot_id="bot-1"))
        assert result["account"] == "test-paper"
        assert result["mode"] == "paper"
        session_id = result["session_id"]

        hb = engine.heartbeat_writer(WriterLeaseRequest(
            account="test-paper", bot_id="bot-1", session_id=session_id,
        ))
        assert hb["session_id"] == session_id

    def test_wrong_bot_rejected(self, tmp_path):
        engine = _make_engine(tmp_path)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            engine.claim_writer(WriterLeaseRequest(account="test-paper", bot_id="wrong-bot"))
        assert exc_info.value.status_code == 403

    def test_second_session_rejected(self, tmp_path):
        engine = _make_engine(tmp_path)
        engine.claim_writer(WriterLeaseRequest(account="test-paper", bot_id="bot-1", session_id="s1"))
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            engine.claim_writer(WriterLeaseRequest(account="test-paper", bot_id="bot-1", session_id="s2"))
        assert exc_info.value.status_code == 409

    def test_expired_lease_allows_reclaim(self, tmp_path):
        clock = FakeClock()
        engine = _make_engine(tmp_path, clock=clock)
        engine.claim_writer(WriterLeaseRequest(account="test-paper", bot_id="bot-1", session_id="s1", ttl_seconds=60))
        clock.advance(120)
        result = engine.claim_writer(WriterLeaseRequest(account="test-paper", bot_id="bot-1", session_id="s2"))
        assert result["session_id"] == "s2"

    def test_unknown_account_404(self, tmp_path):
        engine = _make_engine(tmp_path)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            engine.claim_writer(WriterLeaseRequest(account="nonexistent", bot_id="bot-1"))
        assert exc_info.value.status_code == 404

    def test_claim_rejects_unsafe_session_id_direct_engine_call(self, tmp_path):
        engine = _make_engine(tmp_path)

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            engine.claim_writer(
                type(
                    "Lease",
                    (),
                    {
                        "account": "test-paper",
                        "bot_id": "bot-1",
                        "session_id": "bad/session",
                        "ttl_seconds": 120,
                    },
                )()
            )
        assert exc_info.value.status_code == 400
        assert "unsupported session_id" in str(exc_info.value.detail)

    def test_submit_order_rejects_unsafe_session_id_direct_engine_call(self, tmp_path):
        engine = _make_engine(tmp_path)
        engine.claim_writer(
            type("Lease", (), {"account": "test-paper", "bot_id": "bot-1", "session_id": "owner", "ttl_seconds": 120})()
        )

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            engine.submit_order(
                type(
                    "Order",
                    (),
                    {
                        "account": "test-paper",
                        "bot_id": "bot-1",
                        "session_id": "bad/session",
                        "symbol": "BTCUSDT",
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

    def test_submit_order_rejects_oversized_metadata_direct_engine_call(self, tmp_path):
        engine = _make_engine(tmp_path)
        engine.claim_writer(
            type("Lease", (), {"account": "test-paper", "bot_id": "bot-1", "session_id": "owner", "ttl_seconds": 120})()
        )
        oversized_metadata = {
            "payload": "x" * (binance_server_module.MAX_ORDER_METADATA_BYTES + 128)
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
            f"metadata exceeds {binance_server_module.MAX_ORDER_METADATA_BYTES} bytes "
            f"when serialized (got {encoded_size} bytes)"
        )

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            engine.submit_order(
                type(
                    "Order",
                    (),
                    {
                        "account": "test-paper",
                        "bot_id": "bot-1",
                        "session_id": "owner",
                        "symbol": "BTCUSDT",
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

    def test_submit_order_api_rejects_oversized_metadata(self, tmp_path):
        app = create_app(_make_engine(tmp_path))
        oversized_metadata = {
            "payload": "x" * (binance_server_module.MAX_ORDER_METADATA_BYTES + 128)
        }

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/orders",
                json={
                    "account": "test-paper",
                    "bot_id": "bot-1",
                    "session_id": "owner",
                    "symbol": "BTCUSDT",
                    "side": "buy",
                    "qty": 1.0,
                    "limit_price": 100.0,
                    "execution_mode": "paper",
                    "metadata": oversized_metadata,
                },
            )

        assert response.status_code == 422
        assert "metadata exceeds" in json.dumps(response.json())

    def test_submit_order_rejects_metadata_with_too_many_entries_direct_engine_call(self, tmp_path):
        engine = _make_engine(tmp_path)
        engine.claim_writer(WriterLeaseRequest(account="test-paper", bot_id="bot-1", session_id="owner"))
        oversized_metadata = {
            f"k{i}": i for i in range(binance_server_module.MAX_ORDER_METADATA_ITEMS + 1)
        }

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            engine.submit_order(
                type(
                    "Order",
                    (),
                    {
                        "account": "test-paper",
                        "bot_id": "bot-1",
                        "session_id": "owner",
                        "symbol": "BTCUSDT",
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

    def test_submit_order_rejects_non_finite_metadata_direct_engine_call(self, tmp_path):
        engine = _make_engine(tmp_path)
        engine.claim_writer(WriterLeaseRequest(account="test-paper", bot_id="bot-1", session_id="owner"))

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            engine.submit_order(
                type(
                    "Order",
                    (),
                    {
                        "account": "test-paper",
                        "bot_id": "bot-1",
                        "session_id": "owner",
                        "symbol": "BTCUSDT",
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

    def test_submit_order_rejects_non_finite_qty_and_sanitizes_rejection_log(self, tmp_path):
        engine = _make_engine(tmp_path)
        engine.claim_writer(WriterLeaseRequest(account="test-paper", bot_id="bot-1", session_id="owner"))

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            engine.submit_order(
                type(
                    "Order",
                    (),
                    {
                        "account": "test-paper",
                        "bot_id": "bot-1",
                        "session_id": "owner",
                        "symbol": "BTCUSDT",
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
            tmp_path / "state" / "binance_trading_server" / "events" / "test-paper.rejections.jsonl"
        )
        assert rejected[-1]["qty"] is None
        assert rejected[-1]["limit_price"] == 100.0

    def test_heartbeat_rejection_is_audited(self, tmp_path):
        engine = _make_engine(tmp_path)
        engine.claim_writer(WriterLeaseRequest(account="test-paper", bot_id="bot-1", session_id="owner"))

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            engine.heartbeat_writer(
                WriterLeaseRequest(account="test-paper", bot_id="bot-1", session_id="other")
            )
        assert exc_info.value.status_code == 409

        events = _read_jsonl(tmp_path / "state" / "binance_trading_server" / "events" / "test-paper.audit.jsonl")
        assert [event["event_type"] for event in events][-2:] == ["writer_claimed", "writer_heartbeat_rejected"]
        assert events[-1]["session_id"] == "other"
        assert "does not own" in str(events[-1]["detail"])


class TestPaperOrders:
    def _setup(self, tmp_path):
        clock = FakeClock()
        engine = _make_engine(tmp_path, clock=clock)
        engine.claim_writer(WriterLeaseRequest(account="test-paper", bot_id="bot-1", session_id="s1", ttl_seconds=3600))
        return engine, clock

    def test_buy_fills_immediately(self, tmp_path):
        engine, clock = self._setup(tmp_path)
        result = engine.submit_order(OrderRequest(
            account="test-paper", bot_id="bot-1", session_id="s1",
            symbol="BTCUSDT", side="buy", qty=1.0, limit_price=100.5,
            execution_mode="paper",
        ))
        assert result["filled"] is True
        snap = engine.get_account_snapshot("test-paper")
        assert snap["cash"] < 10000.0
        assert "BTCUSDT" in snap["positions"]

    def test_sell_after_buy(self, tmp_path):
        engine, clock = self._setup(tmp_path)
        engine.submit_order(OrderRequest(
            account="test-paper", bot_id="bot-1", session_id="s1",
            symbol="BTCUSDT", side="buy", qty=1.0, limit_price=100.5,
            execution_mode="paper",
        ))
        clock.advance(2400)
        result = engine.submit_order(OrderRequest(
            account="test-paper", bot_id="bot-1", session_id="s1",
            symbol="BTCUSDT", side="sell", qty=1.0, limit_price=99.5,
            execution_mode="paper",
            allow_loss_exit=True,
            force_exit_reason="test_sell",
        ))
        assert result["filled"] is True
        snap = engine.get_account_snapshot("test-paper")
        assert "BTCUSDT" not in snap["positions"]

    def test_sell_below_entry_within_cooldown_rejected(self, tmp_path):
        engine, clock = self._setup(tmp_path)
        engine.submit_order(OrderRequest(
            account="test-paper", bot_id="bot-1", session_id="s1",
            symbol="BTCUSDT", side="buy", qty=1.0, limit_price=100.5,
            execution_mode="paper",
        ))
        clock.advance(60)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            engine.submit_order(OrderRequest(
                account="test-paper", bot_id="bot-1", session_id="s1",
                symbol="BTCUSDT", side="sell", qty=1.0, limit_price=90.0,
                execution_mode="paper",
            ))
        assert exc_info.value.status_code == 400
        assert "sell rejected" in str(exc_info.value.detail)

    def test_sell_with_force_exit_allowed(self, tmp_path):
        engine, clock = self._setup(tmp_path)
        engine.submit_order(OrderRequest(
            account="test-paper", bot_id="bot-1", session_id="s1",
            symbol="BTCUSDT", side="buy", qty=1.0, limit_price=100.5,
            execution_mode="paper",
        ))
        result = engine.submit_order(OrderRequest(
            account="test-paper", bot_id="bot-1", session_id="s1",
            symbol="BTCUSDT", side="sell", qty=1.0, limit_price=90.0,
            execution_mode="paper",
            allow_loss_exit=True,
            force_exit_reason="max_hold_exceeded",
        ))
        assert result["filled"] is True

    def test_insufficient_cash_rejected(self, tmp_path):
        engine, clock = self._setup(tmp_path)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            engine.submit_order(OrderRequest(
                account="test-paper", bot_id="bot-1", session_id="s1",
                symbol="BTCUSDT", side="buy", qty=200.0, limit_price=100.0,
                execution_mode="paper",
            ))
        assert exc_info.value.status_code == 400
        assert "insufficient" in str(exc_info.value.detail)

    def test_no_writer_claim_rejected(self, tmp_path):
        engine = _make_engine(tmp_path)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            engine.submit_order(OrderRequest(
                account="test-paper", bot_id="bot-1", session_id="s1",
                symbol="BTCUSDT", side="buy", qty=1.0, limit_price=100.0,
                execution_mode="paper",
            ))
        assert exc_info.value.status_code == 409

    def test_mode_mismatch_rejected(self, tmp_path):
        engine, _ = self._setup(tmp_path)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            engine.submit_order(OrderRequest(
                account="test-paper", bot_id="bot-1", session_id="s1",
                symbol="BTCUSDT", side="buy", qty=1.0, limit_price=100.0,
                execution_mode="live",
            ))
        assert exc_info.value.status_code == 400

    def test_fees_deducted(self, tmp_path):
        engine, clock = self._setup(tmp_path)
        engine.submit_order(OrderRequest(
            account="test-paper", bot_id="bot-1", session_id="s1",
            symbol="BTCUSDT", side="buy", qty=1.0, limit_price=100.5,
            execution_mode="paper",
        ))
        snap = engine.get_account_snapshot("test-paper")
        fill_price = snap["positions"]["BTCUSDT"]["avg_entry_price"]
        expected_fee = fill_price * 0.001
        expected_cash = 10000.0 - fill_price - expected_fee
        assert abs(snap["cash"] - expected_cash) < 0.01
        assert snap["total_fees"] > 0

    def test_open_order_not_filled_immediately(self, tmp_path):
        clock = FakeClock()
        engine = _make_engine(tmp_path, clock=clock)
        engine.quote_provider = lambda sym: _fake_quote(sym, 100.0)
        engine.claim_writer(WriterLeaseRequest(account="test-paper", bot_id="bot-1", session_id="s1"))
        result = engine.submit_order(OrderRequest(
            account="test-paper", bot_id="bot-1", session_id="s1",
            symbol="BTCUSDT", side="buy", qty=1.0, limit_price=90.0,
            execution_mode="paper",
        ))
        assert result["filled"] is False
        snap = engine.get_account_snapshot("test-paper")
        assert len(snap["open_orders"]) == 1

    def test_kline_fill(self, tmp_path):
        clock = FakeClock()
        engine = _make_engine(tmp_path, clock=clock)
        engine.claim_writer(WriterLeaseRequest(account="test-paper", bot_id="bot-1", session_id="s1"))
        engine.submit_order(OrderRequest(
            account="test-paper", bot_id="bot-1", session_id="s1",
            symbol="BTCUSDT", side="buy", qty=1.0, limit_price=90.0,
            execution_mode="paper",
        ))
        filled = engine.attempt_open_order_fills("test-paper", klines={
            "BTCUSDT": {"open": 95.0, "high": 95.0, "low": 88.0, "close": 92.0},
        })
        assert len(filled) == 1
        snap = engine.get_account_snapshot("test-paper")
        assert "BTCUSDT" in snap["positions"]
        assert len(snap["open_orders"]) == 0

    def test_refresh_prices_reports_unavailable_and_quote_error_symbols(self, tmp_path):
        clock = FakeClock()

        def quote_provider(symbol: str):
            if symbol == "BTCUSDT":
                return None
            if symbol == "ETHUSDT":
                raise TimeoutError("upstream timeout")
            return _fake_quote(symbol, 100.0)

        engine = BinanceTradingServerEngine(
            registry_path=_make_registry(tmp_path / "config"),
            state_dir=str(tmp_path / "state"),
            quote_provider=quote_provider,
            now_fn=clock,
        )

        result = engine.refresh_prices(account="test-paper")

        assert result == {
            "accounts": [
                {
                    "account": "test-paper",
                    "refreshed_symbols": [],
                    "unavailable_symbols": ["BTCUSDT", "ETHUSDT"],
                    "quote_error_symbols": ["ETHUSDT"],
                    "unavailable_reasons": {
                        "BTCUSDT": "no quote returned",
                        "ETHUSDT": "TimeoutError: upstream timeout",
                    },
                    "filled_orders": [],
                }
            ]
        }

        events = _read_jsonl(tmp_path / "state" / "binance_trading_server" / "events" / "test-paper.audit.jsonl")
        assert events[-1]["event_type"] == "prices_refreshed"
        assert events[-1]["unavailable_symbols"] == ["BTCUSDT", "ETHUSDT"]
        assert events[-1]["quote_error_symbols"] == ["ETHUSDT"]
        assert events[-1]["unavailable_reasons"] == {
            "BTCUSDT": "no quote returned",
            "ETHUSDT": "TimeoutError: upstream timeout",
        }

    def test_refresh_prices_uses_shared_quote_cache_without_persisting_idle_state(self, tmp_path):
        clock = FakeClock()
        quote_sequence = [_fake_quote("BTCUSDT", 100.0)]

        def quote_provider(_symbol: str):
            if quote_sequence:
                return quote_sequence.pop(0)
            return None

        engine = BinanceTradingServerEngine(
            registry_path=_make_registry(tmp_path / "config"),
            state_dir=str(tmp_path / "state"),
            quote_provider=quote_provider,
            now_fn=clock,
        )
        account_path = engine._account_path("test-paper")

        refreshed = engine.refresh_prices(account="test-paper", symbols=["BTCUSDT"])

        assert refreshed["accounts"][0]["refreshed_symbols"] == ["BTCUSDT"]
        assert not account_path.exists()

        engine.claim_writer(WriterLeaseRequest(account="test-paper", bot_id="bot-1", session_id="s1"))
        result = engine.submit_order(OrderRequest(
            account="test-paper", bot_id="bot-1", session_id="s1",
            symbol="BTCUSDT", side="buy", qty=1.0, limit_price=100.1,
            execution_mode="paper",
        ))

        assert result["filled"] is True

    def test_refresh_prices_reuses_fresh_shared_quote_cache_before_refetching(self, tmp_path):
        clock = FakeClock()
        quote_calls: list[str] = []

        def quote_provider(symbol: str):
            normalized = _normalize_symbol(symbol)
            quote_calls.append(normalized)
            return _fake_quote(normalized, 100.0)

        engine = BinanceTradingServerEngine(
            registry_path=_make_registry(tmp_path / "config"),
            state_dir=str(tmp_path / "state"),
            quote_provider=quote_provider,
            now_fn=clock,
        )

        first = engine.refresh_prices(account="test-paper", symbols=["BTCUSDT"])
        second = engine.refresh_prices(account="test-paper", symbols=["BTCUSDT"])

        assert first["accounts"][0]["refreshed_symbols"] == ["BTCUSDT"]
        assert second["accounts"][0]["refreshed_symbols"] == ["BTCUSDT"]
        assert quote_calls == ["BTCUSDT"]

    def test_refresh_prices_coalesces_inflight_quote_fetches_across_requests(self, tmp_path):
        clock = FakeClock()
        call_lock = threading.Lock()
        release_fetch = threading.Event()
        first_fetch_started = threading.Event()
        unexpected_second_fetch = threading.Event()

        def quote_provider(symbol: str):
            normalized = _normalize_symbol(symbol)
            with call_lock:
                if first_fetch_started.is_set():
                    unexpected_second_fetch.set()
                else:
                    first_fetch_started.set()
            assert release_fetch.wait(timeout=1.0)
            return _fake_quote(normalized, 100.0)

        engine = BinanceTradingServerEngine(
            registry_path=_make_registry(
                tmp_path / "config",
                accounts={
                    "test-paper": {
                        "mode": "paper",
                        "allowed_bot_id": "bot-1",
                        "starting_cash": 10000.0,
                        "base_currency": "USDT",
                        "sell_loss_cooldown_seconds": 1800,
                        "min_sell_markup_pct": 0.001,
                        "symbols": ["BTCUSDT"],
                        "margin_enabled": False,
                    }
                },
            ),
            state_dir=str(tmp_path / "state"),
            quote_provider=quote_provider,
            now_fn=clock,
        )

        results: list[dict[str, object]] = []
        errors: list[BaseException] = []
        second_finished = threading.Event()

        def run_refresh(mark_done: threading.Event | None = None) -> None:
            try:
                results.append(engine.refresh_prices(account="test-paper", symbols=["BTCUSDT"]))
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
        assert all(result["accounts"][0]["refreshed_symbols"] == ["BTCUSDT"] for result in results)

    def test_refresh_prices_fetches_multiple_symbols_in_parallel(self, tmp_path):
        clock = FakeClock()
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
            return _fake_quote(_normalize_symbol(symbol), 100.0)

        engine = BinanceTradingServerEngine(
            registry_path=_make_registry(tmp_path / "config"),
            state_dir=str(tmp_path / "state"),
            quote_provider=quote_provider,
            now_fn=clock,
            quote_fetch_workers=2,
        )
        try:
            started_at = time.perf_counter()
            refreshed = engine.refresh_prices(account="test-paper", symbols=["BTCUSDT", "ETHUSDT"])
            duration = time.perf_counter() - started_at
        finally:
            engine.close()

        assert both_started.is_set()
        assert duration < 0.4
        assert refreshed["accounts"][0]["refreshed_symbols"] == ["BTCUSDT", "ETHUSDT"]
        assert refreshed["accounts"][0]["unavailable_symbols"] == []

    def test_refresh_prices_reuses_quote_fetch_executor_across_calls(self, tmp_path, monkeypatch):
        clock = FakeClock()
        created = 0

        class CountingExecutor(RealThreadPoolExecutor):
            def __init__(self, *args, **kwargs):
                nonlocal created
                created += 1
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(binance_server_module, "ThreadPoolExecutor", CountingExecutor)

        engine = BinanceTradingServerEngine(
            registry_path=_make_registry(tmp_path / "config"),
            state_dir=str(tmp_path / "state"),
            quote_provider=lambda symbol: _fake_quote(_normalize_symbol(symbol), 100.0),
            now_fn=clock,
            quote_fetch_workers=2,
        )
        try:
            first = engine.refresh_prices(account="test-paper", symbols=["BTCUSDT", "ETHUSDT"])
            second = engine.refresh_prices(account="test-paper", symbols=["BTCUSDT", "ETHUSDT"])
        finally:
            engine.close()

        assert created == 1
        assert first["accounts"][0]["refreshed_symbols"] == ["BTCUSDT", "ETHUSDT"]
        assert second["accounts"][0]["refreshed_symbols"] == ["BTCUSDT", "ETHUSDT"]

    def test_refresh_prices_shared_quote_cache_evicts_oldest_symbol(self, tmp_path):
        clock = FakeClock()
        quotes = {
            "ADAUSDT": _fake_quote("ADAUSDT", 0.5),
            "BTCUSDT": _fake_quote("BTCUSDT", 100.0),
            "ETHUSDT": _fake_quote("ETHUSDT", 200.0),
        }

        engine = BinanceTradingServerEngine(
            registry_path=_make_registry(tmp_path / "config"),
            state_dir=str(tmp_path / "state"),
            quote_provider=lambda symbol: quotes[_normalize_symbol(symbol)],
            now_fn=clock,
            shared_quote_cache_size=2,
        )

        engine.refresh_prices(account="test-paper", symbols=["BTCUSDT", "ETHUSDT"])
        engine.refresh_prices(account="test-paper", symbols=["BTCUSDT"])
        engine.refresh_prices(account="test-paper", symbols=["ADAUSDT"])

        assert list(engine._shared_quote_cache) == ["BTCUSDT", "ADAUSDT"]

    def test_order_history_is_capped_to_max_order_history(self, tmp_path):
        clock = FakeClock()
        engine = _make_engine(tmp_path, clock=clock, max_order_history=2)
        engine.claim_writer(WriterLeaseRequest(account="test-paper", bot_id="bot-1", session_id="s1"))

        for idx in range(3):
            result = engine.submit_order(
                OrderRequest(
                    account="test-paper",
                    bot_id="bot-1",
                    session_id="s1",
                    symbol="BTCUSDT",
                    side="buy",
                    qty=1.0,
                    limit_price=100.5 + idx,
                    execution_mode="paper",
                    metadata={"seq": idx},
                )
            )
            assert result["filled"] is True

        orders = engine.get_orders("test-paper", include_history=True)
        assert [entry["metadata"]["seq"] for entry in orders["order_history"]] == [1, 2]

        persisted = json.loads(engine._account_path("test-paper").read_text(encoding="utf-8"))
        assert [entry["metadata"]["seq"] for entry in persisted["order_history"]] == [1, 2]

    def test_order_history_preserves_nested_metadata_round_trip(self, tmp_path):
        clock = FakeClock()
        engine = _make_engine(tmp_path, clock=clock)
        engine.claim_writer(WriterLeaseRequest(account="test-paper", bot_id="bot-1", session_id="s1"))
        nested_metadata = {
            "source": "test",
            "context": {"levels": [1, 2], "flags": {"probe": True}},
        }

        result = engine.submit_order(
            OrderRequest(
                account="test-paper",
                bot_id="bot-1",
                session_id="s1",
                symbol="BTCUSDT",
                side="buy",
                qty=1.0,
                limit_price=100.5,
                execution_mode="paper",
                metadata=nested_metadata,
            )
        )

        assert result["order"]["metadata"] == nested_metadata

        orders = engine.get_orders("test-paper", include_history=True)
        assert orders["order_history"][0]["metadata"] == nested_metadata

        persisted = json.loads(engine._account_path("test-paper").read_text(encoding="utf-8"))
        assert persisted["order_history"][0]["metadata"] == nested_metadata


class TestPnlTracking:
    def test_realized_pnl_on_sell(self, tmp_path):
        clock = FakeClock()
        engine = BinanceTradingServerEngine(
            registry_path=_make_registry(tmp_path / "config"),
            state_dir=str(tmp_path / "state"),
            quote_provider=lambda sym: _fake_quote(sym, 100.0),
            now_fn=clock,
        )
        engine.claim_writer(WriterLeaseRequest(account="test-paper", bot_id="bot-1", session_id="s1", ttl_seconds=3600))
        engine.submit_order(OrderRequest(
            account="test-paper", bot_id="bot-1", session_id="s1",
            symbol="BTCUSDT", side="buy", qty=1.0, limit_price=100.5,
            execution_mode="paper",
        ))
        clock.advance(2400)
        engine.quote_provider = lambda sym: _fake_quote(sym, 110.0)
        engine.submit_order(OrderRequest(
            account="test-paper", bot_id="bot-1", session_id="s1",
            symbol="BTCUSDT", side="sell", qty=1.0, limit_price=109.5,
            execution_mode="paper",
        ))
        snap = engine.get_account_snapshot("test-paper")
        assert snap["realized_pnl"] > 0
        assert snap["total_fees"] > 0


class TestAuditTrails:
    def test_fills_logged(self, tmp_path):
        clock = FakeClock()
        engine = _make_engine(tmp_path, clock=clock)
        engine.claim_writer(WriterLeaseRequest(account="test-paper", bot_id="bot-1", session_id="s1"))
        engine.submit_order(OrderRequest(
            account="test-paper", bot_id="bot-1", session_id="s1",
            symbol="BTCUSDT", side="buy", qty=1.0, limit_price=100.5,
            execution_mode="paper",
        ))
        fills_path = engine._fills_path("test-paper")
        assert fills_path.exists()
        lines = fills_path.read_text().strip().split("\n")
        assert len(lines) >= 1
        entry = json.loads(lines[0])
        assert entry["symbol"] == "BTCUSDT"
        assert entry["side"] == "buy"
        assert "fee" in entry

    def test_rejection_logged(self, tmp_path):
        engine = _make_engine(tmp_path)
        engine.claim_writer(WriterLeaseRequest(account="test-paper", bot_id="bot-1", session_id="s1"))
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            engine.submit_order(OrderRequest(
                account="test-paper", bot_id="bot-1", session_id="s1",
                symbol="BTCUSDT", side="buy", qty=999.0, limit_price=100.0,
                execution_mode="paper",
            ))
        rej_path = engine._rejections_path("test-paper")
        assert rej_path.exists()


class TestSymbolNormalization:
    def test_uppercase(self):
        assert _normalize_symbol("btcusdt") == "BTCUSDT"

    def test_slash_removed(self):
        assert _normalize_symbol("BTC/USDT") == "BTCUSDT"

    def test_dash_removed(self):
        assert _normalize_symbol("BTC-USDT") == "BTCUSDT"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _normalize_symbol("")


class TestAccountSnapshot:
    def test_default_state(self, tmp_path):
        engine = _make_engine(tmp_path)
        snap = engine.get_account_snapshot("test-paper")
        assert snap["cash"] == 10000.0
        assert snap["realized_pnl"] == 0.0
        assert snap["positions"] == {}

    def test_configured_accounts(self, tmp_path):
        engine = _make_engine(tmp_path)
        accounts = engine.configured_accounts()
        names = [a["account"] for a in accounts]
        assert "test-paper" in names
        assert "test-live" in names

    def test_invalid_persisted_mode_is_rejected(self, tmp_path):
        engine = _make_engine(tmp_path)
        account_path = engine._account_path("test-paper")
        account_path.parent.mkdir(parents=True, exist_ok=True)
        account_path.write_text(
            json.dumps(
                {
                    "account": "test-paper",
                    "mode": "demo",
                    "cash": 10000.0,
                    "positions": {},
                    "open_orders": [],
                    "order_history": [],
                    "price_cache": {},
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError, match="unsupported mode=demo"):
            engine.get_account_snapshot("test-paper")


class TestLiveOrderGate:
    def test_live_requires_ack(self, tmp_path):
        engine = _make_engine(tmp_path)
        engine.claim_writer(WriterLeaseRequest(account="test-live", bot_id="bot-1", session_id="s1"))
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            engine.submit_order(OrderRequest(
                account="test-live", bot_id="bot-1", session_id="s1",
                symbol="BTCUSDT", side="buy", qty=1.0, limit_price=100.0,
                execution_mode="live",
            ))
        assert exc_info.value.status_code in (400, 403)

    def test_live_success_audits_broker_context(self, tmp_path, monkeypatch):
        clock = FakeClock()
        engine = BinanceTradingServerEngine(
            registry_path=_make_registry(tmp_path / "config"),
            state_dir=str(tmp_path / "state"),
            quote_provider=lambda sym: _fake_quote(sym, 100.0),
            live_executor=lambda order: {"broker_order_id": "broker-9", "status": "accepted"},
            now_fn=clock,
        )
        engine.claim_writer(WriterLeaseRequest(account="test-live", bot_id="bot-1", session_id="s1"))
        monkeypatch.setenv("ALLOW_BINANCE_LIVE_TRADING", "1")

        result = engine.submit_order(OrderRequest(
            account="test-live", bot_id="bot-1", session_id="s1",
            symbol="BTCUSDT", side="buy", qty=1.0, limit_price=100.0,
            execution_mode="live", live_ack="LIVE",
        ))

        assert result["filled"] is False
        events = _read_jsonl(tmp_path / "state" / "binance_trading_server" / "events" / "test-live.audit.jsonl")
        assert events[-1]["event_type"] == "order_submitted"
        assert events[-1]["execution_mode"] == "live"
        assert events[-1]["qty"] == 1.0
        assert events[-1]["limit_price"] == 100.0
        assert events[-1]["broker_order_id"] == "broker-9"
        assert events[-1]["broker_status"] == "accepted"

    def test_live_broker_failure_is_audited_and_logged(self, tmp_path, monkeypatch):
        clock = FakeClock()

        def failing_live_executor(_order: dict[str, object]) -> dict[str, object]:
            raise RuntimeError("binance offline")

        engine = BinanceTradingServerEngine(
            registry_path=_make_registry(tmp_path / "config"),
            state_dir=str(tmp_path / "state"),
            quote_provider=lambda sym: _fake_quote(sym, 100.0),
            live_executor=failing_live_executor,
            now_fn=clock,
        )
        engine.claim_writer(WriterLeaseRequest(account="test-live", bot_id="bot-1", session_id="s1"))
        monkeypatch.setenv("ALLOW_BINANCE_LIVE_TRADING", "1")

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            engine.submit_order(OrderRequest(
                account="test-live", bot_id="bot-1", session_id="s1",
                symbol="BTCUSDT", side="buy", qty=1.0, limit_price=100.0,
                execution_mode="live", live_ack="LIVE",
            ))

        assert exc_info.value.status_code == 502
        assert "RuntimeError: binance offline" in str(exc_info.value.detail)
        events = _read_jsonl(tmp_path / "state" / "binance_trading_server" / "events" / "test-live.audit.jsonl")
        assert events[-1]["event_type"] == "order_submit_failed"
        assert events[-1]["status_code"] == 502
        assert events[-1]["qty"] == 1.0
        assert events[-1]["limit_price"] == 100.0
        assert "binance offline" in str(events[-1]["detail"])
        rejected = _read_jsonl(tmp_path / "state" / "binance_trading_server" / "events" / "test-live.rejections.jsonl")
        assert rejected[-1]["reason"] == "order_submit_failed"
        assert "binance offline" in str(rejected[-1]["detail"])
