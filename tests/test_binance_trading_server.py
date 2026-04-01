from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from src.binance_trading_server import server as binance_server_module
from src.binance_trading_server.server import (
    BinanceTradingServerEngine,
    OrderRequest,
    QuotePayload,
    WriterLeaseRequest,
    _normalize_symbol,
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
) -> BinanceTradingServerEngine:
    reg = _make_registry(tmp_path / "config", accounts)
    clock = clock or FakeClock()
    return BinanceTradingServerEngine(
        registry_path=reg,
        state_dir=str(tmp_path / "state"),
        quote_provider=lambda sym: _fake_quote(sym, 100.0),
        now_fn=clock,
        max_order_history=max_order_history,
    )


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
                    "filled_orders": [],
                }
            ]
        }

        events = _read_jsonl(tmp_path / "state" / "binance_trading_server" / "events" / "test-paper.audit.jsonl")
        assert events[-1]["event_type"] == "prices_refreshed"
        assert events[-1]["unavailable_symbols"] == ["BTCUSDT", "ETHUSDT"]
        assert events[-1]["quote_error_symbols"] == ["ETHUSDT"]

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
