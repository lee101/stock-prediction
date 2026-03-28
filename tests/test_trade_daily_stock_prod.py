from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import pandas as pd

import trade_daily_stock_prod as daily_stock


def _write_daily_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_local_daily_frames_aligns_common_dates(tmp_path: Path) -> None:
    _write_daily_csv(
        tmp_path / "AAPL.csv",
        [
            {"timestamp": "2026-01-01T00:00:00Z", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 100},
            {"timestamp": "2026-01-02T00:00:00Z", "open": 11, "high": 12, "low": 10, "close": 11.5, "volume": 110},
            {"timestamp": "2026-01-03T00:00:00Z", "open": 12, "high": 13, "low": 11, "close": 12.5, "volume": 120},
        ],
    )
    _write_daily_csv(
        tmp_path / "MSFT.csv",
        [
            {"timestamp": "2026-01-02T00:00:00Z", "open": 21, "high": 22, "low": 20, "close": 21.5, "volume": 210},
            {"timestamp": "2026-01-03T00:00:00Z", "open": 22, "high": 23, "low": 21, "close": 22.5, "volume": 220},
            {"timestamp": "2026-01-04T00:00:00Z", "open": 23, "high": 24, "low": 22, "close": 23.5, "volume": 230},
        ],
    )

    frames = daily_stock.load_local_daily_frames(["AAPL", "MSFT"], data_dir=str(tmp_path), min_days=2)

    assert list(frames) == ["AAPL", "MSFT"]
    assert frames["AAPL"]["timestamp"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-02", "2026-01-03"]
    assert frames["MSFT"]["timestamp"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-02", "2026-01-03"]


def test_frames_from_alpaca_bars_handles_single_symbol_without_multiindex() -> None:
    bars_df = pd.DataFrame(
        [
            {"timestamp": "2026-01-01T00:00:00Z", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 100},
            {"timestamp": "2026-01-02T00:00:00Z", "open": 11, "high": 12, "low": 10, "close": 11.5, "volume": 110},
        ]
    )

    frames = daily_stock._frames_from_alpaca_bars(
        symbols=["AAPL"],
        bars_df=bars_df,
        now=datetime(2026, 1, 3, tzinfo=timezone.utc),
    )

    assert list(frames) == ["AAPL"]
    assert len(frames["AAPL"]) == 2


def test_load_alpaca_daily_frames_recovers_missing_symbol_with_single_symbol_retry(monkeypatch) -> None:
    fake_alpaca_data_enums = ModuleType("alpaca.data.enums")
    fake_alpaca_data_enums.DataFeed = SimpleNamespace(IEX="iex", SIP="sip")
    monkeypatch.setitem(sys.modules, "alpaca.data.enums", fake_alpaca_data_enums)

    aapl = daily_stock._normalize_daily_frame(
        pd.DataFrame(
            [
                {"timestamp": "2026-01-01T00:00:00Z", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 100},
                {"timestamp": "2026-01-02T00:00:00Z", "open": 11, "high": 12, "low": 10, "close": 11.5, "volume": 110},
            ]
        )
    )
    msft = daily_stock._normalize_daily_frame(
        pd.DataFrame(
            [
                {"timestamp": "2026-01-01T00:00:00Z", "open": 20, "high": 21, "low": 19, "close": 20.5, "volume": 200},
                {"timestamp": "2026-01-02T00:00:00Z", "open": 21, "high": 22, "low": 20, "close": 21.5, "volume": 210},
            ]
        )
    )

    def _fake_request(*, symbols, feed, **_kwargs):
        key = (tuple(symbols), feed)
        if key == (("AAPL", "MSFT"), "iex"):
            return {"AAPL": aapl}
        if key == (("MSFT",), "iex"):
            return {"MSFT": msft}
        return {}

    monkeypatch.setattr(daily_stock, "_request_alpaca_daily_frames", _fake_request)

    frames = daily_stock.load_alpaca_daily_frames(
        ["AAPL", "MSFT"],
        paper=True,
        min_days=2,
        data_client=object(),
        now=datetime(2026, 1, 3, tzinfo=timezone.utc),
    )

    assert list(frames) == ["AAPL", "MSFT"]
    assert len(frames["AAPL"]) == 2
    assert len(frames["MSFT"]) == 2


def test_should_run_today_only_after_market_open_window() -> None:
    now = datetime(2026, 3, 16, 13, 40, tzinfo=timezone.utc)  # 09:40 ET

    assert daily_stock.should_run_today(now=now, is_market_open=True, last_run_date=None) is True
    assert daily_stock.should_run_today(now=now, is_market_open=False, last_run_date=None) is False
    assert daily_stock.should_run_today(
        now=now,
        is_market_open=True,
        last_run_date="2026-03-16",
    ) is False


def test_compute_target_qty_respects_buying_power_cap() -> None:
    account = SimpleNamespace(portfolio_value=10_000.0, buying_power=2_000.0)

    qty = daily_stock.compute_target_qty(account=account, price=100.0, allocation_pct=50.0)

    assert qty == 19.0


class _FakeClient:
    def __init__(self, positions: list[object], *, portfolio_value: float = 10_000.0, buying_power: float = 10_000.0):
        self._positions = positions
        self._account = SimpleNamespace(portfolio_value=portfolio_value, buying_power=buying_power)

    def get_all_positions(self):
        return list(self._positions)

    def get_account(self):
        return self._account


def test_execute_signal_closes_managed_position_then_opens_new_one(monkeypatch) -> None:
    orders: list[tuple[str, float, str]] = []

    def _fake_submit_market_order(client, *, symbol: str, qty: float, side: str):
        orders.append((symbol, qty, side))
        return SimpleNamespace(id=f"{symbol}-{side}")

    monkeypatch.setattr(daily_stock, "submit_market_order", _fake_submit_market_order)

    client = _FakeClient([SimpleNamespace(symbol="AAPL", qty="10", side="long")])
    state = daily_stock.StrategyState(active_symbol="AAPL", active_qty=10.0)
    signal = SimpleNamespace(symbol="MSFT", direction="long", action="long_MSFT")

    changed = daily_stock.execute_signal(
        signal,
        client=client,
        quotes={"AAPL": 100.0, "MSFT": 50.0},
        state=state,
        symbols=["AAPL", "MSFT"],
        allocation_pct=25.0,
        dry_run=False,
    )

    assert changed is True
    assert orders == [("AAPL", 10.0, "sell"), ("MSFT", 50.0, "buy")]
    assert state.active_symbol == "MSFT"
    assert state.active_qty == 50.0


def test_execute_signal_refuses_to_trade_with_unmanaged_position(monkeypatch) -> None:
    called = False

    def _fake_submit_market_order(client, *, symbol: str, qty: float, side: str):
        nonlocal called
        called = True
        return SimpleNamespace(id="unexpected")

    monkeypatch.setattr(daily_stock, "submit_market_order", _fake_submit_market_order)

    client = _FakeClient([SimpleNamespace(symbol="AAPL", qty="5", side="long")])
    state = daily_stock.StrategyState(active_symbol=None, active_qty=0.0)
    signal = SimpleNamespace(symbol="MSFT", direction="long", action="long_MSFT")

    changed = daily_stock.execute_signal(
        signal,
        client=client,
        quotes={"MSFT": 50.0},
        state=state,
        symbols=["AAPL", "MSFT"],
        allocation_pct=25.0,
        dry_run=False,
    )

    assert changed is False
    assert called is False
    assert state.active_symbol is None


def test_execute_signal_with_open_gate_only_closes_existing_position(monkeypatch) -> None:
    orders: list[tuple[str, float, str]] = []

    def _fake_submit_market_order(client, *, symbol: str, qty: float, side: str):
        orders.append((symbol, qty, side))
        return SimpleNamespace(id=f"{symbol}-{side}")

    monkeypatch.setattr(daily_stock, "submit_market_order", _fake_submit_market_order)

    client = _FakeClient([SimpleNamespace(symbol="AAPL", qty="10", side="long")])
    state = daily_stock.StrategyState(active_symbol="AAPL", active_qty=10.0)
    signal = SimpleNamespace(symbol="MSFT", direction="long", action="long_MSFT")

    changed = daily_stock.execute_signal(
        signal,
        client=client,
        quotes={"AAPL": 100.0, "MSFT": 50.0},
        state=state,
        symbols=["AAPL", "MSFT"],
        allocation_pct=25.0,
        dry_run=False,
        allow_open=False,
    )

    assert changed is True
    assert orders == [("AAPL", 10.0, "sell")]
    assert state.active_symbol is None
    assert state.active_qty == 0.0


def test_adopt_existing_position_populates_state() -> None:
    state = daily_stock.StrategyState()
    adopted = daily_stock.adopt_existing_position(
        state=state,
        live_positions={
            "NVDA": SimpleNamespace(symbol="NVDA", qty="3", side="long", avg_entry_price="121.5"),
        },
        now=datetime(2026, 3, 16, 13, 40, tzinfo=timezone.utc),
    )

    assert adopted is True
    assert state.active_symbol == "NVDA"
    assert state.active_qty == 3.0
    assert state.entry_price == 121.5
    assert state.entry_date == "2026-03-16"


def test_load_latest_quotes_falls_back_to_close_prices() -> None:
    class _BrokenQuoteClient:
        def get_stock_latest_quote(self, request):
            raise RuntimeError("quotes unavailable")

    prices = daily_stock.load_latest_quotes(
        ["AAPL", "MSFT"],
        paper=True,
        fallback_prices={"AAPL": 101.0, "MSFT": 202.0},
        data_client=_BrokenQuoteClient(),
    )

    assert prices == {"AAPL": 101.0, "MSFT": 202.0}

    prices_with_source, source, source_by_symbol = daily_stock.load_latest_quotes_with_source(
        ["AAPL", "MSFT"],
        paper=True,
        fallback_prices={"AAPL": 101.0, "MSFT": 202.0},
        data_client=_BrokenQuoteClient(),
    )
    assert prices_with_source == {"AAPL": 101.0, "MSFT": 202.0}
    assert source == "close_fallback"
    assert source_by_symbol == {"AAPL": "close_fallback", "MSFT": "close_fallback"}


def test_load_latest_quotes_tracks_partial_symbol_fallbacks() -> None:
    class _QuoteClient:
        def get_stock_latest_quote(self, request):
            return {
                "AAPL": SimpleNamespace(ask_price=150.0, bid_price=149.5),
                "MSFT": SimpleNamespace(ask_price=0.0, bid_price=0.0),
            }

    prices, source, source_by_symbol = daily_stock.load_latest_quotes_with_source(
        ["AAPL", "MSFT"],
        paper=True,
        fallback_prices={"AAPL": 101.0, "MSFT": 202.0},
        data_client=_QuoteClient(),
    )

    assert prices == {"AAPL": 150.0, "MSFT": 202.0}
    assert source == "mixed_fallback"
    assert source_by_symbol == {"AAPL": "alpaca", "MSFT": "close_fallback"}


def test_bars_are_fresh_uses_age_gate() -> None:
    latest_bar = pd.Timestamp("2026-03-14T00:00:00Z")

    assert daily_stock.bars_are_fresh(
        latest_bar=latest_bar,
        now=datetime(2026, 3, 16, 13, 40, tzinfo=timezone.utc),
        max_age_days=5,
    ) is True
    assert daily_stock.bars_are_fresh(
        latest_bar=latest_bar,
        now=datetime(2026, 3, 25, 13, 40, tzinfo=timezone.utc),
        max_age_days=5,
    ) is False


def test_run_once_falls_back_to_local_daily_frames(monkeypatch, tmp_path: Path) -> None:
    rows = []
    for idx in range(130):
        ts = pd.Timestamp("2025-09-01T00:00:00Z") + pd.Timedelta(days=idx)
        rows.append(
            {
                "timestamp": ts.isoformat(),
                "open": 100 + idx,
                "high": 101 + idx,
                "low": 99 + idx,
                "close": 100.5 + idx,
                "volume": 1_000 + idx,
            }
        )
    _write_daily_csv(tmp_path / "AAPL.csv", rows)
    _write_daily_csv(tmp_path / "MSFT.csv", rows)

    monkeypatch.setattr(
        daily_stock,
        "load_alpaca_daily_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("alpaca bars unavailable")),
    )
    monkeypatch.setattr(
        daily_stock,
        "build_trading_client",
        lambda paper: SimpleNamespace(
            get_all_positions=lambda: [],
            get_account=lambda: SimpleNamespace(cash=10_000.0, buying_power=10_000.0, portfolio_value=10_000.0),
        ),
    )
    monkeypatch.setattr(daily_stock, "build_data_client", lambda paper: object())
    monkeypatch.setattr(daily_stock, "execute_signal", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        daily_stock,
        "build_signal",
        lambda checkpoint, frames, portfolio=daily_stock.PortfolioContext(), device="cpu", extra_checkpoints=None: (
            SimpleNamespace(
                action="long_AAPL",
                symbol="AAPL",
                direction="long",
                confidence=0.75,
                value_estimate=1.25,
            ),
            {"AAPL": 123.0, "MSFT": 234.0},
        ),
    )
    monkeypatch.setattr(
        daily_stock,
        "load_latest_quotes_with_source",
        lambda symbols, paper, fallback_prices, data_client=None: (
            {"AAPL": 124.0, "MSFT": 235.0},
            "alpaca",
            {"AAPL": "alpaca", "MSFT": "alpaca"},
        ),
    )

    state_path = tmp_path / "state.json"
    payload = daily_stock.run_once(
        checkpoint="dummy.ckpt",
        symbols=["AAPL", "MSFT"],
        paper=True,
        allocation_pct=25.0,
        dry_run=True,
        data_source="alpaca",
        data_dir=str(tmp_path),
        state_path=state_path,
    )

    assert payload["bar_data_source"] == "local_fallback"
    assert payload["quote_data_source"] == "alpaca"
    assert payload["symbol"] == "AAPL"
    assert state_path.exists() is False


def test_run_once_dry_run_does_not_advance_state(monkeypatch, tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        '{"active_symbol": null, "active_qty": 0.0, "last_run_date": null, "last_signal_action": null, "last_signal_timestamp": null, "last_order_id": null}'
    )

    monkeypatch.setattr(
        daily_stock,
        "build_signal",
        lambda checkpoint, frames, portfolio=daily_stock.PortfolioContext(), device="cpu", extra_checkpoints=None: (
            SimpleNamespace(
                action="flat",
                symbol=None,
                direction=None,
                confidence=0.5,
                value_estimate=0.0,
            ),
            {"AAPL": 123.0, "MSFT": 234.0},
        ),
    )
    monkeypatch.setattr(
        daily_stock,
        "load_local_daily_frames",
        lambda symbols, data_dir, min_days=120: {
            "AAPL": pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2026-03-13T00:00:00Z"]),
                    "open": [1.0],
                    "high": [1.0],
                    "low": [1.0],
                    "close": [1.0],
                    "volume": [1.0],
                }
            ),
            "MSFT": pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2026-03-13T00:00:00Z"]),
                    "open": [1.0],
                    "high": [1.0],
                    "low": [1.0],
                    "close": [1.0],
                    "volume": [1.0],
                }
            ),
        },
    )

    before = state_path.read_text()
    daily_stock.run_once(
        checkpoint="dummy.ckpt",
        symbols=["AAPL", "MSFT"],
        paper=True,
        allocation_pct=25.0,
        dry_run=True,
        data_source="local",
        data_dir=str(tmp_path),
        state_path=state_path,
    )
    after = state_path.read_text()

    assert after == before


def test_run_once_market_closed_does_not_advance_state(monkeypatch, tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        '{"active_symbol": null, "active_qty": 0.0, "last_run_date": null, "last_signal_action": null, "last_signal_timestamp": null, "last_order_id": null}'
    )

    monkeypatch.setattr(
        daily_stock,
        "load_inference_frames",
        lambda symbols, paper, data_dir, now, data_client=None: (
            {
                "AAPL": pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(["2026-03-13T00:00:00Z"]),
                        "open": [1.0],
                        "high": [1.0],
                        "low": [1.0],
                        "close": [1.0],
                        "volume": [1.0],
                    }
                ),
                "MSFT": pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(["2026-03-13T00:00:00Z"]),
                        "open": [1.0],
                        "high": [1.0],
                        "low": [1.0],
                        "close": [1.0],
                        "volume": [1.0],
                    }
                ),
            },
            "alpaca",
        ),
    )
    monkeypatch.setattr(
        daily_stock,
        "build_trading_client",
        lambda paper: SimpleNamespace(
            get_all_positions=lambda: [],
            get_account=lambda: SimpleNamespace(cash=10_000.0, buying_power=10_000.0, portfolio_value=10_000.0),
            get_clock=lambda: SimpleNamespace(is_open=False),
        ),
    )
    monkeypatch.setattr(daily_stock, "build_data_client", lambda paper: object())
    monkeypatch.setattr(
        daily_stock,
        "load_latest_quotes_with_source",
        lambda symbols, paper, fallback_prices, data_client=None: (
            {"AAPL": 1.0, "MSFT": 1.0},
            "alpaca",
            {"AAPL": "alpaca", "MSFT": "alpaca"},
        ),
    )
    monkeypatch.setattr(
        daily_stock,
        "build_signal",
        lambda checkpoint, frames, portfolio=daily_stock.PortfolioContext(), device="cpu", extra_checkpoints=None: (
            SimpleNamespace(
                action="long_AAPL",
                symbol="AAPL",
                direction="long",
                confidence=0.5,
                value_estimate=0.0,
            ),
            {"AAPL": 1.0, "MSFT": 1.0},
        ),
    )
    execute_calls: list[object] = []
    monkeypatch.setattr(
        daily_stock,
        "execute_signal",
        lambda *args, **kwargs: execute_calls.append((args, kwargs)) or False,
    )

    before = state_path.read_text()
    daily_stock.run_once(
        checkpoint="dummy.ckpt",
        symbols=["AAPL", "MSFT"],
        paper=True,
        allocation_pct=25.0,
        dry_run=False,
        data_source="alpaca",
        data_dir=str(tmp_path),
        state_path=state_path,
    )
    after = state_path.read_text()

    assert execute_calls == []
    assert after == before
