from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.binance_hourly_xgb_margin_trader as trader


def _candidate(symbol: str = "DOGEUSDT", qty: float = 1200.0) -> trader.XGBLiveCandidate:
    return trader.XGBLiveCandidate(
        symbol=symbol,
        entry_price=0.1005,
        exit_price=0.098,
        edge=0.02,
        trade_amount=100.0,
        current_price=0.10,
        required_move_frac=0.01,
        target_notional_usdt=120.0,
        raw_qty=qty,
        quantized_qty=qty,
        quantized_entry_price=0.1005,
        quantized_exit_price=0.098,
        notional_usdt=qty * 0.1005,
    )


def test_build_order_payloads_short_entry_and_post_fill_exit() -> None:
    candidate = trader.XGBLiveCandidate(
        symbol="DOGEUSDT",
        entry_price=0.1005,
        exit_price=0.098,
        edge=0.02,
        trade_amount=100.0,
        current_price=0.10,
        required_move_frac=0.01,
        target_notional_usdt=120.0,
        raw_qty=1200.0,
        quantized_qty=1200.0,
        quantized_entry_price=0.1005,
        quantized_exit_price=0.098,
        notional_usdt=120.6,
    )

    entry, exit_after_fill = trader.build_order_payloads(candidate)

    assert entry.side == "SELL"
    assert entry.side_effect_type == "AUTO_BORROW_REPAY"
    assert entry.kind == "short_entry"
    assert exit_after_fill.side == "BUY"
    assert exit_after_fill.side_effect_type == "AUTO_REPAY"
    assert exit_after_fill.kind == "short_exit_after_fill"


def test_filter_live_tradable_frames_drops_stablecoin_pairs() -> None:
    frame = pd.DataFrame({"timestamp": [pd.Timestamp("2026-05-01T00:00:00Z")], "close": [1.0]})

    filtered = trader._filter_live_tradable_frames(
        {
            "FDUSDUSDT": frame,
            "USDCUSDT": frame,
            "BTCUSDT": frame,
        }
    )

    assert sorted(filtered) == ["BTCUSDT"]


def test_load_and_score_filters_stables_before_liquidity(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    timestamps = pd.date_range("2026-05-01T00:00:00Z", periods=6, freq="h")
    frame = pd.DataFrame({"timestamp": timestamps, "close": [1, 2, 3, 4, 5, 6]})
    frames = {"FDUSDUSDT": frame, "BTCUSDT": frame, "ETHUSDT": frame}
    seen: dict[str, list[str]] = {}

    monkeypatch.setattr(trader, "_discover_symbols", lambda *_args, **_kwargs: list(frames))
    monkeypatch.setattr(trader, "_load_hourly_frames", lambda *_args, **_kwargs: frames)

    def fake_filter_liquid_frames(frames_arg, **_kwargs):
        seen["liquidity_input"] = sorted(frames_arg)
        return frames_arg, pd.DataFrame(
            {
                "symbol": sorted(frames_arg),
                "median_dollar_volume": [100.0 for _ in frames_arg],
            }
        )

    monkeypatch.setattr(trader, "_filter_liquid_frames", fake_filter_liquid_frames)
    monkeypatch.setattr(trader, "build_model_frame", lambda *_args, **_kwargs: (pd.DataFrame({"f": [1.0]}), ["f"]))
    monkeypatch.setattr(trader, "fit_forecasters", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(trader, "score_eval_rows", lambda *_args, **_kwargs: pd.DataFrame({"f": [1.0]}))
    monkeypatch.setattr(
        trader,
        "build_actions_and_bars",
        lambda *_args, **_kwargs: (
            pd.DataFrame({"timestamp": [timestamps[-3]], "symbol": ["BTCUSDT"], "close": [100.0]}),
            pd.DataFrame({"timestamp": [timestamps[-3]], "symbol": ["BTCUSDT"]}),
        ),
    )

    args = SimpleNamespace(
        hourly_root=tmp_path,
        symbols="",
        min_bars=1,
        min_symbols_per_hour=1,
        liquidity_lookback_days=1,
        min_median_dollar_volume=0.0,
        max_symbols_by_dollar_volume=2,
        decision_lag=2,
        label_horizon=12,
        train_days=10,
        rounds=1,
        device="cpu",
        min_take_profit_bps=35.0,
        max_entry_gap_bps=120.0,
        max_exit_gap_bps=250.0,
        fee_rate=0.001,
        top_candidates_per_hour=6,
        entry_block_hours_utc="",
    )

    _end, _decision_ts, _bars, _actions, _frames, selected_symbols = trader._load_and_score_latest(args)

    assert seen["liquidity_input"] == ["BTCUSDT", "ETHUSDT"]
    assert selected_symbols == ["BTCUSDT", "ETHUSDT"]


def test_select_live_candidates_requires_two_corr_safe_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    ts = pd.Timestamp("2026-05-01T00:00:00Z")
    actions = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "AAAUSDT",
                "sell_price": 101.0,
                "buy_price": 99.0,
                "sell_amount": 100.0,
                "xgb_edge": 0.03,
                "close": 100.0,
            },
            {
                "timestamp": ts,
                "symbol": "BBBUSDT",
                "sell_price": 102.0,
                "buy_price": 98.0,
                "sell_amount": 80.0,
                "xgb_edge": 0.02,
                "close": 100.0,
            },
        ]
    )
    hist_ts = pd.date_range("2026-04-20T00:00:00Z", periods=240, freq="h")
    frames = {
        "AAAUSDT": pd.DataFrame({"timestamp": hist_ts, "close": range(100, 340)}),
        "BBBUSDT": pd.DataFrame({"timestamp": hist_ts, "close": range(300, 540)}),
    }
    monkeypatch.setattr(
        trader,
        "resolve_symbol_rules",
        lambda _symbol: SimpleNamespace(tick_size=0.01, step_size=0.1, min_qty=0.1, min_notional=12.0),
    )
    monkeypatch.setattr(trader, "quantize_price", lambda price, *, tick_size, side: price)
    monkeypatch.setattr(trader, "quantize_qty", lambda qty, *, step_size: qty)

    selected = trader.select_live_candidates(
        actions,
        frames,
        decision_ts=ts,
        equity_usdt=1000.0,
        risk_scale=1.0,
        use_live_prices=False,
        active_symbols=set(),
        max_total_entry_notional_usdt=0.0,
    )

    # The two synthetic series are nearly perfectly correlated, so the
    # correlation gate leaves fewer than the required two positions.
    assert selected == []


def test_select_live_candidates_allocates_two_short_orders(monkeypatch: pytest.MonkeyPatch) -> None:
    ts = pd.Timestamp("2026-05-01T00:00:00Z")
    actions = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "AAAUSDT",
                "sell_price": 101.0,
                "buy_price": 99.0,
                "sell_amount": 100.0,
                "xgb_edge": 0.03,
                "close": 100.0,
            },
            {
                "timestamp": ts,
                "symbol": "BBBUSDT",
                "sell_price": 102.0,
                "buy_price": 98.0,
                "sell_amount": 80.0,
                "xgb_edge": 0.02,
                "close": 100.0,
            },
        ]
    )
    hist_ts = pd.date_range("2026-04-20T00:00:00Z", periods=240, freq="h")
    frames = {
        "AAAUSDT": pd.DataFrame({"timestamp": hist_ts, "close": [100 + i for i in range(240)]}),
        "BBBUSDT": pd.DataFrame({"timestamp": hist_ts, "close": [300 + (-1) ** i * i for i in range(240)]}),
    }
    monkeypatch.setattr(
        trader,
        "resolve_symbol_rules",
        lambda _symbol: SimpleNamespace(tick_size=0.01, step_size=0.1, min_qty=0.1, min_notional=12.0),
    )
    monkeypatch.setattr(trader, "quantize_price", lambda price, *, tick_size, side: price)
    monkeypatch.setattr(trader, "quantize_qty", lambda qty, *, step_size: qty)

    selected = trader.select_live_candidates(
        actions,
        frames,
        decision_ts=ts,
        equity_usdt=1000.0,
        risk_scale=1.0,
        use_live_prices=False,
        active_symbols=set(),
        max_total_entry_notional_usdt=500.0,
    )

    assert [candidate.symbol for candidate in selected] == ["AAAUSDT", "BBBUSDT"]
    assert sum(candidate.notional_usdt for candidate in selected) <= 500.0 + 1e-9
    assert min(candidate.notional_usdt for candidate in selected) >= 100.0 - 1e-9


def test_place_entry_orders_places_exit_for_immediate_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_create_margin_order(symbol, side, order_type, quantity, **kwargs):
        calls.append({"symbol": symbol, "side": side, "type": order_type, "quantity": quantity, **kwargs})
        if side == "SELL":
            return {"orderId": 101, "status": "FILLED", "executedQty": str(quantity)}
        return {"orderId": 202, "status": "NEW", "executedQty": "0"}

    monkeypatch.setattr(trader, "create_margin_order", fake_create_margin_order)
    monkeypatch.setattr(trader, "get_margin_order", lambda *_args, **_kwargs: {"orderId": 101, "status": "FILLED", "executedQty": "1200"})
    monkeypatch.setattr(trader, "cancel_margin_order", lambda *_args, **_kwargs: {"status": "CANCELED"})
    monkeypatch.setattr(
        trader,
        "resolve_symbol_rules",
        lambda _symbol: SimpleNamespace(tick_size=0.01, step_size=0.1, min_qty=0.1, min_notional=12.0),
    )
    monkeypatch.setattr(trader, "quantize_qty", lambda qty, *, step_size: qty)

    placed = trader._place_entry_orders(
        [_candidate()],
        wait_seconds=0.0,
        poll_seconds=0.1,
        cancel_unfilled_entries=True,
    )

    assert [call["side"] for call in calls] == ["SELL", "BUY"]
    assert calls[0]["side_effect_type"] == "AUTO_BORROW_REPAY"
    assert calls[1]["side_effect_type"] == "AUTO_REPAY"
    assert placed[0]["exit_orders"][0]["status"] == "placed"
    assert placed[0]["cancel_order"] is None


def test_place_entry_orders_cancels_partial_entry_and_covers_filled_qty(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_create_margin_order(symbol, side, order_type, quantity, **kwargs):
        calls.append({"symbol": symbol, "side": side, "type": order_type, "quantity": quantity, **kwargs})
        if side == "SELL":
            return {"orderId": 101, "status": "NEW", "executedQty": "0"}
        return {"orderId": 202, "status": "NEW", "executedQty": "0"}

    monkeypatch.setattr(trader, "create_margin_order", fake_create_margin_order)
    monkeypatch.setattr(
        trader,
        "get_margin_order",
        lambda *_args, **_kwargs: {"orderId": 101, "status": "PARTIALLY_FILLED", "executedQty": "200"},
    )
    monkeypatch.setattr(trader, "cancel_margin_order", lambda *_args, **_kwargs: {"orderId": 101, "status": "CANCELED"})
    monkeypatch.setattr(
        trader,
        "resolve_symbol_rules",
        lambda _symbol: SimpleNamespace(tick_size=0.01, step_size=0.1, min_qty=0.1, min_notional=12.0),
    )
    monkeypatch.setattr(trader, "quantize_qty", lambda qty, *, step_size: qty)

    placed = trader._place_entry_orders(
        [_candidate(qty=1200.0)],
        wait_seconds=0.0,
        poll_seconds=0.1,
        cancel_unfilled_entries=True,
    )

    assert [call["side"] for call in calls] == ["SELL", "BUY"]
    assert calls[1]["quantity"] == pytest.approx(200.0)
    assert placed[0]["cancel_order"]["status"] == "CANCELED"
    assert placed[0]["exit_orders"][0]["quantity"] == pytest.approx(200.0)


def test_post_execute_coverage_waits_for_margin_settlement(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    monkeypatch.setattr(trader.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(trader, "load_coverage", lambda **_kwargs: ["covered"])

    rows = trader._load_post_execute_coverage(SimpleNamespace(post_cycle_settle_seconds=7.5))

    assert rows == ["covered"]
    assert sleeps == [7.5]
