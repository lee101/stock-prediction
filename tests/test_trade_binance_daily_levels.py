from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pandas as pd

from binanceneural import trade_binance_daily_levels as module


class _FakeWrapper:
    def predict_ohlc_batch(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        idx = pd.DatetimeIndex([pd.Timestamp("2026-03-17T00:00:00Z")], name="timestamp")
        buy_frame = pd.DataFrame({"low": [105.0]}, index=idx)
        sell_frame = pd.DataFrame({"high": [99.0]}, index=idx)
        p50_frame = pd.DataFrame({"close": [101.0]}, index=idx)
        return [
            SimpleNamespace(
                quantile_frames={
                    0.35: buy_frame,
                    0.5: p50_frame,
                    0.65: sell_frame,
                }
            )
        ]


class _CaptureWrapper:
    def __init__(self) -> None:
        self.last_context_end = None

    def predict_ohlc_batch(self, contexts, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.last_context_end = pd.Timestamp(contexts[0]["timestamp"].iloc[-1])
        idx = pd.DatetimeIndex([pd.Timestamp("2026-03-17T00:00:00Z")], name="timestamp")
        buy_frame = pd.DataFrame({"low": [100.0]}, index=idx)
        sell_frame = pd.DataFrame({"high": [110.0]}, index=idx)
        p50_frame = pd.DataFrame({"close": [105.0]}, index=idx)
        return [
            SimpleNamespace(
                quantile_frames={
                    0.35: buy_frame,
                    0.5: p50_frame,
                    0.65: sell_frame,
                }
            )
        ]


def test_forecast_today_levels_repairs_buy_sell_levels(monkeypatch, tmp_path) -> None:
    daily_root = tmp_path / "daily"
    daily_root.mkdir()
    symbol = "SOLUSD"
    (daily_root / f"{symbol}.csv").write_text("timestamp,open,high,low,close\n")

    index = pd.date_range("2026-02-10T00:00:00Z", periods=35, freq="D", tz="UTC")
    history = pd.DataFrame(
        {
            "timestamp": index,
            "open": [98.0 + i * 0.1 for i in range(len(index))],
            "high": [101.0 + i * 0.1 for i in range(len(index))],
            "low": [97.0 + i * 0.1 for i in range(len(index))],
            "close": [99.0 + i * 0.1 for i in range(len(index))],
        }
    )

    monkeypatch.setattr(module, "_load_daily_history", lambda path, symbol: history.copy())
    monkeypatch.setattr(module, "_utc_now", lambda: pd.Timestamp("2026-03-17T12:00:00Z").to_pydatetime())
    monkeypatch.setattr(
        module,
        "resolve_chronos2_params",
        lambda *args, **kwargs: {"context_length": 32, "batch_size": 4},
    )
    monkeypatch.setattr(
        module.Chronos2OHLCWrapper,
        "from_pretrained",
        classmethod(lambda cls, **kwargs: _FakeWrapper()),
    )

    levels = module._forecast_today_levels(
        symbol=symbol,
        daily_root=daily_root,
        buy_quantile=0.35,
        sell_quantile=0.65,
        context_length=32,
        batch_size=4,
        model_id="dummy",
        device_map="cpu",
    )

    assert levels.buy_price <= levels.predicted_close_p50 <= levels.sell_price
    assert levels.prev_close == float(history["close"].iloc[-1])
    assert levels.predicted_close_p50 == 101.0
    assert levels.predicted_close_return == (101.0 - float(history["close"].iloc[-1])) / float(history["close"].iloc[-1])


def test_forecast_today_levels_ignores_same_day_partial_bar(monkeypatch, tmp_path) -> None:
    daily_root = tmp_path / "daily"
    daily_root.mkdir()
    symbol = "BTCUSD"
    (daily_root / f"{symbol}.csv").write_text("timestamp,open,high,low,close\n")

    index = pd.date_range("2026-02-10T00:00:00Z", periods=36, freq="D", tz="UTC")
    history = pd.DataFrame(
        {
            "timestamp": index,
            "open": [100.0 + i for i in range(len(index))],
            "high": [101.0 + i for i in range(len(index))],
            "low": [99.0 + i for i in range(len(index))],
            "close": [100.5 + i for i in range(len(index))],
        }
    )
    capture = _CaptureWrapper()

    monkeypatch.setattr(module, "_load_daily_history", lambda path, symbol: history.copy())
    monkeypatch.setattr(module, "_utc_now", lambda: pd.Timestamp("2026-03-17T12:00:00Z").to_pydatetime())
    monkeypatch.setattr(
        module,
        "resolve_chronos2_params",
        lambda *args, **kwargs: {"context_length": 32, "batch_size": 4},
    )
    monkeypatch.setattr(
        module.Chronos2OHLCWrapper,
        "from_pretrained",
        classmethod(lambda cls, **kwargs: capture),
    )

    module._forecast_today_levels(
        symbol=symbol,
        daily_root=daily_root,
        buy_quantile=0.35,
        sell_quantile=0.65,
        context_length=32,
        batch_size=4,
        model_id="dummy",
        device_map="cpu",
    )

    assert capture.last_context_end == pd.Timestamp("2026-03-16T00:00:00Z")


def test_main_routes_btc_to_fdusd_and_funds_from_convertible_stables(monkeypatch, tmp_path: Path) -> None:
    day_start = pd.Timestamp("2026-03-17T00:00:00Z")
    issued_at = pd.Timestamp("2026-03-16T00:00:00Z")
    monkeypatch.setattr(
        module,
        "_forecast_today_levels",
        lambda **kwargs: module.DailyLevels(
            day_start=day_start,
            issued_at=issued_at,
            buy_price=100.0,
            sell_price=110.0,
            predicted_close_p50=106.0,
            prev_close=100.0,
            predicted_close_return=0.06,
        ),
    )
    monkeypatch.setattr(module, "_utc_now", lambda: pd.Timestamp("2026-03-17T12:00:00Z").to_pydatetime())
    monkeypatch.setattr(
        module,
        "resolve_symbol_rules",
        lambda symbol: SimpleNamespace(
            min_notional=10.0,
            min_qty=0.0001,
            step_size=None,
            tick_size=None,
            min_price=None,
        ),
    )
    monkeypatch.setattr(module, "get_free_balances", lambda symbol: (0.0, 0.0))
    monkeypatch.setattr(module, "get_spendable_quote_balance", lambda symbol: 100.0)
    monkeypatch.setattr(module.binance_wrapper, "get_symbol_price", lambda symbol: 100.1)

    funding_calls: list[tuple[str, float, bool]] = []
    monkeypatch.setattr(
        module,
        "ensure_stable_quote_balance",
        lambda symbol, needed_quote, dry_run=False: funding_calls.append((symbol, needed_quote, dry_run)) or True,
    )

    watcher_plans: list[module.WatcherPlan] = []
    monkeypatch.setattr(
        module,
        "spawn_watcher",
        lambda plan: watcher_plans.append(plan) or (tmp_path / "watcher.json"),
    )
    monkeypatch.setattr(module, "_wait_for_watcher", lambda path, poll_seconds: {"active": False, "state": "expired", "fill_qty": 0.0})

    module.main(
        [
            "--symbol",
            "BTCUSD",
            "--allocation-usdt",
            "50",
            "--max-cycles",
            "1",
            "--poll-seconds",
            "1",
            "--dry-run",
        ]
    )

    assert len(watcher_plans) == 1
    assert watcher_plans[0].symbol == "BTCUSD"
    assert watcher_plans[0].exchange_symbol == "BTCFDUSD"
    assert funding_calls
    assert funding_calls[0][0] == "BTCFDUSD"
    assert funding_calls[0][1] > 0.0


def test_main_skips_far_entry_without_funding_or_watcher(monkeypatch, tmp_path: Path) -> None:
    day_times = iter(
        [
            pd.Timestamp("2026-03-17T12:00:00Z").to_pydatetime(),
            pd.Timestamp("2026-03-18T00:00:01Z").to_pydatetime(),
        ]
    )
    monkeypatch.setattr(module, "_utc_now", lambda: next(day_times))
    monkeypatch.setattr(
        module,
        "_forecast_today_levels",
        lambda **kwargs: module.DailyLevels(
            day_start=pd.Timestamp("2026-03-17T00:00:00Z"),
            issued_at=pd.Timestamp("2026-03-16T00:00:00Z"),
            buy_price=100.0,
            sell_price=110.0,
            predicted_close_p50=106.0,
            prev_close=100.0,
            predicted_close_return=0.06,
        ),
    )
    monkeypatch.setattr(
        module,
        "resolve_symbol_rules",
        lambda symbol: SimpleNamespace(
            min_notional=10.0,
            min_qty=0.0001,
            step_size=None,
            tick_size=None,
            min_price=None,
        ),
    )
    monkeypatch.setattr(module, "get_free_balances", lambda symbol: (0.0, 0.0))
    monkeypatch.setattr(module, "get_spendable_quote_balance", lambda symbol: 100.0)
    monkeypatch.setattr(module.binance_wrapper, "get_symbol_price", lambda symbol: 101.0)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)

    funding_calls: list[tuple[str, float, bool]] = []
    monkeypatch.setattr(
        module,
        "ensure_stable_quote_balance",
        lambda symbol, needed_quote, dry_run=False: funding_calls.append((symbol, needed_quote, dry_run)) or True,
    )

    watcher_plans: list[module.WatcherPlan] = []
    monkeypatch.setattr(
        module,
        "spawn_watcher",
        lambda plan: watcher_plans.append(plan) or (tmp_path / "watcher.json"),
    )

    module.main(
        [
            "--symbol",
            "BTCUSD",
            "--allocation-usdt",
            "50",
            "--poll-seconds",
            "1",
            "--entry-proximity-bps",
            "25",
            "--dry-run",
        ]
    )

    assert funding_calls == []
    assert watcher_plans == []


def test_main_routes_existing_eth_inventory_sell_to_fdusd(monkeypatch, tmp_path: Path) -> None:
    day_times = iter(
        [
            pd.Timestamp("2026-03-17T12:00:00Z").to_pydatetime(),
            pd.Timestamp("2026-03-18T00:00:01Z").to_pydatetime(),
        ]
    )
    monkeypatch.setattr(module, "_utc_now", lambda: next(day_times))
    monkeypatch.setattr(
        module,
        "_forecast_today_levels",
        lambda **kwargs: module.DailyLevels(
            day_start=pd.Timestamp("2026-03-17T00:00:00Z"),
            issued_at=pd.Timestamp("2026-03-16T00:00:00Z"),
            buy_price=2000.0,
            sell_price=2100.0,
            predicted_close_p50=2050.0,
            prev_close=2000.0,
            predicted_close_return=0.025,
        ),
    )
    monkeypatch.setattr(
        module,
        "resolve_symbol_rules",
        lambda symbol: SimpleNamespace(
            min_notional=10.0,
            min_qty=0.0001,
            step_size=None,
            tick_size=None,
            min_price=None,
        ),
    )
    monkeypatch.setattr(module, "get_free_balances", lambda symbol: (0.0, 0.75))
    monkeypatch.setattr(module, "_wait_for_watcher", lambda path, poll_seconds: {"active": False, "state": "filled"})

    watcher_plans: list[module.WatcherPlan] = []
    monkeypatch.setattr(
        module,
        "spawn_watcher",
        lambda plan: watcher_plans.append(plan) or (tmp_path / "watcher.json"),
    )

    module.main(
        [
            "--symbol",
            "ETHUSD",
            "--poll-seconds",
            "1",
            "--dry-run",
        ]
    )

    assert watcher_plans
    assert watcher_plans[0].symbol == "ETHUSD"
    assert watcher_plans[0].side == "sell"
    assert watcher_plans[0].exchange_symbol == "ETHFDUSD"
