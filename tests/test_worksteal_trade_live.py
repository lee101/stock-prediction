"""Tests for live work-steal runtime config wiring."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binance_worksteal.trade_live import (
    DEFAULT_CONFIG,
    EVENTS_FILE,
    _relative_bps_distance,
    _stage_entry_candidates,
    build_arg_parser,
    build_runtime_config,
    load_neural_model,
    log_event,
    normalize_live_positions,
    plan_legacy_rebalance_exits,
    prepare_neural_features,
    reconcile_exit_orders,
    reconcile_pending_entries,
    run_daily_cycle,
    run_entry_scan,
    run_health_report,
    run_neural_inference,
    synchronize_positions_from_exchange,
)
from binance_worksteal.model import PerSymbolWorkStealPolicy
from binance_worksteal.data import FEATURE_NAMES


def make_bars(prices: list[float], symbol: str) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-01-01", tz="UTC")
    for idx, close in enumerate(prices):
        ts = start + pd.Timedelta(days=idx)
        rows.append(
            {
                "timestamp": ts,
                "open": float(close),
                "high": float(close),
                "low": float(close),
                "close": float(close),
                "volume": 1000.0,
                "symbol": symbol,
            }
        )
    return pd.DataFrame(rows)


def make_bars_ohlc(bars_data: list[dict], symbol: str) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-01-01", tz="UTC")
    for idx, d in enumerate(bars_data):
        ts = start + pd.Timedelta(days=idx)
        rows.append({
            "timestamp": ts,
            "open": float(d.get("open", d["close"])),
            "high": float(d.get("high", d["close"])),
            "low": float(d.get("low", d["close"])),
            "close": float(d["close"]),
            "volume": float(d.get("volume", 1000.0)),
            "symbol": symbol,
        })
    return pd.DataFrame(rows)


def test_build_runtime_config_defaults_match_default_config():
    parser = build_arg_parser()
    args = parser.parse_args([])
    config = build_runtime_config(args)

    assert config == DEFAULT_CONFIG
    assert args.run_on_start is True
    assert args.startup_preview_only is True


def test_build_runtime_config_accepts_explicit_runtime_overrides():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--dip-pct", "0.20",
            "--proximity-pct", "0.01",
            "--profit-target", "0.05",
            "--stop-loss", "0.10",
            "--max-positions", "3",
            "--max-position-pct", "0.18",
            "--max-hold-days", "7",
            "--lookback-days", "10",
            "--ref-method", "high",
            "--sma-filter", "20",
            "--market-breadth-filter", "0.6",
            "--trailing-stop", "0.0",
            "--entry-proximity-bps", "15",
            "--risk-off-ref-method", "high",
            "--risk-off-market-breadth-filter", "0.7",
            "--risk-off-trigger-sma-period", "30",
            "--risk-off-trigger-momentum-period", "7",
            "--rebalance-seeded-positions",
        ]
    )
    config = build_runtime_config(args)

    assert config.dip_pct == 0.20
    assert config.proximity_pct == 0.01
    assert config.profit_target_pct == 0.05
    assert config.stop_loss_pct == 0.10
    assert config.max_positions == 3
    assert config.max_position_pct == 0.18
    assert config.max_hold_days == 7
    assert config.lookback_days == 10
    assert config.ref_price_method == "high"
    assert config.sma_filter_period == 20
    assert config.market_breadth_filter == 0.6
    assert config.trailing_stop_pct == 0.0
    assert config.entry_proximity_bps == 15.0
    assert config.risk_off_ref_price_method == "high"
    assert config.risk_off_market_breadth_filter == 0.7
    assert config.risk_off_trigger_sma_period == 30
    assert config.risk_off_trigger_momentum_period == 7
    assert config.rebalance_seeded_positions is True
    assert config.max_drawdown_exit == DEFAULT_CONFIG.max_drawdown_exit
    assert config.enable_shorts == DEFAULT_CONFIG.enable_shorts


def test_normalize_live_positions_defaults_legacy_metadata():
    config = DEFAULT_CONFIG
    positions = normalize_live_positions(
        {
            "ethusd": {
                "entry_price": "2000",
                "entry_date": "2026-03-17T00:00:00+00:00",
                "quantity": "0.5",
            }
        },
        config,
    )

    assert list(positions) == ["ETHUSD"]
    position = positions["ETHUSD"]
    assert position["entry_price"] == 2000.0
    assert position["quantity"] == 0.5
    assert position["peak_price"] == 2000.0
    assert position["target_sell"] == 2000.0 * (1.0 + config.profit_target_pct)
    assert position["stop_price"] == 2000.0 * (1.0 - config.stop_loss_pct)
    assert position["source"] == "legacy"


def test_normalize_live_positions_preserves_exit_order_metadata():
    positions = normalize_live_positions(
        {
            "adausd": {
                "entry_price": "0.2726",
                "entry_date": "2026-03-25T10:12:00+00:00",
                "quantity": "837.4",
                "target_sell": "0.278",
                "stop_price": "0.2453",
                "exit_order_id": 8538428237,
                "exit_order_symbol": "ADAUSDT",
                "exit_order_status": "NEW",
                "exit_price": "0.278",
                "exit_reason": "profit_target",
            }
        },
        DEFAULT_CONFIG,
    )

    position = positions["ADAUSD"]
    assert position["exit_order_id"] == 8538428237
    assert position["exit_order_symbol"] == "ADAUSDT"
    assert position["exit_order_status"] == "NEW"
    assert position["exit_price"] == 0.278
    assert position["exit_reason"] == "profit_target"


def test_plan_legacy_rebalance_exits_only_non_candidates():
    history = {
        "ALTUSD": make_bars([100.0] * 30, "ALTUSD"),
        "DIPUSD": make_bars([100.0] * 29 + [90.0], "DIPUSD"),
        "BTCUSD": make_bars([100.0] * 30, "BTCUSD"),
    }
    current_bars = {sym: bars.iloc[-1] for sym, bars in history.items()}
    positions = {
        "ALTUSD": {
            "entry_price": 100.0,
            "entry_date": "2026-03-01T00:00:00+00:00",
            "quantity": 1.0,
            "peak_price": 100.0,
            "target_sell": 103.0,
            "stop_price": 92.0,
            "source": "legacy",
        },
        "DIPUSD": {
            "entry_price": 90.0,
            "entry_date": "2026-03-01T00:00:00+00:00",
            "quantity": 1.0,
            "peak_price": 90.0,
            "target_sell": 92.7,
            "stop_price": 82.8,
            "source": "legacy",
        },
        "BTCUSD": {
            "entry_price": 100.0,
            "entry_date": "2026-03-01T00:00:00+00:00",
            "quantity": 1.0,
            "peak_price": 100.0,
            "target_sell": 103.0,
            "stop_price": 92.0,
            "source": "strategy",
        },
    }
    config = build_runtime_config(
        build_arg_parser().parse_args(
            [
                "--dip-pct",
                "0.10",
                "--proximity-pct",
                "0.02",
                "--lookback-days",
                "20",
                "--profit-target",
                "0.03",
                "--stop-loss",
                "0.08",
                "--rebalance-seeded-positions",
            ]
        )
    )

    exits, rebalance_symbols = plan_legacy_rebalance_exits(
        now=datetime(2026, 3, 18, tzinfo=timezone.utc),
        positions=positions,
        current_bars=current_bars,
        history=history,
        last_exit={},
        config=config,
    )

    assert rebalance_symbols == {"ALTUSD"}
    assert exits == [("ALTUSD", 100.0, "legacy_rebalance", positions["ALTUSD"])]
    assert positions["DIPUSD"]["source"] == "strategy"
    assert positions["BTCUSD"]["source"] == "strategy"


def test_build_runtime_config_can_disable_seeded_rebalance():
    config = build_runtime_config(build_arg_parser().parse_args(["--no-rebalance-seeded-positions"]))
    assert config.rebalance_seeded_positions is False


def test_parser_can_disable_startup_preview():
    args = build_arg_parser().parse_args(["--no-run-on-start", "--startup-live-cycle"])
    assert args.run_on_start is False
    assert args.startup_preview_only is False


def test_reconcile_pending_entries_promotes_filled_orders(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")

    class DummyClient:
        def get_margin_order(self, symbol, orderId):
            assert symbol == "ETHFDUSD"
            assert orderId == 123
            return {"status": "FILLED", "executedQty": "0.5", "price": "2000"}

    pending_entries = {
        "ETHUSD": {
            "buy_price": 1995.0,
            "quantity": 0.5,
            "target_sell": 2100.0,
            "stop_price": 1900.0,
            "placed_at": "2026-03-18T00:00:00+00:00",
            "expires_at": "2026-03-19T00:00:00+00:00",
            "order_id": 123,
            "source": "rule",
        }
    }
    positions = {}

    recent = reconcile_pending_entries(
        client=DummyClient(),
        pending_entries=pending_entries,
        positions=positions,
        now=datetime(2026, 3, 18, 12, tzinfo=timezone.utc),
        dry_run=False,
    )

    assert pending_entries == {}
    assert positions["ETHUSD"]["entry_price"] == 2000.0
    assert positions["ETHUSD"]["quantity"] == 0.5
    assert recent == [
        {
            "timestamp": "2026-03-18T12:00:00+00:00",
            "symbol": "ETHUSD",
            "side": "buy",
            "price": 2000.0,
            "quantity": 0.5,
            "pnl": 0.0,
            "reason": "pending_fill(rule)",
        }
    ]


def test_reconcile_exit_orders_removes_position_when_order_fills(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")

    class DummyClient:
        def get_margin_order(self, symbol, orderId):
            assert symbol == "ADAUSDT"
            assert orderId == 987
            return {
                "status": "FILLED",
                "executedQty": "837.4",
                "price": "0.278",
                "time": int(pd.Timestamp("2026-03-25T19:27:00Z").timestamp() * 1000),
            }

    positions = {
        "ADAUSD": {
            "entry_price": 0.2726,
            "entry_date": "2026-03-25T10:12:00+00:00",
            "quantity": 837.4,
            "peak_price": 0.285,
            "target_sell": 0.278,
            "stop_price": 0.2453,
            "source": "exchange_sync",
            "exit_order_id": 987,
            "exit_order_symbol": "ADAUSDT",
            "exit_order_status": "NEW",
            "exit_price": 0.278,
            "exit_reason": "profit_target",
        }
    }
    last_exit = {}

    recent = reconcile_exit_orders(
        client=DummyClient(),
        positions=positions,
        last_exit=last_exit,
        now=datetime(2026, 3, 25, 20, tzinfo=timezone.utc),
    )

    assert positions == {}
    assert last_exit["ADAUSD"] == "2026-03-25T19:27:00+00:00"
    assert recent == [
        {
            "timestamp": "2026-03-25T19:27:00+00:00",
            "symbol": "ADAUSD",
            "side": "sell",
            "price": 0.278,
            "quantity": 837.4,
            "reason": "profit_target",
            "pnl": (0.278 - 0.2726) * 837.4,
            "dry_run": False,
        }
    ]


def test_synchronize_positions_from_exchange_rebuilds_missing_positions(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")

    class DummyClient:
        def get_margin_account(self):
            return {
                "userAssets": [
                    {"asset": "ADA", "free": "419.8626", "netAsset": "836.5626", "borrowed": "0"},
                    {"asset": "ZEC", "free": "3.190806", "netAsset": "3.190806", "borrowed": "0"},
                    {"asset": "DOGE", "free": "0", "netAsset": "0.69314091", "borrowed": "1253.29493773"},
                ]
            }

        def get_open_margin_orders(self, **kwargs):
            return [
                {
                    "symbol": "ADAUSDT",
                    "orderId": 8538428237,
                    "status": "NEW",
                    "side": "SELL",
                    "price": "0.278",
                    "updateTime": 1774466874374,
                }
            ]

        def get_all_margin_orders(self, symbol, **kwargs):
            if symbol == "ADAUSDT":
                return [
                    {
                        "symbol": "ADAUSDT",
                        "status": "FILLED",
                        "side": "BUY",
                        "price": "0.2726",
                        "executedQty": "837.4",
                        "time": 1774433541859,
                        "updateTime": 1774433718086,
                    }
                ]
            if symbol == "ZECUSDT":
                return [
                    {
                        "symbol": "ZECUSDT",
                        "status": "FILLED",
                        "side": "BUY",
                        "price": "238.05",
                        "executedQty": "3.194",
                        "time": 1774440129390,
                        "updateTime": 1774440959689,
                    }
                ]
            return []

    positions = {}
    current_bars = {
        "ADAUSD": pd.Series({"close": 0.2694, "high": 0.2710}),
        "ZECUSD": pd.Series({"close": 230.87, "high": 241.00}),
        "DOGEUSD": pd.Series({"close": 0.095, "high": 0.096}),
    }

    events = synchronize_positions_from_exchange(
        client=DummyClient(),
        symbols=["ADAUSD", "ZECUSD", "DOGEUSD"],
        positions=positions,
        current_bars=current_bars,
        config=DEFAULT_CONFIG,
        now=datetime(2026, 3, 25, 21, tzinfo=timezone.utc),
    )

    assert set(positions) == {"ADAUSD", "ZECUSD"}
    assert positions["ADAUSD"]["entry_price"] == 0.2726
    assert positions["ADAUSD"]["quantity"] == 836.5626
    assert positions["ADAUSD"]["exit_order_id"] == 8538428237
    assert positions["ADAUSD"]["exit_order_symbol"] == "ADAUSDT"
    assert positions["ADAUSD"]["target_sell"] == 0.278
    assert positions["ZECUSD"]["entry_price"] == 238.05
    assert positions["ZECUSD"]["quantity"] == 3.190806
    assert "exit_order_id" not in positions["ZECUSD"]
    assert "DOGEUSD" not in positions
    assert [event["symbol"] for event in events] == ["ADAUSD", "ZECUSD"]


def test_run_daily_cycle_keeps_position_until_live_exit_fills(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.get_account_equity", lambda client: 10000.0)

    bars = make_bars_ohlc(
        [{"close": 100.0}] * 29 + [{"open": 100.0, "high": 120.0, "low": 99.0, "close": 110.0}],
        "DIPUSD",
    )
    monkeypatch.setattr("binance_worksteal.trade_live._fetch_all_bars", lambda client, symbols, lookback_days: {"DIPUSD": bars})

    state = {
        "positions": {
            "DIPUSD": {
                "entry_price": 100.0,
                "entry_date": "2026-01-20T00:00:00+00:00",
                "quantity": 5.0,
                "peak_price": 100.0,
                "target_sell": 115.0,
                "stop_price": 90.0,
                "source": "strategy",
            }
        },
        "pending_entries": {},
        "last_exit": {},
        "recent_trades": [],
        "peak_equity": 0.0,
    }
    (tmp_path / "live_state.json").write_text(json.dumps(state))

    class DummyClient:
        def get_margin_account(self):
            return {
                "userAssets": [
                    {"asset": "DIP", "free": "5.0", "netAsset": "5.0", "borrowed": "0"},
                ]
            }

        def get_open_margin_orders(self, **kwargs):
            return []

        def get_all_margin_orders(self, symbol, **kwargs):
            return []

        def create_margin_order(self, symbol, side, type, timeInForce, quantity, price):
            assert symbol == "DIPUSDT"
            assert side == "SELL"
            return {
                "symbol": symbol,
                "orderId": 4321,
                "status": "NEW",
            }

    config = build_runtime_config(
        build_arg_parser().parse_args(
            [
                "--dip-pct", "0.10",
                "--lookback-days", "20",
                "--profit-target", "0.15",
                "--stop-loss", "0.10",
                "--sma-filter", "0",
                "--risk-off-trigger-sma-period", "0",
                "--risk-off-trigger-momentum-period", "0",
                "--risk-off-market-breadth-filter", "0",
            ]
        )
    )

    run_daily_cycle(client=DummyClient(), symbols=["DIPUSD"], config=config, dry_run=False)

    payload = json.loads((tmp_path / "live_state.json").read_text())
    assert "DIPUSD" in payload["positions"]
    position = payload["positions"]["DIPUSD"]
    assert position["exit_order_id"] == 4321
    assert position["exit_order_symbol"] == "DIPUSDT"
    assert position["exit_order_status"] == "NEW"
    assert position["exit_reason"] == "profit_target"
    assert payload["last_exit"] == {}

    trades = [json.loads(line) for line in (tmp_path / "trade_log.jsonl").read_text().strip().split("\n")]
    assert any(trade["side"] == "staged_sell" for trade in trades)
    assert not any(trade["side"] == "sell" and trade.get("dry_run") is False for trade in trades)


def test_run_daily_cycle_stages_pending_entries_without_open_position(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")

    bars = make_bars([100.0] * 29 + [90.0], "DIPUSD")
    monkeypatch.setattr("binance_worksteal.trade_live.fetch_daily_bars", lambda client, symbol, lookback_days=30: bars)

    config = build_runtime_config(
        build_arg_parser().parse_args(
            [
                "--dip-pct",
                "0.10",
                "--proximity-pct",
                "0.02",
                "--lookback-days",
                "20",
                "--profit-target",
                "0.03",
                    "--stop-loss",
                    "0.08",
                    "--sma-filter",
                    "0",
                    "--risk-off-trigger-sma-period",
                    "0",
                    "--risk-off-trigger-momentum-period",
                    "0",
                    "--risk-off-market-breadth-filter",
                    "0",
                ]
            )
        )

    run_daily_cycle(client=None, symbols=["DIPUSD"], config=config, dry_run=True)

    state_path = tmp_path / "live_state.json"
    assert state_path.exists()
    payload = json.loads(state_path.read_text())
    assert payload["positions"] == {}
    assert "DIPUSD" in payload["pending_entries"]

    events_path = tmp_path / "events.jsonl"
    assert events_path.exists()
    events = [json.loads(line) for line in events_path.read_text().strip().split("\n")]
    entry_events = [e for e in events if e.get("type") == "entry_scan"]
    assert len(entry_events) >= 1
    assert entry_events[0]["n_checked"] >= 1


def test_run_daily_cycle_does_not_stage_failed_live_entry_orders(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")

    bars = make_bars([100.0] * 29 + [90.0], "DIPUSD")
    monkeypatch.setattr("binance_worksteal.trade_live.fetch_daily_bars", lambda client, symbol, lookback_days=30: bars)
    monkeypatch.setattr("binance_worksteal.trade_live.place_limit_buy", lambda *args, **kwargs: None)
    monkeypatch.setattr("binance_worksteal.trade_live.get_account_equity", lambda client: 10_000.0)

    config = build_runtime_config(
        build_arg_parser().parse_args(
            [
                "--dip-pct",
                "0.10",
                "--proximity-pct",
                "0.02",
                "--lookback-days",
                "20",
                "--profit-target",
                "0.03",
                "--stop-loss",
                "0.08",
                "--sma-filter",
                "0",
                "--risk-off-trigger-sma-period",
                "0",
                "--risk-off-trigger-momentum-period",
                "0",
                "--risk-off-market-breadth-filter",
                "0",
            ]
        )
    )

    run_daily_cycle(client=None, symbols=["DIPUSD"], config=config, dry_run=False)

    payload = json.loads((tmp_path / "live_state.json").read_text())
    assert payload["positions"] == {}
    assert payload["pending_entries"] == {}

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().strip().split("\n")]
    assert any(event.get("type") == "entry_order_failed" and event.get("symbol") == "DIPUSD" for event in events)
    entry_scan = [event for event in events if event.get("type") == "entry_scan"][-1]
    assert entry_scan["n_staged"] == 0
    assert entry_scan["n_order_fail"] == 1

    log_path = tmp_path / "trade_log.jsonl"
    assert not log_path.exists() or not log_path.read_text().strip()


def test_default_entry_proximity_bps_is_3000():
    assert DEFAULT_CONFIG.entry_proximity_bps == 3000.0


def test_relative_bps_distance_basic():
    assert _relative_bps_distance(100.0, 100.0) == 0.0
    assert abs(_relative_bps_distance(100.0, 80.0) - 2000.0) < 0.01
    assert _relative_bps_distance(0.0, 100.0) == float("inf")
    assert abs(_relative_bps_distance(100.0, 95.0) - 500.0) < 0.01


def test_entry_proximity_bps_3000_allows_dip_entries(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")

    # close=90, ref_high=100, dip=10% -> buy_target=90, dist=0.0 <= proximity 0.02
    # buy_price=90, close=90 -> 0 bps distance, well within 3000
    bars = make_bars([100.0] * 29 + [90.0], "DIPUSD")
    monkeypatch.setattr("binance_worksteal.trade_live.fetch_daily_bars", lambda client, symbol, lookback_days=30: bars)

    config = build_runtime_config(
        build_arg_parser().parse_args(
            [
                "--dip-pct", "0.10",
                "--proximity-pct", "0.02",
                "--lookback-days", "20",
                "--profit-target", "0.15",
                "--stop-loss", "0.10",
                "--sma-filter", "0",
                "--entry-proximity-bps", "3000",
                "--risk-off-trigger-sma-period", "0",
                "--risk-off-trigger-momentum-period", "0",
                "--risk-off-market-breadth-filter", "0",
            ]
        )
    )
    assert config.entry_proximity_bps == 3000.0

    run_daily_cycle(client=None, symbols=["DIPUSD"], config=config, dry_run=True)

    payload = json.loads((tmp_path / "live_state.json").read_text())
    assert "DIPUSD" in payload["pending_entries"]


def test_entry_proximity_bps_25_blocks_dip_entries(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")

    # last bar: close=91, low=88. ref_high=100, dip=10% -> buy_target=90
    # dist_long = (91-90)/100 = 0.01 <= proximity_pct 0.02 -> candidate generated
    # fill_price = max(buy_target=90, low_bar=88) = 90
    # buy_price=90, close=91 -> |91-90|/91*10000 = 109.9 bps > 25 bps -> BLOCKED
    ohlc = [{"close": 100.0}] * 29 + [{"open": 92.0, "high": 92.0, "low": 88.0, "close": 91.0}]
    bars = make_bars_ohlc(ohlc, "DIPUSD")
    monkeypatch.setattr("binance_worksteal.trade_live.fetch_daily_bars", lambda client, symbol, lookback_days=30: bars)

    config = build_runtime_config(
        build_arg_parser().parse_args(
            [
                "--dip-pct", "0.10",
                "--proximity-pct", "0.02",
                "--lookback-days", "20",
                "--profit-target", "0.15",
                "--stop-loss", "0.10",
                "--sma-filter", "0",
                "--entry-proximity-bps", "25",
                "--risk-off-trigger-sma-period", "0",
                "--risk-off-trigger-momentum-period", "0",
                "--risk-off-market-breadth-filter", "0",
            ]
        )
    )
    assert config.entry_proximity_bps == 25.0

    run_daily_cycle(client=None, symbols=["DIPUSD"], config=config, dry_run=True)

    payload = json.loads((tmp_path / "live_state.json").read_text())
    assert payload["pending_entries"] == {}
    events = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().strip().split("\n")]
    scan = [e for e in events if e.get("type") == "entry_scan"][0]
    assert scan["n_proximity_skip"] >= 1


def test_log_event_writes_jsonl(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")
    log_event({"type": "test", "value": 42})
    log_event({"type": "test", "value": 99})
    lines = (tmp_path / "events.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["value"] == 42
    assert json.loads(lines[1])["value"] == 99
    assert "ts" in json.loads(lines[0])


def test_parser_entry_poll_and_health_report_args():
    parser = build_arg_parser()
    args = parser.parse_args(["--entry-poll-hours", "2", "--health-report-hours", "12"])
    assert args.entry_poll_hours == 2
    assert args.health_report_hours == 12


def test_parser_default_entry_poll_and_health_report():
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.entry_poll_hours == 4
    assert args.health_report_hours == 6


def test_run_daily_cycle_logs_entry_summary_event(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")

    bars = make_bars([100.0] * 30, "BTCUSD")
    monkeypatch.setattr("binance_worksteal.trade_live.fetch_daily_bars", lambda client, symbol, lookback_days=30: bars)

    config = build_runtime_config(
        build_arg_parser().parse_args(
            [
                "--dip-pct", "0.10",
                "--lookback-days", "20",
                "--sma-filter", "0",
                "--risk-off-trigger-sma-period", "0",
                "--risk-off-trigger-momentum-period", "0",
                "--risk-off-market-breadth-filter", "0",
            ]
        )
    )

    run_daily_cycle(client=None, symbols=["BTCUSD"], config=config, dry_run=True)

    events_path = tmp_path / "events.jsonl"
    assert events_path.exists()
    events = [json.loads(line) for line in events_path.read_text().strip().split("\n")]
    entry_events = [e for e in events if e.get("type") == "entry_scan"]
    assert len(entry_events) == 1
    evt = entry_events[0]
    assert "n_checked" in evt
    assert "n_candidates" in evt
    assert "n_staged" in evt
    assert "n_proximity_skip" in evt
    assert "risk_off" in evt


# ---- Neural integration tests ----

def _make_dummy_checkpoint(tmp_path, n_symbols=3, seq_len=10, hidden_dim=32):
    symbols = [f"SYM{i}USDT" for i in range(n_symbols)]
    model = PerSymbolWorkStealPolicy(
        n_features=len(FEATURE_NAMES), n_symbols=n_symbols,
        hidden_dim=hidden_dim, num_temporal_layers=1, num_cross_layers=1,
        num_heads=2, seq_len=seq_len, dropout=0.0,
    )
    ckpt_path = tmp_path / "test_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "epoch": 1,
        "metrics": {"sortino": 1.0},
        "config": {
            "n_features": len(FEATURE_NAMES),
            "n_symbols": n_symbols,
            "hidden_dim": hidden_dim,
            "num_layers": 2,
            "num_heads": 2,
            "seq_len": seq_len,
            "dropout": 0.0,
            "symbols": symbols,
            "model_type": "persymbol",
        },
    }, ckpt_path)
    return ckpt_path, symbols


def test_load_neural_model(tmp_path):
    ckpt_path, symbols = _make_dummy_checkpoint(tmp_path)
    model, loaded_symbols, cfg = load_neural_model(str(ckpt_path))
    assert isinstance(model, PerSymbolWorkStealPolicy)
    assert loaded_symbols == symbols
    assert cfg["seq_len"] == 10
    assert not model.training


def test_load_neural_model_flat(tmp_path):
    from binance_worksteal.model import DailyWorkStealPolicy
    n_symbols = 3
    symbols = [f"SYM{i}USDT" for i in range(n_symbols)]
    model = DailyWorkStealPolicy(
        n_features=len(FEATURE_NAMES), n_symbols=n_symbols,
        hidden_dim=32, num_layers=2, num_heads=2, seq_len=10,
    )
    ckpt_path = tmp_path / "flat_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "epoch": 1,
        "metrics": {},
        "config": {
            "n_features": len(FEATURE_NAMES),
            "n_symbols": n_symbols,
            "hidden_dim": 32,
            "num_layers": 2,
            "num_heads": 2,
            "seq_len": 10,
            "dropout": 0.0,
            "symbols": symbols,
            "model_type": "flat",
        },
    }, ckpt_path)
    loaded_model, loaded_symbols, cfg = load_neural_model(str(ckpt_path))
    assert isinstance(loaded_model, DailyWorkStealPolicy)
    assert loaded_symbols == symbols


def _make_fake_bars_for_symbols(model_symbols, n_days=40, seed=None):
    rng = np.random.RandomState(seed)
    all_bars = {}
    for sym in model_symbols:
        strategy_sym = f"{sym[:-4]}USD"
        dates = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
        prices = 100.0 + rng.randn(n_days).cumsum()
        prices = np.maximum(prices, 1.0)
        all_bars[strategy_sym] = pd.DataFrame({
            "timestamp": dates,
            "open": prices + 0.5,
            "high": prices + 2.0,
            "low": prices - 2.0,
            "close": prices,
            "volume": rng.rand(n_days) * 1e6,
            "symbol": strategy_sym,
        })
    return all_bars


def test_prepare_neural_features():
    symbols = ["SYM0USDT", "SYM1USDT"]
    all_bars = _make_fake_bars_for_symbols(symbols, n_days=40, seed=42)

    tensor = prepare_neural_features(all_bars, symbols, seq_len=10)
    assert tensor is not None
    assert tensor.shape == (1, 10, 2, len(FEATURE_NAMES))
    assert not torch.isnan(tensor).any()


def test_prepare_neural_features_insufficient_data():
    all_bars = {
        "SYM0USD": pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
            "open": [100] * 5, "high": [101] * 5, "low": [99] * 5,
            "close": [100] * 5, "volume": [1000] * 5, "symbol": "SYM0USD",
        })
    }
    result = prepare_neural_features(all_bars, ["SYM0USDT"], seq_len=30)
    assert result is None


def test_run_neural_inference(tmp_path):
    n_symbols = 2
    seq_len = 10
    ckpt_path, symbols = _make_dummy_checkpoint(tmp_path, n_symbols=n_symbols, seq_len=seq_len)
    model, model_symbols, cfg = load_neural_model(str(ckpt_path))

    all_bars = _make_fake_bars_for_symbols(model_symbols, n_days=40, seed=42)

    predictions = run_neural_inference(model, all_bars, model_symbols, seq_len)
    assert predictions is not None
    for sym in model_symbols:
        strategy_sym = f"{sym[:-4]}USD"
        assert strategy_sym in predictions
        pred = predictions[strategy_sym]
        assert 0.0 <= pred["buy_offset"] <= 0.30
        assert 0.0 <= pred["sell_offset"] <= 0.30
        assert 0.0 <= pred["intensity"] <= 1.0


def test_run_neural_inference_no_data():
    model = PerSymbolWorkStealPolicy(
        n_features=len(FEATURE_NAMES), n_symbols=2,
        hidden_dim=32, num_temporal_layers=1, num_cross_layers=1,
        num_heads=2, seq_len=10,
    )
    model.eval()
    predictions = run_neural_inference(model, {}, ["SYM0USDT", "SYM1USDT"], 10)
    assert predictions is None


def test_stage_entry_candidates_with_neural_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")

    bars = make_bars([100.0] * 29 + [90.0], "DIPUSD")
    all_bars = {"DIPUSD": bars}
    current_bar = bars.iloc[-1]

    neural_predictions = {
        "DIPUSD": {
            "buy_offset": 0.05,
            "sell_offset": 0.10,
            "intensity": 0.7,
        }
    }

    config = build_runtime_config(
        build_arg_parser().parse_args([
            "--dip-pct", "0.10", "--lookback-days", "20",
            "--sma-filter", "0", "--entry-proximity-bps", "5000",
            "--risk-off-trigger-sma-period", "0",
            "--risk-off-trigger-momentum-period", "0",
            "--risk-off-market-breadth-filter", "0",
        ])
    )

    candidates = [("DIPUSD", "long", 0.5, 90.0, current_bar)]
    pending_entries = {}
    recent_trades = []
    now = datetime(2026, 3, 18, tzinfo=timezone.utc)

    counts = _stage_entry_candidates(
        client=None, candidates=candidates, all_bars=all_bars,
        staged_symbols=set(), pending_entries=pending_entries,
        recent_trades=recent_trades, entry_config=config, config=config,
        equity=10000.0, now=now, dry_run=True, slots=5,
        neural_predictions=neural_predictions,
    )

    assert counts["n_staged"] == 1
    assert "DIPUSD" in pending_entries
    entry = pending_entries["DIPUSD"]
    assert "neural" in entry["source"]
    assert entry["neural_override"]["buy_offset"] == 0.05
    assert entry["neural_override"]["sell_offset"] == 0.10
    assert entry["neural_override"]["intensity"] == 0.7
    expected_buy = 90.0 * (1.0 - 0.05)
    assert abs(entry["buy_price"] - expected_buy) < 0.01

    trades = [json.loads(l) for l in (tmp_path / "trade_log.jsonl").read_text().strip().split("\n")]
    assert trades[0].get("neural_override") is not None


def test_stage_entry_candidates_neural_skip_low_intensity(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")

    bars = make_bars([100.0] * 29 + [90.0], "DIPUSD")
    current_bar = bars.iloc[-1]

    neural_predictions = {
        "DIPUSD": {"buy_offset": 0.05, "sell_offset": 0.10, "intensity": 0.05},
    }

    config = build_runtime_config(build_arg_parser().parse_args([
        "--dip-pct", "0.10", "--sma-filter", "0", "--entry-proximity-bps", "5000",
        "--risk-off-trigger-sma-period", "0", "--risk-off-trigger-momentum-period", "0",
        "--risk-off-market-breadth-filter", "0",
    ]))

    counts = _stage_entry_candidates(
        client=None, candidates=[("DIPUSD", "long", 0.5, 90.0, current_bar)],
        all_bars={"DIPUSD": bars}, staged_symbols=set(), pending_entries={},
        recent_trades=[], entry_config=config, config=config,
        equity=10000.0, now=datetime(2026, 3, 18, tzinfo=timezone.utc),
        dry_run=True, slots=5, neural_predictions=neural_predictions,
    )

    assert counts["n_neural_skip"] == 1
    assert counts["n_staged"] == 0


def test_stage_entry_candidates_no_neural_falls_back_to_rules(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")

    bars = make_bars([100.0] * 29 + [90.0], "DIPUSD")
    current_bar = bars.iloc[-1]

    config = build_runtime_config(build_arg_parser().parse_args([
        "--dip-pct", "0.10", "--sma-filter", "0", "--entry-proximity-bps", "5000",
        "--risk-off-trigger-sma-period", "0", "--risk-off-trigger-momentum-period", "0",
        "--risk-off-market-breadth-filter", "0",
    ]))

    counts = _stage_entry_candidates(
        client=None, candidates=[("DIPUSD", "long", 0.5, 90.0, current_bar)],
        all_bars={"DIPUSD": bars}, staged_symbols=set(), pending_entries={},
        recent_trades=[], entry_config=config, config=config,
        equity=10000.0, now=datetime(2026, 3, 18, tzinfo=timezone.utc),
        dry_run=True, slots=5, neural_predictions=None,
    )

    assert counts["n_staged"] == 1
    assert counts["n_neural_skip"] == 0


def test_stage_entry_candidates_neural_high_intensity_boosts_confidence(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")

    bars = make_bars([100.0] * 29 + [90.0], "DIPUSD")
    current_bar = bars.iloc[-1]

    neural_predictions = {
        "DIPUSD": {"buy_offset": 0.05, "sell_offset": 0.10, "intensity": 0.9},
    }

    config = build_runtime_config(build_arg_parser().parse_args([
        "--dip-pct", "0.10", "--sma-filter", "0", "--entry-proximity-bps", "5000",
        "--risk-off-trigger-sma-period", "0", "--risk-off-trigger-momentum-period", "0",
        "--risk-off-market-breadth-filter", "0",
    ]))

    pending_entries = {}
    _stage_entry_candidates(
        client=None, candidates=[("DIPUSD", "long", 0.5, 90.0, current_bar)],
        all_bars={"DIPUSD": bars}, staged_symbols=set(), pending_entries=pending_entries,
        recent_trades=[], entry_config=config, config=config,
        equity=10000.0, now=datetime(2026, 3, 18, tzinfo=timezone.utc),
        dry_run=True, slots=5, neural_predictions=neural_predictions,
    )

    entry = pending_entries["DIPUSD"]
    assert entry["confidence"] > 1.0


def test_run_daily_cycle_with_neural_model(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")

    n_symbols = 1
    symbols = ["DIPUSDT"]
    model = PerSymbolWorkStealPolicy(
        n_features=len(FEATURE_NAMES), n_symbols=n_symbols,
        hidden_dim=32, num_temporal_layers=1, num_cross_layers=1,
        num_heads=2, seq_len=10, dropout=0.0,
    )
    model.eval()

    bars = make_bars([100.0] * 29 + [90.0], "DIPUSD")
    monkeypatch.setattr("binance_worksteal.trade_live.fetch_daily_bars",
                        lambda client, symbol, lookback_days=30: bars)

    config = build_runtime_config(build_arg_parser().parse_args([
        "--dip-pct", "0.10", "--lookback-days", "20", "--sma-filter", "0",
        "--risk-off-trigger-sma-period", "0", "--risk-off-trigger-momentum-period", "0",
        "--risk-off-market-breadth-filter", "0",
    ]))

    run_daily_cycle(
        client=None, symbols=["DIPUSD"], config=config, dry_run=True,
        neural_model=model, neural_model_symbols=symbols, neural_seq_len=10,
    )

    events_path = tmp_path / "events.jsonl"
    assert events_path.exists()
    events = [json.loads(line) for line in events_path.read_text().strip().split("\n")]
    neural_events = [e for e in events if e.get("type") == "neural_inference"]
    assert len(neural_events) >= 1
    assert "predictions" in neural_events[0]


def test_run_daily_cycle_neural_fallback_on_error(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")

    class BrokenModel:
        def __call__(self, x):
            raise RuntimeError("model error")

    bars = make_bars([100.0] * 29 + [90.0], "DIPUSD")
    monkeypatch.setattr("binance_worksteal.trade_live.fetch_daily_bars",
                        lambda client, symbol, lookback_days=30: bars)

    config = build_runtime_config(build_arg_parser().parse_args([
        "--dip-pct", "0.10", "--lookback-days", "20", "--sma-filter", "0",
        "--risk-off-trigger-sma-period", "0", "--risk-off-trigger-momentum-period", "0",
        "--risk-off-market-breadth-filter", "0",
    ]))

    run_daily_cycle(
        client=None, symbols=["DIPUSD"], config=config, dry_run=True,
        neural_model=BrokenModel(), neural_model_symbols=["DIPUSDT"],
        neural_seq_len=10,
    )

    state = json.loads((tmp_path / "live_state.json").read_text())
    assert "DIPUSD" in state["pending_entries"]


def test_parser_neural_model_args():
    parser = build_arg_parser()
    args = parser.parse_args(["--neural-model", "/tmp/model.pt", "--neural-symbols", "BTCUSDT", "ETHUSDT"])
    assert args.neural_model == "/tmp/model.pt"
    assert args.neural_symbols == ["BTCUSDT", "ETHUSDT"]


def test_parser_neural_model_args_defaults():
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.neural_model is None
    assert args.neural_symbols is None
