"""Tests for live work-steal runtime config wiring."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binance_worksteal.trade_live import (
    DEFAULT_CONFIG,
    EVENTS_FILE,
    _relative_bps_distance,
    build_arg_parser,
    build_runtime_config,
    log_event,
    normalize_live_positions,
    plan_legacy_rebalance_exits,
    reconcile_pending_entries,
    run_daily_cycle,
    run_entry_scan,
    run_health_report,
)


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
