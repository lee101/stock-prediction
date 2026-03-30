"""Tests for live work-steal runtime config wiring."""
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binance_worksteal.trade_live import (
    DEFAULT_CONFIG,
    _coerce_order_id,
    _merge_peak_equity,
    _order_avg_price,
    _relative_bps_distance,
    _resolve_symbol_selection_context,
    _stage_entry_candidates,
    _submitted_margin_order_or_none,
    build_arg_parser,
    build_runtime_config,
    fetch_daily_bars,
    get_account_equity,
    load_state,
    load_neural_model,
    log_event,
    log_trade,
    main as trade_live_main,
    normalize_live_positions,
    _normalize_pending_entries,
    place_limit_buy,
    place_limit_sell,
    plan_legacy_rebalance_exits,
    prepare_neural_features,
    reconcile_exit_orders,
    reconcile_pending_entries,
    run_daily_cycle,
    run_entry_scan,
    run_health_report,
    run_neural_inference,
    save_state,
    synchronize_positions_from_exchange,
)
from binance_worksteal.data import FEATURE_NAMES
from binance_worksteal.model import DailyWorkStealPolicy, PerSymbolWorkStealPolicy
from binance_worksteal.strategy import WorkStealConfig


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


def test_build_runtime_config_uses_config_file_and_cli_overrides(tmp_path):
    config_path = tmp_path / "live_config.yaml"
    config_path.write_text(
        "config:\n"
        "  dip_pct: 0.18\n"
        "  market_breadth_filter: 0.45\n"
        "  dip_pct_fallback:\n"
        "    - 0.18\n"
        "    - 0.15\n",
        encoding="utf-8",
    )
    parser = build_arg_parser()
    argv = ["--config-file", str(config_path), "--dip-pct", "0.25"]
    args = parser.parse_args(argv)

    config = build_runtime_config(args, argv)

    assert config.dip_pct == pytest.approx(0.25)
    assert config.market_breadth_filter == pytest.approx(0.45)
    assert config.dip_pct_fallback == (0.18, 0.15)


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


def test_normalize_live_positions_repairs_invalid_targets_and_stops():
    positions = normalize_live_positions(
        {
            "adausd": {
                "entry_price": "0.2726",
                "entry_date": "2026-03-25T10:12:00+00:00",
                "quantity": "837.4",
                "target_sell": "0.265",
                "stop_price": "0.400",
            }
        },
        DEFAULT_CONFIG,
    )

    position = positions["ADAUSD"]
    assert position["target_sell"] == pytest.approx(0.2726 * (1.0 + DEFAULT_CONFIG.profit_target_pct))
    assert position["stop_price"] == pytest.approx(0.2726 * (1.0 - DEFAULT_CONFIG.stop_loss_pct))
    assert position["peak_price"] == pytest.approx(0.2726)


def test_normalize_live_positions_recovers_from_invalid_nested_fields():
    positions = normalize_live_positions(
        {
            "ethusd": {
                "entry_price": "2000",
                "entry_date": "not-a-date",
                "quantity": "0.5",
                "peak_price": "oops",
                "target_sell": "oops",
                "stop_price": "oops",
                "exit_order_id": 123,
                "exit_price": "oops",
            }
        },
        DEFAULT_CONFIG,
    )

    position = positions["ETHUSD"]
    assert pd.Timestamp(position["entry_date"]).tzinfo is not None
    assert position["peak_price"] == pytest.approx(2000.0)
    assert position["target_sell"] == pytest.approx(2000.0 * (1.0 + DEFAULT_CONFIG.profit_target_pct))
    assert position["stop_price"] == pytest.approx(2000.0 * (1.0 - DEFAULT_CONFIG.stop_loss_pct))
    assert position["exit_price"] == pytest.approx(position["target_sell"])


def test_normalize_live_positions_and_pending_entries_coerce_valid_order_ids_and_drop_invalid_ones():
    positions = normalize_live_positions(
        {
            "ethusd": {
                "entry_price": "2000",
                "entry_date": "2026-03-17T00:00:00+00:00",
                "quantity": "0.5",
                "exit_order_id": "123",
            },
            "solusd": {
                "entry_price": "150",
                "entry_date": "2026-03-17T00:00:00+00:00",
                "quantity": "2",
                "exit_order_id": "bad-id",
            },
        },
        DEFAULT_CONFIG,
    )

    pending = _normalize_pending_entries(
        {
            "btcusd": {
                "buy_price": "100000",
                "quantity": "0.01",
                "target_sell": "103000",
                "stop_price": "97000",
                "placed_at": "2026-03-17T00:00:00+00:00",
                "order_id": "456",
            },
            "adausd": {
                "buy_price": "0.28",
                "quantity": "10",
                "target_sell": "0.30",
                "stop_price": "0.25",
                "placed_at": "2026-03-17T00:00:00+00:00",
                "order_id": "",
            },
        }
    )

    assert positions["ETHUSD"]["exit_order_id"] == 123
    assert "exit_order_id" not in positions["SOLUSD"]
    assert pending["BTCUSD"]["order_id"] == 456
    assert pending["ADAUSD"]["order_id"] is None


def test_order_id_helpers_only_accept_positive_integer_ids():
    assert _coerce_order_id("123") == 123
    assert _coerce_order_id(456.0) == 456
    assert _coerce_order_id("0") is None
    assert _coerce_order_id(12.5) is None
    assert _coerce_order_id(True) is None
    assert _coerce_order_id("bad-id") is None

    order, order_id = _submitted_margin_order_or_none({"orderId": "789"}, context="test order")
    assert order == {"orderId": "789"}
    assert order_id == 789

    order, order_id = _submitted_margin_order_or_none({"orderId": ""}, context="test order")
    assert order == {"orderId": ""}
    assert order_id is None


def test_normalize_live_positions_drops_non_finite_core_values_and_repairs_nested_non_finite_values():
    positions = normalize_live_positions(
        {
            "ethusd": {
                "entry_price": "nan",
                "quantity": "0.5",
            },
            "btcusd": {
                "entry_price": "2000",
                "quantity": "nan",
            },
            "solusd": {
                "entry_price": "2000",
                "entry_date": "2026-03-17T00:00:00+00:00",
                "quantity": "1.25",
                "peak_price": "nan",
                "target_sell": "nan",
                "stop_price": "nan",
                "exit_order_id": 123,
                "exit_price": "nan",
            },
        },
        DEFAULT_CONFIG,
    )

    assert list(positions) == ["SOLUSD"]
    position = positions["SOLUSD"]
    assert position["entry_price"] == pytest.approx(2000.0)
    assert position["quantity"] == pytest.approx(1.25)
    assert position["peak_price"] == pytest.approx(2000.0)
    assert position["target_sell"] == pytest.approx(2000.0 * (1.0 + DEFAULT_CONFIG.profit_target_pct))
    assert position["stop_price"] == pytest.approx(2000.0 * (1.0 - DEFAULT_CONFIG.stop_loss_pct))
    assert position["exit_price"] == pytest.approx(position["target_sell"])


def test_normalize_pending_entries_replaces_non_finite_numeric_fields():
    entries = _normalize_pending_entries(
        {
            "ethusd": {
                "buy_price": "nan",
                "quantity": "nan",
                "target_sell": "nan",
                "stop_price": "nan",
                "confidence": "nan",
                "placed_at": "2026-03-17T00:00:00+00:00",
            }
        }
    )

    assert entries["ETHUSD"]["buy_price"] == 0.0
    assert entries["ETHUSD"]["quantity"] == 0.0
    assert entries["ETHUSD"]["target_sell"] == 0.0
    assert entries["ETHUSD"]["stop_price"] == 0.0
    assert entries["ETHUSD"]["confidence"] == 1.0


def test_order_avg_price_prefers_executed_average_over_avg_and_limit_price():
    assert _order_avg_price(
        {
            "executedQty": "0.5",
            "cummulativeQuoteQty": "995",
            "avgPrice": "1998",
            "price": "2000",
        }
    ) == pytest.approx(1990.0)


def test_order_avg_price_falls_back_to_avg_price_before_limit_price():
    assert _order_avg_price(
        {
            "executedQty": "0",
            "cummulativeQuoteQty": "0",
            "avgPrice": "1998",
            "price": "2000",
        }
    ) == pytest.approx(1998.0)


def test_order_avg_price_uses_fallback_when_execution_fields_are_invalid():
    assert _order_avg_price(
        {
            "executedQty": "nan",
            "cummulativeQuoteQty": "nan",
            "avgPrice": "nan",
            "price": "nan",
        },
        fallback=1987.5,
    ) == pytest.approx(1987.5)


def test_place_limit_buy_quantizes_price_and_quantity_to_exchange_filters(monkeypatch):
    monkeypatch.setattr("binance_worksteal.trade_live._ORDER_RULES_CACHE", {})
    captured = {}

    class DummyClient:
        def get_symbol_info(self, symbol):
            assert symbol == "ENJUSDT"
            return {
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.00001", "minPrice": "0.00001"},
                    {"filterType": "LOT_SIZE", "stepSize": "1", "minQty": "1"},
                    {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                ]
            }

        def create_margin_order(self, **kwargs):
            captured.update(kwargs)
            return {"symbol": kwargs["symbol"], "orderId": 123, "status": "NEW"}

    order = place_limit_buy(
        DummyClient(),
        "ENJUSD",
        price=0.025830000000000002,
        quantity=28530.521935401997,
        config=DEFAULT_CONFIG,
    )

    assert order is not None
    assert captured["symbol"] == "ENJUSDT"
    assert captured["price"] == "0.02583"
    assert captured["quantity"] == "28530"


def test_place_limit_buy_rejects_order_below_exchange_min_notional(monkeypatch):
    monkeypatch.setattr("binance_worksteal.trade_live._ORDER_RULES_CACHE", {})
    class DummyClient:
        def __init__(self):
            self.called = False

        def get_symbol_info(self, symbol):
            return {
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01", "minPrice": "0.01"},
                    {"filterType": "LOT_SIZE", "stepSize": "1", "minQty": "1"},
                    {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                ]
            }

        def create_margin_order(self, **kwargs):
            self.called = True
            return {"symbol": kwargs["symbol"], "orderId": 123, "status": "NEW"}

    client = DummyClient()
    order = place_limit_buy(
        client,
        "ENAUSD",
        price=0.50,
        quantity=1.0,
        config=DEFAULT_CONFIG,
    )

    assert order is None
    assert client.called is False


def test_place_limit_sell_rounds_price_up_for_exchange_tick(monkeypatch):
    monkeypatch.setattr("binance_worksteal.trade_live._ORDER_RULES_CACHE", {})
    captured = {}

    class DummyClient:
        def get_symbol_info(self, symbol):
            assert symbol == "ADAUSDT"
            return {
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.001", "minPrice": "0.001"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.1", "minQty": "0.1"},
                    {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                ]
            }

        def create_margin_order(self, **kwargs):
            captured.update(kwargs)
            return {"symbol": kwargs["symbol"], "orderId": 456, "status": "NEW"}

    order = place_limit_sell(DummyClient(), "ADAUSD", price=0.27804, quantity=837.47)

    assert order is not None
    assert captured["price"] == "0.279"
    assert captured["quantity"] == "837.4"


def test_place_limit_buy_uses_cached_exchange_filters_after_first_lookup(monkeypatch):
    monkeypatch.setattr("binance_worksteal.trade_live._ORDER_RULES_CACHE", {})
    captured = []

    class DummyClient:
        def __init__(self):
            self.calls = 0
            self.fail_lookup = False

        def get_symbol_info(self, symbol):
            self.calls += 1
            if self.fail_lookup:
                raise RuntimeError("metadata unavailable")
            assert symbol == "ENAUSDT"
            return {
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01", "minPrice": "0.01"},
                    {"filterType": "LOT_SIZE", "stepSize": "1", "minQty": "1"},
                    {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                ]
            }

        def create_margin_order(self, **kwargs):
            captured.append(kwargs)
            return {"symbol": kwargs["symbol"], "orderId": len(captured), "status": "NEW"}

    client = DummyClient()
    first = place_limit_buy(client, "ENAUSD", price=0.534, quantity=12.8, config=DEFAULT_CONFIG)
    client.fail_lookup = True
    second = place_limit_buy(client, "ENAUSD", price=0.544, quantity=13.1, config=DEFAULT_CONFIG)

    assert first is not None
    assert second is not None
    assert client.calls == 1
    assert [order["price"] for order in captured] == ["0.53", "0.54"]
    assert [order["quantity"] for order in captured] == ["12", "13"]


def test_place_limit_buy_rejects_when_symbol_rules_unavailable(monkeypatch):
    monkeypatch.setattr("binance_worksteal.trade_live._ORDER_RULES_CACHE", {})

    class DummyClient:
        def __init__(self):
            self.called = False

        def get_symbol_info(self, symbol):
            raise RuntimeError("metadata unavailable")

        def create_margin_order(self, **kwargs):
            self.called = True
            return {"symbol": kwargs["symbol"], "orderId": 123, "status": "NEW"}

    client = DummyClient()
    order = place_limit_buy(
        client,
        "ENAUSD",
        price=0.50,
        quantity=10.0,
        config=DEFAULT_CONFIG,
    )

    assert order is None
    assert client.called is False


def test_place_limit_sell_rejects_when_symbol_rules_unavailable(monkeypatch):
    monkeypatch.setattr("binance_worksteal.trade_live._ORDER_RULES_CACHE", {})

    class DummyClient:
        def __init__(self):
            self.called = False

        def get_symbol_info(self, symbol):
            raise RuntimeError("metadata unavailable")

        def create_margin_order(self, **kwargs):
            self.called = True
            return {"symbol": kwargs["symbol"], "orderId": 456, "status": "NEW"}

    client = DummyClient()
    order = place_limit_sell(client, "ENAUSD", price=0.50, quantity=10.0)

    assert order is None
    assert client.called is False


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
        now=datetime(2026, 3, 18, tzinfo=UTC),
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


def test_plan_legacy_rebalance_exits_skips_rebalance_in_risk_off_regime():
    history = {
        "ALTUSD": make_bars([100.0] * 20 + [90.0] * 7, "ALTUSD"),
        "DIPUSD": make_bars([100.0] * 20 + [90.0] * 7, "DIPUSD"),
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
    }
    config = build_runtime_config(
        build_arg_parser().parse_args(
            [
                "--dip-pct", "0.10",
                "--proximity-pct", "0.02",
                "--lookback-days", "20",
                "--profit-target", "0.03",
                "--stop-loss", "0.08",
                "--rebalance-seeded-positions",
                "--market-breadth-filter", "0.0",
                "--risk-off-trigger-sma-period", "0",
                "--risk-off-trigger-momentum-period", "7",
            ]
        )
    )

    exits, rebalance_symbols = plan_legacy_rebalance_exits(
        now=datetime(2026, 3, 18, tzinfo=UTC),
        positions=positions,
        current_bars=current_bars,
        history=history,
        last_exit={},
        config=config,
    )

    assert exits == []
    assert rebalance_symbols == set()
    assert positions["ALTUSD"]["source"] == "legacy"
    assert positions["DIPUSD"]["source"] == "legacy"


def test_build_runtime_config_can_disable_seeded_rebalance():
    config = build_runtime_config(build_arg_parser().parse_args(["--no-rebalance-seeded-positions"]))
    assert config.rebalance_seeded_positions is False


def test_parser_can_disable_startup_preview():
    args = build_arg_parser().parse_args(["--no-run-on-start", "--startup-live-cycle"])
    assert args.run_on_start is False
    assert args.startup_preview_only is False


def test_fetch_daily_bars_skips_malformed_kline_rows(monkeypatch):
    warnings = []

    class DummyLogger:
        def warning(self, message):
            warnings.append(message)

        def error(self, message):
            raise AssertionError(f"unexpected error log: {message}")

    class DummyClient:
        def get_klines(self, **kwargs):
            return [
                [int(pd.Timestamp("2026-01-01T00:00:00Z").timestamp() * 1000), "100", "101", "99", "100", "1000"],
                ["bad-ts", "101", "102", "100", "101", "1000"],
                [int(pd.Timestamp("2026-01-03T00:00:00Z").timestamp() * 1000), "oops", "103", "101", "102", "1000"],
                [int(pd.Timestamp("2026-01-04T00:00:00Z").timestamp() * 1000), "103", "104", "102", "103", "1000"],
                {"bad": "row"},
            ]

    monkeypatch.setattr("binance_worksteal.trade_live.logger", DummyLogger())

    bars = fetch_daily_bars(DummyClient(), "BTCUSD", lookback_days=3)

    assert list(bars["timestamp"]) == [
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-04T00:00:00Z"),
    ]
    assert list(bars["close"]) == [100.0, 103.0]
    assert len(warnings) == 3
    assert all("Skipping malformed kline row" in message for message in warnings)


def test_fetch_daily_bars_rejects_invalid_response_shape(monkeypatch):
    warnings = []

    class DummyLogger:
        def warning(self, message):
            warnings.append(message)

        def error(self, message):
            raise AssertionError(f"unexpected error log: {message}")

    class DummyClient:
        def get_klines(self, **kwargs):
            return {"code": -1, "msg": "bad response"}

    monkeypatch.setattr("binance_worksteal.trade_live.logger", DummyLogger())

    bars = fetch_daily_bars(DummyClient(), "BTCUSD", lookback_days=3)

    assert bars.empty
    assert warnings == ["Invalid kline response for BTCUSDT: expected a list, got dict"]


def test_fetch_daily_bars_local_ignores_invalid_directory_response_shape(monkeypatch):
    calls = []

    def mock_load_daily_bars(data_dir, requested_symbols):
        calls.append((data_dir, list(requested_symbols)))
        if data_dir == "trainingdatadailybinance":
            return ["bad"]
        if data_dir == "trainingdata/train":
            return {"BTCUSD": make_bars([100.0, 101.0, 102.0, 103.0], "BTCUSD")}
        raise AssertionError(f"unexpected data_dir: {data_dir}")

    monkeypatch.setattr("binance_worksteal.trade_live.load_daily_bars", mock_load_daily_bars)

    bars = fetch_daily_bars(None, "BTCUSD", lookback_days=3)

    assert calls == [
        ("trainingdatadailybinance", ["BTCUSD"]),
        ("trainingdata/train", ["BTCUSD"]),
    ]
    assert list(bars["close"]) == [100.0, 101.0, 102.0, 103.0]



def test_fetch_daily_bars_local_ignores_invalid_symbol_payload_and_tries_next_directory(monkeypatch):
    calls = []

    def mock_load_daily_bars(data_dir, requested_symbols):
        calls.append((data_dir, list(requested_symbols)))
        if data_dir == "trainingdatadailybinance":
            return {"BTCUSD": {"close": 100.0}}
        if data_dir == "trainingdata/train":
            return {"BTCUSD": make_bars([100.0, 101.0, 102.0, 103.0, 104.0], "BTCUSD")}
        raise AssertionError(f"unexpected data_dir: {data_dir}")

    monkeypatch.setattr("binance_worksteal.trade_live.load_daily_bars", mock_load_daily_bars)

    bars = fetch_daily_bars(None, "BTCUSD", lookback_days=3)

    assert calls == [
        ("trainingdatadailybinance", ["BTCUSD"]),
        ("trainingdata/train", ["BTCUSD"]),
    ]
    assert list(bars["close"]) == [100.0, 101.0, 102.0, 103.0, 104.0]


def test_get_account_equity_rejects_invalid_response_shapes(monkeypatch):
    warnings = []
    errors = []

    class DummyLogger:
        def warning(self, message):
            warnings.append(message)

        def error(self, message):
            errors.append(message)

    class DummyClient:
        def get_margin_account(self):
            return []

        def get_symbol_ticker(self, symbol):
            raise AssertionError("ticker lookup should not run when account payload is invalid")

    monkeypatch.setattr("binance_worksteal.trade_live.logger", DummyLogger())

    equity = get_account_equity(DummyClient())

    assert equity == 0.0
    assert warnings == ["Invalid margin account response for equity: expected dict, got list"]
    assert errors == []


@pytest.mark.parametrize(
    ("account_payload", "ticker_payload", "expected_warning"),
    [
        (
            {"totalNetAssetOfBtc": "nan"},
            {"price": "65000"},
            "Invalid margin account totalNetAssetOfBtc: expected a finite numeric value, got 'nan'",
        ),
        (
            {"totalNetAssetOfBtc": "1.25"},
            {"price": "nan"},
            "Invalid BTCUSDT ticker price: expected a finite numeric value, got 'nan'",
        ),
        (
            {"totalNetAssetOfBtc": "1.25"},
            {"price": "0"},
            "Invalid BTCUSDT ticker price: expected a positive numeric value, got '0'",
        ),
    ],
)
def test_get_account_equity_rejects_invalid_numeric_values(
    monkeypatch, account_payload, ticker_payload, expected_warning
):
    warnings = []
    errors = []

    class DummyLogger:
        def warning(self, message):
            warnings.append(message)

        def error(self, message):
            errors.append(message)

    class DummyClient:
        def get_margin_account(self):
            return dict(account_payload)

        def get_symbol_ticker(self, symbol):
            assert symbol == "BTCUSDT"
            return dict(ticker_payload)

    monkeypatch.setattr("binance_worksteal.trade_live.logger", DummyLogger())

    equity = get_account_equity(DummyClient())

    assert equity == 0.0
    assert warnings == [expected_warning]
    assert errors == []


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
        now=datetime(2026, 3, 18, 12, tzinfo=UTC),
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


def test_reconcile_pending_entries_prefers_executed_average_price_over_limit_price(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")

    class DummyClient:
        def get_margin_order(self, symbol, orderId):
            assert symbol == "ETHFDUSD"
            assert orderId == 123
            return {
                "status": "FILLED",
                "executedQty": "0.5",
                "cummulativeQuoteQty": "995",
                "price": "2000",
            }

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
        now=datetime(2026, 3, 18, 12, tzinfo=UTC),
        dry_run=False,
    )

    assert pending_entries == {}
    assert positions["ETHUSD"]["entry_price"] == 1990.0
    assert positions["ETHUSD"]["quantity"] == 0.5
    assert recent == [
        {
            "timestamp": "2026-03-18T12:00:00+00:00",
            "symbol": "ETHUSD",
            "side": "buy",
            "price": 1990.0,
            "quantity": 0.5,
            "pnl": 0.0,
            "reason": "pending_fill(rule)",
        }
    ]


def test_reconcile_pending_entries_uses_stored_quantity_when_fill_qty_is_nan(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")

    class DummyClient:
        def get_margin_order(self, symbol, orderId):
            assert symbol == "ETHFDUSD"
            assert orderId == 123
            return {"status": "FILLED", "executedQty": "nan", "price": "2000"}

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
        now=datetime(2026, 3, 18, 12, tzinfo=UTC),
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


def test_reconcile_pending_entries_cancels_expired_orders_using_recorded_order_symbol():
    cancelled = []

    class DummyClient:
        def cancel_margin_order(self, symbol, orderId):
            cancelled.append((symbol, orderId))

    pending_entries = {
        "ADAUSD": {
            "buy_price": 0.27,
            "quantity": 100.0,
            "target_sell": 0.29,
            "stop_price": 0.25,
            "placed_at": "2026-03-18T00:00:00+00:00",
            "expires_at": "2026-03-18T12:00:00+00:00",
            "order_id": 123,
            "order_symbol": "ADAUSDT",
            "source": "rule",
        }
    }

    recent = reconcile_pending_entries(
        client=DummyClient(),
        pending_entries=pending_entries,
        positions={},
        now=datetime(2026, 3, 18, 13, tzinfo=UTC),
        dry_run=False,
    )

    assert recent == []
    assert pending_entries == {}
    assert cancelled == [("ADAUSDT", 123)]


def test_reconcile_pending_entries_invalid_expiry_cancels_entry():
    cancelled = []

    class DummyClient:
        def cancel_margin_order(self, symbol, orderId):
            cancelled.append((symbol, orderId))

    pending_entries = {
        "ADAUSD": {
            "buy_price": 0.27,
            "quantity": 100.0,
            "target_sell": 0.29,
            "stop_price": 0.25,
            "placed_at": "2026-03-18T00:00:00+00:00",
            "expires_at": "definitely-not-a-timestamp",
            "order_id": 123,
            "order_symbol": "ADAUSDT",
            "source": "rule",
        }
    }

    recent = reconcile_pending_entries(
        client=DummyClient(),
        pending_entries=pending_entries,
        positions={},
        now=datetime(2026, 3, 18, 13, tzinfo=UTC),
        dry_run=False,
    )

    assert recent == []
    assert pending_entries == {}
    assert cancelled == [("ADAUSDT", 123)]


def test_reconcile_pending_entries_keeps_expired_order_when_cancel_fails():
    class DummyClient:
        def cancel_margin_order(self, symbol, orderId):
            raise RuntimeError("network down")

    pending_entries = {
        "ADAUSD": {
            "buy_price": 0.27,
            "quantity": 100.0,
            "target_sell": 0.29,
            "stop_price": 0.25,
            "placed_at": "2026-03-18T00:00:00+00:00",
            "expires_at": "2026-03-18T12:00:00+00:00",
            "order_id": 123,
            "order_symbol": "ADAUSDT",
            "source": "rule",
        }
    }

    recent = reconcile_pending_entries(
        client=DummyClient(),
        pending_entries=pending_entries,
        positions={},
        now=datetime(2026, 3, 18, 13, tzinfo=UTC),
        dry_run=False,
    )

    assert recent == []
    assert "ADAUSD" in pending_entries
    assert pending_entries["ADAUSD"]["order_id"] == 123


def test_reconcile_pending_entries_keeps_entry_when_fill_details_cannot_be_confirmed():
    class DummyClient:
        def get_margin_order(self, symbol, orderId):
            assert symbol == "ETHFDUSD"
            assert orderId == 123
            return {"status": "FILLED", "executedQty": "0", "price": "0"}

    pending_entries = {
        "ETHUSD": {
            "buy_price": 0.0,
            "quantity": 0.0,
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
        now=datetime(2026, 3, 18, 12, tzinfo=UTC),
        dry_run=False,
    )

    assert recent == []
    assert positions == {}
    assert pending_entries["ETHUSD"]["status"] == "fill_unconfirmed"
    assert pending_entries["ETHUSD"]["order_id"] == 123


def test_reconcile_pending_entries_ignores_invalid_order_response_shape():
    class DummyClient:
        def get_margin_order(self, symbol, orderId):
            assert symbol == "ETHFDUSD"
            assert orderId == 123
            return []

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

    recent = reconcile_pending_entries(
        client=DummyClient(),
        pending_entries=pending_entries,
        positions={},
        now=datetime(2026, 3, 18, 12, tzinfo=UTC),
        dry_run=False,
    )

    assert recent == []
    assert pending_entries["ETHUSD"]["order_id"] == 123


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
        now=datetime(2026, 3, 25, 20, tzinfo=UTC),
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


def test_reconcile_exit_orders_prefers_executed_average_price_over_limit_price(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")

    class DummyClient:
        def get_margin_order(self, symbol, orderId):
            assert symbol == "ADAUSDT"
            assert orderId == 987
            return {
                "status": "FILLED",
                "executedQty": "837.4",
                "cummulativeQuoteQty": str(0.277 * 837.4),
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
        now=datetime(2026, 3, 25, 20, tzinfo=UTC),
    )

    assert positions == {}
    assert last_exit["ADAUSD"] == "2026-03-25T19:27:00+00:00"
    assert recent == [
        {
            "timestamp": "2026-03-25T19:27:00+00:00",
            "symbol": "ADAUSD",
            "side": "sell",
            "price": pytest.approx(0.277),
            "quantity": 837.4,
            "reason": "profit_target",
            "pnl": pytest.approx((0.277 - 0.2726) * 837.4),
            "dry_run": False,
        }
    ]


def test_reconcile_exit_orders_uses_stored_quantity_when_fill_qty_is_nan(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")

    class DummyClient:
        def get_margin_order(self, symbol, orderId):
            assert symbol == "ADAUSDT"
            assert orderId == 987
            return {
                "status": "FILLED",
                "executedQty": "nan",
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
        now=datetime(2026, 3, 25, 20, tzinfo=UTC),
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


def test_reconcile_exit_orders_keeps_position_when_fill_details_cannot_be_confirmed():
    class DummyClient:
        def get_margin_order(self, symbol, orderId):
            assert symbol == "ADAUSDT"
            assert orderId == 987
            return {
                "status": "FILLED",
                "executedQty": "0",
                "price": "0",
                "time": int(pd.Timestamp("2026-03-25T19:27:00Z").timestamp() * 1000),
            }

    positions = {
        "ADAUSD": {
            "entry_price": 0.0,
            "entry_date": "2026-03-25T10:12:00+00:00",
            "quantity": 0.0,
            "peak_price": 0.285,
            "target_sell": 0.0,
            "stop_price": 0.2453,
            "source": "exchange_sync",
            "exit_order_id": 987,
            "exit_order_symbol": "ADAUSDT",
            "exit_order_status": "NEW",
            "exit_price": 0.0,
            "exit_reason": "profit_target",
        }
    }
    last_exit = {}

    recent = reconcile_exit_orders(
        client=DummyClient(),
        positions=positions,
        last_exit=last_exit,
        now=datetime(2026, 3, 25, 20, tzinfo=UTC),
    )

    assert recent == []
    assert "ADAUSD" in positions
    assert positions["ADAUSD"]["exit_order_status"] == "FILLED"
    assert last_exit == {}


def test_reconcile_exit_orders_ignores_invalid_order_response_shape():
    class DummyClient:
        def get_margin_order(self, symbol, orderId):
            assert symbol == "ADAUSDT"
            assert orderId == 987
            return []

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
        now=datetime(2026, 3, 25, 20, tzinfo=UTC),
    )

    assert recent == []
    assert "ADAUSD" in positions
    assert positions["ADAUSD"]["exit_order_status"] == "NEW"
    assert last_exit == {}


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
        now=datetime(2026, 3, 25, 21, tzinfo=UTC),
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


def test_synchronize_positions_from_exchange_skips_positions_with_invalid_borrowed_quantity():
    class DummyClient:
        def get_margin_account(self):
            return {
                "userAssets": [
                    {"asset": "ADA", "free": "419.8626", "netAsset": "836.5626", "borrowed": "nan"},
                ]
            }

        def get_open_margin_orders(self, **kwargs):
            return []

        def get_all_margin_orders(self, symbol, **kwargs):
            raise AssertionError("recent orders should not be fetched when borrowed quantity is invalid")

    positions = {}
    current_bars = {
        "ADAUSD": pd.Series({"close": 0.2694, "high": 0.2710}),
    }

    events = synchronize_positions_from_exchange(
        client=DummyClient(),
        symbols=["ADAUSD"],
        positions=positions,
        current_bars=current_bars,
        config=DEFAULT_CONFIG,
        now=datetime(2026, 3, 25, 21, tzinfo=UTC),
    )

    assert events == []
    assert positions == {}



def test_synchronize_positions_from_exchange_preserves_existing_position_when_balance_quantity_is_non_finite():
    class DummyClient:
        def get_margin_account(self):
            return {
                "userAssets": [
                    {"asset": "ADA", "free": "nan", "netAsset": "nan", "borrowed": "0"},
                ]
            }

        def get_open_margin_orders(self, **kwargs):
            return []

        def get_all_margin_orders(self, symbol, **kwargs):
            raise AssertionError("recent orders should not be fetched for an existing tracked position")

    positions = {
        "ADAUSD": {
            "entry_price": 0.2726,
            "entry_date": "2026-03-25T10:12:00+00:00",
            "quantity": 837.4,
            "peak_price": 0.285,
            "target_sell": 0.278,
            "stop_price": 0.2453,
            "source": "exchange_sync",
        }
    }
    current_bars = {
        "ADAUSD": pd.Series({"close": 0.2694, "high": 0.2710}),
    }

    events = synchronize_positions_from_exchange(
        client=DummyClient(),
        symbols=["ADAUSD"],
        positions=positions,
        current_bars=current_bars,
        config=DEFAULT_CONFIG,
        now=datetime(2026, 3, 25, 21, tzinfo=UTC),
    )

    assert events == []
    assert positions["ADAUSD"]["quantity"] == pytest.approx(837.4)
    assert positions["ADAUSD"]["target_sell"] == pytest.approx(0.278)



def test_synchronize_positions_from_exchange_ignores_invalid_recent_orders_response():
    class DummyClient:
        def get_margin_account(self):
            return {
                "userAssets": [
                    {"asset": "ZEC", "free": "3.190806", "netAsset": "3.190806", "borrowed": "0"},
                ]
            }

        def get_open_margin_orders(self, **kwargs):
            return []

        def get_all_margin_orders(self, symbol, **kwargs):
            assert symbol == "ZECUSDT"
            return {}

    positions = {}
    current_bars = {
        "ZECUSD": pd.Series({"close": 230.87, "high": 241.00}),
    }

    events = synchronize_positions_from_exchange(
        client=DummyClient(),
        symbols=["ZECUSD"],
        positions=positions,
        current_bars=current_bars,
        config=DEFAULT_CONFIG,
        now=datetime(2026, 3, 25, 21, tzinfo=UTC),
    )

    assert [event["symbol"] for event in events] == ["ZECUSD"]
    assert positions["ZECUSD"]["entry_price"] == 230.87
    assert positions["ZECUSD"]["quantity"] == 3.190806


def test_synchronize_positions_from_exchange_ignores_invalid_account_response():
    class DummyClient:
        def get_margin_account(self):
            return []

        def get_open_margin_orders(self, **kwargs):
            raise AssertionError("open orders should not be fetched when account payload is invalid")

    positions = {"ADAUSD": {"entry_price": 0.27}}
    current_bars = {"ADAUSD": pd.Series({"close": 0.2694, "high": 0.2710})}

    events = synchronize_positions_from_exchange(
        client=DummyClient(),
        symbols=["ADAUSD"],
        positions=positions,
        current_bars=current_bars,
        config=DEFAULT_CONFIG,
        now=datetime(2026, 3, 25, 21, tzinfo=UTC),
    )

    assert events == []
    assert positions == {"ADAUSD": {"entry_price": 0.27}}


def test_synchronize_positions_from_exchange_ignores_invalid_open_orders_response():
    class DummyClient:
        def get_margin_account(self):
            return {"userAssets": []}

        def get_open_margin_orders(self, **kwargs):
            return {}

    positions = {"ADAUSD": {"entry_price": 0.27}}
    current_bars = {"ADAUSD": pd.Series({"close": 0.2694, "high": 0.2710})}

    events = synchronize_positions_from_exchange(
        client=DummyClient(),
        symbols=["ADAUSD"],
        positions=positions,
        current_bars=current_bars,
        config=DEFAULT_CONFIG,
        now=datetime(2026, 3, 25, 21, tzinfo=UTC),
    )

    assert events == []
    assert positions == {"ADAUSD": {"entry_price": 0.27}}


def test_synchronize_positions_from_exchange_ignores_invalid_recent_order_timestamps():
    class DummyClient:
        def get_margin_account(self):
            return {
                "userAssets": [
                    {"asset": "ZEC", "free": "3.190806", "netAsset": "3.190806", "borrowed": "0"},
                ]
            }

        def get_open_margin_orders(self, **kwargs):
            return []

        def get_all_margin_orders(self, symbol, **kwargs):
            assert symbol == "ZECUSDT"
            return [
                {
                    "symbol": "ZECUSDT",
                    "status": "FILLED",
                    "side": "BUY",
                    "price": "999.0",
                    "executedQty": "3.194",
                    "time": "bad-timestamp",
                    "updateTime": "bad-timestamp",
                },
                {
                    "symbol": "ZECUSDT",
                    "status": "FILLED",
                    "side": "BUY",
                    "price": "238.05",
                    "executedQty": "3.194",
                    "time": 1774440129390,
                    "updateTime": 1774440959689,
                },
            ]

    positions = {}
    current_bars = {
        "ZECUSD": pd.Series({"close": 230.87, "high": 241.00}),
    }

    events = synchronize_positions_from_exchange(
        client=DummyClient(),
        symbols=["ZECUSD"],
        positions=positions,
        current_bars=current_bars,
        config=DEFAULT_CONFIG,
        now=datetime(2026, 3, 25, 21, tzinfo=UTC),
    )

    assert [event["symbol"] for event in events] == ["ZECUSD"]
    assert positions["ZECUSD"]["entry_price"] == 238.05
    assert positions["ZECUSD"]["quantity"] == 3.190806



def test_synchronize_positions_from_exchange_ignores_invalid_open_order_timestamps():
    class DummyClient:
        def get_margin_account(self):
            return {
                "userAssets": [
                    {"asset": "ADA", "free": "419.8626", "netAsset": "836.5626", "borrowed": "0"},
                ]
            }

        def get_open_margin_orders(self, **kwargs):
            return [
                {
                    "symbol": "ADAUSDT",
                    "orderId": 111,
                    "status": "NEW",
                    "side": "SELL",
                    "price": "0.280",
                    "updateTime": "bad-timestamp",
                },
                {
                    "symbol": "ADAUSDT",
                    "orderId": 222,
                    "status": "NEW",
                    "side": "SELL",
                    "price": "0.278",
                    "updateTime": 1774466874374,
                },
            ]

        def get_all_margin_orders(self, symbol, **kwargs):
            assert symbol == "ADAUSDT"
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

    positions = {}
    current_bars = {
        "ADAUSD": pd.Series({"close": 0.2694, "high": 0.2710}),
    }

    events = synchronize_positions_from_exchange(
        client=DummyClient(),
        symbols=["ADAUSD"],
        positions=positions,
        current_bars=current_bars,
        config=DEFAULT_CONFIG,
        now=datetime(2026, 3, 25, 21, tzinfo=UTC),
    )

    assert [event["symbol"] for event in events] == ["ADAUSD"]
    assert positions["ADAUSD"]["exit_order_id"] == 222
    assert positions["ADAUSD"]["target_sell"] == 0.278


def test_synchronize_positions_from_exchange_preserves_existing_exit_state_when_open_order_id_is_invalid():
    class DummyClient:
        def get_margin_account(self):
            return {
                "userAssets": [
                    {"asset": "ADA", "free": "419.8626", "netAsset": "836.5626", "borrowed": "0"},
                ]
            }

        def get_open_margin_orders(self, **kwargs):
            return [
                {
                    "symbol": "ADAUSDT",
                    "orderId": "bad-id",
                    "status": "NEW",
                    "side": "SELL",
                    "price": "0.280",
                    "updateTime": 1774466874374,
                }
            ]

        def get_all_margin_orders(self, symbol, **kwargs):
            raise AssertionError("recent orders should not be fetched for an existing tracked position")

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
    current_bars = {
        "ADAUSD": pd.Series({"close": 0.2694, "high": 0.2710}),
    }

    events = synchronize_positions_from_exchange(
        client=DummyClient(),
        symbols=["ADAUSD"],
        positions=positions,
        current_bars=current_bars,
        config=DEFAULT_CONFIG,
        now=datetime(2026, 3, 25, 21, tzinfo=UTC),
    )

    assert events == []
    assert positions["ADAUSD"]["exit_order_id"] == 987
    assert positions["ADAUSD"]["exit_order_symbol"] == "ADAUSDT"
    assert positions["ADAUSD"]["exit_order_status"] == "NEW"
    assert positions["ADAUSD"]["exit_price"] == 0.278
    assert positions["ADAUSD"]["target_sell"] == 0.278


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

        def get_symbol_info(self, symbol):
            assert symbol == "DIPUSDT"
            return {
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01", "minPrice": "0.01"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.0001", "minQty": "0.0001"},
                    {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                ]
            }

        def get_open_margin_orders(self, **kwargs):
            return []

        def get_all_margin_orders(self, symbol, **kwargs):
            return []

        def create_margin_order(self, symbol, side, timeInForce, quantity, price, **kwargs):
            assert symbol == "DIPUSDT"
            assert side == "SELL"
            assert kwargs["type"] == "LIMIT"
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


def test_run_daily_cycle_keeps_position_when_live_exit_order_response_is_invalid(monkeypatch, tmp_path):
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

        def get_symbol_info(self, symbol):
            assert symbol == "DIPUSDT"
            return {
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01", "minPrice": "0.01"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.0001", "minQty": "0.0001"},
                    {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                ]
            }

        def get_open_margin_orders(self, **kwargs):
            return []

        def get_all_margin_orders(self, symbol, **kwargs):
            return []

        def create_margin_order(self, symbol, side, timeInForce, quantity, price, **kwargs):
            assert symbol == "DIPUSDT"
            assert side == "SELL"
            assert kwargs["type"] == "LIMIT"
            return []

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
    assert "exit_order_id" not in position
    assert payload["last_exit"] == {}

    log_path = tmp_path / "trade_log.jsonl"
    assert not log_path.exists() or not any(
        json.loads(line)["side"] == "staged_sell"
        for line in log_path.read_text().strip().split("\n")
        if line.strip()
    )


def test_run_daily_cycle_keeps_position_when_live_exit_order_response_is_missing_order_id(monkeypatch, tmp_path):
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

        def get_symbol_info(self, symbol):
            assert symbol == "DIPUSDT"
            return {
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01", "minPrice": "0.01"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.0001", "minQty": "0.0001"},
                    {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                ]
            }

        def get_open_margin_orders(self, **kwargs):
            return []

        def get_all_margin_orders(self, symbol, **kwargs):
            return []

        def create_margin_order(self, symbol, side, timeInForce, quantity, price, **kwargs):
            assert symbol == "DIPUSDT"
            assert side == "SELL"
            assert kwargs["type"] == "LIMIT"
            return {"symbol": symbol, "status": "NEW"}

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
    assert "exit_order_id" not in position
    assert payload["last_exit"] == {}

    log_path = tmp_path / "trade_log.jsonl"
    assert not log_path.exists() or not any(
        json.loads(line)["side"] == "staged_sell"
        for line in log_path.read_text().strip().split("\n")
        if line.strip()
    )


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


def test_run_daily_cycle_does_not_stage_invalid_live_entry_order_responses(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")

    bars = make_bars([100.0] * 29 + [90.0], "DIPUSD")
    monkeypatch.setattr("binance_worksteal.trade_live.fetch_daily_bars", lambda client, symbol, lookback_days=30: bars)
    monkeypatch.setattr("binance_worksteal.trade_live.place_limit_buy", lambda *args, **kwargs: [])
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


def test_run_daily_cycle_does_not_stage_live_entry_order_response_missing_order_id(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")

    bars = make_bars([100.0] * 29 + [90.0], "DIPUSD")
    monkeypatch.setattr("binance_worksteal.trade_live.fetch_daily_bars", lambda client, symbol, lookback_days=30: bars)
    monkeypatch.setattr(
        "binance_worksteal.trade_live.place_limit_buy",
        lambda *args, **kwargs: {"symbol": "DIPUSDT", "status": "NEW"},
    )
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


def test_run_daily_cycle_keeps_pending_entry_when_cancel_fails(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")

    hold_bars = make_bars([100.0] * 30, "HOLDUSD")
    dip_bars = make_bars([100.0] * 29 + [90.0], "DIPUSD")
    monkeypatch.setattr(
        "binance_worksteal.trade_live._fetch_all_bars",
        lambda client, symbols, lookback_days: {"HOLDUSD": hold_bars, "DIPUSD": dip_bars},
    )

    state = {
        "positions": {
            "HOLDUSD": {
                "entry_price": 100.0,
                "entry_date": "2026-01-20T00:00:00+00:00",
                "quantity": 1.0,
                "peak_price": 100.0,
                "target_sell": 103.0,
                "stop_price": 92.0,
                "source": "strategy",
            }
        },
        "pending_entries": {
            "DIPUSD": {
                "buy_price": 90.0,
                "quantity": 1.0,
                "target_sell": 92.7,
                "stop_price": 82.8,
                "placed_at": "2026-03-18T00:00:00+00:00",
                "expires_at": "2026-03-19T00:00:00+00:00",
                "order_id": 123,
                "order_symbol": "DIPUSDT",
                "source": "rule",
            }
        },
        "last_exit": {},
        "recent_trades": [],
        "peak_equity": 0.0,
    }
    (tmp_path / "live_state.json").write_text(json.dumps(state), encoding="utf-8")

    class DummyClient:
        def cancel_margin_order(self, symbol, orderId):
            raise RuntimeError("network down")

    config = build_runtime_config(
        build_arg_parser().parse_args(
            [
                "--max-positions", "1",
                "--dip-pct", "0.10",
                "--proximity-pct", "0.02",
                "--lookback-days", "20",
                "--profit-target", "0.03",
                "--stop-loss", "0.08",
                "--sma-filter", "0",
                "--risk-off-trigger-sma-period", "0",
                "--risk-off-trigger-momentum-period", "0",
                "--risk-off-market-breadth-filter", "0",
            ]
        )
    )

    run_daily_cycle(client=DummyClient(), symbols=["HOLDUSD", "DIPUSD"], config=config, dry_run=True)

    payload = json.loads((tmp_path / "live_state.json").read_text())
    assert "DIPUSD" in payload["pending_entries"]
    assert payload["pending_entries"]["DIPUSD"]["order_id"] == 123


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
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().strip().split("\n")]
    scan = next(event for event in events if event.get("type") == "entry_scan")
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


@pytest.mark.parametrize(
    ("func_name", "path_attr", "payload", "kind_label"),
    [
        ("log_trade", "LOG_FILE", {"symbol": "BTCUSD", "side": "buy"}, "trade log entry"),
        ("log_event", "EVENTS_FILE", {"type": "test", "value": 42}, "event log entry"),
    ],
)
def test_jsonl_logging_write_failures_warn_and_do_not_raise(
    monkeypatch, tmp_path, func_name, path_attr, payload, kind_label
):
    target_path = tmp_path / f"{func_name}.jsonl"
    monkeypatch.setattr(f"binance_worksteal.trade_live.{path_attr}", target_path)

    original_open = Path.open

    def fail_open(self, *args, **kwargs):
        if self == target_path:
            raise OSError("disk full")
        return original_open(self, *args, **kwargs)

    class DummyLogger:
        def __init__(self):
            self.messages = []

        def warning(self, message):
            self.messages.append(message)

    dummy_logger = DummyLogger()
    monkeypatch.setattr(Path, "open", fail_open)
    monkeypatch.setattr("binance_worksteal.trade_live.logger", dummy_logger)

    {"log_trade": log_trade, "log_event": log_event}[func_name](payload)

    assert not target_path.exists()
    assert len(dummy_logger.messages) == 1
    assert f"Failed to append {kind_label}" in dummy_logger.messages[0]
    assert str(target_path) in dummy_logger.messages[0]


def test_load_state_recovers_from_invalid_json_and_quarantines_file(monkeypatch, tmp_path):
    state_path = tmp_path / "live_state.json"
    state_path.write_text('{"positions":', encoding="utf-8")
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", state_path)

    class DummyLogger:
        def __init__(self):
            self.messages = []

        def warning(self, message):
            self.messages.append(message)

    dummy_logger = DummyLogger()
    monkeypatch.setattr("binance_worksteal.trade_live.logger", dummy_logger)

    state = load_state()

    backups = sorted(tmp_path.glob("live_state.corrupt.*.json"))
    assert state == {
        "positions": {},
        "pending_entries": {},
        "last_exit": {},
        "recent_trades": [],
        "peak_equity": 0.0,
    }
    assert not state_path.exists()
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == '{"positions":'
    assert len(dummy_logger.messages) == 1
    assert "Moved corrupt state to" in dummy_logger.messages[0]
    assert str(backups[0]) in dummy_logger.messages[0]


def test_load_state_resets_invalid_top_level_fields(monkeypatch, tmp_path):
    state_path = tmp_path / "live_state.json"
    state_path.write_text(
        json.dumps(
            {
                "positions": [],
                "pending_entries": None,
                "last_exit": "bad",
                "recent_trades": {"unexpected": True},
                "peak_equity": "oops",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", state_path)

    state = load_state()

    assert state["positions"] == {}
    assert state["pending_entries"] == {}
    assert state["last_exit"] == {}
    assert state["recent_trades"] == []
    assert state["peak_equity"] == 0.0


def test_merge_peak_equity_ignores_non_finite_inputs():
    assert _merge_peak_equity(float("nan"), 1234.5) == pytest.approx(1234.5)
    assert _merge_peak_equity(1234.5, float("nan")) == pytest.approx(1234.5)
    assert _merge_peak_equity(float("nan"), float("nan")) == 0.0



def test_run_daily_cycle_recovers_non_finite_peak_equity_when_saving_state(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")
    monkeypatch.setattr(
        "binance_worksteal.trade_live.load_state",
        lambda: {
            "positions": {},
            "pending_entries": {},
            "last_exit": {},
            "recent_trades": [],
            "peak_equity": float("nan"),
        },
    )
    monkeypatch.setattr("binance_worksteal.trade_live._fetch_all_bars", lambda client, symbols, lookback_days: {})
    monkeypatch.setattr("binance_worksteal.trade_live.get_account_equity", lambda client: 1234.0)

    config = WorkStealConfig(lookback_days=20)

    run_daily_cycle(client=None, symbols=["BTCUSD"], config=config, dry_run=False)

    payload = json.loads((tmp_path / "live_state.json").read_text(encoding="utf-8"))
    assert payload["peak_equity"] == pytest.approx(1234.0)



def test_load_state_resets_non_finite_peak_equity(monkeypatch, tmp_path):
    state_path = tmp_path / "live_state.json"
    state_path.write_text(
        json.dumps(
            {
                "positions": {},
                "pending_entries": {},
                "last_exit": {},
                "recent_trades": [],
                "peak_equity": "nan",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", state_path)

    state = load_state()

    assert state["peak_equity"] == 0.0


def test_load_state_quarantines_valid_json_with_invalid_top_level_type(monkeypatch, tmp_path):
    state_path = tmp_path / "live_state.json"
    state_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", state_path)

    class DummyLogger:
        def __init__(self):
            self.messages = []

        def warning(self, message):
            self.messages.append(message)

    dummy_logger = DummyLogger()
    monkeypatch.setattr("binance_worksteal.trade_live.logger", dummy_logger)

    state = load_state()

    backups = sorted(tmp_path.glob("live_state.corrupt.*.json"))
    assert state == {
        "positions": {},
        "pending_entries": {},
        "last_exit": {},
        "recent_trades": [],
        "peak_equity": 0.0,
    }
    assert not state_path.exists()
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "[]"
    assert len(dummy_logger.messages) == 1
    assert "expected a JSON object, got list" in dummy_logger.messages[0]
    assert "Moved corrupt state to" in dummy_logger.messages[0]
    assert str(backups[0]) in dummy_logger.messages[0]


def test_load_state_invalid_top_level_type_falls_back_cleanly_when_quarantine_unavailable(monkeypatch, tmp_path):
    state_path = tmp_path / "live_state.json"
    state_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", state_path)
    monkeypatch.setattr("binance_worksteal.trade_live._quarantine_invalid_state_file", lambda path: None)

    class DummyLogger:
        def __init__(self):
            self.messages = []

        def warning(self, message):
            self.messages.append(message)

    dummy_logger = DummyLogger()
    monkeypatch.setattr("binance_worksteal.trade_live.logger", dummy_logger)

    state = load_state()

    assert state == {
        "positions": {},
        "pending_entries": {},
        "last_exit": {},
        "recent_trades": [],
        "peak_equity": 0.0,
    }
    assert state_path.exists()
    assert state_path.read_text(encoding="utf-8") == "[]"
    assert len(dummy_logger.messages) == 1
    assert "expected a JSON object, got list" in dummy_logger.messages[0]
    assert "Moved corrupt state to" not in dummy_logger.messages[0]
    assert "Starting with empty state." in dummy_logger.messages[0]


def test_save_state_writes_valid_json_without_temp_files(monkeypatch, tmp_path):
    state_path = tmp_path / "live_state.json"
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", state_path)

    save_state(
        {
            "positions": {"BTCUSD": {"entry_price": 100.0}},
            "pending_entries": {"ETHUSD": {"buy_price": 90.0}},
            "last_exit": {"SOLUSD": "2026-03-30T00:00:00+00:00"},
            "recent_trades": [{"symbol": "BTCUSD", "side": "buy"}],
            "peak_equity": "12345.5",
        }
    )

    assert state_path.exists()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["positions"]["BTCUSD"]["entry_price"] == 100.0
    assert payload["pending_entries"]["ETHUSD"]["buy_price"] == 90.0
    assert payload["last_exit"]["SOLUSD"] == "2026-03-30T00:00:00+00:00"
    assert payload["recent_trades"] == [{"symbol": "BTCUSD", "side": "buy"}]
    assert payload["peak_equity"] == 12345.5
    assert [path.name for path in tmp_path.iterdir()] == ["live_state.json"]


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


def test_resolve_symbol_selection_context_caps_and_summarizes_routes():
    context = _resolve_symbol_selection_context(
        ["BTCUSD", "DOGEUSD", "ETHUSD"],
        "command line --symbols",
        max_symbols=2,
    )

    assert context == {
        "symbol_source": "command line --symbols (capped to --max-symbols=2)",
        "symbols": ["BTCUSD", "DOGEUSD"],
        "requested_symbol_count": 3,
        "symbol_count": 2,
        "was_capped": True,
        "omitted_symbol_count": 1,
        "omitted_symbols": ["ETHUSD"],
        "pair_routing": [
            {"symbol": "BTCUSD", "data_pair": "BTCUSDT", "order_pair": "BTCFDUSD"},
            {"symbol": "DOGEUSD", "data_pair": "DOGEUSDT", "order_pair": "DOGEUSDT"},
        ],
        "pair_routing_summary": {
            "route_count": 2,
            "same_pair_count": 1,
            "cross_quote_count": 1,
            "cross_quote_symbols": ["BTCUSD"],
            "data_quote_counts": {"USDT": 2},
            "order_quote_counts": {"FDUSD": 1, "USDT": 1},
            "required_order_quotes": ["FDUSD", "USDT"],
        },
    }


def test_trade_live_list_symbols_uses_shared_resolution_and_max_symbols(tmp_path, capsys):
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text(
        "symbols:\n"
        "  - symbol: BTCUSD\n"
        "  - symbol: ETHUSD\n"
        "  - symbol: SOLUSD\n",
        encoding="utf-8",
    )

    rc = trade_live_main(["--universe-file", str(universe_path), "--max-symbols", "2", "--list-symbols"])

    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 0
    assert out[0] == (
        f"Resolved 2 symbols from universe file {universe_path} (capped to --max-symbols=2):"
    )
    assert out[1:3] == ["BTCUSD", "ETHUSD"]
    assert out[3] == "Excluded by --max-symbols:"
    assert out[4] == "  SOLUSD"
    assert out[5] == "Pair routing:"
    assert out[6:8] == [
        "  BTCUSD: data=BTCUSDT order=BTCFDUSD",
        "  ETHUSD: data=ETHUSDT order=ETHFDUSD",
    ]
    assert out[8:] == [
        "Routing summary:",
        "  total routes: 2",
        "  same-pair routes: 0",
        "  cross-quote routes: 2",
        "  cross-quote symbols: BTCUSD, ETHUSD",
        "  data quote mix: USDT=2",
        "  order quote mix: FDUSD=2",
        "  required order quotes: FDUSD",
    ]


def test_trade_live_list_symbols_summary_dash_prints_json_to_stdout(tmp_path, capsys):
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text(
        "symbols:\n"
        "  - symbol: BTCUSD\n"
        "  - symbol: ETHUSD\n"
        "  - symbol: SOLUSD\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "live.yaml"
    config_path.write_text(
        "config:\n"
        "  dip_pct: 0.18\n",
        encoding="utf-8",
    )

    rc = trade_live_main([
        "--universe-file",
        str(universe_path),
        "--config-file",
        str(config_path),
        "--max-symbols",
        "2",
        "--list-symbols",
        "--summary-json",
        "-",
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["tool"] == "trade_live"
    assert payload["list_symbols_only"] is True
    assert payload["config_file"] == str(config_path)
    assert payload["requested_symbol_count"] == 3
    assert payload["symbol_count"] == 2
    assert payload["omitted_symbol_count"] == 1
    assert payload["omitted_symbols"] == ["SOLUSD"]
    assert payload["symbols"] == ["BTCUSD", "ETHUSD"]
    assert payload["pair_routing"] == [
        {"symbol": "BTCUSD", "data_pair": "BTCUSDT", "order_pair": "BTCFDUSD"},
        {"symbol": "ETHUSD", "data_pair": "ETHUSDT", "order_pair": "ETHFDUSD"},
    ]
    assert payload["pair_routing_summary"] == {
        "route_count": 2,
        "same_pair_count": 0,
        "cross_quote_count": 2,
        "cross_quote_symbols": ["BTCUSD", "ETHUSD"],
        "data_quote_counts": {"USDT": 2},
        "order_quote_counts": {"FDUSD": 2},
        "required_order_quotes": ["FDUSD"],
    }
    assert payload["was_capped"] is True
    assert payload["max_symbols"] == 2
    assert payload["status"] == "success"
    assert payload["exit_code"] == 0
    assert payload["invocation"]["module"] == "binance_worksteal.trade_live"
    assert payload["invocation"]["argv"][-2:] == ["--summary-json", "-"]
    assert captured.err.splitlines()[0] == (
        f"Resolved 2 symbols from universe file {universe_path} (capped to --max-symbols=2):"
    )


def test_trade_live_list_symbols_summary_file_prints_artifacts(tmp_path, capsys):
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text(
        "symbols:\n"
        "  - symbol: BTCUSD\n"
        "  - symbol: ETHUSD\n"
        "  - symbol: SOLUSD\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "live.yaml"
    config_path.write_text(
        "config:\n"
        "  dip_pct: 0.18\n",
        encoding="utf-8",
    )
    summary_path = tmp_path / "live_symbols_summary.json"

    rc = trade_live_main([
        "--universe-file",
        str(universe_path),
        "--config-file",
        str(config_path),
        "--max-symbols",
        "2",
        "--list-symbols",
        "--summary-json",
        str(summary_path),
    ])

    out = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert f"Resolved 2 symbols from universe file {universe_path} (capped to --max-symbols=2):" in out
    assert "Excluded by --max-symbols:" in out
    assert "  SOLUSD" in out
    assert "Pair routing:" in out
    assert "  BTCUSD: data=BTCUSDT order=BTCFDUSD" in out
    assert "  ETHUSD: data=ETHUSDT order=ETHFDUSD" in out
    assert "Routing summary:" in out
    assert "  cross-quote symbols: BTCUSD, ETHUSD" in out
    assert "  data quote mix: USDT=2" in out
    assert "  order quote mix: FDUSD=2" in out
    assert "  required order quotes: FDUSD" in out
    assert "Generated artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert payload["invocation"]["command"] in out
    assert "Wrote summary JSON" not in out
    assert payload["tool"] == "trade_live"
    assert payload["list_symbols_only"] is True
    assert payload["requested_symbol_count"] == 3
    assert payload["symbol_count"] == 2
    assert payload["pair_routing"] == [
        {"symbol": "BTCUSD", "data_pair": "BTCUSDT", "order_pair": "BTCFDUSD"},
        {"symbol": "ETHUSD", "data_pair": "ETHUSDT", "order_pair": "ETHFDUSD"},
    ]
    assert payload["pair_routing_summary"] == {
        "route_count": 2,
        "same_pair_count": 0,
        "cross_quote_count": 2,
        "cross_quote_symbols": ["BTCUSD", "ETHUSD"],
        "data_quote_counts": {"USDT": 2},
        "order_quote_counts": {"FDUSD": 2},
        "required_order_quotes": ["FDUSD"],
    }
    assert payload["summary_json_file"] == str(summary_path)
    assert payload["artifacts"] == [
        {
            "name": "summary_json",
            "path": str(summary_path),
            "description": "Structured JSON run summary.",
        }
    ]


def test_trade_live_invalid_universe_file_returns_error(tmp_path, capsys):
    missing_path = tmp_path / "missing.yaml"

    rc = trade_live_main(["--universe-file", str(missing_path), "--list-symbols"])

    out = capsys.readouterr().out.strip()
    assert rc == 1
    assert out == f"ERROR: Universe file not found: {missing_path}"


def test_trade_live_invalid_universe_file_summary_dash_prints_structured_error(tmp_path, capsys):
    missing_path = tmp_path / "missing.yaml"

    rc = trade_live_main(["--universe-file", str(missing_path), "--list-symbols", "--summary-json", "-"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["tool"] == "trade_live"
    assert payload["error"] == f"ERROR: Universe file not found: {missing_path}"
    assert payload["error_type"] == "FileNotFoundError"
    assert payload["config_file"] is None
    assert payload["list_symbols_only"] is True
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert payload["invocation"]["module"] == "binance_worksteal.trade_live"
    assert captured.err.strip() == f"ERROR: Universe file not found: {missing_path}"


def test_trade_live_invalid_universe_entry_returns_error(tmp_path, capsys):
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text("symbols:\n  - fee_tier: usdt\n", encoding="utf-8")

    rc = trade_live_main(["--universe-file", str(bad_path), "--list-symbols"])

    out = capsys.readouterr().out.strip()
    assert rc == 1
    assert out.startswith(f"ERROR: Symbol entry missing 'symbol' field at index 0 in {bad_path}:")


def test_trade_live_invalid_universe_entry_summary_dash_prints_structured_error(tmp_path, capsys):
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text("symbols:\n  - fee_tier: usdt\n", encoding="utf-8")

    rc = trade_live_main(["--universe-file", str(bad_path), "--list-symbols", "--summary-json", "-"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["tool"] == "trade_live"
    assert payload["error"].startswith(
        f"ERROR: Symbol entry missing 'symbol' field at index 0 in {bad_path}:"
    )
    assert payload["error_type"] == "ValueError"
    assert payload["list_symbols_only"] is True
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert payload["invocation"]["module"] == "binance_worksteal.trade_live"
    assert captured.err.strip().startswith(
        f"ERROR: Symbol entry missing 'symbol' field at index 0 in {bad_path}:"
    )


def test_trade_live_invalid_universe_boolean_field_summary_dash_prints_structured_error(tmp_path, capsys):
    bad_path = tmp_path / "bad_bool.yaml"
    bad_path.write_text(
        "symbols:\n"
        "  - symbol: BTCUSD\n"
        "    margin_eligible: maybe\n",
        encoding="utf-8",
    )

    rc = trade_live_main(["--universe-file", str(bad_path), "--list-symbols", "--summary-json", "-"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["tool"] == "trade_live"
    assert payload["error"] == f"ERROR: Invalid margin_eligible for BTCUSD in {bad_path}: 'maybe'"
    assert payload["error_type"] == "ValueError"
    assert payload["list_symbols_only"] is True
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert payload["invocation"]["module"] == "binance_worksteal.trade_live"
    assert captured.err.strip() == f"ERROR: Invalid margin_eligible for BTCUSD in {bad_path}: 'maybe'"


def test_trade_live_missing_config_summary_dash_prints_structured_error(tmp_path, capsys):
    missing_path = tmp_path / "missing.yaml"

    rc = trade_live_main([
        "--symbols",
        "BTCUSD",
        "--config-file",
        str(missing_path),
        "--summary-json",
        "-",
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["tool"] == "trade_live"
    assert payload["error"] == f"ERROR: Config file not found: {missing_path}"
    assert payload["error_type"] == "FileNotFoundError"
    assert payload["config_file"] == str(missing_path)
    assert payload["dry_run"] is True
    assert payload["live_mode"] is False
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert payload["invocation"]["module"] == "binance_worksteal.trade_live"
    assert captured.err.strip() == f"ERROR: Config file not found: {missing_path}"


def test_trade_live_print_config_exits_before_symbol_resolution(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "live_config.yaml"
    config_path.write_text(
        "config:\n"
        "  dip_pct: 0.18\n"
        "  market_breadth_filter: 0.45\n",
        encoding="utf-8",
    )

    def fail_resolve(*args, **kwargs):
        raise AssertionError("symbol resolution should not run for --print-config")

    monkeypatch.setattr("binance_worksteal.trade_live.resolve_cli_symbols_with_error", fail_resolve)

    rc = trade_live_main(["--print-config", "--config-file", str(config_path), "--dip-pct", "0.25"])

    out = capsys.readouterr().out
    payload = yaml.safe_load(out)
    assert rc == 0
    assert payload["config"]["dip_pct"] == pytest.approx(0.25)
    assert payload["config"]["market_breadth_filter"] == pytest.approx(0.45)


def test_trade_live_explain_config_exits_before_symbol_resolution(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "live_config.yaml"
    config_path.write_text(
        "config:\n"
        "  dip_pct: 0.18\n"
        "  market_breadth_filter: 0.45\n",
        encoding="utf-8",
    )

    def fail_resolve(*args, **kwargs):
        raise AssertionError("symbol resolution should not run for --explain-config")

    monkeypatch.setattr("binance_worksteal.trade_live.resolve_cli_symbols_with_error", fail_resolve)

    rc = trade_live_main(["--explain-config", "--config-file", str(config_path), "--dip-pct", "0.25"])

    out = capsys.readouterr().out
    payload = yaml.safe_load(out)
    assert rc == 0
    assert payload["sources"]["dip_pct"] == "cli"
    assert payload["sources"]["market_breadth_filter"] == "config_file"
    assert payload["changed_fields"]["dip_pct"]["config_file_value"] == pytest.approx(0.18)


def test_trade_live_preview_run_exits_before_client_and_model_init(monkeypatch, capsys):
    def fail_client(*args, **kwargs):
        raise AssertionError("Binance client should not initialize for --preview-run")

    def fail_model(*args, **kwargs):
        raise AssertionError("neural model should not load for --preview-run")

    monkeypatch.setattr("binance_worksteal.trade_live.BinanceClient", fail_client)
    monkeypatch.setattr("binance_worksteal.trade_live.load_neural_model", fail_model)

    rc = trade_live_main(
        [
            "--symbols",
            "btcusd",
            "ethusd",
            "--max-symbols",
            "1",
            "--live",
            "--daemon",
            "--gemini",
            "--neural-model",
            "model.pt",
            "--neural-symbols",
            "solusd",
            "adausd",
            "--preview-run",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "trade_live run preview:" in out
    assert "symbol_source: command line --symbols (capped to --max-symbols=1)" in out
    assert "symbol_count: 1" in out
    assert "omitted_symbol_count: 1" in out
    assert "omitted_symbols: ETHUSD" in out
    assert "symbols: BTCUSD" in out
    assert "Pair routing:" in out
    assert "  BTCUSD: data=BTCUSDT order=BTCFDUSD" in out
    assert "Routing summary:" in out
    assert "  cross-quote routes: 1" in out
    assert "  data quote mix: USDT=1" in out
    assert "  order quote mix: FDUSD=1" in out
    assert "  required order quotes: FDUSD" in out
    assert "dry_run: no" in out
    assert "live_mode: yes" in out
    assert "daemon: yes" in out
    assert "gemini_enabled: yes" in out
    assert "gemini_model: gemini-2.5-flash" in out
    assert "neural_model: model.pt" in out
    assert "neural_symbols: SOLUSD, ADAUSD" in out


def test_trade_live_preview_run_summary_dash_redirects_human_output(monkeypatch, capsys):
    def fail_client(*args, **kwargs):
        raise AssertionError("Binance client should not initialize for --preview-run")

    def fail_model(*args, **kwargs):
        raise AssertionError("neural model should not load for --preview-run")

    monkeypatch.setattr("binance_worksteal.trade_live.BinanceClient", fail_client)
    monkeypatch.setattr("binance_worksteal.trade_live.load_neural_model", fail_model)

    rc = trade_live_main(
        [
            "--symbols",
            "btcusd",
            "ethusd",
            "--max-symbols",
            "1",
            "--live",
            "--daemon",
            "--gemini",
            "--neural-model",
            "model.pt",
            "--neural-symbols",
            "solusd",
            "adausd",
            "--preview-run",
            "--summary-json",
            "-",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["tool"] == "trade_live"
    assert payload["preview_only"] is True
    assert payload["requested_symbol_count"] == 2
    assert payload["symbol_count"] == 1
    assert payload["omitted_symbol_count"] == 1
    assert payload["omitted_symbols"] == ["ETHUSD"]
    assert payload["symbols"] == ["BTCUSD"]
    assert payload["pair_routing"] == [
        {"symbol": "BTCUSD", "data_pair": "BTCUSDT", "order_pair": "BTCFDUSD"},
    ]
    assert payload["pair_routing_summary"] == {
        "route_count": 1,
        "same_pair_count": 0,
        "cross_quote_count": 1,
        "cross_quote_symbols": ["BTCUSD"],
        "data_quote_counts": {"USDT": 1},
        "order_quote_counts": {"FDUSD": 1},
        "required_order_quotes": ["FDUSD"],
    }
    assert payload["dry_run"] is False
    assert payload["live_mode"] is True
    assert payload["daemon"] is True
    assert payload["neural_symbols"] == ["SOLUSD", "ADAUSD"]
    assert payload["status"] == "success"
    assert payload["exit_code"] == 0
    assert payload["invocation"]["module"] == "binance_worksteal.trade_live"
    assert payload["invocation"]["argv"][-2:] == ["--summary-json", "-"]
    assert payload["invocation"]["command"].startswith("python -m binance_worksteal.trade_live ")
    assert "trade_live run preview:" in captured.err
    assert "summary_json: -" in captured.err


def test_trade_live_preview_run_surfaces_config_override_sources(monkeypatch, tmp_path, capsys):
    def fail_client(*args, **kwargs):
        raise AssertionError("Binance client should not initialize for --preview-run")

    def fail_model(*args, **kwargs):
        raise AssertionError("neural model should not load for --preview-run")

    monkeypatch.setattr("binance_worksteal.trade_live.BinanceClient", fail_client)
    monkeypatch.setattr("binance_worksteal.trade_live.load_neural_model", fail_model)

    config_path = tmp_path / "live_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "config": {
                    "market_breadth_filter": 0.55,
                    "rebalance_seeded_positions": False,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    rc = trade_live_main(
        [
            "--symbols",
            "btcusd",
            "--config-file",
            str(config_path),
            "--dip-pct",
            "0.25",
            "--preview-run",
            "--summary-json",
            "-",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert "Config overrides:" in captured.err
    assert "config_override_count: 3" in captured.err
    assert "config_file_override_fields: market_breadth_filter, rebalance_seeded_positions" in captured.err
    assert "cli_override_fields: dip_pct" in captured.err
    assert "dip_pct=0.25 (cli)" in captured.err
    assert "market_breadth_filter=0.55 (config_file)" in captured.err
    assert "rebalance_seeded_positions=no (config_file)" in captured.err
    assert payload["config_override_count"] == 3
    assert payload["config_override_fields"] == [
        "dip_pct",
        "market_breadth_filter",
        "rebalance_seeded_positions",
    ]
    assert payload["config_file_override_fields"] == [
        "market_breadth_filter",
        "rebalance_seeded_positions",
    ]
    assert payload["cli_override_fields"] == ["dip_pct"]
    assert payload["config_changed_fields"]["dip_pct"]["source"] == "cli"
    assert payload["config_changed_fields"]["dip_pct"]["value"] == pytest.approx(0.25)
    assert payload["config_changed_fields"]["market_breadth_filter"]["source"] == "config_file"
    assert payload["config_changed_fields"]["market_breadth_filter"]["value"] == pytest.approx(0.55)
    assert payload["config_changed_fields"]["rebalance_seeded_positions"]["source"] == "config_file"
    assert payload["config_changed_fields"]["rebalance_seeded_positions"]["value"] is False


def test_trade_live_preview_run_summary_file_prints_artifacts(monkeypatch, tmp_path, capsys):
    def fail_client(*args, **kwargs):
        raise AssertionError("Binance client should not initialize for --preview-run")

    def fail_model(*args, **kwargs):
        raise AssertionError("neural model should not load for --preview-run")

    monkeypatch.setattr("binance_worksteal.trade_live.BinanceClient", fail_client)
    monkeypatch.setattr("binance_worksteal.trade_live.load_neural_model", fail_model)

    summary_path = tmp_path / "live_preview_summary.json"

    rc = trade_live_main(
        [
            "--symbols",
            "btcusd",
            "ethusd",
            "--max-symbols",
            "1",
            "--live",
            "--daemon",
            "--gemini",
            "--neural-model",
            "model.pt",
            "--neural-symbols",
            "solusd",
            "adausd",
            "--preview-run",
            "--summary-json",
            str(summary_path),
        ]
    )

    out = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert "trade_live run preview:" in out
    assert "Pair routing:" in out
    assert "  BTCUSD: data=BTCUSDT order=BTCFDUSD" in out
    assert "Routing summary:" in out
    assert "  cross-quote symbols: BTCUSD" in out
    assert "  data quote mix: USDT=1" in out
    assert "  order quote mix: FDUSD=1" in out
    assert "  required order quotes: FDUSD" in out
    assert "Generated artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert payload["invocation"]["command"] in out
    assert "Wrote summary JSON" not in out
    assert payload["tool"] == "trade_live"
    assert payload["preview_only"] is True
    assert payload["requested_symbol_count"] == 2
    assert payload["symbol_count"] == 1
    assert payload["pair_routing"] == [
        {"symbol": "BTCUSD", "data_pair": "BTCUSDT", "order_pair": "BTCFDUSD"},
    ]
    assert payload["pair_routing_summary"] == {
        "route_count": 1,
        "same_pair_count": 0,
        "cross_quote_count": 1,
        "cross_quote_symbols": ["BTCUSD"],
        "data_quote_counts": {"USDT": 1},
        "order_quote_counts": {"FDUSD": 1},
        "required_order_quotes": ["FDUSD"],
    }
    assert payload["summary_json_file"] == str(summary_path)
    assert payload["artifacts"] == [
        {
            "name": "summary_json",
            "path": str(summary_path),
            "description": "Structured JSON run summary.",
        }
    ]


def test_trade_live_live_mode_client_unavailable_summary_dash_prints_structured_error(monkeypatch, capsys):
    class DummyLogger:
        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

        def error(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr("binance_worksteal.trade_live.BinanceClient", None)
    monkeypatch.setattr("binance_worksteal.trade_live.logger", DummyLogger())

    rc = trade_live_main([
        "--symbols",
        "BTCUSD",
        "ETHUSD",
        "--max-symbols",
        "1",
        "--live",
        "--summary-json",
        "-",
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["tool"] == "trade_live"
    assert payload["error"] == "ERROR: Binance client required for live mode but unavailable"
    assert payload["error_type"] == "RuntimeError"
    assert payload["symbol_source"] == "command line --symbols (capped to --max-symbols=1)"
    assert payload["requested_symbol_count"] == 2
    assert payload["symbol_count"] == 1
    assert payload["omitted_symbol_count"] == 1
    assert payload["omitted_symbols"] == ["ETHUSD"]
    assert payload["symbols"] == ["BTCUSD"]
    assert payload["pair_routing"] == [
        {"symbol": "BTCUSD", "data_pair": "BTCUSDT", "order_pair": "BTCFDUSD"},
    ]
    assert payload["pair_routing_summary"] == {
        "route_count": 1,
        "same_pair_count": 0,
        "cross_quote_count": 1,
        "cross_quote_symbols": ["BTCUSD"],
        "data_quote_counts": {"USDT": 1},
        "order_quote_counts": {"FDUSD": 1},
        "required_order_quotes": ["FDUSD"],
    }
    assert payload["dry_run"] is False
    assert payload["live_mode"] is True
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert payload["invocation"]["module"] == "binance_worksteal.trade_live"
    assert captured.err.strip() == "ERROR: Binance client required for live mode but unavailable"

def test_trade_live_live_mode_client_unavailable_summary_dash_includes_config_override_context(monkeypatch, tmp_path, capsys):
    class DummyLogger:
        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

        def error(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr("binance_worksteal.trade_live.BinanceClient", None)
    monkeypatch.setattr("binance_worksteal.trade_live.logger", DummyLogger())

    config_path = tmp_path / "live_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "config": {
                    "market_breadth_filter": 0.55,
                    "rebalance_seeded_positions": False,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    rc = trade_live_main([
        "--symbols",
        "BTCUSD",
        "ETHUSD",
        "--max-symbols",
        "1",
        "--config-file",
        str(config_path),
        "--dip-pct",
        "0.25",
        "--live",
        "--summary-json",
        "-",
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["config_override_count"] == 3
    assert payload["config_override_fields"] == [
        "dip_pct",
        "market_breadth_filter",
        "rebalance_seeded_positions",
    ]
    assert payload["config_file_override_fields"] == [
        "market_breadth_filter",
        "rebalance_seeded_positions",
    ]
    assert payload["cli_override_fields"] == ["dip_pct"]
    assert payload["config_changed_fields"]["dip_pct"]["source"] == "cli"
    assert payload["config_changed_fields"]["dip_pct"]["value"] == pytest.approx(0.25)
    assert payload["config_changed_fields"]["market_breadth_filter"]["source"] == "config_file"
    assert payload["config_changed_fields"]["market_breadth_filter"]["value"] == pytest.approx(0.55)
    assert payload["config_changed_fields"]["rebalance_seeded_positions"]["source"] == "config_file"
    assert payload["config_changed_fields"]["rebalance_seeded_positions"]["value"] is False



def test_trade_live_neural_model_load_failure_returns_error_before_run(monkeypatch, capsys):
    class DummyLogger:
        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

        def error(self, *_args, **_kwargs):
            pass

    def fail_model(*_args, **_kwargs):
        raise RuntimeError("bad model")

    def fail_run_daily_cycle(*_args, **_kwargs):
        raise AssertionError("run_daily_cycle should not run when neural model loading fails")

    monkeypatch.setattr("binance_worksteal.trade_live.logger", DummyLogger())
    monkeypatch.setattr("binance_worksteal.trade_live.load_neural_model", fail_model)
    monkeypatch.setattr("binance_worksteal.trade_live.run_daily_cycle", fail_run_daily_cycle)

    rc = trade_live_main([
        "--symbols",
        "BTCUSD",
        "--neural-model",
        "bad.pt",
    ])

    out = capsys.readouterr().out.strip()
    assert rc == 1
    assert out == "ERROR: Failed to load neural model: bad model"


def test_trade_live_neural_model_load_failure_summary_dash_prints_structured_error(monkeypatch, capsys):
    class DummyLogger:
        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

        def error(self, *_args, **_kwargs):
            pass

    def fail_model(*_args, **_kwargs):
        raise RuntimeError("bad model")

    def fail_run_daily_cycle(*_args, **_kwargs):
        raise AssertionError("run_daily_cycle should not run when neural model loading fails")

    monkeypatch.setattr("binance_worksteal.trade_live.logger", DummyLogger())
    monkeypatch.setattr("binance_worksteal.trade_live.load_neural_model", fail_model)
    monkeypatch.setattr("binance_worksteal.trade_live.run_daily_cycle", fail_run_daily_cycle)

    rc = trade_live_main([
        "--symbols",
        "BTCUSD",
        "ETHUSD",
        "--max-symbols",
        "1",
        "--neural-model",
        "bad.pt",
        "--summary-json",
        "-",
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["tool"] == "trade_live"
    assert payload["error"] == "ERROR: Failed to load neural model: bad model"
    assert payload["error_type"] == "RuntimeError"
    assert payload["symbol_source"] == "command line --symbols (capped to --max-symbols=1)"
    assert payload["requested_symbol_count"] == 2
    assert payload["symbol_count"] == 1
    assert payload["omitted_symbol_count"] == 1
    assert payload["omitted_symbols"] == ["ETHUSD"]
    assert payload["symbols"] == ["BTCUSD"]
    assert payload["pair_routing"] == [
        {"symbol": "BTCUSD", "data_pair": "BTCUSDT", "order_pair": "BTCFDUSD"},
    ]
    assert payload["pair_routing_summary"] == {
        "route_count": 1,
        "same_pair_count": 0,
        "cross_quote_count": 1,
        "cross_quote_symbols": ["BTCUSD"],
        "data_quote_counts": {"USDT": 1},
        "order_quote_counts": {"FDUSD": 1},
        "required_order_quotes": ["FDUSD"],
    }
    assert payload["dry_run"] is True
    assert payload["live_mode"] is False
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert payload["invocation"]["module"] == "binance_worksteal.trade_live"
    assert captured.err.strip() == "ERROR: Failed to load neural model: bad model"


def test_trade_live_neural_model_load_failure_summary_dash_includes_config_override_context(monkeypatch, tmp_path, capsys):
    class DummyLogger:
        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

        def error(self, *_args, **_kwargs):
            pass

    def fail_model(*_args, **_kwargs):
        raise RuntimeError("bad model")

    def fail_run_daily_cycle(*_args, **_kwargs):
        raise AssertionError("run_daily_cycle should not run when neural model loading fails")

    monkeypatch.setattr("binance_worksteal.trade_live.logger", DummyLogger())
    monkeypatch.setattr("binance_worksteal.trade_live.load_neural_model", fail_model)
    monkeypatch.setattr("binance_worksteal.trade_live.run_daily_cycle", fail_run_daily_cycle)

    config_path = tmp_path / "live_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "config": {
                    "market_breadth_filter": 0.55,
                    "rebalance_seeded_positions": False,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    rc = trade_live_main([
        "--symbols",
        "BTCUSD",
        "ETHUSD",
        "--max-symbols",
        "1",
        "--config-file",
        str(config_path),
        "--dip-pct",
        "0.25",
        "--neural-model",
        "bad.pt",
        "--summary-json",
        "-",
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["config_override_count"] == 3
    assert payload["config_override_fields"] == [
        "dip_pct",
        "market_breadth_filter",
        "rebalance_seeded_positions",
    ]
    assert payload["config_file_override_fields"] == [
        "market_breadth_filter",
        "rebalance_seeded_positions",
    ]
    assert payload["cli_override_fields"] == ["dip_pct"]
    assert payload["config_changed_fields"]["dip_pct"]["source"] == "cli"
    assert payload["config_changed_fields"]["dip_pct"]["value"] == pytest.approx(0.25)
    assert payload["config_changed_fields"]["market_breadth_filter"]["source"] == "config_file"
    assert payload["config_changed_fields"]["market_breadth_filter"]["value"] == pytest.approx(0.55)
    assert payload["config_changed_fields"]["rebalance_seeded_positions"]["source"] == "config_file"
    assert payload["config_changed_fields"]["rebalance_seeded_positions"]["value"] is False



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


def test_run_daily_cycle_reuses_cached_regime_breadth_for_market_logging(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")

    bars = make_bars([100.0] * 30, "BTCUSD")
    monkeypatch.setattr("binance_worksteal.trade_live.fetch_daily_bars", lambda client, symbol, lookback_days=30: bars)
    monkeypatch.setattr("binance_worksteal.trade_live.plan_legacy_rebalance_exits", lambda **kwargs: ([], False))

    calls = {"count": 0}

    def counting_breadth(current_bars, history, symbol_metrics=None):
        calls["count"] += 1
        return 0.25, 1, 1

    monkeypatch.setattr("binance_worksteal.strategy.compute_breadth_ratio", counting_breadth)
    monkeypatch.setattr("binance_worksteal.trade_live.compute_breadth_ratio", counting_breadth)

    config = build_runtime_config(
        build_arg_parser().parse_args(
            [
                "--dip-pct", "0.10",
                "--lookback-days", "20",
                "--sma-filter", "0",
                "--market-breadth-filter", "0.50",
                "--risk-off-trigger-sma-period", "0",
                "--risk-off-trigger-momentum-period", "0",
                "--risk-off-market-breadth-filter", "0",
            ]
        )
    )

    run_daily_cycle(client=None, symbols=["BTCUSD"], config=config, dry_run=True)

    assert calls["count"] == 1


def test_run_daily_cycle_passes_symbol_metrics_cache_to_strategy_calls(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.load_state", lambda: {})

    bars = {
        "BTCUSD": make_bars([100.0] * 29 + [90.0], "BTCUSD"),
    }
    monkeypatch.setattr("binance_worksteal.trade_live._fetch_all_bars", lambda client, symbols, lookback_days: bars)
    monkeypatch.setattr("binance_worksteal.trade_live.plan_legacy_rebalance_exits", lambda **kwargs: ([], set()))

    sentinel_metrics = {"BTCUSD": object()}
    calls = {"resolve": 0, "build": 0, "sma": 0}

    monkeypatch.setattr(
        "binance_worksteal.trade_live._build_symbol_metric_cache",
        lambda current_bars, history: sentinel_metrics,
    )

    class FakeRegime:
        def __init__(self, config):
            self.config = config
            self.skip_entries = False
            self.risk_off = False
            self.market_breadth_skip = False
            self.market_breadth_ratio = 0.0
            self.market_breadth_dipping_count = 0
            self.market_breadth_total_count = 0

    def fake_resolve_entry_regime(*, current_bars, history, config, symbol_metrics=None):
        assert symbol_metrics is sentinel_metrics
        calls["resolve"] += 1
        return FakeRegime(config)

    def fake_build_entry_candidates(*, symbol_metrics=None, **kwargs):
        assert symbol_metrics is sentinel_metrics
        calls["build"] += 1
        return []

    def fake_count_sma_pass_fail(all_bars, config, *, symbol_metrics=None):
        assert symbol_metrics is sentinel_metrics
        calls["sma"] += 1
        return 1, 0

    monkeypatch.setattr("binance_worksteal.trade_live.resolve_entry_regime", fake_resolve_entry_regime)
    monkeypatch.setattr("binance_worksteal.trade_live.build_entry_candidates", fake_build_entry_candidates)
    monkeypatch.setattr("binance_worksteal.trade_live._count_sma_pass_fail", fake_count_sma_pass_fail)
    monkeypatch.setattr("binance_worksteal.trade_live.log_event", lambda payload: None)

    config = WorkStealConfig(
        dip_pct=0.10,
        proximity_pct=0.02,
        lookback_days=20,
        sma_filter_period=0,
        market_breadth_filter=0.0,
        risk_off_market_breadth_filter=0.0,
        risk_off_trigger_sma_period=0,
        risk_off_trigger_momentum_period=0,
    )

    run_daily_cycle(client=None, symbols=["BTCUSD"], config=config, dry_run=True)

    assert calls == {"resolve": 1, "build": 1, "sma": 1}



def test_run_daily_cycle_logs_no_market_data_event(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.load_state", lambda: {})
    monkeypatch.setattr("binance_worksteal.trade_live._fetch_all_bars", lambda client, symbols, lookback_days: {})
    monkeypatch.setattr("binance_worksteal.trade_live.get_account_equity", lambda client: 1234.0)

    def fail_sync(**kwargs):
        raise AssertionError("sync should not run when market data is unavailable")

    monkeypatch.setattr("binance_worksteal.trade_live.synchronize_positions_from_exchange", fail_sync)
    events = []
    monkeypatch.setattr("binance_worksteal.trade_live.log_event", lambda payload: events.append(payload))

    config = WorkStealConfig(lookback_days=20)

    run_daily_cycle(client=None, symbols=["BTCUSD", "ETHUSD"], config=config, dry_run=False)

    assert len(events) == 1
    event = events[0]
    assert event["type"] == "daily_cycle"
    assert event["status"] == "no_market_data"
    assert event["cycle_status"] == "no_market_data"
    assert event["n_symbols_requested"] == 2
    assert event["n_symbols_with_data"] == 0

    saved_state = json.loads((tmp_path / "live_state.json").read_text())
    assert saved_state["positions"] == {}
    assert saved_state["pending_entries"] == {}


def test_run_health_report_logs_actual_risk_off_trigger(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.load_state", lambda: {})

    risk_off_bars = {
        "BTCUSD": make_bars([100.0] * 20 + [90.0] * 7, "BTCUSD"),
        "ETHUSD": make_bars([120.0] * 20 + [108.0] * 7, "ETHUSD"),
    }
    monkeypatch.setattr("binance_worksteal.trade_live._fetch_all_bars", lambda client, symbols, lookback_days: risk_off_bars)
    events = []
    monkeypatch.setattr("binance_worksteal.trade_live.log_event", lambda payload: events.append(payload))

    config = WorkStealConfig(
        dip_pct=0.10,
        lookback_days=20,
        sma_filter_period=0,
        market_breadth_filter=0.0,
        risk_off_market_breadth_filter=0.70,
        risk_off_trigger_sma_period=0,
        risk_off_trigger_momentum_period=7,
        risk_off_momentum_threshold=-0.05,
    )

    run_health_report(client=None, symbols=["BTCUSD", "ETHUSD"], config=config, dry_run=True)

    assert len(events) == 1
    event = events[0]
    assert event["type"] == "health_report"
    assert event["risk_off"] is True
    assert event["market_breadth_skip"] is False
    assert event["entry_skip"] is True


def test_run_health_report_logs_no_market_data_event(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.load_state", lambda: {})
    monkeypatch.setattr("binance_worksteal.trade_live._fetch_all_bars", lambda client, symbols, lookback_days: {})
    monkeypatch.setattr("binance_worksteal.trade_live.get_account_equity", lambda client: 1234.0)

    def fail_sync(**kwargs):
        raise AssertionError("sync should not run when market data is unavailable")

    monkeypatch.setattr("binance_worksteal.trade_live.synchronize_positions_from_exchange", fail_sync)
    events = []
    monkeypatch.setattr("binance_worksteal.trade_live.log_event", lambda payload: events.append(payload))

    config = WorkStealConfig(lookback_days=20)

    run_health_report(client=None, symbols=["BTCUSD", "ETHUSD"], config=config, dry_run=False)

    assert len(events) == 1
    event = events[0]
    assert event["type"] == "health_report"
    assert event["status"] == "no_market_data"
    assert event["health_status"] == "no_market_data"
    assert event["n_symbols_requested"] == 2
    assert event["n_symbols_with_data"] == 0
    assert event["entry_skip"] is False


def test_run_health_report_ignores_invalid_recent_trade_timestamps(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")
    monkeypatch.setattr(
        "binance_worksteal.trade_live.load_state",
        lambda: {"recent_trades": [{"timestamp": "not-a-date"}]},
    )

    bars = {
        "BTCUSD": make_bars([100.0] * 29 + [90.0], "BTCUSD"),
    }
    monkeypatch.setattr("binance_worksteal.trade_live._fetch_all_bars", lambda client, symbols, lookback_days: bars)
    events = []
    monkeypatch.setattr("binance_worksteal.trade_live.log_event", lambda payload: events.append(payload))

    config = WorkStealConfig(
        dip_pct=0.10,
        lookback_days=20,
        sma_filter_period=0,
        market_breadth_filter=0.0,
        risk_off_market_breadth_filter=0.0,
        risk_off_trigger_sma_period=0,
        risk_off_trigger_momentum_period=0,
    )

    run_health_report(client=None, symbols=["BTCUSD"], config=config, dry_run=True)

    assert len(events) == 1
    assert events[0]["type"] == "health_report"
    assert events[0]["days_since_trade"] == -1


def test_run_entry_scan_passes_symbol_metrics_cache_to_strategy_calls(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.load_state", lambda: {})

    bars = {
        "BTCUSD": make_bars([100.0] * 29 + [90.0], "BTCUSD"),
    }
    monkeypatch.setattr("binance_worksteal.trade_live._fetch_all_bars", lambda client, symbols, lookback_days: bars)

    sentinel_metrics = {"BTCUSD": object()}
    calls = {"resolve": 0, "build": 0}

    monkeypatch.setattr(
        "binance_worksteal.trade_live._build_symbol_metric_cache",
        lambda current_bars, history: sentinel_metrics,
    )

    class FakeRegime:
        def __init__(self, config):
            self.config = config
            self.skip_entries = False
            self.risk_off = False
            self.market_breadth_skip = False
            self.market_breadth_ratio = 0.0
            self.market_breadth_dipping_count = 0
            self.market_breadth_total_count = 0

    def fake_resolve_entry_regime(*, current_bars, history, config, symbol_metrics=None):
        assert symbol_metrics is sentinel_metrics
        calls["resolve"] += 1
        return FakeRegime(config)

    def fake_build_entry_candidates(*, symbol_metrics=None, **kwargs):
        assert symbol_metrics is sentinel_metrics
        calls["build"] += 1
        return []

    monkeypatch.setattr("binance_worksteal.trade_live.resolve_entry_regime", fake_resolve_entry_regime)
    monkeypatch.setattr("binance_worksteal.trade_live.build_entry_candidates", fake_build_entry_candidates)
    monkeypatch.setattr(
        "binance_worksteal.trade_live._stage_entry_candidates",
        lambda **kwargs: {
            "n_staged": 0,
            "n_proximity_skip": 0,
            "n_gemini_skip": 0,
            "n_already_held": 0,
            "n_neural_skip": 0,
            "n_order_fail": 0,
        },
    )
    events = []
    monkeypatch.setattr("binance_worksteal.trade_live.log_event", lambda payload: events.append(payload))

    config = WorkStealConfig(
        dip_pct=0.10,
        proximity_pct=0.02,
        lookback_days=20,
        sma_filter_period=0,
        market_breadth_filter=0.0,
        risk_off_market_breadth_filter=0.0,
        risk_off_trigger_sma_period=0,
        risk_off_trigger_momentum_period=0,
    )

    run_entry_scan(client=None, symbols=["BTCUSD"], config=config, dry_run=True)

    assert calls == {"resolve": 1, "build": 1}
    assert len(events) == 1
    assert events[0]["type"] == "entry_scan"



def test_run_entry_scan_logs_regime_fields_on_non_skip_path(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.load_state", lambda: {})

    bars = {
        "BTCUSD": make_bars([100.0] * 29 + [90.0], "BTCUSD"),
    }
    monkeypatch.setattr("binance_worksteal.trade_live._fetch_all_bars", lambda client, symbols, lookback_days: bars)
    monkeypatch.setattr(
        "binance_worksteal.trade_live._stage_entry_candidates",
        lambda **kwargs: {
            "n_staged": 0,
            "n_proximity_skip": 0,
            "n_gemini_skip": 0,
            "n_already_held": 0,
            "n_neural_skip": 0,
            "n_order_fail": 0,
        },
    )
    events = []
    monkeypatch.setattr("binance_worksteal.trade_live.log_event", lambda payload: events.append(payload))

    config = WorkStealConfig(
        dip_pct=0.10,
        proximity_pct=0.02,
        lookback_days=20,
        sma_filter_period=0,
        market_breadth_filter=0.0,
        risk_off_market_breadth_filter=0.0,
        risk_off_trigger_sma_period=0,
        risk_off_trigger_momentum_period=0,
    )

    run_entry_scan(client=None, symbols=["BTCUSD"], config=config, dry_run=True)

    assert len(events) == 1
    event = events[0]
    assert event["type"] == "entry_scan"
    assert event["risk_off"] is False
    assert event["risk_off_triggered"] is False
    assert event["market_breadth_skip"] is False


def test_run_entry_scan_logs_no_market_data_skip(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.load_state", lambda: {})
    monkeypatch.setattr("binance_worksteal.trade_live._fetch_all_bars", lambda client, symbols, lookback_days: {})
    monkeypatch.setattr("binance_worksteal.trade_live.get_account_equity", lambda client: 1234.0)

    def fail_sync(**kwargs):
        raise AssertionError("sync should not run when market data is unavailable")

    monkeypatch.setattr("binance_worksteal.trade_live.synchronize_positions_from_exchange", fail_sync)
    events = []
    monkeypatch.setattr("binance_worksteal.trade_live.log_event", lambda payload: events.append(payload))

    config = WorkStealConfig(lookback_days=20, max_positions=5)

    run_entry_scan(client=None, symbols=["BTCUSD"], config=config, dry_run=False)

    assert len(events) == 1
    event = events[0]
    assert event["type"] == "entry_scan"
    assert event["status"] == "no_market_data"
    assert event["skip_reason"] == "no_market_data"
    assert event["n_checked"] == 0
    assert event["n_candidates"] == 0
    assert event["n_staged"] == 0


def test_run_entry_scan_ignores_invalid_last_exit_timestamps(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_worksteal.trade_live.STATE_FILE", tmp_path / "live_state.json")
    monkeypatch.setattr("binance_worksteal.trade_live.LOG_FILE", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr("binance_worksteal.trade_live.EVENTS_FILE", tmp_path / "events.jsonl")
    monkeypatch.setattr(
        "binance_worksteal.trade_live.load_state",
        lambda: {"last_exit": {"BTCUSD": "not-a-date"}},
    )

    bars = {
        "BTCUSD": make_bars([100.0] * 29 + [90.0], "BTCUSD"),
    }
    monkeypatch.setattr("binance_worksteal.trade_live._fetch_all_bars", lambda client, symbols, lookback_days: bars)
    monkeypatch.setattr(
        "binance_worksteal.trade_live._stage_entry_candidates",
        lambda **kwargs: {
            "n_staged": 0,
            "n_proximity_skip": 0,
            "n_gemini_skip": 0,
            "n_already_held": 0,
            "n_neural_skip": 0,
            "n_order_fail": 0,
        },
    )
    events = []
    monkeypatch.setattr("binance_worksteal.trade_live.log_event", lambda payload: events.append(payload))

    config = WorkStealConfig(
        dip_pct=0.10,
        proximity_pct=0.02,
        lookback_days=20,
        sma_filter_period=0,
        market_breadth_filter=0.0,
        risk_off_market_breadth_filter=0.0,
        risk_off_trigger_sma_period=0,
        risk_off_trigger_momentum_period=0,
    )

    run_entry_scan(client=None, symbols=["BTCUSD"], config=config, dry_run=True)

    assert len(events) == 1
    assert events[0]["type"] == "entry_scan"


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
    loaded_model, loaded_symbols, _cfg = load_neural_model(str(ckpt_path))
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
    ckpt_path, _symbols = _make_dummy_checkpoint(tmp_path, n_symbols=n_symbols, seq_len=seq_len)
    model, model_symbols, _cfg = load_neural_model(str(ckpt_path))

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
    now = datetime(2026, 3, 18, tzinfo=UTC)

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

    trades = [json.loads(line) for line in (tmp_path / "trade_log.jsonl").read_text().strip().split("\n")]
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
        equity=10000.0, now=datetime(2026, 3, 18, tzinfo=UTC),
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
        equity=10000.0, now=datetime(2026, 3, 18, tzinfo=UTC),
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
        equity=10000.0, now=datetime(2026, 3, 18, tzinfo=UTC),
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
