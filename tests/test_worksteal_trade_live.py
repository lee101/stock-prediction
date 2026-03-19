"""Tests for live work-steal runtime config wiring."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binance_worksteal.trade_live import (
    DEFAULT_CONFIG,
    build_arg_parser,
    build_runtime_config,
    normalize_live_positions,
    plan_legacy_rebalance_exits,
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
