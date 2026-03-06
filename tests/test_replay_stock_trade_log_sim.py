from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import unified_hourly_experiment.replay_stock_trade_log_sim as replay_mod
from unified_hourly_experiment.replay_stock_trade_log_sim import (
    compare_counts,
    compare_entries,
    load_live_entry_fills,
    load_live_entries,
    parse_bool_list,
    run_replay,
)


def _write_log(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_load_live_entries_parses_and_deduplicates_hour_symbol(tmp_path: Path) -> None:
    log_path = tmp_path / "stock_trade_log.jsonl"
    _write_log(
        log_path,
        [
            {
                "event": "entry",
                "symbol": "NVDA",
                "side": "long",
                "price": 100.0,
                "exit_price": 101.0,
                "intensity": 0.25,
                "logged_at": "2026-03-03T15:01:00Z",
            },
            # Same symbol/hour should keep latest row.
            {
                "event": "entry",
                "symbol": "NVDA",
                "side": "long",
                "price": 99.0,
                "exit_price": 100.5,
                "intensity": 0.50,
                "logged_at": "2026-03-03T15:40:00Z",
            },
            {
                "event": "entry",
                "symbol": "MTCH",
                "side": "short",
                "price": 30.5,
                "exit_price": 30.0,
                "intensity": 0.10,
                "logged_at": "2026-03-03T16:05:00Z",
            },
            # Non-entry events must be ignored.
            {
                "event": "exit_filled",
                "symbol": "NVDA",
                "side": "long",
                "price": 101.0,
                "exit_price": 0.0,
                "logged_at": "2026-03-03T17:00:00Z",
            },
        ],
    )

    out = load_live_entries(
        trade_log=log_path,
        symbols=None,
        start=pd.Timestamp("2026-03-03T15:00:00Z"),
        end=pd.Timestamp("2026-03-03T17:00:00Z"),
    )
    assert len(out) == 2

    nvda = out[out["symbol"] == "NVDA"].iloc[0]
    assert float(nvda["buy_price"]) == 99.0
    assert float(nvda["sell_price"]) == 100.5
    assert float(nvda["buy_amount"]) == 50.0
    assert float(nvda["sell_amount"]) == 0.0
    assert nvda["timestamp"] == pd.Timestamp("2026-03-03T15:00:00Z")

    mtch = out[out["symbol"] == "MTCH"].iloc[0]
    # Short entry should map to sell=open and buy=cover.
    assert float(mtch["buy_price"]) == 30.0
    assert float(mtch["sell_price"]) == 30.5
    assert float(mtch["buy_amount"]) == 0.0
    assert float(mtch["sell_amount"]) == 10.0
    assert float(mtch["entry_price"]) == 30.5
    assert float(mtch["qty"]) == 0.0
    assert mtch["timestamp"] == pd.Timestamp("2026-03-03T16:00:00Z")


def test_load_live_entry_fills_uses_broker_closed_entry_orders(tmp_path: Path) -> None:
    event_log = tmp_path / "stock_event_log.jsonl"
    _write_log(
        event_log,
        [
            {
                "event_type": "entry_order_submit_succeeded",
                "symbol": "NVDA",
                "side": "long",
                "order_id": "entry-1",
                "logged_at": "2026-03-03T15:01:00Z",
            },
            {
                "event_type": "exit_order_submit_succeeded",
                "symbol": "NVDA",
                "side": "sell",
                "order_id": "exit-1",
                "logged_at": "2026-03-03T15:10:00Z",
            },
            {
                "event_type": "broker_closed_order",
                "event_ts": "2026-03-03T15:05:00Z",
                "order": {
                    "id": "entry-1",
                    "symbol": "NVDA",
                    "status": "filled",
                    "filled_qty": 7,
                    "filled_avg_price": 100.25,
                    "filled_at": "2026-03-03T15:05:00Z",
                },
            },
            {
                "event_type": "broker_closed_order",
                "event_ts": "2026-03-03T15:55:00Z",
                "order": {
                    "id": "exit-1",
                    "symbol": "NVDA",
                    "status": "filled",
                    "filled_qty": 7,
                    "filled_avg_price": 101.0,
                    "filled_at": "2026-03-03T15:55:00Z",
                },
            },
        ],
    )

    out = load_live_entry_fills(
        event_log=event_log,
        symbols=None,
        start=pd.Timestamp("2026-03-03T15:00:00Z"),
        end=pd.Timestamp("2026-03-03T16:00:00Z"),
    )

    assert len(out) == 1
    row = out.iloc[0]
    assert row["symbol"] == "NVDA"
    assert row["side"] == "long"
    assert float(row["qty"]) == 7.0
    assert float(row["entry_price"]) == 100.25
    assert row["timestamp"] == pd.Timestamp("2026-03-03T15:00:00Z")


def test_compare_counts_computes_alignment_metrics() -> None:
    live_counts = pd.DataFrame(
        [
            {"hour": pd.Timestamp("2026-03-03T15:00:00Z"), "symbol": "NVDA", "side": "long", "count": 2},
            {"hour": pd.Timestamp("2026-03-03T16:00:00Z"), "symbol": "MTCH", "side": "short", "count": 1},
        ]
    )
    sim_counts = pd.DataFrame(
        [
            {"hour": pd.Timestamp("2026-03-03T15:00:00Z"), "symbol": "NVDA", "side": "long", "count": 1},
            {"hour": pd.Timestamp("2026-03-03T16:00:00Z"), "symbol": "MTCH", "side": "short", "count": 1},
            {"hour": pd.Timestamp("2026-03-03T16:00:00Z"), "symbol": "GOOG", "side": "long", "count": 1},
        ]
    )
    out = compare_counts(live_counts, sim_counts)
    assert out["live_entries"] == 3
    assert out["sim_entries"] == 3
    assert out["rows_compared"] == 3
    # |2-1| + |1-1| + |0-1| = 2
    assert out["hourly_abs_count_delta_total"] == 2.0
    # Only MTCH row matches exactly.
    assert out["exact_row_ratio"] == (1.0 / 3.0)


def test_parse_bool_list_accepts_common_tokens() -> None:
    assert parse_bool_list("1,true,yes,on") == [True, True, True, True]
    assert parse_bool_list("0,false,no,off") == [False, False, False, False]


def test_parse_bool_list_rejects_invalid_token() -> None:
    try:
        parse_bool_list("1,maybe")
    except ValueError as exc:
        assert "Invalid boolean token" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid bool token")


def test_compare_entries_tracks_qty_and_price_error() -> None:
    live_entries = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-03T15:00:00Z"),
                "symbol": "NVDA",
                "side": "long",
                "qty": 10.0,
                "entry_price": 100.0,
            },
            {
                "timestamp": pd.Timestamp("2026-03-03T16:00:00Z"),
                "symbol": "MTCH",
                "side": "short",
                "qty": 5.0,
                "entry_price": 30.5,
            },
        ]
    )
    sim_entries = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-03T15:00:00Z"),
                "symbol": "NVDA",
                "side": "long",
                "qty": 8.0,
                "entry_price": 101.0,
            },
            {
                "timestamp": pd.Timestamp("2026-03-03T16:00:00Z"),
                "symbol": "MTCH",
                "side": "short",
                "qty": 5.0,
                "entry_price": 30.25,
            },
        ]
    )

    out = compare_entries(live_entries, sim_entries)

    assert out["live_entries"] == 2
    assert out["sim_entries"] == 2
    assert out["hourly_abs_count_delta_total"] == 0.0
    assert out["hourly_abs_qty_delta_total"] == 2.0
    assert out["matched_price_mae"] == 0.625


def test_run_replay_passes_market_order_entry(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _DummyResult:
        trades = []
        metrics = {}

    def _fake_run_portfolio_simulation(bars, actions, config, horizon):
        captured["market_order_entry"] = config.market_order_entry
        return _DummyResult()

    monkeypatch.setattr(replay_mod, "run_portfolio_simulation", _fake_run_portfolio_simulation)

    bars = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-03T15:00:00Z"),
                "symbol": "NVDA",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
            }
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-03T15:00:00Z"),
                "symbol": "NVDA",
                "buy_price": 100.0,
                "sell_price": 101.0,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
            }
        ]
    )

    run_replay(
        bars=bars,
        actions=actions,
        symbols=["NVDA"],
        initial_cash=50_000.0,
        max_positions=1,
        max_hold_hours=6,
        min_edge=-1.0,
        fee_rate=0.001,
        leverage=2.0,
        decision_lag_bars=0,
        bar_margin=0.0005,
        entry_order_ttl_hours=0,
        market_order_entry=True,
        sim_backend="python",
    )
    assert captured["market_order_entry"] is True


def test_run_replay_hourly_trader_backend_generates_entry_rows() -> None:
    ts0 = pd.Timestamp("2026-03-03T15:00:00Z")
    ts1 = ts0 + pd.Timedelta(hours=1)
    ts2 = ts1 + pd.Timedelta(hours=1)

    bars = pd.DataFrame(
        [
            {"timestamp": ts0, "symbol": "NVDA", "open": 100.0, "high": 101.0, "low": 100.0, "close": 100.0},
            {"timestamp": ts1, "symbol": "NVDA", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            {"timestamp": ts2, "symbol": "NVDA", "open": 110.0, "high": 111.0, "low": 109.0, "close": 110.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "timestamp": ts0,
                "symbol": "NVDA",
                "side": "long",
                "qty": 10.0,
                "entry_price": 100.0,
                "buy_price": 100.0,
                "sell_price": 110.0,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
            },
            {
                "timestamp": ts1,
                "symbol": "NVDA",
                "side": "long",
                "qty": 0.0,
                "entry_price": 100.0,
                "buy_price": 100.0,
                "sell_price": 110.0,
                "buy_amount": 0.0,
                "sell_amount": 100.0,
            },
            {
                "timestamp": ts2,
                "symbol": "NVDA",
                "side": "long",
                "qty": 0.0,
                "entry_price": 100.0,
                "buy_price": 100.0,
                "sell_price": 110.0,
                "buy_amount": 0.0,
                "sell_amount": 0.0,
            },
        ]
    )

    out = run_replay(
        bars=bars,
        actions=actions,
        symbols=["NVDA"],
        initial_cash=1_000.0,
        max_positions=1,
        max_hold_hours=6,
        min_edge=-1.0,
        fee_rate=0.0,
        leverage=2.0,
        decision_lag_bars=1,
        bar_margin=0.0,
        entry_order_ttl_hours=0,
        market_order_entry=False,
        sim_backend="hourly_trader",
        cancel_ack_delay_bars=1,
        partial_fill_on_touch=False,
    )

    assert out["sim_counts"]["count"].sum() == 1
    entry = out["sim_entries"].iloc[0]
    assert entry["side"] == "long"
    assert entry["qty"] == 20.0
    assert entry["entry_price"] == 100.0
