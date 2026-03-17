from __future__ import annotations

import json

import pandas as pd

from newnanoalpacahourlyexp.run_hourly_trader_sim import (
    _best_mode_summary,
    _build_starting_position_scenarios,
    _common_eval_end_timestamp,
    _trim_frame_for_inference,
    _load_initial_state,
    _parse_checkpoint_map,
    _summarize_scenario_results,
)


def test_parse_checkpoint_map(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("x")

    parsed = _parse_checkpoint_map(f"ETHUSD={checkpoint}")

    assert parsed == {"ETHUSD": checkpoint.resolve()}


def test_load_initial_state_supports_alias_fields(tmp_path) -> None:
    path = tmp_path / "initial_state.json"
    path.write_text(
        json.dumps(
            {
                "cash": 47000.5,
                "positions": [
                    {"symbol": "ETH/USD", "quantity": "0.25"},
                ],
                "open_orders": [
                    {
                        "symbol": "ETH/USD",
                        "side": "buy",
                        "quantity": "6.1",
                        "price": "1928.73",
                        "kind": "entry",
                        "created_at": "2026-03-05T17:00:07Z",
                    }
                ],
            }
        )
    )

    initial_cash, positions, open_orders = _load_initial_state(path)

    assert initial_cash == 47000.5
    assert positions == {"ETHUSD": 0.25}
    assert len(open_orders) == 1
    assert open_orders[0].symbol == "ETHUSD"
    assert open_orders[0].qty == 6.1
    assert open_orders[0].limit_price == 1928.73
    assert open_orders[0].placed_at.isoformat() == "2026-03-05T17:00:07+00:00"


def test_build_starting_position_scenarios_includes_flat_provided_and_seeded_variants() -> None:
    bars = pd.DataFrame(
        [
            {"timestamp": "2026-03-01T00:00:00Z", "symbol": "ETHUSD", "close": 2000.0},
            {"timestamp": "2026-03-01T00:00:00Z", "symbol": "BTCUSD", "close": 100000.0},
        ]
    )

    scenarios = _build_starting_position_scenarios(
        bars=bars,
        symbols=["ETHUSD", "BTCUSD"],
        initial_cash=10_000.0,
        allocation_usd=500.0,
        allocation_pct=None,
        allocation_mode="per_symbol",
        allow_short=True,
        initial_positions={"ETHUSD": 0.25},
        initial_open_orders=[],
        symbol_limit=2,
    )

    names = {scenario.name for scenario in scenarios}
    assert "flat" in names
    assert "provided_state" in names
    assert "basket_long" in names
    assert "basket_short" in names
    assert any(name.startswith("long_") for name in names)
    assert any(name.startswith("short_") for name in names)


def test_summarize_scenario_results_ranks_by_sortino_then_return() -> None:
    summary = _summarize_scenario_results(
        [
            {"scenario": "flat", "metrics": {"sortino": 0.5, "total_return": 0.20, "max_drawdown": -0.10}},
            {"scenario": "basket_long", "metrics": {"sortino": 1.2, "total_return": 0.05, "max_drawdown": -0.08}},
            {"scenario": "basket_short", "metrics": {"sortino": 1.2, "total_return": 0.03, "max_drawdown": -0.02}},
        ]
    )

    assert summary["scenario_count"] == 3
    assert summary["best_scenario"] == "basket_long"


def test_best_mode_summary_prefers_higher_sortino_then_return() -> None:
    best = _best_mode_summary(
        [
            {"mode": "baseline", "best_metrics": {"sortino": 0.4, "total_return": 0.08, "max_drawdown": -0.05}},
            {"mode": "gemini", "best_metrics": {"sortino": 0.8, "total_return": 0.03, "max_drawdown": -0.04}},
            {"mode": "amount_reforecasting", "best_metrics": {"sortino": 0.8, "total_return": 0.05, "max_drawdown": -0.07}},
        ]
    )

    assert best is not None
    assert best["mode"] == "amount_reforecasting"


def test_common_eval_end_timestamp_uses_overlapping_latest_timestamp() -> None:
    early = pd.DataFrame({"timestamp": pd.date_range("2026-03-01", periods=4, freq="h", tz="UTC")})
    late = pd.DataFrame({"timestamp": pd.date_range("2026-03-01", periods=8, freq="h", tz="UTC")})

    ts_end = _common_eval_end_timestamp([early, late])

    assert ts_end == pd.Timestamp("2026-03-01T03:00:00Z")


def test_trim_frame_for_inference_respects_common_end_before_tail_selection() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-03-01", periods=8, freq="h", tz="UTC"),
            "symbol": ["SOLUSD"] * 8,
            "close": range(8),
        }
    )

    trimmed = _trim_frame_for_inference(
        frame,
        rows_needed=3,
        ts_end=pd.Timestamp("2026-03-01T04:00:00Z"),
    )

    assert trimmed["timestamp"].tolist() == [
        pd.Timestamp("2026-03-01T02:00:00Z"),
        pd.Timestamp("2026-03-01T03:00:00Z"),
        pd.Timestamp("2026-03-01T04:00:00Z"),
    ]
