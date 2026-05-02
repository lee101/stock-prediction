from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from xgbnew.backtest import BacktestConfig, DayResult, DayTrade
from xgbnew.symbol_attribution import (
    _resolve_model_paths,
    aggregate_symbol_attribution,
    parse_args,
)


def _trade(symbol: str, net_return_pct: float, *, score: float = 0.8, side: int = 1) -> DayTrade:
    return DayTrade(
        symbol=symbol,
        score=score,
        actual_open=100.0,
        actual_close=100.0 * (1.0 + net_return_pct / 100.0),
        entry_fill_price=100.0,
        exit_fill_price=100.0 * (1.0 + net_return_pct / 100.0),
        spread_bps=2.0,
        commission_bps=0.0,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        leverage=1.0,
        gross_return_pct=net_return_pct,
        net_return_pct=net_return_pct,
        intraday_worst_dd_pct=max(-net_return_pct, 0.0),
        intraday_best_runup_pct=max(net_return_pct, 0.0),
        side=side,
    )


def test_aggregate_symbol_attribution_uses_portfolio_weights() -> None:
    config = BacktestConfig(top_n=2, allocation_mode="equal")
    result = SimpleNamespace(
        total_return_pct=3.0,
        sortino_ratio=1.25,
        max_drawdown_pct=4.0,
        day_results=[
            DayResult(
                day=date(2026, 1, 2),
                equity_start=10_000.0,
                equity_end=10_300.0,
                daily_return_pct=3.0,
                trades=[
                    _trade("AAA", 10.0),
                    _trade("BBB", -4.0),
                ],
            ),
            DayResult(
                day=date(2026, 1, 3),
                equity_start=10_300.0,
                equity_end=10_506.0,
                daily_return_pct=2.0,
                trades=[_trade("AAA", 2.0)],
            ),
        ],
    )

    rows, windows = aggregate_symbol_attribution(
        window_results=[(date(2026, 1, 2), date(2026, 1, 31), 21, result)],
        config=config,
    )

    by_symbol = {row["symbol"]: row for row in rows}
    assert by_symbol["AAA"]["portfolio_contribution_pct"] == pytest.approx(7.0)
    assert by_symbol["BBB"]["portfolio_contribution_pct"] == pytest.approx(-2.0)
    assert by_symbol["AAA"]["unique_traded_days"] == 2
    assert by_symbol["BBB"]["win_rate_pct"] == 0.0
    assert windows[0].top_symbol == "AAA"
    assert windows[0].worst_symbol == "BBB"


def test_aggregate_symbol_attribution_tracks_short_weight_scale() -> None:
    config = BacktestConfig(
        top_n=1,
        short_n=1,
        allocation_mode="equal",
        short_allocation_scale=0.5,
    )
    result = SimpleNamespace(
        total_return_pct=4.0,
        sortino_ratio=2.0,
        max_drawdown_pct=1.0,
        day_results=[
            DayResult(
                day=date(2026, 2, 2),
                equity_start=10_000.0,
                equity_end=10_400.0,
                daily_return_pct=4.0,
                trades=[
                    _trade("LONG", 6.0, side=1),
                    _trade("SHORT", 6.0, score=0.2, side=-1),
                ],
            ),
        ],
    )

    rows, _ = aggregate_symbol_attribution(
        window_results=[(date(2026, 2, 2), date(2026, 2, 28), 21, result)],
        config=config,
    )

    by_symbol = {row["symbol"]: row for row in rows}
    assert by_symbol["LONG"]["portfolio_contribution_pct"] == pytest.approx(4.0)
    assert by_symbol["SHORT"]["portfolio_contribution_pct"] == pytest.approx(2.0)
    assert by_symbol["SHORT"]["short_trades"] == 1


def test_resolve_model_paths_expands_globs_and_deduplicates(tmp_path) -> None:
    first = tmp_path / "a.pkl"
    second = tmp_path / "b.pkl"
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    paths = _resolve_model_paths(f"{tmp_path}/*.pkl,{first}")

    assert paths == [first, second]


def test_parse_args_exposes_realism_knobs() -> None:
    args = parse_args(
        [
            "--symbols-file",
            "symbols.txt",
            "--model-paths",
            "m.pkl",
            "--hold-through",
            "--leverage",
            "2.25",
            "--inference-min-dolvol",
            "50000000",
            "--inference-min-vol-20d",
            "0.12",
            "--fill-buffer-bps",
            "5",
        ],
    )

    assert args.hold_through is True
    assert args.leverage == 2.25
    assert args.inference_min_dolvol == 50_000_000
    assert args.inference_min_vol_20d == 0.12
    assert args.fill_buffer_bps == 5
