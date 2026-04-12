from __future__ import annotations

import math
from pathlib import Path

import pytest
import pandas as pd

from trade_stock_wide.planner import (
    WidePlannerConfig,
    build_wide_plan,
    candidate_from_row,
    render_plan_text,
    validate_long_price_levels,
)
from trade_stock_wide.replay import simulate_day, simulate_wide_strategy
from trade_stock_wide.types import WideCandidate
import trade_stock_wide.run as run_mod
from trade_stock_wide.intraday import load_hourly_symbol_history, simulate_intraday_day
from trade_stock_wide.runtime_logging import WideRunLogger


def _candidate(
    symbol: str,
    *,
    forecasted_pnl: float,
    avg_return: float,
    last_close: float,
    entry_price: float,
    take_profit_price: float,
    realized_close: float,
    realized_high: float,
    realized_low: float,
    score: float | None = None,
    dollar_vol_20d: float | None = None,
    spread_bps_estimate: float | None = None,
    allocation_fraction_of_equity: float | None = None,
) -> WideCandidate:
    return WideCandidate(
        symbol=symbol,
        strategy="maxdiff",
        forecasted_pnl=forecasted_pnl,
        avg_return=avg_return,
        last_close=last_close,
        entry_price=entry_price,
        take_profit_price=take_profit_price,
        predicted_high=take_profit_price,
        predicted_low=entry_price,
        realized_close=realized_close,
        realized_high=realized_high,
        realized_low=realized_low,
        score=score if score is not None else forecasted_pnl,
        day_index=0,
        dollar_vol_20d=dollar_vol_20d,
        spread_bps_estimate=spread_bps_estimate,
        allocation_fraction_of_equity=allocation_fraction_of_equity,
    )


def test_candidate_from_row_prefers_highest_valid_forecast_strategy():
    row = {
        "close": 100.0,
        "predicted_high": 108.0,
        "predicted_low": 96.0,
        "high": 109.0,
        "low": 95.0,
        "maxdiffprofit_profit": 0.07,
        "maxdiff_avg_daily_return": 0.03,
        "maxdiffprofit_low_price": 97.5,
        "maxdiffprofit_high_price": 106.0,
        "entry_takeprofit_profit": 0.04,
        "entry_takeprofit_avg_daily_return": 0.02,
        "takeprofit_low_price": 98.5,
        "takeprofit_high_price": 104.5,
        "simple_strategy_return": 0.01,
        "simple_strategy_avg_daily_return": 0.01,
    }

    candidate = candidate_from_row("AAPL", row, day_index=3)

    assert candidate is not None
    assert candidate.symbol == "AAPL"
    assert candidate.strategy == "maxdiff"
    assert candidate.entry_price == pytest.approx(97.5)
    assert candidate.take_profit_price == pytest.approx(106.0)
    assert candidate.day_index == 3
    assert candidate.session_date is None


def test_candidate_from_row_refits_prices_from_modifier_fields():
    row = {
        "close": 100.0,
        "predicted_high": 108.0,
        "predicted_low": 96.0,
        "high": 109.0,
        "low": 95.0,
        "maxdiffprofit_profit": 0.07,
        "maxdiff_avg_daily_return": 0.03,
        "maxdiffprofit_low_price": 98.0,
        "maxdiffprofit_high_price": 106.0,
        "maxdiffprofit_profit_low_multiplier": -0.01,
        "maxdiffprofit_profit_high_multiplier": 0.04,
    }

    candidate = candidate_from_row("AAPL", row, day_index=0, require_realized_ohlc=True)

    assert candidate is not None
    assert candidate.entry_price == pytest.approx(98.5)
    assert candidate.take_profit_price == pytest.approx(105.0)


def test_candidate_from_row_requires_realized_ohlc_for_replay():
    row = {
        "close": 100.0,
        "predicted_high": 108.0,
        "predicted_low": 96.0,
        "maxdiffprofit_profit": 0.07,
        "maxdiff_avg_daily_return": 0.03,
        "maxdiffprofit_low_price": 97.5,
        "maxdiffprofit_high_price": 106.0,
    }

    candidate = candidate_from_row("AAPL", row, day_index=0, require_realized_ohlc=True)

    assert candidate is None


def test_candidate_from_row_sets_session_date_when_timestamp_present():
    row = {
        "timestamp": "2024-03-04T05:00:00Z",
        "close": 100.0,
        "predicted_high": 108.0,
        "predicted_low": 96.0,
        "high": 109.0,
        "low": 95.0,
        "maxdiffprofit_profit": 0.07,
        "maxdiff_avg_daily_return": 0.03,
        "maxdiffprofit_low_price": 97.5,
        "maxdiffprofit_high_price": 106.0,
    }

    candidate = candidate_from_row("AAPL", row, day_index=0, require_realized_ohlc=True)

    assert candidate is not None
    assert candidate.session_date == "2024-03-04"


def test_build_wide_plan_uses_half_equity_pair_size():
    config = WidePlannerConfig(top_k=2, pair_notional_fraction=0.5, max_pair_notional_fraction=0.5, max_leverage=2.0)
    candidates = [
        _candidate(
            "AAPL",
            forecasted_pnl=0.05,
            avg_return=0.02,
            last_close=100.0,
            entry_price=98.0,
            take_profit_price=104.0,
            realized_close=103.0,
            realized_high=105.0,
            realized_low=97.5,
        ),
        _candidate(
            "MSFT",
            forecasted_pnl=0.04,
            avg_return=0.02,
            last_close=100.0,
            entry_price=99.0,
            take_profit_price=103.0,
            realized_close=101.0,
            realized_high=103.5,
            realized_low=98.5,
        ),
    ]

    orders = build_wide_plan(candidates, account_equity=20_000.0, config=config)

    assert len(orders) == 2
    assert orders[0].reserved_notional == pytest.approx(10_000.0)
    assert orders[1].reserved_fraction_of_equity == pytest.approx(0.5)


def test_build_wide_plan_respects_candidate_allocation_override_and_cap():
    config = WidePlannerConfig(top_k=2, pair_notional_fraction=0.5, max_pair_notional_fraction=0.5, max_leverage=2.0)
    candidates = [
        _candidate(
            "AAPL",
            forecasted_pnl=0.05,
            avg_return=0.02,
            last_close=100.0,
            entry_price=98.0,
            take_profit_price=104.0,
            realized_close=103.0,
            realized_high=105.0,
            realized_low=97.5,
            allocation_fraction_of_equity=0.25,
        ),
        _candidate(
            "MSFT",
            forecasted_pnl=0.04,
            avg_return=0.02,
            last_close=100.0,
            entry_price=99.0,
            take_profit_price=103.0,
            realized_close=101.0,
            realized_high=103.5,
            realized_low=98.5,
            allocation_fraction_of_equity=0.90,
        ),
    ]

    orders = build_wide_plan(candidates, account_equity=20_000.0, config=config)

    assert orders[0].reserved_fraction_of_equity == pytest.approx(0.25)
    assert orders[0].reserved_notional == pytest.approx(5_000.0)
    assert orders[1].reserved_fraction_of_equity == pytest.approx(0.5)
    assert orders[1].reserved_notional == pytest.approx(10_000.0)


def test_validate_long_price_levels_rejects_flipped_exit_prices():
    with pytest.raises(ValueError, match="take_profit_price"):
        validate_long_price_levels(
            symbol="AAPL",
            entry_price=100.0,
            take_profit_price=99.0,
        )


def test_build_wide_plan_rejects_invalid_top_candidate():
    config = WidePlannerConfig(top_k=1)
    candidates = [
        _candidate(
            "AAPL",
            forecasted_pnl=0.05,
            avg_return=0.02,
            last_close=100.0,
            entry_price=100.0,
            take_profit_price=99.0,
            realized_close=101.0,
            realized_high=102.0,
            realized_low=98.0,
        ),
    ]

    with pytest.raises(ValueError, match="invalid long price plan"):
        build_wide_plan(candidates, account_equity=20_000.0, config=config)


def test_simulate_day_applies_work_steal_priority_and_leverage_cap():
    config = WidePlannerConfig(top_k=6, pair_notional_fraction=0.5, max_leverage=2.0, fee_bps=0.0, fill_buffer_bps=0.0)
    candidates = [
        _candidate("AAA", forecasted_pnl=0.09, avg_return=0.03, last_close=100.0, entry_price=99.0, take_profit_price=103.0, realized_close=102.0, realized_high=104.0, realized_low=98.5),
        _candidate("BBB", forecasted_pnl=0.08, avg_return=0.03, last_close=100.0, entry_price=98.0, take_profit_price=103.0, realized_close=102.0, realized_high=104.0, realized_low=97.5),
        _candidate("CCC", forecasted_pnl=0.07, avg_return=0.03, last_close=100.0, entry_price=97.0, take_profit_price=103.0, realized_close=102.0, realized_high=104.0, realized_low=96.5),
        _candidate("DDD", forecasted_pnl=0.06, avg_return=0.03, last_close=100.0, entry_price=96.0, take_profit_price=103.0, realized_close=102.0, realized_high=104.0, realized_low=95.5),
        _candidate("EEE", forecasted_pnl=0.05, avg_return=0.03, last_close=100.0, entry_price=95.0, take_profit_price=103.0, realized_close=102.0, realized_high=104.0, realized_low=94.5),
    ]

    result = simulate_day(candidates, account_equity=1_000.0, config=config)

    filled = [fill.order.candidate.symbol for fill in result.fills if fill.filled]
    assert filled == ["AAA", "BBB", "CCC", "DDD"]
    assert result.end_equity > result.start_equity


def test_simulate_wide_strategy_reports_summary_and_plan_text():
    config = WidePlannerConfig(top_k=2, pair_notional_fraction=0.5, max_leverage=2.0, fee_bps=10.0, fill_buffer_bps=5.0)
    day_one = [
        _candidate("AAPL", forecasted_pnl=0.05, avg_return=0.02, last_close=100.0, entry_price=98.0, take_profit_price=104.0, realized_close=103.0, realized_high=105.0, realized_low=97.0),
        _candidate("MSFT", forecasted_pnl=0.04, avg_return=0.02, last_close=100.0, entry_price=99.0, take_profit_price=102.0, realized_close=100.5, realized_high=101.0, realized_low=99.5),
    ]
    day_two = [
        _candidate("NVDA", forecasted_pnl=0.06, avg_return=0.03, last_close=100.0, entry_price=97.5, take_profit_price=104.0, realized_close=98.0, realized_high=99.0, realized_low=97.0),
    ]

    summary = simulate_wide_strategy([day_one, day_two], starting_equity=10_000.0, config=config)
    plan = build_wide_plan(day_one, account_equity=10_000.0, config=config)
    rendered = render_plan_text(plan, account_equity=10_000.0, config=config)

    assert summary.trade_count == 3
    assert summary.filled_count == 2
    assert math.isfinite(summary.monthly_return)
    assert "Wide plan for" in rendered
    assert "Watch activation=0.50%" in rendered
    assert "Deleverage companion" in rendered


def test_load_hourly_symbol_history_finds_nested_file(tmp_path: Path):
    hourly_root = tmp_path / "trainingdatahourly"
    nested = hourly_root / "stocks"
    nested.mkdir(parents=True)
    pd.DataFrame(
        [
            {"timestamp": "2024-03-04T14:30:00Z", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5},
            {"timestamp": "2024-03-04T15:30:00Z", "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5},
        ]
    ).to_csv(nested / "AAPL.csv", index=False)

    frame = load_hourly_symbol_history("AAPL", hourly_root)

    assert len(frame) == 2
    assert frame["timestamp"].iloc[0].isoformat() == "2024-03-04T14:30:00+00:00"


def test_simulate_intraday_day_uses_hourly_sequence_and_logs(tmp_path: Path):
    logger = WideRunLogger.create(tmp_path / "logs")
    config = WidePlannerConfig(top_k=2, pair_notional_fraction=0.5, max_leverage=1.0, fee_bps=0.0, fill_buffer_bps=5.0)
    candidates = [
        WideCandidate(
            symbol="AAPL",
            strategy="maxdiff",
            forecasted_pnl=0.05,
            avg_return=0.02,
            last_close=100.0,
            entry_price=99.0,
            take_profit_price=103.0,
            predicted_high=103.0,
            predicted_low=99.0,
            realized_close=101.0,
            realized_high=103.0,
            realized_low=99.0,
            score=0.05,
            day_index=0,
            session_date="2024-03-04",
            dollar_vol_20d=30_000_000.0,
        ),
        WideCandidate(
            symbol="MSFT",
            strategy="maxdiff",
            forecasted_pnl=0.04,
            avg_return=0.02,
            last_close=100.0,
            entry_price=98.0,
            take_profit_price=102.0,
            predicted_high=102.0,
            predicted_low=98.0,
            realized_close=101.0,
            realized_high=102.0,
            realized_low=98.0,
            score=0.04,
            day_index=0,
            session_date="2024-03-04",
            dollar_vol_20d=30_000_000.0,
        ),
    ]
    hourly_by_symbol = {
        "AAPL": pd.DataFrame(
            [
                {"timestamp": pd.Timestamp("2024-03-04T14:30:00Z"), "open": 100.0, "high": 102.0, "low": 98.5, "close": 101.0},
                {"timestamp": pd.Timestamp("2024-03-04T15:30:00Z"), "open": 101.0, "high": 103.5, "low": 100.5, "close": 103.0},
            ]
        ),
        "MSFT": pd.DataFrame(
            [
                {"timestamp": pd.Timestamp("2024-03-04T14:30:00Z"), "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0},
                {"timestamp": pd.Timestamp("2024-03-04T15:30:00Z"), "open": 100.0, "high": 100.5, "low": 97.5, "close": 98.5},
            ]
        ),
    }

    result = simulate_intraday_day(
        candidates,
        account_equity=1_000.0,
        hourly_by_symbol=hourly_by_symbol,
        config=config,
        logger=logger,
    )

    filled = [fill.order.candidate.symbol for fill in result.fills if fill.filled]
    assert filled == ["AAPL", "MSFT"]
    assert result.end_equity > result.start_equity
    assert (logger.run_dir / "central.log").exists()
    assert (logger.run_dir / "symbols" / "AAPL.log").exists()
    assert "filled at 99.0000" in (logger.run_dir / "symbols" / "AAPL.log").read_text()
    assert "penetration=5.0bp" in (logger.run_dir / "symbols" / "AAPL.log").read_text()


def test_simulate_intraday_day_requires_fill_through_buffer():
    config = WidePlannerConfig(top_k=1, pair_notional_fraction=0.5, max_leverage=1.0, fee_bps=0.0, fill_buffer_bps=5.0)
    candidate = WideCandidate(
        symbol="AAPL",
        strategy="maxdiff",
        forecasted_pnl=0.05,
        avg_return=0.02,
        last_close=100.0,
        entry_price=99.0,
        take_profit_price=103.0,
        predicted_high=103.0,
        predicted_low=99.0,
        realized_close=100.0,
        realized_high=103.0,
        realized_low=99.0,
        score=0.05,
        day_index=0,
        session_date="2024-03-04",
        dollar_vol_20d=30_000_000.0,
    )
    hourly_by_symbol = {
        "AAPL": pd.DataFrame(
            [{"timestamp": pd.Timestamp("2024-03-04T14:30:00Z"), "open": 100.0, "high": 101.0, "low": 98.97, "close": 100.0}]
        )
    }

    result = simulate_intraday_day([candidate], account_equity=1_000.0, hourly_by_symbol=hourly_by_symbol, config=config)

    assert not any(fill.filled for fill in result.fills)


def test_simulate_intraday_day_widens_buffer_for_low_volume_names():
    config = WidePlannerConfig(top_k=1, pair_notional_fraction=0.5, max_leverage=1.0, fee_bps=0.0, fill_buffer_bps=5.0)
    candidate = WideCandidate(
        symbol="AAVEUSD",
        strategy="maxdiff",
        forecasted_pnl=0.05,
        avg_return=0.02,
        last_close=100.0,
        entry_price=99.0,
        take_profit_price=103.0,
        predicted_high=103.0,
        predicted_low=99.0,
        realized_close=100.0,
        realized_high=103.0,
        realized_low=99.0,
        score=0.05,
        day_index=0,
        session_date="2024-03-04",
        dollar_vol_20d=400_000.0,
    )
    hourly_by_symbol = {
        "AAVEUSD": pd.DataFrame(
            [{"timestamp": pd.Timestamp("2024-03-04T14:30:00Z"), "open": 100.0, "high": 101.0, "low": 98.90, "close": 100.0}]
        )
    }

    result = simulate_intraday_day([candidate], account_equity=1_000.0, hourly_by_symbol=hourly_by_symbol, config=config)

    assert not any(fill.filled for fill in result.fills)


def test_simulate_intraday_day_work_steals_active_watch_when_cap_full(tmp_path: Path):
    logger = WideRunLogger.create(tmp_path / "logs")
    config = WidePlannerConfig(
        top_k=2,
        pair_notional_fraction=0.5,
        max_leverage=0.5,
        fee_bps=0.0,
        fill_buffer_bps=5.0,
        watch_activation_pct=0.005,
        steal_protection_pct=0.004,
    )
    candidates = [
        WideCandidate(
            symbol="AAA",
            strategy="maxdiff",
            forecasted_pnl=0.03,
            avg_return=0.01,
            last_close=100.0,
            entry_price=99.0,
            take_profit_price=103.0,
            predicted_high=103.0,
            predicted_low=99.0,
            realized_close=100.0,
            realized_high=101.0,
            realized_low=99.0,
            score=0.03,
            day_index=0,
            session_date="2024-03-04",
            dollar_vol_20d=30_000_000.0,
        ),
        WideCandidate(
            symbol="BBB",
            strategy="maxdiff",
            forecasted_pnl=0.06,
            avg_return=0.02,
            last_close=100.0,
            entry_price=98.0,
            take_profit_price=102.0,
            predicted_high=102.0,
            predicted_low=98.0,
            realized_close=100.0,
            realized_high=100.5,
            realized_low=98.0,
            score=0.06,
            day_index=0,
            session_date="2024-03-04",
            dollar_vol_20d=30_000_000.0,
        ),
    ]
    hourly_by_symbol = {
        "AAA": pd.DataFrame(
            [
                {"timestamp": pd.Timestamp("2024-03-04T14:30:00Z"), "open": 100.0, "high": 100.5, "low": 99.45, "close": 100.0},
                {"timestamp": pd.Timestamp("2024-03-04T15:30:00Z"), "open": 100.0, "high": 100.2, "low": 99.45, "close": 100.0},
            ]
        ),
        "BBB": pd.DataFrame(
            [
                {"timestamp": pd.Timestamp("2024-03-04T14:30:00Z"), "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0},
                {"timestamp": pd.Timestamp("2024-03-04T15:30:00Z"), "open": 100.0, "high": 100.3, "low": 98.45, "close": 99.0},
            ]
        ),
    }

    result = simulate_intraday_day(
        candidates,
        account_equity=1_000.0,
        hourly_by_symbol=hourly_by_symbol,
        config=config,
        logger=logger,
    )

    assert not any(fill.filled for fill in result.fills)
    central_log = (logger.run_dir / "central.log").read_text()
    assert "work steal" in central_log
    assert "watch canceled by work steal from BBB" in (logger.run_dir / "symbols" / "AAA.log").read_text()


def test_compute_backtest_frame_falls_back_when_primary_wrapper_fails(monkeypatch):
    fallback_frame = pd.DataFrame([{"close": 100.0, "predicted_high": 101.0, "predicted_low": 99.0}])

    def _boom(*args, **kwargs):
        raise RuntimeError("primary failed")

    monkeypatch.setattr(run_mod, "backtest_forecasts", _boom)
    monkeypatch.setattr(run_mod.fallback_backtest_module, "_fallback_backtest", lambda symbol, num_simulations=None: fallback_frame)

    result = run_mod._compute_backtest_frame("AAPL", num_simulations=5, model_override="chronos2")

    assert result is fallback_frame


def test_attach_realized_ohlc_falls_back_to_session_day_alignment(tmp_path: Path):
    data_root = tmp_path / "trainingdata"
    data_root.mkdir()
    pd.DataFrame(
        [
            {"timestamp": "2026-02-05T05:00:00Z", "open": 55.0, "high": 58.0, "low": 54.0, "close": 56.0},
        ]
    ).to_csv(data_root / "AA.csv", index=False)
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-02-05T20:00:00Z",
                "close": 56.37,
                "predicted_high": 57.2,
                "predicted_low": 56.21,
                "open": float("nan"),
                "high": float("nan"),
                "low": float("nan"),
            }
        ]
    )

    enriched = run_mod._attach_realized_ohlc(frame, "AA", data_root)

    assert "timestamp" in enriched.columns
    assert enriched.iloc[0]["open"] == pytest.approx(55.0)
    assert enriched.iloc[0]["high"] == pytest.approx(58.0)
    assert enriched.iloc[0]["low"] == pytest.approx(54.0)


def test_attach_realized_ohlc_falls_back_to_tail_alignment_when_dates_do_not_match(tmp_path: Path):
    data_root = tmp_path / "trainingdata"
    data_root.mkdir()
    pd.DataFrame(
        [
            {"timestamp": "2021-12-03T05:00:00Z", "open": 44.0, "high": 45.0, "low": 43.0, "close": 44.5},
            {"timestamp": "2021-12-06T05:00:00Z", "open": 45.0, "high": 46.0, "low": 44.0, "close": 45.5},
        ]
    ).to_csv(data_root / "AA.csv", index=False)
    frame = pd.DataFrame(
        [
            {"timestamp": "2026-02-05T20:00:00Z", "close": 56.37, "predicted_high": 57.2, "predicted_low": 56.21, "open": float("nan"), "high": float("nan"), "low": float("nan")},
            {"timestamp": "2026-02-05T19:00:00Z", "close": 57.06, "predicted_high": 57.2, "predicted_low": 56.21, "open": float("nan"), "high": float("nan"), "low": float("nan")},
        ]
    )

    enriched = run_mod._attach_realized_ohlc(frame, "AA", data_root)

    assert enriched.iloc[0]["open"] == pytest.approx(45.0)
    assert enriched.iloc[0]["high"] == pytest.approx(46.0)
    assert enriched.iloc[1]["open"] == pytest.approx(44.0)
    assert enriched.iloc[1]["low"] == pytest.approx(43.0)


def test_run_main_can_submit_plan_through_trading_server(monkeypatch, tmp_path: Path, capsys):
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2024-03-04T14:30:00Z",
                "close": 100.0,
                "high": 103.0,
                "low": 98.0,
                "predicted_high": 103.0,
                "predicted_low": 99.0,
                "maxdiffprofit_profit": 0.05,
                "maxdiff_avg_daily_return": 0.02,
                "maxdiffprofit_low_price": 99.0,
                "maxdiffprofit_high_price": 103.0,
            }
        ]
    )

    monkeypatch.setattr(run_mod, "load_backtests", lambda *args, **kwargs: {"AAPL": frame})
    monkeypatch.setattr(run_mod, "load_hourly_histories", lambda *args, **kwargs: {"AAPL": pd.DataFrame()})
    monkeypatch.setattr(
        run_mod,
        "simulate_intraday_day",
        lambda *args, **kwargs: type(
            "DayResult",
            (),
            {"end_equity": 10_000.0, "fills": tuple()},
        )(),
    )
    monkeypatch.setattr(run_mod, "release_model_resources", lambda force=True: None)

    submitted_payloads: list[dict[str, object]] = []

    class _FakeTradingServerClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def claim_writer(self, *, ttl_seconds=None):
            return {"account": "test-paper", "session_id": "s", "expires_at": "later"}

        def heartbeat_writer(self, *, ttl_seconds=None):
            return {"account": "test-paper", "session_id": "s", "expires_at": "later"}

        def submit_limit_order(self, **kwargs):
            submitted_payloads.append(kwargs)
            return {"order": kwargs, "quote": None, "filled": False}

    monkeypatch.setattr(run_mod, "TradingServerClient", _FakeTradingServerClient)

    exit_code = run_mod.main(
        [
            "--symbols",
            "AAPL",
            "--account-equity",
            "10000",
            "--backtest-days",
            "1",
            "--submit-plan",
            "--trading-account",
            "test-paper",
            "--trading-execution-mode",
            "live",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "submitted 1 entry orders via trading_server" in captured.out
    assert len(submitted_payloads) == 1
    assert submitted_payloads[0]["side"] == "buy"
    assert submitted_payloads[0]["live_ack"] == "LIVE"
