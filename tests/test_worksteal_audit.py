from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binance_worksteal import sim_vs_live_audit
from binance_worksteal.strategy import WorkStealConfig


def test_audit_entries_keeps_pre_start_history_and_deduplicates():
    raw = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-03T00:00:00Z",
                "open": 90.0,
                "high": 91.0,
                "low": 85.0,
                "close": 90.0,
                "volume": 1000.0,
                "symbol": "BTCUSD",
            },
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000.0,
                "symbol": "BTCUSD",
            },
            {
                "timestamp": "2026-01-02T00:00:00Z",
                "open": 80.0,
                "high": 82.0,
                "low": 79.0,
                "close": 80.0,
                "volume": 1000.0,
                "symbol": "BTCUSD",
            },
            {
                "timestamp": "2026-01-02T00:00:00Z",
                "open": 110.0,
                "high": 112.0,
                "low": 109.0,
                "close": 110.0,
                "volume": 1200.0,
                "symbol": "BTCUSD",
            },
        ]
    )
    config = WorkStealConfig(
        dip_pct=0.10,
        proximity_pct=0.50,
        lookback_days=3,
        sma_filter_period=0,
    )

    audit_df = sim_vs_live_audit.audit_entries(
        {"BTCUSD": raw},
        config,
        start_date="2026-01-03",
        end_date="2026-01-03",
    )

    assert len(audit_df) == 1
    row = audit_df.iloc[0]
    assert row["date"] == pd.Timestamp("2026-01-03", tz="UTC")
    assert row["close"] == pytest.approx(90.0)
    assert row["ref_high"] == pytest.approx(112.0)
    assert row["buy_target"] == pytest.approx(100.8)


def test_audit_entries_does_not_mutate_input_frames():
    raw = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-03T00:00:00Z",
                "open": 90.0,
                "high": 91.0,
                "low": 85.0,
                "close": 90.0,
                "volume": 1000.0,
                "symbol": "BTCUSD",
            },
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000.0,
                "symbol": "BTCUSD",
            },
            {
                "timestamp": "2026-01-02T00:00:00Z",
                "open": 110.0,
                "high": 112.0,
                "low": 109.0,
                "close": 110.0,
                "volume": 1200.0,
                "symbol": "BTCUSD",
            },
        ]
    )
    bars = {"BTCUSD": raw}
    original_timestamps = list(raw["timestamp"])

    sim_vs_live_audit.audit_entries(
        bars,
        WorkStealConfig(dip_pct=0.10, proximity_pct=0.50, lookback_days=3, sma_filter_period=0),
        start_date="2026-01-03",
        end_date="2026-01-03",
    )

    assert bars["BTCUSD"] is raw
    assert list(raw["timestamp"]) == original_timestamps
    assert isinstance(raw.iloc[0]["timestamp"], str)


def test_run_comparison_uses_precomputed_summary(capsys, monkeypatch):
    def fail_build(*args, **kwargs):
        raise AssertionError("build_comparison_summary should not be called when comparison is provided")

    monkeypatch.setattr(sim_vs_live_audit, "build_comparison_summary", fail_build)

    sim_vs_live_audit.run_comparison(
        {},
        WorkStealConfig(),
        start_date="2026-01-01",
        end_date="2026-01-31",
        comparison={
            "default": {
                "total_return_pct": 1.1,
                "sortino": 1.2,
                "max_drawdown_pct": -2.0,
                "win_rate": 50.0,
                "entries_executed": 4,
                "candidates_generated": 8,
                "candidates_visible": 6,
                "fill_rate": 0.50,
                "visible_fill_rate": 0.66,
            },
            "realistic": {
                "total_return_pct": 0.8,
                "sortino": 0.9,
                "max_drawdown_pct": -1.5,
                "win_rate": 40.0,
                "entries_executed": 2,
                "candidates_generated": 8,
                "candidates_visible": 5,
                "fill_rate": 0.25,
                "visible_fill_rate": 0.40,
            },
        },
    )

    out = capsys.readouterr().out
    assert "BACKTEST COMPARISON: default vs realistic fill" in out
    assert "entries_executed" in out
    assert "fill_rate" in out


def test_audit_entries_reuses_input_mapping(monkeypatch):
    all_bars = {"BTCUSD": pd.DataFrame()}

    def fake_prepare(passed_bars, start_date=None, end_date=None):
        assert passed_bars is all_bars
        return {}, [], None

    monkeypatch.setattr(sim_vs_live_audit, "_prepare_backtest_symbol_data", fake_prepare)

    audit_df = sim_vs_live_audit.audit_entries(
        all_bars,
        WorkStealConfig(),
        start_date="2026-01-01",
        end_date="2026-01-02",
    )

    assert audit_df.empty


def test_build_comparison_summary_reuses_input_mapping(monkeypatch):
    all_bars = {"BTCUSD": pd.DataFrame()}
    seen = []

    def fake_backtest(passed_bars, config, start_date, end_date):
        seen.append(passed_bars is all_bars)
        return pd.DataFrame(), [], {}

    monkeypatch.setattr(sim_vs_live_audit, "run_worksteal_backtest", fake_backtest)

    sim_vs_live_audit.build_comparison_summary(
        all_bars,
        WorkStealConfig(),
        start_date="2026-01-01",
        end_date="2026-01-31",
    )

    assert seen == [True, True]


def test_build_comparison_summary_can_reuse_same_input_frames_without_mutation():
    raw = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price,
                "volume": 1000.0,
                "symbol": "BTCUSD",
            }
            for ts, price in zip(
                pd.date_range("2026-01-01", periods=35, tz="UTC", freq="D").strftime("%Y-%m-%dT%H:%M:%SZ"),
                [100.0 + i for i in range(35)],
            )
        ]
    ).iloc[::-1].reset_index(drop=True)
    bars = {"BTCUSD": raw}
    original_timestamps = list(raw["timestamp"])
    config = WorkStealConfig(
        dip_pct=0.10,
        proximity_pct=0.02,
        lookback_days=20,
        sma_filter_period=0,
        risk_off_trigger_sma_period=0,
        risk_off_trigger_momentum_period=0,
    )

    first = sim_vs_live_audit.build_comparison_summary(
        bars,
        config,
        start_date="2026-01-20",
        end_date="2026-01-31",
    )
    second = sim_vs_live_audit.build_comparison_summary(
        bars,
        config,
        start_date="2026-01-20",
        end_date="2026-01-31",
    )

    assert first == second
    assert bars["BTCUSD"] is raw
    assert list(raw["timestamp"]) == original_timestamps
    assert isinstance(raw.iloc[0]["timestamp"], str)


def test_audit_entries_resolves_entry_regime_once_per_day(monkeypatch):
    bars = {
        "BTCUSD": _make_daily_bars([100.0 + i for i in range(25)], "BTCUSD"),
        "ETHUSD": _make_daily_bars([120.0 + i for i in range(25)], "ETHUSD"),
    }
    counter = {"count": 0}
    original = sim_vs_live_audit.resolve_entry_regime

    def counted_regime(*args, **kwargs):
        counter["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(sim_vs_live_audit, "resolve_entry_regime", counted_regime)

    sim_vs_live_audit.audit_entries(
        bars,
        WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.02,
            lookback_days=20,
            sma_filter_period=0,
            risk_off_trigger_sma_period=0,
            risk_off_trigger_momentum_period=7,
        ),
    )

    assert counter["count"] == 25


def _make_daily_bars(prices: list[float], symbol: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": ts,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1000.0,
                "symbol": symbol,
            }
            for ts, price in zip(
                pd.date_range("2026-01-01", periods=len(prices), tz="UTC", freq="D").strftime("%Y-%m-%dT%H:%M:%SZ"),
                prices,
            )
        ]
    )


def test_audit_entries_marks_market_breadth_global_skip():
    bars = {
        "AAAUSD": _make_daily_bars([100.0, 100.0, 100.0, 100.0, 100.0, 90.0], "AAAUSD"),
        "BBBUSD": _make_daily_bars([100.0, 100.0, 100.0, 100.0, 100.0, 90.0], "BBBUSD"),
    }
    config = WorkStealConfig(
        dip_pct=0.10,
        proximity_pct=0.02,
        lookback_days=5,
        sma_filter_period=0,
        market_breadth_filter=0.5,
        risk_off_trigger_sma_period=0,
        risk_off_trigger_momentum_period=0,
    )

    audit_df = sim_vs_live_audit.audit_entries(
        bars,
        config,
        start_date="2026-01-06",
        end_date="2026-01-06",
    )

    assert len(audit_df) == 2
    assert audit_df["local_candidate"].tolist() == [True, True]
    assert audit_df["is_candidate"].tolist() == [False, False]
    assert audit_df["market_breadth_blocks"].tolist() == [True, True]
    assert audit_df["filter_reason"].tolist() == ["market_breadth", "market_breadth"]

    summary = sim_vs_live_audit.build_audit_summary(audit_df)
    assert summary["blocked_by_market_breadth"] == 2
    assert summary["candidate_count"] == 0


def test_audit_entries_marks_risk_off_global_skip():
    bars = {
        "AAAUSD": _make_daily_bars([110.0, 110.0, 110.0, 110.0, 100.0, 90.0], "AAAUSD"),
        "BBBUSD": _make_daily_bars([110.0, 110.0, 110.0, 110.0, 100.0, 90.0], "BBBUSD"),
    }
    config = WorkStealConfig(
        dip_pct=0.10,
        proximity_pct=0.02,
        lookback_days=5,
        sma_filter_period=0,
        market_breadth_filter=0.0,
        risk_off_trigger_sma_period=3,
        risk_off_trigger_momentum_period=0,
    )

    audit_df = sim_vs_live_audit.audit_entries(
        bars,
        config,
        start_date="2026-01-06",
        end_date="2026-01-06",
    )

    assert len(audit_df) == 2
    assert audit_df["local_candidate"].tolist() == [True, True]
    assert audit_df["is_candidate"].tolist() == [False, False]
    assert audit_df["risk_off_blocks"].tolist() == [True, True]
    assert audit_df["filter_reason"].tolist() == ["risk_off", "risk_off"]

    summary = sim_vs_live_audit.build_audit_summary(audit_df)
    assert summary["blocked_by_risk_off"] == 2
    assert summary["candidate_count"] == 0


def test_build_audit_summary_counts_effective_global_blocker_only():
    audit_df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-06", tz="UTC"),
                "symbol": "AAAUSD",
                "filter_reason": "market_breadth",
                "market_breadth_blocks": True,
                "risk_off_blocks": False,
                "is_candidate": False,
                "would_fill_realistic": False,
            },
            {
                "date": pd.Timestamp("2026-01-06", tz="UTC"),
                "symbol": "BBBUSD",
                "filter_reason": "base_asset",
                "market_breadth_blocks": True,
                "risk_off_blocks": False,
                "is_candidate": False,
                "would_fill_realistic": False,
            },
            {
                "date": pd.Timestamp("2026-01-06", tz="UTC"),
                "symbol": "CCCUSD",
                "filter_reason": "risk_off",
                "market_breadth_blocks": True,
                "risk_off_blocks": True,
                "is_candidate": False,
                "would_fill_realistic": False,
            },
        ]
    )

    summary = sim_vs_live_audit.build_audit_summary(audit_df)

    assert summary["blocked_by_market_breadth"] == 1
    assert summary["blocked_by_base_asset"] == 1
    assert summary["blocked_by_risk_off"] == 1


def test_build_audit_summary_falls_back_to_boolean_blockers_without_filter_reason():
    audit_df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-06", tz="UTC"),
                "symbol": "AAAUSD",
                "sma_blocks": True,
                "proximity_blocks": False,
                "momentum_blocks": False,
                "market_breadth_blocks": False,
                "risk_off_blocks": False,
                "is_candidate": False,
                "would_fill_realistic": False,
            },
            {
                "date": pd.Timestamp("2026-01-06", tz="UTC"),
                "symbol": "BBBUSD",
                "sma_blocks": False,
                "proximity_blocks": True,
                "momentum_blocks": False,
                "market_breadth_blocks": False,
                "risk_off_blocks": False,
                "is_candidate": False,
                "would_fill_realistic": False,
            },
            {
                "date": pd.Timestamp("2026-01-06", tz="UTC"),
                "symbol": "CCCUSD",
                "sma_blocks": False,
                "proximity_blocks": False,
                "momentum_blocks": True,
                "market_breadth_blocks": True,
                "risk_off_blocks": True,
                "is_candidate": False,
                "would_fill_realistic": False,
            },
        ]
    )

    summary = sim_vs_live_audit.build_audit_summary(audit_df)

    assert summary["blocked_by_sma"] == 1
    assert summary["blocked_by_proximity"] == 1
    assert summary["blocked_by_momentum"] == 1
    assert summary["blocked_by_market_breadth"] == 1
    assert summary["blocked_by_risk_off"] == 1
