from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone

import pytest

from src.leverage_settings import LeverageSettings, reset_leverage_settings, set_leverage_settings


@pytest.fixture(autouse=True)
def leverage_override():
    set_leverage_settings(LeverageSettings())
    yield
    reset_leverage_settings()


@pytest.fixture
def risk_module(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "test_stock.db"))
    module = importlib.import_module("src.portfolio_risk")
    module = importlib.reload(module)
    yield module
    importlib.reload(module)


def test_global_risk_defaults_to_minimum(risk_module):
    risk_module.reset_cached_threshold()
    assert risk_module.get_global_risk_threshold() == pytest.approx(risk_module.DEFAULT_MIN_RISK_THRESHOLD)


def test_risk_threshold_updates_with_portfolio_performance(risk_module):
    risk_module.reset_cached_threshold()
    day1 = datetime(2025, 10, 13, 16, 0, tzinfo=timezone.utc)
    day2 = day1 + timedelta(days=1)
    day3 = day2 + timedelta(days=1)

    snap1 = risk_module.record_portfolio_snapshot(1000.0, observed_at=day1)
    assert snap1.risk_threshold == pytest.approx(risk_module.DEFAULT_MIN_RISK_THRESHOLD)

    snap2 = risk_module.record_portfolio_snapshot(1100.0, observed_at=day2)
    assert snap2.risk_threshold == pytest.approx(risk_module.get_configured_max_risk_threshold())

    snap3 = risk_module.record_portfolio_snapshot(900.0, observed_at=day3)
    assert snap3.risk_threshold == pytest.approx(risk_module.DEFAULT_MIN_RISK_THRESHOLD)


def test_fetch_snapshots_returns_ordered_records(risk_module):
    risk_module.reset_cached_threshold()
    start = datetime(2025, 10, 12, 14, 0, tzinfo=timezone.utc)
    for offset in range(3):
        risk_module.record_portfolio_snapshot(
            1000 + (offset * 50),
            observed_at=start + timedelta(days=offset),
        )

    snapshots = risk_module.fetch_snapshots()
    assert len(snapshots) == 3
    assert snapshots[0].portfolio_value < snapshots[-1].portfolio_value
    assert all(prev.observed_at <= curr.observed_at for prev, curr in zip(snapshots, snapshots[1:]))


def test_fetch_latest_snapshot_returns_most_recent(risk_module):
    risk_module.reset_cached_threshold()
    start = datetime(2025, 10, 12, 14, 0, tzinfo=timezone.utc)
    for offset in range(3):
        risk_module.record_portfolio_snapshot(
            1000 + (offset * 50),
            observed_at=start + timedelta(days=offset),
        )

    latest = risk_module.fetch_latest_snapshot()
    assert latest is not None
    expected_ts = start + timedelta(days=2)
    if latest.observed_at.tzinfo is None:
        latest_ts = latest.observed_at.replace(tzinfo=timezone.utc)
    else:
        latest_ts = latest.observed_at.astimezone(timezone.utc)
    assert latest_ts == expected_ts
    assert latest.portfolio_value == pytest.approx(1100)


def test_day_pl_overrides_reference_logic(risk_module):
    risk_module.reset_cached_threshold()
    day1 = datetime(2025, 10, 13, 14, 0, tzinfo=timezone.utc)
    day2 = day1 + timedelta(hours=1)

    snap1 = risk_module.record_portfolio_snapshot(1000.0, observed_at=day1, day_pl=-10.0)
    assert snap1.risk_threshold == pytest.approx(risk_module.DEFAULT_MIN_RISK_THRESHOLD)

    snap2 = risk_module.record_portfolio_snapshot(900.0, observed_at=day2, day_pl=25.0)
    assert snap2.risk_threshold == pytest.approx(risk_module.get_configured_max_risk_threshold())
