from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def risk_state_module(monkeypatch, tmp_path):
    suffix = f"riskstate_{datetime.now().timestamp()}"
    monkeypatch.setenv("TRADE_STATE_SUFFIX", suffix)
    module = importlib.import_module("src.risk_state")
    module = importlib.reload(module)
    yield module
    try:
        module.RISK_STATE_FILE.unlink()
    except FileNotFoundError:
        pass


def test_record_day_pl_schedules_probe_day(risk_state_module):
    day = datetime(2025, 11, 10, 21, 0, tzinfo=timezone.utc)
    risk_state_module.record_day_pl(-150.0, observed_at=day)

    next_day = day + timedelta(days=1)
    probe_state = risk_state_module.resolve_probe_state(next_day)
    assert probe_state.force_probe is True
    assert probe_state.reason.startswith("Previous day loss")

    # Positive day clears enforcement
    risk_state_module.record_day_pl(200.0, observed_at=next_day + timedelta(hours=6))
    cleared_state = risk_state_module.resolve_probe_state(next_day + timedelta(days=1))
    assert cleared_state.force_probe is False


def test_record_day_pl_no_probe_when_positive(risk_state_module):
    day = datetime(2025, 11, 10, 21, 0, tzinfo=timezone.utc)
    risk_state_module.record_day_pl(75.0, observed_at=day)
    probe_state = risk_state_module.resolve_probe_state(day + timedelta(days=1))
    assert probe_state.force_probe is False
