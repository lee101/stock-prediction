from __future__ import annotations

import json
from functools import lru_cache
from datetime import datetime, timezone

import pytest

from stock import state as state_module
from stock import state_utils


def _install_temp_state_dir(monkeypatch: pytest.MonkeyPatch, tmp_path):
    state_module.get_state_dir.cache_clear()

    def _tmp_state_dir():
        return tmp_path

    monkeypatch.setattr(state_module, "get_state_dir", lru_cache(maxsize=1)(_tmp_state_dir))
    state_module.ensure_state_dir()


def test_collect_probe_statuses(monkeypatch: pytest.MonkeyPatch, tmp_path):
    _install_temp_state_dir(monkeypatch, tmp_path)
    monkeypatch.setenv("TRADE_STATE_SUFFIX", "test")

    paths = state_module.get_default_state_paths()
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    (paths["trade_learning"]).write_text(
        json.dumps(
            {
                "AAPL|buy": {
                    "pending_probe": True,
                    "probe_active": False,
                    "updated_at": "2025-01-02T00:00:00+00:00",
                }
            }
        )
    )
    (paths["trade_outcomes"]).write_text(
        json.dumps(
            {
                "AAPL|buy": {
                    "pnl": 42.5,
                    "reason": "profit_target",
                    "closed_at": "2025-01-01T00:00:00+00:00",
                }
            }
        )
    )
    (paths["active_trades"]).write_text(
        json.dumps(
            {
                "AAPL|buy": {
                    "mode": "probe",
                    "qty": 1.0,
                    "opened_at": "2025-01-03T00:00:00+00:00",
                }
            }
        )
    )
    (paths["trade_history"]).write_text(json.dumps({}))

    statuses = state_utils.collect_probe_statuses()
    assert len(statuses) == 1
    status = statuses[0]
    assert status.symbol == "AAPL"
    assert status.pending_probe is True
    assert status.active_mode == "probe"
    assert status.last_pnl == pytest.approx(42.5)
    assert status.last_closed_at == datetime(2025, 1, 1, tzinfo=timezone.utc)


def test_render_ascii_line_downsamples():
    values = list(range(100))
    ascii_lines = state_utils.render_ascii_line(values, width=10)
    assert len(ascii_lines) == 1
    assert len(ascii_lines[0]) == 10
