from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from stockagent.reporting import format_summary, load_state_snapshot, summarize_trades


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_summarize_trades_handles_basic_history(tmp_path: Path) -> None:
    suffix = "test"
    history = {
        "AAPL|buy": [
            {
                "pnl": 10.0,
                "qty": 1,
                "mode": "probe",
                "closed_at": datetime(2025, 1, 2, tzinfo=timezone.utc).isoformat(),
            },
            {
                "pnl": -5.0,
                "qty": 1,
                "mode": "normal",
                "closed_at": datetime(2025, 1, 3, tzinfo=timezone.utc).isoformat(),
            },
        ]
    }
    suffix_tag = f"_{suffix}"
    _write_json(tmp_path / f"trade_history{suffix_tag}.json", history)
    _write_json(tmp_path / f"trade_outcomes{suffix_tag}.json", {})
    _write_json(tmp_path / f"trade_learning{suffix_tag}.json", {})
    _write_json(tmp_path / f"active_trades{suffix_tag}.json", {})

    snapshot = load_state_snapshot(state_dir=tmp_path, state_suffix=suffix)
    summary = summarize_trades(snapshot=snapshot, directory=tmp_path, suffix=suffix)

    assert summary.total_trades == 2
    assert summary.total_pnl == 5.0
    assert summary.win_rate == 0.5
    assert summary.max_drawdown == 5.0

    output = format_summary(summary, label="unit-test")
    assert "unit-test" in output
    assert "Trades: 2" in output or "Closed trades: 2" in output
