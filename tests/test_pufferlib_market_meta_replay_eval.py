from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pufferlib_market.meta_replay_eval as module
from pufferlib_market.meta_replay_eval import (
    align_daily_returns_by_intersection,
    combine_actions_by_winners,
    summarize_winner_series,
)


def test_align_daily_returns_by_intersection_uses_common_days() -> None:
    a_idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True)
    b_idx = pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04"], utc=True)
    aligned = align_daily_returns_by_intersection(
        {
            "A": pd.Series([0.1, 0.2, 0.3], index=a_idx),
            "B": pd.Series([0.4, 0.5, 0.6], index=b_idx),
        }
    )

    expected_idx = pd.to_datetime(["2026-01-02", "2026-01-03"], utc=True)
    assert list(aligned.keys()) == ["A", "B"]
    assert aligned["A"].index.equals(expected_idx)
    assert aligned["B"].index.equals(expected_idx)
    assert aligned["A"].tolist() == [0.2, 0.3]
    assert aligned["B"].tolist() == [0.4, 0.5]


def test_combine_actions_by_winners_flattens_cash_days() -> None:
    decision_days = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True)
    winners = pd.Series(["A", None, "B"], index=decision_days)

    combined = combine_actions_by_winners(
        {
            "A": np.asarray([1, 1, 1], dtype=np.int32),
            "B": np.asarray([2, 2, 2], dtype=np.int32),
        },
        winners,
        decision_days=decision_days,
    )

    assert combined.tolist() == [1, 0, 2]


def test_summarize_winner_series_counts_switches_and_cash() -> None:
    winners = pd.Series(
        ["A", "A", None, "B", "B"],
        index=pd.to_datetime(
            ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
            utc=True,
        ),
    )

    summary = summarize_winner_series(winners)

    assert summary["switch_count"] == 2
    assert summary["winner_counts"] == {"A": 2, "B": 2, "cash": 1}
    assert summary["latest_winner"] == "B"


def test_main_writes_output_json_atomically(monkeypatch, tmp_path: Path, capsys) -> None:
    output_json = tmp_path / "meta_report.json"
    writes: list[tuple[Path, dict, dict]] = []

    monkeypatch.setattr(module, "run_meta_replay", lambda **_kwargs: {"b": 2, "a": 1})
    monkeypatch.setattr(
        module,
        "write_json_atomic",
        lambda path, payload, **kwargs: writes.append((Path(path), payload, kwargs)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "meta_replay_eval.py",
            "--checkpoint",
            str(tmp_path / "candidate.pt"),
            "--daily-data-path",
            str(tmp_path / "daily.bin"),
            "--hourly-data-root",
            str(tmp_path / "hourly"),
            "--start-date",
            "2026-01-01",
            "--end-date",
            "2026-01-03",
            "--max-steps",
            "2",
            "--cpu",
            "--output-json",
            str(output_json),
        ],
    )

    module.main()

    assert json.loads(capsys.readouterr().out) == {"a": 1, "b": 2}
    assert writes == [(output_json, {"b": 2, "a": 1}, {"sort_keys": True})]
    assert not output_json.exists()


def test_meta_replay_eval_uses_atomic_json_helper() -> None:
    source = Path("pufferlib_market/meta_replay_eval.py").read_text(encoding="utf-8")

    assert "write_json_atomic(" in source
    assert "write_text(" not in source
