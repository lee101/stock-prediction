"""Smoke test for the fp4 multi-seed sweep harness (Phase 4, Unit P4-6).

Runs ``sweep.main(['--smoke','--algos','ppo'])`` and asserts the expected
output files exist and parse. Cells that don't produce metrics are OK as
long as the harness itself emits a summary + leaderboard -- the whole point
of the sweep is to tolerate trainer-level skips/errors without aborting.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from fp4.bench import sweep


def test_sweep_smoke_ppo_only(tmp_path: Path):
    out_dir = tmp_path / "sweep_out"
    rc = sweep.main([
        "--smoke",
        "--algos", "ppo",
        "--constrained", "off",
        "--out-dir", str(out_dir),
    ])
    assert rc == 0

    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "leaderboard.md"
    assert csv_path.exists(), f"summary.csv missing under {out_dir}"
    assert md_path.exists(), f"leaderboard.md missing under {out_dir}"

    # summary.csv parses and has the expected schema.
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1, "expected at least one cell row"
    required = {
        "algo", "constrained", "seed", "steps", "status",
        "median_5bps", "p10_5bps", "sortino_5bps", "max_dd_5bps",
    }
    assert required.issubset(rows[0].keys())
    assert rows[0]["algo"] == "ppo"
    assert rows[0]["constrained"] == "off"

    # leaderboard.md parses (is non-empty, has header + sort-key line).
    md = md_path.read_text()
    assert md.strip(), "leaderboard.md is empty"
    assert "fp4 sweep leaderboard" in md
    assert "p10@5bps" in md
    assert "sort key" in md


def test_extract_metrics_nested_eval():
    rec = {
        "eval": {"by_slippage": {"5": {"summary": {
            "median_total_return": 0.12,
            "p10_total_return": -0.03,
            "median_sortino": 1.8,
            "median_max_drawdown": -0.09,
        }}}}
    }
    m = sweep._extract_metrics(rec)
    assert m["median_5bps"] == pytest.approx(0.12)
    assert m["p10_5bps"] == pytest.approx(-0.03)
    assert m["sortino_5bps"] == pytest.approx(1.8)
    assert m["max_dd_5bps"] == pytest.approx(-0.09)


def test_extract_metrics_trainer_fallback():
    rec = {
        "train": {"trainer_output": {
            "final_sortino": 0.5,
            "final_p10": -1.2,
            "mean_return": 0.4,
        }}
    }
    m = sweep._extract_metrics(rec)
    assert m["sortino_5bps"] == pytest.approx(0.5)
    assert m["p10_5bps"] == pytest.approx(-1.2)
    assert m["median_5bps"] == pytest.approx(0.4)
