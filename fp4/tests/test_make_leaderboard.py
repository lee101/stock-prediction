"""Tests for fp4/bench/make_leaderboard.py (Unit P4-8)."""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "bench"
sys.path.insert(0, str(BENCH))

import make_leaderboard  # noqa: E402


def _write_synthetic_csv(path: Path):
    fields = [
        "algorithm", "constrained", "seed", "status",
        "p10_5bps", "median_5bps", "sortino_5bps", "maxdd_5bps",
        "steps_per_sec", "gpu_peak_mb",
    ]
    rows = [
        # ppo unconstrained — modest
        {"algorithm": "ppo", "constrained": "off", "seed": 0, "status": "ok",
         "p10_5bps": 0.10, "median_5bps": 0.20, "sortino_5bps": 1.5,
         "maxdd_5bps": -0.15, "steps_per_sec": 5200, "gpu_peak_mb": 2048},
        {"algorithm": "ppo", "constrained": "off", "seed": 1, "status": "ok",
         "p10_5bps": 0.12, "median_5bps": 0.22, "sortino_5bps": 1.7,
         "maxdd_5bps": -0.17, "steps_per_sec": 5100, "gpu_peak_mb": 2050},
        # sac constrained — strongest p10
        {"algorithm": "sac", "constrained": "on", "seed": 0, "status": "ok",
         "p10_5bps": 0.30, "median_5bps": 0.45, "sortino_5bps": 2.8,
         "maxdd_5bps": -0.12, "steps_per_sec": 3200, "gpu_peak_mb": 3100},
        {"algorithm": "sac", "constrained": "on", "seed": 1, "status": "ok",
         "p10_5bps": 0.28, "median_5bps": 0.42, "sortino_5bps": 2.6,
         "maxdd_5bps": -0.14, "steps_per_sec": 3100, "gpu_peak_mb": 3150},
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_make_leaderboard_end_to_end(tmp_path):
    synthetic = tmp_path / "synthetic_summary.csv"
    _write_synthetic_csv(synthetic)
    out = tmp_path / "leaderboard.md"

    rc = make_leaderboard.main([
        "--summary", str(synthetic),
        "--out", str(out),
    ])
    assert rc == 0
    assert out.exists()

    md = out.read_text()
    # Report structure checks.
    assert "# Algorithm Leaderboard" in md
    assert "## Ranked leaderboard" in md
    assert "## Risk-adjusted ranking" in md
    assert "## Recommendation" in md
    # Both algos appear.
    assert "ppo" in md and "sac" in md
    # Recommendation line exists and names the winner (sac beats ppo on p10).
    rec_idx = md.index("## Recommendation")
    rec_block = md[rec_idx:]
    assert "**Recommendation:" in rec_block
    assert "sac" in rec_block  # sac should win the feasible ranking
    # Upstream commit cross-refs present.
    assert "f9806fe0" in md
    assert "d8eb1a5b" in md
    assert "3a1a29c2" in md
    # 66.2% ensemble bar referenced.
    assert "0.662" in md


def test_aggregate_drops_nonfinite(tmp_path):
    p = tmp_path / "s.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["algorithm", "constrained", "seed",
                                          "status", "p10_5bps", "median_5bps",
                                          "sortino_5bps", "maxdd_5bps"])
        w.writeheader()
        w.writerow({"algorithm": "ppo", "constrained": "off", "seed": 0,
                    "status": "ok", "p10_5bps": "nan", "median_5bps": 0.1,
                    "sortino_5bps": 1.0, "maxdd_5bps": -0.1})
        w.writerow({"algorithm": "ppo", "constrained": "off", "seed": 1,
                    "status": "ok", "p10_5bps": 0.2, "median_5bps": 0.1,
                    "sortino_5bps": 1.0, "maxdd_5bps": -0.1})
    rows = make_leaderboard.load_rows(p)
    agg = make_leaderboard.aggregate(rows)
    assert len(agg) == 1
    # mean of only the finite value
    assert abs(agg[0]["p10@5bps_mean"] - 0.2) < 1e-9
