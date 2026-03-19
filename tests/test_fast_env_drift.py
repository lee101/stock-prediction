from __future__ import annotations

import csv
from pathlib import Path

import pytest
from analysis.fast_env_drift import DriftThresholds, evaluate_drift


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "timestamp",
        "symbol",
        "backend",
        "steps",
        "total_time_s",
        "avg_step_ms",
        "reward_sum",
        "equity_final",
        "position_final",
        "reward_mae",
        "reward_max",
        "equity_mae",
        "equity_max",
        "obs_max",
        "speedup",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_fast_env_drift_passes_within_thresholds(tmp_path: Path) -> None:
    csv_path = tmp_path / "bench.csv"
    _write_csv(
        csv_path,
        [
            {
                "timestamp": "2025-03-19T00:00:00Z",
                "symbol": "AAPL",
                "backend": "python",
                "steps": "128",
                "avg_step_ms": "0.80",
                "reward_sum": "1.23",
                "equity_final": "101.0",
            },
            {
                "timestamp": "2025-03-19T00:00:00Z",
                "symbol": "AAPL",
                "backend": "fast",
                "steps": "128",
                "avg_step_ms": "0.40",
                "reward_sum": "1.24",
                "equity_final": "101.1",
            },
            {
                "timestamp": "2025-03-19T00:00:00Z",
                "symbol": "AAPL",
                "backend": "delta",
                "steps": "128",
                "reward_mae": "1e-06",
                "reward_max": "2e-06",
                "equity_mae": "1e-06",
                "equity_max": "3e-06",
                "obs_max": "4e-06",
                "speedup": "2.0",
            },
        ],
    )
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    thresholds = DriftThresholds(
        max_reward_diff=1e-4,
        max_equity_diff=1e-4,
        max_obs_diff=1e-4,
        min_speedup=1.1,
    )
    errors, warnings, summaries = evaluate_drift(rows, thresholds=thresholds)
    assert errors == []
    assert warnings == []
    assert summaries["AAPL"].speedup == pytest.approx(2.0)


def test_fast_env_drift_reports_errors(tmp_path: Path) -> None:
    csv_path = tmp_path / "bench.csv"
    _write_csv(
        csv_path,
        [
            {
                "timestamp": "2025-03-19T00:00:00Z",
                "symbol": "MSFT",
                "backend": "python",
                "steps": "64",
                "avg_step_ms": "0.50",
            },
            {
                "timestamp": "2025-03-19T00:00:00Z",
                "symbol": "MSFT",
                "backend": "fast",
                "steps": "64",
                "avg_step_ms": "0.55",
            },
            {
                "timestamp": "2025-03-19T00:00:00Z",
                "symbol": "MSFT",
                "backend": "delta",
                "steps": "64",
                "reward_mae": "5e-03",
                "reward_max": "6e-03",
                "equity_mae": "1e-02",
                "equity_max": "2e-02",
                "obs_max": "1e-02",
                "speedup": "0.5",
            },
        ],
    )
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    thresholds = DriftThresholds(
        max_reward_diff=1e-4,
        max_equity_diff=1e-4,
        max_obs_diff=1e-4,
        min_speedup=0.9,
    )
    errors, warnings, _ = evaluate_drift(rows, thresholds=thresholds)
    assert any("reward delta" in msg for msg in errors)
    assert any("speedup" in msg for msg in errors)
    assert warnings == []
