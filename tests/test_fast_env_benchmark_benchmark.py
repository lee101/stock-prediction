from __future__ import annotations

from pathlib import Path

import pytest
from analysis.fast_env_drift import BenchmarkRow, DriftError, load_rows, validate_rows
from scripts import fast_env_benchmark


def test_fast_env_benchmark_produces_outputs(monkeypatch, tmp_path: Path) -> None:
    # Use the python MarketEnv stub for fast env to avoid compiling the C++ extension in unit tests.
    class _StubFastEnv(fast_env_benchmark.MarketEnv):
        def __init__(self, *args, device: str | None = None, **kwargs):
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(fast_env_benchmark, "FastMarketEnv", _StubFastEnv)

    rows, meta = fast_env_benchmark.run_benchmark(num_steps=16, context_len=8, horizon=1, seed=7)
    metrics = {row["metric"] for row in rows}
    assert {"reward", "gross", "trading_cost", "financing_cost", "equity"}.issubset(metrics)
    assert "runtime_seconds" in metrics

    csv_path = tmp_path / "bench.csv"
    json_path = tmp_path / "bench.json"
    fast_env_benchmark._write_csv(rows, csv_path)
    fast_env_benchmark._write_json(meta, rows, json_path)

    loaded_rows = load_rows(csv_path)
    # Allow generous tolerances for the stubbed benchmark to ensure the validator passes.
    validate_rows(loaded_rows, max_abs_diff=1e-2, max_rel_diff=0.2, runtime_slack=1.0)


def test_fast_env_drift_detection() -> None:
    bad_rows = [
        BenchmarkRow(
            metric="reward",
            count=10,
            python_mean=1.0,
            fast_mean=1.5,
            python_std=0.1,
            fast_std=0.1,
            mean_delta=0.5,
            abs_diff_mean=0.4,
            rel_diff_mean=0.5,
            max_abs_diff=0.4,
        ),
        BenchmarkRow(
            metric="runtime_seconds",
            count=10,
            python_mean=1.0,
            fast_mean=2.0,
            python_std=0.0,
            fast_std=0.0,
            mean_delta=1.0,
            abs_diff_mean=1.0,
            rel_diff_mean=1.0,
            max_abs_diff=1.0,
        ),
    ]

    with pytest.raises(DriftError):
        validate_rows(bad_rows, max_abs_diff=1e-3, max_rel_diff=0.05, runtime_slack=0.1)
