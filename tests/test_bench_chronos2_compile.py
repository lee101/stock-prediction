from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from scripts.bench_chronos2_compile import (
    build_benchmark_command,
    compare_reports,
    report_passes_gate,
    resolve_modes,
)


def test_resolve_modes_includes_backend_and_compile_flags():
    modes = resolve_modes(["eager_fp32", "cute_compiled_fp32"])

    assert [mode.name for mode in modes] == ["eager_fp32", "cute_compiled_fp32"]
    assert modes[0].torch_compile is False
    assert modes[1].torch_compile is True
    assert modes[1].pipeline_backend == "cutechronos"


def test_resolve_modes_rejects_unknown_names():
    with pytest.raises(KeyError):
        resolve_modes(["not_a_real_mode"])


def test_build_benchmark_command_includes_precision_and_backend(tmp_path: Path):
    args = Namespace(
        context_lengths=[512],
        batch_sizes=[128],
        aggregations=["median"],
        sample_counts=[0],
        scalers=["none"],
        device_map="cuda",
        verbose=False,
    )
    mode = resolve_modes(["compiled_fp32"])[0]

    cmd = build_benchmark_command(args, symbol="AAPL", mode=mode, output_dir=tmp_path)

    assert "--torch-dtype" in cmd
    assert "float32" in cmd
    assert "--pipeline-backend" in cmd
    assert "chronos" in cmd
    assert "--torch-compile" in cmd


def test_compare_reports_and_gate_detect_positive_mae_regression():
    baseline = {
        "validation": {"price_mae": 1.0, "pct_return_mae": 0.10, "latency_s": 2.0},
        "test": {"price_mae": 1.2, "pct_return_mae": 0.12, "latency_s": 2.4},
    }
    candidate = {
        "validation": {"price_mae": 1.01, "pct_return_mae": 0.1005, "latency_s": 1.5},
        "test": {"price_mae": 1.212, "pct_return_mae": 0.1206, "latency_s": 1.2},
    }

    deltas = compare_reports(baseline, candidate)

    assert deltas["test_latency_speedup"] == pytest.approx(2.0)
    assert deltas["test_price_mae_delta_pct"] == pytest.approx(1.0)
    assert deltas["test_pct_return_mae_delta_pct"] == pytest.approx(0.5)
    assert report_passes_gate(deltas, max_price_mae_regression_pct=0.25, max_return_mae_regression_pct=0.25) is False
    assert report_passes_gate(deltas, max_price_mae_regression_pct=1.1, max_return_mae_regression_pct=0.6) is True
