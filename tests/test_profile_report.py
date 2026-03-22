"""Tests for pufferlib_market.profile_training.generate_markdown_report
and tools/profile_report.py CLI.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pufferlib_market.profile_training import (
    generate_markdown_report,
    _parse_chrome_trace_top_kernels,
    _parse_speedscope_top_functions,
    _parse_gprof,
)

# ---------------------------------------------------------------------------
# Fixtures: minimal mock profile data
# ---------------------------------------------------------------------------

MOCK_TRACE = {
    "schemaVersion": 1,
    "deviceProperties": [{"id": 0, "name": "NVIDIA Test GPU", "totalGlobalMem": 8000000000}],
    "traceEvents": [
        # Two kernel events: sgemm (dominant) and elementwise
        {
            "ph": "X", "cat": "kernel",
            "name": "void cutlass::sgemm_64x64()",
            "pid": 0, "tid": 7, "ts": 1000.0, "dur": 500.0,
            "args": {},
        },
        {
            "ph": "X", "cat": "kernel",
            "name": "void cutlass::sgemm_64x64()",
            "pid": 0, "tid": 7, "ts": 2000.0, "dur": 300.0,
            "args": {},
        },
        {
            "ph": "X", "cat": "kernel",
            "name": "void at::native::elementwise_kernel()",
            "pid": 0, "tid": 7, "ts": 3000.0, "dur": 100.0,
            "args": {},
        },
        # A CPU op (should not appear in kernel list)
        {
            "ph": "X", "cat": "cpu_op",
            "name": "aten::linear",
            "pid": 1, "tid": 1, "ts": 1000.0, "dur": 200.0,
        },
    ],
}

MOCK_SPEEDSCOPE = {
    "$schema": "https://www.speedscope.app/file-format-schema.json",
    "profiles": [
        {
            "type": "sampled",
            "name": "MainThread",
            "unit": "seconds",
            "startValue": 0.0,
            "endValue": 10.0,
            "samples": [
                [0, 1, 2],   # ppo_update -> _run_ppo -> optimizer.step
                [0, 1, 2],
                [0, 1, 3],   # ppo_update -> _run_ppo -> loss.backward
                [0, 1, 2],
                [0, 4],      # ppo_update -> collect_rollout
            ],
            "weights": [0.01, 0.01, 0.01, 0.01, 0.01],
        }
    ],
    "shared": {
        "frames": [
            {"name": "ppo_update", "file": "pufferlib_market/train.py", "line": 100},
            {"name": "_run_ppo", "file": "pufferlib_market/train.py", "line": 200},
            {"name": "optimizer.step", "file": "torch/optim/adamw.py", "line": 50},
            {"name": "loss.backward", "file": "torch/autograd/__init__.py", "line": 30},
            {"name": "collect_rollout", "file": "pufferlib_market/train.py", "line": 150},
        ]
    },
    "activeProfileIndex": 0,
    "exporter": "py-spy@0.3.14",
    "name": "profile",
}

MOCK_TIMING = {
    "steps_per_sec": 47200.0,
    "updates_per_sec": 10.6,
    "wall_time_s": 94.3,
    "n_updates": 1000,
    "gpu_name": "NVIDIA Test GPU",
    "gpu_vram_gb": 8.0,
}

MOCK_GPROF = """\
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total
 time   seconds   seconds    calls  ms/call  ms/call  name
 45.23      1.20     1.20      500     2.40     2.40  TradingEnv_step
 20.11      1.73     0.53     1000     0.53     0.53  compute_reward
 10.05      2.00     0.27      500     0.54     0.54  reset_episode
"""


# ---------------------------------------------------------------------------
# Unit tests: individual parsers
# ---------------------------------------------------------------------------

def test_parse_chrome_trace_top_kernels() -> None:
    """Top kernels should be sorted by total ms, correct aggregation."""
    with tempfile.TemporaryDirectory() as td:
        trace_path = Path(td) / "trace.json"
        trace_path.write_text(json.dumps(MOCK_TRACE))

        result = _parse_chrome_trace_top_kernels(trace_path, top_n=10)
        assert isinstance(result, tuple), "Expected (kernels, total_ms) tuple"
        kernels, total_ms = result

        # sgemm: 500+300=800us -> 0.8ms; elementwise: 100us -> 0.1ms
        assert len(kernels) == 2, f"Expected 2 kernels, got {len(kernels)}"
        names = [k[0] for k in kernels]
        assert "void cutlass::sgemm_64x64()" == names[0], "sgemm should be top kernel"
        assert abs(kernels[0][1] - 0.8) < 0.01, f"sgemm total ms should be ~0.8, got {kernels[0][1]}"
        assert kernels[0][2] == 2, f"sgemm call count should be 2, got {kernels[0][2]}"
        assert abs(total_ms - 0.9) < 0.01, f"Total ms should be ~0.9, got {total_ms}"


def test_parse_chrome_trace_missing_file() -> None:
    """Missing trace file should return ([], 1.0) without crashing."""
    kernels, total_ms = _parse_chrome_trace_top_kernels(Path("/nonexistent/trace.json"))
    assert kernels == [], f"Expected empty kernel list, got {kernels}"
    assert total_ms == 1.0


def test_parse_speedscope_top_functions() -> None:
    """Top functions should exclude bootstrap frames and rank by sample count."""
    with tempfile.TemporaryDirectory() as td:
        sg_path = Path(td) / "flamegraph.svg"
        sg_path.write_text(json.dumps(MOCK_SPEEDSCOPE))

        result = _parse_speedscope_top_functions(sg_path, top_n=5)
        assert len(result) > 0, "Should return at least one function"
        # ppo_update should appear (5 samples = 100%)
        names = [r[0] for r in result]
        assert "ppo_update" in names, f"ppo_update not found in {names}"


def test_parse_speedscope_missing_file() -> None:
    """Missing speedscope file should return empty list."""
    result = _parse_speedscope_top_functions(Path("/nonexistent/flamegraph.svg"))
    assert result == []


def test_parse_gprof() -> None:
    """gprof parser should extract function names and percentages."""
    with tempfile.TemporaryDirectory() as td:
        gprof_path = Path(td) / "gprof_output.txt"
        gprof_path.write_text(MOCK_GPROF)
        result = _parse_gprof(gprof_path)
        assert len(result) >= 2
        func_names = [r[0] for r in result]
        assert "TradingEnv_step" in func_names
        assert "compute_reward" in func_names


def test_parse_gprof_missing_file() -> None:
    """Missing gprof file should return empty list."""
    result = _parse_gprof(Path("/nonexistent/gprof.txt"))
    assert result == []


# ---------------------------------------------------------------------------
# Integration test: generate_markdown_report with mock files
# ---------------------------------------------------------------------------

def test_generate_markdown_report_creates_file() -> None:
    """generate_markdown_report should create report.md with expected sections."""
    with tempfile.TemporaryDirectory() as td:
        profiles_dir = Path(td)

        # Write mock profile files
        (profiles_dir / "pufferlib_cuda_trace.json").write_text(json.dumps(MOCK_TRACE))
        (profiles_dir / "pufferlib_flamegraph.svg").write_text(json.dumps(MOCK_SPEEDSCOPE))
        (profiles_dir / "timing.json").write_text(json.dumps(MOCK_TIMING))
        (profiles_dir / "gprof_output.txt").write_text(MOCK_GPROF)

        report_path = generate_markdown_report(profiles_dir)

        assert report_path.exists(), "report.md was not created"
        content = report_path.read_text()

        # Check all required sections are present
        assert "# Training Profile Report" in content
        assert "## Throughput Summary" in content
        assert "## Top CPU Functions (py-spy)" in content
        assert "## Top CUDA Kernels (torch.profiler)" in content
        assert "## Memory Allocation Hotspots" in content
        assert "## C Env (gprof)" in content
        assert "## Optimization Recommendations" in content


def test_generate_markdown_report_throughput_data() -> None:
    """Throughput stats from timing.json should appear in report."""
    with tempfile.TemporaryDirectory() as td:
        profiles_dir = Path(td)
        (profiles_dir / "timing.json").write_text(json.dumps(MOCK_TIMING))

        report_path = generate_markdown_report(profiles_dir)
        content = report_path.read_text()

        assert "47200" in content, "steps_per_sec not in report"
        assert "NVIDIA Test GPU" in content, "gpu_name not in report"
        assert "8.0" in content, "gpu_vram not in report"


def test_generate_markdown_report_cuda_kernels() -> None:
    """Top CUDA kernels table should be populated from trace file."""
    with tempfile.TemporaryDirectory() as td:
        profiles_dir = Path(td)
        (profiles_dir / "pufferlib_cuda_trace.json").write_text(json.dumps(MOCK_TRACE))

        report_path = generate_markdown_report(profiles_dir)
        content = report_path.read_text()

        assert "sgemm_64x64" in content, "sgemm kernel should appear in report"
        assert "elementwise_kernel" in content, "elementwise kernel should appear in report"


def test_generate_markdown_report_cpu_functions() -> None:
    """Top CPU functions from speedscope should appear in report."""
    with tempfile.TemporaryDirectory() as td:
        profiles_dir = Path(td)
        (profiles_dir / "pufferlib_flamegraph.svg").write_text(json.dumps(MOCK_SPEEDSCOPE))

        report_path = generate_markdown_report(profiles_dir)
        content = report_path.read_text()

        assert "ppo_update" in content, "ppo_update function should appear in report"


def test_generate_markdown_report_gprof() -> None:
    """C env gprof section should be populated when gprof_output.txt is present."""
    with tempfile.TemporaryDirectory() as td:
        profiles_dir = Path(td)
        (profiles_dir / "gprof_output.txt").write_text(MOCK_GPROF)

        report_path = generate_markdown_report(profiles_dir)
        content = report_path.read_text()

        assert "TradingEnv_step" in content, "TradingEnv_step should appear in gprof section"


def test_generate_markdown_report_no_files() -> None:
    """Report should still be created (with placeholder messages) when no profile files exist."""
    with tempfile.TemporaryDirectory() as td:
        profiles_dir = Path(td)
        report_path = generate_markdown_report(profiles_dir)
        assert report_path.exists()
        content = report_path.read_text()
        assert "# Training Profile Report" in content
        # Should have placeholder messages, not crash
        assert "No" in content or "not found" in content or "N/A" in content


def test_generate_markdown_report_returns_path() -> None:
    """generate_markdown_report must return a Path to report.md."""
    with tempfile.TemporaryDirectory() as td:
        profiles_dir = Path(td)
        result = generate_markdown_report(profiles_dir)
        assert isinstance(result, Path)
        assert result.name == "report.md"
        assert result.parent == profiles_dir


# ---------------------------------------------------------------------------
# CLI tests: tools/profile_report.py
# ---------------------------------------------------------------------------

PROFILE_REPORT_SCRIPT = PROJECT_ROOT / "tools" / "profile_report.py"


def test_profile_report_cli_help() -> None:
    """python tools/profile_report.py --help should exit 0."""
    result = subprocess.run(
        [sys.executable, str(PROFILE_REPORT_SCRIPT), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"--help failed:\n{result.stderr}"
    assert "profiles-dir" in result.stdout or "profiles_dir" in result.stdout


def test_profile_report_cli_nonexistent_dir() -> None:
    """CLI with nonexistent dir should print 'No profile files found' and exit 0."""
    result = subprocess.run(
        [sys.executable, str(PROFILE_REPORT_SCRIPT), "--profiles-dir", "/nonexistent/path/xyz"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Expected exit 0, got {result.returncode}"
    assert "No profile files found" in result.stdout, (
        f"Expected 'No profile files found' in output:\n{result.stdout}"
    )


def test_profile_report_cli_generates_report() -> None:
    """CLI should generate report.md when profile files are present."""
    with tempfile.TemporaryDirectory() as td:
        profiles_dir = Path(td)
        (profiles_dir / "pufferlib_cuda_trace.json").write_text(json.dumps(MOCK_TRACE))
        (profiles_dir / "timing.json").write_text(json.dumps(MOCK_TIMING))

        result = subprocess.run(
            [sys.executable, str(PROFILE_REPORT_SCRIPT), "--profiles-dir", str(profiles_dir)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert "Report written to" in result.stdout
        assert (profiles_dir / "report.md").exists()


# ---------------------------------------------------------------------------
# Module API: profile_training.py exports
# ---------------------------------------------------------------------------

def test_profile_training_module_exports() -> None:
    """profile_training.py must export generate_markdown_report and new flags in main."""
    import pufferlib_market.profile_training as mod
    assert hasattr(mod, "generate_markdown_report"), "Missing generate_markdown_report"
    assert hasattr(mod, "_parse_chrome_trace_top_kernels"), "Missing _parse_chrome_trace_top_kernels"
    assert hasattr(mod, "_parse_speedscope_top_functions"), "Missing _parse_speedscope_top_functions"
    assert hasattr(mod, "_parse_gprof"), "Missing _parse_gprof"
    assert hasattr(mod, "run_pufferlib_profiling"), "Missing run_pufferlib_profiling"
    assert hasattr(mod, "main"), "Missing main"
