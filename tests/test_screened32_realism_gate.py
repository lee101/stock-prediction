"""Smoke + invariant tests for scripts/screened32_realism_gate.py.

Realism gate is a deploy-blocking script: if it lies about cells, we ship
configs that fail at the live limit-order buffer. These tests exercise the
small pure-python helpers and end-to-end on a 2-window subset of the prod
val data.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "screened32_realism_gate.py"
VAL_PATH = REPO_ROOT / "pufferlib_market" / "data" / "screened32_single_offset_val_full.bin"


def test_monthly_from_total_compounds_correctly():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts.screened32_realism_gate import _monthly_from_total
    finally:
        sys.path.pop(0)

    # 21d total of +21% → +21%/mo (1d ≈ 1%, 21d ≈ 21%/mo via compounding)
    assert _monthly_from_total(0.21, 21) == pytest.approx(0.21, abs=1e-6)
    # 50d total of +21% → about (1.21)^(21/50) - 1 ≈ +8.6%/mo
    expected = math.expm1(math.log1p(0.21) * (21.0 / 50.0))
    assert _monthly_from_total(0.21, 50) == pytest.approx(expected, abs=1e-9)
    # zero / negative-days guard
    assert _monthly_from_total(0.5, 0) == 0.0
    # totally degenerate input does not raise
    assert _monthly_from_total(-2.0, 50) == 0.0


def test_percentile_handles_empty_and_basic_cases():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts.screened32_realism_gate import _percentile
    finally:
        sys.path.pop(0)
    assert _percentile([], 50) == 0.0
    assert _percentile([1.0, 2.0, 3.0], 50) == pytest.approx(2.0)
    assert _percentile([1.0, 2.0, 3.0], 10) == pytest.approx(1.2)


@pytest.mark.skipif(not VAL_PATH.exists(), reason="prod val data missing")
def test_realism_gate_smoke_runs_and_emits_artifacts(tmp_path: Path):
    """Run a 1×1 grid for 2 windows and assert the script produces a parseable JSON+MD.

    The 2-window cap keeps this test under 30s; we only check structural invariants,
    not the absolute return numbers (those are documented in alpacaprod.md and
    re-validated by the full sweep).
    """
    out_dir = tmp_path / "rg"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--val-data", str(VAL_PATH),
        "--window-days", "30",
        "--fill-buffer-bps-grid", "5",
        "--max-leverage-grid", "1.0",
        "--max-windows", "2",
        "--out-dir", str(out_dir),
        "--device", "cpu",
        "--decision-lag", "2",
    ]
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert r.returncode == 0, f"realism_gate exited {r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    json_path = out_dir / f"{VAL_PATH.stem}_realism_gate.json"
    md_path = out_dir / f"{VAL_PATH.stem}_realism_gate.md"
    assert json_path.exists(), f"missing {json_path}"
    assert md_path.exists(), f"missing {md_path}"
    payload = json.loads(json_path.read_text())
    assert payload["window_days"] == 30
    assert payload["decision_lag"] == 2
    assert payload["disable_shorts"] is True
    assert payload["fill_buffer_bps_grid"] == [5.0]
    assert payload["max_leverage_grid"] == [1.0]
    assert payload["n_windows_per_cell"] == 2
    assert payload["ensemble_size"] >= 2, "should have loaded the prod multi-model ensemble"
    cells = payload["cells"]
    assert len(cells) == 1
    cell = cells[0]
    assert cell["fill_buffer_bps"] == 5.0
    assert cell["max_leverage"] == 1.0
    assert cell["n_windows"] == 2
    # Total return must be a real number; sortino can be inf when there are no
    # negative returns in the window, but it must not be NaN.
    assert isinstance(cell["median_total_return"], float)
    assert not math.isnan(cell["median_total_return"])
    md_text = md_path.read_text()
    assert "Screened32 realism gate" in md_text
    assert "fill_bps" in md_text
