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
from scripts import screened32_realism_gate as mod
from scripts.screened32_realism_gate import (
    CellResult,
    _can_use_gpu_path,
    _monthly_from_total,
    _parse_float_grid,
    _percentile,
    _promotion_gate,
    _write_text_atomic,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "screened32_realism_gate.py"
VAL_PATH = REPO_ROOT / "pufferlib_market" / "data" / "screened32_single_offset_val_full.bin"


def test_monthly_from_total_compounds_correctly():
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
    assert _percentile([], 50) == 0.0
    assert _percentile([1.0, 2.0, 3.0], 50) == pytest.approx(2.0)
    assert _percentile([1.0, 2.0, 3.0], 10) == pytest.approx(1.2)


def test_parse_float_grid_rejects_non_finite_duplicate_and_invalid_values():
    assert _parse_float_grid("0,5,10", name="fill_buffer_bps_grid", min_value=0.0) == [
        0.0,
        5.0,
        10.0,
    ]
    with pytest.raises(ValueError, match="finite"):
        _parse_float_grid("5,nan", name="fill_buffer_bps_grid", min_value=0.0)
    with pytest.raises(ValueError, match="duplicate"):
        _parse_float_grid("5,5", name="fill_buffer_bps_grid", min_value=0.0)
    with pytest.raises(ValueError, match=">= 0"):
        _parse_float_grid("-1,5", name="fill_buffer_bps_grid", min_value=0.0)
    with pytest.raises(ValueError, match="> 0"):
        _parse_float_grid("0,1.5", name="max_leverage_grid", strictly_positive=True)
    with pytest.raises(ValueError, match="at least one"):
        _parse_float_grid(" , ", name="fill_buffer_bps_grid", min_value=0.0)


def test_write_text_atomic_creates_parent_and_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "gate.json"

    _write_text_atomic(path, "old")
    _write_text_atomic(path, "new")

    assert path.read_text(encoding="utf-8") == "new"
    assert not list(path.parent.glob(".*.tmp"))


def test_write_text_atomic_cleans_temp_file_on_replace_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "gate.json"
    original_replace = Path.replace

    def fail_replace(self: Path, target: Path) -> Path:
        if self.name.startswith(".gate.json."):
            raise OSError("simulated replace failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        _write_text_atomic(path, "body")

    assert not path.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_gpu_path_is_disabled_when_borrow_apr_is_charged(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    assert (
        _can_use_gpu_path(
            decision_lag=2,
            disable_shorts=True,
            deterministic=True,
            alloc_bins=1,
            level_bins=1,
            action_max_offset_bps=0.0,
            ensemble_mode="softmax_avg",
            short_borrow_apr=0.0625,
        )
        is False
    )
    assert (
        _can_use_gpu_path(
            decision_lag=2,
            disable_shorts=True,
            deterministic=True,
            alloc_bins=1,
            level_bins=1,
            action_max_offset_bps=0.0,
            ensemble_mode="softmax_avg",
            short_borrow_apr=0.0,
        )
        is True
    )


def test_promotion_gate_binds_worst_cell_and_target() -> None:
    good = CellResult(
        slippage_bps=0.0,
        fill_buffer_bps=5.0,
        max_leverage=1.0,
        median_total_return=0.50,
        p10_total_return=0.20,
        p90_total_return=0.70,
        median_monthly_return=0.30,
        p10_monthly_return=0.12,
        median_sortino=4.0,
        median_max_dd=0.10,
        n_neg=0,
        n_windows=30,
    )
    bad = CellResult(
        slippage_bps=20.0,
        fill_buffer_bps=5.0,
        max_leverage=1.0,
        median_total_return=0.10,
        p10_total_return=-0.02,
        p90_total_return=0.20,
        median_monthly_return=0.20,
        p10_monthly_return=-0.01,
        median_sortino=1.0,
        median_max_dd=0.18,
        n_neg=3,
        n_windows=30,
    )

    gate = _promotion_gate([good, bad], monthly_target=0.27)

    assert gate["passed"] is False
    assert gate["enforced"] is True
    assert gate["worst_median_monthly_return"] == pytest.approx(0.20)
    assert gate["worst_cell"]["slippage_bps"] == 20.0
    assert gate["failures"] == ["worst_median_monthly_return 0.2 < target 0.27"]

    relaxed = _promotion_gate([good], monthly_target=0.27, enforce=False)
    assert relaxed["passed"] is True
    assert relaxed["enforced"] is False


def test_realism_gate_returns_3_after_writing_artifacts_when_enforced_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    val_path = tmp_path / "tiny.bin"
    val_path.write_bytes(b"stub")

    class FakeData:
        num_symbols = 1
        num_timesteps = 2
        features = type("Features", (), {"shape": (2, 1, 1)})()

    monkeypatch.setattr(mod, "read_mktd", lambda path: FakeData())
    monkeypatch.setattr(
        mod,
        "_run_cell",
        lambda **kwargs: mod.CellResult(
            slippage_bps=float(kwargs["slippage_bps"]),
            fill_buffer_bps=float(kwargs["fill_buffer_bps"]),
            max_leverage=float(kwargs["max_leverage"]),
            median_total_return=0.0,
            p10_total_return=0.0,
            p90_total_return=0.0,
            median_monthly_return=0.0,
            p10_monthly_return=0.0,
            median_sortino=0.0,
            median_max_dd=0.0,
            n_neg=0,
            n_windows=1,
        ),
    )
    out_dir = tmp_path / "out"

    rc = mod.main(
        [
            "--val-data",
            str(val_path),
            "--window-days",
            "1",
            "--fill-buffer-bps-grid",
            "5",
            "--max-leverage-grid",
            "1",
            "--slippage-bps",
            "5",
            "--monthly-target",
            "0.27",
            "--checkpoints",
            "AGENTS.md",
            "--out-dir",
            str(out_dir),
            "--device",
            "cpu",
        ]
    )

    assert rc == 3
    payload = json.loads((out_dir / "tiny_realism_gate.json").read_text())
    assert payload["promotion_gate"]["passed"] is False
    assert payload["promotion_gate"]["enforced"] is True
    assert payload["promotion_gate"]["failures"] == [
        "worst_median_monthly_return 0 < target 0.27"
    ]
    assert (out_dir / "tiny_realism_gate.md").exists()


@pytest.mark.parametrize(
    "args",
    [
        ["--fill-buffer-bps-grid", "nan"],
        ["--fill-buffer-bps-grid", "5,5"],
        ["--slippage-bps-grid", "5,nan"],
        ["--slippage-bps-grid", "5,5"],
        ["--max-leverage-grid", "0"],
        ["--fee-rate", "-0.001"],
        ["--slippage-bps", "inf"],
        ["--short-borrow-apr", "-0.1"],
        ["--decision-lag", "-1"],
        ["--window-days", "0"],
        ["--max-windows", "0"],
    ],
)
def test_realism_gate_rejects_invalid_numeric_inputs_before_data_load(
    tmp_path: Path,
    args: list[str],
) -> None:
    missing_val = tmp_path / "missing.bin"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--val-data",
        str(missing_val),
        "--out-dir",
        str(tmp_path / "out"),
        *args,
    ]

    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)

    assert result.returncode == 2
    assert "realism_gate:" in result.stderr
    assert "val data not found" not in result.stderr


@pytest.mark.skipif(not VAL_PATH.exists(), reason="prod val data missing")
def test_realism_gate_smoke_runs_and_emits_artifacts(tmp_path: Path):
    """Run a 4x1x1 grid for 1 window and assert the script produces parseable JSON+MD.

    The 1-window cap keeps this test fast while still exercising the default
    production slippage grid.
    """
    out_dir = tmp_path / "rg"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--val-data", str(VAL_PATH),
        "--window-days", "30",
        "--fill-buffer-bps-grid", "5",
        "--max-leverage-grid", "1.0",
        "--max-windows", "1",
        "--no-enforce-gate",
        "--out-dir", str(out_dir),
        "--device", "cpu",
        "--decision-lag", "2",
    ]
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    assert r.returncode == 0, f"realism_gate exited {r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    json_path = out_dir / f"{VAL_PATH.stem}_realism_gate.json"
    md_path = out_dir / f"{VAL_PATH.stem}_realism_gate.md"
    assert json_path.exists(), f"missing {json_path}"
    assert md_path.exists(), f"missing {md_path}"
    payload = json.loads(json_path.read_text())
    assert payload["window_days"] == 30
    assert payload["decision_lag"] == 2
    assert payload["short_borrow_apr"] == 0.0625
    assert payload["disable_shorts"] is True
    assert payload["slippage_bps_grid"] == [0.0, 5.0, 10.0, 20.0]
    assert payload["slippage_bps"] == [0.0, 5.0, 10.0, 20.0]
    assert payload["fill_buffer_bps_grid"] == [5.0]
    assert payload["max_leverage_grid"] == [1.0]
    assert payload["n_windows_per_cell"] == 1
    assert payload["promotion_gate"]["enforced"] is False
    assert payload["promotion_gate"]["n_cells"] == 4
    assert "worst_cell" in payload["promotion_gate"]
    assert payload["ensemble_size"] >= 2, "should have loaded the prod multi-model ensemble"
    cells = payload["cells"]
    assert len(cells) == 4
    assert {cell["slippage_bps"] for cell in cells} == {0.0, 5.0, 10.0, 20.0}
    for cell in cells:
        assert cell["fill_buffer_bps"] == 5.0
        assert cell["max_leverage"] == 1.0
        assert cell["n_windows"] == 1
    # Total return must be a real number; sortino can be inf when there are no
    # negative returns in the window, but it must not be NaN.
    for cell in cells:
        assert isinstance(cell["median_total_return"], float)
        assert not math.isnan(cell["median_total_return"])
    md_text = md_path.read_text()
    assert "Screened32 realism gate" in md_text
    assert "short_borrow_apr=0.0625" in md_text
    assert "slippage_bps_grid=0,5,10,20" in md_text
    assert "promotion_gate:" in md_text
    assert "slip_bps" in md_text
    assert "fill_bps" in md_text
