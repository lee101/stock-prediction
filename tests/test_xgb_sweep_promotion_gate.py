from __future__ import annotations

import json
import hashlib
import re
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "xgb_sweep_promotion_gate.py"


def _load_module():
    spec = spec_from_file_location("xgb_sweep_promotion_gate", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_sweep(
    tmp_path: Path,
    *,
    cells: list[dict],
    oos_start="2025-12-18",
    oos_end="2026-04-17",
    complete: bool | None = True,
    model_paths: list[str] | None = None,
    model_sha256: list[str] | None = None,
    ensemble_manifest: dict | None = None,
    symbols_file: str | None = None,
    data_root: str | None = None,
    spy_csv: str | None = None,
    spy_csv_sha256: str | None = None,
    blend_mode: str | None = None,
) -> Path:
    path = tmp_path / "sweep.json"
    payload = {
        "oos_start": oos_start,
        "oos_end": oos_end,
        "cells": cells,
    }
    if complete is not None:
        payload["complete"] = complete
    if model_paths is not None:
        payload["model_paths"] = model_paths
    if model_sha256 is not None:
        payload["model_sha256"] = model_sha256
    if ensemble_manifest is not None:
        payload["ensemble_manifest"] = ensemble_manifest
    if symbols_file is not None:
        payload["symbols_file"] = symbols_file
    if data_root is not None:
        payload["data_root"] = data_root
    if spy_csv is not None:
        payload["spy_csv"] = spy_csv
    if spy_csv_sha256 is not None:
        payload["spy_csv_sha256"] = spy_csv_sha256
    if blend_mode is not None:
        payload["blend_mode"] = blend_mode
    path.write_text(json.dumps(payload))
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_evaluate_sweep_passes_best_stress_cell(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[
        {
            "fee_regime": "stress36x",
            "median_monthly_pct": 35.0,
            "p10_monthly_pct": 12.0,
            "worst_dd_pct": 10.0,
            "n_neg": 0,
            "n_windows": 8,
        },
        {
            "fee_regime": "deploy",
            "median_monthly_pct": 80.0,
            "p10_monthly_pct": 50.0,
            "worst_dd_pct": 5.0,
            "n_neg": 0,
            "n_windows": 8,
        },
    ])
    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=5,
        min_p10_monthly_pct=None,
    ))

    assert result.passed is True
    assert result.oos_days == 121
    assert result.best_cell["fee_regime"] == "stress36x"


def test_evaluate_sweep_fails_short_oos_before_cell_scores(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(
        tmp_path,
        oos_start="2026-04-01",
        oos_end="2026-04-20",
        cells=[{
            "fee_regime": "stress36x",
            "median_monthly_pct": 200.0,
            "p10_monthly_pct": 100.0,
            "worst_dd_pct": 0.0,
            "n_neg": 0,
            "n_windows": 4,
        }],
    )
    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
    ))

    assert result.passed is False
    assert result.reason == "oos_days 20 < 100"
    assert result.best_cell is None


def test_evaluate_sweep_can_require_positive_p10(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 50.0,
        "p10_monthly_pct": -1.0,
        "worst_dd_pct": 10.0,
        "n_neg": 0,
        "n_windows": 8,
    }])
    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=0.0,
    ))

    assert result.passed is False
    assert result.reason == "p10_monthly_pct -1.00 < 0.00"


def test_evaluate_sweep_rejects_nan_core_metric(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": float("nan"),
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 8,
    }])
    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
    ))

    assert result.passed is False
    assert result.reason == "median_monthly_pct non-finite"


def test_evaluate_sweep_rejects_missing_risk_metric(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "n_neg": 0,
        "n_windows": 8,
    }])
    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
    ))

    assert result.passed is False
    assert result.reason == "worst_dd_pct missing"


def test_evaluate_sweep_rejects_invalid_window_count(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 1.5,
    }])
    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
    ))

    assert result.passed is False
    assert result.reason == "n_windows invalid"


def test_evaluate_sweep_rejects_bool_integer_counts(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": False,
        "n_windows": 8,
    }])
    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
    ))

    assert result.passed is False
    assert result.reason == "n_neg invalid"


def test_evaluate_sweep_rejects_partial_expected_windows(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 3,
        "expected_n_windows": 4,
    }])
    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
    ))

    assert result.passed is False
    assert result.reason == "n_windows 3 < 4"


def test_evaluate_sweep_can_ignore_expected_windows_for_legacy_inspection(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 3,
        "expected_n_windows": 4,
    }])
    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        require_expected_windows=False,
    ))

    assert result.passed is True


def test_evaluate_sweep_rejects_invalid_expected_windows(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 4,
        "expected_n_windows": 4.5,
    }])
    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
    ))

    assert result.passed is False
    assert result.reason == "expected_n_windows invalid"


def test_evaluate_sweep_rejects_fail_fast_cell_even_when_metrics_clear(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 8,
        "fail_fast_triggered": True,
        "fail_fast_reason": "intraday_dd_pct>=40",
    }])
    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
    ))

    assert result.passed is False
    assert result.reason == "fail_fast_triggered: intraday_dd_pct>=40"


def test_evaluate_sweep_can_allow_fail_fast_cells_for_legacy_inspection(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 8,
        "fail_fast_triggered": True,
        "fail_fast_reason": "intraday_dd_pct>=40",
    }])
    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        reject_fail_fast=False,
    ))

    assert result.passed is True


def test_evaluate_sweep_rejects_skip_stress_cells_by_default(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 8,
        "skip_prob": 0.25,
        "skip_seed": 7,
    }])
    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
    ))

    assert result.passed is False
    assert result.reason == "skip_prob 0.2500 != 0.0"


def test_evaluate_sweep_can_allow_skip_stress_cells_for_research_inspection(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 8,
        "skip_prob": 0.25,
        "skip_seed": 7,
    }])
    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        reject_skip_stress=False,
    ))

    assert result.passed is True


def test_evaluate_sweep_rejects_low_fill_buffer_when_required(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 8,
        "fill_buffer_bps": 5.0,
    }])
    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        min_fill_buffer_bps=15.0,
    ))

    assert result.passed is False
    assert result.reason == "fill_buffer_bps 5.00 < 15.00"


def test_evaluate_sweep_rejects_incomplete_sweep_even_when_cell_passes(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, complete=False, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 8,
    }])
    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
    ))

    assert result.passed is False
    assert result.reason == "sweep complete flag is False"
    assert result.best_cell is None


def test_evaluate_sweep_rejects_missing_complete_flag_by_default(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, complete=None, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 8,
    }])
    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
    ))

    assert result.passed is False
    assert result.reason == "sweep complete flag is '<missing>'"


def test_evaluate_sweep_can_allow_partial_for_research_inspection(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, complete=False, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 8,
    }])
    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        require_complete=False,
    ))

    assert result.passed is True


def test_evaluate_sweep_selects_best_passing_cell_not_best_raw_return(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[
        {
            "fee_regime": "stress36x",
            "median_monthly_pct": 90.0,
            "p10_monthly_pct": 50.0,
            "worst_dd_pct": 5.0,
            "n_neg": 0,
            "n_windows": 8,
            "fail_fast_triggered": True,
            "fail_fast_reason": "intraday_dd_pct>=40",
        },
        {
            "fee_regime": "stress36x",
            "median_monthly_pct": 40.0,
            "p10_monthly_pct": 15.0,
            "worst_dd_pct": 8.0,
            "n_neg": 0,
            "n_windows": 8,
        },
    ])
    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
    ))

    assert result.passed is True
    assert result.best_cell["median_monthly_pct"] == 40.0


def test_launch_script_filter_requires_exact_live_knob_match(tmp_path: Path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --symbols-file symbols.txt \\\n"
        "  --model-paths a.pkl,b.pkl \\\n"
        "  --top-n 2 \\\n"
        "  --min-picks 1 \\\n"
        "  --allocation 2.5 \\\n"
        "  --allocation-mode softmax \\\n"
        "  --allocation-temp 0.25 \\\n"
        "  --score-uncertainty-penalty 0.40 \\\n"
        "  --min-score 0.62 \\\n"
        "  --hold-through \\\n"
        "  --min-dollar-vol 50000000 \\\n"
        "  --max-spread-bps 12 \\\n"
        "  --min-vol-20d 0.12 \\\n"
        "  --max-vol-20d 0.55 \\\n"
        "  --max-ret-20d-rank-pct 0.80 \\\n"
        "  --min-ret-5d-rank-pct 0.20 \\\n"
        "  --regime-gate-window 20 \\\n"
        "  --inv-vol-target-ann 0.20 \\\n"
        "  --inv-vol-floor 0.08 \\\n"
        "  --inv-vol-cap 2.0 \\\n"
        "  --vol-target-ann 0.15 \\\n"
        "  --no-picks-fallback SPY \\\n"
        "  --no-picks-fallback-alloc 0.25 \\\n"
        "  --conviction-scaled-alloc \\\n"
        "  --conviction-alloc-low 0.52 \\\n"
        "  --conviction-alloc-high 0.72 \\\n"
        "  --regime-cs-skew-min 1.0 \\\n"
        "  --live\n"
    )
    live_config = mod.extract_live_config_from_launch(launch)
    assert live_config["top_n"] == 2
    assert live_config["min_picks"] == 1
    assert live_config["opportunistic_watch_n"] == 0
    assert live_config["opportunistic_entry_discount_bps"] == 0.0
    assert live_config["leverage"] == 2.5
    assert live_config["hold_through"] is True
    assert live_config["no_picks_fallback_symbol"] == "SPY"
    assert live_config["no_picks_fallback_alloc_scale"] == 0.25
    assert live_config["inference_max_spread_bps"] == 12.0
    assert live_config["inference_max_vol_20d"] == 0.55
    assert live_config["max_ret_20d_rank_pct"] == 0.80
    assert live_config["min_ret_5d_rank_pct"] == 0.20
    assert live_config["skip_prob"] == 0.0
    assert live_config["skip_seed"] == 0
    assert live_config["regime_gate_window"] == 20
    assert live_config["vol_target_ann"] == 0.15
    assert live_config["inv_vol_target_ann"] == 0.20
    assert live_config["inv_vol_floor"] == 0.08
    assert live_config["inv_vol_cap"] == 2.0
    assert live_config["conviction_scaled_alloc"] is True
    assert live_config["conviction_alloc_low"] == 0.52
    assert live_config["conviction_alloc_high"] == 0.72
    assert live_config["allocation_mode"] == "softmax"
    assert live_config["allocation_temp"] == 0.25
    assert live_config["score_uncertainty_penalty"] == 0.40

    path = _write_sweep(tmp_path, cells=[
        {
            "fee_regime": "stress36x",
            "leverage": 2.5,
            "top_n": 1,
            "min_picks": 1,
            "min_score": 0.62,
            "hold_through": True,
            "inference_min_dolvol": 50000000,
            "inference_max_spread_bps": 12.0,
            "inference_min_vol_20d": 0.12,
            "inference_max_vol_20d": 0.0,
            "max_ret_20d_rank_pct": 1.0,
            "min_ret_5d_rank_pct": 0.0,
            "regime_cs_iqr_max": 0.0,
            "regime_cs_skew_min": 1.0,
            "inv_vol_target_ann": 0.0,
            "inv_vol_floor": 0.05,
            "inv_vol_cap": 3.0,
            "no_picks_fallback_symbol": "",
            "no_picks_fallback_alloc_scale": 0.0,
            "allocation_mode": "equal",
            "score_uncertainty_penalty": 0.0,
            "median_monthly_pct": 80.0,
            "p10_monthly_pct": 20.0,
            "worst_dd_pct": 5.0,
            "n_neg": 0,
            "n_windows": 8,
        },
        {
            "fee_regime": "stress36x",
            "leverage": 2.5,
            "top_n": 2,
            "min_picks": 0,
            "min_score": 0.62,
            "hold_through": True,
            "inference_min_dolvol": 50000000,
            "inference_max_spread_bps": 12.0,
            "inference_min_vol_20d": 0.12,
            "inference_max_vol_20d": 0.55,
            "max_ret_20d_rank_pct": 0.80,
            "min_ret_5d_rank_pct": 0.20,
            "regime_gate_window": 20,
            "regime_cs_iqr_max": 0.0,
            "regime_cs_skew_min": 1.0,
            "inv_vol_target_ann": 0.0,
            "inv_vol_floor": 0.05,
            "inv_vol_cap": 3.0,
            "no_picks_fallback_symbol": "",
            "no_picks_fallback_alloc_scale": 0.0,
            "allocation_mode": "equal",
            "score_uncertainty_penalty": 0.0,
            "median_monthly_pct": 70.0,
            "p10_monthly_pct": 20.0,
            "worst_dd_pct": 5.0,
            "n_neg": 0,
            "n_windows": 8,
        },
        {
            "fee_regime": "stress36x",
            "leverage": 2.5,
            "top_n": 2,
            "min_picks": 0,
            "min_score": 0.62,
            "hold_through": True,
            "inference_min_dolvol": 50000000,
            "inference_max_spread_bps": 12.0,
            "inference_min_vol_20d": 0.12,
            "inference_max_vol_20d": 0.0,
            "max_ret_20d_rank_pct": 1.0,
            "min_ret_5d_rank_pct": 0.0,
            "regime_cs_iqr_max": 0.0,
            "regime_cs_skew_min": 1.0,
            "inv_vol_target_ann": 0.0,
            "inv_vol_floor": 0.05,
            "inv_vol_cap": 3.0,
            "no_picks_fallback_symbol": "SPY",
            "no_picks_fallback_alloc_scale": 0.25,
            "conviction_scaled_alloc": True,
            "conviction_alloc_low": 0.52,
            "conviction_alloc_high": 0.72,
            "allocation_mode": "softmax",
            "allocation_temp": 0.25,
            "score_uncertainty_penalty": 0.0,
            "median_monthly_pct": 50.0,
            "p10_monthly_pct": 12.0,
            "worst_dd_pct": 6.0,
            "n_neg": 0,
            "n_windows": 8,
        },
        {
            "fee_regime": "stress36x",
            "leverage": 2.5,
            "top_n": 2,
            "min_picks": 0,
            "min_score": 0.62,
            "hold_through": True,
            "inference_min_dolvol": 50000000,
            "inference_min_vol_20d": 0.12,
            "inference_max_vol_20d": 0.55,
            "max_ret_20d_rank_pct": 0.80,
            "min_ret_5d_rank_pct": 0.20,
            "regime_gate_window": 20,
            "regime_cs_iqr_max": 0.0,
            "regime_cs_skew_min": 1.0,
            "inv_vol_target_ann": 0.0,
            "inv_vol_floor": 0.05,
            "inv_vol_cap": 3.0,
            "no_picks_fallback_symbol": "SPY",
            "no_picks_fallback_alloc_scale": 0.25,
            "conviction_scaled_alloc": False,
            "conviction_alloc_low": 0.55,
            "conviction_alloc_high": 0.85,
            "allocation_mode": "equal",
            "allocation_temp": 1.0,
            "score_uncertainty_penalty": 0.0,
            "median_monthly_pct": 65.0,
            "p10_monthly_pct": 15.0,
            "worst_dd_pct": 6.0,
            "n_neg": 0,
            "n_windows": 8,
        },
        {
            "fee_regime": "stress36x",
            "leverage": 2.5,
            "top_n": 2,
            "min_picks": 0,
            "min_score": 0.62,
            "hold_through": True,
            "inference_min_dolvol": 50000000,
            "inference_min_vol_20d": 0.12,
            "inference_max_vol_20d": 0.55,
            "max_ret_20d_rank_pct": 0.80,
            "min_ret_5d_rank_pct": 0.20,
            "regime_gate_window": 20,
            "regime_cs_iqr_max": 0.0,
            "regime_cs_skew_min": 1.0,
            "inv_vol_target_ann": 0.0,
            "inv_vol_floor": 0.05,
            "inv_vol_cap": 3.0,
            "no_picks_fallback_symbol": "SPY",
            "no_picks_fallback_alloc_scale": 0.25,
            "conviction_scaled_alloc": False,
            "conviction_alloc_low": 0.55,
            "conviction_alloc_high": 0.85,
            "allocation_mode": "softmax",
            "allocation_temp": 0.25,
            "score_uncertainty_penalty": 0.0,
            "median_monthly_pct": 55.0,
            "p10_monthly_pct": 12.0,
            "worst_dd_pct": 6.0,
            "n_neg": 0,
            "n_windows": 8,
        },
        {
            "fee_regime": "stress36x",
            "leverage": 2.5,
            "top_n": 2,
            "min_picks": 1,
            "min_score": 0.62,
            "hold_through": True,
            "inference_min_dolvol": 50000000,
            "inference_max_spread_bps": 12.0,
            "inference_min_vol_20d": 0.12,
            "inference_max_vol_20d": 0.55,
            "max_ret_20d_rank_pct": 0.80,
            "min_ret_5d_rank_pct": 0.20,
            "regime_gate_window": 20,
            "regime_cs_iqr_max": 0.0,
            "regime_cs_skew_min": 1.0,
            "vol_target_ann": 0.15,
            "inv_vol_target_ann": 0.20,
            "inv_vol_floor": 0.08,
            "inv_vol_cap": 2.0,
            "no_picks_fallback_symbol": "SPY",
            "no_picks_fallback_alloc_scale": 0.25,
            "conviction_scaled_alloc": True,
            "conviction_alloc_low": 0.52,
            "conviction_alloc_high": 0.72,
            "allocation_mode": "softmax",
            "allocation_temp": 0.25,
            "score_uncertainty_penalty": 0.40,
            "median_monthly_pct": 40.0,
            "p10_monthly_pct": 10.0,
            "worst_dd_pct": 6.0,
            "n_neg": 0,
            "n_windows": 8,
        },
    ])

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_config=live_config,
    ))

    assert result.passed is True
    assert result.best_cell["top_n"] == 2
    assert result.best_cell["min_picks"] == 1
    assert result.best_cell["no_picks_fallback_symbol"] == "SPY"
    assert result.best_cell["conviction_scaled_alloc"] is True
    assert result.best_cell["allocation_mode"] == "softmax"
    assert result.best_cell["score_uncertainty_penalty"] == 0.40
    assert result.best_cell["inference_max_spread_bps"] == 12.0
    assert result.best_cell["regime_gate_window"] == 20
    assert result.best_cell["vol_target_ann"] == 0.15
    assert result.best_cell["inv_vol_target_ann"] == 0.20
    assert result.best_cell["inv_vol_floor"] == 0.08
    assert result.best_cell["inv_vol_cap"] == 2.0
    assert result.best_cell["median_monthly_pct"] == 40.0


def test_live_config_rejects_opportunistic_sweep_cells(tmp_path: Path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --model-paths a.pkl,b.pkl \\\n"
        "  --top-n 2 \\\n"
        "  --allocation 2.0 \\\n"
        "  --live\n"
    )
    live_config = mod.extract_live_config_from_launch(launch)
    assert live_config["opportunistic_watch_n"] == 0
    assert live_config["opportunistic_entry_discount_bps"] == 0.0

    path = _write_sweep(tmp_path, cells=[
        {
            "fee_regime": "stress36x",
            "leverage": 2.0,
            "top_n": 2,
            "min_picks": 0,
            "opportunistic_watch_n": 5,
            "opportunistic_entry_discount_bps": 30.0,
            "min_score": 0.0,
            "hold_through": False,
            "median_monthly_pct": 80.0,
            "p10_monthly_pct": 20.0,
            "worst_dd_pct": 5.0,
            "n_neg": 0,
            "n_windows": 8,
        },
        {
            "fee_regime": "stress36x",
            "leverage": 2.0,
            "top_n": 2,
            "min_picks": 0,
            "opportunistic_watch_n": 0,
            "opportunistic_entry_discount_bps": 0.0,
            "min_score": 0.0,
            "hold_through": False,
            "median_monthly_pct": 40.0,
            "p10_monthly_pct": 12.0,
            "worst_dd_pct": 6.0,
            "n_neg": 0,
            "n_windows": 8,
        },
    ])

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_config=live_config,
    ))

    assert result.passed is True
    assert result.best_cell["median_monthly_pct"] == 40.0
    assert result.best_cell["opportunistic_watch_n"] == 0
    assert result.best_cell["opportunistic_entry_discount_bps"] == 0.0


def test_live_config_matches_spy_regime_and_vol_target_cells(tmp_path: Path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --model-paths a.pkl,b.pkl \\\n"
        "  --top-n 2 \\\n"
        "  --allocation 2.0 \\\n"
        "  --regime-gate-window 20 \\\n"
        "  --vol-target-ann 0.15 \\\n"
        "  --live\n"
    )
    live_config = mod.extract_live_config_from_launch(launch)
    assert live_config["regime_gate_window"] == 20
    assert live_config["vol_target_ann"] == 0.15

    path = _write_sweep(tmp_path, cells=[
        {
            "fee_regime": "stress36x",
            "leverage": 2.0,
            "top_n": 2,
            "regime_gate_window": 20,
            "vol_target_ann": 0.15,
            "median_monthly_pct": 80.0,
            "p10_monthly_pct": 20.0,
            "worst_dd_pct": 5.0,
            "n_neg": 0,
            "n_windows": 8,
        },
        {
            "fee_regime": "stress36x",
            "leverage": 2.0,
            "top_n": 2,
            "regime_gate_window": 0,
            "vol_target_ann": 0.15,
            "median_monthly_pct": 70.0,
            "p10_monthly_pct": 18.0,
            "worst_dd_pct": 5.0,
            "n_neg": 0,
            "n_windows": 8,
        },
        {
            "fee_regime": "stress36x",
            "leverage": 2.0,
            "top_n": 2,
            "regime_gate_window": 0,
            "vol_target_ann": 0.0,
            "median_monthly_pct": 40.0,
            "p10_monthly_pct": 12.0,
            "worst_dd_pct": 6.0,
            "n_neg": 0,
            "n_windows": 8,
        },
    ])

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_config=live_config,
    ))

    assert result.passed is True
    assert result.best_cell["median_monthly_pct"] == 80.0
    assert result.best_cell["regime_gate_window"] == 20
    assert result.best_cell["vol_target_ann"] == 0.15


def test_live_config_rejects_missed_order_stress_cells(tmp_path: Path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --model-paths a.pkl,b.pkl \\\n"
        "  --top-n 2 \\\n"
        "  --allocation 2.0 \\\n"
        "  --live\n"
    )
    live_config = mod.extract_live_config_from_launch(launch)
    assert live_config["skip_prob"] == 0.0
    assert live_config["skip_seed"] == 0

    path = _write_sweep(tmp_path, cells=[
        {
            "fee_regime": "stress36x",
            "leverage": 2.0,
            "top_n": 2,
            "skip_prob": 0.25,
            "skip_seed": 7,
            "median_monthly_pct": 90.0,
            "p10_monthly_pct": 25.0,
            "worst_dd_pct": 4.0,
            "n_neg": 0,
            "n_windows": 8,
        },
        {
            "fee_regime": "stress36x",
            "leverage": 2.0,
            "top_n": 2,
            "skip_prob": 0.0,
            "skip_seed": 0,
            "median_monthly_pct": 40.0,
            "p10_monthly_pct": 12.0,
            "worst_dd_pct": 6.0,
            "n_neg": 0,
            "n_windows": 8,
        },
    ])

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_config=live_config,
    ))

    assert result.passed is True
    assert result.best_cell["median_monthly_pct"] == 40.0
    assert result.best_cell["skip_prob"] == 0.0
    assert result.best_cell["skip_seed"] == 0


def test_launch_script_model_paths_resolve_shell_vars(tmp_path: Path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "MODEL_DIR=\"analysis/xgbnew_daily/custom\"\n"
        "MODEL_PATHS=\"${MODEL_DIR}/alltrain_seed0.pkl,${MODEL_DIR}/alltrain_seed7.pkl\"\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --model-paths \"${MODEL_PATHS}\" \\\n"
        "  --live\n"
    )

    assert mod.extract_model_paths_from_launch(launch) == (
        str((REPO / "analysis/xgbnew_daily/custom/alltrain_seed0.pkl").resolve(strict=False)),
        str((REPO / "analysis/xgbnew_daily/custom/alltrain_seed7.pkl").resolve(strict=False)),
    )


def test_launch_script_model_paths_respect_unset(tmp_path: Path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "MODEL_DIR=\"analysis/xgbnew_daily/custom\"\n"
        "MODEL_PATHS=\"${MODEL_DIR}/alltrain_seed0.pkl\"\n"
        "unset MODEL_PATHS\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --model-paths \"${MODEL_PATHS}\" \\\n"
        "  --live\n"
    )

    assert mod.extract_model_paths_from_launch(launch) == ()


def test_launch_script_symbols_file_resolves_shell_vars(tmp_path: Path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "SYMBOL_DIR=\"symbol_lists\"\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --symbols-file \"${SYMBOL_DIR}/custom.txt\" \\\n"
        "  --model-paths a.pkl \\\n"
        "  --live\n"
    )

    assert mod.extract_symbols_file_from_launch(launch) == str(
        (REPO / "symbol_lists/custom.txt").resolve(strict=False)
    )


def test_launch_script_data_root_resolves_shell_vars(tmp_path: Path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "DATA_ROOT=\"trainingdata_custom\"\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --symbols-file symbol_lists/custom.txt \\\n"
        "  --data-root \"${DATA_ROOT}\" \\\n"
        "  --model-paths a.pkl \\\n"
        "  --live\n"
    )

    assert mod.extract_data_root_from_launch(launch) == str(
        (REPO / "trainingdata_custom").resolve(strict=False)
    )


def test_launch_script_spy_csv_resolves_shell_vars(tmp_path: Path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "SPY_CSV=\"market_inputs/custom_spy.csv\"\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --data-root trainingdata_ignored \\\n"
        "  --spy-csv \"${SPY_CSV}\" \\\n"
        "  --model-paths a.pkl \\\n"
        "  --live\n"
    )

    assert mod.extract_spy_csv_from_launch(launch) == str(
        (REPO / "market_inputs/custom_spy.csv").resolve(strict=False)
    )


def test_launch_audit_requires_symbols_file_match(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(
        tmp_path,
        symbols_file="symbol_lists/sweep_universe.txt",
        cells=[
            {
                "fee_regime": "stress36x",
                "median_monthly_pct": 35.0,
                "p10_monthly_pct": 12.0,
                "worst_dd_pct": 10.0,
                "n_neg": 0,
                "n_windows": 8,
            },
        ],
    )

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_symbols_file=str((REPO / "symbol_lists/live_universe.txt").resolve(strict=False)),
    ))

    assert result.passed is False
    assert result.reason == "launch symbols_file does not match sweep symbols_file"


def test_launch_audit_requires_data_root_match(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(
        tmp_path,
        data_root="trainingdata_sweep",
        cells=[
            {
                "fee_regime": "stress36x",
                "median_monthly_pct": 35.0,
                "p10_monthly_pct": 12.0,
                "worst_dd_pct": 10.0,
                "n_neg": 0,
                "n_windows": 8,
            },
        ],
    )

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_data_root=str((REPO / "trainingdata_live").resolve(strict=False)),
    ))

    assert result.passed is False
    assert result.reason == "launch data_root does not match sweep data_root"


def test_launch_audit_rejects_missing_sweep_symbols_file(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(
        tmp_path,
        cells=[
            {
                "fee_regime": "stress36x",
                "median_monthly_pct": 35.0,
                "p10_monthly_pct": 12.0,
                "worst_dd_pct": 10.0,
                "n_neg": 0,
                "n_windows": 8,
            },
        ],
    )

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_symbols_file=str((REPO / "symbol_lists/live_universe.txt").resolve(strict=False)),
    ))

    assert result.passed is False
    assert result.reason == "sweep symbols_file missing"


def test_launch_audit_rejects_missing_sweep_data_root(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(
        tmp_path,
        cells=[
            {
                "fee_regime": "stress36x",
                "median_monthly_pct": 35.0,
                "p10_monthly_pct": 12.0,
                "worst_dd_pct": 10.0,
                "n_neg": 0,
                "n_windows": 8,
            },
        ],
    )

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_data_root=str((REPO / "trainingdata").resolve(strict=False)),
    ))

    assert result.passed is False
    assert result.reason == "sweep data_root missing"


def test_launch_audit_requires_spy_csv_match_when_vol_target_enabled(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(
        tmp_path,
        spy_csv="trainingdata_sweep/SPY.csv",
        cells=[
            {
                "fee_regime": "stress36x",
                "vol_target_ann": 0.15,
                "median_monthly_pct": 35.0,
                "p10_monthly_pct": 12.0,
                "worst_dd_pct": 10.0,
                "n_neg": 0,
                "n_windows": 8,
            },
        ],
    )

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_config={"vol_target_ann": 0.15},
        live_spy_csv=str((REPO / "trainingdata_live/SPY.csv").resolve(strict=False)),
    ))

    assert result.passed is False
    assert result.reason == "launch spy_csv does not match sweep spy_csv"


def test_launch_audit_rejects_invalid_spy_csv_hash_metadata(tmp_path: Path) -> None:
    mod = _load_module()
    spy_csv = tmp_path / "SPY.csv"
    spy_csv.write_text("timestamp,close\n2026-01-02T21:00:00Z,100\n", encoding="utf-8")
    path = _write_sweep(
        tmp_path,
        spy_csv=str(spy_csv),
        spy_csv_sha256="not-a-sha",
        cells=[
            {
                "fee_regime": "stress36x",
                "vol_target_ann": 0.15,
                "median_monthly_pct": 35.0,
                "p10_monthly_pct": 12.0,
                "worst_dd_pct": 10.0,
                "n_neg": 0,
                "n_windows": 8,
            },
        ],
    )

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_config={"vol_target_ann": 0.15},
        live_spy_csv=str(spy_csv.resolve(strict=False)),
    ))

    assert result.passed is False
    assert result.reason == "sweep spy_csv_sha256 invalid"


def test_launch_audit_rejects_spy_csv_hash_mismatch(tmp_path: Path) -> None:
    mod = _load_module()
    spy_csv = tmp_path / "SPY.csv"
    spy_csv.write_text("timestamp,close\n2026-01-02T21:00:00Z,100\n", encoding="utf-8")
    path = _write_sweep(
        tmp_path,
        spy_csv=str(spy_csv),
        spy_csv_sha256=hashlib.sha256(b"different spy data").hexdigest(),
        cells=[
            {
                "fee_regime": "stress36x",
                "vol_target_ann": 0.15,
                "median_monthly_pct": 35.0,
                "p10_monthly_pct": 12.0,
                "worst_dd_pct": 10.0,
                "n_neg": 0,
                "n_windows": 8,
            },
        ],
    )

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_config={"vol_target_ann": 0.15},
        live_spy_csv=str(spy_csv.resolve(strict=False)),
    ))

    assert result.passed is False
    assert result.reason == "launch spy_csv_sha256 does not match sweep spy_csv_sha256"


def test_launch_audit_accepts_matching_spy_csv_hash(tmp_path: Path) -> None:
    mod = _load_module()
    spy_csv = tmp_path / "SPY.csv"
    spy_csv.write_text("timestamp,close\n2026-01-02T21:00:00Z,100\n", encoding="utf-8")
    path = _write_sweep(
        tmp_path,
        spy_csv=str(spy_csv),
        spy_csv_sha256=_sha256(spy_csv),
        cells=[
            {
                "fee_regime": "stress36x",
                "vol_target_ann": 0.15,
                "median_monthly_pct": 35.0,
                "p10_monthly_pct": 12.0,
                "worst_dd_pct": 10.0,
                "n_neg": 0,
                "n_windows": 8,
            },
        ],
    )

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_config={"vol_target_ann": 0.15},
        live_spy_csv=str(spy_csv.resolve(strict=False)),
    ))

    assert result.passed is True


def test_launch_audit_rejects_missing_sweep_spy_csv_when_vol_target_enabled(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(
        tmp_path,
        cells=[
            {
                "fee_regime": "stress36x",
                "vol_target_ann": 0.15,
                "median_monthly_pct": 35.0,
                "p10_monthly_pct": 12.0,
                "worst_dd_pct": 10.0,
                "n_neg": 0,
                "n_windows": 8,
            },
        ],
    )

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_config={"vol_target_ann": 0.15},
        live_spy_csv=str((REPO / "trainingdata/SPY.csv").resolve(strict=False)),
    ))

    assert result.passed is False
    assert result.reason == "sweep spy_csv missing"


def test_launch_audit_rejects_missing_sweep_spy_csv_when_regime_gate_enabled(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(
        tmp_path,
        cells=[
            {
                "fee_regime": "stress36x",
                "regime_gate_window": 20,
                "median_monthly_pct": 35.0,
                "p10_monthly_pct": 12.0,
                "worst_dd_pct": 10.0,
                "n_neg": 0,
                "n_windows": 8,
            },
        ],
    )

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_config={"regime_gate_window": 20},
        live_spy_csv=str((REPO / "trainingdata/SPY.csv").resolve(strict=False)),
    ))

    assert result.passed is False
    assert result.reason == "sweep spy_csv missing"


def test_launch_audit_requires_live_mean_blend_mode(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(
        tmp_path,
        blend_mode="median",
        cells=[
            {
                "fee_regime": "stress36x",
                "median_monthly_pct": 35.0,
                "p10_monthly_pct": 12.0,
                "worst_dd_pct": 10.0,
                "n_neg": 0,
                "n_windows": 8,
            },
        ],
    )

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_blend_mode="mean",
    ))

    assert result.passed is False
    assert result.reason == "launch blend_mode does not match sweep blend_mode"


def test_launch_audit_rejects_missing_sweep_blend_mode(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(
        tmp_path,
        cells=[
            {
                "fee_regime": "stress36x",
                "median_monthly_pct": 35.0,
                "p10_monthly_pct": 12.0,
                "worst_dd_pct": 10.0,
                "n_neg": 0,
                "n_windows": 8,
            },
        ],
    )

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_blend_mode="mean",
    ))

    assert result.passed is False
    assert result.reason == "sweep blend_mode missing"


def test_launch_script_gate_requires_matching_sweep_model_paths(tmp_path: Path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "MODEL_DIR=\"analysis/xgbnew_daily/custom\"\n"
        "MODEL_PATHS=\"${MODEL_DIR}/alltrain_seed0.pkl,${MODEL_DIR}/alltrain_seed7.pkl\"\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --model-paths \"${MODEL_PATHS}\" \\\n"
        "  --top-n 2 \\\n"
        "  --allocation 2.5 \\\n"
        "  --live\n"
    )
    cell = {
        "fee_regime": "stress36x",
        "leverage": 2.5,
        "top_n": 2,
        "median_monthly_pct": 40.0,
        "p10_monthly_pct": 10.0,
        "worst_dd_pct": 6.0,
        "n_neg": 0,
        "n_windows": 8,
        "fill_buffer_bps": 15.0,
    }
    matching = _write_sweep(
        tmp_path,
        cells=[cell],
        model_paths=[
            "analysis/xgbnew_daily/custom/alltrain_seed0.pkl",
            "analysis/xgbnew_daily/custom/alltrain_seed7.pkl",
        ],
    )
    mismatched = tmp_path / "mismatched.json"
    mismatched.write_text(json.dumps({
        "oos_start": "2025-12-18",
        "oos_end": "2026-04-17",
        "complete": True,
        "model_paths": [
            "analysis/xgbnew_daily/other/alltrain_seed0.pkl",
            "analysis/xgbnew_daily/other/alltrain_seed7.pkl",
        ],
        "cells": [cell],
    }))

    live_config = mod.extract_live_config_from_launch(launch)
    live_model_paths = mod.extract_model_paths_from_launch(launch)
    config = mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_config=live_config,
        live_model_paths=live_model_paths,
    )

    assert mod.evaluate_sweep(matching, config).passed is True
    result = mod.evaluate_sweep(mismatched, config)
    assert result.passed is False
    assert result.reason == "launch model_paths do not match sweep model_paths"


def test_launch_script_gate_rejects_missing_sweep_model_paths(tmp_path: Path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --model-paths a.pkl,b.pkl \\\n"
        "  --top-n 2 \\\n"
        "  --allocation 2.5 \\\n"
        "  --live\n"
    )
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "leverage": 2.5,
        "top_n": 2,
        "median_monthly_pct": 40.0,
        "p10_monthly_pct": 10.0,
        "worst_dd_pct": 6.0,
        "n_neg": 0,
        "n_windows": 8,
        "fill_buffer_bps": 15.0,
    }])

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_config=mod.extract_live_config_from_launch(launch),
        live_model_paths=mod.extract_model_paths_from_launch(launch),
    ))

    assert result.passed is False
    assert result.reason == "sweep model_paths missing"


def test_launch_script_gate_rejects_model_hash_mismatch(tmp_path: Path) -> None:
    mod = _load_module()
    model = tmp_path / "alltrain_seed0.pkl"
    model.write_bytes(b"live model bytes")
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        f"exec python -u -m xgbnew.live_trader --model-paths {model} --live\n"
    )
    path = _write_sweep(
        tmp_path,
        cells=[{
            "fee_regime": "stress36x",
            "median_monthly_pct": 40.0,
            "p10_monthly_pct": 10.0,
            "worst_dd_pct": 6.0,
            "n_neg": 0,
            "n_windows": 8,
            "fill_buffer_bps": 15.0,
        }],
        model_paths=[str(model)],
        model_sha256=[hashlib.sha256(b"different bytes").hexdigest()],
    )

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_config=mod.extract_live_config_from_launch(launch),
        live_model_paths=mod.extract_model_paths_from_launch(launch),
    ))

    assert result.passed is False
    assert result.reason == "launch model_sha256 do not match sweep model_sha256"


def test_launch_script_gate_rejects_invalid_model_hash_metadata(tmp_path: Path) -> None:
    mod = _load_module()
    model = tmp_path / "alltrain_seed0.pkl"
    model.write_bytes(b"live model bytes")
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        f"exec python -u -m xgbnew.live_trader --model-paths {model} --live\n"
    )
    path = _write_sweep(
        tmp_path,
        cells=[{
            "fee_regime": "stress36x",
            "median_monthly_pct": 40.0,
            "p10_monthly_pct": 10.0,
            "worst_dd_pct": 6.0,
            "n_neg": 0,
            "n_windows": 8,
            "fill_buffer_bps": 15.0,
        }],
        model_paths=[str(model)],
        model_sha256=["not-a-sha"],
    )

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_config=mod.extract_live_config_from_launch(launch),
        live_model_paths=mod.extract_model_paths_from_launch(launch),
    ))

    assert result.passed is False
    assert result.reason == "sweep model_sha256 invalid"


def test_launch_script_gate_accepts_matching_model_hashes(tmp_path: Path) -> None:
    mod = _load_module()
    model = tmp_path / "alltrain_seed0.pkl"
    model.write_bytes(b"live model bytes")
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        f"exec python -u -m xgbnew.live_trader --model-paths {model} --live\n"
    )
    path = _write_sweep(
        tmp_path,
        cells=[{
            "fee_regime": "stress36x",
            "median_monthly_pct": 40.0,
            "p10_monthly_pct": 10.0,
            "worst_dd_pct": 6.0,
            "n_neg": 0,
            "n_windows": 8,
            "fill_buffer_bps": 15.0,
        }],
        model_paths=[str(model)],
        model_sha256=[_sha256(model)],
    )

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_config=mod.extract_live_config_from_launch(launch),
        live_model_paths=mod.extract_model_paths_from_launch(launch),
    ))

    assert result.passed is True


def test_launch_script_gate_rejects_ensemble_manifest_hash_mismatch(tmp_path: Path) -> None:
    mod = _load_module()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model = model_dir / "alltrain_seed0.pkl"
    manifest = model_dir / "alltrain_ensemble.json"
    model.write_bytes(b"live model bytes")
    manifest.write_text(json.dumps({"seeds": [0], "train_end": "2026-04-25"}))
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        f"exec python -u -m xgbnew.live_trader --model-paths {model} --live\n"
    )
    path = _write_sweep(
        tmp_path,
        cells=[{
            "fee_regime": "stress36x",
            "median_monthly_pct": 40.0,
            "p10_monthly_pct": 10.0,
            "worst_dd_pct": 6.0,
            "n_neg": 0,
            "n_windows": 8,
            "fill_buffer_bps": 15.0,
        }],
        model_paths=[str(model)],
        model_sha256=[_sha256(model)],
        ensemble_manifest={
            "path": str(manifest),
            "sha256": hashlib.sha256(b"different manifest").hexdigest(),
        },
    )

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_config=mod.extract_live_config_from_launch(launch),
        live_model_paths=mod.extract_model_paths_from_launch(launch),
    ))

    assert result.passed is False
    assert result.reason == (
        "launch ensemble_manifest sha256 does not match sweep ensemble_manifest sha256"
    )


def test_launch_script_gate_accepts_matching_ensemble_manifest(tmp_path: Path) -> None:
    mod = _load_module()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model = model_dir / "alltrain_seed0.pkl"
    manifest = model_dir / "alltrain_ensemble.json"
    model.write_bytes(b"live model bytes")
    manifest.write_text(json.dumps({"seeds": [0], "train_end": "2026-04-25"}))
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        f"exec python -u -m xgbnew.live_trader --model-paths {model} --live\n"
    )
    path = _write_sweep(
        tmp_path,
        cells=[{
            "fee_regime": "stress36x",
            "median_monthly_pct": 40.0,
            "p10_monthly_pct": 10.0,
            "worst_dd_pct": 6.0,
            "n_neg": 0,
            "n_windows": 8,
            "fill_buffer_bps": 15.0,
        }],
        model_paths=[str(model)],
        model_sha256=[_sha256(model)],
        ensemble_manifest={"path": str(manifest), "sha256": _sha256(manifest)},
    )

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_config=mod.extract_live_config_from_launch(launch),
        live_model_paths=mod.extract_model_paths_from_launch(launch),
    ))

    assert result.passed is True


def test_main_launch_script_rejects_missing_model_paths(tmp_path: Path, capsys) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --top-n 2 \\\n"
        "  --allocation 2.5 \\\n"
        "  --live\n"
    )
    path = _write_sweep(
        tmp_path,
        cells=[{
            "fee_regime": "stress36x",
            "leverage": 2.5,
            "top_n": 2,
            "median_monthly_pct": 40.0,
            "p10_monthly_pct": 10.0,
            "worst_dd_pct": 6.0,
            "n_neg": 0,
            "n_windows": 8,
            "fill_buffer_bps": 15.0,
        }],
        model_paths=["analysis/xgbnew_daily/custom/alltrain_seed0.pkl"],
        symbols_file="symbol_lists/stocks_wide_1000_v1.txt",
        data_root="trainingdata",
        blend_mode="mean",
    )

    assert mod.main([str(path), "--launch-script", str(launch)]) == 3
    assert "launch-script omits --model-paths" in capsys.readouterr().err


def test_main_launch_script_rejects_paper_mode(tmp_path: Path, capsys) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --model-paths a.pkl,b.pkl\n"
    )
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 8,
        "fill_buffer_bps": 15.0,
    }])

    assert mod.main([str(path), "--launch-script", str(launch)]) == 3
    assert "launch-script omits --live" in capsys.readouterr().err


def test_main_launch_script_rejects_dry_run(tmp_path: Path, capsys) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --model-paths a.pkl,b.pkl \\\n"
        "  --live \\\n"
        "  --dry-run\n"
    )
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 8,
        "fill_buffer_bps": 15.0,
    }])

    assert mod.main([str(path), "--launch-script", str(launch)]) == 3
    assert "launch-script enables --dry-run" in capsys.readouterr().err


def test_main_launch_script_rejects_non_live_trader_command(
    tmp_path: Path,
    capsys,
) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m other.trader \\\n"
        "  --model-paths a.pkl,b.pkl \\\n"
        "  --live\n"
    )
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 8,
        "fill_buffer_bps": 15.0,
    }])

    assert mod.main([str(path), "--launch-script", str(launch)]) == 3
    assert "does not invoke xgbnew.live_trader" in capsys.readouterr().err


def test_main_launch_script_allows_live_trader_script_path(tmp_path: Path) -> None:
    mod = _load_module()
    model = tmp_path / "alltrain_seed0.pkl"
    model.write_bytes(b"model")
    symbols = tmp_path / "symbols.txt"
    symbols.write_text("AAPL\n")
    data_root = tmp_path / "trainingdata"
    data_root.mkdir()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        f"exec python -u xgbnew/live_trader.py --model-paths {model} "
        f"--symbols-file {symbols} --data-root {data_root} --live\n"
    )
    path = _write_sweep(
        tmp_path,
        cells=[{
            "fee_regime": "stress36x",
            "median_monthly_pct": 80.0,
            "p10_monthly_pct": 40.0,
            "worst_dd_pct": 5.0,
            "n_neg": 0,
            "n_windows": 8,
            "fill_buffer_bps": 15.0,
        }],
        model_paths=[str(model)],
        model_sha256=[_sha256(model)],
        symbols_file=str(symbols),
        data_root=str(data_root),
        blend_mode="mean",
    )

    assert mod.main([str(path), "--launch-script", str(launch)]) == 0


def test_launch_script_filter_uses_defaults_for_legacy_cells(tmp_path: Path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --symbols-file symbols.txt \\\n"
        "  --model-paths a.pkl,b.pkl \\\n"
        "  --top-n 2 \\\n"
        "  --allocation 2.5 \\\n"
        "  --live\n"
    )
    live_config = mod.extract_live_config_from_launch(launch)
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "leverage": 2.5,
        "top_n": 2,
        "median_monthly_pct": 40.0,
        "p10_monthly_pct": 10.0,
        "worst_dd_pct": 6.0,
        "n_neg": 0,
        "n_windows": 8,
    }])

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_config=live_config,
    ))

    assert result.passed is True
    assert result.best_cell["median_monthly_pct"] == 40.0


def test_launch_script_repeated_value_flags_use_live_argparse_last_value(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --symbols-file symbols.txt \\\n"
        "  --model-paths a.pkl,b.pkl \\\n"
        "  --top-n 1 \\\n"
        "  --allocation 1.0 \\\n"
        "  --top-n 3 \\\n"
        "  --allocation 2.5 \\\n"
        "  --live\n"
    )
    live_config = mod.extract_live_config_from_launch(launch)
    assert live_config["top_n"] == 3
    assert live_config["leverage"] == 2.5

    path = _write_sweep(tmp_path, cells=[
        {
            "fee_regime": "stress36x",
            "leverage": 1.0,
            "top_n": 1,
            "median_monthly_pct": 90.0,
            "p10_monthly_pct": 40.0,
            "worst_dd_pct": 4.0,
            "n_neg": 0,
            "n_windows": 8,
        },
        {
            "fee_regime": "stress36x",
            "leverage": 2.5,
            "top_n": 3,
            "median_monthly_pct": 40.0,
            "p10_monthly_pct": 10.0,
            "worst_dd_pct": 6.0,
            "n_neg": 0,
            "n_windows": 8,
        },
    ])

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_config=live_config,
    ))

    assert result.passed is True
    assert result.best_cell["top_n"] == 3
    assert result.best_cell["leverage"] == 2.5


def test_launch_script_parse_rejects_malformed_numeric_flags(tmp_path: Path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --symbols-file symbols.txt \\\n"
        "  --model-paths a.pkl,b.pkl \\\n"
        "  --top-n 2.5 \\\n"
        "  --allocation nan \\\n"
        "  --live\n"
    )

    try:
        mod.extract_live_config_from_launch(launch)
    except ValueError as exc:
        assert "--top-n must be an integer" in str(exc)
    else:
        raise AssertionError("expected malformed launch numeric flag to fail")


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--inv-vol-target-ann", "-0.1", "--inv-vol-target-ann must be >= 0"),
        ("--inv-vol-floor", "0", "--inv-vol-floor must be > 0"),
        ("--inv-vol-cap", "0.5", "--inv-vol-cap must be >= 1"),
    ],
)
def test_launch_script_parse_rejects_invalid_inv_vol_flags(
    tmp_path: Path,
    flag: str,
    value: str,
    message: str,
) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --model-paths a.pkl,b.pkl \\\n"
        f"  {flag} {value} \\\n"
        "  --live\n"
    )

    with pytest.raises(ValueError, match=message):
        mod.extract_live_config_from_launch(launch)


@pytest.mark.parametrize(
    ("flag_lines", "message"),
    [
        ("  --top-n 0 \\\n", "--top-n must be >= 1"),
        ("  --top-n 2 \\\n  --min-picks -1 \\\n", "--min-picks must be >= 0"),
        (
            "  --top-n 2 \\\n  --min-picks 3 \\\n",
            "--min-picks must be <= --top-n",
        ),
    ],
)
def test_launch_script_parse_rejects_invalid_pick_counts(
    tmp_path: Path,
    flag_lines: str,
    message: str,
) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --symbols-file symbols.txt \\\n"
        "  --model-paths a.pkl,b.pkl \\\n"
        f"{flag_lines}"
        "  --live\n"
    )

    try:
        mod.extract_live_config_from_launch(launch)
    except ValueError as exc:
        assert message in str(exc)
    else:
        raise AssertionError("expected invalid pick count to fail")


@pytest.mark.parametrize(
    ("flag_lines", "message"),
    [
        ("  --allocation 0 \\\n", "--allocation must be > 0"),
        ("  --allocation-temp 0 \\\n", "--allocation-temp must be > 0"),
        ("  --min-score 1.5 \\\n", "--min-score must be between 0 and 1"),
        ("  --min-dollar-vol -1 \\\n", "--min-dollar-vol must be >= 0"),
        ("  --max-spread-bps -1 \\\n", "--max-spread-bps must be >= 0"),
        ("  --min-vol-20d -0.1 \\\n", "--min-vol-20d must be >= 0"),
        ("  --max-vol-20d -0.1 \\\n", "--max-vol-20d must be >= 0"),
        (
            "  --max-ret-20d-rank-pct 1.5 \\\n",
            "--max-ret-20d-rank-pct must be between 0 and 1",
        ),
        (
            "  --min-ret-5d-rank-pct -0.1 \\\n",
            "--min-ret-5d-rank-pct must be between 0 and 1",
        ),
        ("  --regime-cs-iqr-max -0.1 \\\n", "--regime-cs-iqr-max must be >= 0"),
        ("  --regime-gate-window -1 \\\n", "--regime-gate-window must be >= 0"),
        ("  --vol-target-ann -0.1 \\\n", "--vol-target-ann must be >= 0"),
        (
            "  --no-picks-fallback SPY \\\n  --no-picks-fallback-alloc -0.1 \\\n",
            "--no-picks-fallback-alloc must be >= 0",
        ),
        (
            "  --conviction-alloc-low 0.9 \\\n  --conviction-alloc-high 0.8 \\\n",
            "--conviction-alloc-high must be > --conviction-alloc-low",
        ),
    ],
)
def test_launch_script_parse_rejects_invalid_live_numeric_domains(
    tmp_path: Path,
    flag_lines: str,
    message: str,
) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --symbols-file symbols.txt \\\n"
        "  --model-paths a.pkl,b.pkl \\\n"
        f"{flag_lines}"
        "  --live\n"
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        mod.extract_live_config_from_launch(launch)


def test_launch_script_parse_rejects_negative_uncertainty_penalty(tmp_path: Path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --symbols-file symbols.txt \\\n"
        "  --model-paths a.pkl,b.pkl \\\n"
        "  --score-uncertainty-penalty -0.1 \\\n"
        "  --live\n"
    )

    try:
        mod.extract_live_config_from_launch(launch)
    except ValueError as exc:
        assert "--score-uncertainty-penalty must be >= 0" in str(exc)
    else:
        raise AssertionError("expected negative uncertainty penalty to fail")


def test_main_returns_usage_error_for_bad_launch_numeric_flag(
    tmp_path: Path,
    capsys,
) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --top-n 2 \\\n"
        "  --allocation nope \\\n"
        "  --live\n"
    )
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 8,
        "fill_buffer_bps": 15.0,
    }])

    assert mod.main([str(path), "--launch-script", str(launch)]) == 2
    assert "launch-script parse error" in capsys.readouterr().err


def test_launch_script_parse_rejects_unknown_allocation_mode(tmp_path: Path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --allocation-mode softmax \\\n"
        "  --allocation-mode martingale \\\n"
        "  --live\n"
    )

    try:
        mod.extract_live_config_from_launch(launch)
    except ValueError as exc:
        assert "--allocation-mode must be one of" in str(exc)
        assert "martingale" in str(exc)
    else:
        raise AssertionError("expected unknown allocation mode to fail")


def test_main_returns_usage_error_for_bad_launch_allocation_mode(
    tmp_path: Path,
    capsys,
) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --allocation-mode nope \\\n"
        "  --live\n"
    )
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 8,
        "fill_buffer_bps": 15.0,
    }])

    assert mod.main([str(path), "--launch-script", str(launch)]) == 2
    captured = capsys.readouterr()
    assert "launch-script parse error" in captured.err
    assert "--allocation-mode" in captured.err


def test_main_rejects_launch_with_unmodeled_live_sidecars_by_default(
    tmp_path: Path,
    capsys,
) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --model-paths analysis/xgbnew_daily/custom/alltrain_seed0.pkl \\\n"
        "  --top-n 2 \\\n"
        "  --allocation 2.0 \\\n"
        "  --crypto-weekend \\\n"
        "  --eod-deleverage \\\n"
        "  --live\n"
    )
    path = _write_sweep(
        tmp_path,
        cells=[{
            "fee_regime": "stress36x",
            "leverage": 2.0,
            "top_n": 2,
            "median_monthly_pct": 80.0,
            "p10_monthly_pct": 40.0,
            "worst_dd_pct": 5.0,
            "n_neg": 0,
            "n_windows": 8,
            "fill_buffer_bps": 15.0,
        }],
        model_paths=["analysis/xgbnew_daily/custom/alltrain_seed0.pkl"],
    )

    assert mod.main([str(path), "--launch-script", str(launch)]) == 3
    captured = capsys.readouterr()
    assert "unmodeled live sidecars" in captured.err
    assert "--crypto-weekend" in captured.err
    assert "--eod-deleverage" in captured.err


def test_main_can_allow_unmodeled_live_sidecars_for_ops_inspection(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --model-paths analysis/xgbnew_daily/custom/alltrain_seed0.pkl \\\n"
        "  --top-n 2 \\\n"
        "  --allocation 2.0 \\\n"
        "  --crypto-weekend \\\n"
        "  --eod-deleverage \\\n"
        "  --live\n"
    )
    path = _write_sweep(
        tmp_path,
        cells=[{
            "fee_regime": "stress36x",
            "leverage": 2.0,
            "top_n": 2,
            "median_monthly_pct": 80.0,
            "p10_monthly_pct": 40.0,
            "worst_dd_pct": 5.0,
            "n_neg": 0,
            "n_windows": 8,
            "fill_buffer_bps": 15.0,
        }],
        model_paths=["analysis/xgbnew_daily/custom/alltrain_seed0.pkl"],
        symbols_file="symbol_lists/stocks_wide_1000_v1.txt",
        data_root="trainingdata",
        blend_mode="mean",
    )

    assert mod.main([
        str(path),
        "--launch-script", str(launch),
        "--allow-unmodeled-live-sidecars",
    ]) == 0


def test_launch_script_filter_parses_string_booleans_without_truthiness_bug(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[
        {
            "fee_regime": "stress36x",
            "leverage": 2.5,
            "top_n": 2,
            "hold_through": "False",
            "median_monthly_pct": 90.0,
            "p10_monthly_pct": 30.0,
            "worst_dd_pct": 5.0,
            "n_neg": 0,
            "n_windows": 8,
        },
        {
            "fee_regime": "stress36x",
            "leverage": 2.5,
            "top_n": 2,
            "hold_through": "true",
            "median_monthly_pct": 40.0,
            "p10_monthly_pct": 10.0,
            "worst_dd_pct": 6.0,
            "n_neg": 0,
            "n_windows": 8,
        },
    ])

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_config={
            "leverage": 2.5,
            "top_n": 2,
            "hold_through": True,
        },
    ))

    assert result.passed is True
    assert result.best_cell["median_monthly_pct"] == 40.0
    assert result.best_cell["hold_through"] == "true"


def test_launch_script_filter_rejects_fractional_integer_knobs(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[
        {
            "fee_regime": "stress36x",
            "leverage": 2.5,
            "top_n": 2.5,
            "median_monthly_pct": 90.0,
            "p10_monthly_pct": 30.0,
            "worst_dd_pct": 5.0,
            "n_neg": 0,
            "n_windows": 8,
        },
        {
            "fee_regime": "stress36x",
            "leverage": 2.5,
            "top_n": "2",
            "median_monthly_pct": 40.0,
            "p10_monthly_pct": 10.0,
            "worst_dd_pct": 6.0,
            "n_neg": 0,
            "n_windows": 8,
        },
    ])

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
        live_config={
            "leverage": 2.5,
            "top_n": 2,
        },
    ))

    assert result.passed is True
    assert result.best_cell["median_monthly_pct"] == 40.0
    assert result.best_cell["top_n"] == "2"


def test_fail_fast_string_false_does_not_reject_cell(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 40.0,
        "p10_monthly_pct": 10.0,
        "worst_dd_pct": 6.0,
        "n_neg": 0,
        "n_windows": 8,
        "fail_fast_triggered": "False",
    }])

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
    ))

    assert result.passed is True


def test_malformed_fail_fast_boolean_fails_closed(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 40.0,
        "p10_monthly_pct": 10.0,
        "worst_dd_pct": 6.0,
        "n_neg": 0,
        "n_windows": 8,
        "fail_fast_triggered": "definitely-not",
    }])

    result = mod.evaluate_sweep(path, mod.GateConfig(
        fee_regime="stress36x",
        min_median_monthly_pct=27.0,
        max_worst_dd_pct=25.0,
        max_neg_windows=0,
        min_oos_days=100,
        min_windows=1,
        min_p10_monthly_pct=None,
    ))

    assert result.passed is False
    assert result.reason == "fail_fast_triggered invalid"


def test_main_returns_nonzero_when_no_cell_passes(tmp_path: Path, capsys) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 26.0,
        "p10_monthly_pct": 1.0,
        "worst_dd_pct": 10.0,
        "n_neg": 0,
        "n_windows": 8,
    }])

    assert mod.main([str(path)]) == 3
    assert "FAIL" in capsys.readouterr().out


def test_main_default_requires_stress_fill_buffer(tmp_path: Path, capsys) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 8,
        "fill_buffer_bps": 5.0,
    }])

    assert mod.main([str(path)]) == 3
    out = capsys.readouterr().out
    assert "fill_buffer_bps 5.00 < 15.00" in out


def test_main_can_disable_fill_buffer_gate_for_legacy_inspection(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 8,
        "fill_buffer_bps": 5.0,
    }])

    assert mod.main([str(path), "--min-fill-buffer-bps", "-1"]) == 0


def test_main_default_rejects_skip_stress_cells(tmp_path: Path, capsys) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 8,
        "fill_buffer_bps": 15.0,
        "skip_prob": 0.25,
        "skip_seed": 7,
    }])

    assert mod.main([str(path)]) == 3
    assert "skip_prob 0.2500 != 0.0" in capsys.readouterr().out


def test_main_can_allow_skip_stress_cells_for_research_inspection(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 8,
        "fill_buffer_bps": 15.0,
        "skip_prob": 0.25,
        "skip_seed": 7,
    }])

    assert mod.main([str(path), "--allow-skip-stress-cells"]) == 0


def test_main_deploy_fee_regime_does_not_inherit_stress_fill_buffer(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "deploy",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 8,
        "fill_buffer_bps": 5.0,
    }])

    assert mod.main([str(path), "--fee-regime", "deploy"]) == 0


def test_main_default_rejects_partial_sweep(tmp_path: Path, capsys) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, complete=False, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 8,
    }])

    assert mod.main([str(path)]) == 3
    assert "sweep complete flag is False" in capsys.readouterr().out


def test_main_can_allow_partial_sweep_for_research_inspection(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, complete=False, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 8,
        "fill_buffer_bps": 15.0,
    }])

    assert mod.main([str(path), "--allow-partial-sweep"]) == 0


def test_main_default_rejects_partial_expected_windows(tmp_path: Path, capsys) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 3,
        "expected_n_windows": 4,
        "fill_buffer_bps": 15.0,
    }])

    assert mod.main([str(path)]) == 3
    assert "n_windows 3 < 4" in capsys.readouterr().out


def test_main_can_ignore_expected_windows_for_legacy_inspection(tmp_path: Path) -> None:
    mod = _load_module()
    path = _write_sweep(tmp_path, cells=[{
        "fee_regime": "stress36x",
        "median_monthly_pct": 80.0,
        "p10_monthly_pct": 40.0,
        "worst_dd_pct": 5.0,
        "n_neg": 0,
        "n_windows": 3,
        "expected_n_windows": 4,
        "fill_buffer_bps": 15.0,
    }])

    assert mod.main([str(path), "--ignore-expected-windows"]) == 0
