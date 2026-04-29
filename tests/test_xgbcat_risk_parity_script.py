"""Tests for scripts/xgbcat_risk_parity_wide_120d.sh."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "xgbcat_risk_parity_wide_120d.sh"
COMPLETION_SCRIPT = REPO / "scripts" / "write_xgbcat_risk_parity_completion.py"
UNIT_MODEL_BYTES = b"unit model\n"
UNIT_MODEL_SHA256 = hashlib.sha256(UNIT_MODEL_BYTES).hexdigest()
UNIT_SYMBOLS_BYTES = b"AAPL\nMSFT\n"
UNIT_SYMBOLS_SHA256 = hashlib.sha256(UNIT_SYMBOLS_BYTES).hexdigest()
UNIT_MONTHLY_RETURN_PCTS = [
    12.0,
    22.0,
    30.0,
    30.0,
    30.0,
    30.0,
    30.0,
    30.0,
    30.0,
    30.0,
]
UNIT_N_WINDOWS = len(UNIT_MONTHLY_RETURN_PCTS)
UNIT_WINDOW_SORTINO_VALUES = [2.0] * UNIT_N_WINDOWS
UNIT_WINDOW_DRAWDOWN_PCTS = [20.0] * UNIT_N_WINDOWS
UNIT_WINDOW_TIME_UNDER_WATER_PCTS = [0.0] * UNIT_N_WINDOWS
UNIT_WINDOW_ULCER_INDEXES = [0.0] * UNIT_N_WINDOWS
UNIT_WINDOW_ACTIVE_DAY_PCTS = [100.0] * UNIT_N_WINDOWS
UNIT_WINDOW_WORST_INTRADAY_DD_PCTS = [3.0] * UNIT_N_WINDOWS
UNIT_WINDOW_AVG_INTRADAY_DD_PCTS = [1.0] * UNIT_N_WINDOWS
UNIT_WINDOW_START_DATES = [
    "2025-12-18",
    "2025-12-21",
    "2025-12-24",
    "2025-12-27",
    "2025-12-30",
    "2026-01-02",
    "2026-01-05",
    "2026-01-08",
    "2026-01-11",
    "2026-01-14",
]
UNIT_WINDOW_END_DATES = [
    "2026-01-16",
    "2026-01-19",
    "2026-01-22",
    "2026-01-25",
    "2026-01-28",
    "2026-01-31",
    "2026-02-03",
    "2026-02-06",
    "2026-02-09",
    "2026-02-12",
]
STRATEGY_PARAM_DEFAULTS = {
    "allocation_mode": "equal",
    "allocation_temp": 1.0,
    "conviction_alloc_high": 0.85,
    "conviction_alloc_low": 0.55,
    "conviction_scaled_alloc": False,
    "hold_through": True,
    "inference_max_spread_bps": 30.0,
    "inference_max_vol_20d": 0.0,
    "inference_min_dolvol": 5_000_000.0,
    "inference_min_vol_20d": 0.0,
    "inv_vol_cap": 2.0,
    "inv_vol_floor": 0.08,
    "inv_vol_target_ann": 0.0,
    "leverage": 2.0,
    "max_ret_20d_rank_pct": 1.0,
    "min_picks": 0,
    "min_ret_5d_rank_pct": 0.0,
    "min_score": 0.58,
    "no_picks_fallback_alloc_scale": 0.0,
    "no_picks_fallback_symbol": "",
    "opportunistic_entry_discount_bps": 0.0,
    "opportunistic_watch_n": 0,
    "regime_cs_iqr_max": 0.0,
    "regime_cs_skew_min": -1e9,
    "regime_gate_window": 0,
    "score_uncertainty_penalty": 0.0,
    "top_n": 3,
    "vol_target_ann": 0.0,
}


def _valid_result_payload() -> dict[str, object]:
    passing_strategy = {
        **STRATEGY_PARAM_DEFAULTS,
        "any_fail_fast_triggered": False,
        "fee_regimes": ["deploy", "prod10bps", "stress36x"],
        "fill_buffer_bps_values": [0.0, 5.0, 10.0, 20.0],
        "max_n_neg": 0,
        "max_avg_intraday_dd_pct": 1.0,
        "max_time_under_water_pct": 0.0,
        "max_ulcer_index": 0.0,
        "max_expected_n_windows": UNIT_N_WINDOWS,
        "max_worst_intraday_dd_pct": 3.0,
        "max_worst_dd_pct": 20.0,
        "min_n_windows": UNIT_N_WINDOWS,
        "n_friction_cells": 12,
        "production_target_pass": True,
        "required_min_n_windows": UNIT_N_WINDOWS,
        "skip_prob_values": [0.0],
        "skip_seed_values": [0],
        "worst_fee_regime_by_pain": "stress36x",
        "worst_fill_buffer_bps_by_pain": 20.0,
        "worst_goodness_score": 1.0,
        "worst_median_active_day_pct": 100.0,
        "worst_median_monthly_pct": 30.0,
        "worst_median_sortino": 2.0,
        "worst_min_active_day_pct": 100.0,
        "worst_pain_adjusted_goodness_score": -9.0,
        "worst_p10_monthly_pct": 21.0,
        "worst_robust_goodness_score": -9.0,
    }
    return {
        "best_production_target_strategy": passing_strategy,
        "cells": [
            {
                **STRATEGY_PARAM_DEFAULTS,
                "expected_n_windows": UNIT_N_WINDOWS,
                "avg_intraday_dd_pct": 1.0,
                "fail_fast_triggered": False,
                "fee_regime": fee_regime,
                "fill_buffer_bps": fill_buffer_bps,
                "goodness_score": 1.0,
                "median_active_day_pct": 100.0,
                "median_monthly_pct": 30.0,
                "median_sortino": 2.0,
                "mean_abs_neg_monthly_pct": 0.0,
                "min_active_day_pct": 100.0,
                "monthly_return_pcts": UNIT_MONTHLY_RETURN_PCTS,
                "n_neg": 0,
                "n_windows": UNIT_N_WINDOWS,
                "pain_adjusted_goodness_score": -9.0,
                "p10_monthly_pct": 21.0,
                "robust_goodness_score": -9.0,
                "time_under_water_pct": 0.0,
                "ulcer_index": 0.0,
                "window_drawdown_pcts": UNIT_WINDOW_DRAWDOWN_PCTS,
                "window_sortino_values": UNIT_WINDOW_SORTINO_VALUES,
                "window_time_under_water_pcts": UNIT_WINDOW_TIME_UNDER_WATER_PCTS,
                "window_ulcer_indexes": UNIT_WINDOW_ULCER_INDEXES,
                "window_active_day_pcts": UNIT_WINDOW_ACTIVE_DAY_PCTS,
                "window_worst_intraday_dd_pcts": UNIT_WINDOW_WORST_INTRADAY_DD_PCTS,
                "window_avg_intraday_dd_pcts": UNIT_WINDOW_AVG_INTRADAY_DD_PCTS,
                "window_start_dates": UNIT_WINDOW_START_DATES,
                "window_end_dates": UNIT_WINDOW_END_DATES,
                "worst_intraday_dd_pct": 3.0,
                "worst_dd_pct": 20.0,
            }
            for fee_regime in ("deploy", "prod10bps", "stress36x")
            for fill_buffer_bps in (0.0, 5.0, 10.0, 20.0)
        ],
        "complete": True,
        "fail_fast": {
            "max_dd_pct": 20.0,
            "max_intraday_dd_pct": 20.0,
            "neg_windows": 1,
            "score": -1_000_000_000.0,
        },
        "fee_regimes": {
            "deploy": {
                "commission_bps": 0.0,
                "fee_rate": 0.0000278,
                "fill_buffer_bps": 5.0,
            },
            "prod10bps": {
                "commission_bps": 0.0,
                "fee_rate": 0.001,
                "fill_buffer_bps": 5.0,
            },
            "stress36x": {
                "commission_bps": 10.0,
                "fee_rate": 0.001,
                "fill_buffer_bps": 15.0,
            },
        },
        "friction_robust_strategies": [passing_strategy],
        "goodness_weights": {
            "dd_coef": 1.0,
            "neg_coef": 100.0,
            "p10_coef": 1.0,
        },
        "model_paths": ["models/model0.pkl"],
        "model_sha256": [UNIT_MODEL_SHA256],
        "n_cells": 12,
        "n_friction_robust_strategies": 1,
        "n_production_target_pass": 1,
        "oos_end": "2026-04-20",
        "oos_start": "2025-12-18",
        "production_target": {
            "expected_windows_required": True,
            "max_dd_pct": 25.0,
            "max_neg_windows": 0,
            "median_monthly_pct": 27.0,
            "min_windows": 1,
        },
        "pain_adjusted_goodness_weights": {
            "tuw_coef": 0.25,
            "ulcer_coef": 1.0,
        },
        "robust_goodness_weights": {
            "dd_coef": 1.5,
            "neg_count_coef": 50.0,
            "neg_magnitude_coef": 2.0,
            "p10_coef": 1.0,
        },
        "stride_days": 3,
        "symbols_file": "symbols.txt",
        "window_days": 30,
    }


def _write_valid_result(path: Path, payload: dict[str, object] | None = None) -> None:
    path.write_text(
        json.dumps(payload or _valid_result_payload(), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_manifest(
    path: Path,
    *,
    fee_regimes: str = "deploy,prod10bps,stress36x",
    fill_buffer_bps_grid: str = "0,5,10,20",
    leverage_grid: str = "1.5,2.0,2.25,2.5",
    fail_fast_max_dd_pct: str = "20",
    fail_fast_max_intraday_dd_pct: str = "20",
    fail_fast_neg_windows: str = "1",
) -> None:
    repo = path.parent
    model_path = repo / "models" / "model0.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(UNIT_MODEL_BYTES)
    (repo / "model_paths.txt").write_text(f"{model_path}\n", encoding="utf-8")
    symbols_path = repo / "symbols.txt"
    symbols_path.write_bytes(UNIT_SYMBOLS_BYTES)
    path.write_text(
        json.dumps(
            {
                "config": {
                    "allocation_mode_grid": "equal,score_norm,softmax",
                    "allocation_temp_grid": "0.5,1.0",
                    "checkpoint_every_cells": "50",
                    "fee_regimes": fee_regimes,
                    "fail_fast_max_dd_pct": fail_fast_max_dd_pct,
                    "fail_fast_max_intraday_dd_pct": fail_fast_max_intraday_dd_pct,
                    "fail_fast_neg_windows": fail_fast_neg_windows,
                    "fill_buffer_bps_grid": fill_buffer_bps_grid,
                    "inference_max_spread_bps_grid": "20,30",
                    "inference_max_vol_grid": "0.0,0.80",
                    "inference_min_dolvol_grid": "50000000,100000000",
                    "inference_min_vol_grid": "0.12,0.15",
                    "inv_vol_cap": "2.0",
                    "inv_vol_floor": "0.08",
                    "inv_vol_target_grid": "0.0,0.20,0.25,0.30",
                    "leverage_grid": leverage_grid,
                    "min_dollar_vol": "50000000",
                    "min_score_grid": "0.58,0.62,0.66,0.70",
                    "oos_end": "2026-04-20",
                    "oos_start": "2025-12-18",
                    "regime_cs_skew_min_grid": "0.50,0.75,1.00",
                    "score_uncertainty_penalty_grid": "0.0,0.5,1.0",
                    "stride_days": "3",
                    "top_n_grid": "2,3,4",
                    "vol_target_ann_grid": "0.0,0.18,0.22",
                    "window_days": "30",
                },
                "models": [
                    {
                        "family": "xgb",
                        "path": str(model_path),
                        "sha256": UNIT_MODEL_SHA256,
                    },
                ],
                "cat_dir": str(repo / "cat_models"),
                "cat_model_count": 0,
                "fixed_flags": {
                    "fast_features": True,
                    "hold_through": True,
                    "require_production_target": True,
                },
                "done_marker": str(repo / "sweep.done"),
                "model_paths_file": str(repo / "model_paths.txt"),
                "output_dir": str(repo),
                "repo": str(repo),
                "result_artifacts_file": str(repo / "result_artifacts.json"),
                "run_log": str(repo / "sweep_stdout.log"),
                "script": "scripts/xgbcat_risk_parity_wide_120d.sh",
                "symbols_file": str(symbols_path),
                "symbols_sha256": UNIT_SYMBOLS_SHA256,
                "sweep_args_file": str(repo / "sweep_args.txt"),
                "venv": ".venv313",
                "xgb_dir": str(repo / "models"),
                "xgb_model_count": 1,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_sweep_args(path: Path) -> None:
    repo = path.parent
    path.write_text(
        "\n".join(
            [
                "--symbols-file",
                str(repo / "symbols.txt"),
                "--model-paths",
                str(repo / "models" / "model0.pkl"),
                "--oos-start",
                "2025-12-18",
                "--oos-end",
                "2026-04-20",
                "--window-days",
                "30",
                "--stride-days",
                "3",
                "--leverage-grid",
                "1.5,2.0,2.25,2.5",
                "--min-score-grid",
                "0.58,0.62,0.66,0.70",
                "--top-n-grid",
                "2,3,4",
                "--fill-buffer-bps-grid",
                "0,5,10,20",
                "--fee-regimes",
                "deploy,prod10bps,stress36x",
                "--min-dollar-vol",
                "50000000",
                "--inference-min-dolvol-grid",
                "50000000,100000000",
                "--inference-min-vol-grid",
                "0.12,0.15",
                "--inference-max-vol-grid",
                "0.0,0.80",
                "--inference-max-spread-bps-grid",
                "20,30",
                "--vol-target-ann-grid",
                "0.0,0.18,0.22",
                "--inv-vol-target-grid",
                "0.0,0.20,0.25,0.30",
                "--inv-vol-floor",
                "0.08",
                "--inv-vol-cap",
                "2.0",
                "--regime-cs-skew-min-grid",
                "0.50,0.75,1.00",
                "--allocation-mode-grid",
                "equal,score_norm,softmax",
                "--allocation-temp-grid",
                "0.5,1.0",
                "--score-uncertainty-penalty-grid",
                "0.0,0.5,1.0",
                "--fail-fast-max-dd-pct",
                "20",
                "--fail-fast-max-intraday-dd-pct",
                "20",
                "--fail-fast-neg-windows",
                "1",
                "--checkpoint-every-cells",
                "50",
                "--output-dir",
                str(repo),
                "--hold-through",
                "--require-production-target",
                "--fast-features",
                "--verbose",
            ],
        )
        + "\n",
        encoding="utf-8",
    )


def _load_completion_module():
    spec = importlib.util.spec_from_file_location("xgbcat_completion", COMPLETION_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_fake_repo(tmp_path: Path) -> Path:
    fake_repo = tmp_path / "repo"
    (fake_repo / ".venv313" / "bin").mkdir(parents=True)
    (fake_repo / ".venv313" / "bin" / "activate").write_text(
        "export XGBCAT_TEST_VENV=1\n",
        encoding="utf-8",
    )
    symbols = fake_repo / "symbol_lists" / "stocks_wide_fresh0401_photonics_2500_v1.txt"
    symbols.parent.mkdir(parents=True)
    symbols.write_text("AAPL\nMSFT\n", encoding="utf-8")
    for model_dir in [
        fake_repo / "analysis" / "xgbnew_daily" / "track1_oos120d_xgb",
        fake_repo / "analysis" / "xgbnew_daily" / "track1_oos120d_cat",
    ]:
        model_dir.mkdir(parents=True)
        (model_dir / "alltrain_seed0.pkl").write_text("model\n", encoding="utf-8")
    return fake_repo


def _install_fake_sweep_module(fake_repo: Path) -> None:
    package = fake_repo / "xgbnew"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "sweep_ensemble_grid.py").write_text(
        """
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import sys


def _arg_value(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


print("fake sweep launched")
print("fill_buffer_bps_grid=" + _arg_value("--fill-buffer-bps-grid"))
print("fee_regimes=" + _arg_value("--fee-regimes"))
out = Path(_arg_value("--output-dir"))
out.mkdir(parents=True, exist_ok=True)
(out / "argv_seen.json").write_text(json.dumps(sys.argv[1:]) + "\\n", encoding="utf-8")
if not os.environ.get("FAKE_SWEEP_NO_RESULT"):
    model_paths = [
        item.strip()
        for item in _arg_value("--model-paths").split(",")
    ]
    fee_regimes = [item.strip() for item in _arg_value("--fee-regimes").split(",")]
    fill_buffers = [
        float(item.strip())
        for item in _arg_value("--fill-buffer-bps-grid").split(",")
    ]
    strategy_params = {
        "allocation_mode": "equal",
        "allocation_temp": 1.0,
        "conviction_alloc_high": 0.85,
        "conviction_alloc_low": 0.55,
        "conviction_scaled_alloc": False,
        "hold_through": True,
        "inference_max_spread_bps": 30.0,
        "inference_max_vol_20d": 0.0,
        "inference_min_dolvol": 5000000.0,
        "inference_min_vol_20d": 0.0,
        "inv_vol_cap": 2.0,
        "inv_vol_floor": 0.08,
        "inv_vol_target_ann": 0.0,
        "leverage": 2.0,
        "max_ret_20d_rank_pct": 1.0,
        "min_picks": 0,
        "min_ret_5d_rank_pct": 0.0,
        "min_score": 0.58,
        "no_picks_fallback_alloc_scale": 0.0,
        "no_picks_fallback_symbol": "",
        "opportunistic_entry_discount_bps": 0.0,
        "opportunistic_watch_n": 0,
        "regime_cs_iqr_max": 0.0,
        "regime_cs_skew_min": -1e9,
        "regime_gate_window": 0,
        "score_uncertainty_penalty": 0.0,
        "top_n": 3,
        "vol_target_ann": 0.0,
    }
    passing_strategy = {
        **strategy_params,
        "any_fail_fast_triggered": False,
        "fee_regimes": fee_regimes,
        "fill_buffer_bps_values": fill_buffers,
        "max_n_neg": 0,
        "max_avg_intraday_dd_pct": 1.0,
        "max_time_under_water_pct": 0.0,
        "max_ulcer_index": 0.0,
        "max_expected_n_windows": 10,
        "max_worst_intraday_dd_pct": 3.0,
        "max_worst_dd_pct": 20.0,
        "min_n_windows": 10,
        "n_friction_cells": len(fee_regimes) * len(fill_buffers),
        "production_target_pass": True,
        "required_min_n_windows": 10,
        "skip_prob_values": [0.0],
        "skip_seed_values": [0],
        "worst_fee_regime_by_pain": "stress36x",
        "worst_fill_buffer_bps_by_pain": max(fill_buffers),
        "worst_goodness_score": 1.0,
        "worst_median_active_day_pct": 100.0,
        "worst_median_monthly_pct": 30.0,
        "worst_median_sortino": 2.0,
        "worst_min_active_day_pct": 100.0,
        "worst_pain_adjusted_goodness_score": -9.0,
        "worst_p10_monthly_pct": 21.0,
        "worst_robust_goodness_score": -9.0,
    }
    payload = {
        "best_production_target_strategy": passing_strategy,
        "cells": [
            {
                **strategy_params,
                "expected_n_windows": 10,
                "avg_intraday_dd_pct": 1.0,
                "fail_fast_triggered": False,
                "fee_regime": fee_regime,
                "fill_buffer_bps": fill_buffer,
                "goodness_score": 1.0,
                "median_active_day_pct": 100.0,
                "median_monthly_pct": 30.0,
                "median_sortino": 2.0,
                "mean_abs_neg_monthly_pct": 0.0,
                "min_active_day_pct": 100.0,
                "monthly_return_pcts": [
                    12.0,
                    22.0,
                    30.0,
                    30.0,
                    30.0,
                    30.0,
                    30.0,
                    30.0,
                    30.0,
                    30.0,
                ],
                "n_neg": 0,
                "n_windows": 10,
                "pain_adjusted_goodness_score": -9.0,
                "p10_monthly_pct": 21.0,
                "robust_goodness_score": -9.0,
                "time_under_water_pct": 0.0,
                "ulcer_index": 0.0,
                "window_drawdown_pcts": [20.0] * 10,
                "window_sortino_values": [2.0] * 10,
                "window_time_under_water_pcts": [0.0] * 10,
                "window_ulcer_indexes": [0.0] * 10,
                "window_active_day_pcts": [100.0] * 10,
                "window_worst_intraday_dd_pcts": [3.0] * 10,
                "window_avg_intraday_dd_pcts": [1.0] * 10,
                "window_start_dates": [
                    "2025-12-18",
                    "2025-12-21",
                    "2025-12-24",
                    "2025-12-27",
                    "2025-12-30",
                    "2026-01-02",
                    "2026-01-05",
                    "2026-01-08",
                    "2026-01-11",
                    "2026-01-14",
                ],
                "window_end_dates": [
                    "2026-01-16",
                    "2026-01-19",
                    "2026-01-22",
                    "2026-01-25",
                    "2026-01-28",
                    "2026-01-31",
                    "2026-02-03",
                    "2026-02-06",
                    "2026-02-09",
                    "2026-02-12",
                ],
                "worst_intraday_dd_pct": 3.0,
                "worst_dd_pct": 20.0,
            }
            for fee_regime in fee_regimes
            for fill_buffer in fill_buffers
        ],
        "complete": True,
        "friction_robust_strategies": [passing_strategy],
        "fail_fast": {
            "max_dd_pct": 20.0,
            "max_intraday_dd_pct": 20.0,
            "neg_windows": 1,
            "score": -1_000_000_000.0,
        },
        "fee_regimes": {
            "deploy": {
                "commission_bps": 0.0,
                "fee_rate": 0.0000278,
                "fill_buffer_bps": 5.0,
            },
            "prod10bps": {
                "commission_bps": 0.0,
                "fee_rate": 0.001,
                "fill_buffer_bps": 5.0,
            },
            "stress36x": {
                "commission_bps": 10.0,
                "fee_rate": 0.001,
                "fill_buffer_bps": 15.0,
            },
        },
        "model_paths": model_paths,
        "goodness_weights": {
            "dd_coef": 1.0,
            "neg_coef": 100.0,
            "p10_coef": 1.0,
        },
        "model_sha256": [_sha256(Path(item)) for item in model_paths],
        "n_cells": len(fee_regimes) * len(fill_buffers),
        "n_friction_robust_strategies": 1,
        "n_production_target_pass": 1,
        "oos_end": _arg_value("--oos-end"),
        "oos_start": _arg_value("--oos-start"),
        "production_target": {
            "expected_windows_required": True,
            "max_dd_pct": 25.0,
            "max_neg_windows": 0,
            "median_monthly_pct": 27.0,
            "min_windows": 1,
        },
        "pain_adjusted_goodness_weights": {
            "tuw_coef": 0.25,
            "ulcer_coef": 1.0,
        },
        "robust_goodness_weights": {
            "dd_coef": 1.5,
            "neg_count_coef": 50.0,
            "neg_magnitude_coef": 2.0,
            "p10_coef": 1.0,
        },
        "stride_days": int(_arg_value("--stride-days")),
        "symbols_file": _arg_value("--symbols-file"),
        "window_days": int(_arg_value("--window-days")),
    }
    if os.environ.get("FAKE_SWEEP_INCOMPLETE_RESULT"):
        payload["complete"] = False
    if os.environ.get("FAKE_SWEEP_NO_PRODUCTION_PASS"):
        payload["best_production_target_strategy"] = None
        payload["friction_robust_strategies"][0]["production_target_pass"] = False
        payload["friction_robust_strategies"][0]["worst_median_monthly_pct"] = 10.0
        payload["friction_robust_strategies"][0]["worst_p10_monthly_pct"] = 10.0
        payload["friction_robust_strategies"][0]["worst_goodness_score"] = -10.0
        payload["friction_robust_strategies"][0]["worst_robust_goodness_score"] = -20.0
        payload["friction_robust_strategies"][0]["worst_pain_adjusted_goodness_score"] = -20.0
        for cell in payload["cells"]:
            cell["median_monthly_pct"] = 10.0
            cell["monthly_return_pcts"] = [10.0] * 10
            cell["p10_monthly_pct"] = 10.0
            cell["goodness_score"] = -10.0
            cell["robust_goodness_score"] = -20.0
            cell["pain_adjusted_goodness_score"] = -20.0
        payload["n_production_target_pass"] = 0
    (out / "sweep_20000101_000000.json").write_text(
        json.dumps(payload, sort_keys=True) + "\\n",
        encoding="utf-8",
    )
if os.environ.get("FAKE_SWEEP_EXIT"):
    print("fake sweep failing")
    raise SystemExit(int(os.environ["FAKE_SWEEP_EXIT"]))
""".lstrip(),
        encoding="utf-8",
    )


def _run_script(fake_repo: Path, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "REPO": str(fake_repo),
        "DRY_RUN": "1",
    }
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(SCRIPT)],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_key_value_file(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        rows[key] = value
    return rows


def _publish_valid_completion(out: Path) -> tuple[Path, Path]:
    module = _load_completion_module()
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )
    return manifest, done_marker


def _rewrite_marker_manifest_hash(done_marker: Path, manifest: Path) -> None:
    done = _read_key_value_file(done_marker)
    done["manifest_sha256"] = _sha256(manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )


def _verify_completion_marker(done_marker: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_xgbcat_risk_parity_script_syntax_is_valid() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_xgbcat_completion_writer_publishes_hash_bound_marker(tmp_path: Path) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)

    module.publish_completion(
        done_marker=out / "sweep.done",
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )

    done = _read_key_value_file(out / "sweep.done")
    results_manifest = json.loads((out / "result_artifacts.json").read_text(encoding="utf-8"))
    assert results_manifest["results"] == [
        {
            "path": str(result.resolve()),
            "sha256": _sha256(result),
            "size_bytes": result.stat().st_size,
        },
    ]
    assert done["schema_version"] == "1"
    assert done["manifest_sha256"] == _sha256(manifest)
    assert done["sweep_args_sha256"] == _sha256(sweep_args)
    assert done["run_log_sha256"] == _sha256(run_log)
    assert done["result_artifacts_sha256"] == _sha256(out / "result_artifacts.json")
    assert done["result_count"] == "1"


def test_xgbcat_completion_writer_verifies_hash_bound_marker(tmp_path: Path) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )

    module.verify_completion(done_marker)


def test_xgbcat_completion_writer_verify_rejects_sweep_args_manifest_drift(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )
    args = sweep_args.read_text(encoding="utf-8").splitlines()
    args[args.index("--leverage-grid") + 1] = "1.5,2.0"
    sweep_args.write_text("\n".join(args) + "\n", encoding="utf-8")
    done = _read_key_value_file(done_marker)
    done["sweep_args_sha256"] = _sha256(sweep_args)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "sweep args --leverage-grid does not match manifest" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_unchecked_sweep_arg_drift(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )
    args = sweep_args.read_text(encoding="utf-8").splitlines()
    args[args.index("--min-score-grid") + 1] = "0.01"
    sweep_args.write_text("\n".join(args) + "\n", encoding="utf-8")
    done = _read_key_value_file(done_marker)
    done["sweep_args_sha256"] = _sha256(sweep_args)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "sweep args --min-score-grid does not match manifest" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_unexpected_sweep_arg_flag(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )
    args = sweep_args.read_text(encoding="utf-8").splitlines()
    args.extend(["--soft-fill-cheat", "1"])
    sweep_args.write_text("\n".join(args) + "\n", encoding="utf-8")
    done = _read_key_value_file(done_marker)
    done["sweep_args_sha256"] = _sha256(sweep_args)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "sweep args unexpected flag: --soft-fill-cheat" in proc.stderr


def test_xgbcat_completion_writer_verify_requires_sweep_safety_flags(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )
    args = [
        arg
        for arg in sweep_args.read_text(encoding="utf-8").splitlines()
        if arg != "--require-production-target"
    ]
    sweep_args.write_text("\n".join(args) + "\n", encoding="utf-8")
    done = _read_key_value_file(done_marker)
    done["sweep_args_sha256"] = _sha256(sweep_args)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "sweep args missing required safety flag: --require-production-target" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_sweep_args_model_path_drift(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )
    args = sweep_args.read_text(encoding="utf-8").splitlines()
    args[args.index("--model-paths") + 1] = str(out / "models" / "other.pkl")
    sweep_args.write_text("\n".join(args) + "\n", encoding="utf-8")
    done = _read_key_value_file(done_marker)
    done["sweep_args_sha256"] = _sha256(sweep_args)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "sweep args --model-paths does not match manifest models" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_empty_sweep_model_path_token(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )
    args = sweep_args.read_text(encoding="utf-8").splitlines()
    model_paths_idx = args.index("--model-paths") + 1
    args[model_paths_idx] = f"{args[model_paths_idx]},"
    sweep_args.write_text("\n".join(args) + "\n", encoding="utf-8")
    done = _read_key_value_file(done_marker)
    done["sweep_args_sha256"] = _sha256(sweep_args)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "sweep args --model-paths must not contain empty paths" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_sweep_args_output_dir_drift(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )
    args = sweep_args.read_text(encoding="utf-8").splitlines()
    args[args.index("--output-dir") + 1] = str(tmp_path / "other_out")
    sweep_args.write_text("\n".join(args) + "\n", encoding="utf-8")
    done = _read_key_value_file(done_marker)
    done["sweep_args_sha256"] = _sha256(sweep_args)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "sweep args --output-dir does not match manifest output_dir" in proc.stderr


def test_xgbcat_completion_writer_refuses_to_overwrite_existing_marker(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    results_manifest = out / "result_artifacts.json"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=results_manifest,
        output_dir=out,
        run_started_ns=0,
    )
    marker_before = done_marker.read_bytes()
    results_manifest_before = results_manifest.read_bytes()
    run_log.write_text("replacement log\n", encoding="utf-8")

    try:
        module.publish_completion(
            done_marker=done_marker,
            manifest=manifest,
            sweep_args=sweep_args,
            run_log=run_log,
            results_manifest=results_manifest,
            output_dir=out,
            run_started_ns=0,
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("publish_completion overwrote an existing completion marker")
    assert done_marker.read_bytes() == marker_before
    assert results_manifest.read_bytes() == results_manifest_before


def test_xgbcat_completion_writer_refuses_concurrent_publish(tmp_path: Path) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    results_manifest = out / "result_artifacts.json"
    lock_file = out / ".xgbcat_completion_publish.lock"
    locker = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl, pathlib, sys, time\n"
                "path = pathlib.Path(sys.argv[1])\n"
                "path.parent.mkdir(parents=True, exist_ok=True)\n"
                "handle = path.open('w')\n"
                "fcntl.flock(handle, fcntl.LOCK_EX)\n"
                "print('locked', flush=True)\n"
                "time.sleep(30)\n"
            ),
            str(lock_file),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert locker.stdout is not None
        assert locker.stdout.readline().strip() == "locked"
        try:
            module.publish_completion(
                done_marker=done_marker,
                manifest=manifest,
                sweep_args=sweep_args,
                run_log=run_log,
                results_manifest=results_manifest,
                output_dir=out,
                run_started_ns=0,
            )
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("publish_completion ignored an active publish lock")
        assert not done_marker.exists()
        assert not results_manifest.exists()
    finally:
        locker.terminate()
        locker.communicate(timeout=5)


def test_xgbcat_completion_writer_self_verifies_and_quarantines_bad_publish(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    noncanonical_results_manifest = out / "custom_result_artifacts.json"

    try:
        module.publish_completion(
            done_marker=done_marker,
            manifest=manifest,
            sweep_args=sweep_args,
            run_log=run_log,
            results_manifest=noncanonical_results_manifest,
            output_dir=out,
            run_started_ns=0,
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("publish_completion accepted self-inconsistent evidence")
    assert not done_marker.exists()
    failed_marker = out / "sweep.done.failed.1"
    assert failed_marker.exists()
    assert "result_artifacts=/" in failed_marker.read_text(encoding="utf-8")
    assert not noncanonical_results_manifest.exists()


def test_xgbcat_completion_writer_self_verify_failure_restores_previous_sidecar(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    noncanonical_results_manifest = out / "custom_result_artifacts.json"
    previous_sidecar = b'{"previous": true}\n'
    noncanonical_results_manifest.write_bytes(previous_sidecar)

    try:
        module.publish_completion(
            done_marker=done_marker,
            manifest=manifest,
            sweep_args=sweep_args,
            run_log=run_log,
            results_manifest=noncanonical_results_manifest,
            output_dir=out,
            run_started_ns=0,
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("publish_completion accepted self-inconsistent evidence")
    assert not done_marker.exists()
    assert (out / "sweep.done.failed.1").exists()
    assert noncanonical_results_manifest.read_bytes() == previous_sidecar


def test_xgbcat_completion_writer_verify_fails_on_tampered_result(tmp_path: Path) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )
    tampered_payload = _valid_result_payload()
    tampered_payload["n_production_target_pass"] = 2
    _write_valid_result(result, tampered_payload)

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "result artifact 0 sha256 mismatch" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_marker_path_drift(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    alt_manifest = out / "alt_manifest.json"
    alt_manifest.write_bytes(manifest.read_bytes())
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )
    done_marker.write_text(
        done_marker.read_text(encoding="utf-8").replace(
            f"manifest={manifest.resolve()}",
            f"manifest={alt_manifest.resolve()}",
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "completion marker manifest path drift" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_manifest_sidecar_path_drift(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["run_log"] = str(tmp_path / "other" / "sweep_stdout.log")
    manifest.write_text(json.dumps(manifest_payload, sort_keys=True) + "\n", encoding="utf-8")
    done = _read_key_value_file(done_marker)
    done["manifest_sha256"] = _sha256(manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "manifest run_log path drift" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_bad_completed_at(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )
    done = _read_key_value_file(done_marker)
    done["completed_at"] = "2026-04-28 12:00:00"
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "completed_at must be a UTC ISO timestamp" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_schema_version_drift(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    done = _read_key_value_file(done_marker)
    done["schema_version"] = "2"
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = _verify_completion_marker(done_marker)

    assert manifest.exists()
    assert proc.returncode == 2
    assert "completion marker schema_version mismatch" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_result_path_drift(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    results_manifest = out / "result_artifacts.json"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=results_manifest,
        output_dir=out,
        run_started_ns=0,
    )
    other_result = other / result.name
    other_result.write_bytes(result.read_bytes())
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["path"] = str(other_result.resolve())
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "result artifact 0 path drift" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_duplicate_result_artifacts(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    results_manifest = out / "result_artifacts.json"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=results_manifest,
        output_dir=out,
        run_started_ns=0,
    )
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"].append(dict(results_payload["results"][0]))
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done["result_count"] = "2"
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "duplicates earlier result" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_extra_result_artifacts_root_key(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    results_manifest = out / "result_artifacts.json"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=results_manifest,
        output_dir=out,
        run_started_ns=0,
    )
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["result_count"] = 99
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "result artifacts JSON unexpected key: result_count" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_extra_result_artifacts_row_key(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    results_manifest = out / "result_artifacts.json"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=results_manifest,
        output_dir=out,
        run_started_ns=0,
    )
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["note"] = "preferred"
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "result artifacts row 0 unexpected key: note" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_result_pass_count_drift(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    results_manifest = out / "result_artifacts.json"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=results_manifest,
        output_dir=out,
        run_started_ns=0,
    )
    tampered_payload = _valid_result_payload()
    tampered_payload["n_production_target_pass"] = 2
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "production-target pass count mismatch" in proc.stderr


def test_xgbcat_completion_writer_rejects_tampered_production_pass_metrics(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    results_manifest = out / "result_artifacts.json"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=results_manifest,
        output_dir=out,
        run_started_ns=0,
    )
    tampered_payload = _valid_result_payload()
    robust_strategies = tampered_payload["friction_robust_strategies"]
    assert isinstance(robust_strategies, list)
    strategy = robust_strategies[0]
    assert isinstance(strategy, dict)
    strategy["worst_median_monthly_pct"] = 10.0
    strategy["worst_p10_monthly_pct"] = 10.0
    strategy["worst_goodness_score"] = -10.0
    strategy["worst_robust_goodness_score"] = -20.0
    strategy["worst_pain_adjusted_goodness_score"] = -20.0
    cells = tampered_payload["cells"]
    assert isinstance(cells, list)
    for cell in cells:
        assert isinstance(cell, dict)
        cell["median_monthly_pct"] = 10.0
        cell["monthly_return_pcts"] = [10.0] * UNIT_N_WINDOWS
        cell["p10_monthly_pct"] = 10.0
        cell["goodness_score"] = -10.0
        cell["robust_goodness_score"] = -20.0
        cell["pain_adjusted_goodness_score"] = -20.0
    tampered_payload["best_production_target_strategy"] = strategy
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "production_target_pass does not match metrics" in proc.stderr


def test_xgbcat_completion_writer_rejects_robust_summary_metric_drift(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    out = manifest.parent
    result = out / "sweep_20000101_000000.json"
    results_manifest = out / "result_artifacts.json"
    tampered_payload = _valid_result_payload()
    robust_strategies = tampered_payload["friction_robust_strategies"]
    assert isinstance(robust_strategies, list)
    strategy = robust_strategies[0]
    assert isinstance(strategy, dict)
    strategy["worst_median_monthly_pct"] = 35.0
    tampered_payload["best_production_target_strategy"] = strategy
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "worst_median_monthly_pct does not match raw cells" in proc.stderr


def test_xgbcat_completion_writer_rejects_expected_window_summary_drift(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    out = manifest.parent
    result = out / "sweep_20000101_000000.json"
    results_manifest = out / "result_artifacts.json"
    tampered_payload = _valid_result_payload()
    robust_strategies = tampered_payload["friction_robust_strategies"]
    assert isinstance(robust_strategies, list)
    strategy = robust_strategies[0]
    assert isinstance(strategy, dict)
    strategy["max_expected_n_windows"] = 99
    tampered_payload["best_production_target_strategy"] = strategy
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "max_expected_n_windows does not match raw cells" in proc.stderr


def test_xgbcat_completion_writer_rejects_monthly_return_summary_drift(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    out = manifest.parent
    result = out / "sweep_20000101_000000.json"
    results_manifest = out / "result_artifacts.json"
    tampered_payload = _valid_result_payload()
    cells = tampered_payload["cells"]
    assert isinstance(cells, list)
    for cell in cells:
        assert isinstance(cell, dict)
        cell["p10_monthly_pct"] = 30.0
        cell["goodness_score"] = 10.0
        cell["robust_goodness_score"] = 0.0
        cell["pain_adjusted_goodness_score"] = 0.0
    robust_strategies = tampered_payload["friction_robust_strategies"]
    assert isinstance(robust_strategies, list)
    strategy = robust_strategies[0]
    assert isinstance(strategy, dict)
    strategy["worst_p10_monthly_pct"] = 30.0
    strategy["worst_goodness_score"] = 10.0
    strategy["worst_robust_goodness_score"] = 0.0
    strategy["worst_pain_adjusted_goodness_score"] = 0.0
    tampered_payload["best_production_target_strategy"] = strategy
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "p10_monthly_pct does not match monthly_return_pcts" in proc.stderr


def test_xgbcat_completion_writer_rejects_window_pain_summary_drift(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    out = manifest.parent
    result = out / "sweep_20000101_000000.json"
    results_manifest = out / "result_artifacts.json"
    tampered_payload = _valid_result_payload()
    cells = tampered_payload["cells"]
    assert isinstance(cells, list)
    for cell in cells:
        assert isinstance(cell, dict)
        cell["window_time_under_water_pcts"] = [12.0] * UNIT_N_WINDOWS
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "time_under_water_pct does not match window_time_under_water_pcts" in proc.stderr


def test_xgbcat_completion_writer_rejects_active_day_summary_drift(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    out = manifest.parent
    result = out / "sweep_20000101_000000.json"
    results_manifest = out / "result_artifacts.json"
    tampered_payload = _valid_result_payload()
    cells = tampered_payload["cells"]
    assert isinstance(cells, list)
    for cell in cells:
        assert isinstance(cell, dict)
        cell["window_active_day_pcts"] = [40.0] * UNIT_N_WINDOWS
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "median_active_day_pct does not match window_active_day_pcts" in proc.stderr


def test_xgbcat_completion_writer_rejects_intraday_summary_drift(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    out = manifest.parent
    result = out / "sweep_20000101_000000.json"
    results_manifest = out / "result_artifacts.json"
    tampered_payload = _valid_result_payload()
    cells = tampered_payload["cells"]
    assert isinstance(cells, list)
    for cell in cells:
        assert isinstance(cell, dict)
        cell["window_worst_intraday_dd_pcts"] = [7.0] * UNIT_N_WINDOWS
        cell["window_avg_intraday_dd_pcts"] = [4.0] * UNIT_N_WINDOWS
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert (
        "worst_intraday_dd_pct does not match window_worst_intraday_dd_pcts"
        in proc.stderr
    )


def test_xgbcat_completion_writer_rejects_avg_intraday_summary_drift(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    out = manifest.parent
    result = out / "sweep_20000101_000000.json"
    results_manifest = out / "result_artifacts.json"
    tampered_payload = _valid_result_payload()
    cells = tampered_payload["cells"]
    assert isinstance(cells, list)
    for cell in cells:
        assert isinstance(cell, dict)
        cell["window_avg_intraday_dd_pcts"] = [4.0] * UNIT_N_WINDOWS
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "avg_intraday_dd_pct does not match window_avg_intraday_dd_pcts" in proc.stderr


def test_xgbcat_completion_writer_rejects_out_of_range_window_dates(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    out = manifest.parent
    result = out / "sweep_20000101_000000.json"
    results_manifest = out / "result_artifacts.json"
    tampered_payload = _valid_result_payload()
    cells = tampered_payload["cells"]
    assert isinstance(cells, list)
    for cell in cells:
        assert isinstance(cell, dict)
        cell["window_end_dates"] = [*UNIT_WINDOW_END_DATES[:-1], "2026-05-01"]
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "window date range 9 is outside OOS span" in proc.stderr


def test_xgbcat_completion_writer_rejects_raw_cell_score_drift(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    out = manifest.parent
    result = out / "sweep_20000101_000000.json"
    results_manifest = out / "result_artifacts.json"
    tampered_payload = _valid_result_payload()
    cells = tampered_payload["cells"]
    assert isinstance(cells, list)
    for cell in cells:
        assert isinstance(cell, dict)
        cell["pain_adjusted_goodness_score"] = 5.0
    robust_strategies = tampered_payload["friction_robust_strategies"]
    assert isinstance(robust_strategies, list)
    strategy = robust_strategies[0]
    assert isinstance(strategy, dict)
    strategy["worst_pain_adjusted_goodness_score"] = 5.0
    tampered_payload["best_production_target_strategy"] = strategy
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "pain_adjusted_goodness_score does not match raw metrics" in proc.stderr


def test_xgbcat_completion_writer_rejects_raw_cell_robust_score_drift(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    out = manifest.parent
    result = out / "sweep_20000101_000000.json"
    results_manifest = out / "result_artifacts.json"
    tampered_payload = _valid_result_payload()
    cells = tampered_payload["cells"]
    assert isinstance(cells, list)
    for cell in cells:
        assert isinstance(cell, dict)
        cell["robust_goodness_score"] = 5.0
        cell["pain_adjusted_goodness_score"] = 5.0
    robust_strategies = tampered_payload["friction_robust_strategies"]
    assert isinstance(robust_strategies, list)
    strategy = robust_strategies[0]
    assert isinstance(strategy, dict)
    strategy["worst_robust_goodness_score"] = 5.0
    strategy["worst_pain_adjusted_goodness_score"] = 5.0
    tampered_payload["best_production_target_strategy"] = strategy
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "robust_goodness_score does not match raw metrics" in proc.stderr


def test_xgbcat_completion_writer_rejects_negative_cell_without_loss_magnitude(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    out = manifest.parent
    result = out / "sweep_20000101_000000.json"
    results_manifest = out / "result_artifacts.json"
    tampered_payload = _valid_result_payload()
    cells = tampered_payload["cells"]
    assert isinstance(cells, list)
    extra_cell = dict(cells[0])
    extra_cell["min_score"] = 0.62
    extra_cell["expected_n_windows"] = 2
    extra_cell["n_windows"] = 2
    extra_cell["n_neg"] = 1
    extra_cell["mean_abs_neg_monthly_pct"] = 0.0
    extra_cell["median_monthly_pct"] = 10.0
    extra_cell["monthly_return_pcts"] = [-10.0, 30.0]
    extra_cell["window_drawdown_pcts"] = [20.0, 20.0]
    extra_cell["window_sortino_values"] = [2.0, 2.0]
    extra_cell["window_time_under_water_pcts"] = [0.0, 0.0]
    extra_cell["window_ulcer_indexes"] = [0.0, 0.0]
    extra_cell["window_active_day_pcts"] = [100.0, 100.0]
    extra_cell["window_worst_intraday_dd_pcts"] = [3.0, 3.0]
    extra_cell["window_avg_intraday_dd_pcts"] = [1.0, 1.0]
    extra_cell["window_start_dates"] = UNIT_WINDOW_START_DATES[:2]
    extra_cell["window_end_dates"] = UNIT_WINDOW_END_DATES[:2]
    extra_cell["p10_monthly_pct"] = -6.0
    extra_cell["goodness_score"] = -76.0
    extra_cell["robust_goodness_score"] = -61.0
    extra_cell["pain_adjusted_goodness_score"] = -61.0
    cells.append(extra_cell)
    tampered_payload["n_cells"] = len(cells)
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "mean_abs_neg_monthly_pct does not match monthly_return_pcts" in proc.stderr


def test_xgbcat_completion_writer_rejects_raw_cell_missing_strategy_field(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    payload = _valid_result_payload()
    cells = payload["cells"]
    assert isinstance(cells, list)
    cell = cells[0]
    assert isinstance(cell, dict)
    cell.pop("top_n")
    _write_valid_result(result, payload)

    try:
        module.publish_completion(
            done_marker=out / "sweep.done",
            manifest=manifest,
            sweep_args=sweep_args,
            run_log=run_log,
            results_manifest=out / "result_artifacts.json",
            output_dir=out,
            run_started_ns=0,
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("publish_completion accepted raw cell missing strategy field")
    assert not (out / "sweep.done").exists()
    assert not (out / "result_artifacts.json").exists()


def test_xgbcat_completion_writer_rejects_robust_strategy_missing_field(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    out = manifest.parent
    result = out / "sweep_20000101_000000.json"
    results_manifest = out / "result_artifacts.json"
    tampered_payload = _valid_result_payload()
    robust_strategies = tampered_payload["friction_robust_strategies"]
    assert isinstance(robust_strategies, list)
    strategy = robust_strategies[0]
    assert isinstance(strategy, dict)
    strategy.pop("top_n")
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "missing strategy field: top_n" in proc.stderr


def test_xgbcat_completion_writer_rejects_robust_pain_metric_drift(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    out = manifest.parent
    result = out / "sweep_20000101_000000.json"
    results_manifest = out / "result_artifacts.json"
    tampered_payload = _valid_result_payload()
    robust_strategies = tampered_payload["friction_robust_strategies"]
    assert isinstance(robust_strategies, list)
    strategy = robust_strategies[0]
    assert isinstance(strategy, dict)
    strategy["max_time_under_water_pct"] = 1.0
    tampered_payload["best_production_target_strategy"] = strategy
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "max_time_under_water_pct does not match raw cells" in proc.stderr


def test_xgbcat_completion_writer_rejects_incomplete_robust_stress_coverage(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    out = manifest.parent
    result = out / "sweep_20000101_000000.json"
    results_manifest = out / "result_artifacts.json"
    tampered_payload = _valid_result_payload()
    robust_strategies = tampered_payload["friction_robust_strategies"]
    assert isinstance(robust_strategies, list)
    strategy = robust_strategies[0]
    assert isinstance(strategy, dict)
    strategy["fill_buffer_bps_values"] = [0.0, 5.0, 10.0]
    tampered_payload["best_production_target_strategy"] = strategy
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "fill_buffer_bps_values do not cover manifest stress grid" in proc.stderr


def test_xgbcat_completion_writer_rejects_unrequested_skip_stress_axis(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    out = manifest.parent
    result = out / "sweep_20000101_000000.json"
    results_manifest = out / "result_artifacts.json"
    tampered_payload = _valid_result_payload()
    robust_strategies = tampered_payload["friction_robust_strategies"]
    assert isinstance(robust_strategies, list)
    strategy = robust_strategies[0]
    assert isinstance(strategy, dict)
    strategy["skip_prob_values"] = [0.0, 0.05]
    strategy["skip_seed_values"] = [0, 1, 2]
    tampered_payload["best_production_target_strategy"] = strategy
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "skip_prob_values do not match no-skip runner contract" in proc.stderr


def test_xgbcat_completion_writer_rejects_robust_stress_cell_count_drift(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    out = manifest.parent
    result = out / "sweep_20000101_000000.json"
    results_manifest = out / "result_artifacts.json"
    tampered_payload = _valid_result_payload()
    robust_strategies = tampered_payload["friction_robust_strategies"]
    assert isinstance(robust_strategies, list)
    strategy = robust_strategies[0]
    assert isinstance(strategy, dict)
    strategy["n_friction_cells"] = 24
    tampered_payload["best_production_target_strategy"] = strategy
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "friction stress cell count does not match manifest grid" in proc.stderr


def test_xgbcat_completion_writer_rejects_non_best_production_strategy(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    results_manifest = out / "result_artifacts.json"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=results_manifest,
        output_dir=out,
        run_started_ns=0,
    )
    tampered_payload = _valid_result_payload()
    robust_strategies = tampered_payload["friction_robust_strategies"]
    assert isinstance(robust_strategies, list)
    first_strategy = robust_strategies[0]
    assert isinstance(first_strategy, dict)
    first_strategy["worst_pain_adjusted_goodness_score"] = -9.0
    better_strategy = dict(first_strategy)
    better_strategy["min_score"] = 0.62
    better_strategy["worst_pain_adjusted_goodness_score"] = 2.0
    better_strategy["worst_median_monthly_pct"] = 40.0
    better_strategy["worst_p10_monthly_pct"] = 32.0
    better_strategy["worst_goodness_score"] = 12.0
    better_strategy["worst_robust_goodness_score"] = 2.0
    robust_strategies.append(better_strategy)
    cells = tampered_payload["cells"]
    assert isinstance(cells, list)
    better_cells = []
    for cell in cells:
        assert isinstance(cell, dict)
        better_cells.append(
            {
                **cell,
                "min_score": 0.62,
                "goodness_score": 12.0,
                "median_monthly_pct": 40.0,
                "monthly_return_pcts": [23.0, 33.0] + [40.0] * 8,
                "pain_adjusted_goodness_score": 2.0,
                "p10_monthly_pct": 32.0,
                "robust_goodness_score": 2.0,
            },
        )
    cells.extend(better_cells)
    tampered_payload["n_cells"] = len(cells)
    tampered_payload["n_friction_robust_strategies"] = 2
    tampered_payload["n_production_target_pass"] = 2
    tampered_payload["best_production_target_strategy"] = first_strategy
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "best production-target strategy is not highest producer-ranked pass" in proc.stderr


def test_xgbcat_completion_writer_rejects_duplicate_robust_strategy(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    out = manifest.parent
    result = out / "sweep_20000101_000000.json"
    results_manifest = out / "result_artifacts.json"
    tampered_payload = _valid_result_payload()
    robust_strategies = tampered_payload["friction_robust_strategies"]
    assert isinstance(robust_strategies, list)
    first_strategy = robust_strategies[0]
    assert isinstance(first_strategy, dict)
    robust_strategies.append(dict(first_strategy))
    tampered_payload["n_friction_robust_strategies"] = 2
    tampered_payload["n_production_target_pass"] = 2
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "duplicates an earlier robust strategy" in proc.stderr


def test_xgbcat_completion_writer_rejects_lower_median_best_tiebreak(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    results_manifest = out / "result_artifacts.json"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=results_manifest,
        output_dir=out,
        run_started_ns=0,
    )
    tampered_payload = _valid_result_payload()
    robust_strategies = tampered_payload["friction_robust_strategies"]
    assert isinstance(robust_strategies, list)
    lower_median_strategy = robust_strategies[0]
    assert isinstance(lower_median_strategy, dict)
    lower_median_strategy["worst_median_monthly_pct"] = 28.0
    cells = tampered_payload["cells"]
    assert isinstance(cells, list)
    for cell in cells:
        assert isinstance(cell, dict)
        cell["median_monthly_pct"] = 28.0
        cell["monthly_return_pcts"] = [12.0, 22.0] + [28.0] * 8
    higher_median_strategy = dict(lower_median_strategy)
    higher_median_strategy["min_score"] = 0.62
    higher_median_strategy["worst_median_monthly_pct"] = 30.0
    robust_strategies.append(higher_median_strategy)
    higher_median_cells = []
    for cell in cells:
        assert isinstance(cell, dict)
        higher_median_cells.append(
            {
                **cell,
                "median_monthly_pct": 30.0,
                "min_score": 0.62,
                "monthly_return_pcts": UNIT_MONTHLY_RETURN_PCTS,
            },
        )
    cells.extend(higher_median_cells)
    tampered_payload["n_cells"] = len(cells)
    tampered_payload["n_friction_robust_strategies"] = 2
    tampered_payload["n_production_target_pass"] = 2
    tampered_payload["best_production_target_strategy"] = lower_median_strategy
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "best production-target strategy is not highest producer-ranked pass" in proc.stderr


def test_xgbcat_completion_writer_accepts_tied_best_production_strategy(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    results_manifest = out / "result_artifacts.json"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=results_manifest,
        output_dir=out,
        run_started_ns=0,
    )
    payload = _valid_result_payload()
    robust_strategies = payload["friction_robust_strategies"]
    assert isinstance(robust_strategies, list)
    first_strategy = robust_strategies[0]
    assert isinstance(first_strategy, dict)
    tied_strategy = dict(first_strategy)
    tied_strategy["min_score"] = 0.62
    robust_strategies.append(tied_strategy)
    cells = payload["cells"]
    assert isinstance(cells, list)
    tied_cells = []
    for cell in cells:
        assert isinstance(cell, dict)
        tied_cells.append({**cell, "min_score": 0.62})
    cells.extend(tied_cells)
    payload["n_cells"] = len(cells)
    payload["n_friction_robust_strategies"] = 2
    payload["n_production_target_pass"] = 2
    payload["best_production_target_strategy"] = tied_strategy
    _write_valid_result(result, payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    module.verify_completion(done_marker)


def test_xgbcat_completion_writer_rejects_result_model_path_drift(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    results_manifest = out / "result_artifacts.json"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=results_manifest,
        output_dir=out,
        run_started_ns=0,
    )
    tampered_payload = _valid_result_payload()
    tampered_payload["model_paths"] = ["models/other.pkl"]
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "model_paths do not match manifest" in proc.stderr


def test_xgbcat_completion_writer_rejects_result_model_hash_drift(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    results_manifest = out / "result_artifacts.json"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=results_manifest,
        output_dir=out,
        run_started_ns=0,
    )
    tampered_payload = _valid_result_payload()
    tampered_payload["model_sha256"] = ["1" * 64]
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "model_sha256 does not match manifest" in proc.stderr


def test_xgbcat_completion_writer_rejects_result_fail_fast_drift(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    results_manifest = out / "result_artifacts.json"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=results_manifest,
        output_dir=out,
        run_started_ns=0,
    )
    tampered_payload = _valid_result_payload()
    fail_fast = tampered_payload["fail_fast"]
    assert isinstance(fail_fast, dict)
    fail_fast["max_dd_pct"] = 40.0
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "result JSON fail_fast max_dd_pct does not match manifest" in proc.stderr


def test_xgbcat_completion_writer_rejects_result_fee_regime_drift(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    results_manifest = out / "result_artifacts.json"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=results_manifest,
        output_dir=out,
        run_started_ns=0,
    )
    tampered_payload = _valid_result_payload()
    fee_regimes = tampered_payload["fee_regimes"]
    assert isinstance(fee_regimes, dict)
    stress36x = fee_regimes["stress36x"]
    assert isinstance(stress36x, dict)
    stress36x["commission_bps"] = 0.0
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "result JSON fee_regime stress36x commission_bps mismatch" in proc.stderr


def test_xgbcat_completion_writer_rejects_result_goodness_weight_drift(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    results_manifest = out / "result_artifacts.json"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=results_manifest,
        output_dir=out,
        run_started_ns=0,
    )
    tampered_payload = _valid_result_payload()
    goodness_weights = tampered_payload["goodness_weights"]
    assert isinstance(goodness_weights, dict)
    goodness_weights["neg_coef"] = 0.0
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "result JSON goodness_weights neg_coef mismatch" in proc.stderr


def test_xgbcat_completion_writer_rejects_unexpected_result_field(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    results_manifest = out / "result_artifacts.json"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=results_manifest,
        output_dir=out,
        run_started_ns=0,
    )
    tampered_payload = _valid_result_payload()
    tampered_payload["preferred_for_promotion"] = True
    _write_valid_result(result, tampered_payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "result JSON unexpected field: preferred_for_promotion" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_model_file_hash_drift(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )
    (out / "models" / "model0.pkl").write_bytes(b"changed model bytes\n")

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "model 0 sha256 mismatch" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_symbols_file_hash_drift(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )
    (out / "symbols.txt").write_text("TSLA\n", encoding="utf-8")

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "symbols file sha256 mismatch" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_missing_symbols_file(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    Path(manifest_payload["symbols_file"]).unlink()

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "symbols file missing" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_unsafe_manifest_leverage(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["config"]["leverage_grid"] = "1.5,3.0"
    manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args = sweep_args.read_text(encoding="utf-8").splitlines()
    args[args.index("--leverage-grid") + 1] = "1.5,3.0"
    sweep_args.write_text("\n".join(args) + "\n", encoding="utf-8")
    done = _read_key_value_file(done_marker)
    done["manifest_sha256"] = _sha256(manifest)
    done["sweep_args_sha256"] = _sha256(sweep_args)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "manifest config leverage_grid exceeds production limit" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_missing_manifest_stress_regime(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["config"]["fee_regimes"] = "deploy,prod10bps"
    manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args = sweep_args.read_text(encoding="utf-8").splitlines()
    args[args.index("--fee-regimes") + 1] = "deploy,prod10bps"
    sweep_args.write_text("\n".join(args) + "\n", encoding="utf-8")
    done = _read_key_value_file(done_marker)
    done["manifest_sha256"] = _sha256(manifest)
    done["sweep_args_sha256"] = _sha256(sweep_args)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "manifest config fee_regimes missing required regime: stress36x" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_false_manifest_fixed_flag(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["fixed_flags"]["require_production_target"] = False
    manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["manifest_sha256"] = _sha256(manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "manifest fixed_flags require_production_target must be true" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_unexpected_manifest_field(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["preferred_for_promotion"] = True
    manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["manifest_sha256"] = _sha256(manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "manifest unexpected field: preferred_for_promotion" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_model_count_drift(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["cat_model_count"] = 1
    manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["manifest_sha256"] = _sha256(manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "manifest cat_model_count does not match models" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_family_directory_drift(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["models"][0]["family"] = "cat"
    manifest_payload["xgb_model_count"] = 0
    manifest_payload["cat_model_count"] = 1
    manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rewrite_marker_manifest_hash(done_marker, manifest)

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "manifest model 0 path is outside declared cat directory" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_model_paths_file_drift(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    model_paths_file = manifest.parent / "model_paths.txt"
    model_paths_file.write_text(
        f"{manifest.parent / 'models' / 'other.pkl'}\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["model_paths_sha256"] = _sha256(model_paths_file)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "model paths file does not match manifest models" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_mutated_model_artifact(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    model_path = Path(manifest_payload["models"][0]["path"])
    model_path.write_bytes(b"tampered model bytes\n")

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "manifest model 0 sha256 mismatch" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_missing_manifest_config_key(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    del manifest_payload["config"]["min_score_grid"]
    manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rewrite_marker_manifest_hash(done_marker, manifest)

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "manifest config missing key: min_score_grid" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_unexpected_model_field(
    tmp_path: Path,
) -> None:
    manifest, done_marker = _publish_valid_completion(tmp_path / "out")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["models"][0]["rank_hint"] = 1
    manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rewrite_marker_manifest_hash(done_marker, manifest)

    proc = _verify_completion_marker(done_marker)

    assert proc.returncode == 2
    assert "manifest model 0 unexpected field: rank_hint" in proc.stderr


def test_xgbcat_completion_writer_verify_rejects_missing_required_grid_cell(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    results_manifest = out / "result_artifacts.json"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=results_manifest,
        output_dir=out,
        run_started_ns=0,
    )
    payload = _valid_result_payload()
    cells = payload["cells"]
    assert isinstance(cells, list)
    payload["cells"] = [
        cell
        for cell in cells
        if not (
            isinstance(cell, dict)
            and cell.get("fee_regime") == "deploy"
            and cell.get("fill_buffer_bps") == 5.0
        )
    ]
    payload["n_cells"] = len(payload["cells"])
    _write_valid_result(result, payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "missing required production-realism cell" in proc.stderr
    assert "fee_regime=deploy fill_buffer_bps=5" in proc.stderr


def test_xgbcat_completion_writer_accepts_multiple_strategy_cells_per_stress_pair(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    results_manifest = out / "result_artifacts.json"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=results_manifest,
        output_dir=out,
        run_started_ns=0,
    )
    payload = _valid_result_payload()
    cells = payload["cells"]
    assert isinstance(cells, list)
    payload["cells"] = [*cells, *[dict(cell, min_score=0.70) for cell in cells]]
    payload["n_cells"] = len(payload["cells"])
    _write_valid_result(result, payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    module.verify_completion(done_marker)


def test_xgbcat_completion_writer_verify_rejects_unexpected_result_grid_cell(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)
    done_marker = out / "sweep.done"
    results_manifest = out / "result_artifacts.json"
    module.publish_completion(
        done_marker=done_marker,
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=results_manifest,
        output_dir=out,
        run_started_ns=0,
    )
    payload = _valid_result_payload()
    cells = payload["cells"]
    assert isinstance(cells, list)
    cells.append(
        {
            "fee_regime": "deploy",
            "fill_buffer_bps": 25.0,
            "goodness_score": 0.5,
        },
    )
    payload["n_cells"] = len(cells)
    _write_valid_result(result, payload)
    results_payload = json.loads(results_manifest.read_text(encoding="utf-8"))
    results_payload["results"][0]["sha256"] = _sha256(result)
    results_payload["results"][0]["size_bytes"] = result.stat().st_size
    results_manifest.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    done = _read_key_value_file(done_marker)
    done["manifest_sha256"] = _sha256(manifest)
    done["result_artifacts_sha256"] = _sha256(results_manifest)
    done_marker.write_text(
        "".join(f"{key}={value}\n" for key, value in done.items()),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(done_marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "cell has unexpected fill_buffer_bps: 25" in proc.stderr


def test_xgbcat_completion_writer_rejects_boolean_numeric_result_fields(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    payload = _valid_result_payload()
    cells = payload["cells"]
    assert isinstance(cells, list)
    cell = cells[0]
    assert isinstance(cell, dict)
    cell["fill_buffer_bps"] = True
    payload["n_cells"] = True
    _write_valid_result(result, payload)

    try:
        module.publish_completion(
            done_marker=out / "sweep.done",
            manifest=manifest,
            sweep_args=sweep_args,
            run_log=run_log,
            results_manifest=out / "result_artifacts.json",
            output_dir=out,
            run_started_ns=0,
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("publish_completion accepted boolean numeric fields")
    assert not (out / "sweep.done").exists()
    assert not (out / "result_artifacts.json").exists()


def test_xgbcat_completion_writer_rejects_nonboolean_raw_fail_fast_flag(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    payload = _valid_result_payload()
    cells = payload["cells"]
    assert isinstance(cells, list)
    cell = cells[0]
    assert isinstance(cell, dict)
    cell["fail_fast_triggered"] = 0
    _write_valid_result(result, payload)

    try:
        module.publish_completion(
            done_marker=out / "sweep.done",
            manifest=manifest,
            sweep_args=sweep_args,
            run_log=run_log,
            results_manifest=out / "result_artifacts.json",
            output_dir=out,
            run_started_ns=0,
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("publish_completion accepted non-boolean raw fail-fast flag")
    assert not (out / "sweep.done").exists()
    assert not (out / "result_artifacts.json").exists()


def test_xgbcat_completion_writer_accepts_valid_optional_result_metadata(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    spy_csv = out / "trainingdata" / "SPY.csv"
    spy_csv.parent.mkdir()
    spy_csv.write_text("timestamp,close\n2026-01-02,500\n", encoding="utf-8")
    fm_latents = out / "analysis" / "fm_latents.parquet"
    fm_latents.parent.mkdir()
    fm_latents.write_bytes(b"fm-latents")
    ensemble_manifest_path = out / "models" / "alltrain_ensemble.json"
    ensemble_manifest_payload = {
        "config": {"max_depth": 5, "n_estimators": 400},
        "seeds": [0, 7],
        "train_end": "2026-04-20",
        "train_start": "2020-01-01",
        "trained_at": "2026-04-21T00:00:00Z",
    }
    ensemble_manifest_path.write_text(
        json.dumps(ensemble_manifest_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = out / "sweep_20000101_000000.json"
    payload = _valid_result_payload()
    payload["blend_mode"] = "mean"
    payload["data_root"] = "trainingdata"
    payload["spy_csv"] = "trainingdata/SPY.csv"
    payload["spy_csv_sha256"] = _sha256(spy_csv)
    payload["fm_latents_path"] = "analysis/fm_latents.parquet"
    payload["fm_latents_sha256"] = _sha256(fm_latents)
    payload["fm_n_latents"] = 32
    payload["ensemble_feature_mode"] = {
        "needs_dispersion": False,
        "needs_ranks": False,
    }
    payload["ensemble_manifest"] = {
        "path": str(ensemble_manifest_path),
        "sha256": _sha256(ensemble_manifest_path),
        **ensemble_manifest_payload,
    }
    _write_valid_result(result, payload)

    module.publish_completion(
        done_marker=out / "sweep.done",
        manifest=manifest,
        sweep_args=sweep_args,
        run_log=run_log,
        results_manifest=out / "result_artifacts.json",
        output_dir=out,
        run_started_ns=0,
    )

    assert (out / "sweep.done").exists()


def test_xgbcat_completion_writer_rejects_tampered_optional_result_metadata(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    payload = _valid_result_payload()
    payload["blend_mode"] = "median"
    _write_valid_result(result, payload)

    try:
        module.publish_completion(
            done_marker=out / "sweep.done",
            manifest=manifest,
            sweep_args=sweep_args,
            run_log=run_log,
            results_manifest=out / "result_artifacts.json",
            output_dir=out,
            run_started_ns=0,
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("publish_completion accepted tampered optional metadata")
    assert not (out / "sweep.done").exists()
    assert not (out / "result_artifacts.json").exists()


def test_xgbcat_completion_writer_rejects_tampered_fm_latent_hash(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    fm_latents = out / "analysis" / "fm_latents.parquet"
    fm_latents.parent.mkdir()
    fm_latents.write_bytes(b"fm-latents")
    result = out / "sweep_20000101_000000.json"
    payload = _valid_result_payload()
    payload["fm_latents_path"] = "analysis/fm_latents.parquet"
    payload["fm_latents_sha256"] = "0" * 64
    payload["fm_n_latents"] = 32
    _write_valid_result(result, payload)

    try:
        module.publish_completion(
            done_marker=out / "sweep.done",
            manifest=manifest,
            sweep_args=sweep_args,
            run_log=run_log,
            results_manifest=out / "result_artifacts.json",
            output_dir=out,
            run_started_ns=0,
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("publish_completion accepted tampered FM latent hash")
    assert not (out / "sweep.done").exists()
    assert not (out / "result_artifacts.json").exists()


def test_xgbcat_completion_writer_rejects_fm_latent_hash_manifest_mismatch(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    fm_latents = out / "analysis" / "fm_latents.parquet"
    stale_fm_latents = out / "analysis" / "stale_fm_latents.parquet"
    fm_latents.parent.mkdir()
    fm_latents.write_bytes(b"fm-latents")
    stale_fm_latents.write_bytes(b"stale-fm-latents")
    ensemble_manifest_path = out / "models" / "alltrain_ensemble.json"
    ensemble_manifest_payload = {
        "config": {
            "fm_latents_path": str(stale_fm_latents),
            "fm_latents_sha256": _sha256(stale_fm_latents),
            "fm_n_latents": 32,
        },
        "seeds": [0, 7],
    }
    ensemble_manifest_path.write_text(
        json.dumps(ensemble_manifest_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = out / "sweep_20000101_000000.json"
    payload = _valid_result_payload()
    payload["fm_latents_path"] = "analysis/fm_latents.parquet"
    payload["fm_latents_sha256"] = _sha256(fm_latents)
    payload["fm_n_latents"] = 32
    payload["ensemble_manifest"] = {
        "path": str(ensemble_manifest_path),
        "sha256": _sha256(ensemble_manifest_path),
        **ensemble_manifest_payload,
    }
    _write_valid_result(result, payload)

    try:
        module.publish_completion(
            done_marker=out / "sweep.done",
            manifest=manifest,
            sweep_args=sweep_args,
            run_log=run_log,
            results_manifest=out / "result_artifacts.json",
            output_dir=out,
            run_started_ns=0,
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("publish_completion accepted mismatched FM latent provenance")
    assert not (out / "sweep.done").exists()
    assert not (out / "result_artifacts.json").exists()


def test_xgbcat_completion_writer_rejects_unbacked_ensemble_feature_mode(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    payload = _valid_result_payload()
    payload["ensemble_feature_mode"] = {
        "needs_dispersion": False,
        "needs_ranks": True,
    }
    _write_valid_result(result, payload)

    try:
        module.publish_completion(
            done_marker=out / "sweep.done",
            manifest=manifest,
            sweep_args=sweep_args,
            run_log=run_log,
            results_manifest=out / "result_artifacts.json",
            output_dir=out,
            run_started_ns=0,
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("publish_completion accepted unbacked ensemble feature mode")
    assert not (out / "sweep.done").exists()
    assert not (out / "result_artifacts.json").exists()


def test_xgbcat_completion_writer_rejects_tampered_spy_csv_hash(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    spy_csv = out / "trainingdata" / "SPY.csv"
    spy_csv.parent.mkdir()
    spy_csv.write_text("timestamp,close\n2026-01-02,500\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    payload = _valid_result_payload()
    payload["spy_csv"] = "trainingdata/SPY.csv"
    payload["spy_csv_sha256"] = "0" * 64
    _write_valid_result(result, payload)

    try:
        module.publish_completion(
            done_marker=out / "sweep.done",
            manifest=manifest,
            sweep_args=sweep_args,
            run_log=run_log,
            results_manifest=out / "result_artifacts.json",
            output_dir=out,
            run_started_ns=0,
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("publish_completion accepted tampered SPY CSV hash")
    assert not (out / "sweep.done").exists()
    assert not (out / "result_artifacts.json").exists()


def test_xgbcat_completion_writer_rejects_tampered_ensemble_manifest_copy(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    ensemble_manifest_path = out / "models" / "alltrain_ensemble.json"
    ensemble_manifest_path.write_text(
        json.dumps(
            {
                "config": {"max_depth": 5},
                "seeds": [0, 7],
                "train_end": "2026-04-20",
                "train_start": "2020-01-01",
                "trained_at": "2026-04-21T00:00:00Z",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result = out / "sweep_20000101_000000.json"
    payload = _valid_result_payload()
    payload["ensemble_manifest"] = {
        "path": str(ensemble_manifest_path),
        "sha256": _sha256(ensemble_manifest_path),
        "train_end": "2026-04-26",
    }
    _write_valid_result(result, payload)

    try:
        module.publish_completion(
            done_marker=out / "sweep.done",
            manifest=manifest,
            sweep_args=sweep_args,
            run_log=run_log,
            results_manifest=out / "result_artifacts.json",
            output_dir=out,
            run_started_ns=0,
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("publish_completion accepted tampered ensemble manifest copy")
    assert not (out / "sweep.done").exists()
    assert not (out / "result_artifacts.json").exists()


def test_xgbcat_completion_writer_rejects_boolean_production_target_fields(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    payload = _valid_result_payload()
    production_target = payload["production_target"]
    assert isinstance(production_target, dict)
    production_target["min_windows"] = True
    _write_valid_result(result, payload)

    try:
        module.publish_completion(
            done_marker=out / "sweep.done",
            manifest=manifest,
            sweep_args=sweep_args,
            run_log=run_log,
            results_manifest=out / "result_artifacts.json",
            output_dir=out,
            run_started_ns=0,
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("publish_completion accepted boolean production target fields")
    assert not (out / "sweep.done").exists()
    assert not (out / "result_artifacts.json").exists()


def test_xgbcat_completion_writer_rejects_missing_model_paths_before_publish(
    tmp_path: Path,
) -> None:
    module = _load_completion_module()
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    (out / "model_paths.txt").unlink()
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    _write_valid_result(result)

    try:
        module.publish_completion(
            done_marker=out / "sweep.done",
            manifest=manifest,
            sweep_args=sweep_args,
            run_log=run_log,
            results_manifest=out / "result_artifacts.json",
            output_dir=out,
            run_started_ns=0,
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("publish_completion accepted a missing model paths sidecar")
    assert not (out / "sweep.done").exists()
    assert not (out / "result_artifacts.json").exists()


def test_xgbcat_completion_writer_cli_fails_on_bad_result(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "run_manifest.json"
    _write_manifest(manifest)
    sweep_args = out / "sweep_args.txt"
    _write_sweep_args(sweep_args)
    run_log = out / "sweep_stdout.log"
    run_log.write_text("log\n", encoding="utf-8")
    result = out / "sweep_20000101_000000.json"
    result.write_text('{"complete": false}\n', encoding="utf-8")

    proc = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--done-marker",
            str(out / "sweep.done"),
            "--manifest",
            str(manifest),
            "--sweep-args",
            str(sweep_args),
            "--run-log",
            str(run_log),
            "--results-manifest",
            str(out / "result_artifacts.json"),
            "--output-dir",
            str(out),
            "--run-started-ns",
            "0",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "result JSON is not complete" in proc.stderr
    assert not (out / "sweep.done").exists()
    assert not (out / "result_artifacts.json").exists()


def test_xgbcat_risk_parity_preflight_passes_in_dry_run(tmp_path: Path) -> None:
    fake_repo = _make_fake_repo(tmp_path)

    proc = _run_script(fake_repo)

    assert proc.returncode == 0
    assert "dry run: preflight passed" in proc.stdout
    assert "xgb_model_count=1" in proc.stdout
    assert "cat_model_count=1" in proc.stdout
    assert "manifest=" in proc.stdout
    assert not (fake_repo / "logs" / "track1" / "xgbcat_risk_parity_wide_120d.log").exists()
    manifest_path = (
        fake_repo
        / "analysis"
        / "xgbnew_daily"
        / "xgbcat_risk_parity_wide_120d"
        / "run_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["script"] == "scripts/xgbcat_risk_parity_wide_120d.sh"
    assert manifest["venv"] == ".venv313"
    assert manifest["xgb_model_count"] == 1
    assert manifest["cat_model_count"] == 1
    assert manifest["symbols_sha256"] == _sha256(
        fake_repo / "symbol_lists" / "stocks_wide_fresh0401_photonics_2500_v1.txt",
    )
    assert [model["family"] for model in manifest["models"]] == ["xgb", "cat"]
    assert all(len(model["sha256"]) == 64 for model in manifest["models"])
    assert manifest["config"]["oos_start"] == "2025-12-18"
    assert manifest["config"]["fill_buffer_bps_grid"] == "0,5,10,20"
    assert manifest["config"]["fee_regimes"] == "deploy,prod10bps,stress36x"
    assert manifest["fixed_flags"]["require_production_target"] is True
    model_list = Path(manifest["model_paths_file"])
    assert model_list.read_text(encoding="utf-8").count("alltrain_seed0.pkl") == 2
    assert Path(manifest["run_log"]).name == "sweep_stdout.log"
    assert not Path(manifest["run_log"]).exists()
    assert Path(manifest["done_marker"]).name == "sweep.done"
    assert not Path(manifest["done_marker"]).exists()
    sweep_args = Path(manifest["sweep_args_file"]).read_text(encoding="utf-8").splitlines()
    assert sweep_args[sweep_args.index("--fill-buffer-bps-grid") + 1] == "0,5,10,20"
    assert sweep_args[sweep_args.index("--fee-regimes") + 1] == "deploy,prod10bps,stress36x"
    assert "--require-production-target" in sweep_args
    assert "--fast-features" in sweep_args


def test_xgbcat_risk_parity_refuses_concurrent_same_output_run(tmp_path: Path) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    out = fake_repo / "analysis" / "xgbnew_daily" / "xgbcat_risk_parity_wide_120d"
    lock_file = out / ".xgbcat_risk_parity.lock"
    locker = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl, pathlib, sys, time\n"
                "path = pathlib.Path(sys.argv[1])\n"
                "path.parent.mkdir(parents=True, exist_ok=True)\n"
                "handle = path.open('w')\n"
                "fcntl.flock(handle, fcntl.LOCK_EX)\n"
                "print('locked', flush=True)\n"
                "time.sleep(30)\n"
            ),
            str(lock_file),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert locker.stdout is not None
        assert locker.stdout.readline().strip() == "locked"

        proc = _run_script(fake_repo)

        assert proc.returncode == 2
        assert "another xgbcat risk-parity sweep is already running" in proc.stderr
        assert not (out / "run_manifest.json").exists()
        assert not (out / "sweep_args.txt").exists()
    finally:
        locker.terminate()
        locker.communicate(timeout=5)


def test_xgbcat_risk_parity_manifest_records_config_overrides(tmp_path: Path) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    out = fake_repo / "custom-out"

    proc = _run_script(
        fake_repo,
        OOS_START="2026-01-05",
        LEVERAGE_GRID="1.5",
        OUT_SHOULD_NOT_BE_USED="ignored",
    )

    assert proc.returncode == 0
    default_manifest = (
        fake_repo
        / "analysis"
        / "xgbnew_daily"
        / "xgbcat_risk_parity_wide_120d"
        / "run_manifest.json"
    )
    assert default_manifest.exists()
    manifest = json.loads(default_manifest.read_text(encoding="utf-8"))
    assert manifest["config"]["oos_start"] == "2026-01-05"
    assert manifest["config"]["leverage_grid"] == "1.5"
    sweep_args = Path(manifest["sweep_args_file"]).read_text(encoding="utf-8").splitlines()
    assert sweep_args[sweep_args.index("--oos-start") + 1] == "2026-01-05"
    assert sweep_args[sweep_args.index("--leverage-grid") + 1] == "1.5"
    assert not out.exists()


def test_xgbcat_risk_parity_non_dry_run_writes_run_local_log(tmp_path: Path) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    _install_fake_sweep_module(fake_repo)

    proc = _run_script(fake_repo, DRY_RUN="0")

    assert proc.returncode == 0
    out = fake_repo / "analysis" / "xgbnew_daily" / "xgbcat_risk_parity_wide_120d"
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    run_log = Path(manifest["run_log"])
    assert run_log == (out / "sweep_stdout.log").resolve()
    log_text = run_log.read_text(encoding="utf-8")
    assert "fake sweep launched" in log_text
    assert "fill_buffer_bps_grid=0,5,10,20" in log_text
    assert "fee_regimes=deploy,prod10bps,stress36x" in log_text
    sweep_args = Path(manifest["sweep_args_file"]).read_text(encoding="utf-8").splitlines()
    argv_seen = json.loads((out / "argv_seen.json").read_text(encoding="utf-8"))
    assert argv_seen == sweep_args
    result_path = out / "sweep_20000101_000000.json"
    assert result_path.exists()
    assert (fake_repo / "logs" / "track1" / "xgbcat_risk_parity_wide_120d.log").exists()
    results_manifest = json.loads(
        Path(manifest["result_artifacts_file"]).read_text(encoding="utf-8"),
    )
    assert results_manifest == {
        "results": [
            {
                "path": str(result_path.resolve()),
                "sha256": _sha256(result_path),
                "size_bytes": result_path.stat().st_size,
            },
        ],
    }
    done = _read_key_value_file(Path(manifest["done_marker"]))
    assert done["manifest"] == str((out / "run_manifest.json").resolve())
    assert done["manifest_sha256"] == _sha256(out / "run_manifest.json")
    assert done["model_paths"] == str((out / "model_paths.txt").resolve())
    assert done["model_paths_sha256"] == _sha256(out / "model_paths.txt")
    assert done["sweep_args"] == str((out / "sweep_args.txt").resolve())
    assert done["sweep_args_sha256"] == _sha256(out / "sweep_args.txt")
    assert done["run_log"] == str((out / "sweep_stdout.log").resolve())
    assert done["run_log_sha256"] == _sha256(out / "sweep_stdout.log")
    assert done["result_artifacts"] == str((out / "result_artifacts.json").resolve())
    assert done["result_artifacts_sha256"] == _sha256(out / "result_artifacts.json")
    assert done["result_count"] == "1"
    verify = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(COMPLETION_SCRIPT),
            "--verify",
            "--done-marker",
            str(out / "sweep.done"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert verify.returncode == 0


def test_xgbcat_risk_parity_verify_only_rejects_completed_run_without_models(
    tmp_path: Path,
) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    _install_fake_sweep_module(fake_repo)
    first = _run_script(fake_repo, DRY_RUN="0")
    assert first.returncode == 0
    for model_path in fake_repo.glob("analysis/xgbnew_daily/track1_oos120d_*/*.pkl"):
        model_path.unlink()

    proc = _run_script(fake_repo, VERIFY_ONLY="1")

    assert proc.returncode == 2
    assert "manifest model 0 file missing" in proc.stderr
    assert "fake sweep launched" not in proc.stdout


def test_xgbcat_risk_parity_verify_only_fails_without_completion_marker(
    tmp_path: Path,
) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    out = fake_repo / "analysis" / "xgbnew_daily" / "xgbcat_risk_parity_wide_120d"

    proc = _run_script(fake_repo, VERIFY_ONLY="1")

    assert proc.returncode == 2
    assert "completion marker missing" in proc.stderr
    assert not out.exists()


def test_xgbcat_risk_parity_result_manifest_excludes_stale_results(tmp_path: Path) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    _install_fake_sweep_module(fake_repo)
    out = fake_repo / "analysis" / "xgbnew_daily" / "xgbcat_risk_parity_wide_120d"
    out.mkdir(parents=True)
    stale_result = out / "sweep_19990101_000000.json"
    stale_result.write_text('{"stale": true}\n', encoding="utf-8")

    proc = _run_script(fake_repo, DRY_RUN="0")

    assert proc.returncode == 0
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    results_manifest = json.loads(
        Path(manifest["result_artifacts_file"]).read_text(encoding="utf-8"),
    )
    assert [Path(row["path"]).name for row in results_manifest["results"]] == [
        "sweep_20000101_000000.json",
    ]


def test_xgbcat_risk_parity_failed_sweep_does_not_write_done_marker(tmp_path: Path) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    _install_fake_sweep_module(fake_repo)

    proc = _run_script(fake_repo, DRY_RUN="0", FAKE_SWEEP_EXIT="7")

    assert proc.returncode == 7
    out = fake_repo / "analysis" / "xgbnew_daily" / "xgbcat_risk_parity_wide_120d"
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    run_log = Path(manifest["run_log"])
    assert "fake sweep failing" in run_log.read_text(encoding="utf-8")
    assert not Path(manifest["done_marker"]).exists()


def test_xgbcat_risk_parity_fails_completion_without_final_result(tmp_path: Path) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    _install_fake_sweep_module(fake_repo)

    proc = _run_script(fake_repo, DRY_RUN="0", FAKE_SWEEP_NO_RESULT="1")

    assert proc.returncode == 2
    assert "completed without a final sweep_*.json result" in proc.stderr
    out = fake_repo / "analysis" / "xgbnew_daily" / "xgbcat_risk_parity_wide_120d"
    assert (out / "sweep_stdout.log").exists()
    assert not (out / "sweep.done").exists()
    assert not (out / "result_artifacts.json").exists()


def test_xgbcat_risk_parity_fails_completion_with_only_stale_result(tmp_path: Path) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    _install_fake_sweep_module(fake_repo)
    out = fake_repo / "analysis" / "xgbnew_daily" / "xgbcat_risk_parity_wide_120d"
    out.mkdir(parents=True)
    stale_result = out / "sweep_19990101_000000.json"
    stale_result.write_text('{"stale": true}\n', encoding="utf-8")

    proc = _run_script(fake_repo, DRY_RUN="0", FAKE_SWEEP_NO_RESULT="1")

    assert proc.returncode == 2
    assert "completed without a final sweep_*.json result" in proc.stderr
    assert stale_result.exists()
    assert not (out / "sweep.done").exists()
    assert not (out / "result_artifacts.json").exists()


def test_xgbcat_risk_parity_fails_completion_on_incomplete_result(tmp_path: Path) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    _install_fake_sweep_module(fake_repo)

    proc = _run_script(fake_repo, DRY_RUN="0", FAKE_SWEEP_INCOMPLETE_RESULT="1")

    assert proc.returncode == 2
    assert "result JSON is not complete" in proc.stderr
    out = fake_repo / "analysis" / "xgbnew_daily" / "xgbcat_risk_parity_wide_120d"
    assert (out / "sweep_stdout.log").exists()
    assert not (out / "sweep.done").exists()
    assert not (out / "result_artifacts.json").exists()


def test_xgbcat_risk_parity_fails_completion_without_production_pass(
    tmp_path: Path,
) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    _install_fake_sweep_module(fake_repo)

    proc = _run_script(fake_repo, DRY_RUN="0", FAKE_SWEEP_NO_PRODUCTION_PASS="1")

    assert proc.returncode == 2
    assert "result JSON has no production-target pass" in proc.stderr
    out = fake_repo / "analysis" / "xgbnew_daily" / "xgbcat_risk_parity_wide_120d"
    assert (out / "sweep_stdout.log").exists()
    assert not (out / "sweep.done").exists()
    assert not (out / "result_artifacts.json").exists()


def test_xgbcat_risk_parity_quarantines_marker_when_self_verify_fails(
    tmp_path: Path,
) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    _install_fake_sweep_module(fake_repo)
    venv_bin = fake_repo / ".venv313" / "bin"
    python_wrapper = venv_bin / "python"
    python_wrapper.write_text(
        f"""#!/usr/bin/env bash
set -e
real_python={sys.executable!r}
"$real_python" "$@"
rc=$?
if [ "$rc" -eq 0 ] && [ "${{1:-}}" = {str(COMPLETION_SCRIPT)!r} ]; then
  is_verify=0
  done_marker=""
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --verify)
        is_verify=1
        ;;
      --done-marker)
        shift
        done_marker="${{1:-}}"
        ;;
    esac
    shift || true
  done
  if [ "$is_verify" = "0" ] && [ -n "$done_marker" ]; then
    printf 'unexpected_key=corrupt-after-publish\\n' >> "$done_marker"
  fi
fi
exit "$rc"
""",
        encoding="utf-8",
    )
    python_wrapper.chmod(0o755)
    (venv_bin / "activate").write_text(
        f"export PATH={venv_bin}:$PATH\nexport XGBCAT_TEST_VENV=1\n",
        encoding="utf-8",
    )

    proc = _run_script(fake_repo, DRY_RUN="0")

    assert proc.returncode == 2
    assert "completion verification failed" in proc.stderr
    out = fake_repo / "analysis" / "xgbnew_daily" / "xgbcat_risk_parity_wide_120d"
    assert not (out / "sweep.done").exists()
    failed_marker = out / "sweep.done.failed.1"
    assert failed_marker.exists()
    assert "unexpected_key=corrupt-after-publish" in failed_marker.read_text(
        encoding="utf-8",
    )
    assert (out / "sweep_stdout.log").exists()


def test_xgbcat_risk_parity_refuses_to_overwrite_completed_run(tmp_path: Path) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    _install_fake_sweep_module(fake_repo)
    first = _run_script(fake_repo, DRY_RUN="0")
    assert first.returncode == 0
    out = fake_repo / "analysis" / "xgbnew_daily" / "xgbcat_risk_parity_wide_120d"
    done_marker = out / "sweep.done"
    done_before = done_marker.read_bytes()

    proc = _run_script(fake_repo, DRY_RUN="0")

    assert proc.returncode == 2
    assert "completed sweep already exists" in proc.stderr
    assert done_marker.read_bytes() == done_before


def test_xgbcat_risk_parity_dry_run_does_not_overwrite_completed_evidence(
    tmp_path: Path,
) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    _install_fake_sweep_module(fake_repo)
    first = _run_script(fake_repo, DRY_RUN="0")
    assert first.returncode == 0
    out = fake_repo / "analysis" / "xgbnew_daily" / "xgbcat_risk_parity_wide_120d"
    done_marker = out / "sweep.done"
    done_before = done_marker.read_bytes()

    proc = _run_script(fake_repo, DRY_RUN="1", ALLOW_OVERWRITE="1")

    assert proc.returncode == 2
    assert "dry-run would overwrite evidence" in proc.stderr
    assert done_marker.read_bytes() == done_before


def test_xgbcat_risk_parity_overwrite_failure_removes_stale_done_marker(
    tmp_path: Path,
) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    _install_fake_sweep_module(fake_repo)
    first = _run_script(fake_repo, DRY_RUN="0")
    assert first.returncode == 0
    out = fake_repo / "analysis" / "xgbnew_daily" / "xgbcat_risk_parity_wide_120d"
    done_marker = out / "sweep.done"
    assert done_marker.exists()

    proc = _run_script(fake_repo, DRY_RUN="0", ALLOW_OVERWRITE="1", FAKE_SWEEP_EXIT="7")

    assert proc.returncode == 7
    assert not done_marker.exists()
    assert "fake sweep failing" in (out / "sweep_stdout.log").read_text(encoding="utf-8")


def test_xgbcat_risk_parity_bad_overwrite_preserves_completed_evidence(
    tmp_path: Path,
) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    _install_fake_sweep_module(fake_repo)
    first = _run_script(fake_repo, DRY_RUN="0")
    assert first.returncode == 0
    out = fake_repo / "analysis" / "xgbnew_daily" / "xgbcat_risk_parity_wide_120d"
    done_marker = out / "sweep.done"
    done_before = done_marker.read_bytes()
    manifest_before = (out / "run_manifest.json").read_bytes()
    sweep_args_before = (out / "sweep_args.txt").read_bytes()

    proc = _run_script(
        fake_repo,
        DRY_RUN="0",
        ALLOW_OVERWRITE="1",
        FILL_BUFFER_BPS_GRID="0,5,10",
    )

    assert proc.returncode == 2
    assert "fill_buffer_bps_grid missing required cell: 20 bps" in proc.stderr
    assert done_marker.read_bytes() == done_before
    assert (out / "run_manifest.json").read_bytes() == manifest_before
    assert (out / "sweep_args.txt").read_bytes() == sweep_args_before


def test_xgbcat_risk_parity_uses_venv313_by_default() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'VENV="${VENV:-.venv313}"' in text
    assert 'source "$VENV/bin/activate"' in text
    assert "XGB=$(ls " not in text
    assert "CAT=$(ls " not in text


def test_xgbcat_risk_parity_fails_before_sweep_without_required_slippage_cell(
    tmp_path: Path,
) -> None:
    fake_repo = _make_fake_repo(tmp_path)

    proc = _run_script(fake_repo, DRY_RUN="0", FILL_BUFFER_BPS_GRID="0,5,10")

    assert proc.returncode == 2
    assert "fill_buffer_bps_grid missing required cell: 20 bps" in proc.stderr
    assert "dry run: preflight passed" not in proc.stdout


def test_xgbcat_risk_parity_fails_before_sweep_when_leverage_is_unsafe(
    tmp_path: Path,
) -> None:
    fake_repo = _make_fake_repo(tmp_path)

    proc = _run_script(fake_repo, DRY_RUN="0", LEVERAGE_GRID="1.5,3.0")

    assert proc.returncode == 2
    assert "leverage_grid exceeds production limit: 3 > 2.5" in proc.stderr
    assert "dry run: preflight passed" not in proc.stdout


def test_xgbcat_risk_parity_fails_before_sweep_on_empty_numeric_grid_token(
    tmp_path: Path,
) -> None:
    fake_repo = _make_fake_repo(tmp_path)

    proc = _run_script(fake_repo, DRY_RUN="0", FILL_BUFFER_BPS_GRID="0,,5,10,20")

    assert proc.returncode == 2
    assert "fill_buffer_bps_grid must be a finite comma-separated numeric grid" in proc.stderr
    assert "dry run: preflight passed" not in proc.stdout


def test_xgbcat_risk_parity_fails_before_sweep_on_nonfinite_numeric_grid(
    tmp_path: Path,
) -> None:
    fake_repo = _make_fake_repo(tmp_path)

    proc = _run_script(fake_repo, DRY_RUN="0", FILL_BUFFER_BPS_GRID="0,5,10,20,nan")

    assert proc.returncode == 2
    assert "fill_buffer_bps_grid must be a finite comma-separated numeric grid" in proc.stderr
    assert "dry run: preflight passed" not in proc.stdout


def test_xgbcat_risk_parity_fails_before_sweep_when_fail_fast_dd_is_unsafe(
    tmp_path: Path,
) -> None:
    fake_repo = _make_fake_repo(tmp_path)

    proc = _run_script(fake_repo, DRY_RUN="0", FAIL_FAST_MAX_DD_PCT="40")

    assert proc.returncode == 2
    assert "fail_fast_max_dd_pct exceeds production limit: 40 > 20" in proc.stderr
    assert "dry run: preflight passed" not in proc.stdout


def test_xgbcat_risk_parity_fails_before_sweep_when_negative_window_gate_is_loose(
    tmp_path: Path,
) -> None:
    fake_repo = _make_fake_repo(tmp_path)

    proc = _run_script(fake_repo, DRY_RUN="0", FAIL_FAST_NEG_WINDOWS="2")

    assert proc.returncode == 2
    assert "fail_fast_neg_windows exceeds production limit: 2 > 1" in proc.stderr
    assert "dry run: preflight passed" not in proc.stdout


def test_xgbcat_risk_parity_fails_before_sweep_without_stress_fee_regime(
    tmp_path: Path,
) -> None:
    fake_repo = _make_fake_repo(tmp_path)

    proc = _run_script(fake_repo, DRY_RUN="0", FEE_REGIMES="deploy,prod10bps")

    assert proc.returncode == 2
    assert "fee_regimes missing required regime: stress36x" in proc.stderr
    assert "dry run: preflight passed" not in proc.stdout


def test_xgbcat_risk_parity_fails_before_sweep_without_prod10bps_fee_regime(
    tmp_path: Path,
) -> None:
    fake_repo = _make_fake_repo(tmp_path)

    proc = _run_script(fake_repo, DRY_RUN="0", FEE_REGIMES="deploy,stress36x")

    assert proc.returncode == 2
    assert "fee_regimes missing required regime: prod10bps" in proc.stderr
    assert "dry run: preflight passed" not in proc.stdout


def test_xgbcat_risk_parity_fails_before_sweep_with_unknown_fee_regime(
    tmp_path: Path,
) -> None:
    fake_repo = _make_fake_repo(tmp_path)

    proc = _run_script(fake_repo, DRY_RUN="0", FEE_REGIMES="deploy,prod10bps,stress36x,paper")

    assert proc.returncode == 2
    assert "fee_regimes contains unknown regime: paper" in proc.stderr
    assert "dry run: preflight passed" not in proc.stdout


def test_xgbcat_risk_parity_fails_before_sweep_on_empty_fee_regime_token(
    tmp_path: Path,
) -> None:
    fake_repo = _make_fake_repo(tmp_path)

    proc = _run_script(fake_repo, DRY_RUN="0", FEE_REGIMES="deploy,,prod10bps,stress36x")

    assert proc.returncode == 2
    assert "fee_regimes must not be empty" in proc.stderr
    assert "dry run: preflight passed" not in proc.stdout


def test_xgbcat_risk_parity_fails_before_sweep_when_symbols_missing(tmp_path: Path) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    (fake_repo / "symbol_lists" / "stocks_wide_fresh0401_photonics_2500_v1.txt").unlink()
    (fake_repo / "symbol_lists" / "stocks_wide_1000_v1.txt").unlink(missing_ok=True)

    proc = _run_script(fake_repo)

    assert proc.returncode == 2
    assert "missing symbols file" in proc.stderr
    assert "dry run: preflight passed" not in proc.stdout


def test_xgbcat_risk_parity_fails_before_sweep_when_models_missing(tmp_path: Path) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    for path in (fake_repo / "analysis" / "xgbnew_daily" / "track1_oos120d_cat").glob("*.pkl"):
        path.unlink()

    proc = _run_script(fake_repo)

    assert proc.returncode == 2
    assert "no CatBoost alltrain_seed*.pkl models found" in proc.stderr
    assert "dry run: preflight passed" not in proc.stdout


def test_xgbcat_risk_parity_rejects_duplicate_model_artifacts(tmp_path: Path) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    shared_dir = fake_repo / "analysis" / "xgbnew_daily" / "track1_oos120d_xgb"

    proc = _run_script(
        fake_repo,
        CAT_DIR=str(shared_dir),
    )

    assert proc.returncode == 2
    assert "duplicate model artifact" in proc.stderr
    assert "dry run: preflight passed" not in proc.stdout
