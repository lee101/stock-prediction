#!/usr/bin/env python3
"""Publish completion evidence for xgbcat_risk_parity_wide_120d.sh."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sys


COMPLETION_SCHEMA_VERSION = 1
PRODUCTION_TARGET_MEDIAN_MONTHLY_PCT = 27.0
PRODUCTION_TARGET_MAX_DD_PCT = 25.0
PRODUCTION_TARGET_MAX_NEG_WINDOWS = 0
PRODUCTION_TARGET_MIN_WINDOWS = 1
REQUIRED_FEE_REGIMES = {"deploy", "prod10bps", "stress36x"}
EXPECTED_FEE_REGIME_CONFIG = {
    "deploy": {"commission_bps": 0.0, "fee_rate": 0.0000278, "fill_buffer_bps": 5.0},
    "prod10bps": {"commission_bps": 0.0, "fee_rate": 0.001, "fill_buffer_bps": 5.0},
    "stress36x": {"commission_bps": 10.0, "fee_rate": 0.001, "fill_buffer_bps": 15.0},
}
EXPECTED_GOODNESS_WEIGHTS = {
    "p10_coef": 1.0,
    "dd_coef": 1.0,
    "neg_coef": 100.0,
}
EXPECTED_ROBUST_GOODNESS_WEIGHTS = {
    "p10_coef": 1.0,
    "dd_coef": 1.5,
    "neg_count_coef": 50.0,
    "neg_magnitude_coef": 2.0,
}
EXPECTED_PAIN_ADJUSTED_GOODNESS_WEIGHTS = {
    "tuw_coef": 0.25,
    "ulcer_coef": 1.0,
}
REQUIRED_FILL_BUFFER_BPS = {0.0, 5.0, 10.0, 20.0}
MAX_ALLOWED_LEVERAGE = 2.5
MAX_ALLOWED_FAIL_FAST_DD_PCT = 20.0
MAX_ALLOWED_FAIL_FAST_NEG_WINDOWS = 1
FAIL_FAST_SCORE = -1_000_000_000.0
SWEEP_ARG_CONFIG_FLAGS = {
    "--allocation-mode-grid": "allocation_mode_grid",
    "--allocation-temp-grid": "allocation_temp_grid",
    "--checkpoint-every-cells": "checkpoint_every_cells",
    "--fee-regimes": "fee_regimes",
    "--fail-fast-max-dd-pct": "fail_fast_max_dd_pct",
    "--fail-fast-max-intraday-dd-pct": "fail_fast_max_intraday_dd_pct",
    "--fail-fast-neg-windows": "fail_fast_neg_windows",
    "--fill-buffer-bps-grid": "fill_buffer_bps_grid",
    "--inference-max-spread-bps-grid": "inference_max_spread_bps_grid",
    "--inference-max-vol-grid": "inference_max_vol_grid",
    "--inference-min-dolvol-grid": "inference_min_dolvol_grid",
    "--inference-min-vol-grid": "inference_min_vol_grid",
    "--inv-vol-cap": "inv_vol_cap",
    "--inv-vol-floor": "inv_vol_floor",
    "--inv-vol-target-grid": "inv_vol_target_grid",
    "--leverage-grid": "leverage_grid",
    "--min-dollar-vol": "min_dollar_vol",
    "--min-score-grid": "min_score_grid",
    "--oos-end": "oos_end",
    "--oos-start": "oos_start",
    "--regime-cs-skew-min-grid": "regime_cs_skew_min_grid",
    "--score-uncertainty-penalty-grid": "score_uncertainty_penalty_grid",
    "--stride-days": "stride_days",
    "--top-n-grid": "top_n_grid",
    "--vol-target-ann-grid": "vol_target_ann_grid",
    "--window-days": "window_days",
}
REQUIRED_SWEEP_FLAGS = {
    "--fast-features",
    "--hold-through",
    "--require-production-target",
}
EXPECTED_FIXED_FLAGS = {
    "fast_features": True,
    "hold_through": True,
    "require_production_target": True,
}
ALLOWED_VALUELESS_SWEEP_FLAGS = REQUIRED_SWEEP_FLAGS | {"--verbose"}
SPECIAL_VALUE_SWEEP_FLAGS = {
    "--model-paths",
    "--output-dir",
    "--symbols-file",
}
EXPECTED_MANIFEST_ROOT_KEYS = {
    "cat_dir",
    "cat_model_count",
    "config",
    "done_marker",
    "fixed_flags",
    "model_paths_file",
    "models",
    "output_dir",
    "repo",
    "result_artifacts_file",
    "run_log",
    "script",
    "sweep_args_file",
    "symbols_file",
    "symbols_sha256",
    "venv",
    "xgb_dir",
    "xgb_model_count",
}
EXPECTED_MODEL_ROW_KEYS = {"family", "path", "sha256"}
EXPECTED_SCRIPT = "scripts/xgbcat_risk_parity_wide_120d.sh"
STRATEGY_PARAM_FIELDS = (
    "leverage",
    "min_score",
    "hold_through",
    "top_n",
    "min_picks",
    "opportunistic_watch_n",
    "opportunistic_entry_discount_bps",
    "score_uncertainty_penalty",
    "inference_min_dolvol",
    "inference_max_spread_bps",
    "inference_min_vol_20d",
    "inference_max_vol_20d",
    "regime_gate_window",
    "vol_target_ann",
    "inv_vol_target_ann",
    "inv_vol_floor",
    "inv_vol_cap",
    "max_ret_20d_rank_pct",
    "min_ret_5d_rank_pct",
    "regime_cs_iqr_max",
    "regime_cs_skew_min",
    "no_picks_fallback_symbol",
    "no_picks_fallback_alloc_scale",
    "conviction_scaled_alloc",
    "conviction_alloc_low",
    "conviction_alloc_high",
    "allocation_mode",
    "allocation_temp",
)
ROBUST_SUMMARY_HIGH_GOOD_METRICS = (
    "median_monthly_pct",
    "p10_monthly_pct",
    "median_sortino",
    "goodness_score",
    "robust_goodness_score",
    "pain_adjusted_goodness_score",
    "median_active_day_pct",
    "min_active_day_pct",
)
ROBUST_SUMMARY_LOW_GOOD_METRICS = (
    "worst_dd_pct",
    "worst_intraday_dd_pct",
    "avg_intraday_dd_pct",
    "time_under_water_pct",
    "ulcer_index",
)
EXPECTED_RESULT_ROOT_KEYS = {
    "best_production_target_strategy",
    "blend_mode",
    "cells",
    "complete",
    "data_root",
    "ensemble_feature_mode",
    "ensemble_manifest",
    "fail_fast",
    "fee_regimes",
    "fm_latents_path",
    "fm_latents_sha256",
    "fm_n_latents",
    "friction_robust_strategies",
    "goodness_weights",
    "model_paths",
    "model_sha256",
    "n_cells",
    "n_friction_robust_strategies",
    "n_production_target_pass",
    "oos_end",
    "oos_start",
    "pain_adjusted_goodness_weights",
    "production_target",
    "robust_goodness_weights",
    "spy_csv",
    "spy_csv_sha256",
    "stride_days",
    "symbols_file",
    "window_days",
}
ENSEMBLE_MANIFEST_OPTIONAL_KEYS = {"trained_at", "train_start", "train_end", "seeds", "config"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    _write_bytes_atomic(path, text.encode("utf-8"))


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


@contextmanager
def _completion_publish_lock(done_marker: Path) -> Iterator[None]:
    lock_path = done_marker.with_name(".xgbcat_completion_publish.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail(f"completion publication already in progress: {done_marker}")
        yield


def _quarantine_completion_marker(done_marker: Path) -> Path:
    idx = 1
    failed_marker = done_marker.with_name(f"{done_marker.name}.failed.{idx}")
    while failed_marker.exists():
        idx += 1
        failed_marker = done_marker.with_name(f"{done_marker.name}.failed.{idx}")
    if done_marker.exists():
        done_marker.replace(failed_marker)
    return failed_marker


def _restore_sidecar(path: Path, previous_bytes: bytes | None) -> None:
    if previous_bytes is None:
        path.unlink(missing_ok=True)
        return
    _write_bytes_atomic(path, previous_bytes)


def _read_key_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        _fail(f"completion marker missing: {path}")
    rows: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line or "=" not in line:
            _fail(f"malformed completion marker line {lineno}: {path}")
        key, value = line.split("=", 1)
        if not key or key in rows:
            _fail(f"malformed completion marker key on line {lineno}: {path}")
        rows[key] = value
    return rows


def _validate_completed_at(value: str) -> None:
    if not value.endswith("Z"):
        _fail("completion marker completed_at must be a UTC ISO timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        _fail("completion marker completed_at is not a valid ISO timestamp")
    if parsed.tzinfo != UTC:
        _fail("completion marker completed_at must be UTC")


def _fail(message: str) -> None:
    print(f"[xgbcat-risk] ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _number_value(row: dict[str, object], key: str, label: str) -> float:
    value = row.get(key)
    if not _is_number(value) or not math.isfinite(float(value)):
        _fail(f"{label} missing finite numeric {key}")
    return float(value)


def _int_value(row: dict[str, object], key: str, label: str) -> int:
    value = row.get(key)
    if not _is_int(value):
        _fail(f"{label} missing integer {key}")
    return int(value)


def _validate_numeric_mapping(
    actual: object,
    expected: dict[str, float],
    *,
    label: str,
    path: Path,
) -> None:
    if not isinstance(actual, dict):
        _fail(f"result JSON {label} must be an object: {path}")
    actual_keys = {str(key) for key in actual}
    if actual_keys != set(expected):
        _fail(f"result JSON {label} keys do not match verifier contract: {path}")
    for key, expected_value in sorted(expected.items()):
        actual_value = actual.get(key)
        if not _is_number(actual_value) or float(actual_value) != expected_value:
            _fail(f"result JSON {label} {key} mismatch: {path}")


def _same_float(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)


def _percentile_linear(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower_idx = int(math.floor(position))
    upper_idx = int(math.ceil(position))
    if lower_idx == upper_idx:
        return ordered[lower_idx]
    lower = ordered[lower_idx]
    upper = ordered[upper_idx]
    return lower + (upper - lower) * (position - lower_idx)


def _numeric_vector_values(row: dict[str, object], key: str, label: str) -> list[float]:
    values = row.get(key)
    if not isinstance(values, list):
        _fail(f"{label} missing {key} list")
    parsed: list[float] = []
    for idx, value in enumerate(values):
        if not _is_number(value) or not math.isfinite(float(value)):
            _fail(f"{label} has non-finite {key} entry {idx}")
        parsed.append(float(value))
    return parsed


def _date_value(value: object, label: str) -> date:
    if not isinstance(value, str):
        _fail(f"{label} must be an ISO date string")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        _fail(f"{label} must be an ISO date string")
    if parsed.isoformat() != value:
        _fail(f"{label} must be a canonical ISO date")
    return parsed


def _date_vector_values(
    row: dict[str, object],
    key: str,
    label: str,
    n_windows: int,
) -> list[date]:
    values = row.get(key)
    if not isinstance(values, list):
        _fail(f"{label} missing {key} list")
    if len(values) != n_windows:
        _fail(f"{label} {key} length does not match n_windows")
    return [
        _date_value(value, f"{label} {key} entry {idx}")
        for idx, value in enumerate(values)
    ]


def _validate_raw_cell_scores(
    row: dict[str, object],
    label: str,
    *,
    oos_start: str,
    oos_end: str,
) -> None:
    n_windows = _int_value(row, "n_windows", label)
    n_neg = _int_value(row, "n_neg", label)
    if n_windows < 0 or n_neg < 0 or n_neg > n_windows:
        _fail(f"{label} has invalid n_windows/n_neg")
    expected_oos_start = _date_value(oos_start, f"{label} oos_start")
    expected_oos_end = _date_value(oos_end, f"{label} oos_end")
    if expected_oos_start > expected_oos_end:
        _fail(f"{label} has invalid OOS date range")
    window_starts = _date_vector_values(row, "window_start_dates", label, n_windows)
    window_ends = _date_vector_values(row, "window_end_dates", label, n_windows)
    previous_start = None
    previous_end = None
    for idx, (start, end) in enumerate(zip(window_starts, window_ends, strict=True)):
        if start > end:
            _fail(f"{label} window date range {idx} has start after end")
        if start < expected_oos_start or end > expected_oos_end:
            _fail(f"{label} window date range {idx} is outside OOS span")
        if previous_start is not None and start <= previous_start:
            _fail(f"{label} window_start_dates are not strictly increasing")
        if previous_end is not None and end <= previous_end:
            _fail(f"{label} window_end_dates are not strictly increasing")
        previous_start = start
        previous_end = end
    monthly_returns = _numeric_vector_values(row, "monthly_return_pcts", label)
    if len(monthly_returns) != n_windows:
        _fail(f"{label} monthly_return_pcts length does not match n_windows")
    window_sortinos = _numeric_vector_values(row, "window_sortino_values", label)
    if len(window_sortinos) != n_windows:
        _fail(f"{label} window_sortino_values length does not match n_windows")
    window_drawdowns = _numeric_vector_values(row, "window_drawdown_pcts", label)
    if len(window_drawdowns) != n_windows:
        _fail(f"{label} window_drawdown_pcts length does not match n_windows")
    window_tuws = _numeric_vector_values(row, "window_time_under_water_pcts", label)
    if len(window_tuws) != n_windows:
        _fail(f"{label} window_time_under_water_pcts length does not match n_windows")
    window_ulcers = _numeric_vector_values(row, "window_ulcer_indexes", label)
    if len(window_ulcers) != n_windows:
        _fail(f"{label} window_ulcer_indexes length does not match n_windows")
    window_active_days = _numeric_vector_values(row, "window_active_day_pcts", label)
    if len(window_active_days) != n_windows:
        _fail(f"{label} window_active_day_pcts length does not match n_windows")
    window_worst_intraday_dds = _numeric_vector_values(
        row,
        "window_worst_intraday_dd_pcts",
        label,
    )
    if len(window_worst_intraday_dds) != n_windows:
        _fail(f"{label} window_worst_intraday_dd_pcts length does not match n_windows")
    window_avg_intraday_dds = _numeric_vector_values(
        row,
        "window_avg_intraday_dd_pcts",
        label,
    )
    if len(window_avg_intraday_dds) != n_windows:
        _fail(f"{label} window_avg_intraday_dd_pcts length does not match n_windows")
    derived_n_neg = sum(1 for value in monthly_returns if value < 0.0)
    if n_neg != derived_n_neg:
        _fail(f"{label} n_neg does not match monthly_return_pcts")
    derived_median = _percentile_linear(monthly_returns, 50.0)
    if not _same_float(_number_value(row, "median_monthly_pct", label), derived_median):
        _fail(f"{label} median_monthly_pct does not match monthly_return_pcts")
    derived_p10 = _percentile_linear(monthly_returns, 10.0)
    if not _same_float(_number_value(row, "p10_monthly_pct", label), derived_p10):
        _fail(f"{label} p10_monthly_pct does not match monthly_return_pcts")
    derived_median_sortino = _percentile_linear(window_sortinos, 50.0)
    if not _same_float(_number_value(row, "median_sortino", label), derived_median_sortino):
        _fail(f"{label} median_sortino does not match window_sortino_values")
    derived_worst_dd = max(window_drawdowns) if window_drawdowns else 0.0
    if not _same_float(_number_value(row, "worst_dd_pct", label), derived_worst_dd):
        _fail(f"{label} worst_dd_pct does not match window_drawdown_pcts")
    derived_tuw = _percentile_linear(window_tuws, 50.0)
    if not _same_float(_number_value(row, "time_under_water_pct", label), derived_tuw):
        _fail(
            f"{label} time_under_water_pct does not match "
            "window_time_under_water_pcts",
        )
    derived_ulcer = _percentile_linear(window_ulcers, 50.0)
    if not _same_float(_number_value(row, "ulcer_index", label), derived_ulcer):
        _fail(f"{label} ulcer_index does not match window_ulcer_indexes")
    derived_median_active_day = _percentile_linear(window_active_days, 50.0)
    if not _same_float(
        _number_value(row, "median_active_day_pct", label),
        derived_median_active_day,
    ):
        _fail(f"{label} median_active_day_pct does not match window_active_day_pcts")
    derived_min_active_day = min(window_active_days) if window_active_days else 0.0
    if not _same_float(
        _number_value(row, "min_active_day_pct", label),
        derived_min_active_day,
    ):
        _fail(f"{label} min_active_day_pct does not match window_active_day_pcts")
    derived_worst_intraday_dd = (
        max(window_worst_intraday_dds) if window_worst_intraday_dds else 0.0
    )
    if not _same_float(
        _number_value(row, "worst_intraday_dd_pct", label),
        derived_worst_intraday_dd,
    ):
        _fail(
            f"{label} worst_intraday_dd_pct does not match "
            "window_worst_intraday_dd_pcts",
        )
    derived_avg_intraday_dd = (
        sum(window_avg_intraday_dds) / len(window_avg_intraday_dds)
        if window_avg_intraday_dds
        else 0.0
    )
    if not _same_float(
        _number_value(row, "avg_intraday_dd_pct", label),
        derived_avg_intraday_dd,
    ):
        _fail(
            f"{label} avg_intraday_dd_pct does not match "
            "window_avg_intraday_dd_pcts",
        )
    derived_mean_abs_neg = (
        -sum(value for value in monthly_returns if value < 0.0) / n_windows
        if n_windows > 0 and derived_n_neg
        else 0.0
    )
    mean_abs_neg = _number_value(row, "mean_abs_neg_monthly_pct", label)
    if mean_abs_neg < 0.0:
        _fail(f"{label} mean_abs_neg_monthly_pct must be non-negative")
    if not _same_float(mean_abs_neg, derived_mean_abs_neg):
        _fail(f"{label} mean_abs_neg_monthly_pct does not match monthly_return_pcts")
    if n_neg == 0 and mean_abs_neg != 0.0:
        _fail(f"{label} mean_abs_neg_monthly_pct does not match n_neg")
    if n_neg > 0 and mean_abs_neg <= 0.0:
        _fail(f"{label} mean_abs_neg_monthly_pct does not match n_neg")

    fail_fast = row.get("fail_fast_triggered")
    if not isinstance(fail_fast, bool):
        _fail(f"{label} missing boolean fail_fast_triggered")

    goodness_score = _number_value(row, "goodness_score", label)
    robust_goodness_score = _number_value(row, "robust_goodness_score", label)
    pain_adjusted_score = _number_value(row, "pain_adjusted_goodness_score", label)
    if fail_fast:
        if (
            goodness_score != FAIL_FAST_SCORE
            or robust_goodness_score != FAIL_FAST_SCORE
            or pain_adjusted_score != FAIL_FAST_SCORE
        ):
            _fail(f"{label} fail-fast scores do not match verifier contract")
        return

    expected_goodness = (
        EXPECTED_GOODNESS_WEIGHTS["p10_coef"]
        * _number_value(row, "p10_monthly_pct", label)
        - EXPECTED_GOODNESS_WEIGHTS["dd_coef"]
        * _number_value(row, "worst_dd_pct", label)
        - EXPECTED_GOODNESS_WEIGHTS["neg_coef"]
        * (float(n_neg) / float(max(n_windows, 1)))
    )
    if not _same_float(goodness_score, expected_goodness):
        _fail(f"{label} goodness_score does not match raw metrics")

    expected_robust_goodness = (
        EXPECTED_ROBUST_GOODNESS_WEIGHTS["p10_coef"]
        * _number_value(row, "p10_monthly_pct", label)
        - EXPECTED_ROBUST_GOODNESS_WEIGHTS["dd_coef"]
        * _number_value(row, "worst_dd_pct", label)
        - EXPECTED_ROBUST_GOODNESS_WEIGHTS["neg_count_coef"]
        * (float(n_neg) / float(max(n_windows, 1)))
        - EXPECTED_ROBUST_GOODNESS_WEIGHTS["neg_magnitude_coef"] * mean_abs_neg
    )
    if not _same_float(robust_goodness_score, expected_robust_goodness):
        _fail(f"{label} robust_goodness_score does not match raw metrics")

    expected_pain_adjusted = (
        robust_goodness_score
        - EXPECTED_PAIN_ADJUSTED_GOODNESS_WEIGHTS["tuw_coef"]
        * _number_value(row, "time_under_water_pct", label)
        - EXPECTED_PAIN_ADJUSTED_GOODNESS_WEIGHTS["ulcer_coef"]
        * _number_value(row, "ulcer_index", label)
    )
    if not _same_float(pain_adjusted_score, expected_pain_adjusted):
        _fail(f"{label} pain_adjusted_goodness_score does not match raw metrics")


def _validate_robust_summary_metric(
    row: dict[str, object],
    matching_cells: list[dict[str, object]],
    *,
    source_metric: str,
    summary_key: str,
    reducer: str,
    label: str,
) -> None:
    summary_value = _number_value(row, summary_key, label)
    raw_values = [
        _number_value(cell, source_metric, f"{label} raw cell")
        for cell in matching_cells
    ]
    if reducer == "min":
        derived = min(raw_values)
    elif reducer == "max":
        derived = max(raw_values)
    else:
        _fail(f"internal verifier error: unknown reducer {reducer}")
    if not _same_float(summary_value, derived):
        _fail(f"{label} {summary_key} does not match raw cells")


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"invalid {label} JSON {path}: {exc}")
    if not isinstance(payload, dict):
        _fail(f"{label} JSON must be an object: {path}")
    return payload


def _load_sweep_args(path: Path) -> list[str]:
    if not path.is_file():
        _fail(f"sweep args missing: {path}")
    args = path.read_text(encoding="utf-8").splitlines()
    if not args:
        _fail(f"sweep args file is empty: {path}")
    return args


def _parse_csv(value: object, label: str) -> list[str]:
    if not isinstance(value, str):
        _fail(f"{label} must be a comma-separated string")
    parsed = [item.strip() for item in value.split(",")]
    if not parsed or any(not item for item in parsed):
        _fail(f"{label} must not contain empty values")
    return parsed


def _parse_float_grid(value: object, label: str) -> list[float]:
    parsed_strings = _parse_csv(value, label)
    try:
        parsed = [float(item) for item in parsed_strings]
    except ValueError:
        _fail(f"{label} must be a finite comma-separated numeric grid")
    if any(not math.isfinite(item) for item in parsed):
        _fail(f"{label} must be finite")
    return parsed


def _parse_float_config(config: dict[object, object], key: str, label: str) -> float:
    value = config.get(key)
    if not isinstance(value, str):
        _fail(f"{label} must be a string")
    try:
        parsed = float(value)
    except ValueError:
        _fail(f"{label} must be numeric")
    if not math.isfinite(parsed):
        _fail(f"{label} must be finite")
    return parsed


def _parse_int_config(config: dict[object, object], key: str, label: str) -> int:
    value = config.get(key)
    if not isinstance(value, str):
        _fail(f"{label} must be a string")
    try:
        return int(value)
    except ValueError:
        _fail(f"{label} must be an integer")


def _validate_manifest_production_contract(config: dict[object, object], manifest_path: Path) -> None:
    fee_regimes = set(_parse_csv(config.get("fee_regimes"), "manifest config fee_regimes"))
    unknown_fee_regimes = sorted(fee_regimes - REQUIRED_FEE_REGIMES)
    if unknown_fee_regimes:
        _fail(f"manifest config fee_regimes has unknown regime: {unknown_fee_regimes[0]}")
    missing_fee_regimes = sorted(REQUIRED_FEE_REGIMES - fee_regimes)
    if missing_fee_regimes:
        _fail(f"manifest config fee_regimes missing required regime: {missing_fee_regimes[0]}")

    fill_buffers = set(
        _parse_float_grid(
            config.get("fill_buffer_bps_grid"),
            "manifest config fill_buffer_bps_grid",
        ),
    )
    negative_fill_buffers = sorted(value for value in fill_buffers if value < 0.0)
    if negative_fill_buffers:
        _fail(
            "manifest config fill_buffer_bps_grid must be non-negative: "
            f"{negative_fill_buffers[0]:g}",
        )
    missing_fill_buffers = sorted(REQUIRED_FILL_BUFFER_BPS - fill_buffers)
    if missing_fill_buffers:
        _fail(
            "manifest config fill_buffer_bps_grid missing required cell: "
            f"{missing_fill_buffers[0]:g} bps",
        )

    leverage_grid = _parse_float_grid(
        config.get("leverage_grid"),
        "manifest config leverage_grid",
    )
    unsafe_leverage = sorted(
        value
        for value in leverage_grid
        if value <= 0.0 or value > MAX_ALLOWED_LEVERAGE
    )
    if unsafe_leverage:
        _fail(
            "manifest config leverage_grid exceeds production limit: "
            f"{unsafe_leverage[-1]:g} > {MAX_ALLOWED_LEVERAGE:g}",
        )

    for key in ("fail_fast_max_dd_pct", "fail_fast_max_intraday_dd_pct"):
        value = _parse_float_config(config, key, f"manifest config {key}")
        if value <= 0.0 or value > MAX_ALLOWED_FAIL_FAST_DD_PCT:
            _fail(
                f"manifest config {key} exceeds production limit: "
                f"{value:g} > {MAX_ALLOWED_FAIL_FAST_DD_PCT:g}",
            )
    neg_windows = _parse_int_config(
        config,
        "fail_fast_neg_windows",
        "manifest config fail_fast_neg_windows",
    )
    if neg_windows < 0 or neg_windows > MAX_ALLOWED_FAIL_FAST_NEG_WINDOWS:
        _fail(
            "manifest config fail_fast_neg_windows exceeds production limit: "
            f"{neg_windows} > {MAX_ALLOWED_FAIL_FAST_NEG_WINDOWS}",
        )


def _validate_manifest_fixed_flags(manifest: dict[str, object], manifest_path: Path) -> None:
    fixed_flags = manifest.get("fixed_flags")
    if not isinstance(fixed_flags, dict):
        _fail(f"manifest missing fixed_flags object: {manifest_path}")
    missing = sorted(set(EXPECTED_FIXED_FLAGS) - set(fixed_flags))
    if missing:
        _fail(f"manifest fixed_flags missing key: {missing[0]}")
    unexpected = sorted(set(fixed_flags) - set(EXPECTED_FIXED_FLAGS))
    if unexpected:
        _fail(f"manifest fixed_flags unexpected key: {unexpected[0]}")
    for key, expected_value in sorted(EXPECTED_FIXED_FLAGS.items()):
        if fixed_flags.get(key) is not expected_value:
            _fail(f"manifest fixed_flags {key} must be true: {manifest_path}")


def _validate_manifest_schema(manifest: dict[str, object], manifest_path: Path) -> None:
    missing = sorted(EXPECTED_MANIFEST_ROOT_KEYS - set(manifest))
    if missing:
        _fail(f"manifest missing field: {missing[0]}")
    unexpected = sorted(set(manifest) - EXPECTED_MANIFEST_ROOT_KEYS)
    if unexpected:
        _fail(f"manifest unexpected field: {unexpected[0]}")
    if manifest.get("script") != EXPECTED_SCRIPT:
        _fail(f"manifest script mismatch: {manifest.get('script')}")
    if not isinstance(manifest.get("venv"), str) or not manifest["venv"]:
        _fail(f"manifest venv must be a non-empty string: {manifest_path}")
    config = manifest.get("config")
    if not isinstance(config, dict):
        _fail(f"manifest missing config object: {manifest_path}")
    expected_config_keys = set(SWEEP_ARG_CONFIG_FLAGS.values())
    missing_config = sorted(expected_config_keys - set(config))
    if missing_config:
        _fail(f"manifest config missing key: {missing_config[0]}")
    unexpected_config = sorted(set(config) - expected_config_keys)
    if unexpected_config:
        _fail(f"manifest config unexpected key: {unexpected_config[0]}")
    xgb_count = manifest.get("xgb_model_count")
    cat_count = manifest.get("cat_model_count")
    if not _is_int(xgb_count) or xgb_count < 0:
        _fail(f"manifest xgb_model_count must be a non-negative integer: {manifest_path}")
    if not _is_int(cat_count) or cat_count < 0:
        _fail(f"manifest cat_model_count must be a non-negative integer: {manifest_path}")
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        _fail(f"manifest missing models list: {manifest_path}")
    repo = _resolve_manifest_path(
        manifest.get("repo"),
        repo=manifest_path.parent,
        label="repo",
    )
    family_dirs = {
        "xgb": _resolve_manifest_path(
            manifest.get("xgb_dir"),
            repo=repo,
            label="xgb_dir",
        ),
        "cat": _resolve_manifest_path(
            manifest.get("cat_dir"),
            repo=repo,
            label="cat_dir",
        ),
    }
    if family_dirs["xgb"] == family_dirs["cat"]:
        _fail(f"manifest xgb_dir and cat_dir must be distinct: {manifest_path}")
    observed_counts = {"xgb": 0, "cat": 0}
    for idx, row in enumerate(models):
        if not isinstance(row, dict):
            _fail(f"manifest model {idx} must be an object: {manifest_path}")
        missing_row = sorted(EXPECTED_MODEL_ROW_KEYS - set(row))
        if missing_row:
            _fail(f"manifest model {idx} missing field: {missing_row[0]}")
        unexpected_row = sorted(set(row) - EXPECTED_MODEL_ROW_KEYS)
        if unexpected_row:
            _fail(f"manifest model {idx} unexpected field: {unexpected_row[0]}")
        family = row.get("family")
        if family not in observed_counts:
            _fail(f"manifest model {idx} has invalid family: {family}")
        model_path = _resolve_manifest_path(
            row.get("path"),
            repo=repo,
            label=f"model {idx} path",
        )
        model_sha = row.get("sha256")
        if (
            not isinstance(model_sha, str)
            or len(model_sha) != 64
            or any(char not in "0123456789abcdef" for char in model_sha)
        ):
            _fail(f"manifest model {idx} has invalid sha256: {manifest_path}")
        if not model_path.is_relative_to(family_dirs[family]):
            _fail(
                "manifest model "
                f"{idx} path is outside declared {family} directory: {model_path}",
            )
        if not model_path.is_file():
            _fail(f"manifest model {idx} file missing: {model_path}")
        if sha256(model_path) != model_sha:
            _fail(f"manifest model {idx} sha256 mismatch: {model_path}")
        observed_counts[family] += 1
    if observed_counts["xgb"] != xgb_count:
        _fail(f"manifest xgb_model_count does not match models: {manifest_path}")
    if observed_counts["cat"] != cat_count:
        _fail(f"manifest cat_model_count does not match models: {manifest_path}")


def _sweep_arg_values(args: list[str], flag: str) -> list[str]:
    values: list[str] = []
    idx = 0
    while idx < len(args):
        if args[idx] != flag:
            idx += 1
            continue
        if idx + 1 >= len(args) or args[idx + 1].startswith("--"):
            _fail(f"sweep args {flag} missing value")
        values.append(args[idx + 1])
        idx += 2
    return values


def _validate_known_sweep_args(args: list[str]) -> None:
    value_flags = set(SWEEP_ARG_CONFIG_FLAGS) | SPECIAL_VALUE_SWEEP_FLAGS
    idx = 0
    while idx < len(args):
        flag = args[idx]
        if not flag.startswith("--"):
            _fail(f"sweep args unexpected positional token: {flag}")
        if flag in value_flags:
            if idx + 1 >= len(args) or args[idx + 1].startswith("--"):
                _fail(f"sweep args {flag} missing value")
            idx += 2
            continue
        if flag in ALLOWED_VALUELESS_SWEEP_FLAGS:
            idx += 1
            continue
        _fail(f"sweep args unexpected flag: {flag}")


def _validate_sweep_args_contract(sweep_args_path: Path, manifest_path: Path) -> None:
    args = _load_sweep_args(sweep_args_path)
    _validate_known_sweep_args(args)
    manifest = _load_json_object(manifest_path, "manifest")
    _validate_manifest_schema(manifest, manifest_path)
    _validate_manifest_fixed_flags(manifest, manifest_path)
    config = manifest.get("config")
    if not isinstance(config, dict):
        _fail(f"manifest missing config object: {manifest_path}")
    repo = _resolve_manifest_path(
        manifest.get("repo", str(manifest_path.parent)),
        repo=manifest_path.parent,
        label="repo",
    )
    for flag in sorted(REQUIRED_SWEEP_FLAGS):
        if flag not in args:
            _fail(f"sweep args missing required safety flag: {flag}")
    for flag, config_key in sorted(SWEEP_ARG_CONFIG_FLAGS.items()):
        values = _sweep_arg_values(args, flag)
        if len(values) != 1:
            _fail(f"sweep args must include {flag} exactly once")
        expected_value = config.get(config_key)
        if not isinstance(expected_value, str):
            _fail(f"manifest config {config_key} must be a string")
        if values[0] != expected_value:
            _fail(
                f"sweep args {flag} does not match manifest config {config_key}: "
                f"{values[0]} != {expected_value}",
            )
    output_dir_values = _sweep_arg_values(args, "--output-dir")
    if len(output_dir_values) != 1:
        _fail("sweep args must include --output-dir exactly once")
    manifest_output_dir = _resolve_manifest_path(
        manifest.get("output_dir"),
        repo=repo,
        label="output_dir",
    )
    sweep_output_dir = _resolve_manifest_path(
        output_dir_values[0],
        repo=repo,
        label="sweep args output_dir",
    )
    if sweep_output_dir != manifest_output_dir:
        _fail("sweep args --output-dir does not match manifest output_dir")

    symbols_values = _sweep_arg_values(args, "--symbols-file")
    if len(symbols_values) != 1:
        _fail("sweep args must include --symbols-file exactly once")
    manifest_symbols_file = _resolve_manifest_path(
        manifest.get("symbols_file"),
        repo=repo,
        label="symbols_file",
    )
    sweep_symbols_file = _resolve_manifest_path(
        symbols_values[0],
        repo=repo,
        label="sweep args symbols_file",
    )
    if sweep_symbols_file != manifest_symbols_file:
        _fail("sweep args --symbols-file does not match manifest symbols_file")

    model_values = _sweep_arg_values(args, "--model-paths")
    if len(model_values) != 1:
        _fail("sweep args must include --model-paths exactly once")
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        _fail(f"manifest missing models list: {manifest_path}")
    manifest_model_paths: list[Path] = []
    for idx, row in enumerate(models):
        if not isinstance(row, dict):
            _fail(f"manifest model {idx} must be an object: {manifest_path}")
        manifest_model_paths.append(
            _resolve_manifest_path(
                row.get("path"),
                repo=repo,
                label=f"model {idx} path",
            ),
        )
    raw_sweep_model_paths = [value.strip() for value in model_values[0].split(",")]
    if not raw_sweep_model_paths or any(not value for value in raw_sweep_model_paths):
        _fail("sweep args --model-paths must not contain empty paths")
    sweep_model_paths = [
        _resolve_manifest_path(
            value,
            repo=repo,
            label=f"sweep args model_paths[{idx}]",
        )
        for idx, value in enumerate(raw_sweep_model_paths)
    ]
    if sweep_model_paths != manifest_model_paths:
        _fail("sweep args --model-paths does not match manifest models")


def _resolve_manifest_path(value: object, *, repo: Path, label: str) -> Path:
    if not isinstance(value, str):
        _fail(f"manifest {label} must be a path string")
    path = Path(value)
    if not path.is_absolute():
        path = repo / path
    return path.resolve(strict=False)


def _validate_manifest_sidecar_paths(
    manifest_path: Path,
    *,
    marker_dir: Path,
    expected_paths: dict[str, Path],
) -> None:
    manifest = _load_json_object(manifest_path, "manifest")
    repo = _resolve_manifest_path(
        manifest.get("repo", str(marker_dir)),
        repo=marker_dir,
        label="repo",
    )
    expected_manifest_paths = {
        "output_dir": marker_dir,
        "model_paths_file": marker_dir / "model_paths.txt",
        "run_log": expected_paths["run_log"],
        "sweep_args_file": expected_paths["sweep_args"],
        "done_marker": marker_dir / "sweep.done",
        "result_artifacts_file": expected_paths["result_artifacts"],
    }
    for key, expected_path in expected_manifest_paths.items():
        actual_path = _resolve_manifest_path(
            manifest.get(key),
            repo=repo,
            label=key,
        )
        if actual_path != expected_path.resolve(strict=False):
            _fail(f"manifest {key} path drift: {actual_path}")


def _validate_model_paths_file(model_paths_file: Path, manifest_path: Path) -> None:
    manifest = _load_json_object(manifest_path, "manifest")
    repo = _resolve_manifest_path(
        manifest.get("repo"),
        repo=manifest_path.parent,
        label="repo",
    )
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        _fail(f"manifest missing models list: {manifest_path}")
    expected_paths = [
        _resolve_manifest_path(
            row.get("path"),
            repo=repo,
            label=f"model {idx} path",
        )
        for idx, row in enumerate(models)
        if isinstance(row, dict)
    ]
    if len(expected_paths) != len(models):
        _fail(f"manifest models must be objects: {manifest_path}")
    if not model_paths_file.is_file():
        _fail(f"model paths file missing: {model_paths_file}")
    rows = model_paths_file.read_text(encoding="utf-8").splitlines()
    if not rows:
        _fail(f"model paths file is empty: {model_paths_file}")
    if any(not row.strip() for row in rows):
        _fail(f"model paths file contains empty rows: {model_paths_file}")
    actual_paths = [
        _resolve_manifest_path(
            row.strip(),
            repo=repo,
            label=f"model_paths_file[{idx}]",
        )
        for idx, row in enumerate(rows)
    ]
    if actual_paths != expected_paths:
        _fail("model paths file does not match manifest models")


def _result_path(value: object, *, repo: Path, label: str) -> Path:
    if not isinstance(value, str):
        _fail(f"result JSON {label} must be a path string")
    path = Path(value)
    if not path.is_absolute():
        path = repo / path
    return path.resolve(strict=False)


def _strategy_identity(row: dict[str, object], label: str) -> tuple[tuple[str, str], ...]:
    _validate_strategy_params(row, label)
    identity_parts: list[tuple[str, str]] = []
    for field in STRATEGY_PARAM_FIELDS:
        value = row[field]
        if isinstance(value, dict | list):
            _fail(f"{label} has non-scalar strategy field: {field}")
        identity_parts.append(
            (
                field,
                json.dumps(value, sort_keys=True, separators=(",", ":")),
            ),
        )
    return tuple(identity_parts)


def _validate_strategy_params(row: dict[str, object], label: str) -> None:
    missing = [field for field in STRATEGY_PARAM_FIELDS if field not in row]
    if missing:
        _fail(f"{label} missing strategy field: {missing[0]}")
    for field in STRATEGY_PARAM_FIELDS:
        if isinstance(row[field], dict | list):
            _fail(f"{label} has non-scalar strategy field: {field}")


def _validate_sha256_string(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _validate_optional_result_metadata(
    payload: dict[str, object],
    *,
    cells: list[object],
    expected_metadata: dict[str, object],
    path: Path,
) -> None:
    repo = expected_metadata["repo"]
    if not isinstance(repo, Path):
        _fail(f"internal verifier error: invalid repo for {path}")
    if "blend_mode" in payload and payload["blend_mode"] != "mean":
        _fail(f"result JSON blend_mode does not match runner contract: {path}")

    if "data_root" in payload:
        data_root = _result_path(payload.get("data_root"), repo=repo, label="data_root")
        expected_data_root = (repo / "trainingdata").resolve(strict=False)
        if data_root != expected_data_root:
            _fail(f"result JSON data_root does not match runner contract: {path}")

    if "spy_csv_sha256" in payload and "spy_csv" not in payload:
        _fail(f"result JSON spy_csv_sha256 present without spy_csv: {path}")
    if "spy_csv" in payload:
        spy_csv = _result_path(payload.get("spy_csv"), repo=repo, label="spy_csv")
        expected_spy_csv = (repo / "trainingdata" / "SPY.csv").resolve(strict=False)
        if spy_csv != expected_spy_csv:
            _fail(f"result JSON spy_csv does not match runner contract: {path}")
        if "spy_csv_sha256" in payload:
            _expect_sha(
                spy_csv,
                _validate_sha256_string(payload.get("spy_csv_sha256"), "result JSON spy_csv_sha256"),
                "SPY CSV",
            )

    if "fm_latents_sha256" in payload and "fm_latents_path" not in payload:
        _fail(f"result JSON fm_latents_sha256 present without fm_latents_path: {path}")
    result_fm_n_latents = None
    if "fm_n_latents" in payload:
        try:
            fm_n_latents = int(payload["fm_n_latents"])
        except (TypeError, ValueError):
            _fail(f"result JSON fm_n_latents must be an integer: {path}")
        if fm_n_latents <= 0:
            _fail(f"result JSON fm_n_latents must be positive: {path}")
        result_fm_n_latents = fm_n_latents
    result_fm_sha256 = None
    if "fm_latents_path" in payload:
        fm_latents_path = _result_path(
            payload.get("fm_latents_path"),
            repo=repo,
            label="fm_latents_path",
        )
        if "fm_latents_sha256" in payload:
            result_fm_sha256 = _validate_sha256_string(
                payload.get("fm_latents_sha256"),
                "result JSON fm_latents_sha256",
            )
            _expect_sha(
                fm_latents_path,
                result_fm_sha256,
                "FM latents",
            )

    ensemble_feature_mode = payload.get("ensemble_feature_mode")
    if ensemble_feature_mode is not None:
        if not isinstance(ensemble_feature_mode, dict):
            _fail(f"result JSON ensemble_feature_mode must be an object: {path}")
        expected_keys = {"needs_ranks", "needs_dispersion"}
        missing_keys = sorted(expected_keys - set(ensemble_feature_mode))
        if missing_keys:
            _fail(f"result JSON ensemble_feature_mode missing key: {missing_keys[0]}")
        unexpected_keys = sorted(set(ensemble_feature_mode) - expected_keys)
        if unexpected_keys:
            _fail(f"result JSON ensemble_feature_mode unexpected key: {unexpected_keys[0]}")
        derived_flags = {
            "needs_ranks": _derive_raw_cell_bool_flag(cells, "ensemble_needs_ranks", path),
            "needs_dispersion": _derive_raw_cell_bool_flag(
                cells,
                "ensemble_needs_dispersion",
                path,
            ),
        }
        for key, expected_value in sorted(derived_flags.items()):
            if ensemble_feature_mode.get(key) is not expected_value:
                _fail(f"result JSON ensemble_feature_mode {key} does not match raw cells")

    ensemble_manifest = payload.get("ensemble_manifest")
    if ensemble_manifest is not None:
        if not isinstance(ensemble_manifest, dict):
            _fail(f"result JSON ensemble_manifest must be an object: {path}")
        unexpected_keys = sorted(
            set(ensemble_manifest) - ({"path", "sha256"} | ENSEMBLE_MANIFEST_OPTIONAL_KEYS),
        )
        if unexpected_keys:
            _fail(f"result JSON ensemble_manifest unexpected key: {unexpected_keys[0]}")
        manifest_path = _result_path(
            ensemble_manifest.get("path"),
            repo=repo,
            label="ensemble_manifest.path",
        )
        model_paths = expected_metadata.get("model_paths")
        if not isinstance(model_paths, list) or not model_paths:
            _fail(f"internal verifier error: invalid model paths for {path}")
        model_parent_dirs = {Path(str(model_path)).resolve(strict=False).parent for model_path in model_paths}
        if len(model_parent_dirs) != 1:
            _fail(f"result JSON ensemble_manifest is inconsistent with split model dirs: {path}")
        expected_manifest_path = next(iter(model_parent_dirs)) / "alltrain_ensemble.json"
        if manifest_path != expected_manifest_path.resolve(strict=False):
            _fail(f"result JSON ensemble_manifest path does not match model directory: {path}")
        _expect_sha(
            manifest_path,
            _validate_sha256_string(
                ensemble_manifest.get("sha256"),
                "result JSON ensemble_manifest.sha256",
            ),
            "ensemble manifest",
        )
        copied_keys = sorted(ENSEMBLE_MANIFEST_OPTIONAL_KEYS & set(ensemble_manifest))
        manifest_payload = None
        if copied_keys or result_fm_sha256 is not None or result_fm_n_latents is not None:
            manifest_payload = _load_json_object(manifest_path, "ensemble manifest")
        if copied_keys:
            assert manifest_payload is not None
            for key in copied_keys:
                if ensemble_manifest[key] != manifest_payload.get(key):
                    _fail(f"result JSON ensemble_manifest {key} does not match manifest file")
        if manifest_payload is not None:
            manifest_config = manifest_payload.get("config")
            if isinstance(manifest_config, dict):
                manifest_fm_sha256 = manifest_config.get("fm_latents_sha256")
                if manifest_fm_sha256 not in (None, ""):
                    manifest_fm_sha256 = _validate_sha256_string(
                        manifest_fm_sha256,
                        "ensemble manifest config.fm_latents_sha256",
                    )
                    if result_fm_sha256 is None:
                        _fail("result JSON missing FM latents hash from ensemble manifest")
                    if result_fm_sha256 != manifest_fm_sha256:
                        _fail("result JSON fm_latents_sha256 does not match ensemble manifest")
                manifest_fm_n_latents = manifest_config.get("fm_n_latents")
                if manifest_fm_n_latents not in (None, 0):
                    if not _is_int(manifest_fm_n_latents) or manifest_fm_n_latents <= 0:
                        _fail("ensemble manifest config.fm_n_latents must be positive")
                    if result_fm_n_latents is None:
                        _fail("result JSON missing fm_n_latents from ensemble manifest")
                    if result_fm_n_latents != manifest_fm_n_latents:
                        _fail("result JSON fm_n_latents does not match ensemble manifest")


def _derive_raw_cell_bool_flag(cells: list[object], field: str, path: Path) -> bool:
    observed = []
    for idx, cell in enumerate(cells):
        if not isinstance(cell, dict):
            _fail(f"result JSON contains non-object cells: {path}")
        value = cell.get(field, False)
        if not isinstance(value, bool):
            _fail(f"result JSON cell {idx} {field} must be boolean when present")
        observed.append(value)
    return any(observed)


def _expected_result_contract(
    manifest_path: Path,
) -> tuple[set[str], set[float], dict[str, object]]:
    manifest = _load_json_object(manifest_path, "manifest")
    _validate_manifest_schema(manifest, manifest_path)
    _validate_manifest_fixed_flags(manifest, manifest_path)
    config = manifest.get("config")
    if not isinstance(config, dict):
        _fail(f"manifest missing config object: {manifest_path}")
    _validate_manifest_production_contract(config, manifest_path)
    fee_regimes = set(_parse_csv(config.get("fee_regimes"), "manifest config fee_regimes"))
    fill_buffers = set(
        _parse_float_grid(
            config.get("fill_buffer_bps_grid"),
            "manifest config fill_buffer_bps_grid",
        ),
    )
    repo = _resolve_manifest_path(
        manifest.get("repo", str(manifest_path.parent)),
        repo=manifest_path.parent,
        label="repo",
    )
    symbols_file = _resolve_manifest_path(
        manifest.get("symbols_file"),
        repo=repo,
        label="symbols_file",
    )
    symbols_sha256 = manifest.get("symbols_sha256")
    if (
        not isinstance(symbols_sha256, str)
        or len(symbols_sha256) != 64
        or any(char not in "0123456789abcdef" for char in symbols_sha256)
    ):
        _fail(f"manifest has invalid symbols_sha256: {manifest_path}")
    _expect_sha(symbols_file, symbols_sha256, "symbols file")
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        _fail(f"manifest missing models list: {manifest_path}")
    model_paths: list[str] = []
    model_sha256: list[str] = []
    for idx, row in enumerate(models):
        if not isinstance(row, dict):
            _fail(f"manifest model {idx} must be an object: {manifest_path}")
        model_path = _resolve_manifest_path(
            row.get("path"),
            repo=repo,
            label=f"model {idx} path",
        )
        model_paths.append(str(model_path))
        sha_value = row.get("sha256")
        if not isinstance(sha_value, str) or len(sha_value) != 64:
            _fail(f"manifest model {idx} missing sha256: {manifest_path}")
        if model_path.is_file():
            _expect_sha(model_path, sha_value, f"model {idx}")
        model_sha256.append(sha_value)
    metadata: dict[str, object] = {
        "repo": repo,
        "symbols_file": str(symbols_file),
        "model_paths": model_paths,
        "model_sha256": model_sha256,
        "fail_fast_max_dd_pct": _parse_float_config(
            config,
            "fail_fast_max_dd_pct",
            "manifest config fail_fast_max_dd_pct",
        ),
        "fail_fast_max_intraday_dd_pct": _parse_float_config(
            config,
            "fail_fast_max_intraday_dd_pct",
            "manifest config fail_fast_max_intraday_dd_pct",
        ),
        "fail_fast_neg_windows": _parse_int_config(
            config,
            "fail_fast_neg_windows",
            "manifest config fail_fast_neg_windows",
        ),
        "oos_start": config.get("oos_start"),
        "oos_end": config.get("oos_end"),
        "window_days": config.get("window_days"),
        "stride_days": config.get("stride_days"),
    }
    return fee_regimes, fill_buffers, metadata


def _validate_result(
    path: Path,
    *,
    expected_fee_regimes: set[str],
    expected_fill_buffers: set[float],
    expected_metadata: dict[str, object],
) -> None:
    payload = _load_json_object(path, "result")
    unexpected = sorted(set(payload) - EXPECTED_RESULT_ROOT_KEYS)
    if unexpected:
        _fail(f"result JSON unexpected field: {unexpected[0]}")
    if payload.get("complete") is not True:
        _fail(f"result JSON is not complete: {path}")
    repo = expected_metadata["repo"]
    if not isinstance(repo, Path):
        _fail(f"internal verifier error: invalid repo for {path}")
    result_symbols_file = _result_path(
        payload.get("symbols_file"),
        repo=repo,
        label="symbols_file",
    )
    if str(result_symbols_file) != expected_metadata["symbols_file"]:
        _fail(f"result JSON symbols_file does not match manifest: {path}")
    result_model_values = payload.get("model_paths")
    if not isinstance(result_model_values, list) or not result_model_values:
        _fail(f"result JSON missing model_paths: {path}")
    result_model_paths = [
        str(_result_path(value, repo=repo, label=f"model_paths[{idx}]"))
        for idx, value in enumerate(result_model_values)
    ]
    if result_model_paths != expected_metadata["model_paths"]:
        _fail(f"result JSON model_paths do not match manifest: {path}")
    result_model_sha256 = payload.get("model_sha256")
    if (
        not isinstance(result_model_sha256, list)
        or not result_model_sha256
        or any(not isinstance(value, str) for value in result_model_sha256)
    ):
        _fail(f"result JSON missing model_sha256: {path}")
    if result_model_sha256 != expected_metadata["model_sha256"]:
        _fail(f"result JSON model_sha256 does not match manifest: {path}")
    for key in ("oos_start", "oos_end"):
        if payload.get(key) != expected_metadata[key]:
            _fail(f"result JSON {key} does not match manifest: {path}")
    for key in ("window_days", "stride_days"):
        expected_value = expected_metadata[key]
        try:
            expected_int = int(expected_value)  # manifest config values are recorded from argv strings.
        except (TypeError, ValueError):
            _fail(f"manifest config {key} must be an integer")
        if payload.get(key) != expected_int:
            _fail(f"result JSON {key} does not match manifest: {path}")
    fail_fast = payload.get("fail_fast")
    if not isinstance(fail_fast, dict):
        _fail(f"result JSON missing fail-fast contract: {path}")
    fail_fast_numeric = {
        "max_dd_pct": expected_metadata["fail_fast_max_dd_pct"],
        "max_intraday_dd_pct": expected_metadata["fail_fast_max_intraday_dd_pct"],
    }
    for key, expected_value in fail_fast_numeric.items():
        actual_value = fail_fast.get(key)
        if not _is_number(actual_value) or float(actual_value) != expected_value:
            _fail(f"result JSON fail_fast {key} does not match manifest: {path}")
    expected_neg_windows = expected_metadata["fail_fast_neg_windows"]
    actual_neg_windows = fail_fast.get("neg_windows")
    if not _is_int(actual_neg_windows) or actual_neg_windows != expected_neg_windows:
        _fail(f"result JSON fail_fast neg_windows does not match manifest: {path}")
    fail_fast_score = fail_fast.get("score")
    if not _is_number(fail_fast_score) or float(fail_fast_score) != FAIL_FAST_SCORE:
        _fail(f"result JSON fail_fast score does not match verifier contract: {path}")
    fee_regimes = payload.get("fee_regimes")
    if not isinstance(fee_regimes, dict):
        _fail(f"result JSON missing fee_regimes: {path}")
    actual_fee_regimes = {str(key) for key in fee_regimes}
    if actual_fee_regimes != expected_fee_regimes:
        _fail(f"result JSON fee_regimes do not match manifest: {path}")
    for regime in sorted(expected_fee_regimes):
        expected_config = EXPECTED_FEE_REGIME_CONFIG.get(regime)
        if expected_config is None:
            _fail(f"result JSON fee_regime has no verifier contract: {regime}")
        actual_config = fee_regimes.get(regime)
        if not isinstance(actual_config, dict):
            _fail(f"result JSON fee_regime {regime} must be an object: {path}")
        for key, expected_value in sorted(expected_config.items()):
            actual_value = actual_config.get(key)
            if not _is_number(actual_value) or float(actual_value) != expected_value:
                _fail(f"result JSON fee_regime {regime} {key} mismatch: {path}")
    _validate_numeric_mapping(
        payload.get("goodness_weights"),
        EXPECTED_GOODNESS_WEIGHTS,
        label="goodness_weights",
        path=path,
    )
    _validate_numeric_mapping(
        payload.get("robust_goodness_weights"),
        EXPECTED_ROBUST_GOODNESS_WEIGHTS,
        label="robust_goodness_weights",
        path=path,
    )
    _validate_numeric_mapping(
        payload.get("pain_adjusted_goodness_weights"),
        EXPECTED_PAIN_ADJUSTED_GOODNESS_WEIGHTS,
        label="pain_adjusted_goodness_weights",
        path=path,
    )
    production_target = payload.get("production_target")
    if not isinstance(production_target, dict):
        _fail(f"result JSON missing production target contract: {path}")
    numeric_contract = {
        "median_monthly_pct": PRODUCTION_TARGET_MEDIAN_MONTHLY_PCT,
        "max_dd_pct": PRODUCTION_TARGET_MAX_DD_PCT,
    }
    for key, expected in numeric_contract.items():
        actual = production_target.get(key)
        if not _is_number(actual) or float(actual) != expected:
            _fail(f"result JSON production target contract mismatch for {key}: {path}")
    integer_contract = {
        "max_neg_windows": PRODUCTION_TARGET_MAX_NEG_WINDOWS,
        "min_windows": PRODUCTION_TARGET_MIN_WINDOWS,
    }
    for key, expected in integer_contract.items():
        actual = production_target.get(key)
        if not _is_int(actual) or actual != expected:
            _fail(f"result JSON production target contract mismatch for {key}: {path}")
    if production_target.get("expected_windows_required") is not True:
        _fail(
            "result JSON production target contract mismatch for "
            f"expected_windows_required: {path}",
        )
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        _fail(f"result JSON has no cells: {path}")
    if any(not isinstance(row, dict) for row in cells):
        _fail(f"result JSON contains non-object cells: {path}")
    _validate_optional_result_metadata(
        payload,
        cells=cells,
        expected_metadata=expected_metadata,
        path=path,
    )
    observed_pairs: set[tuple[str, float]] = set()
    for row in cells:
        fee_regime = row.get("fee_regime")
        fill_buffer = row.get("fill_buffer_bps")
        if not isinstance(fee_regime, str):
            _fail(f"result JSON cell missing fee_regime: {path}")
        if fee_regime not in expected_fee_regimes:
            _fail(f"result JSON cell has unexpected fee_regime: {fee_regime}")
        if not _is_number(fill_buffer) or not math.isfinite(float(fill_buffer)):
            _fail(f"result JSON cell missing finite fill_buffer_bps: {path}")
        fill_buffer_float = float(fill_buffer)
        if fill_buffer_float not in expected_fill_buffers:
            _fail(f"result JSON cell has unexpected fill_buffer_bps: {fill_buffer_float:g}")
        _validate_strategy_params(row, "result JSON raw cell")
        _validate_raw_cell_scores(
            row,
            "result JSON raw cell",
            oos_start=str(payload.get("oos_start")),
            oos_end=str(payload.get("oos_end")),
        )
        observed_pairs.add((fee_regime, fill_buffer_float))
    for fee_regime in sorted(expected_fee_regimes):
        for fill_buffer in sorted(expected_fill_buffers):
            if (fee_regime, fill_buffer) not in observed_pairs:
                _fail(
                    "result JSON missing required production-realism cell: "
                    f"fee_regime={fee_regime} fill_buffer_bps={fill_buffer:g}",
                )
    n_cells = payload.get("n_cells")
    if not _is_int(n_cells) or n_cells != len(cells):
        _fail(f"result JSON n_cells does not match cells: {path}")
    robust_strategies = payload.get("friction_robust_strategies")
    if not isinstance(robust_strategies, list) or not robust_strategies:
        _fail(f"result JSON has no friction-robust strategies: {path}")
    if any(not isinstance(row, dict) for row in robust_strategies):
        _fail(f"result JSON contains non-object friction-robust strategies: {path}")
    n_robust = payload.get("n_friction_robust_strategies")
    if not _is_int(n_robust) or n_robust != len(robust_strategies):
        _fail(f"result JSON robust-strategy count mismatch: {path}")
    production_passes: list[dict[str, object]] = []
    best_production_rank = (float("-inf"), float("-inf"))
    seen_strategy_identities: set[tuple[tuple[str, str], ...]] = set()
    for idx, row in enumerate(robust_strategies):
        label = f"result JSON friction-robust strategy {idx}"
        strategy_identity = _strategy_identity(row, label)
        if strategy_identity in seen_strategy_identities:
            _fail(f"{label} duplicates an earlier robust strategy")
        seen_strategy_identities.add(strategy_identity)
        pass_flag = row.get("production_target_pass")
        if not isinstance(pass_flag, bool):
            _fail(f"{label} missing boolean production_target_pass")
        worst_goodness = _number_value(row, "worst_goodness_score", label)
        worst_robust_goodness = _number_value(row, "worst_robust_goodness_score", label)
        ranking_score = _number_value(row, "worst_pain_adjusted_goodness_score", label)
        worst_median = _number_value(row, "worst_median_monthly_pct", label)
        max_dd = _number_value(row, "max_worst_dd_pct", label)
        max_n_neg = _int_value(row, "max_n_neg", label)
        min_n_windows = _int_value(row, "min_n_windows", label)
        max_expected_n_windows = _int_value(row, "max_expected_n_windows", label)
        required_min_n_windows = _int_value(row, "required_min_n_windows", label)
        expected_min_friction_cells = len(expected_fee_regimes) * len(expected_fill_buffers)
        n_friction_cells = _int_value(row, "n_friction_cells", label)
        if n_friction_cells != expected_min_friction_cells:
            _fail(f"{label} friction stress cell count does not match manifest grid")
        matching_cells = [
            cell
            for cell in cells
            if all(
                field not in row
                or (
                    isinstance(cell, dict)
                    and field in cell
                    and cell[field] == row[field]
                )
                for field in STRATEGY_PARAM_FIELDS
            )
        ]
        if len(matching_cells) != n_friction_cells:
            _fail(f"{label} raw cell count does not match robust summary")
        fee_values = row.get("fee_regimes")
        if not isinstance(fee_values, list) or any(
            not isinstance(value, str) for value in fee_values
        ):
            _fail(f"{label} missing fee_regimes stress coverage")
        if set(fee_values) != expected_fee_regimes:
            _fail(f"{label} fee_regimes do not cover manifest stress grid")
        fill_values = row.get("fill_buffer_bps_values")
        if not isinstance(fill_values, list) or not fill_values:
            _fail(f"{label} missing fill_buffer_bps_values stress coverage")
        fill_value_set: set[float] = set()
        for value in fill_values:
            if not _is_number(value) or not math.isfinite(float(value)):
                _fail(f"{label} has non-finite fill_buffer_bps_values entry")
            fill_value_set.add(float(value))
        if fill_value_set != expected_fill_buffers:
            _fail(f"{label} fill_buffer_bps_values do not cover manifest stress grid")
        skip_prob_values = row.get("skip_prob_values")
        if not isinstance(skip_prob_values, list) or not skip_prob_values:
            _fail(f"{label} missing skip_prob_values stress coverage")
        skip_prob_set: set[float] = set()
        for value in skip_prob_values:
            if not _is_number(value) or not math.isfinite(float(value)):
                _fail(f"{label} has non-finite skip_prob_values entry")
            skip_prob_set.add(float(value))
        if skip_prob_set != {0.0}:
            _fail(f"{label} skip_prob_values do not match no-skip runner contract")
        skip_seed_values = row.get("skip_seed_values")
        if not isinstance(skip_seed_values, list) or not skip_seed_values:
            _fail(f"{label} missing skip_seed_values stress coverage")
        skip_seed_set: set[int] = set()
        for value in skip_seed_values:
            if not _is_int(value):
                _fail(f"{label} has non-integer skip_seed_values entry")
            skip_seed_set.add(int(value))
        if skip_seed_set != {0}:
            _fail(f"{label} skip_seed_values do not match no-skip runner contract")
        worst_fee_regime = row.get("worst_fee_regime_by_pain")
        if not isinstance(worst_fee_regime, str) or worst_fee_regime not in expected_fee_regimes:
            _fail(f"{label} worst_fee_regime_by_pain is outside manifest stress grid")
        worst_fill_buffer = row.get("worst_fill_buffer_bps_by_pain")
        if (
            not _is_number(worst_fill_buffer)
            or not math.isfinite(float(worst_fill_buffer))
            or float(worst_fill_buffer) not in expected_fill_buffers
        ):
            _fail(f"{label} worst_fill_buffer_bps_by_pain is outside manifest stress grid")
        fail_fast = row.get("any_fail_fast_triggered")
        if not isinstance(fail_fast, bool):
            _fail(f"{label} missing boolean any_fail_fast_triggered")
        if any(not isinstance(cell, dict) for cell in matching_cells):
            _fail(f"{label} contains non-object matching raw cells")
        derived_worst_median = min(
            _number_value(cell, "median_monthly_pct", f"{label} raw cell")
            for cell in matching_cells
        )
        derived_worst_goodness = min(
            _number_value(cell, "goodness_score", f"{label} raw cell")
            for cell in matching_cells
        )
        derived_worst_robust_goodness = min(
            _number_value(cell, "robust_goodness_score", f"{label} raw cell")
            for cell in matching_cells
        )
        derived_ranking_score = min(
            _number_value(cell, "pain_adjusted_goodness_score", f"{label} raw cell")
            for cell in matching_cells
        )
        derived_max_dd = max(
            _number_value(cell, "worst_dd_pct", f"{label} raw cell")
            for cell in matching_cells
        )
        derived_max_n_neg = max(
            _int_value(cell, "n_neg", f"{label} raw cell")
            for cell in matching_cells
        )
        derived_min_n_windows = min(
            _int_value(cell, "n_windows", f"{label} raw cell")
            for cell in matching_cells
        )
        derived_max_expected_n_windows = max(
            _int_value(cell, "expected_n_windows", f"{label} raw cell")
            for cell in matching_cells
        )
        derived_required_min_windows = max(
            PRODUCTION_TARGET_MIN_WINDOWS,
            derived_max_expected_n_windows,
        )
        raw_fail_fast_flags = []
        for cell in matching_cells:
            raw_fail_fast = cell.get("fail_fast_triggered")
            if not isinstance(raw_fail_fast, bool):
                _fail(f"{label} raw cell missing boolean fail_fast_triggered")
            raw_fail_fast_flags.append(raw_fail_fast)
        derived_any_fail_fast = any(raw_fail_fast_flags)
        if worst_median != derived_worst_median:
            _fail(f"{label} worst_median_monthly_pct does not match raw cells")
        if worst_goodness != derived_worst_goodness:
            _fail(f"{label} worst_goodness_score does not match raw cells")
        if worst_robust_goodness != derived_worst_robust_goodness:
            _fail(f"{label} worst_robust_goodness_score does not match raw cells")
        if ranking_score != derived_ranking_score:
            _fail(f"{label} worst_pain_adjusted_goodness_score does not match raw cells")
        if max_dd != derived_max_dd:
            _fail(f"{label} max_worst_dd_pct does not match raw cells")
        if max_n_neg != derived_max_n_neg:
            _fail(f"{label} max_n_neg does not match raw cells")
        if min_n_windows != derived_min_n_windows:
            _fail(f"{label} min_n_windows does not match raw cells")
        if max_expected_n_windows != derived_max_expected_n_windows:
            _fail(f"{label} max_expected_n_windows does not match raw cells")
        if required_min_n_windows != derived_required_min_windows:
            _fail(f"{label} required_min_n_windows does not match raw cells")
        if fail_fast != derived_any_fail_fast:
            _fail(f"{label} any_fail_fast_triggered does not match raw cells")
        for metric in ROBUST_SUMMARY_HIGH_GOOD_METRICS:
            _validate_robust_summary_metric(
                row,
                matching_cells,
                source_metric=metric,
                summary_key=f"worst_{metric}",
                reducer="min",
                label=label,
            )
        for metric in ROBUST_SUMMARY_LOW_GOOD_METRICS:
            _validate_robust_summary_metric(
                row,
                matching_cells,
                source_metric=metric,
                summary_key=f"max_{metric}",
                reducer="max",
                label=label,
            )
        derived_pass = (
            not fail_fast
            and worst_median >= PRODUCTION_TARGET_MEDIAN_MONTHLY_PCT
            and max_dd <= PRODUCTION_TARGET_MAX_DD_PCT
            and max_n_neg <= PRODUCTION_TARGET_MAX_NEG_WINDOWS
            and min_n_windows >= required_min_n_windows
            and required_min_n_windows >= PRODUCTION_TARGET_MIN_WINDOWS
        )
        if pass_flag != derived_pass:
            _fail(f"{label} production_target_pass does not match metrics")
        if pass_flag:
            production_passes.append(row)
            best_production_rank = max(best_production_rank, (ranking_score, worst_median))
    pass_count = payload.get("n_production_target_pass")
    if not _is_int(pass_count) or pass_count != len(production_passes):
        _fail(f"result JSON production-target pass count mismatch: {path}")
    if pass_count < 1:
        _fail(f"result JSON has no production-target pass: {path}")
    best_strategy = payload.get("best_production_target_strategy")
    if not isinstance(best_strategy, dict):
        _fail(f"result JSON missing best production-target strategy: {path}")
    if best_strategy not in production_passes:
        _fail(f"result JSON best production-target strategy is not a production pass: {path}")
    best_strategy_goodness = _number_value(
        best_strategy,
        "worst_pain_adjusted_goodness_score",
        "result JSON best production-target strategy",
    )
    best_strategy_median = _number_value(
        best_strategy,
        "worst_median_monthly_pct",
        "result JSON best production-target strategy",
    )
    if (best_strategy_goodness, best_strategy_median) != best_production_rank:
        _fail(
            "result JSON best production-target strategy is not highest "
            f"producer-ranked pass: {path}",
        )


def _current_run_results(output_dir: Path, run_started_ns: int) -> list[Path]:
    return sorted(
        path.resolve()
        for path in output_dir.glob("sweep_*.json")
        if ".partial." not in path.name
        and path.stat().st_mtime_ns >= run_started_ns
    )


def _expect_sha(path: Path, expected_sha: str, label: str) -> None:
    if not path.is_file():
        _fail(f"{label} missing: {path}")
    actual_sha = sha256(path)
    if actual_sha != expected_sha:
        _fail(f"{label} sha256 mismatch: {path}")


def _load_results_manifest(path: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"invalid result artifacts JSON {path}: {exc}")
    if not isinstance(payload, dict):
        _fail(f"result artifacts JSON must be an object: {path}")
    unexpected = sorted(set(payload) - {"results"})
    if unexpected:
        _fail(f"result artifacts JSON unexpected key: {unexpected[0]}")
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        _fail(f"result artifacts JSON has no results: {path}")
    return results


def verify_completion(done_marker: Path) -> None:
    marker_dir = done_marker.resolve().parent
    marker = _read_key_values(done_marker)
    required_keys = {
        "schema_version",
        "completed_at",
        "manifest",
        "manifest_sha256",
        "model_paths",
        "model_paths_sha256",
        "sweep_args",
        "sweep_args_sha256",
        "run_log",
        "run_log_sha256",
        "result_artifacts",
        "result_artifacts_sha256",
        "result_count",
    }
    missing = sorted(required_keys - marker.keys())
    if missing:
        _fail(f"completion marker missing key: {missing[0]}")
    unexpected = sorted(marker.keys() - required_keys)
    if unexpected:
        _fail(f"completion marker unexpected key: {unexpected[0]}")
    if marker["schema_version"] != str(COMPLETION_SCHEMA_VERSION):
        _fail("completion marker schema_version mismatch")
    _validate_completed_at(marker["completed_at"])

    expected_paths = {
        "manifest": marker_dir / "run_manifest.json",
        "model_paths": marker_dir / "model_paths.txt",
        "sweep_args": marker_dir / "sweep_args.txt",
        "run_log": marker_dir / "sweep_stdout.log",
        "result_artifacts": marker_dir / "result_artifacts.json",
    }
    for key, expected_path in expected_paths.items():
        if Path(marker[key]).resolve() != expected_path:
            _fail(f"completion marker {key} path drift: {marker[key]}")

    manifest_path = Path(marker["manifest"])
    _expect_sha(manifest_path, marker["manifest_sha256"], "manifest")
    _validate_manifest_sidecar_paths(
        manifest_path,
        marker_dir=marker_dir,
        expected_paths=expected_paths,
    )
    model_paths_file = Path(marker["model_paths"])
    _expect_sha(model_paths_file, marker["model_paths_sha256"], "model paths")
    _validate_model_paths_file(model_paths_file, manifest_path)
    _expect_sha(Path(marker["sweep_args"]), marker["sweep_args_sha256"], "sweep args")
    _expect_sha(Path(marker["run_log"]), marker["run_log_sha256"], "run log")
    results_manifest = Path(marker["result_artifacts"])
    _expect_sha(
        results_manifest,
        marker["result_artifacts_sha256"],
        "result artifacts",
    )

    results = _load_results_manifest(results_manifest)
    try:
        expected_count = int(marker["result_count"])
    except ValueError:
        _fail("completion marker result_count is not an integer")
    if expected_count != len(results):
        _fail("completion marker result_count does not match result artifacts")
    sweep_args_path = Path(marker["sweep_args"])
    _validate_sweep_args_contract(sweep_args_path, manifest_path)
    expected_fee_regimes, expected_fill_buffers, expected_metadata = _expected_result_contract(manifest_path)
    seen_result_paths: set[Path] = set()
    expected_result_row_keys = {"path", "sha256", "size_bytes"}
    for idx, row in enumerate(results):
        if not isinstance(row, dict):
            _fail(f"result artifacts row {idx} must be an object")
        unexpected_row_keys = sorted(set(row) - expected_result_row_keys)
        if unexpected_row_keys:
            _fail(f"result artifacts row {idx} unexpected key: {unexpected_row_keys[0]}")
        path_value = row.get("path")
        sha_value = row.get("sha256")
        size_value = row.get("size_bytes")
        if not isinstance(path_value, str) or not isinstance(sha_value, str):
            _fail(f"result artifacts row {idx} missing path or sha256")
        path = Path(path_value)
        resolved_path = path.resolve()
        if (
            resolved_path.parent != marker_dir
            or not resolved_path.name.startswith("sweep_")
            or resolved_path.suffix != ".json"
            or ".partial." in resolved_path.name
        ):
            _fail(f"result artifact {idx} path drift: {path}")
        if resolved_path in seen_result_paths:
            _fail(f"result artifact {idx} duplicates earlier result: {path}")
        seen_result_paths.add(resolved_path)
        _expect_sha(path, sha_value, f"result artifact {idx}")
        if not _is_int(size_value) or path.stat().st_size != size_value:
            _fail(f"result artifact {idx} size mismatch: {path}")
        _validate_result(
            path,
            expected_fee_regimes=expected_fee_regimes,
            expected_fill_buffers=expected_fill_buffers,
            expected_metadata=expected_metadata,
        )


def publish_completion(
    *,
    done_marker: Path,
    manifest: Path,
    sweep_args: Path,
    run_log: Path,
    results_manifest: Path,
    output_dir: Path,
    run_started_ns: int,
) -> None:
    with _completion_publish_lock(done_marker):
        if done_marker.exists():
            _fail(f"completion marker already exists: {done_marker}")
        _validate_sweep_args_contract(sweep_args, manifest)
        resolved_model_paths_file = manifest.resolve().parent / "model_paths.txt"
        _validate_model_paths_file(resolved_model_paths_file, manifest)
        result_paths = _current_run_results(output_dir.resolve(), run_started_ns)
        if not result_paths:
            _fail("sweep completed without a final sweep_*.json result")
        expected_fee_regimes, expected_fill_buffers, expected_metadata = _expected_result_contract(manifest)
        for result_path in result_paths:
            _validate_result(
                result_path,
                expected_fee_regimes=expected_fee_regimes,
                expected_fill_buffers=expected_fill_buffers,
                expected_metadata=expected_metadata,
            )
        result_rows = [
            {
                "path": str(path),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in result_paths
        ]
        resolved_results_manifest = results_manifest.resolve()
        previous_results_manifest = (
            resolved_results_manifest.read_bytes()
            if resolved_results_manifest.exists()
            else None
        )
        _write_text_atomic(
            resolved_results_manifest,
            json.dumps({"results": result_rows}, indent=2, sort_keys=True) + "\n",
        )
        lines = [
            f"schema_version={COMPLETION_SCHEMA_VERSION}",
            f"completed_at={datetime.now(UTC).isoformat().replace('+00:00', 'Z')}",
            f"manifest={manifest.resolve()}",
            f"manifest_sha256={sha256(manifest)}",
            f"model_paths={resolved_model_paths_file}",
            f"model_paths_sha256={sha256(resolved_model_paths_file)}",
            f"sweep_args={sweep_args.resolve()}",
            f"sweep_args_sha256={sha256(sweep_args)}",
            f"run_log={run_log.resolve()}",
            f"run_log_sha256={sha256(run_log)}",
            f"result_artifacts={resolved_results_manifest}",
            f"result_artifacts_sha256={sha256(resolved_results_manifest)}",
            f"result_count={len(result_rows)}",
        ]
        _write_text_atomic(done_marker, "\n".join(lines) + "\n")
        try:
            verify_completion(done_marker)
        except SystemExit as exc:
            if exc.code == 2:
                failed_marker = _quarantine_completion_marker(done_marker)
                _restore_sidecar(resolved_results_manifest, previous_results_manifest)
                _fail(
                    "published completion evidence failed verification; "
                    f"quarantined marker={failed_marker}",
                )
            raise


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify an existing completion marker")
    parser.add_argument("--done-marker", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--sweep-args", type=Path)
    parser.add_argument("--run-log", type=Path)
    parser.add_argument("--results-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-started-ns", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.verify:
        verify_completion(args.done_marker)
        return 0
    required = {
        "--manifest": args.manifest,
        "--sweep-args": args.sweep_args,
        "--run-log": args.run_log,
        "--results-manifest": args.results_manifest,
        "--output-dir": args.output_dir,
        "--run-started-ns": args.run_started_ns,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        _fail(f"missing required argument for publish mode: {missing[0]}")
    publish_completion(
        done_marker=args.done_marker,
        manifest=args.manifest,
        sweep_args=args.sweep_args,
        run_log=args.run_log,
        results_manifest=args.results_manifest,
        output_dir=args.output_dir,
        run_started_ns=args.run_started_ns,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
