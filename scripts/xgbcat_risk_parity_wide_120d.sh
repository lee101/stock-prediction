#!/bin/bash
# Research sweep: XGB+Cat wide-universe risk controls.
#
# Goal: test the strongest current XGB+Cat family against a wider symbol
# universe while adding research-motivated risk controls:
#   - more concurrent names (top_n 2/3/4) to use more independent bets
#   - SPY volatility targeting for market-wide risk
#   - per-pick inverse-vol sizing for single-name crash sensitivity
#   - cross-sectional skew regime gates from the prior 120d audit
#   - fail-fast drawdown / negative-window gates so bad cells stop early
#
# This script is intentionally research-only. It does not edit production
# launch files and does not call Alpaca.
set -euo pipefail

die() {
  echo "[xgbcat-risk] ERROR: $*" >&2
  exit 2
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO="${REPO:-/nvme0n1-disk/code/stock-prediction}"
cd "$REPO" || die "repo not found: $REPO"

VENV="${VENV:-.venv313}"
if [ ! -f "$VENV/bin/activate" ]; then
  die "missing virtualenv activation script: $VENV/bin/activate"
fi
source "$VENV/bin/activate"

OUT=${1:-analysis/xgbnew_daily/xgbcat_risk_parity_wide_120d}
MANIFEST="$OUT/run_manifest.json"
MODEL_LIST="$OUT/model_paths.txt"
RUN_LOG="$OUT/sweep_stdout.log"
SWEEP_ARGS_FILE="$OUT/sweep_args.txt"
DONE_MARKER="$OUT/sweep.done"
RESULTS_MANIFEST="$OUT/result_artifacts.json"

if [ "${VERIFY_ONLY:-0}" = "1" ]; then
  python "$SCRIPT_DIR/write_xgbcat_risk_parity_completion.py" \
    --verify \
    --done-marker "$DONE_MARKER"
  echo "[xgbcat-risk] verified; done_marker=$DONE_MARKER"
  exit 0
fi

SYMBOLS=${SYMBOLS:-symbol_lists/stocks_wide_fresh0401_photonics_2500_v1.txt}
if [ ! -f "$SYMBOLS" ]; then
  SYMBOLS=symbol_lists/stocks_wide_1000_v1.txt
fi
if [ ! -f "$SYMBOLS" ]; then
  die "missing symbols file: $SYMBOLS"
fi

mkdir -p "$OUT" logs/track1
LOCK_FILE="$OUT/.xgbcat_risk_parity.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  die "another xgbcat risk-parity sweep is already running for OUT: $OUT"
fi

XGB_DIR=${XGB_DIR:-analysis/xgbnew_daily/track1_oos120d_xgb}
CAT_DIR=${CAT_DIR:-analysis/xgbnew_daily/track1_oos120d_cat}
if [ ! -d "$XGB_DIR" ]; then
  die "missing XGB model dir: $XGB_DIR"
fi
if [ ! -d "$CAT_DIR" ]; then
  die "missing CatBoost model dir: $CAT_DIR"
fi

shopt -s nullglob
XGB_MODELS=("$XGB_DIR"/alltrain_seed*.pkl)
CAT_MODELS=("$CAT_DIR"/alltrain_seed*.pkl)
shopt -u nullglob
if [ "${#XGB_MODELS[@]}" -eq 0 ]; then
  die "no XGB alltrain_seed*.pkl models found in $XGB_DIR"
fi
if [ "${#CAT_MODELS[@]}" -eq 0 ]; then
  die "no CatBoost alltrain_seed*.pkl models found in $CAT_DIR"
fi

MODEL_PATHS=("${XGB_MODELS[@]}" "${CAT_MODELS[@]}")
declare -A SEEN_MODELS=()
for model_path in "${MODEL_PATHS[@]}"; do
  canonical_model_path=$(realpath "$model_path")
  if [ -n "${SEEN_MODELS[$canonical_model_path]:-}" ]; then
    die "duplicate model artifact: $model_path"
  fi
  SEEN_MODELS["$canonical_model_path"]=1
done
PATHS=$(IFS=,; echo "${MODEL_PATHS[*]}")
REPLACING_COMPLETED=0
if [ -e "$DONE_MARKER" ] && [ "${ALLOW_OVERWRITE:-0}" != "1" ]; then
  die "completed sweep already exists: $DONE_MARKER (set ALLOW_OVERWRITE=1 to replace)"
fi
if [ -e "$DONE_MARKER" ]; then
  if [ "${DRY_RUN:-0}" = "1" ]; then
    die "completed sweep already exists: $DONE_MARKER (dry-run would overwrite evidence; choose a new OUT)"
  fi
  REPLACING_COMPLETED=1
fi
SWEEP_ARGS=(
  --symbols-file "$SYMBOLS"
  --model-paths "$PATHS"
  --oos-start "${OOS_START:-2025-12-18}"
  --oos-end "${OOS_END:-2026-04-20}"
  --window-days "${WINDOW_DAYS:-30}"
  --stride-days "${STRIDE_DAYS:-3}"
  --leverage-grid "${LEVERAGE_GRID:-1.5,2.0,2.25,2.5}"
  --min-score-grid "${MIN_SCORE_GRID:-0.58,0.62,0.66,0.70}"
  --top-n-grid "${TOP_N_GRID:-2,3,4}"
  --hold-through
  --min-dollar-vol "${MIN_DOLLAR_VOL:-50000000}"
  --inference-min-dolvol-grid "${INFERENCE_MIN_DOLVOL_GRID:-50000000,100000000}"
  --inference-min-vol-grid "${INFERENCE_MIN_VOL_GRID:-0.12,0.15}"
  --inference-max-vol-grid "${INFERENCE_MAX_VOL_GRID:-0.0,0.80}"
  --inference-max-spread-bps-grid "${INFERENCE_MAX_SPREAD_BPS_GRID:-20,30}"
  --vol-target-ann-grid "${VOL_TARGET_ANN_GRID:-0.0,0.18,0.22}"
  --inv-vol-target-grid "${INV_VOL_TARGET_GRID:-0.0,0.20,0.25,0.30}"
  --inv-vol-floor "${INV_VOL_FLOOR:-0.08}"
  --inv-vol-cap "${INV_VOL_CAP:-2.0}"
  --regime-cs-skew-min-grid "${REGIME_CS_SKEW_MIN_GRID:-0.50,0.75,1.00}"
  --allocation-mode-grid "${ALLOCATION_MODE_GRID:-equal,score_norm,softmax}"
  --allocation-temp-grid "${ALLOCATION_TEMP_GRID:-0.5,1.0}"
  --score-uncertainty-penalty-grid "${SCORE_UNCERTAINTY_PENALTY_GRID:-0.0,0.5,1.0}"
  --fill-buffer-bps-grid "${FILL_BUFFER_BPS_GRID:-0,5,10,20}"
  --fee-regimes "${FEE_REGIMES:-deploy,prod10bps,stress36x}"
  --fail-fast-max-dd-pct "${FAIL_FAST_MAX_DD_PCT:-20}"
  --fail-fast-max-intraday-dd-pct "${FAIL_FAST_MAX_INTRADAY_DD_PCT:-20}"
  --fail-fast-neg-windows "${FAIL_FAST_NEG_WINDOWS:-1}"
  --checkpoint-every-cells "${CHECKPOINT_EVERY_CELLS:-50}"
  --require-production-target
  --fast-features
  --output-dir "$OUT"
  --verbose
)

export XGBCAT_REPO="$REPO"
export XGBCAT_VENV="$VENV"
export XGBCAT_SYMBOLS="$SYMBOLS"
export XGBCAT_OUT="$OUT"
export XGBCAT_XGB_DIR="$XGB_DIR"
export XGBCAT_CAT_DIR="$CAT_DIR"
export XGBCAT_XGB_COUNT="${#XGB_MODELS[@]}"
export XGBCAT_MODEL_LIST="$MODEL_LIST"
export XGBCAT_RUN_LOG="$RUN_LOG"
export XGBCAT_SWEEP_ARGS_FILE="$SWEEP_ARGS_FILE"
export XGBCAT_DONE_MARKER="$DONE_MARKER"
export XGBCAT_RESULTS_MANIFEST="$RESULTS_MANIFEST"
export XGBCAT_REPLACING_COMPLETED="$REPLACING_COMPLETED"
python - "$MANIFEST" "${MODEL_PATHS[@]}" <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def die(message: str) -> None:
    print(f"[xgbcat-risk] ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_float_grid(config: dict[str, str], key: str) -> list[float]:
    value = config[key]
    tokens = [item.strip() for item in value.split(",")]
    if not tokens or any(not item for item in tokens):
        die(f"{key} must be a finite comma-separated numeric grid: {value}")
    try:
        parsed = [float(item) for item in tokens]
    except ValueError:
        die(f"{key} must be a comma-separated numeric grid: {value}")
    if not parsed or any(not math.isfinite(item) for item in parsed):
        die(f"{key} must be a finite comma-separated numeric grid: {value}")
    return parsed


def parse_csv_list(config: dict[str, str], key: str) -> list[str]:
    value = config[key]
    parsed = [item.strip() for item in value.split(",")]
    if not parsed or any(not item for item in parsed):
        die(f"{key} must not be empty")
    return parsed


def parse_float(config: dict[str, str], key: str) -> float:
    value = config[key]
    try:
        parsed = float(value)
    except ValueError:
        die(f"{key} must be numeric: {value}")
    if not math.isfinite(parsed):
        die(f"{key} must be finite: {value}")
    return parsed


def parse_int(config: dict[str, str], key: str) -> int:
    value = config[key]
    try:
        return int(value)
    except ValueError:
        die(f"{key} must be an integer: {value}")


def validate_production_contract(config: dict[str, str]) -> None:
    fee_regimes = parse_csv_list(config, "fee_regimes")
    allowed_fee_regimes = {"deploy", "prod10bps", "stress36x"}
    unknown_fee_regimes = [value for value in fee_regimes if value not in allowed_fee_regimes]
    if unknown_fee_regimes:
        die(f"fee_regimes contains unknown regime: {unknown_fee_regimes[0]}")
    required_fee_regimes = ("deploy", "prod10bps", "stress36x")
    missing_fee_regimes = [
        required
        for required in required_fee_regimes
        if required not in fee_regimes
    ]
    if missing_fee_regimes:
        die(f"fee_regimes missing required regime: {missing_fee_regimes[0]}")

    leverage_grid = parse_float_grid(config, "leverage_grid")
    bad_leverage = [value for value in leverage_grid if value <= 0.0 or value > 2.5]
    if bad_leverage:
        die(f"leverage_grid exceeds production limit: {bad_leverage[0]:g} > 2.5")

    fill_buffer_grid = parse_float_grid(config, "fill_buffer_bps_grid")
    bad_fill_buffer = [value for value in fill_buffer_grid if value < 0.0]
    if bad_fill_buffer:
        die(f"fill_buffer_bps_grid must be non-negative: {bad_fill_buffer[0]:g}")
    required_fill_buffers = (0.0, 5.0, 10.0, 20.0)
    missing_fill_buffers = [
        required
        for required in required_fill_buffers
        if required not in fill_buffer_grid
    ]
    if missing_fill_buffers:
        die(f"fill_buffer_bps_grid missing required cell: {missing_fill_buffers[0]:g} bps")

    max_dd = parse_float(config, "fail_fast_max_dd_pct")
    if max_dd <= 0.0 or max_dd > 20.0:
        die(f"fail_fast_max_dd_pct exceeds production limit: {max_dd:g} > 20")
    max_intraday_dd = parse_float(config, "fail_fast_max_intraday_dd_pct")
    if max_intraday_dd <= 0.0 or max_intraday_dd > 20.0:
        die(f"fail_fast_max_intraday_dd_pct exceeds production limit: {max_intraday_dd:g} > 20")
    max_neg_windows = parse_int(config, "fail_fast_neg_windows")
    if max_neg_windows < 0 or max_neg_windows > 1:
        die(f"fail_fast_neg_windows exceeds production limit: {max_neg_windows} > 1")


manifest_path = Path(sys.argv[1])
model_list = Path(os.environ["XGBCAT_MODEL_LIST"])
model_paths = [Path(arg) for arg in sys.argv[2:]]
xgb_count = int(os.environ["XGBCAT_XGB_COUNT"])
config_defaults = {
    "OOS_START": "2025-12-18",
    "OOS_END": "2026-04-20",
    "WINDOW_DAYS": "30",
    "STRIDE_DAYS": "3",
    "LEVERAGE_GRID": "1.5,2.0,2.25,2.5",
    "MIN_SCORE_GRID": "0.58,0.62,0.66,0.70",
    "TOP_N_GRID": "2,3,4",
    "MIN_DOLLAR_VOL": "50000000",
    "INFERENCE_MIN_DOLVOL_GRID": "50000000,100000000",
    "INFERENCE_MIN_VOL_GRID": "0.12,0.15",
    "INFERENCE_MAX_VOL_GRID": "0.0,0.80",
    "INFERENCE_MAX_SPREAD_BPS_GRID": "20,30",
    "VOL_TARGET_ANN_GRID": "0.0,0.18,0.22",
    "INV_VOL_TARGET_GRID": "0.0,0.20,0.25,0.30",
    "INV_VOL_FLOOR": "0.08",
    "INV_VOL_CAP": "2.0",
    "REGIME_CS_SKEW_MIN_GRID": "0.50,0.75,1.00",
    "ALLOCATION_MODE_GRID": "equal,score_norm,softmax",
    "ALLOCATION_TEMP_GRID": "0.5,1.0",
    "SCORE_UNCERTAINTY_PENALTY_GRID": "0.0,0.5,1.0",
    "FILL_BUFFER_BPS_GRID": "0,5,10,20",
    "FEE_REGIMES": "deploy,prod10bps,stress36x",
    "FAIL_FAST_MAX_DD_PCT": "20",
    "FAIL_FAST_MAX_INTRADAY_DD_PCT": "20",
    "FAIL_FAST_NEG_WINDOWS": "1",
    "CHECKPOINT_EVERY_CELLS": "50",
}
config = {
    key.lower(): os.environ.get(key, default)
    for key, default in config_defaults.items()
}
validate_production_contract(config)
if os.environ.get("XGBCAT_REPLACING_COMPLETED") == "1":
    Path(os.environ["XGBCAT_DONE_MARKER"]).unlink(missing_ok=True)
    Path(os.environ["XGBCAT_RESULTS_MANIFEST"]).unlink(missing_ok=True)
manifest = {
    "script": "scripts/xgbcat_risk_parity_wide_120d.sh",
    "repo": str(Path(os.environ["XGBCAT_REPO"]).resolve()),
    "venv": os.environ["XGBCAT_VENV"],
    "symbols_file": str(Path(os.environ["XGBCAT_SYMBOLS"]).resolve()),
    "symbols_sha256": sha256(Path(os.environ["XGBCAT_SYMBOLS"])),
    "output_dir": str(Path(os.environ["XGBCAT_OUT"]).resolve()),
    "xgb_dir": str(Path(os.environ["XGBCAT_XGB_DIR"]).resolve()),
    "cat_dir": str(Path(os.environ["XGBCAT_CAT_DIR"]).resolve()),
    "xgb_model_count": xgb_count,
    "cat_model_count": len(model_paths) - xgb_count,
    "model_paths_file": str(model_list.resolve()),
    "run_log": str(Path(os.environ["XGBCAT_RUN_LOG"]).resolve()),
    "sweep_args_file": str(Path(os.environ["XGBCAT_SWEEP_ARGS_FILE"]).resolve()),
    "done_marker": str(Path(os.environ["XGBCAT_DONE_MARKER"]).resolve()),
    "result_artifacts_file": str(Path(os.environ["XGBCAT_RESULTS_MANIFEST"]).resolve()),
    "models": [
        {
            "family": "xgb" if idx < xgb_count else "cat",
            "path": str(path.resolve()),
            "sha256": sha256(path),
        }
        for idx, path in enumerate(model_paths)
    ],
    "config": config,
    "fixed_flags": {
        "hold_through": True,
        "require_production_target": True,
        "fast_features": True,
    },
}
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
printf '%s\n' "${MODEL_PATHS[@]}" > "$MODEL_LIST"
printf '%s\n' "${SWEEP_ARGS[@]}" > "$SWEEP_ARGS_FILE"

echo "[xgbcat-risk] symbols=$SYMBOLS ($(wc -l < "$SYMBOLS") names)"
echo "[xgbcat-risk] output=$OUT"
echo "[xgbcat-risk] xgb_model_count=${#XGB_MODELS[@]}"
echo "[xgbcat-risk] cat_model_count=${#CAT_MODELS[@]}"
echo "[xgbcat-risk] model_count=${#MODEL_PATHS[@]}"
echo "[xgbcat-risk] manifest=$MANIFEST"
echo "[xgbcat-risk] run_log=$RUN_LOG"
echo "[xgbcat-risk] sweep_args=$SWEEP_ARGS_FILE"
echo "[xgbcat-risk] done_marker=$DONE_MARKER"
echo "[xgbcat-risk] result_artifacts=$RESULTS_MANIFEST"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[xgbcat-risk] dry run: preflight passed; sweep not launched"
  exit 0
fi

RUN_STARTED_NS=$(python - <<'PY'
import time

print(time.time_ns())
PY
)
python -m xgbnew.sweep_ensemble_grid "${SWEEP_ARGS[@]}" \
  2>&1 | tee "$RUN_LOG" "logs/track1/xgbcat_risk_parity_wide_120d.log"

python "$SCRIPT_DIR/write_xgbcat_risk_parity_completion.py" \
  --done-marker "$DONE_MARKER" \
  --manifest "$MANIFEST" \
  --sweep-args "$SWEEP_ARGS_FILE" \
  --run-log "$RUN_LOG" \
  --results-manifest "$RESULTS_MANIFEST" \
  --output-dir "$OUT" \
  --run-started-ns "$RUN_STARTED_NS"
if ! python "$SCRIPT_DIR/write_xgbcat_risk_parity_completion.py" \
  --verify \
  --done-marker "$DONE_MARKER"; then
  FAILED_MARKER="$DONE_MARKER.failed.1"
  marker_idx=1
  while [ -e "$FAILED_MARKER" ]; do
    marker_idx=$((marker_idx + 1))
    FAILED_MARKER="$DONE_MARKER.failed.$marker_idx"
  done
  if [ -e "$DONE_MARKER" ]; then
    mv "$DONE_MARKER" "$FAILED_MARKER"
  fi
  die "completion verification failed; quarantined marker=$FAILED_MARKER"
fi
echo "[xgbcat-risk] completed; done_marker=$DONE_MARKER"
