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

REPO="${REPO:-/nvme0n1-disk/code/stock-prediction}"
cd "$REPO" || die "repo not found: $REPO"

VENV="${VENV:-.venv313}"
if [ ! -f "$VENV/bin/activate" ]; then
  die "missing virtualenv activation script: $VENV/bin/activate"
fi
source "$VENV/bin/activate"

SYMBOLS=${SYMBOLS:-symbol_lists/stocks_wide_fresh0401_photonics_2500_v1.txt}
if [ ! -f "$SYMBOLS" ]; then
  SYMBOLS=symbol_lists/stocks_wide_1000_v1.txt
fi
if [ ! -f "$SYMBOLS" ]; then
  die "missing symbols file: $SYMBOLS"
fi

OUT=${1:-analysis/xgbnew_daily/xgbcat_risk_parity_wide_120d}
mkdir -p "$OUT" logs/track1

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
MANIFEST="$OUT/run_manifest.json"
MODEL_LIST="$OUT/model_paths.txt"
printf '%s\n' "${MODEL_PATHS[@]}" > "$MODEL_LIST"

export XGBCAT_REPO="$REPO"
export XGBCAT_VENV="$VENV"
export XGBCAT_SYMBOLS="$SYMBOLS"
export XGBCAT_OUT="$OUT"
export XGBCAT_XGB_DIR="$XGB_DIR"
export XGBCAT_CAT_DIR="$CAT_DIR"
export XGBCAT_XGB_COUNT="${#XGB_MODELS[@]}"
python - "$MODEL_LIST" "$MANIFEST" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


model_list = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
model_paths = [
    Path(line.strip())
    for line in model_list.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
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
    "FILL_BUFFER_BPS_GRID": "-1,10,20",
    "FEE_REGIMES": "deploy,stress36x",
    "FAIL_FAST_MAX_DD_PCT": "20",
    "FAIL_FAST_MAX_INTRADAY_DD_PCT": "20",
    "FAIL_FAST_NEG_WINDOWS": "1",
    "CHECKPOINT_EVERY_CELLS": "50",
}
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
    "models": [
        {
            "family": "xgb" if idx < xgb_count else "cat",
            "path": str(path.resolve()),
            "sha256": sha256(path),
        }
        for idx, path in enumerate(model_paths)
    ],
    "config": {
        key.lower(): os.environ.get(key, default)
        for key, default in config_defaults.items()
    },
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

echo "[xgbcat-risk] symbols=$SYMBOLS ($(wc -l < "$SYMBOLS") names)"
echo "[xgbcat-risk] output=$OUT"
echo "[xgbcat-risk] xgb_model_count=${#XGB_MODELS[@]}"
echo "[xgbcat-risk] cat_model_count=${#CAT_MODELS[@]}"
echo "[xgbcat-risk] model_count=${#MODEL_PATHS[@]}"
echo "[xgbcat-risk] manifest=$MANIFEST"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[xgbcat-risk] dry run: preflight passed; sweep not launched"
  exit 0
fi

python -m xgbnew.sweep_ensemble_grid \
  --symbols-file "$SYMBOLS" \
  --model-paths "$PATHS" \
  --oos-start "${OOS_START:-2025-12-18}" \
  --oos-end "${OOS_END:-2026-04-20}" \
  --window-days "${WINDOW_DAYS:-30}" \
  --stride-days "${STRIDE_DAYS:-3}" \
  --leverage-grid "${LEVERAGE_GRID:-1.5,2.0,2.25,2.5}" \
  --min-score-grid "${MIN_SCORE_GRID:-0.58,0.62,0.66,0.70}" \
  --top-n-grid "${TOP_N_GRID:-2,3,4}" \
  --hold-through \
  --min-dollar-vol "${MIN_DOLLAR_VOL:-50000000}" \
  --inference-min-dolvol-grid "${INFERENCE_MIN_DOLVOL_GRID:-50000000,100000000}" \
  --inference-min-vol-grid "${INFERENCE_MIN_VOL_GRID:-0.12,0.15}" \
  --inference-max-vol-grid "${INFERENCE_MAX_VOL_GRID:-0.0,0.80}" \
  --inference-max-spread-bps-grid "${INFERENCE_MAX_SPREAD_BPS_GRID:-20,30}" \
  --vol-target-ann-grid "${VOL_TARGET_ANN_GRID:-0.0,0.18,0.22}" \
  --inv-vol-target-grid "${INV_VOL_TARGET_GRID:-0.0,0.20,0.25,0.30}" \
  --inv-vol-floor "${INV_VOL_FLOOR:-0.08}" \
  --inv-vol-cap "${INV_VOL_CAP:-2.0}" \
  --regime-cs-skew-min-grid "${REGIME_CS_SKEW_MIN_GRID:-0.50,0.75,1.00}" \
  --allocation-mode-grid "${ALLOCATION_MODE_GRID:-equal,score_norm,softmax}" \
  --allocation-temp-grid "${ALLOCATION_TEMP_GRID:-0.5,1.0}" \
  --score-uncertainty-penalty-grid "${SCORE_UNCERTAINTY_PENALTY_GRID:-0.0,0.5,1.0}" \
  --fill-buffer-bps-grid "${FILL_BUFFER_BPS_GRID:--1,10,20}" \
  --fee-regimes "${FEE_REGIMES:-deploy,stress36x}" \
  --fail-fast-max-dd-pct "${FAIL_FAST_MAX_DD_PCT:-20}" \
  --fail-fast-max-intraday-dd-pct "${FAIL_FAST_MAX_INTRADAY_DD_PCT:-20}" \
  --fail-fast-neg-windows "${FAIL_FAST_NEG_WINDOWS:-1}" \
  --checkpoint-every-cells "${CHECKPOINT_EVERY_CELLS:-50}" \
  --require-production-target \
  --fast-features \
  --output-dir "$OUT" \
  --verbose \
  2>&1 | tee "logs/track1/xgbcat_risk_parity_wide_120d.log"
