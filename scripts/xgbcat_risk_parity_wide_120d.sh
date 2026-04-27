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

cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate

SYMBOLS=${SYMBOLS:-symbol_lists/stocks_wide_fresh0401_photonics_2500_v1.txt}
if [ ! -f "$SYMBOLS" ]; then
  SYMBOLS=symbol_lists/stocks_wide_1000_v1.txt
fi

OUT=${1:-analysis/xgbnew_daily/xgbcat_risk_parity_wide_120d}
mkdir -p "$OUT" logs/track1

XGB_DIR=${XGB_DIR:-analysis/xgbnew_daily/track1_oos120d_xgb}
CAT_DIR=${CAT_DIR:-analysis/xgbnew_daily/track1_oos120d_cat}
XGB=$(ls "$XGB_DIR"/alltrain_seed*.pkl)
CAT=$(ls "$CAT_DIR"/alltrain_seed*.pkl)
PATHS=$(printf '%s\n%s\n' "$XGB" "$CAT" | paste -sd,)

echo "[xgbcat-risk] symbols=$SYMBOLS ($(wc -l < "$SYMBOLS") names)"
echo "[xgbcat-risk] output=$OUT"
echo "[xgbcat-risk] model_count=$(printf '%s' "$PATHS" | tr ',' '\n' | wc -l)"

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
