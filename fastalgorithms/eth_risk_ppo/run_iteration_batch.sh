#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ITER_TAG="${1:-iter1}"
NUM_TIMESTEPS="${2:-120000}"
VENV_PATH="${VENV_PATH:-.venv312}"
SYMBOL="${SYMBOL:-ETHUSD}"
WINDOWS_HOURS="${WINDOWS_HOURS:-24,168,720}"
FILL_BUFFERS_BPS="${FILL_BUFFERS_BPS:-0,5,10}"

PYTHON_BIN="${VENV_PATH}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing python executable: ${PYTHON_BIN}" >&2
  exit 1
fi

TS="$(date -u +%Y%m%d_%H%M%S)"
LEADERBOARD="fastalgorithms/eth_risk_ppo/artifacts/${ITER_TAG}_${TS}_leaderboard.csv"
ITER_FEATURE_CACHE="fastalgorithms/eth_risk_ppo/artifacts/${ITER_TAG}_${TS}_features_cache.npz"
mkdir -p "$(dirname "${LEADERBOARD}")"
echo "run_name,variant,num_timesteps,robust_score,long_return_5_10bp,short_return_5_10bp,long_sortino_5_10bp,all_returns_positive_5_10bp,summary_csv" > "${LEADERBOARD}"

declare -a VARIANTS=(
  "regime_core|LEARNING_RATE=4e-5 BATCH_SIZE=512 N_STEPS=2048 N_EPOCHS=10 GAMMA=0.997 GAE_LAMBDA=0.99 TARGET_KL=0.010 LEVERAGE_HEAD=0 MAX_GROSS_LEVERAGE=1.0 INTRADAY_LEVERAGE_CAP=1.2 DRAWDOWN_PENALTY=0.09 CVAR_PENALTY=0.40 TURNOVER_PENALTY=0.0018 REGIME_DRAWDOWN_THRESHOLD=0.02 REGIME_LEVERAGE_SCALE=0.20"
  "regime_core_s7|SEED=7 LEARNING_RATE=4e-5 BATCH_SIZE=512 N_STEPS=2048 N_EPOCHS=10 GAMMA=0.997 GAE_LAMBDA=0.99 TARGET_KL=0.010 LEVERAGE_HEAD=0 MAX_GROSS_LEVERAGE=1.0 INTRADAY_LEVERAGE_CAP=1.2 DRAWDOWN_PENALTY=0.09 CVAR_PENALTY=0.40 TURNOVER_PENALTY=0.0018 REGIME_DRAWDOWN_THRESHOLD=0.02 REGIME_LEVERAGE_SCALE=0.20"
  "regime_core_s99|SEED=99 LEARNING_RATE=4e-5 BATCH_SIZE=512 N_STEPS=2048 N_EPOCHS=10 GAMMA=0.997 GAE_LAMBDA=0.99 TARGET_KL=0.010 LEVERAGE_HEAD=0 MAX_GROSS_LEVERAGE=1.0 INTRADAY_LEVERAGE_CAP=1.2 DRAWDOWN_PENALTY=0.09 CVAR_PENALTY=0.40 TURNOVER_PENALTY=0.0018 REGIME_DRAWDOWN_THRESHOLD=0.02 REGIME_LEVERAGE_SCALE=0.20"
  "regime_soft|LEARNING_RATE=4e-5 BATCH_SIZE=512 N_STEPS=2048 N_EPOCHS=10 GAMMA=0.996 GAE_LAMBDA=0.99 TARGET_KL=0.012 LEVERAGE_HEAD=0 MAX_GROSS_LEVERAGE=1.0 INTRADAY_LEVERAGE_CAP=1.1 DRAWDOWN_PENALTY=0.06 CVAR_PENALTY=0.28 TURNOVER_PENALTY=0.0022 REGIME_DRAWDOWN_THRESHOLD=0.03 REGIME_LEVERAGE_SCALE=0.25"
  "regime_tight|LEARNING_RATE=4e-5 BATCH_SIZE=512 N_STEPS=2048 N_EPOCHS=12 GAMMA=0.997 GAE_LAMBDA=0.99 TARGET_KL=0.008 LEVERAGE_HEAD=0 MAX_GROSS_LEVERAGE=0.9 INTRADAY_LEVERAGE_CAP=1.0 WEIGHT_CAP=0.75 DRAWDOWN_PENALTY=0.12 CVAR_PENALTY=0.50 TURNOVER_PENALTY=0.0040 REGIME_DRAWDOWN_THRESHOLD=0.015 REGIME_LEVERAGE_SCALE=0.15"
  "low_turnover|LEARNING_RATE=4e-5 BATCH_SIZE=1024 N_STEPS=4096 N_EPOCHS=12 GAMMA=0.997 GAE_LAMBDA=0.99 TARGET_KL=0.010 LEVERAGE_HEAD=0 MAX_GROSS_LEVERAGE=1.0 INTRADAY_LEVERAGE_CAP=1.1 DRAWDOWN_PENALTY=0.11 CVAR_PENALTY=0.48 TURNOVER_PENALTY=0.0100 REGIME_DRAWDOWN_THRESHOLD=0.02 REGIME_LEVERAGE_SCALE=0.15"
  "cash_high|LEARNING_RATE=4e-5 BATCH_SIZE=512 N_STEPS=2048 N_EPOCHS=12 GAMMA=0.996 GAE_LAMBDA=0.99 TARGET_KL=0.010 LEVERAGE_HEAD=0 MAX_GROSS_LEVERAGE=1.0 INTRADAY_LEVERAGE_CAP=1.0 WEIGHT_CAP=0.55 DRAWDOWN_PENALTY=0.12 CVAR_PENALTY=0.45 TURNOVER_PENALTY=0.0035 REGIME_DRAWDOWN_THRESHOLD=0.02 REGIME_LEVERAGE_SCALE=0.15"
  "cost_realistic|LEARNING_RATE=4e-5 BATCH_SIZE=512 N_STEPS=2048 N_EPOCHS=10 GAMMA=0.996 GAE_LAMBDA=0.99 TARGET_KL=0.010 LEVERAGE_HEAD=0 MAX_GROSS_LEVERAGE=1.0 INTRADAY_LEVERAGE_CAP=1.1 COSTS_BPS=8.0 DRAWDOWN_PENALTY=0.11 CVAR_PENALTY=0.50 TURNOVER_PENALTY=0.0030 REGIME_DRAWDOWN_THRESHOLD=0.02 REGIME_LEVERAGE_SCALE=0.15"
  "cost_stress|LEARNING_RATE=4e-5 BATCH_SIZE=512 N_STEPS=2048 N_EPOCHS=10 GAMMA=0.997 GAE_LAMBDA=0.99 TARGET_KL=0.008 LEVERAGE_HEAD=0 MAX_GROSS_LEVERAGE=1.0 INTRADAY_LEVERAGE_CAP=1.1 COSTS_BPS=12.0 DRAWDOWN_PENALTY=0.14 CVAR_PENALTY=0.55 TURNOVER_PENALTY=0.0045 REGIME_DRAWDOWN_THRESHOLD=0.015 REGIME_LEVERAGE_SCALE=0.12"
  "mild_leverage|LEARNING_RATE=4e-5 BATCH_SIZE=512 N_STEPS=2048 N_EPOCHS=10 GAMMA=0.996 GAE_LAMBDA=0.99 TARGET_KL=0.012 LEVERAGE_HEAD=1 BASE_GROSS_EXPOSURE=1.0 MAX_GROSS_LEVERAGE=1.2 INTRADAY_LEVERAGE_CAP=1.4 DRAWDOWN_PENALTY=0.10 CVAR_PENALTY=0.40 TURNOVER_PENALTY=0.0028 REGIME_DRAWDOWN_THRESHOLD=0.02 REGIME_LEVERAGE_SCALE=0.18"
  "short_balanced|LEARNING_RATE=4e-5 BATCH_SIZE=512 N_STEPS=2048 N_EPOCHS=10 GAMMA=0.997 GAE_LAMBDA=0.99 TARGET_KL=0.010 LEVERAGE_HEAD=0 ALLOW_SHORT=1 LEVERAGE_CAP=1.05 MAX_GROSS_LEVERAGE=1.05 INTRADAY_LEVERAGE_CAP=1.10 DRAWDOWN_PENALTY=0.10 CVAR_PENALTY=0.45 TURNOVER_PENALTY=0.0030 REGIME_DRAWDOWN_THRESHOLD=0.02 REGIME_LEVERAGE_SCALE=0.15"
  "short_tight|LEARNING_RATE=4e-5 BATCH_SIZE=512 N_STEPS=2048 N_EPOCHS=12 GAMMA=0.997 GAE_LAMBDA=0.99 TARGET_KL=0.008 LEVERAGE_HEAD=0 ALLOW_SHORT=1 LEVERAGE_CAP=1.02 MAX_GROSS_LEVERAGE=1.02 INTRADAY_LEVERAGE_CAP=1.05 COSTS_BPS=10.0 DRAWDOWN_PENALTY=0.14 CVAR_PENALTY=0.60 TURNOVER_PENALTY=0.0050 REGIME_DRAWDOWN_THRESHOLD=0.015 REGIME_LEVERAGE_SCALE=0.10"
)

for spec in "${VARIANTS[@]}"; do
  IFS="|" read -r VARIANT_NAME OVERRIDES <<< "${spec}"
  RUN_NAME="eth_${ITER_TAG}_${VARIANT_NAME}_${TS}"
  echo "=== Training ${RUN_NAME} (${NUM_TIMESTEPS} steps) ==="

  # shellcheck disable=SC2086
  env VENV_PATH="${VENV_PATH}" FEATURES_CACHE="${ITER_FEATURE_CACHE}" ${OVERRIDES} \
    bash fastalgorithms/eth_risk_ppo/run_train_local.sh "${NUM_TIMESTEPS}" "${RUN_NAME}"

  CKPT="fastalgorithms/eth_risk_ppo/artifacts/${RUN_NAME}/ppo_allocator_final.zip"
  if [[ ! -f "${CKPT}" ]]; then
    echo "Checkpoint missing for ${RUN_NAME}: ${CKPT}" >&2
    exit 1
  fi

  EVAL_DIR="analysis/eth_risk_ppo/${RUN_NAME}_eval"
  mkdir -p "${EVAL_DIR}"
  "${PYTHON_BIN}" fastalgorithms/eth_risk_ppo/evaluate_checkpoint_windows.py \
    --checkpoint "${CKPT}" \
    --symbol "${SYMBOL}" \
    --windows-hours "${WINDOWS_HOURS}" \
    --fill-buffers-bps "${FILL_BUFFERS_BPS}" \
    --forecast-cache-root binanceneural/forecast_cache \
    --cache-only \
    --output-dir "${EVAL_DIR}"

  SCORE_LINE="$("${PYTHON_BIN}" - <<'PY' "${EVAL_DIR}/summary.csv"
import pandas as pd
import sys
df = pd.read_csv(sys.argv[1])
subset = df[df["fill_buffer_bps"].isin([5.0, 10.0])]
if subset.empty:
    subset = df.copy()
long_h = subset[subset["window_hours"] >= 168]
short_h = subset[subset["window_hours"] <= 24]
if long_h.empty:
    long_h = subset
if short_h.empty:
    short_h = subset
long_ret = float(long_h["total_return"].mean()) if not long_h.empty else 0.0
short_ret = float(short_h["total_return"].mean()) if not short_h.empty else 0.0
long_sort = float(long_h["sortino"].mean()) if not long_h.empty else 0.0
worst_long = float(long_h["total_return"].min()) if not long_h.empty else 0.0
mean_drawdown = float(long_h["max_drawdown"].mean()) if ("max_drawdown" in long_h.columns and not long_h.empty) else 0.0
all_positive = int(bool((subset["total_return"] > 0).all())) if not subset.empty else 0
mean_fills = float(long_h["fills_total"].mean()) if ("fills_total" in long_h.columns and not long_h.empty) else 0.0
mean_turnover = float(long_h["mean_turnover"].mean()) if ("mean_turnover" in long_h.columns and not long_h.empty) else 0.0
score = (
    (long_ret * 140.0)
    + (short_ret * 25.0)
    + (0.06 * long_sort)
    + (worst_long * 60.0)
    - (mean_drawdown * 30.0)
)
if all_positive:
    score += 5.0
if mean_fills < 1.0:
    score -= 30.0
elif mean_fills < 5.0:
    score -= 12.0
if mean_turnover < 1e-4:
    score -= 12.0
elif mean_turnover > 0.45:
    score -= 8.0
print(f"{score:.6f},{long_ret:.6f},{short_ret:.6f},{long_sort:.6f},{all_positive}")
PY
)"
  SCORE="$(echo "${SCORE_LINE}" | cut -d',' -f1)"
  LONG_RET="$(echo "${SCORE_LINE}" | cut -d',' -f2)"
  SHORT_RET="$(echo "${SCORE_LINE}" | cut -d',' -f3)"
  LONG_SORT="$(echo "${SCORE_LINE}" | cut -d',' -f4)"
  ALL_POS="$(echo "${SCORE_LINE}" | cut -d',' -f5)"
  echo "${RUN_NAME},${VARIANT_NAME},${NUM_TIMESTEPS},${SCORE},${LONG_RET},${SHORT_RET},${LONG_SORT},${ALL_POS},${EVAL_DIR}/summary.csv" >> "${LEADERBOARD}"
done

echo "=== Leaderboard ==="
"${PYTHON_BIN}" - <<'PY' "${LEADERBOARD}"
import pandas as pd
import sys
df = pd.read_csv(sys.argv[1]).sort_values(["robust_score", "long_return_5_10bp"], ascending=False)
print(df.to_string(index=False))
PY
echo "Wrote: ${LEADERBOARD}"
