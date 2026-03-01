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
mkdir -p "$(dirname "${LEADERBOARD}")"
echo "run_name,variant,num_timesteps,robust_score,long_return_5_10bp,short_return_5_10bp,long_sortino_5_10bp,all_returns_positive_5_10bp,summary_csv" > "${LEADERBOARD}"

declare -a VARIANTS=(
  "base_long|LEARNING_RATE=5e-5 BATCH_SIZE=512 N_STEPS=2048 GAMMA=0.995 GAE_LAMBDA=0.98 DRAWDOWN_PENALTY=0.05 CVAR_PENALTY=0.25 TURNOVER_PENALTY=0.001 REGIME_LEVERAGE_SCALE=0.35"
  "lr_lo|LEARNING_RATE=3e-5 BATCH_SIZE=512 N_STEPS=2048 GAMMA=0.995 GAE_LAMBDA=0.98 DRAWDOWN_PENALTY=0.05 CVAR_PENALTY=0.25 TURNOVER_PENALTY=0.001 REGIME_LEVERAGE_SCALE=0.35"
  "risk_tight|LEARNING_RATE=5e-5 BATCH_SIZE=512 N_STEPS=2048 GAMMA=0.995 GAE_LAMBDA=0.98 DRAWDOWN_PENALTY=0.08 CVAR_PENALTY=0.35 TURNOVER_PENALTY=0.0015 REGIME_LEVERAGE_SCALE=0.30"
  "cost_high|LEARNING_RATE=5e-5 BATCH_SIZE=512 N_STEPS=2048 GAMMA=0.995 GAE_LAMBDA=0.98 COSTS_BPS=6.0 DRAWDOWN_PENALTY=0.06 CVAR_PENALTY=0.30 TURNOVER_PENALTY=0.0020 REGIME_LEVERAGE_SCALE=0.30"
  "regime_hard|LEARNING_RATE=4e-5 BATCH_SIZE=512 N_STEPS=2048 GAMMA=0.997 GAE_LAMBDA=0.99 DRAWDOWN_PENALTY=0.09 CVAR_PENALTY=0.40 TURNOVER_PENALTY=0.0018 REGIME_DRAWDOWN_THRESHOLD=0.02 REGIME_LEVERAGE_SCALE=0.20"
  "horizon_long|LEARNING_RATE=4e-5 BATCH_SIZE=1024 N_STEPS=4096 GAMMA=0.997 GAE_LAMBDA=0.99 DRAWDOWN_PENALTY=0.06 CVAR_PENALTY=0.30 TURNOVER_PENALTY=0.001 REGIME_LEVERAGE_SCALE=0.30"
  "entropy_small|LEARNING_RATE=5e-5 BATCH_SIZE=512 N_STEPS=2048 GAMMA=0.995 GAE_LAMBDA=0.98 ENT_COEF=0.001 DRAWDOWN_PENALTY=0.05 CVAR_PENALTY=0.25 TURNOVER_PENALTY=0.0012 REGIME_LEVERAGE_SCALE=0.35"
  "cash_buffer|LEARNING_RATE=5e-5 BATCH_SIZE=512 N_STEPS=2048 GAMMA=0.995 GAE_LAMBDA=0.98 WEIGHT_CAP=0.65 DRAWDOWN_PENALTY=0.06 CVAR_PENALTY=0.30 TURNOVER_PENALTY=0.0010 REGIME_LEVERAGE_SCALE=0.30"
  "short_balanced|LEARNING_RATE=5e-5 BATCH_SIZE=512 N_STEPS=2048 GAMMA=0.995 GAE_LAMBDA=0.98 ALLOW_SHORT=1 LEVERAGE_CAP=1.30 DRAWDOWN_PENALTY=0.06 CVAR_PENALTY=0.30 TURNOVER_PENALTY=0.0012 REGIME_LEVERAGE_SCALE=0.30"
  "short_tight|LEARNING_RATE=4e-5 BATCH_SIZE=512 N_STEPS=2048 GAMMA=0.997 GAE_LAMBDA=0.99 ALLOW_SHORT=1 LEVERAGE_CAP=1.10 DRAWDOWN_PENALTY=0.09 CVAR_PENALTY=0.40 TURNOVER_PENALTY=0.0018 REGIME_DRAWDOWN_THRESHOLD=0.02 REGIME_LEVERAGE_SCALE=0.20"
)

for spec in "${VARIANTS[@]}"; do
  IFS="|" read -r VARIANT_NAME OVERRIDES <<< "${spec}"
  RUN_NAME="eth_${ITER_TAG}_${VARIANT_NAME}_${TS}"
  echo "=== Training ${RUN_NAME} (${NUM_TIMESTEPS} steps) ==="

  # shellcheck disable=SC2086
  env VENV_PATH="${VENV_PATH}" ${OVERRIDES} bash fastalgorithms/eth_risk_ppo/run_train_local.sh "${NUM_TIMESTEPS}" "${RUN_NAME}"

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
all_positive = int(bool((subset["total_return"] > 0).all())) if not subset.empty else 0
score = (long_ret * 100.0) + (0.05 * long_sort) + (worst_long * 50.0)
if all_positive:
    score += 5.0
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
