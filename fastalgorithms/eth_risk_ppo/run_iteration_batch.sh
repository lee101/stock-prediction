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
echo "run_name,variant,num_timesteps,score_sortino_5_10bp,mean_return_5_10bp,summary_csv" > "${LEADERBOARD}"

declare -a VARIANTS=(
  "base|LEARNING_RATE=5e-5 BATCH_SIZE=512 N_STEPS=2048 GAMMA=0.995 GAE_LAMBDA=0.98 DRAWDOWN_PENALTY=0.05 CVAR_PENALTY=0.25 TURNOVER_PENALTY=0.001 REGIME_LEVERAGE_SCALE=0.35"
  "lr_hi|LEARNING_RATE=8e-5 BATCH_SIZE=512 N_STEPS=2048 GAMMA=0.995 GAE_LAMBDA=0.98 DRAWDOWN_PENALTY=0.05 CVAR_PENALTY=0.25 TURNOVER_PENALTY=0.001 REGIME_LEVERAGE_SCALE=0.35"
  "lr_lo|LEARNING_RATE=3e-5 BATCH_SIZE=512 N_STEPS=2048 GAMMA=0.995 GAE_LAMBDA=0.98 DRAWDOWN_PENALTY=0.05 CVAR_PENALTY=0.25 TURNOVER_PENALTY=0.001 REGIME_LEVERAGE_SCALE=0.35"
  "risk_tight|LEARNING_RATE=5e-5 BATCH_SIZE=512 N_STEPS=2048 GAMMA=0.995 GAE_LAMBDA=0.98 DRAWDOWN_PENALTY=0.08 CVAR_PENALTY=0.35 TURNOVER_PENALTY=0.0015 REGIME_LEVERAGE_SCALE=0.30"
  "risk_loose|LEARNING_RATE=5e-5 BATCH_SIZE=512 N_STEPS=2048 GAMMA=0.995 GAE_LAMBDA=0.98 DRAWDOWN_PENALTY=0.03 CVAR_PENALTY=0.15 TURNOVER_PENALTY=0.0007 REGIME_LEVERAGE_SCALE=0.45"
  "batch_big|LEARNING_RATE=5e-5 BATCH_SIZE=1024 N_STEPS=4096 GAMMA=0.995 GAE_LAMBDA=0.985 DRAWDOWN_PENALTY=0.05 CVAR_PENALTY=0.25 TURNOVER_PENALTY=0.001 REGIME_LEVERAGE_SCALE=0.35"
  "batch_small|LEARNING_RATE=5e-5 BATCH_SIZE=256 N_STEPS=1024 GAMMA=0.99 GAE_LAMBDA=0.95 DRAWDOWN_PENALTY=0.05 CVAR_PENALTY=0.25 TURNOVER_PENALTY=0.0012 REGIME_LEVERAGE_SCALE=0.35"
  "horizon_long|LEARNING_RATE=4e-5 BATCH_SIZE=1024 N_STEPS=4096 GAMMA=0.997 GAE_LAMBDA=0.99 DRAWDOWN_PENALTY=0.06 CVAR_PENALTY=0.30 TURNOVER_PENALTY=0.001 REGIME_LEVERAGE_SCALE=0.30"
  "clip_tight|LEARNING_RATE=5e-5 BATCH_SIZE=512 N_STEPS=2048 GAMMA=0.995 GAE_LAMBDA=0.98 CLIP_RANGE=0.15 DRAWDOWN_PENALTY=0.05 CVAR_PENALTY=0.25 TURNOVER_PENALTY=0.001 REGIME_LEVERAGE_SCALE=0.35"
  "clip_loose|LEARNING_RATE=5e-5 BATCH_SIZE=512 N_STEPS=2048 GAMMA=0.995 GAE_LAMBDA=0.98 CLIP_RANGE=0.25 DRAWDOWN_PENALTY=0.05 CVAR_PENALTY=0.20 TURNOVER_PENALTY=0.0009 REGIME_LEVERAGE_SCALE=0.40"
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
    subset = df
score = float(subset["sortino"].mean()) if not subset.empty else 0.0
ret = float(subset["total_return"].mean()) if not subset.empty else 0.0
print(f"{score:.6f},{ret:.6f}")
PY
)"
  SCORE="$(echo "${SCORE_LINE}" | cut -d',' -f1)"
  RET="$(echo "${SCORE_LINE}" | cut -d',' -f2)"
  echo "${RUN_NAME},${VARIANT_NAME},${NUM_TIMESTEPS},${SCORE},${RET},${EVAL_DIR}/summary.csv" >> "${LEADERBOARD}"
done

echo "=== Leaderboard ==="
"${PYTHON_BIN}" - <<'PY' "${LEADERBOARD}"
import pandas as pd
import sys
df = pd.read_csv(sys.argv[1]).sort_values("score_sortino_5_10bp", ascending=False)
print(df.to_string(index=False))
PY
echo "Wrote: ${LEADERBOARD}"
