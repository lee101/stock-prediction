#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

NUM_TIMESTEPS="${1:-150000}"
RUN_NAME="${2:-eth_risk_ppo_$(date -u +%Y%m%d_%H%M%S)}"
VENV_PATH="${VENV_PATH:-.venv312}"
DATA_DIR="${DATA_DIR:-trainingdatahourly}"

LEARNING_RATE="${LEARNING_RATE:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-512}"
N_STEPS="${N_STEPS:-2048}"
N_EPOCHS="${N_EPOCHS:-10}"
GAMMA="${GAMMA:-0.995}"
GAE_LAMBDA="${GAE_LAMBDA:-0.98}"
CLIP_RANGE="${CLIP_RANGE:-0.2}"
CLIP_RANGE_VF="${CLIP_RANGE_VF:-}"
VF_COEF="${VF_COEF:-0.5}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"
TARGET_KL="${TARGET_KL:-0.015}"
ENT_COEF="${ENT_COEF:-0.0}"
COSTS_BPS="${COSTS_BPS:-3.0}"
TURNOVER_PENALTY="${TURNOVER_PENALTY:-0.001}"
DRAWDOWN_PENALTY="${DRAWDOWN_PENALTY:-0.05}"
CVAR_PENALTY="${CVAR_PENALTY:-0.25}"
UNCERTAINTY_PENALTY="${UNCERTAINTY_PENALTY:-0.05}"
REGIME_DRAWDOWN_THRESHOLD="${REGIME_DRAWDOWN_THRESHOLD:-0.03}"
REGIME_LEVERAGE_SCALE="${REGIME_LEVERAGE_SCALE:-0.35}"
WEIGHT_CAP="${WEIGHT_CAP:-}"
ALLOW_SHORT="${ALLOW_SHORT:-0}"
INCLUDE_CASH="${INCLUDE_CASH:-1}"
LEVERAGE_CAP="${LEVERAGE_CAP:-1.0}"
POLICY_DTYPE="${POLICY_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"

PYTHON_BIN="${VENV_PATH}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing python executable: ${PYTHON_BIN}" >&2
  exit 1
fi

OUT_DIR="fastalgorithms/eth_risk_ppo/artifacts/${RUN_NAME}"
TB_DIR="fastalgorithms/eth_risk_ppo/runs"
CACHE_FEATURES_TO="${CACHE_FEATURES_TO:-${OUT_DIR}/features_cache.npz}"
mkdir -p "${OUT_DIR}" "${TB_DIR}"

# Support both layouts:
# - trainingdatahourly/ETHUSD.csv
# - trainingdatahourly/crypto/ETHUSD.csv
if [[ ! -f "${DATA_DIR}/ETHUSD.csv" && -f "${DATA_DIR}/crypto/ETHUSD.csv" ]]; then
  DATA_DIR="${DATA_DIR}/crypto"
fi

CMD=(
  "${PYTHON_BIN}" -m gymrl.train_ppo_allocator
  --data-dir "${DATA_DIR}"
  --symbols ETHUSD
  --forecast-backend bootstrap
  --num-samples 128
  --context-window 128
  --prediction-length 1
  --realized-horizon 1
  --num-timesteps "${NUM_TIMESTEPS}"
  --learning-rate "${LEARNING_RATE}"
  --batch-size "${BATCH_SIZE}"
  --n-steps "${N_STEPS}"
  --n-epochs "${N_EPOCHS}"
  --gamma "${GAMMA}"
  --gae-lambda "${GAE_LAMBDA}"
  --clip-range "${CLIP_RANGE}"
  --vf-coef "${VF_COEF}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --target-kl "${TARGET_KL}"
  --ent-coef "${ENT_COEF}"
  --costs-bps "${COSTS_BPS}"
  --turnover-penalty "${TURNOVER_PENALTY}"
  --drawdown-penalty "${DRAWDOWN_PENALTY}"
  --cvar-penalty "${CVAR_PENALTY}"
  --uncertainty-penalty "${UNCERTAINTY_PENALTY}"
  --regime-filters-enabled
  --regime-drawdown-threshold "${REGIME_DRAWDOWN_THRESHOLD}"
  --regime-leverage-scale "${REGIME_LEVERAGE_SCALE}"
  --policy-dtype "${POLICY_DTYPE}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --no-wandb
  --run-name "${RUN_NAME}"
  --tensorboard-log "${TB_DIR}"
  --cache-features-to "${CACHE_FEATURES_TO}"
  --output-dir "${OUT_DIR}"
)

if [[ -n "${WEIGHT_CAP}" ]]; then
  CMD+=(--weight-cap "${WEIGHT_CAP}")
fi

if [[ -n "${CLIP_RANGE_VF}" ]]; then
  CMD+=(--clip-range-vf "${CLIP_RANGE_VF}")
fi

if [[ "${ALLOW_SHORT}" == "1" ]]; then
  CMD+=(--allow-short --leverage-cap "${LEVERAGE_CAP}" --no-include-cash)
elif [[ "${INCLUDE_CASH}" == "0" ]]; then
  CMD+=(--no-include-cash)
else
  CMD+=(--include-cash)
fi

"${CMD[@]}"

echo "Completed run: ${RUN_NAME}"
echo "Artifacts: ${OUT_DIR}"
