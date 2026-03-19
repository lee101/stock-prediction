#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${REPO_ROOT}/.venv313"
LOG_DIR="${REPO_ROOT}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${LOG_DIR}"

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
else
  echo "[ERROR] venv not found at ${VENV_PATH}." >&2
  exit 1
fi

run_train() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}_${TIMESTAMP}.log"
  echo "[$(date +%F_%T)] Starting ${name}" | tee -a "${log_file}"
  # Ensure unbuffered logs for long-running training
  PYTHONUNBUFFERED=1 "$@" 2>&1 | tee -a "${log_file}"
  local rc=${PIPESTATUS[0]}
  if [[ ${rc} -ne 0 ]]; then
    echo "[$(date +%F_%T)] ${name} failed with exit code ${rc}" | tee -a "${log_file}"
  else
    echo "[$(date +%F_%T)] ${name} completed" | tee -a "${log_file}"
  fi
  return ${rc}
}

run_train "train_LINKUSD" python train_hourlyv5.py \
  --symbol LINKUSD \
  --epochs 100 \
  --batch-size 64 \
  --checkpoint-dir neuralhourlytradingv5/checkpoints_link

run_train "train_UNIUSD" python train_hourlyv5.py \
  --symbol UNIUSD \
  --epochs 100 \
  --batch-size 64 \
  --checkpoint-dir neuralhourlytradingv5/checkpoints_uni

run_train "train_NFLX" python train_stocks_hourlyv5.py \
  --symbols NFLX \
  --epochs 100 \
  --batch-size 64 \
  --checkpoint-dir neuralhourlystocksv5/checkpoints_nflx
