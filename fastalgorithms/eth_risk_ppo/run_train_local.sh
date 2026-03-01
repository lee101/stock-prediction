#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

NUM_TIMESTEPS="${1:-150000}"
RUN_NAME="${2:-eth_risk_ppo_$(date -u +%Y%m%d_%H%M%S)}"
VENV_PATH="${VENV_PATH:-.venv312}"
DATA_DIR="${DATA_DIR:-trainingdatahourly}"

PYTHON_BIN="${VENV_PATH}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing python executable: ${PYTHON_BIN}" >&2
  exit 1
fi

OUT_DIR="fastalgorithms/eth_risk_ppo/artifacts/${RUN_NAME}"
TB_DIR="fastalgorithms/eth_risk_ppo/runs"
mkdir -p "${OUT_DIR}" "${TB_DIR}"

# Support both layouts:
# - trainingdatahourly/ETHUSD.csv
# - trainingdatahourly/crypto/ETHUSD.csv
if [[ ! -f "${DATA_DIR}/ETHUSD.csv" && -f "${DATA_DIR}/crypto/ETHUSD.csv" ]]; then
  DATA_DIR="${DATA_DIR}/crypto"
fi

"${PYTHON_BIN}" -m gymrl.train_ppo_allocator \
  --data-dir "${DATA_DIR}" \
  --symbols ETHUSD \
  --forecast-backend bootstrap \
  --num-samples 128 \
  --context-window 128 \
  --prediction-length 1 \
  --realized-horizon 1 \
  --num-timesteps "${NUM_TIMESTEPS}" \
  --learning-rate 5e-5 \
  --batch-size 512 \
  --n-steps 2048 \
  --gamma 0.995 \
  --gae-lambda 0.98 \
  --clip-range 0.2 \
  --costs-bps 3.0 \
  --turnover-penalty 0.001 \
  --drawdown-penalty 0.05 \
  --cvar-penalty 0.25 \
  --uncertainty-penalty 0.05 \
  --regime-filters-enabled \
  --regime-drawdown-threshold 0.03 \
  --regime-leverage-scale 0.35 \
  --policy-dtype bfloat16 \
  --device auto \
  --no-wandb \
  --run-name "${RUN_NAME}" \
  --tensorboard-log "${TB_DIR}" \
  --output-dir "${OUT_DIR}"

echo "Completed run: ${RUN_NAME}"
echo "Artifacts: ${OUT_DIR}"
