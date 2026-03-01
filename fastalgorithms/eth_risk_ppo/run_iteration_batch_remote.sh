#!/usr/bin/env bash
set -euo pipefail

ITER_TAG="${1:-iter1}"
NUM_TIMESTEPS="${2:-120000}"
RUN_ID="${3:-eth_risk_${ITER_TAG}_remote_$(date -u +%Y%m%d_%H%M%S)}"

REMOTE_HOST="${REMOTE_HOST:-administrator@93.127.141.100}"
REMOTE_DIR="${REMOTE_DIR:-/nvme0n1-disk/code/stock-prediction}"
REMOTE_SSH="${REMOTE_SSH:-ssh -o StrictHostKeyChecking=no}"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
VENV_PATH="${VENV_PATH:-.venv312}"
DATA_DIR="${DATA_DIR:-trainingdatahourly}"

REMOTE_CMD="
set -euo pipefail
cd '${REMOTE_DIR}'
git fetch --all --prune
git checkout '${BRANCH}'
git pull --rebase
mkdir -p fastalgorithms/eth_risk_ppo/logs
if [ -n '${WAIT_FOR_PID}' ]; then
  while kill -0 '${WAIT_FOR_PID}' >/dev/null 2>&1; do
    echo waiting for pid '${WAIT_FOR_PID}'...
    sleep 60
  done
fi
nohup env VENV_PATH='${VENV_PATH}' DATA_DIR='${DATA_DIR}' bash fastalgorithms/eth_risk_ppo/run_iteration_batch.sh '${ITER_TAG}' '${NUM_TIMESTEPS}' > 'fastalgorithms/eth_risk_ppo/logs/${RUN_ID}.log' 2>&1 &
echo \$! > 'fastalgorithms/eth_risk_ppo/logs/${RUN_ID}.pid'
echo launched '${RUN_ID}' pid=\$(cat 'fastalgorithms/eth_risk_ppo/logs/${RUN_ID}.pid')
"

# shellcheck disable=SC2086
${REMOTE_SSH} "${REMOTE_HOST}" "${REMOTE_CMD}"

echo "Remote iteration batch launched: ${RUN_ID}"
echo "Check log:"
echo "  ${REMOTE_SSH} ${REMOTE_HOST} 'tail -n 80 ${REMOTE_DIR}/fastalgorithms/eth_risk_ppo/logs/${RUN_ID}.log'"
