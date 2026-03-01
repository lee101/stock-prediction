#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-eth_risk_queue_remote_$(date -u +%Y%m%d_%H%M%S)}"

REMOTE_HOST="${REMOTE_HOST:-administrator@93.127.141.100}"
REMOTE_DIR="${REMOTE_DIR:-/nvme0n1-disk/code/stock-prediction}"
REMOTE_SSH="${REMOTE_SSH:-ssh -o StrictHostKeyChecking=no}"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
ITERATION_SPECS="${ITERATION_SPECS:-iter_explore_2:200000,iter_refine_1:300000}"
VENV_PATH="${VENV_PATH:-.venv312}"
DATA_DIR="${DATA_DIR:-trainingdatahourly}"

REMOTE_CMD="
set -euo pipefail
cd '${REMOTE_DIR}'
git fetch --all --prune
git checkout '${BRANCH}'
git pull --rebase
mkdir -p fastalgorithms/eth_risk_ppo/logs
nohup env WAIT_FOR_PID='${WAIT_FOR_PID}' ITERATION_SPECS='${ITERATION_SPECS}' \\
  VENV_PATH='${VENV_PATH}' DATA_DIR='${DATA_DIR}' \\
  bash fastalgorithms/eth_risk_ppo/run_iteration_queue.sh \\
  > 'fastalgorithms/eth_risk_ppo/logs/${RUN_ID}.log' 2>&1 &
echo \$! > 'fastalgorithms/eth_risk_ppo/logs/${RUN_ID}.pid'
echo launched '${RUN_ID}' pid=\$(cat 'fastalgorithms/eth_risk_ppo/logs/${RUN_ID}.pid')
"

# shellcheck disable=SC2086
${REMOTE_SSH} "${REMOTE_HOST}" "${REMOTE_CMD}"

echo "Remote iteration queue launched: ${RUN_ID}"
echo "Check log:"
echo "  ${REMOTE_SSH} ${REMOTE_HOST} 'tail -n 120 ${REMOTE_DIR}/fastalgorithms/eth_risk_ppo/logs/${RUN_ID}.log'"
