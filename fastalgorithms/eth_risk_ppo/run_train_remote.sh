#!/usr/bin/env bash
set -euo pipefail

NUM_TIMESTEPS="${1:-300000}"
RUN_NAME="${2:-eth_risk_ppo_remote_$(date -u +%Y%m%d_%H%M%S)}"

REMOTE_HOST="${REMOTE_HOST:-administrator@93.127.141.100}"
REMOTE_DIR="${REMOTE_DIR:-/nvme0n1-disk/code/stock-prediction}"
REMOTE_SSH="${REMOTE_SSH:-ssh -o StrictHostKeyChecking=no}"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"

REMOTE_CMD="
set -euo pipefail
cd '${REMOTE_DIR}'
git fetch --all --prune
git checkout '${BRANCH}'
git pull --rebase
if [ -f .venv312/bin/activate ]; then
  source .venv312/bin/activate
elif [ -f .venv313/bin/activate ]; then
  source .venv313/bin/activate
fi
mkdir -p fastalgorithms/eth_risk_ppo/logs
nohup bash fastalgorithms/eth_risk_ppo/run_train_local.sh '${NUM_TIMESTEPS}' '${RUN_NAME}' > 'fastalgorithms/eth_risk_ppo/logs/${RUN_NAME}.log' 2>&1 &
echo \$! > 'fastalgorithms/eth_risk_ppo/logs/${RUN_NAME}.pid'
echo launched '${RUN_NAME}' pid=\$(cat 'fastalgorithms/eth_risk_ppo/logs/${RUN_NAME}.pid')
"

# shellcheck disable=SC2086
${REMOTE_SSH} "${REMOTE_HOST}" "${REMOTE_CMD}"

echo "Remote launch requested: ${RUN_NAME}"
echo "Check log:"
echo "  ${REMOTE_SSH} ${REMOTE_HOST} 'tail -n 50 ${REMOTE_DIR}/fastalgorithms/eth_risk_ppo/logs/${RUN_NAME}.log'"
