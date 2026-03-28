#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

VENV="${REPO_ROOT}/.venv312"
if [[ -f "${VENV}/bin/activate" ]]; then
  source "${VENV}/bin/activate"
else
  echo "venv not found at ${VENV}" >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

PYTHONUNBUFFERED=1 python scripts/train_doge_optimized.py 2>&1 | tee "${LOG_DIR}/train_doge_optimized_${TIMESTAMP}.log"
