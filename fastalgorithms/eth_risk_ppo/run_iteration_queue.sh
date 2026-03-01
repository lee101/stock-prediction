#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

WAIT_FOR_PID="${WAIT_FOR_PID:-}"
ITERATION_SPECS="${ITERATION_SPECS:-iter_explore_2:200000,iter_refine_1:300000}"

if [[ -n "${WAIT_FOR_PID}" ]]; then
  while kill -0 "${WAIT_FOR_PID}" >/dev/null 2>&1; do
    echo "waiting for pid ${WAIT_FOR_PID}..."
    sleep 60
  done
fi

IFS=',' read -r -a SPECS <<< "${ITERATION_SPECS}"
if [[ "${#SPECS[@]}" -eq 0 ]]; then
  echo "No iteration specs provided via ITERATION_SPECS." >&2
  exit 1
fi

for spec in "${SPECS[@]}"; do
  TAG="${spec%%:*}"
  TIMESTEPS="${spec##*:}"
  if [[ -z "${TAG}" || -z "${TIMESTEPS}" || "${TAG}" == "${TIMESTEPS}" ]]; then
    echo "Invalid iteration spec '${spec}'. Expected format tag:timesteps." >&2
    exit 1
  fi
  echo "starting batch ${TAG} (${TIMESTEPS} steps)"
  bash fastalgorithms/eth_risk_ppo/run_iteration_batch.sh "${TAG}" "${TIMESTEPS}"
done

