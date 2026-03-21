#!/usr/bin/env bash
# Profile both pufferlib RL and binanceneural training paths.
#
# Usage:
#   cd /nvme0n1-disk/code/stock-prediction
#   source .venv313/bin/activate
#   bash scripts/profile_all.sh
#
# Outputs in profiles/:
#   pufferlib_cuda_trace.json   — Chrome trace
#   pufferlib_report.md         — top-20 GPU kernels
#   pufferlib_flamegraph.svg    — py-spy flamegraph (if py-spy available)
#   neural_cuda_trace.json      — Chrome trace
#   neural_report.md            — top-20 GPU kernels

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-python}"
PROFILES_DIR="${PROJECT_ROOT}/profiles"

echo "============================================================"
echo "  Stock-Prediction Profiling Harness"
echo "  $(date)"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Profiles dir: ${PROFILES_DIR}"
echo "============================================================"
echo ""

mkdir -p "${PROFILES_DIR}"

# ── 1. Pufferlib RL profiler ──────────────────────────────────────
echo "[1/2] Running pufferlib RL profiler (60s) ..."
T0=$(date +%s)

${PYTHON} pufferlib_market/profile_training.py \
    --duration 60 \
    --output-dir "${PROFILES_DIR}"

T1=$(date +%s)
echo "[1/2] Pufferlib profiler done in $((T1 - T0))s"
echo ""

# ── 2. Binanceneural profiler ─────────────────────────────────────
echo "[2/2] Running binanceneural profiler (2 epochs) ..."
T0=$(date +%s)

${PYTHON} binanceneural/profile_training.py \
    --symbols AAPL,NVDA,DBX \
    --epochs 2 \
    --output-dir "${PROFILES_DIR}"

T1=$(date +%s)
echo "[2/2] Neural profiler done in $((T1 - T0))s"
echo ""

# ── Summary ───────────────────────────────────────────────────────
echo "============================================================"
echo "  Summary of generated profiles"
echo "============================================================"
for f in \
    "${PROFILES_DIR}/pufferlib_cuda_trace.json" \
    "${PROFILES_DIR}/pufferlib_report.md" \
    "${PROFILES_DIR}/pufferlib_flamegraph.svg" \
    "${PROFILES_DIR}/neural_cuda_trace.json" \
    "${PROFILES_DIR}/neural_report.md"
do
    if [ -f "${f}" ]; then
        SIZE=$(du -sh "${f}" | cut -f1)
        echo "  OK  ${SIZE}  ${f}"
    else
        echo "  --  (not generated)  ${f}"
    fi
done
echo ""

# Print first section of each report if it exists
for report in \
    "${PROFILES_DIR}/pufferlib_report.md" \
    "${PROFILES_DIR}/neural_report.md"
do
    if [ -f "${report}" ]; then
        echo "--- $(basename ${report}) ---"
        head -30 "${report}"
        echo ""
    fi
done

echo "Done. Open Chrome trace at chrome://tracing"
