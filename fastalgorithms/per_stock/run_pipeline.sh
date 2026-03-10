#!/usr/bin/env bash
# End-to-end per-stock pipeline: train, evaluate, meta-select, compare.
#
# Usage:
#   ./fastalgorithms/per_stock/run_pipeline.sh                    # full run
#   SYMBOLS="NVDA,PLTR" EPOCHS=5 ./fastalgorithms/per_stock/run_pipeline.sh  # quick test
#   SKIP_TRAIN=1 ./fastalgorithms/per_stock/run_pipeline.sh       # reuse existing checkpoints
set -euo pipefail

PYTHON="${PYTHON:-.venv313/bin/python -u}"
BASE_DIR="fastalgorithms/per_stock"
CKPT_ROOT="${BASE_DIR}/checkpoints"

SYMBOLS="${SYMBOLS:-NVDA,PLTR,GOOG,DBX,TRIP,MTCH}"
HOLDOUT_DAYS="${HOLDOUT_DAYS:-90}"
EPOCHS="${EPOCHS:-20}"

echo "=========================================="
echo " Per-Stock Meta-Selector Pipeline"
echo "=========================================="
echo " Symbols: ${SYMBOLS}"
echo " Holdout: ${HOLDOUT_DAYS} days"
echo " Epochs: ${EPOCHS}"
echo "=========================================="

# Phase 1: Train per-stock models
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
    echo ""
    echo "=== Phase 1: Per-Stock Training ==="
    SYMBOLS="$SYMBOLS" EPOCHS="$EPOCHS" bash ${BASE_DIR}/run_sweep.sh
else
    echo ""
    echo "=== Phase 1: SKIPPED (SKIP_TRAIN=1) ==="
fi

# Phase 2: Evaluate all models and select best per stock
echo ""
echo "=== Phase 2: Evaluate & Select Best Per-Stock ==="
$PYTHON -m fastalgorithms.per_stock.eval_per_stock \
    --select-best \
    --output-equity \
    --symbols "$SYMBOLS" \
    --holdout-days "$HOLDOUT_DAYS" \
    --checkpoint-root "$CKPT_ROOT"

# Phase 3: Compare all strategies
echo ""
echo "=== Phase 3: Strategy Comparison ==="
$PYTHON -m fastalgorithms.per_stock.compare_strategies \
    --symbols "$SYMBOLS" \
    --holdout-days "$HOLDOUT_DAYS" \
    --checkpoint-root "$CKPT_ROOT"

echo ""
echo "=== Pipeline Complete ==="
echo "Results: ${CKPT_ROOT}/comparison_${HOLDOUT_DAYS}d.json"
