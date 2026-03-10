#!/usr/bin/env bash
# Per-stock hyperparameter sweep: train individual models for each stock.
# Usage:
#   ./fastalgorithms/per_stock/run_sweep.sh
#   SYMBOLS="NVDA,PLTR" ./fastalgorithms/per_stock/run_sweep.sh  # subset
#   EPOCHS=5 ./fastalgorithms/per_stock/run_sweep.sh             # quick test
set -euo pipefail

PYTHON="${PYTHON:-.venv313/bin/python -u}"
SCRIPT="fastalgorithms/per_stock/train_per_stock.py"

# Symbols to train (override with SYMBOLS env var)
IFS=',' read -ra SYM_ARRAY <<< "${SYMBOLS:-NVDA,PLTR,GOOG,DBX,TRIP,MTCH}"

# Hyperparameter grid
RW_VALUES=(${RW_VALUES:-0.10 0.15 0.20})
WD_VALUES=(${WD_VALUES:-0.04 0.06})

# Fixed params
EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-42}"
SEQ="${SEQ:-48}"
HIDDEN="${HIDDEN:-512}"
LAYERS="${LAYERS:-6}"
HEADS="${HEADS:-8}"

total=$((${#SYM_ARRAY[@]} * ${#RW_VALUES[@]} * ${#WD_VALUES[@]}))
count=0

echo "=========================================="
echo " Per-Stock Hyperparameter Sweep"
echo "=========================================="
echo " Symbols: ${SYM_ARRAY[*]}"
echo " Return weights: ${RW_VALUES[*]}"
echo " Weight decays: ${WD_VALUES[*]}"
echo " Fixed: h${HIDDEN} ${LAYERS}L ${HEADS}H seq=${SEQ} ep=${EPOCHS} s=${SEED}"
echo " Total runs: ${total}"
echo "=========================================="
echo ""

for SYMBOL in "${SYM_ARRAY[@]}"; do
    for RW in "${RW_VALUES[@]}"; do
        for WD in "${WD_VALUES[@]}"; do
            count=$((count + 1))
            echo ""
            echo "=== [${count}/${total}] ${SYMBOL} rw=${RW} wd=${WD} ==="
            $PYTHON $SCRIPT \
                --symbol "$SYMBOL" \
                --return-weight "$RW" \
                --weight-decay "$WD" \
                --epochs "$EPOCHS" \
                --seed "$SEED" \
                --sequence-length "$SEQ" \
                --hidden-dim "$HIDDEN" \
                --num-layers "$LAYERS" \
                --num-heads "$HEADS" \
                --no-compile \
                || echo "FAILED: ${SYMBOL} rw=${RW} wd=${WD}"
        done
    done
done

echo ""
echo "=== Sweep Complete: ${count}/${total} runs ==="
echo "Checkpoints: fastalgorithms/per_stock/checkpoints/"
ls -d fastalgorithms/per_stock/checkpoints/*/ 2>/dev/null | head -20
