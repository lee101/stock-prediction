#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CRL_DIR="$(dirname "$SCRIPT_DIR")"
cd "$CRL_DIR"

TRAIN_DATA="${TRAIN_DATA:-data/mixed23_latest_train_20260320.bin}"
VAL_DATA="${VAL_DATA:-data/mixed23_latest_val_20250922_20260320.bin}"
TIME_BUDGET="${TIME_BUDGET:-300}"
MAX_TRIALS="${MAX_TRIALS:-15}"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Train data not found: $TRAIN_DATA"
    echo "Available .bin files:"
    ls data/*.bin 2>/dev/null || echo "  none"
    exit 1
fi

echo "=== CRL Autoresearch ==="
echo "Train: $TRAIN_DATA"
echo "Val: $VAL_DATA"
echo "Budget: ${TIME_BUDGET}s/trial, max $MAX_TRIALS trials/round"

./crl_autoresearch \
    --train-data "$TRAIN_DATA" \
    --val-data "$VAL_DATA" \
    --experiments experiments.json \
    --checkpoint-root checkpoints \
    --leaderboard leaderboard.csv \
    --time-budget "$TIME_BUDGET" \
    --max-trials "$MAX_TRIALS" \
    --gpu \
    "$@"
