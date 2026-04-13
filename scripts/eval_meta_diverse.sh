#!/bin/bash
# Evaluate diverse models with meta-selector backtest
# Run on daisy after train_meta_diverse.sh completes
set +e

DATA=trainingdata/fresh
BASE_DIR=pufferlib_market/checkpoints/meta_diverse
OUT_DIR=analysis/meta_diverse_eval

source .venv/bin/activate

mkdir -p "$OUT_DIR"

# Collect all val_best.pt checkpoints
CKPTS=""
for d in "$BASE_DIR"/*/; do
    name=$(basename "$d")
    cp="$d/val_best.pt"
    if [ ! -f "$cp" ]; then
        cp="$d/best.pt"
    fi
    if [ ! -f "$cp" ]; then
        cp="$d/final.pt"
    fi
    if [ -f "$cp" ]; then
        CKPTS="$CKPTS $cp"
        echo "Found: $cp"
    else
        echo "MISSING: $name"
    fi
done

N=$(echo $CKPTS | wc -w)
echo "Total checkpoints: $N"

if [ "$N" -lt 3 ]; then
    echo "ERROR: Need at least 3 models for meta-selection"
    exit 1
fi

# Copy checkpoints to a flat directory for the meta backtest
META_DIR=pufferlib_market/meta_diverse_ensemble
mkdir -p "$META_DIR"
i=0
for cp in $CKPTS; do
    name=$(basename $(dirname "$cp"))
    cp "$cp" "$META_DIR/${name}.pt"
    i=$((i + 1))
done
echo "Copied $i checkpoints to $META_DIR"

# Run meta-selector backtest with OOS data
echo "=== OOS Sweep (95d) ==="
for lb in 3 5 7; do
    for k in 1 2 3; do
        python scripts/meta_strategy_backtest.py \
            --data-dir "$DATA" \
            --eval-days 95 \
            --ensemble-dir "$META_DIR" \
            --top-k $k --lookback $lb \
            --slippage-bps 5 \
            --device cuda \
            --out "$OUT_DIR/oos_lb${lb}_k${k}.json" 2>&1 | grep "Meta momentum"
    done
done

# Slippage sweep at best config
echo "=== Slippage Sweep (lb=3 k=1) ==="
for slip in 0 5 10 20; do
    python scripts/meta_strategy_backtest.py \
        --data-dir "$DATA" \
        --eval-days 95 \
        --ensemble-dir "$META_DIR" \
        --top-k 1 --lookback 3 \
        --slippage-bps $slip \
        --device cuda \
        --out "$OUT_DIR/slip_${slip}bps.json" 2>&1 | grep "Meta momentum"
done

# Full history
echo "=== Full History ==="
python scripts/meta_strategy_backtest.py \
    --data-dir "$DATA" \
    --ensemble-dir "$META_DIR" \
    --top-k 1 --lookback 3 \
    --slippage-bps 5 \
    --device cuda \
    --out "$OUT_DIR/full_history.json" 2>&1 | grep -E "Meta momentum|Best model|Ensemble"

echo "=== Done ==="
echo "Results in $OUT_DIR/"
