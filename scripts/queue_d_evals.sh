#!/bin/bash
source /nvme0n1-disk/code/stock-prediction/.venv/bin/activate
cd /nvme0n1-disk/code/stock-prediction

EVAL_CMD="python -m pufferlib_market.evaluate_holdout --data-path pufferlib_market/data/screened32_single_offset_val_full.bin --eval-hours 50 --n-windows 100 --fee-rate 0.001 --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop"

for s in 72 73 75 76; do
    while pgrep -f "evaluate_holdout" > /dev/null; do
        sleep 30
    done

    if [ ! -s "pufferlib_market/checkpoints/screened32_sweep/D/s${s}/eval_full.json" ]; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Evaluating D/s${s}..."
        $EVAL_CMD \
            --checkpoint "pufferlib_market/checkpoints/screened32_sweep/D/s${s}/val_best.pt" \
            --output "pufferlib_market/checkpoints/screened32_sweep/D/s${s}/eval_full.json"
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] D/s${s} eval done"
    fi
done
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] All D evals complete"
