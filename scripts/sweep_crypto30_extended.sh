#!/usr/bin/env bash
# Extended crypto30 sweep: seeds 24-50 with the proven recipe.
# Goal: find 4-5 more positive seeds to make a bigger, more robust ensemble.
# Current: 4/23 positive = 17% hit rate -> expect ~5 winners from 27 new seeds.
set -euo pipefail

REPO="/media/lee/pcd/code/stock"
source "$REPO/.venv313/bin/activate"
cd "$REPO"

TRAIN_BIN="$REPO/pufferlib_market/data/crypto30_daily_aug_train.bin"
VAL_BIN="$REPO/pufferlib_market/data/crypto30_daily_val.bin"
CKPT_BASE="$REPO/pufferlib_market/checkpoints/crypto30_gpu_extended"

run_one() {
    local seed=$1
    local tag="s${seed}"
    local ckdir="${CKPT_BASE}/${tag}"
    mkdir -p "$ckdir"

    if [ -f "${ckdir}/final.pt" ]; then
        echo "=== Seed $seed already trained, skipping ==="
    else
        echo "=== Training seed $seed ==="
        python -u -m pufferlib_market.train \
            --data-path "$TRAIN_BIN" \
            --val-data-path "$VAL_BIN" \
            --total-timesteps 15000000 \
            --max-steps 365 --trade-penalty 0.02 \
            --hidden-size 256 --anneal-lr --disable-shorts \
            --num-envs 32 --periods-per-year 365.0 \
            --decision-lag 2 --fee-rate 0.001 \
            --reward-scale 10.0 --reward-clip 5.0 \
            --ent-coef 0.01 --weight-decay 0.005 \
            --seed "$seed" --val-eval-windows 50 \
            --checkpoint-dir "$ckdir" > "${ckdir}/train.log" 2>&1 || true
    fi

    echo "=== Evaluating seed $seed ==="
    for pt in val_best final; do
        local ckpt="${ckdir}/${pt}.pt"
        local out_json="${ckdir}/${pt}_eval90d.json"
        [ -f "$ckpt" ] || continue
        [ -f "$out_json" ] && { echo "  ${pt} eval cached"; continue; }
        python -m pufferlib_market.evaluate_holdout \
            --checkpoint "$ckpt" \
            --data-path "$VAL_BIN" \
            --eval-hours 90 --n-windows 50 \
            --fee-rate 0.001 --fill-buffer-bps 5.0 \
            --decision-lag 2 --deterministic --no-early-stop \
            --seed 99 \
            --out "$out_json" 2>&1 | tail -3
    done
}

for seed in $(seq 24 50); do
    run_one $seed
done

echo ""
echo "=== Summary ==="
for seed in $(seq 24 50); do
    json="${CKPT_BASE}/s${seed}/val_best_eval90d.json"
    [ -f "$json" ] || continue
    med=$(grep median_total_return "$json" | head -1 | awk -F: '{print $2}' | tr -d ' ,')
    sort=$(grep median_sortino "$json" | head -1 | awk -F: '{print $2}' | tr -d ' ,')
    echo "s${seed}: med=${med} sort=${sort}"
done
echo "=== Done ==="
