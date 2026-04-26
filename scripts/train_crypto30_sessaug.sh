#!/usr/bin/env bash
# Train crypto30 daily PPO with session-shift augmented data.
# Same recipe as the winning config (h256, wd=0.005, 15M steps).
# Run on daisy GPU.
set -euo pipefail

REPO="/media/lee/pcd/code/stock"
VENV="$REPO/.venv313/bin/activate"
TRAIN_BIN="$REPO/pufferlib_market/data/crypto30_daily_sess_aug_train.bin"
VAL_BIN="$REPO/pufferlib_market/data/crypto30_daily_sess_aug_val.bin"
CKPT_BASE="$REPO/pufferlib_market/checkpoints/crypto30_sessaug"

# Always use original 30-symbol val for eval (session-aug val has 29 symbols)
EVAL_VAL_BIN="$REPO/pufferlib_market/data/crypto30_daily_val.bin"

# Fall back to non-session-augmented val if session-aug val doesn't exist
if [ ! -f "$VAL_BIN" ]; then
    VAL_BIN="$EVAL_VAL_BIN"
    echo "Using original val: $VAL_BIN"
fi

source "$VENV"

run_one() {
    local seed=$1
    local tag="s${seed}"
    local ckdir="${CKPT_BASE}/${tag}"
    mkdir -p "$ckdir"

    echo "=== Training seed $seed ==="
    python -u -m pufferlib_market.train \
        --data-path "$TRAIN_BIN" \
        --val-data-path "$VAL_BIN" \
        --total-timesteps 15000000 \
        --max-steps 365 \
        --trade-penalty 0.02 \
        --hidden-size 256 \
        --anneal-lr \
        --disable-shorts \
        --num-envs 32 \
        --periods-per-year 365.0 \
        --decision-lag 2 \
        --fee-rate 0.001 \
        --reward-scale 10.0 \
        --reward-clip 5.0 \
        --ent-coef 0.01 \
        --weight-decay 0.005 \
        --seed "$seed" \
        --val-eval-windows 50 \
        --checkpoint-dir "$ckdir" > "${ckdir}/train.log" 2>&1 || true

    echo "=== Evaluating seed $seed ==="
    for pt in val_best final; do
        local ckpt="${ckdir}/${pt}.pt"
        [ -f "$ckpt" ] || continue
        python -m pufferlib_market.evaluate_holdout \
            --checkpoint "$ckpt" \
            --data-path "$EVAL_VAL_BIN" \
            --eval-hours 90 --n-windows 50 \
            --fee-rate 0.001 --fill-buffer-bps 5.0 \
            --decision-lag 2 --deterministic --no-early-stop \
            --seed 99 \
            --out "${ckdir}/${pt}_eval90d.json" 2>&1 | tail -5
    done
}

# 5-seed pilot
for seed in 1 2 3 4 5; do
    run_one $seed
done

echo "=== All seeds done ==="
