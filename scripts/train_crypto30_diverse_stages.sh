#!/usr/bin/env bash
# Train crypto30 daily PPO with ORIGINAL data (matching prod config) but new seeds.
# Key insight: prod ensemble diversity comes from different convergence stages.
# Save many periodic checkpoints for later ensemble selection.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

# Use ORIGINAL augmented data (same as prod models)
TRAIN_BIN="pufferlib_market/data/crypto30_daily_aug_train.bin"
VAL_BIN="pufferlib_market/data/crypto30_daily_val.bin"
CKPT_BASE="pufferlib_market/checkpoints/crypto30_stages"

test -f "$TRAIN_BIN" || { echo "ERROR: $TRAIN_BIN missing"; exit 1; }
test -f "$VAL_BIN"   || { echo "ERROR: $VAL_BIN missing"; exit 1; }

run_one() {
    local name=$1 seed=$2 rscale=$3 rclip=$4 extra="${5:-}"
    local tag="${name}_s${seed}"
    local ckdir="${CKPT_BASE}/${tag}"
    mkdir -p "$ckdir"
    echo "[$(date -u +%FT%TZ)] START $tag rs=$rscale rc=$rclip $extra"

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
        --reward-scale "$rscale" \
        --reward-clip "$rclip" \
        --ent-coef 0.01 \
        --weight-decay 0.005 \
        --seed "$seed" \
        --val-eval-windows 50 \
        --val-decision-lag 2 \
        --cpu \
        --save-every 25 \
        --max-periodic-checkpoints 20 \
        --checkpoint-dir "$ckdir" $extra > "${ckdir}/train.log" 2>&1 || true

    echo "[$(date -u +%FT%TZ)] DONE $tag"
}

mkdir -p "$CKPT_BASE"
echo "=== crypto30 diverse stages (original data, save-every-25, 20 periodic ckpts) ==="

# Match prod config (reward_scale=1.0, reward_clip=3.0)
# New seeds: 10, 20, 30, 40, 50 (prod used 2,19,21,23)
run_one "orig" 10 1.0 3.0 &
run_one "orig" 20 1.0 3.0 &
run_one "orig" 30 1.0 3.0 &

wait
echo "=== Batch 1 done ==="

run_one "orig" 40 1.0 3.0 &
run_one "orig" 50 1.0 3.0 &
# Also try with sessaug data for comparison
run_one "sessaug_orig_config" 10 1.0 3.0 "--data-path pufferlib_market/data/crypto30_daily_sess_aug_train.bin" &

wait
echo "=== All done ==="

echo ""
echo "=== CHECKPOINTS ==="
for d in "$CKPT_BASE"/*/; do
    tag=$(basename "$d")
    n_ckpts=$(ls "$d"/*.pt 2>/dev/null | wc -l)
    echo "  $tag: $n_ckpts checkpoints"
done
echo ""
echo "Run scripts/crypto30_stage_ensemble.py to find best ensemble combinations."
