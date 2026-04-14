#!/usr/bin/env bash
# Train crypto30 daily PPO with session-shift augmented data (local machine, CPU).
# Adapted from train_crypto30_sessaug.sh for this machine's paths.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

TRAIN_BIN="pufferlib_market/data/crypto30_daily_sess_aug_train.bin"
VAL_BIN="pufferlib_market/data/crypto30_daily_val.bin"
CKPT_BASE="pufferlib_market/checkpoints/crypto30_sessaug"

test -f "$TRAIN_BIN" || { echo "ERROR: $TRAIN_BIN missing"; exit 1; }
test -f "$VAL_BIN"   || { echo "ERROR: $VAL_BIN missing"; exit 1; }

run_one() {
    local name=$1 seed=$2 hidden=$3 wd=$4 rscale=$5 rclip=$6 extra="${7:-}"
    local tag="${name}_s${seed}"
    local ckdir="${CKPT_BASE}/${tag}"
    mkdir -p "$ckdir"
    echo "[$(date -u +%FT%TZ)] START $tag h=$hidden wd=$wd rs=$rscale rc=$rclip $extra"

    python -u -m pufferlib_market.train \
        --data-path "$TRAIN_BIN" \
        --val-data-path "$VAL_BIN" \
        --total-timesteps 15000000 \
        --max-steps 365 \
        --trade-penalty 0.02 \
        --hidden-size "$hidden" \
        --anneal-lr \
        --disable-shorts \
        --num-envs 32 \
        --periods-per-year 365.0 \
        --decision-lag 2 \
        --fee-rate 0.001 \
        --reward-scale "$rscale" \
        --reward-clip "$rclip" \
        --ent-coef 0.01 \
        --weight-decay "$wd" \
        --seed "$seed" \
        --val-eval-windows 50 \
        --val-decision-lag 2 \
        --cpu \
        --checkpoint-dir "$ckdir" $extra > "${ckdir}/train.log" 2>&1 || true

    echo "[$(date -u +%FT%TZ)] DONE training $tag (exit=$?)"

    # Evaluate
    for pt in val_best best final; do
        local ckpt="${ckdir}/${pt}.pt"
        [ -f "$ckpt" ] || continue
        echo "[$(date -u +%FT%TZ)] eval $tag $pt"
        python -m pufferlib_market.evaluate_holdout \
            --checkpoint "$ckpt" \
            --data-path "$VAL_BIN" \
            --eval-hours 90 --n-windows 50 \
            --fee-rate 0.001 --fill-buffer-bps 5.0 \
            --decision-lag 2 --deterministic --no-early-stop \
            --out "${ckdir}/${pt}_eval90d.json" 2>&1 | tail -5
    done
}

mkdir -p "$CKPT_BASE"

echo "=== crypto30 sessaug training (CPU) ==="

# Config 1: Baseline (same as prod winning config)
run_one "base" 1 256 0.005 10.0 5.0 &
run_one "base" 2 256 0.005 10.0 5.0 &

# Config 2: Higher weight decay
run_one "wd01" 1 256 0.01 10.0 5.0 &

wait
echo "=== Batch 1 done ==="

# Config 3: Lower reward scale (crypto30_daily used 1.0/3.0)
run_one "rs1" 1 256 0.005 1.0 3.0 &

# Config 4: Cosine LR
run_one "cosine" 1 256 0.005 10.0 5.0 "--lr-schedule cosine" &

# Config 5: Larger hidden
run_one "h384" 1 384 0.005 10.0 5.0 &

wait
echo "=== Batch 2 done ==="

# Config 6: Per-sym norm
run_one "psnorm" 1 256 0.005 10.0 5.0 "--per-sym-norm" &

# Extra seeds of best baseline
run_one "base" 3 256 0.005 10.0 5.0 &

wait
echo "=== All done ==="

# Print eval results
echo ""
echo "=== RESULTS ==="
for d in "$CKPT_BASE"/*/; do
    tag=$(basename "$d")
    for f in "$d"/*_eval90d.json; do
        [ -f "$f" ] || continue
        pt=$(basename "$f" _eval90d.json)
        stats=$(python3 -c "
import json, sys
d = json.load(open('$f'))
med = d.get('median_total_return', 0) * 100
sort = d.get('median_sortino', 0)
p10 = d.get('p10_total_return', 0) * 100
print(f'{med:+.2f}% sort={sort:.2f} p10={p10:+.2f}%')
" 2>/dev/null || echo "parse error")
        echo "  $tag/$pt: $stats"
    done
done
