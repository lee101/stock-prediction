#!/usr/bin/env bash
# Crypto30 sessaug batch 2: quick-win improvements (cosine LR, entropy annealing, obs-norm, clip-vloss)
# Run AFTER batch 1 finishes to avoid overloading CPU.
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
        --disable-shorts \
        --num-envs 32 \
        --periods-per-year 365.0 \
        --decision-lag 2 \
        --fee-rate 0.001 \
        --reward-scale "$rscale" \
        --reward-clip "$rclip" \
        --weight-decay "$wd" \
        --seed "$seed" \
        --val-eval-windows 50 \
        --val-decision-lag 2 \
        --cpu \
        --save-every 100 \
        --max-periodic-checkpoints 10 \
        --checkpoint-dir "$ckdir" $extra > "${ckdir}/train.log" 2>&1 || true

    echo "[$(date -u +%FT%TZ)] DONE training $tag (exit=$?)"

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

echo "=== crypto30 sessaug batch 2: improved configs ==="

# Cosine LR + entropy annealing (top priority combo)
run_one "cosine_ent" 1 256 0.005 10.0 5.0 "--lr-schedule cosine --anneal-ent --ent-coef 0.05 --ent-coef-end 0.001" &

# Obs-norm + cosine LR
run_one "obsnorm_cos" 1 256 0.005 10.0 5.0 "--lr-schedule cosine --obs-norm" &

# Clip-vloss + cosine LR + entropy annealing (all quick wins combined)
run_one "allwins" 1 256 0.005 10.0 5.0 "--lr-schedule cosine --anneal-ent --ent-coef 0.05 --ent-coef-end 0.001 --clip-vloss" &

wait
echo "=== Batch 2a done ==="

# Higher wd + all quick wins
run_one "allwins_wd01" 1 256 0.01 10.0 5.0 "--lr-schedule cosine --anneal-ent --ent-coef 0.05 --ent-coef-end 0.001 --clip-vloss" &

# Per-sym norm + cosine (alternative to obs-norm)
run_one "psnorm_cos" 1 256 0.005 10.0 5.0 "--lr-schedule cosine --per-sym-norm" &

# Advantage per_env + all wins
run_one "perenv" 1 256 0.005 10.0 5.0 "--lr-schedule cosine --anneal-ent --ent-coef 0.05 --ent-coef-end 0.001 --advantage-norm per_env" &

wait
echo "=== Batch 2b done ==="

# h384 + all wins
run_one "h384_allwins" 1 384 0.01 10.0 5.0 "--lr-schedule cosine --anneal-ent --ent-coef 0.05 --ent-coef-end 0.001 --clip-vloss" &

# Second seeds of best configs (to check if consistent)
run_one "cosine_ent" 2 256 0.005 10.0 5.0 "--lr-schedule cosine --anneal-ent --ent-coef 0.05 --ent-coef-end 0.001" &
run_one "allwins" 2 256 0.005 10.0 5.0 "--lr-schedule cosine --anneal-ent --ent-coef 0.05 --ent-coef-end 0.001 --clip-vloss" &

wait
echo "=== All batch 2 done ==="

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
neg = d.get('num_negative_windows', '?')
print(f'{med:+.2f}% sort={sort:.2f} p10={p10:+.2f}% neg={neg}')
" 2>/dev/null || echo "parse error")
        echo "  $tag/$pt: $stats"
    done
done
