#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

TRAIN_BIN=pufferlib_market/data/crypto30_daily_aug_train.bin
VAL_BIN=pufferlib_market/data/crypto30_daily_val.bin
CKPT_BASE=pufferlib_market/checkpoints/crypto30_gpu_wd005_20seed
LEADER=${CKPT_BASE}/leaderboard.csv
mkdir -p "$CKPT_BASE"

echo "timestamp,config,seed,med_pct,sort,neg,checkpoint" > "$LEADER"

echo "=== crypto30 GPU wd005 20-seed sweep (15M steps) ==="

run_one() {
    local seed=$1
    local tag="s${seed}"
    local ckdir="${CKPT_BASE}/${tag}"
    mkdir -p "$ckdir"
    echo "[$(date -u +%FT%TZ)] ${tag} seed=${seed}"

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
        --checkpoint-dir "$ckdir" \
        > "${ckdir}/train.log" 2>&1 || true

    echo "[$(date -u +%FT%TZ)] done ${tag} (exit=$?)"

    # independent eval with seed=99
    for pt in val_best final; do
        [ -f "${ckdir}/${pt}.pt" ] || continue
        local result
        result=$(python -m pufferlib_market.evaluate_holdout \
            --checkpoint "${ckdir}/${pt}.pt" \
            --data-path "$VAL_BIN" \
            --eval-hours 90 --n-windows 50 \
            --fee-rate 0.001 --fill-buffer-bps 5.0 \
            --decision-lag 2 --deterministic --no-early-stop \
            --seed 99 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
med=d['median_total_return']*100
sort=d['median_sortino']
neg=d['negative_windows']
print(f'{med:.2f},{sort:.2f},{neg}')
" 2>/dev/null) || result="ERR,ERR,ERR"
        local med=$(echo "$result" | cut -d, -f1)
        local sort=$(echo "$result" | cut -d, -f2)
        local neg=$(echo "$result" | cut -d, -f3)
        echo "  ${tag} ${pt}: med=${med}% sort=${sort} neg=${neg}/50"
        echo "$(date -u +%FT%TZ),wd005_${pt},${seed},${med},${sort},${neg},${ckdir}/${pt}.pt" >> "$LEADER"
    done
}

# 20 seeds (skip s1-3 already done in original sweep)
for seed in $(seq 4 23); do
    run_one "$seed"
done

echo "=== SWEEP COMPLETE ==="
echo "Leaderboard: $LEADER"
cat "$LEADER" | sort -t, -k4 -rn | head -20
