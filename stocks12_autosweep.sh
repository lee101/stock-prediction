#!/bin/bash
# Autonomous stocks12 daily PPO seed sweep + eval loop
# Usage: nohup bash scripts/stocks12_autosweep.sh > /tmp/stocks12_autosweep.log 2>&1 &

set -e
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

TRAIN_DATA="pufferlib_market/data/stocks12_daily_train.bin"
VAL_DATA="pufferlib_market/data/stocks12_daily_val.bin"
CKPT_ROOT="pufferlib_market/checkpoints/stocks12_new_seeds"
LOG="pufferlib_market/stocks12_seed_sweep_leaderboard.csv"
TOTAL_TIMESTEPS=15000000
HIDDEN_SIZE=1024
NUM_ENVS=128
MAX_STEPS=252

echo "timestamp,seed,med_90d,p10_90d,worst_90d,neg_of_50,med_sortino,checkpoint" > $LOG

eval_seed() {
    local seed=$1
    local ckpt="$CKPT_ROOT/tp05_s${seed}/best.pt"
    local out="/tmp/eval_stocks12_s${seed}.json"

    if [ ! -f "$ckpt" ]; then
        echo "  s${seed}: no checkpoint, skip"
        return
    fi

    python -m pufferlib_market.evaluate_holdout \
        --checkpoint "$ckpt" \
        --data-path "$VAL_DATA" \
        --eval-hours 90 \
        --n-windows 50 \
        --seed 42 \
        --fee-rate 0.001 \
        --fill-buffer-bps 5.0 \
        --deterministic \
        --no-early-stop \
        --out "$out" > /dev/null 2>&1

    if [ ! -f "$out" ]; then
        echo "  s${seed}: eval failed"
        return
    fi

    stats=$(python3 -c "
import json, sys
d=json.load(open('$out'))
s=d['summary']
rets=[w['total_return'] for w in d['windows']]
neg=sum(1 for r in rets if r<0)
worst=min(rets)*100
print(f\"{s['median_total_return']*100:.2f},{s['p10_total_return']*100:.2f},{worst:.2f},{neg},{s['median_sortino']:.2f}\")
" 2>/dev/null)

    if [ -z "$stats" ]; then
        echo "  s${seed}: parse failed"
        return
    fi

    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "$ts,$seed,$stats,$ckpt" >> $LOG

    neg=$(echo "$stats" | cut -d, -f4)
    med=$(echo "$stats" | cut -d, -f1)
    p10=$(echo "$stats" | cut -d, -f2)
    echo "  s${seed}: med=${med}%  p10=${p10}%  neg=${neg}/50$([ '$neg' = '0' ] && echo ' *** ZERO-NEG ***' || true)"
}

train_seeds() {
    local seeds=("$@")
    for seed in "${seeds[@]}"; do
        python -u -m pufferlib_market.train \
            --data-path "$TRAIN_DATA" \
            --total-timesteps $TOTAL_TIMESTEPS \
            --trade-penalty 0.05 \
            --hidden-size $HIDDEN_SIZE \
            --anneal-lr \
            --num-envs $NUM_ENVS \
            --seed $seed \
            --max-steps $MAX_STEPS \
            --checkpoint-dir "$CKPT_ROOT/tp05_s${seed}" \
            > "/tmp/train_stocks12_s${seed}.log" 2>&1 &
        echo "  Started seed $seed (PID $!)"
    done
    echo "  Waiting for training to complete..."
    wait
    echo "  Training batch done"
}

# ---- Batch 1 already running (51-59), just wait and eval ----
echo "=== Waiting for batch 1 (seeds 51-59) to finish ==="
while pgrep -f "tp05_s5[0-9]/best" > /dev/null 2>&1; do sleep 10; done
# Also wait for any pufferlib train processes on s5x
sleep 5

echo "=== Evaluating batch 1 (seeds 51-59) ==="
for seed in 51 52 53 54 55 56 57 58 59; do
    eval_seed $seed
done

# ---- Batch 2: seeds 60-67 ----
echo ""
echo "=== Training batch 2 (seeds 60-67) ==="
train_seeds 60 61 62 63 64 65 66 67
echo "=== Evaluating batch 2 ==="
for seed in 60 61 62 63 64 65 66 67; do eval_seed $seed; done

# ---- Batch 3: seeds 70-77 ----
echo ""
echo "=== Training batch 3 (seeds 70-77) ==="
train_seeds 70 71 72 73 74 75 76 77
echo "=== Evaluating batch 3 ==="
for seed in 70 71 72 73 74 75 76 77; do eval_seed $seed; done

# ---- Batch 4: seeds 80-87 ----
echo ""
echo "=== Training batch 4 (seeds 80-87) ==="
train_seeds 80 81 82 83 84 85 86 87
echo "=== Evaluating batch 4 ==="
for seed in 80 81 82 83 84 85 86 87; do eval_seed $seed; done

# ---- Final summary ----
echo ""
echo "=== FINAL LEADERBOARD ==="
cat $LOG | grep ",0," | sort -t, -k3 -rn | head -20
echo ""
echo "Full leaderboard: $LOG"
