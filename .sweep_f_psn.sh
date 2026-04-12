#!/bin/bash
# Variant F: per-sym-norm + tp=0.02 for stocks17
export TMPDIR=/nvme0n1-disk/code/stock-prediction/.tmp_train
mkdir -p "$TMPDIR"
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

CKPT_ROOT="pufferlib_market/checkpoints/stocks17_sweep/F_psn_lotp"
VAL="pufferlib_market/data/stocks17_augmented_val.bin"
TRAIN="pufferlib_market/data/stocks17_augmented_train.bin"
LOG="$CKPT_ROOT/leaderboard.csv"
mkdir -p "$CKPT_ROOT"
echo "timestamp,variant,seed,med_pct,p10_pct,worst_pct,neg_count,med_sortino,checkpoint" > "$LOG"

eval_seed() {
    local seed=$1
    local dir="$CKPT_ROOT/s${seed}"
    local ckpt="$dir/val_best.pt"
    [ -f "$ckpt" ] || ckpt="$dir/best.pt"
    [ -f "$ckpt" ] || { echo "  s${seed}: no checkpoint"; return; }
    local out="$dir/eval_lag2.json"
    python -m pufferlib_market.evaluate_holdout \
        --checkpoint "$ckpt" --data-path "$VAL" \
        --eval-hours 60 --n-windows 50 --fee-rate 0.001 \
        --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop \
        > "$out" 2>/dev/null || { echo "  s${seed}: eval failed"; return; }
    python3 - "$out" "$LOG" "$seed" "$ckpt" << 'PY'
import json, sys
d = json.load(open(sys.argv[1]))
med = d.get("median_total_return", 0)*100
p10 = d.get("p10_total_return", 0)*100
worst = d.get("worst_window", {}).get("total_return", 0)*100
neg = d.get("negative_windows", -1)
sort = d.get("median_sortino", 0)
ts = __import__("datetime").datetime.utcnow().isoformat()
row = f"{ts},F,{sys.argv[3]},{med:.2f},{p10:.2f},{worst:.2f},{neg},{sort:.2f},{sys.argv[4]}"
print(f"  s{sys.argv[3]}: med={med:.2f}% p10={p10:.2f}% neg={neg}/50")
open(sys.argv[2], "a").write(row + "\n")
PY
}

echo "=== Variant F (per-sym-norm) stocks17: tp=0.02, adamw, --per-sym-norm ==="
echo "    train=$TRAIN"
echo

for seed in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    dir="$CKPT_ROOT/s${seed}"
    if [ -f "$dir/eval_lag2.json" ]; then
        echo "[$(date -u +%FT%TZ)] F s${seed}: already done, skip"
        continue
    fi
    mkdir -p "$dir"
    echo "[$(date -u +%FT%TZ)] F_psn_lotp seed $seed"
    python -u -m pufferlib_market.train \
        --data-path "$TRAIN" \
        --val-data-path "$VAL" \
        --total-timesteps 15000000 \
        --max-steps 252 \
        --trade-penalty 0.02 \
        --hidden-size 1024 \
        --anneal-lr \
        --disable-shorts \
        --per-sym-norm \
        --val-eval-windows 50 \
        --num-envs 128 \
        --seed "$seed" \
        --checkpoint-dir "$dir" \
        > "$dir/train.log" 2>&1
    eval_seed "$seed"
done

echo
echo "=== F LEADERBOARD ==="
sort -t, -k4 -rn "$LOG" | head -10
