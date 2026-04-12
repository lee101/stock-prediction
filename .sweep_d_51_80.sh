#!/bin/bash
# D variant (muon) seeds 51-80 for stocks17 sweep
export TMPDIR=/nvme0n1-disk/code/stock-prediction/.tmp_train
mkdir -p "$TMPDIR"
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

CKPT_ROOT="pufferlib_market/checkpoints/stocks17_sweep/D_muon"
VAL="pufferlib_market/data/stocks17_augmented_val.bin"
TRAIN="pufferlib_market/data/stocks17_augmented_train.bin"
LOG="$CKPT_ROOT/leaderboard.csv"
mkdir -p "$CKPT_ROOT"

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
import json, sys, datetime
d = json.load(open(sys.argv[1]))
med = d.get("median_total_return", 0)*100
p10 = d.get("p10_total_return", 0)*100
worst = d.get("worst_window", {}).get("total_return", 0)*100
neg = d.get("negative_windows", -1)
sort = d.get("median_sortino", 0)
ts = datetime.datetime.utcnow().isoformat()
row = f"{ts},D,{sys.argv[3]},{med:.2f},{p10:.2f},{worst:.2f},{neg},{sort:.2f},{sys.argv[4]}"
print(f"  s{sys.argv[3]}: med={med:.2f}% p10={p10:.2f}% neg={neg}/50 sort={sort:.2f}")
open(sys.argv[2], "a").write(row + "\n")
PY
}

echo "=== D_muon seeds 51-80 stocks17 ==="
echo "    train=$TRAIN"

for seed in $(seq 51 80); do
    dir="$CKPT_ROOT/s${seed}"
    if [ -f "$dir/eval_lag2.json" ]; then
        echo "[$(date -u +%FT%TZ)] D s${seed}: already done, skip"
        continue
    fi
    mkdir -p "$dir"
    echo "[$(date -u +%FT%TZ)] D_muon seed $seed"
    python -u -m pufferlib_market.train \
        --data-path "$TRAIN" \
        --val-data-path "$VAL" \
        --total-timesteps 15000000 \
        --max-steps 252 \
        --trade-penalty 0.05 \
        --hidden-size 1024 \
        --anneal-lr \
        --disable-shorts \
        --optimizer muon \
        --num-envs 128 \
        --seed "$seed" \
        --checkpoint-dir "$dir" \
        > "$dir/train.log" 2>&1
    eval_seed "$seed"
done

echo
echo "=== D LEADERBOARD top 10 ==="
sort -t, -k4 -rn "$LOG" 2>/dev/null | head -10
