#!/bin/bash
# Extended-data stocks sweep: train on 7.1yr data (vs 5.2yr standard)
# More diverse market conditions (includes 2019 which had market correction)
# Evaluate on SAME val data as production for fair comparison
#
# Usage: nohup bash scripts/stocks_extended_sweep.sh > /tmp/stocks_extended_sweep.log 2>&1 &

set -e
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

TRAIN_EXT="pufferlib_market/data/stocks12_extended_train.bin"
VAL_STD="pufferlib_market/data/stocks12_daily_val.bin"  # same val as production for fair comparison
TRAIN20="pufferlib_market/data/stocks20_daily_train.bin"
VAL20="pufferlib_market/data/stocks20_daily_val.bin"

CKPT_ROOT="pufferlib_market/checkpoints"
TOTAL_TIMESTEPS=35000000
HIDDEN_SIZE=1024
NUM_ENVS=128
MAX_STEPS=252
PARALLEL=6

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

eval_checkpoint() {
    local label="$1" ckpt="$2" val_data="$3" csv="$4"
    local out="/tmp/eval_${label}.json"
    [ ! -f "$ckpt" ] && log "  ${label}: no checkpoint" && return
    python -m pufferlib_market.evaluate_holdout \
        --checkpoint "$ckpt" --data-path "$val_data" \
        --eval-hours 90 --n-windows 50 --seed 42 \
        --fee-rate 0.001 --fill-buffer-bps 5.0 \
        --deterministic --no-early-stop --out "$out" > /dev/null 2>&1
    [ ! -f "$out" ] && log "  ${label}: eval failed" && return
    stats=$(python3 -c "
import json
d=json.load(open('$out'))
s=d['summary']
rets=[w['total_return'] for w in d['windows']]
neg=sum(1 for r in rets if r<0)
worst=min(rets)*100
print(f\"{s['median_total_return']*100:.2f},{s['p10_total_return']*100:.2f},{worst:.2f},{neg},{s['median_sortino']:.2f}\")
" 2>/dev/null)
    [ -z "$stats" ] && log "  ${label}: parse failed" && return
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),${label},${stats},${ckpt}" >> "$csv"
    med=$(echo "$stats" | cut -d, -f1); neg=$(echo "$stats" | cut -d, -f4)
    log "  ${label}: med=${med}%  neg=${neg}/50"
}

# Wait for deep sweep Phase A to complete
log "Waiting for deep sweep to wind down (checking for active training)..."
while pgrep -f "stocks12_tp05_bigseed\|stocks20_tp05\|stocks12_tp03\|stocks12_tp07\|stocks12_anneal_ent\|stocks12_35m_retrain" > /dev/null 2>&1; do
    sleep 60
done
log "Deep sweep done. Starting extended-data sweep."

# Phase F: stocks12 extended training data, tp05, s1-100
CSV_F="pufferlib_market/stocks12_extended_tp05_leaderboard.csv"
[ ! -f "$CSV_F" ] && echo "timestamp,label,med_90d,p10_90d,worst_90d,neg_of_50,med_sortino,checkpoint" > "$CSV_F"

log "=== Phase F: stocks12 extended (7.1yr) tp05 seeds 1-100 @ 35M ==="
seeds_f=($(seq 1 100))
for i in $(seq 0 $PARALLEL $((${#seeds_f[@]} - 1))); do
    batch=(); for j in $(seq 0 $((PARALLEL-1))); do
        idx=$((i+j)); [ $idx -lt ${#seeds_f[@]} ] && batch+=(${seeds_f[$idx]})
    done
    log "--- Training extended s${batch[*]} ---"
    for seed in "${batch[@]}"; do
        ckpt_dir="${CKPT_ROOT}/stocks12_extended_tp05/s${seed}"
        [ -f "${ckpt_dir}/best.pt" ] && log "  s${seed}: exists, skip" && continue
        python -m pufferlib_market.train \
            --data-path "$TRAIN_EXT" --hidden-size $HIDDEN_SIZE \
            --num-envs $NUM_ENVS --total-timesteps $TOTAL_TIMESTEPS \
            --anneal-lr --trade-penalty 0.05 --fee-rate 0.001 \
            --max-steps $MAX_STEPS --seed "$seed" \
            --checkpoint-dir "$ckpt_dir" \
            > "/tmp/train_ext_s${seed}.log" 2>&1 &
        log "  s${seed}: started (PID $!)"
    done
    wait
    for seed in "${batch[@]}"; do
        eval_checkpoint "stocks12_ext_tp05_s${seed}" \
            "${CKPT_ROOT}/stocks12_extended_tp05/s${seed}/best.pt" \
            "$VAL_STD" "$CSV_F"
    done
done

log "Phase F done. Top results:"
sort -t, -k3 -rn "$CSV_F" | head -10
log "0/50-neg:"
awk -F, '$6=="0"' "$CSV_F" | sort -t, -k3 -rn
