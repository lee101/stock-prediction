#!/bin/bash
# Deep stocks sweep: extended seed range, stocks20 universe, and trade_penalty variants
# Based on crypto70 lesson: need 200+ seeds to find champions
# Uses 35M steps (matching production) for better quality signal
#
# Phase A: stocks12 tp05 seeds 100-299 @ 35M steps
# Phase B: stocks20 tp05 seeds 1-80 @ 35M steps  
# Phase C: stocks12 tp03 seeds 1-60 @ 35M steps
# Phase D: stocks12 tp07 seeds 1-60 @ 35M steps
# Phase E: stocks12 anneal_ent tp05 seeds 1-60 @ 35M steps

set -e
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

TRAIN12="pufferlib_market/data/stocks12_daily_train.bin"
VAL12="pufferlib_market/data/stocks12_daily_val.bin"
TRAIN20="pufferlib_market/data/stocks20_daily_train.bin"
VAL20="pufferlib_market/data/stocks20_daily_val.bin"

CKPT_ROOT="pufferlib_market/checkpoints"
LOG_ROOT="pufferlib_market"
TOTAL_TIMESTEPS=35000000
HIDDEN_SIZE=1024
NUM_ENVS=128
MAX_STEPS=252
PARALLEL=6

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

eval_checkpoint() {
    local label="$1" ckpt="$2" val_data="$3" csv="$4"
    local out="/tmp/eval_${label}.json"
    [ ! -f "$ckpt" ] && log "  ${label}: no checkpoint, skip" && return
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
    med=$(echo "$stats" | cut -d, -f1); p10=$(echo "$stats" | cut -d, -f2); neg=$(echo "$stats" | cut -d, -f4)
    log "  ${label}: med=${med}%  p10=${p10}%  neg=${neg}/50"
}

train_batch() {
    local data="$1" ckpt_prefix="$2" tp="$3" extra_args="${4:-}"
    shift 4
    local seeds=("$@")
    for seed in "${seeds[@]}"; do
        local ckpt_dir="${CKPT_ROOT}/${ckpt_prefix}/s${seed}"
        if [ -f "${ckpt_dir}/best.pt" ]; then
            log "  s${seed}: exists, skip"
        else
            python -m pufferlib_market.train \
                --data-path "$data" --hidden-size $HIDDEN_SIZE \
                --num-envs $NUM_ENVS --total-timesteps $TOTAL_TIMESTEPS \
                --anneal-lr --trade-penalty "$tp" --fee-rate 0.001 \
                --max-steps $MAX_STEPS --seed "$seed" \
                --checkpoint-dir "$ckpt_dir" $extra_args \
                > "/tmp/train_$(basename $ckpt_prefix)_s${seed}.log" 2>&1 &
            log "  s${seed}: started (PID $!)"
        fi
    done
    wait
    log "  batch done"
}

run_phase() {
    local phase="$1" label="$2" ckpt_prefix="$3" data="$4" val="$5" tp="$6" extra_args="${7:-}"
    shift 7
    local seeds=("$@")
    local csv="${LOG_ROOT}/${label}_leaderboard.csv"
    [ ! -f "$csv" ] && echo "timestamp,label,med_90d,p10_90d,worst_90d,neg_of_50,med_sortino,checkpoint" > "$csv"
    log "=== Phase ${phase}: ${label} (${#seeds[@]} seeds, tp=${tp}, 35M steps) ==="
    for i in $(seq 0 $PARALLEL $((${#seeds[@]} - 1))); do
        batch=(); for j in $(seq 0 $((PARALLEL-1))); do
            idx=$((i+j)); [ $idx -lt ${#seeds[@]} ] && batch+=(${seeds[$idx]})
        done
        log "--- Training: ${batch[*]} ---"
        train_batch "$data" "$ckpt_prefix" "$tp" "$extra_args" "${batch[@]}"
        log "--- Evaluating: ${batch[*]} ---"
        for seed in "${batch[@]}"; do
            eval_checkpoint "${label}_s${seed}" "${CKPT_ROOT}/${ckpt_prefix}/s${seed}/best.pt" "$val" "$csv"
        done
    done
    log "Phase ${phase} done. Top results:"
    sort -t, -k3 -rn "$csv" | head -5
}

# Wait for current sweep (s70-87) to finish before hogging all GPU
log "Waiting for current autosweep to finish (s70-87 training)..."
while pgrep -f "stocks12_new_seeds" > /dev/null 2>&1; do sleep 30; done
log "Current sweep done. Starting deep sweep."

run_phase "A" "stocks12_tp05_bigseed" "stocks12_tp05_bigseed" "$TRAIN12" "$VAL12" "0.05" "" \
    $(seq 100 299)

run_phase "B" "stocks20_tp05" "stocks20_tp05" "$TRAIN20" "$VAL20" "0.05" "" \
    $(seq 1 80)

run_phase "C" "stocks12_tp03" "stocks12_tp03" "$TRAIN12" "$VAL12" "0.03" "" \
    $(seq 1 60)

run_phase "D" "stocks12_tp07" "stocks12_tp07" "$TRAIN12" "$VAL12" "0.07" "" \
    $(seq 1 60)

run_phase "E" "stocks12_anneal_ent" "stocks12_anneal_ent" "$TRAIN12" "$VAL12" "0.05" "--anneal-ent" \
    $(seq 1 60)

log "=== ALL PHASES COMPLETE ==="
log "0/50-neg champions across all phases:"
for csv in $LOG_ROOT/stocks12_tp05_bigseed_leaderboard.csv $LOG_ROOT/stocks20_tp05_leaderboard.csv $LOG_ROOT/stocks12_tp03_leaderboard.csv $LOG_ROOT/stocks12_tp07_leaderboard.csv $LOG_ROOT/stocks12_anneal_ent_leaderboard.csv; do
    [ -f "$csv" ] && awk -F, '$6=="0"' "$csv"
done | sort -t, -k3 -rn
