#!/bin/bash
# 2-stage stocks sweep: quick 15M scan → selective 35M retrain of promising seeds
# Addresses overfitting: most seeds hurt at 35M, only scan for intrinsically good ones
#
# Stage 1: Scan s106-s600 at 15M steps (8 parallel)
#   - Covers s106-s299 (deep sweep parent killed, current batch s100-s105 runs to completion at 35M)
#   - Covers s300-s600 (original quickscan range)
#   - Skips seeds where best.pt already exists (e.g., s100-s105 bigseed dir)
# Stage 2: Retrain any seeds with med>10% and neg<15/50 at 35M steps
#
# Usage: nohup bash scripts/stocks_quickscan_sweep.sh > /tmp/stocks_quickscan.log 2>&1 &

set -e
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

TRAIN12="pufferlib_market/data/stocks12_daily_train.bin"
VAL12="pufferlib_market/data/stocks12_daily_val.bin"
CKPT_ROOT="pufferlib_market/checkpoints"
LOG_ROOT="pufferlib_market"
SCAN_STEPS=15000000
RETRAIN_STEPS=35000000
HIDDEN_SIZE=1024
NUM_ENVS=128
MAX_STEPS=252
PARALLEL=8  # more parallel since 15M is quick

# Promotion thresholds
MIN_MED=10.0    # >10% median return
MAX_NEG=15      # <15/50 negative windows

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

eval_checkpoint() {
    local label="$1" ckpt="$2" val_data="$3" csv="$4"
    local out="/tmp/eval_${label}.json"
    [ ! -f "$ckpt" ] && return 1
    python -m pufferlib_market.evaluate_holdout \
        --checkpoint "$ckpt" --data-path "$val_data" \
        --eval-hours 90 --n-windows 50 --seed 42 \
        --fee-rate 0.001 --fill-buffer-bps 5.0 \
        --deterministic --no-early-stop --out "$out" > /dev/null 2>&1
    [ ! -f "$out" ] && return 1
    stats=$(python3 -c "
import json
d=json.load(open('$out'))
s=d['summary']
rets=[w['total_return'] for w in d['windows']]
neg=sum(1 for r in rets if r<0)
worst=min(rets)*100
print(f\"{s['median_total_return']*100:.2f},{s['p10_total_return']*100:.2f},{worst:.2f},{neg},{s['median_sortino']:.2f}\")
" 2>/dev/null)
    [ -z "$stats" ] && return 1
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),${label},${stats},${ckpt}" >> "$csv"
    med=$(echo "$stats" | cut -d, -f1); neg=$(echo "$stats" | cut -d, -f4)
    log "  ${label}: med=${med}%  neg=${neg}/50"
    return 0
}

is_promising() {
    local ckpt="$1"
    local out="/tmp/promising_check_$(basename $(dirname $ckpt)).json"
    python -m pufferlib_market.evaluate_holdout \
        --checkpoint "$ckpt" --data-path "$VAL12" \
        --eval-hours 90 --n-windows 50 --seed 42 \
        --fee-rate 0.001 --fill-buffer-bps 5.0 \
        --deterministic --no-early-stop --out "$out" > /dev/null 2>&1
    [ ! -f "$out" ] && return 1
    python3 -c "
import json, sys
d=json.load(open('$out'))
s=d['summary']
rets=[w['total_return'] for w in d['windows']]
neg=sum(1 for r in rets if r<0)
med=s['median_total_return']*100
if med > $MIN_MED and neg < $MAX_NEG:
    sys.exit(0)  # promising
sys.exit(1)  # not promising
" 2>/dev/null
}

# === Stage 1: Quick scan at 15M steps ===
CSV_SCAN="${LOG_ROOT}/stocks12_quickscan_leaderboard.csv"
CKPT_SCAN="${CKPT_ROOT}/stocks12_quickscan"
[ ! -f "$CSV_SCAN" ] && echo "timestamp,label,med_90d,p10_90d,worst_90d,neg_of_50,med_sortino,checkpoint" > "$CSV_SCAN"

log "=== Stage 1: Quick scan s106-s600 @ 15M steps ==="

scan_seeds=($(seq 106 599))
for i in $(seq 0 $PARALLEL $((${#scan_seeds[@]} - 1))); do
    batch=(); for j in $(seq 0 $((PARALLEL-1))); do
        idx=$((i+j)); [ $idx -lt ${#scan_seeds[@]} ] && batch+=(${scan_seeds[$idx]})
    done
    log "--- Scanning: ${batch[*]} ---"
    for seed in "${batch[@]}"; do
        ckpt_dir="${CKPT_SCAN}/tp05_s${seed}"
        [ -f "${ckpt_dir}/best.pt" ] && continue
        python -m pufferlib_market.train \
            --data-path "$TRAIN12" --hidden-size $HIDDEN_SIZE \
            --num-envs $NUM_ENVS --total-timesteps $SCAN_STEPS \
            --anneal-lr --trade-penalty 0.05 --fee-rate 0.001 \
            --max-steps $MAX_STEPS --seed "$seed" \
            --checkpoint-dir "$ckpt_dir" \
            > "/tmp/scan_s${seed}.log" 2>&1 &
    done
    wait
    for seed in "${batch[@]}"; do
        eval_checkpoint "qs_tp05_s${seed}" "${CKPT_SCAN}/tp05_s${seed}/best.pt" "$VAL12" "$CSV_SCAN" || true
    done
done

log "Stage 1 scan done."
sort -t, -k3 -rn "$CSV_SCAN" | head -10

# === Stage 2: Retrain promising seeds at 35M ===
log ""
log "=== Stage 2: Retrain promising seeds @ 35M ==="
CSV_RETRAIN="${LOG_ROOT}/stocks12_quickscan_retrain_leaderboard.csv"
CKPT_RETRAIN="${CKPT_ROOT}/stocks12_quickscan_retrain"
[ ! -f "$CSV_RETRAIN" ] && echo "timestamp,label,med_90d,p10_90d,worst_90d,neg_of_50,med_sortino,checkpoint" > "$CSV_RETRAIN"

# Find promising seeds from Stage 1
PROMISING_SEEDS=()
while IFS=, read -r ts label med p10 worst neg sort ckpt; do
    [[ "$label" == "label" ]] && continue
    seed_num=$(echo "$label" | grep -oP '\d+$')
    # Check threshold: med > MIN_MED and neg < MAX_NEG
    if python3 -c "import sys; med=float('$med'); neg=int('$neg'); sys.exit(0 if med>$MIN_MED and neg<$MAX_NEG else 1)" 2>/dev/null; then
        PROMISING_SEEDS+=("$seed_num")
        log "  Promising: ${label} (med=${med}%, neg=${neg}/50)"
    fi
done < "$CSV_SCAN"

if [ ${#PROMISING_SEEDS[@]} -eq 0 ]; then
    log "No promising seeds found. Done."
    exit 0
fi

log "Retraining ${#PROMISING_SEEDS[@]} promising seeds: ${PROMISING_SEEDS[*]}"

for i in $(seq 0 6 $((${#PROMISING_SEEDS[@]} - 1))); do
    batch=(); for j in $(seq 0 5); do
        idx=$((i+j)); [ $idx -lt ${#PROMISING_SEEDS[@]} ] && batch+=(${PROMISING_SEEDS[$idx]})
    done
    log "--- Retraining: ${batch[*]} ---"
    for seed in "${batch[@]}"; do
        ckpt_dir="${CKPT_RETRAIN}/tp05_s${seed}"
        [ -f "${ckpt_dir}/best.pt" ] && continue
        python -m pufferlib_market.train \
            --data-path "$TRAIN12" --hidden-size $HIDDEN_SIZE \
            --num-envs $NUM_ENVS --total-timesteps $RETRAIN_STEPS \
            --anneal-lr --trade-penalty 0.05 --fee-rate 0.001 \
            --max-steps $MAX_STEPS --seed "$seed" \
            --checkpoint-dir "$ckpt_dir" \
            > "/tmp/retrain_s${seed}.log" 2>&1 &
    done
    wait
    for seed in "${batch[@]}"; do
        eval_checkpoint "qr_tp05_s${seed}" "${CKPT_RETRAIN}/tp05_s${seed}/best.pt" "$VAL12" "$CSV_RETRAIN" || true
    done
done

log "=== Stage 2 done. 0/50-neg champions: ==="
awk -F, '$6=="0"' "$CSV_RETRAIN" | sort -t, -k3 -rn
