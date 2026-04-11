#!/bin/bash
# Extended seed sweep for C_low_tp (tp=0.02) — the best variant.
# Runs seeds 24-36 to find stable positive p10 performance.
# Wait for D/E variants to finish before starting to avoid GPU saturation.
#
# Usage:
#   nohup bash scripts/sweep_stocks17_c_extended.sh > /tmp/sw17_C_ext.log 2>&1 &

set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

VARIANT="C"
LABEL="low_tp"
TP=0.02
STEPS=15000000
MAX_EP=252
TRAIN_DATA="pufferlib_market/data/stocks17_augmented_train.bin"
VAL_DATA="pufferlib_market/data/stocks17_augmented_val.bin"
SEEDS=(24 25 26 27 28 29 30 31 32 33 34 35 36)

CKPT_ROOT="pufferlib_market/checkpoints/stocks17_sweep/${VARIANT}_${LABEL}"
LOG="$CKPT_ROOT/leaderboard.csv"

mkdir -p "$CKPT_ROOT"
test -f "$TRAIN_DATA" || { echo "ERROR: $TRAIN_DATA missing"; exit 1; }
test -f "$VAL_DATA"   || { echo "ERROR: $VAL_DATA missing";   exit 1; }

# Append to existing leaderboard (don't overwrite seeds 21-23)
if [ ! -f "$LOG" ]; then
    echo "timestamp,variant,seed,med_pct,p10_pct,worst_pct,med_sortino,checkpoint" > "$LOG"
fi

train_one() {
  local seed=$1
  local dir="$CKPT_ROOT/s${seed}"
  mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] [$VARIANT/$LABEL] train seed $seed → $dir"
  python -u -m pufferlib_market.train \
      --data-path       "$TRAIN_DATA" \
      --val-data-path   "$VAL_DATA" \
      --total-timesteps "$STEPS" \
      --max-steps       "$MAX_EP" \
      --trade-penalty   "$TP" \
      --hidden-size     1024 \
      --anneal-lr \
      --disable-shorts \
      --num-envs        128 \
      --seed            "$seed" \
      --checkpoint-dir  "$dir" \
      > "$dir/train.log" 2>&1
  echo "[$(date -u +%FT%TZ)] [$VARIANT/$LABEL] done seed $seed (exit=$?)"
}

eval_one() {
  local seed=$1
  local ckpt="$CKPT_ROOT/s${seed}/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$CKPT_ROOT/s${seed}/best.pt"
  [ -f "$ckpt" ] || { echo "  s${seed}: no checkpoint"; return; }
  local out="$CKPT_ROOT/s${seed}/eval_lag2.json"
  echo "  s${seed}: evaluating (lag=2, 50 windows) ..."
  python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" --data-path "$VAL_DATA" \
      --eval-hours 60 --n-windows 50 --fee-rate 0.001 \
      --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop \
      > "$out" 2>/dev/null || { echo "  s${seed}: eval failed"; return; }
  stats=$(python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
med   = d.get('median_total_return', 0)*100
p10   = d.get('p10_total_return', 0)*100
worst = d.get('worst_window',{}).get('total_return', 0)*100
sort  = d.get('median_sortino', 0)
print(f"{med:.2f},{p10:.2f},{worst:.2f},{sort:.2f}")
PY
  )
  [ -z "$stats" ] && { echo "  s${seed}: parse failed"; return; }
  ts=$(date -u +%FT%TZ)
  echo "$ts,$VARIANT,$seed,$stats,$ckpt" >> "$LOG"
  echo "  s${seed}: med=${stats%%,*}% | full=$stats"
}

echo "=== Extended C/low_tp seed sweep: tp=$TP steps=$STEPS max_ep=$MAX_EP seeds=${SEEDS[*]} ==="
echo

for s in "${SEEDS[@]}"; do train_one "$s"; eval_one "$s"; done

echo
echo "=== [C_ext] LEADERBOARD ==="
sort -t, -k5 -rn "$LOG" | head -10
echo
# Count positive p10
pos_p10=$(tail -n +2 "$LOG" | awk -F, '$5>0 {c++} END {print c+0}')
total=$(tail -n +2 "$LOG" | wc -l | tr -d ' ')
echo "Positive p10 seeds: $pos_p10 / $total"
