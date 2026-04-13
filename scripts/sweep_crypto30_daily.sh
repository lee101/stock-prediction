#!/bin/bash
# Extended seed sweep for crypto30 daily PPO.
# Run after pilot (seeds 1-5) shows promise.
#
# Usage:
#   nohup bash scripts/sweep_crypto30_daily.sh > /tmp/crypto30_sweep.log 2>&1 &

set -u
cd "$(git rev-parse --show-toplevel)"
source .venv313/bin/activate

TRAIN_DATA="pufferlib_market/data/crypto30_daily_aug_train.bin"
VAL_DATA="pufferlib_market/data/crypto30_daily_val.bin"
CKPT_ROOT="pufferlib_market/checkpoints/crypto30_daily"
LOG="$CKPT_ROOT/leaderboard.csv"

TOTAL_TIMESTEPS=15000000
HIDDEN_SIZE=256
NUM_ENVS=32
MAX_STEPS=365
TRADE_PENALTY=0.02
PERIODS_PER_YEAR=365.0
DECISION_LAG=2
FEE_RATE=0.001

test -f "$TRAIN_DATA" || { echo "ERROR: $TRAIN_DATA missing"; exit 1; }
test -f "$VAL_DATA"   || { echo "ERROR: $VAL_DATA missing";   exit 1; }
mkdir -p "$CKPT_ROOT"

if [ ! -f "$LOG" ]; then
    echo "timestamp,seed,med_pct,p10_pct,worst_pct,med_sortino,checkpoint" > "$LOG"
fi

train_one() {
  local seed=$1
  local dir="$CKPT_ROOT/s${seed}"
  mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] train seed $seed -> $dir"
  python -u -m pufferlib_market.train \
      --data-path       "$TRAIN_DATA" \
      --val-data-path   "$VAL_DATA" \
      --total-timesteps "$TOTAL_TIMESTEPS" \
      --max-steps       "$MAX_STEPS" \
      --trade-penalty   "$TRADE_PENALTY" \
      --hidden-size     "$HIDDEN_SIZE" \
      --anneal-lr \
      --disable-shorts \
      --num-envs        "$NUM_ENVS" \
      --periods-per-year "$PERIODS_PER_YEAR" \
      --decision-lag    "$DECISION_LAG" \
      --fee-rate        "$FEE_RATE" \
      --seed            "$seed" \
      --checkpoint-dir  "$dir" \
      > "$dir/train.log" 2>&1
  echo "[$(date -u +%FT%TZ)] done seed $seed (exit=$?)"
}

eval_one() {
  local seed=$1
  local ckpt="$CKPT_ROOT/s${seed}/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$CKPT_ROOT/s${seed}/best.pt"
  [ -f "$ckpt" ] || { echo "  s${seed}: no checkpoint"; return; }
  local out="$CKPT_ROOT/s${seed}/eval_lag2.json"
  echo "  s${seed}: evaluating (lag=$DECISION_LAG, 50 windows) ..."
  python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" --data-path "$VAL_DATA" \
      --eval-hours 60 --n-windows 50 --fee-rate "$FEE_RATE" \
      --fill-buffer-bps 5.0 --decision-lag "$DECISION_LAG" --deterministic --no-early-stop \
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
  echo "$ts,$seed,$stats,$ckpt" >> "$LOG"
  echo "  s${seed}: med=${stats%%,*}% | full=$stats"
}

SEEDS=(6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
echo "=== crypto30 daily extended sweep: seeds=${SEEDS[*]} ==="
echo

for s in "${SEEDS[@]}"; do train_one "$s"; eval_one "$s"; done

echo
echo "=== LEADERBOARD (top 10 by median) ==="
sort -t, -k3 -rn "$LOG" | head -10
echo
pos_p10=$(tail -n +2 "$LOG" | awk -F, '$4>0 {c++} END {print c+0}')
total=$(tail -n +2 "$LOG" | wc -l | tr -d ' ')
echo "Positive p10 seeds: $pos_p10 / $total"
