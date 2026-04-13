#!/bin/bash
# Train daily crypto30 PPO model.
# Recipe adapted from stocks17 C_low_tp (the best-performing variant).
#
# Key diffs from stocks17:
#   --periods-per-year 365 (crypto 24/7, not 252 stock trading days)
#   --max-steps 365 (1yr daily episodes)
#   --decision-lag 2 (native lag=2 in training -- critical for honest eval)
#
# Usage:
#   nohup bash scripts/train_crypto30_daily.sh > /tmp/crypto30_daily.log 2>&1 &

cd "$(git rev-parse --show-toplevel)"
source .venv313/bin/activate

# Disable Triton -- GPU memory is full from other processes
export TRITON_DISABLED=1

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
REWARD_SCALE=1.0      # crypto vol >> stocks, scale down from default 10
REWARD_CLIP=3.0        # tighter clip for stability

test -f "$TRAIN_DATA" || { echo "ERROR: missing $TRAIN_DATA"; exit 1; }
test -f "$VAL_DATA"   || { echo "ERROR: missing $VAL_DATA";   exit 1; }
mkdir -p "$CKPT_ROOT"

echo "timestamp,seed,med_pct,p10_pct,worst_pct,med_sortino,checkpoint" > "$LOG"

train_one() {
  local seed=$1
  local dir="$CKPT_ROOT/s${seed}"
  mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] training seed $seed -> $dir"
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
      --reward-scale    "$REWARD_SCALE" \
      --reward-clip     "$REWARD_CLIP" \
      --cpu \
      --seed            "$seed" \
      --checkpoint-dir  "$dir" \
      > "$dir/train.log" 2>&1
  local exit_code=$?
  echo "[$(date -u +%FT%TZ)] done seed $seed (exit=$exit_code)"
}

eval_one() {
  local seed=$1
  local ckpt="$CKPT_ROOT/s${seed}/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$CKPT_ROOT/s${seed}/best.pt"
  [ -f "$ckpt" ] || { echo "  s${seed}: no checkpoint found"; return; }
  local out="$CKPT_ROOT/s${seed}/eval_lag2.json"
  echo "  s${seed}: evaluating $ckpt"
  python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" \
      --data-path  "$VAL_DATA" \
      --eval-hours 60 \
      --n-windows  50 \
      --fee-rate   "$FEE_RATE" \
      --fill-buffer-bps 5.0 \
      --decision-lag "$DECISION_LAG" \
      --deterministic \
      --no-early-stop \
      > "$out" 2>/dev/null || { echo "  s${seed}: eval failed"; return; }
  stats=$(python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
med   = d.get('median_total_return', 0.0) * 100
p10   = d.get('p10_total_return',    0.0) * 100
worst = d.get('worst_window',  {}).get('total_return', 0.0) * 100
sort  = d.get('median_sortino', 0.0)
print(f"{med:.2f},{p10:.2f},{worst:.2f},{sort:.2f}")
PY
  )
  [ -z "$stats" ] && { echo "  s${seed}: parse failed"; return; }
  ts=$(date -u +%FT%TZ)
  echo "$ts,$seed,$stats,$ckpt" >> "$LOG"
  echo "  s${seed}: med=${stats%%,*}% | $stats"
}

SEEDS=(1 2 3 4 5)
echo "=== crypto30 daily PPO training: seeds=${SEEDS[*]} ==="
echo "    train=$TRAIN_DATA  val=$VAL_DATA"
echo "    tp=$TRADE_PENALTY  h=$HIDDEN_SIZE  steps=$TOTAL_TIMESTEPS  lag=$DECISION_LAG"
echo "    periods_per_year=$PERIODS_PER_YEAR  fee=$FEE_RATE  rscale=$REWARD_SCALE  rclip=$REWARD_CLIP  --disable-shorts"
echo

for s in "${SEEDS[@]}"; do
  train_one "$s"
  eval_one  "$s"
done

echo
echo "=== LEADERBOARD (top 5 by median) ==="
sort -t, -k3 -rn "$LOG" | head -5
echo
pos_p10=$(tail -n +2 "$LOG" | awk -F, '$4>0 {c++} END {print c+0}')
total=$(tail -n +2 "$LOG" | wc -l | tr -d ' ')
echo "Positive p10 seeds: $pos_p10 / $total"
