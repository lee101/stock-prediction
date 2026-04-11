#!/bin/bash
# Train 17-symbol augmented daily stock model.
#
# Dataset: pufferlib_market/data/stocks17_augmented_train.bin
#   - 17 symbols: AAPL MSFT NVDA GOOG META TSLA SPY QQQ PLTR JPM V AMZN AMD NFLX COIN CRWD UBER
#   - 5x augmentation via hourly session-shifting (offsets 0-4 bars)
#   - 1703 timesteps train, 177 timesteps val
#
# Recipe: anneal-lr only — proven working recipe for daily stock data.
#   NO --obs-norm / --bf16 / --cuda-graph-ppo (cause trade collapse on small daily datasets).
#   --disable-shorts: long/flat only (18 actions = 1 flat + 17 long)
#
# Usage:
#   nohup bash scripts/train_stocks17_augmented.sh > /tmp/stocks17_aug.log 2>&1 &

# No set -e: eval failures must not abort seed loop
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

TRAIN_DATA="pufferlib_market/data/stocks17_augmented_train.bin"
VAL_DATA="pufferlib_market/data/stocks17_augmented_val.bin"
CKPT_ROOT="pufferlib_market/checkpoints/stocks17_augmented"
LOG="$CKPT_ROOT/leaderboard.csv"

TOTAL_TIMESTEPS=15000000
HIDDEN_SIZE=1024
NUM_ENVS=128
MAX_STEPS=252       # 1yr daily episodes
TRADE_PENALTY=0.05

test -f "$TRAIN_DATA" || { echo "ERROR: missing $TRAIN_DATA"; exit 1; }
test -f "$VAL_DATA"   || { echo "ERROR: missing $VAL_DATA";   exit 1; }
mkdir -p "$CKPT_ROOT"

echo "timestamp,seed,med_pct,p10_pct,worst_pct,med_sortino,checkpoint" > "$LOG"

train_one() {
  local seed=$1
  local dir="$CKPT_ROOT/tp05_s${seed}"
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
      --seed            "$seed" \
      --checkpoint-dir  "$dir" \
      > "$dir/train.log" 2>&1
  local exit_code=$?
  echo "[$(date -u +%FT%TZ)] done seed $seed (exit=$exit_code)"
}

eval_one() {
  local seed=$1
  local ckpt="$CKPT_ROOT/tp05_s${seed}/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$CKPT_ROOT/tp05_s${seed}/best.pt"
  [ -f "$ckpt" ] || { echo "  s${seed}: no checkpoint found"; return; }
  local out="$CKPT_ROOT/tp05_s${seed}/eval_holdout50.json"
  echo "  s${seed}: evaluating $ckpt"
  python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" \
      --data-path  "$VAL_DATA" \
      --eval-hours 60 \
      --n-windows  30 \
      --fee-rate   0.001 \
      --fill-buffer-bps 5.0 \
      --decision-lag 2 \
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

# Run seeds sequentially on a single GPU (seeds 1-5 already done; resume from 6)
SEEDS=(6 7 8 9 10 11 12 13 14 15)
echo "=== stocks17 augmented training: seeds=${SEEDS[*]} ==="
echo "    train=$TRAIN_DATA  val=$VAL_DATA"
echo "    tp=$TRADE_PENALTY  h=$HIDDEN_SIZE  steps=$TOTAL_TIMESTEPS  --disable-shorts"
echo

for s in "${SEEDS[@]}"; do
  train_one "$s"
  eval_one  "$s"
done

echo
echo "=== LEADERBOARD (top 10 by median) ==="
sort -t, -k3 -rn "$LOG" | head -10
