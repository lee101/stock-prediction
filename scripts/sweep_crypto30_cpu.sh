#!/bin/bash
# Crypto30 CPU training sweep with regularization.
# CPU h=256 num_envs=32 is the only working config (GPU has value collapse bug).
cd "$(git rev-parse --show-toplevel)"
source .venv313/bin/activate
export TRITON_DISABLED=1

TRAIN_DATA="pufferlib_market/data/crypto30_daily_aug_train.bin"
VAL_DATA="pufferlib_market/data/crypto30_daily_val.bin"
CKPT_ROOT="pufferlib_market/checkpoints/crypto30_cpu"
LOG="$CKPT_ROOT/leaderboard.csv"

TOTAL_TIMESTEPS=15000000
HIDDEN_SIZE=256
NUM_ENVS=32
MAX_STEPS=365
PERIODS_PER_YEAR=365.0
DECISION_LAG=2
FEE_RATE=0.001
REWARD_SCALE=10.0
REWARD_CLIP=5.0

test -f "$TRAIN_DATA" || { echo "ERROR: $TRAIN_DATA missing"; exit 1; }
test -f "$VAL_DATA"   || { echo "ERROR: $VAL_DATA missing"; exit 1; }
mkdir -p "$CKPT_ROOT"

echo "timestamp,config,seed,med_pct,p10_pct,worst_pct,med_sortino,checkpoint" > "$LOG"

run_config() {
  local name=$1 tp=$2 wd=$3 seed=$4
  local dir="$CKPT_ROOT/${name}_s${seed}"
  mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] $name s$seed tp=$tp wd=$wd"
  python -u -m pufferlib_market.train \
      --data-path       "$TRAIN_DATA" \
      --val-data-path   "$VAL_DATA" \
      --total-timesteps "$TOTAL_TIMESTEPS" \
      --max-steps       "$MAX_STEPS" \
      --trade-penalty   "$tp" \
      --hidden-size     "$HIDDEN_SIZE" \
      --anneal-lr \
      --disable-shorts \
      --num-envs        "$NUM_ENVS" \
      --periods-per-year "$PERIODS_PER_YEAR" \
      --decision-lag    "$DECISION_LAG" \
      --fee-rate        "$FEE_RATE" \
      --reward-scale    "$REWARD_SCALE" \
      --reward-clip     "$REWARD_CLIP" \
      --weight-decay    "$wd" \
      --cpu \
      --seed            "$seed" \
      --checkpoint-dir  "$dir" \
      > "$dir/train.log" 2>&1
  echo "[$(date -u +%FT%TZ)] done $name s$seed (exit=$?)"

  local ckpt="$dir/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$dir/best.pt"
  [ -f "$ckpt" ] || { echo "  $name s$seed: no checkpoint"; return; }
  local out="$dir/eval_lag2.json"
  CUDA_VISIBLE_DEVICES="" python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" --data-path "$VAL_DATA" \
      --eval-hours 60 --n-windows 50 --fee-rate "$FEE_RATE" \
      --fill-buffer-bps 5.0 --decision-lag "$DECISION_LAG" --deterministic --no-early-stop \
      > "$out" 2>/dev/null || { echo "  $name s$seed: eval failed"; return; }
  stats=$(python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
med   = d.get("median_total_return", 0) * 100
p10   = d.get("p10_total_return", 0) * 100
worst = d.get("worst_window", {}).get("total_return", 0) * 100
sort  = d.get("median_sortino", 0)
print(f"{med:.2f},{p10:.2f},{worst:.2f},{sort:.2f}")
PY
  )
  [ -z "$stats" ] && { echo "  $name s$seed: parse failed"; return; }
  echo "$(date -u +%FT%TZ),$name,$seed,$stats,$ckpt" >> "$LOG"
  echo "  $name s$seed: med=${stats%%,*}% | $stats"
}

echo "=== crypto30 CPU training sweep ==="
echo

# Baseline: no weight decay
for s in 1 2 3 4 5; do
  run_config "base"       0.02  0.0    $s
done

# Weight decay 0.005
for s in 1 2 3; do
  run_config "wd005"      0.02  0.005  $s
done

# Weight decay 0.01
for s in 1 2 3; do
  run_config "wd01"       0.02  0.01   $s
done

# Lower trade penalty
for s in 1 2 3; do
  run_config "tp005"      0.005 0.0    $s
done

# Lower tp + weight decay
for s in 1 2 3; do
  run_config "tp005_wd005" 0.005 0.005 $s
done

echo
echo "=== LEADERBOARD ==="
sort -t, -k4 -rn "$LOG" | head -20
echo
echo "=== BEST PER CONFIG ==="
tail -n +2 "$LOG" | sort -t, -k2,2 -k4 -rn | awk -F, '!seen[$2]++ {print $2": med="$4"% p10="$5"% sort="$7}'
