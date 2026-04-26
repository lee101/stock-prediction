#!/bin/bash
# Crypto30 GPU short sweep: 5M timesteps (early stop regime).
# Key insight: GPU trains too fast, overfits after ~3M steps.
# Best val comes early. Shorter training = better generalization.
cd "$(git rev-parse --show-toplevel)"
source .venv313/bin/activate

TRAIN_DATA="pufferlib_market/data/crypto30_daily_aug_train.bin"
VAL_DATA="pufferlib_market/data/crypto30_daily_val.bin"
CKPT_ROOT="pufferlib_market/checkpoints/crypto30_gpu_short"
LOG="$CKPT_ROOT/leaderboard.csv"

TOTAL_TIMESTEPS=5000000
MAX_STEPS=365
PERIODS_PER_YEAR=365.0
DECISION_LAG=2
FEE_RATE=0.001
NUM_ENVS=32
REWARD_SCALE=10.0
REWARD_CLIP=5.0

test -f "$TRAIN_DATA" || { echo "ERROR: $TRAIN_DATA missing"; exit 1; }
test -f "$VAL_DATA"   || { echo "ERROR: $VAL_DATA missing"; exit 1; }
mkdir -p "$CKPT_ROOT"

echo "timestamp,config,seed,med_pct,p10_pct,worst_pct,med_sortino,checkpoint" > "$LOG"

run_config() {
  local name=$1 hidden=$2 tp=$3 wd=$4 entcoef=$5 seed=$6
  local dir="$CKPT_ROOT/${name}_s${seed}"
  mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] $name s$seed h=$hidden tp=$tp wd=$wd ent=$entcoef"
  python -u -m pufferlib_market.train \
      --data-path       "$TRAIN_DATA" \
      --val-data-path   "$VAL_DATA" \
      --total-timesteps "$TOTAL_TIMESTEPS" \
      --max-steps       "$MAX_STEPS" \
      --trade-penalty   "$tp" \
      --hidden-size     "$hidden" \
      --anneal-lr \
      --disable-shorts \
      --num-envs        "$NUM_ENVS" \
      --periods-per-year "$PERIODS_PER_YEAR" \
      --decision-lag    "$DECISION_LAG" \
      --fee-rate        "$FEE_RATE" \
      --reward-scale    "$REWARD_SCALE" \
      --reward-clip     "$REWARD_CLIP" \
      --ent-coef        "$entcoef" \
      --weight-decay    "$wd" \
      --seed            "$seed" \
      --checkpoint-dir  "$dir" \
      > "$dir/train.log" 2>&1
  echo "[$(date -u +%FT%TZ)] done $name s$seed (exit=$?)"

  local ckpt="$dir/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$dir/best.pt"
  [ -f "$ckpt" ] || { echo "  $name s$seed: no checkpoint"; return; }
  local out="$dir/eval_lag2.json"
  python -m pufferlib_market.evaluate_holdout \
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

echo "=== crypto30 GPU SHORT sweep (5M steps) ==="
echo

# h=256 baseline -- 5 seeds
for s in 1 2 3 4 5; do
  run_config "h256"           256  0.02  0.0    0.01  $s
done

# h=256 + weight decay 0.01
for s in 1 2 3; do
  run_config "h256_wd01"      256  0.02  0.01   0.01  $s
done

# h=256 + weight decay 0.05 (aggressive)
for s in 1 2 3; do
  run_config "h256_wd05"      256  0.02  0.05   0.01  $s
done

# h=256 lower trade penalty
for s in 1 2 3; do
  run_config "h256_tp005"     256  0.005 0.0    0.01  $s
done

# h=128 (smaller model = less overfitting)
for s in 1 2 3; do
  run_config "h128"           128  0.02  0.0    0.01  $s
done

# h=256 higher entropy (more exploration)
for s in 1 2 3; do
  run_config "h256_ent05"     256  0.02  0.0    0.05  $s
done

echo
echo "=== LEADERBOARD (by median, top 20) ==="
sort -t, -k4 -rn "$LOG" | head -20
echo
echo "=== BEST PER CONFIG ==="
tail -n +2 "$LOG" | sort -t, -k2,2 -k4 -rn | awk -F, '!seen[$2]++ {print $2": med="$4"% p10="$5"% sort="$7}'
