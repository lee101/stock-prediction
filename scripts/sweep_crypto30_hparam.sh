#!/bin/bash
# Hyperparameter sweep for crypto30 daily PPO with the short-masking fix.
# Tests reward_scale, trade_penalty, ent_coef, lr combos.
# Short training (3M steps) to find best config fast, then full run.
cd "$(git rev-parse --show-toplevel)"
source .venv313/bin/activate

TRAIN_DATA="pufferlib_market/data/crypto30_daily_aug_train.bin"
VAL_DATA="pufferlib_market/data/crypto30_daily_val.bin"
CKPT_ROOT="pufferlib_market/checkpoints/crypto30_hparam"
LOG="$CKPT_ROOT/leaderboard.csv"

TOTAL_TIMESTEPS=3000000  # short run for fast iteration
HIDDEN_SIZE=1024
NUM_ENVS=128
MAX_STEPS=365
PERIODS_PER_YEAR=365.0
DECISION_LAG=2
FEE_RATE=0.001

test -f "$TRAIN_DATA" || { echo "ERROR: $TRAIN_DATA missing"; exit 1; }
test -f "$VAL_DATA"   || { echo "ERROR: $VAL_DATA missing"; exit 1; }
mkdir -p "$CKPT_ROOT"

echo "timestamp,config,seed,med_pct,p10_pct,worst_pct,med_sortino,checkpoint" > "$LOG"

run_config() {
  local name=$1
  local rscale=$2
  local rclip=$3
  local tp=$4
  local ent=$5
  local lr=$6
  local seed=$7
  local dir="$CKPT_ROOT/${name}_s${seed}"
  mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] $name seed=$seed rscale=$rscale rclip=$rclip tp=$tp ent=$ent lr=$lr"
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
      --reward-scale    "$rscale" \
      --reward-clip     "$rclip" \
      --ent-coef        "$ent" \
      --lr              "$lr" \
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

echo "=== crypto30 hparam sweep (short-mask fix applied) ==="
echo

# Config format: name reward_scale reward_clip trade_penalty ent_coef lr seed
# Baseline (stock defaults + crypto adjustments)
run_config "rs10_tp02"      10.0 5.0  0.02  0.01  3e-4  1
run_config "rs10_tp02"      10.0 5.0  0.02  0.01  3e-4  2
run_config "rs10_tp02"      10.0 5.0  0.02  0.01  3e-4  3

# Lower trade penalty (stocks17 C_low_tp value)
run_config "rs10_tp005"     10.0 5.0  0.005 0.01  3e-4  1
run_config "rs10_tp005"     10.0 5.0  0.005 0.01  3e-4  2
run_config "rs10_tp005"     10.0 5.0  0.005 0.01  3e-4  3

# Medium trade penalty
run_config "rs10_tp01"      10.0 5.0  0.01  0.01  3e-4  1
run_config "rs10_tp01"      10.0 5.0  0.01  0.01  3e-4  2

# Lower entropy coef
run_config "rs10_tp02_ent001" 10.0 5.0 0.02  0.001 3e-4  1
run_config "rs10_tp02_ent001" 10.0 5.0 0.02  0.001 3e-4  2

# Higher LR
run_config "rs10_tp02_lr1e3" 10.0 5.0 0.02  0.01  1e-3  1
run_config "rs10_tp02_lr1e3" 10.0 5.0 0.02  0.01  1e-3  2

# reward_scale=5 (midpoint)
run_config "rs5_tp02"       5.0  3.0  0.02  0.01  3e-4  1
run_config "rs5_tp02"       5.0  3.0  0.02  0.01  3e-4  2

echo
echo "=== LEADERBOARD (by median) ==="
sort -t, -k4 -rn "$LOG" | head -15
echo
echo "=== BEST PER CONFIG ==="
tail -n +2 "$LOG" | sort -t, -k2,2 -k4 -rn | awk -F, '!seen[$2]++ {print $2": med="$4"% p10="$5"% sort="$7}'
