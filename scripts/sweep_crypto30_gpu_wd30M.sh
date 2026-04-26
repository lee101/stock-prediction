#!/bin/bash
# Crypto30 extended weight-decay sweep: 30M steps.
# Key finding: h256_wd005_s2 was still improving at 15M (+4.15% median).
# Weight decay prevents overfitting, model learns bearish conservatism late.
cd "$(git rev-parse --show-toplevel)"
source .venv313/bin/activate

TRAIN_DATA="pufferlib_market/data/crypto30_daily_aug_train.bin"
VAL_DATA="pufferlib_market/data/crypto30_daily_val.bin"
CKPT_ROOT="pufferlib_market/checkpoints/crypto30_gpu_wd30M"
LOG="$CKPT_ROOT/leaderboard.csv"

TOTAL_TIMESTEPS=30000000
HIDDEN_SIZE=256
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

echo "timestamp,config,seed,ckpt_type,med_pct,p10_pct,worst_pct,med_sortino,neg_windows,checkpoint" > "$LOG"

eval_checkpoint() {
  local name=$1 seed=$2 ckpt=$3 ckpt_type=$4
  [ -f "$ckpt" ] || return
  local out="${ckpt%.pt}_eval50w.json"
  python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" --data-path "$VAL_DATA" \
      --eval-hours 60 --n-windows 50 --fee-rate "$FEE_RATE" \
      --fill-buffer-bps 5.0 --decision-lag "$DECISION_LAG" --deterministic --no-early-stop \
      > "$out" 2>/dev/null || { echo "  $name s$seed $ckpt_type: eval failed"; return; }
  stats=$(python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
med   = d.get("median_total_return", 0) * 100
p10   = d.get("p10_total_return", 0) * 100
worst = d.get("worst_window", {}).get("total_return", 0) * 100
sort  = d.get("median_sortino", 0)
neg   = d.get("negative_windows", -1)
tot   = d.get("sampled_window_count", 50)
print(f"{med:.2f},{p10:.2f},{worst:.2f},{sort:.2f},{neg}/{tot}")
PY
  )
  [ -z "$stats" ] && { echo "  $name s$seed $ckpt_type: parse failed"; return; }
  echo "$(date -u +%FT%TZ),$name,$seed,$ckpt_type,$stats,$ckpt" >> "$LOG"
  echo "  $name s$seed $ckpt_type: med=${stats%%,*}% | $stats"
}

run_config() {
  local name=$1 tp=$2 wd=$3 entcoef=$4 seed=$5
  local dir="$CKPT_ROOT/${name}_s${seed}"
  mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] $name s$seed tp=$tp wd=$wd ent=$entcoef (30M steps)"
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
      --ent-coef        "$entcoef" \
      --weight-decay    "$wd" \
      --seed            "$seed" \
      --val-eval-windows 50 \
      --checkpoint-dir  "$dir" \
      > "$dir/train.log" 2>&1
  echo "[$(date -u +%FT%TZ)] done $name s$seed (exit=$?)"

  eval_checkpoint "$name" "$seed" "$dir/val_best.pt" "val_best"
  eval_checkpoint "$name" "$seed" "$dir/final.pt"    "final"
}

echo "=== crypto30 GPU weight-decay 30M sweep ==="
echo

# wd=0.005 -- proven, 5 seeds for statistical power
for s in 1 2 3 4 5; do
  run_config "wd005"           0.02  0.005  0.01  $s
done

# wd=0.01 -- stronger regularization
for s in 1 2 3; do
  run_config "wd01"            0.02  0.01   0.01  $s
done

# wd=0.02 -- aggressive
for s in 1 2 3; do
  run_config "wd02"            0.02  0.02   0.01  $s
done

# wd=0.005 + lower trade penalty
for s in 1 2 3; do
  run_config "wd005_tp005"     0.005 0.005  0.01  $s
done

# wd=0.01 + lower trade penalty
for s in 1 2 3; do
  run_config "wd01_tp005"      0.005 0.01   0.01  $s
done

# h=1024 + wd (h1024_s1 without wd had sort=+0.40, larger model + wd = high signal)
run_config_h() {
  local name=$1 hidden=$2 tp=$3 wd=$4 entcoef=$5 seed=$6
  local dir="$CKPT_ROOT/${name}_s${seed}"
  mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] $name s$seed h=$hidden tp=$tp wd=$wd ent=$entcoef (30M steps)"
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
      --val-eval-windows 50 \
      --checkpoint-dir  "$dir" \
      > "$dir/train.log" 2>&1
  echo "[$(date -u +%FT%TZ)] done $name s$seed (exit=$?)"
  eval_checkpoint "$name" "$seed" "$dir/val_best.pt" "val_best"
  eval_checkpoint "$name" "$seed" "$dir/final.pt"    "final"
}

for s in 1 2 3; do
  run_config_h "h1024_wd005"   1024  0.02  0.005  0.01  $s
done

for s in 1 2 3; do
  run_config_h "h1024_wd01"    1024  0.02  0.01   0.01  $s
done

echo
echo "=== LEADERBOARD (by median, top 20) ==="
sort -t, -k5 -rn "$LOG" | head -20
echo
echo "=== BEST PER CONFIG (val_best only) ==="
grep ",val_best," "$LOG" | sort -t, -k2,2 -k5 -rn | awk -F, '!seen[$2]++ {print $2": med="$5"% p10="$6"% sort="$8" neg="$9}'
