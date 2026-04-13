#!/bin/bash
# Phase 1: Quick-scan sweep for crypto6 margin trading with PPO.
# Tests shorts, leverage, allocation bins, reward shaping, architectures.
# 3M steps each (~10 min/config on 3090 Ti).
cd "$(git rev-parse --show-toplevel)"
source .venv313/bin/activate
export CUDA_VISIBLE_DEVICES=""  # run on CPU (GPU occupied by prod services)

TRAIN_DATA="pufferlib_market/data/crypto6_margin_train.bin"
VAL_DATA="pufferlib_market/data/crypto6_margin_val.bin"
CKPT_ROOT="pufferlib_market/checkpoints/crypto6_margin_p1"
LOG="$CKPT_ROOT/leaderboard.csv"

TOTAL_TIMESTEPS=2000000
MAX_STEPS=720
PERIODS_PER_YEAR=8760.0
DECISION_LAG=2
FEE_RATE=0.001
NUM_ENVS=32
FILL_SLIP=5.0

test -f "$TRAIN_DATA" || { echo "ERROR: $TRAIN_DATA missing"; exit 1; }
test -f "$VAL_DATA"   || { echo "ERROR: $VAL_DATA missing"; exit 1; }
mkdir -p "$CKPT_ROOT"

echo "timestamp,config,seed,med_pct,p10_pct,worst_pct,med_sortino,num_trades,checkpoint" > "$LOG"

run_config() {
  local name=$1; shift
  local seed=$1; shift
  # remaining args are extra train flags
  local dir="$CKPT_ROOT/${name}_s${seed}"
  mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] $name s$seed $@"
  python -u -m pufferlib_market.train \
      --data-path       "$TRAIN_DATA" \
      --val-data-path   "$VAL_DATA" \
      --total-timesteps "$TOTAL_TIMESTEPS" \
      --max-steps       "$MAX_STEPS" \
      --num-envs        "$NUM_ENVS" \
      --periods-per-year "$PERIODS_PER_YEAR" \
      --decision-lag    "$DECISION_LAG" \
      --fee-rate        "$FEE_RATE" \
      --fill-slippage-bps "$FILL_SLIP" \
      --obs-norm \
      --lr 3e-4 \
      --ent-coef        0.02 \
      --reward-scale    10.0 \
      --reward-clip     5.0 \
      --cash-penalty    0.005 \
      --trade-penalty   0.02 \
      --seed            "$seed" \
      --checkpoint-dir  "$dir" \
      "$@" \
      > "$dir/train.log" 2>&1
  echo "[$(date -u +%FT%TZ)] done $name s$seed (exit=$?)"

  local ckpt="$dir/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$dir/best.pt"
  [ -f "$ckpt" ] || { echo "  $name s$seed: no checkpoint"; return; }
  local out="$dir/eval_lag2.json"
  python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" --data-path "$VAL_DATA" \
      --eval-hours 720 --n-windows 50 --fee-rate "$FEE_RATE" \
      --fill-buffer-bps 5.0 --slippage-bps 5.0 \
      --decision-lag "$DECISION_LAG" --deterministic --no-early-stop \
      --max-leverage 2.0 --short-borrow-apr 0.10 \
      > "$out" 2>/dev/null || { echo "  $name s$seed: eval failed"; return; }
  stats=$(python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
med   = d.get("median_total_return", 0) * 100
p10   = d.get("p10_total_return", 0) * 100
worst = d.get("worst_window", {}).get("total_return", 0) * 100
sort  = d.get("median_sortino", 0)
trades = d.get("median_num_trades", d.get("best_window", {}).get("num_trades", 0))
print(f"{med:.2f},{p10:.2f},{worst:.2f},{sort:.2f},{trades}")
PY
  )
  [ -z "$stats" ] && { echo "  $name s$seed: parse failed"; return; }
  echo "$(date -u +%FT%TZ),$name,$seed,$stats,$ckpt" >> "$LOG"
  echo "  $name s$seed: med=${stats%%,*}% | $stats"
}

echo "=== Phase 1: crypto6 margin sweep ==="
echo

# ---- A. MARGIN KNOBS (12 configs x 2 seeds) ----

# baseline: long-only, no leverage
for s in 1 2; do
  run_config "longonly_1x" $s \
    --disable-shorts --max-leverage 1.0
done

# long-only with 2x leverage
for s in 1 2; do
  run_config "longonly_2x" $s \
    --disable-shorts --max-leverage 2.0
done

# margin 2x, shorts enabled, 1 alloc bin (all-in)
for s in 1 2; do
  run_config "margin_2x_a1" $s \
    --max-leverage 2.0 --short-borrow-apr 0.10 \
    --action-allocation-bins 1
done

# margin 2x, 3 alloc bins
for s in 1 2; do
  run_config "margin_2x_a3" $s \
    --max-leverage 2.0 --short-borrow-apr 0.10 \
    --action-allocation-bins 3
done

# margin 2x, 5 alloc bins
for s in 1 2; do
  run_config "margin_2x_a5" $s \
    --max-leverage 2.0 --short-borrow-apr 0.10 \
    --action-allocation-bins 5
done

# margin 3x leverage
for s in 1 2; do
  run_config "margin_3x_a3" $s \
    --max-leverage 3.0 --short-borrow-apr 0.10 \
    --action-allocation-bins 3
done

# smooth downside penalty = 1.0
for s in 1 2; do
  run_config "margin_2x_a3_sd1" $s \
    --max-leverage 2.0 --short-borrow-apr 0.10 \
    --action-allocation-bins 3 \
    --smooth-downside-penalty 1.0
done

# smooth downside penalty = 5.0
for s in 1 2; do
  run_config "margin_2x_a3_sd5" $s \
    --max-leverage 2.0 --short-borrow-apr 0.10 \
    --action-allocation-bins 3 \
    --smooth-downside-penalty 5.0
done

# drawdown penalty
for s in 1 2; do
  run_config "margin_2x_a3_dd" $s \
    --max-leverage 2.0 --short-borrow-apr 0.10 \
    --action-allocation-bins 3 \
    --drawdown-penalty 0.02
done

# lower trade penalty
for s in 1 2; do
  run_config "margin_2x_a3_tp01" $s \
    --max-leverage 2.0 --short-borrow-apr 0.10 \
    --action-allocation-bins 3 \
    --trade-penalty 0.01
done

# max hold 6 hours
for s in 1 2; do
  run_config "margin_2x_a3_hold6" $s \
    --max-leverage 2.0 --short-borrow-apr 0.10 \
    --action-allocation-bins 3 \
    --max-hold-hours 6
done

# max hold 12 hours
for s in 1 2; do
  run_config "margin_2x_a3_hold12" $s \
    --max-leverage 2.0 --short-borrow-apr 0.10 \
    --action-allocation-bins 3 \
    --max-hold-hours 12
done

# ---- B. ARCHITECTURE SWEEP (best = margin_2x_a3, 6 archs x 2 seeds) ----

MARGIN_BASE="--max-leverage 2.0 --short-borrow-apr 0.10 --action-allocation-bins 3"

for s in 1 2; do
  run_config "arch_mlp256" $s \
    $MARGIN_BASE --arch mlp --hidden-size 256
done

for s in 1 2; do
  run_config "arch_mlp512" $s \
    $MARGIN_BASE --arch mlp --hidden-size 512
done

for s in 1 2; do
  run_config "arch_resmlp256" $s \
    $MARGIN_BASE --arch resmlp --hidden-size 256
done

for s in 1 2; do
  run_config "arch_transformer256" $s \
    $MARGIN_BASE --arch transformer --hidden-size 256
done

for s in 1 2; do
  run_config "arch_depthrecur256" $s \
    $MARGIN_BASE --arch depth_recurrence --hidden-size 256
done

for s in 1 2; do
  run_config "arch_gru256" $s \
    $MARGIN_BASE --arch gru --hidden-size 256
done

# ---- C. ENTROPY SWEEP (6 configs x 2 seeds) ----

for s in 1 2; do
  run_config "ent001" $s \
    $MARGIN_BASE --ent-coef 0.001
done

for s in 1 2; do
  run_config "ent005" $s \
    $MARGIN_BASE --ent-coef 0.005
done

for s in 1 2; do
  run_config "ent02" $s \
    $MARGIN_BASE --ent-coef 0.02
done

for s in 1 2; do
  run_config "ent05" $s \
    $MARGIN_BASE --ent-coef 0.05
done

for s in 1 2; do
  run_config "anneal_05_001" $s \
    $MARGIN_BASE --ent-coef 0.05 --ent-coef-end 0.001
done

for s in 1 2; do
  run_config "anneal_02_001" $s \
    $MARGIN_BASE --ent-coef 0.02 --ent-coef-end 0.001
done

echo
echo "=== LEADERBOARD (by median return, top 25) ==="
sort -t, -k4 -rn "$LOG" | head -25
echo
echo "=== BEST PER CONFIG ==="
tail -n +2 "$LOG" | sort -t, -k2,2 -k4 -rn | awk -F, '!seen[$2]++ {print $2": med="$4"% p10="$5"% sort="$7" trades="$8}'
