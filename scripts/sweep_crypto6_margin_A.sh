#!/bin/bash
# Part A: Margin knobs sweep for crypto6 (12 configs x 2 seeds = 24 runs).
# CPU-only, 2M steps each.
cd "$(git rev-parse --show-toplevel)"
source .venv313/bin/activate
export CUDA_VISIBLE_DEVICES=""

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

[ -f "$LOG" ] || echo "timestamp,config,seed,med_pct,p10_pct,worst_pct,med_sortino,num_trades,checkpoint" > "$LOG"

run_config() {
  local name=$1; shift
  local seed=$1; shift
  local dir="$CKPT_ROOT/${name}_s${seed}"
  mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] START $name s$seed $@"
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
      --hidden-size     256 \
      --seed            "$seed" \
      --checkpoint-dir  "$dir" \
      "$@" \
      > "$dir/train.log" 2>&1
  local rc=$?
  echo "[$(date -u +%FT%TZ)] DONE $name s$seed exit=$rc"

  local ckpt="$dir/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$dir/best.pt"
  [ -f "$ckpt" ] || { echo "  $name s$seed: no checkpoint"; return; }

  # eval with matching leverage
  local lev=$(grep -oP '(?<=--max-leverage )\S+' "$dir/train.log" | tail -1)
  [ -z "$lev" ] && lev="1.0"
  local borrow=$(grep -oP '(?<=--short-borrow-apr )\S+' "$dir/train.log" | tail -1)
  [ -z "$borrow" ] && borrow="0.0"

  local out="$dir/eval_lag2.json"
  python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" --data-path "$VAL_DATA" \
      --eval-hours 720 --n-windows 50 --fee-rate "$FEE_RATE" \
      --fill-buffer-bps 5.0 --slippage-bps 5.0 \
      --decision-lag "$DECISION_LAG" --deterministic --no-early-stop \
      --max-leverage "$lev" --short-borrow-apr "$borrow" \
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
  echo "  >>> $name s$seed: med=${stats%%,*}% | $stats"
}

echo "=== Part A: Margin knobs sweep (12 configs x 2 seeds) ==="
echo "Start: $(date -u +%FT%TZ)"
echo

# 1. baseline: long-only, no leverage
for s in 1 2; do run_config "longonly_1x" $s --disable-shorts --max-leverage 1.0; done

# 2. long-only with 2x leverage
for s in 1 2; do run_config "longonly_2x" $s --disable-shorts --max-leverage 2.0; done

# 3. margin 2x, shorts enabled, 1 alloc bin
for s in 1 2; do run_config "margin_2x_a1" $s --max-leverage 2.0 --short-borrow-apr 0.10 --action-allocation-bins 1; done

# 4. margin 2x, 3 alloc bins
for s in 1 2; do run_config "margin_2x_a3" $s --max-leverage 2.0 --short-borrow-apr 0.10 --action-allocation-bins 3; done

# 5. margin 2x, 5 alloc bins
for s in 1 2; do run_config "margin_2x_a5" $s --max-leverage 2.0 --short-borrow-apr 0.10 --action-allocation-bins 5; done

# 6. margin 3x leverage
for s in 1 2; do run_config "margin_3x_a3" $s --max-leverage 3.0 --short-borrow-apr 0.10 --action-allocation-bins 3; done

# 7. smooth downside = 1.0
for s in 1 2; do run_config "margin_2x_sd1" $s --max-leverage 2.0 --short-borrow-apr 0.10 --action-allocation-bins 3 --smooth-downside-penalty 1.0; done

# 8. smooth downside = 5.0
for s in 1 2; do run_config "margin_2x_sd5" $s --max-leverage 2.0 --short-borrow-apr 0.10 --action-allocation-bins 3 --smooth-downside-penalty 5.0; done

# 9. drawdown penalty
for s in 1 2; do run_config "margin_2x_dd" $s --max-leverage 2.0 --short-borrow-apr 0.10 --action-allocation-bins 3 --drawdown-penalty 0.02; done

# 10. lower trade penalty
for s in 1 2; do run_config "margin_2x_tp01" $s --max-leverage 2.0 --short-borrow-apr 0.10 --action-allocation-bins 3 --trade-penalty 0.01; done

# 11. max hold 6h
for s in 1 2; do run_config "margin_2x_hold6" $s --max-leverage 2.0 --short-borrow-apr 0.10 --action-allocation-bins 3 --max-hold-hours 6; done

# 12. max hold 12h
for s in 1 2; do run_config "margin_2x_hold12" $s --max-leverage 2.0 --short-borrow-apr 0.10 --action-allocation-bins 3 --max-hold-hours 12; done

echo
echo "=== LEADERBOARD (by median return, top 25) ==="
sort -t, -k4 -rn "$LOG" | head -25
echo
echo "=== BEST PER CONFIG ==="
tail -n +2 "$LOG" | sort -t, -k2,2 -k4 -rn | awk -F, '!seen[$2]++ {print $2": med="$4"% p10="$5"% sort="$7" trades="$8}'
echo
echo "Finished: $(date -u +%FT%TZ)"
