#!/bin/bash
# Short run ext_shorts sweep: 8M steps instead of 15M
# Hypothesis: models find good short policy early but degrade with more training
# Stopping at 8M should capture the peak before degradation.
#
# Early termination: if first val (step 50/244) shows >25/30 neg, kill training — bad seed.
# This saves ~25 min per bad seed.
#
# Usage:
#   nohup bash scripts/sweep_screened32_ext_shorts_8m.sh > /tmp/s32ext_shorts_8m.log 2>&1 &

set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

export TMPDIR="$(pwd)/.tmp_train"
mkdir -p "$TMPDIR"

TRAIN="pufferlib_market/data/screened32_ext_augmented_train.bin"
VAL="pufferlib_market/data/screened32_ext_augmented_val.bin"
STEPS=8000000
TP=0.02
# Early termination threshold: kill if first val neg > this
# 28 = kill only 29+/30 neg (truly hopeless). 25-28/30 might recover by val 2-4.
EARLY_STOP_NEG=${EARLY_STOP_NEG:-28}

SEEDS=${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20}
NUM_ENVS=${NUM_ENVS:-128}
CKPT_ROOT="pufferlib_market/checkpoints/screened32_ext_shorts_8m_sweep/C"
LOG="$CKPT_ROOT/leaderboard.csv"
mkdir -p "$CKPT_ROOT"
[ -f "$LOG" ] || echo "timestamp,variant,seed,med_pct,p10_pct,worst_pct,neg_count,med_sortino,checkpoint" > "$LOG"

train_one() {
  local seed=$1
  local dir="$CKPT_ROOT/s${seed}"
  mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] [8M_shorts_C] seed ${seed} — training"
  python -u -m pufferlib_market.train \
      --data-path "$TRAIN" --val-data-path "$VAL" \
      --total-timesteps "$STEPS" --max-steps 252 \
      --trade-penalty "$TP" --hidden-size 1024 \
      --anneal-lr --val-eval-windows 30 \
      --num-envs "$NUM_ENVS" \
      --seed "$seed" --checkpoint-dir "$dir" > "$dir/train.log" 2>&1 &
  local train_pid=$!

  # Monitor for early termination: check every 60s for val results
  # Kill if val1 >EARLY_STOP_NEG (hopeless), or if val1+val2 both >20/30 (consistently bad)
  local vals_checked=0
  local first_neg=-1
  while kill -0 $train_pid 2>/dev/null; do
    sleep 60
    local val_count
    val_count=$(grep -c "\[val\]" "$dir/train.log" 2>/dev/null || echo 0)
    if [ "$val_count" -gt "$vals_checked" ]; then
      local latest_val neg n_val
      latest_val=$(grep "\[val\]" "$dir/train.log" 2>/dev/null | sed -n "$((vals_checked+1))p")
      vals_checked=$val_count
      neg=$(echo "$latest_val" | grep -oP 'neg=\K[0-9]+(?=/)')
      n_val=$(echo "$latest_val" | grep -oP 'neg=[0-9]+/\K[0-9]+')
      echo "  s${seed}: val${vals_checked} neg=${neg}/${n_val}"
      if [ -n "$neg" ] && [ -n "$n_val" ] && [ "$n_val" -eq 30 ]; then
        # Kill if first val is truly hopeless (>EARLY_STOP_NEG)
        if [ "$vals_checked" -eq 1 ] && [ "$neg" -gt "$EARLY_STOP_NEG" ]; then
          echo "  s${seed}: early termination val1 (neg=${neg}/${n_val} > ${EARLY_STOP_NEG})"
          kill "$train_pid" 2>/dev/null; wait "$train_pid" 2>/dev/null
          touch "$dir/SKIPPED_EARLY_TERM"
          echo "[$(date -u +%FT%TZ)] [8M_shorts_C] seed ${seed} — early terminated val1 neg=${neg}/${n_val}"
          return 0
        fi
        # Save first neg for val2 check
        [ "$vals_checked" -eq 1 ] && first_neg=$neg
        # Kill if val1+val2 both bad (>20/30) — unlikely to recover
        if [ "$vals_checked" -eq 2 ] && [ "$first_neg" -gt 20 ] && [ "$neg" -gt 20 ]; then
          echo "  s${seed}: early termination val1+val2 (${first_neg}+${neg}/30 both bad)"
          kill "$train_pid" 2>/dev/null; wait "$train_pid" 2>/dev/null
          touch "$dir/SKIPPED_EARLY_TERM"
          echo "[$(date -u +%FT%TZ)] [8M_shorts_C] seed ${seed} — early terminated val1+val2 neg=${first_neg}+${neg}/30"
          return 0
        fi
      fi
    fi
  done
  wait $train_pid
  echo "[$(date -u +%FT%TZ)] [8M_shorts_C] seed ${seed} — training done"
}

eval_one() {
  local seed=$1
  local dir="$CKPT_ROOT/s${seed}"
  # Skip if early-terminated
  if [ -f "$dir/SKIPPED_EARLY_TERM" ]; then
    echo "  s${seed}: skipped (early terminated)"
    return
  fi
  local ckpt="$dir/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$dir/best.pt"
  [ -f "$ckpt" ] || { echo "  s${seed}: no ckpt"; return; }
  local out="$dir/eval_lag2.json"
  CUDA_VISIBLE_DEVICES="" python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" --data-path "$VAL" \
      --eval-hours 90 --n-windows 30 --fee-rate 0.001 \
      --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop \
      --device cpu \
      > "$out" 2>/dev/null || { echo "  s${seed}: eval failed"; return; }
  stats=$(python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{d.get('median_total_return',0)*100:.2f},{d.get('p10_total_return',0)*100:.2f},{(d.get('worst_window') or {}).get('total_return',0)*100:.2f},{d.get('negative_windows',0)},{d.get('median_sortino',0):.2f}")
PY
  )
  [ -z "$stats" ] && return
  ts=$(date -u +%FT%TZ)
  echo "$ts,C,$seed,$stats,$ckpt" >> "$LOG"
  echo "  s${seed}: ${stats}"
}

echo "=== ext_shorts 8M C sweep: tp=$TP, 8M steps, envs=$NUM_ENVS, seeds=$SEEDS, early_stop_neg=$EARLY_STOP_NEG ==="
# Run eval in background (CPU-only) while next training runs on GPU
last_eval_pid=""
for seed in $SEEDS; do
  dir="$CKPT_ROOT/s${seed}"
  if [ -f "$dir/eval_lag2.json" ]; then
    echo "  s${seed}: done, skip"
    continue
  fi
  if [ -f "$dir/SKIPPED_EARLY_TERM" ]; then
    echo "  s${seed}: was early-terminated, skip"
    continue
  fi
  train_one "$seed"
  # Launch eval in background (CPU), don't wait — next seed's training can run concurrently
  eval_one "$seed" &
  last_eval_pid=$!
done
# Wait for last background eval to finish
[ -n "$last_eval_pid" ] && wait "$last_eval_pid" 2>/dev/null

echo "=== LEADERBOARD ==="
sort -t, -k7 -n "$LOG" | head -10
echo "--- top by median ---"
sort -t, -k4 -rn "$LOG" | head -10
