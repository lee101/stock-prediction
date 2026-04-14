#!/bin/bash
# Simple sequential I seed runner - no complex early termination
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate
export TMPDIR="$(pwd)/.tmp_train"
export TRITON_CACHE_DIR="$(pwd)/.tmp_train/triton_cache"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

TRAIN="pufferlib_market/data/screened32_augmented_train.bin"
VAL="pufferlib_market/data/screened32_augmented_val.bin"
FULL_VAL="pufferlib_market/data/screened32_single_offset_val_full.bin"
CKPT_ROOT="pufferlib_market/checkpoints/screened32_sweep/I"

LOG="$CKPT_ROOT/runner.log"
SEEDS_LIST="${SEEDS:-25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50}"
echo "[$(date -u +%FT%TZ)] Starting I seed runner: seeds=$SEEDS_LIST" >> "$LOG"

for s in $SEEDS_LIST; do
  dir="$CKPT_ROOT/s${s}"
  mkdir -p "$dir"

  # Skip if already fully done
  if [ -f "$dir/eval_full.json" ] && [ -s "$dir/eval_full.json" ]; then
    echo "[$(date -u +%FT%TZ)] s${s}: already evaluated, skip" | tee -a "$LOG"
    continue
  fi

  # Skip if already has final.pt (trained) → just eval
  if [ ! -f "$dir/final.pt" ]; then
    # Remove stale SKIPPED flag
    rm -f "$dir/SKIPPED_EARLY_TERM"

    echo "[$(date -u +%FT%TZ)] s${s}: starting training" | tee -a "$LOG"
    # Train with retry loop
    attempt=0
    while [ $attempt -lt 15 ]; do
      [ -f "$dir/final.pt" ] && break
      attempt=$((attempt+1))
      echo "[$(date -u +%FT%TZ)] s${s}: attempt $attempt" | tee -a "$LOG"
      python -u -m pufferlib_market.train \
        --data-path "$TRAIN" --val-data-path "$VAL" \
        --total-timesteps 15000000 --max-steps 252 \
        --trade-penalty 0.03 --hidden-size 1024 \
        --anneal-lr --disable-shorts --val-eval-windows 50 \
        --optimizer muon --num-envs 128 \
        --seed "$s" --checkpoint-dir "$dir" \
        >> "$dir/train.log" 2>&1
      exit_code=$?
      echo "[$(date -u +%FT%TZ)] s${s}: attempt $attempt exit=$exit_code" | tee -a "$LOG"
    done

    if [ ! -f "$dir/final.pt" ]; then
      echo "[$(date -u +%FT%TZ)] s${s}: FAILED after $attempt attempts" | tee -a "$LOG"
      continue
    fi
    echo "[$(date -u +%FT%TZ)] s${s}: training complete" | tee -a "$LOG"
  else
    echo "[$(date -u +%FT%TZ)] s${s}: has final.pt, running eval" | tee -a "$LOG"
  fi

  # Run OOS eval
  ckpt="$dir/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$dir/best.pt"
  [ -f "$ckpt" ] || { echo "  s${s}: no checkpoint for eval" | tee -a "$LOG"; continue; }

  python -m pufferlib_market.evaluate_holdout \
    --checkpoint "$ckpt" --data-path "$FULL_VAL" \
    --eval-hours 50 --n-windows 100 --fee-rate 0.001 \
    --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop \
    > "$dir/eval_full.json" 2>/dev/null

  if [ -s "$dir/eval_full.json" ]; then
    result=$(python3 -c "
import json
d = json.load(open('$dir/eval_full.json'))
neg = d.get('negative_windows',0)
med = d.get('median_total_return',0)*100
p10 = d.get('p10_total_return',0)*100
sort = d.get('median_sortino',0)
print(f'I_s${s}: neg={neg}/100 med={med:.2f}% p10={p10:.2f}% sort={sort:.2f}')
" 2>/dev/null)
    echo "[$(date -u +%FT%TZ)] $result" | tee -a "$LOG"

    # Log to leaderboard
    ts=$(date -u +%FT%TZ)
    python3 -c "
import json
d = json.load(open('$dir/eval_full.json'))
med = d.get('median_total_return',0)*100
p10 = d.get('p10_total_return',0)*100
worst = (d.get('worst_window') or {}).get('total_return',0)*100
neg = d.get('negative_windows',0)
sort = d.get('median_sortino',0)
print(f'${ts},I,${s},{med:.2f},{p10:.2f},{worst:.2f},{neg},{sort:.2f},${ckpt}')
" 2>/dev/null >> "$CKPT_ROOT/leaderboard_fulloos.csv"
  fi
done

echo "[$(date -u +%FT%TZ)] Runner complete" | tee -a "$LOG"
