#!/bin/bash
# I seed runner v3 - uses setsid to isolate training process from runner shell
# This prevents OOM kills of the training process from propagating to the runner
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate
export TMPDIR="$(pwd)/.tmp_train"
export TRITON_CACHE_DIR="$(pwd)/.tmp_train/triton_cache"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

TRAIN="pufferlib_market/data/screened32_augmented_train.bin"
VAL="pufferlib_market/data/screened32_augmented_val.bin"
FULL_VAL="pufferlib_market/data/screened32_single_offset_val_full.bin"
CKPT_ROOT="pufferlib_market/checkpoints/screened32_sweep/I"
LOG="$CKPT_ROOT/runner_v3.log"

SEEDS_LIST="${SEEDS:-25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50}"
GPU_FREE_THRESH_MB="${GPU_FREE_MB:-14000}"

echo "[$(date -u +%FT%TZ)] Starting I runner v3: seeds=$SEEDS_LIST" >> "$LOG"

wait_gpu() {
  local waited=0
  while true; do
    local free_mb
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    [ -z "$free_mb" ] && return 0
    [ "$free_mb" -ge "$GPU_FREE_THRESH_MB" ] && return 0
    [ "$waited" -ge 600 ] && return 0
    echo "[$(date -u +%FT%TZ)] GPU busy (free=${free_mb}MB), waiting 30s..." >> "$LOG"
    sleep 30
    waited=$((waited + 30))
  done
}

for s in $SEEDS_LIST; do
  dir="$CKPT_ROOT/s${s}"
  mkdir -p "$dir"

  # Skip if already has good eval
  if [ -s "$dir/eval_full.json" ]; then
    neg=$(python3 -c "import json; d=json.load(open('$dir/eval_full.json')); print(d.get('negative_windows',99))" 2>/dev/null)
    if [ -n "$neg" ] && [ "$neg" -lt 90 ]; then
      echo "[$(date -u +%FT%TZ)] s${s}: done (neg=$neg), skip" >> "$LOG"
      continue
    fi
    rm -f "$dir/eval_full.json"
  fi

  rm -f "$dir/SKIPPED_EARLY_TERM"

  if [ ! -f "$dir/final.pt" ]; then
    echo "[$(date -u +%FT%TZ)] s${s}: training" >> "$LOG"
    attempt=0
    while [ $attempt -lt 20 ]; do
      [ -f "$dir/final.pt" ] && break
      attempt=$((attempt+1))
      wait_gpu
      echo "[$(date -u +%FT%TZ)] s${s}: attempt $attempt" >> "$LOG"

      # Run training in isolated process group via setsid
      # Redirect stdout/stderr fully to train.log
      setsid python -u -m pufferlib_market.train \
        --data-path "$TRAIN" --val-data-path "$VAL" \
        --total-timesteps 15000000 --max-steps 252 \
        --trade-penalty 0.03 --hidden-size 1024 \
        --anneal-lr --disable-shorts --val-eval-windows 50 \
        --optimizer muon --num-envs 128 \
        --seed "$s" --checkpoint-dir "$dir" \
        >> "$dir/train.log" 2>&1
      exit_code=$?
      echo "[$(date -u +%FT%TZ)] s${s}: attempt $attempt exit=$exit_code" >> "$LOG"

      # Log latest val
      grep "\[val\]" "$dir/train.log" 2>/dev/null | tail -1 >> "$LOG"
    done

    if [ ! -f "$dir/final.pt" ]; then
      echo "[$(date -u +%FT%TZ)] s${s}: FAILED" >> "$LOG"
      continue
    fi
    echo "[$(date -u +%FT%TZ)] s${s}: training done" >> "$LOG"
  else
    echo "[$(date -u +%FT%TZ)] s${s}: has final.pt" >> "$LOG"
  fi

  # Show val trajectory
  grep "\[val\]" "$dir/train.log" 2>/dev/null >> "$LOG"

  # Evaluate on full OOS
  ckpt="$dir/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$dir/best.pt"
  [ -f "$ckpt" ] || { echo "s${s}: no ckpt" >> "$LOG"; continue; }

  python -m pufferlib_market.evaluate_holdout \
    --checkpoint "$ckpt" --data-path "$FULL_VAL" \
    --eval-hours 50 --n-windows 100 --fee-rate 0.001 \
    --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop \
    > "$dir/eval_full.json" 2>/dev/null

  if [ -s "$dir/eval_full.json" ]; then
    neg=$(python3 -c "import json; d=json.load(open('$dir/eval_full.json')); print(d.get('negative_windows',99))" 2>/dev/null)
    med=$(python3 -c "import json; d=json.load(open('$dir/eval_full.json')); print(f'{d.get(\"median_total_return\",0)*100:.2f}')" 2>/dev/null)
    echo "[$(date -u +%FT%TZ)] I/s$s OOS: neg=$neg med=$med%" >> "$LOG"

    ts=$(date -u +%FT%TZ)
    python3 -c "
import json
d = json.load(open('$dir/eval_full.json'))
med = d.get('median_total_return',0)*100
p10 = d.get('p10_total_return',0)*100
worst = (d.get('worst_window') or {}).get('total_return',0)*100
neg = d.get('negative_windows',0)
sort = d.get('median_sortino',0)
print(f'$ts,I,$s,{med:.2f},{p10:.2f},{worst:.2f},{neg},{sort:.2f},$ckpt')
" 2>/dev/null >> "$CKPT_ROOT/leaderboard_fulloos.csv"
  fi
done

echo "[$(date -u +%FT%TZ)] I runner v3 complete" >> "$LOG"
