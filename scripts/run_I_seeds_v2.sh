#!/bin/bash
# I seed runner v2 - waits for GPU headroom before launching training
# Skips seeds that already have good eval_full.json (neg < 90)
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate
export TMPDIR="$(pwd)/.tmp_train"
export TRITON_CACHE_DIR="$(pwd)/.tmp_train/triton_cache"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

TRAIN="pufferlib_market/data/screened32_augmented_train.bin"
VAL="pufferlib_market/data/screened32_augmented_val.bin"
FULL_VAL="pufferlib_market/data/screened32_single_offset_val_full.bin"
CKPT_ROOT="pufferlib_market/checkpoints/screened32_sweep/I"
LOG="$CKPT_ROOT/runner_v2.log"

SEEDS_LIST="${SEEDS:-25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50}"
GPU_FREE_THRESH_MB="${GPU_FREE_MB:-16000}"  # need at least 16GB free before launching

echo "[$(date -u +%FT%TZ)] Starting I runner v2: seeds=$SEEDS_LIST, gpu_thresh=${GPU_FREE_THRESH_MB}MB" >> "$LOG"

wait_gpu() {
  local max_wait=600  # max 10 minutes
  local waited=0
  while true; do
    local free_mb
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -z "$free_mb" ] || [ "$free_mb" -ge "$GPU_FREE_THRESH_MB" ]; then
      return 0
    fi
    if [ "$waited" -ge "$max_wait" ]; then
      echo "[$(date -u +%FT%TZ)] GPU wait timeout (free=${free_mb}MB), proceeding anyway" | tee -a "$LOG"
      return 0
    fi
    echo "[$(date -u +%FT%TZ)] GPU busy (free=${free_mb}MB < ${GPU_FREE_THRESH_MB}MB), waiting 30s..." | tee -a "$LOG"
    sleep 30
    waited=$((waited + 30))
  done
}

for s in $SEEDS_LIST; do
  dir="$CKPT_ROOT/s${s}"
  mkdir -p "$dir"

  # Skip if already has good eval (neg < 90)
  if [ -s "$dir/eval_full.json" ]; then
    neg=$(python3 -c "import json; d=json.load(open('$dir/eval_full.json')); print(d.get('negative_windows',99))" 2>/dev/null)
    if [ "$neg" != "" ] && [ "$neg" -lt 90 ]; then
      echo "[$(date -u +%FT%TZ)] s${s}: already evaluated (neg=$neg), skip" | tee -a "$LOG"
      continue
    else
      echo "[$(date -u +%FT%TZ)] s${s}: bad eval (neg=$neg), rerunning" | tee -a "$LOG"
      rm -f "$dir/eval_full.json"
    fi
  fi

  rm -f "$dir/SKIPPED_EARLY_TERM"

  if [ ! -f "$dir/final.pt" ]; then
    echo "[$(date -u +%FT%TZ)] s${s}: starting training" | tee -a "$LOG"
    attempt=0
    while [ $attempt -lt 20 ]; do
      [ -f "$dir/final.pt" ] && break
      attempt=$((attempt+1))

      # Wait for GPU headroom
      wait_gpu

      echo "[$(date -u +%FT%TZ)] s${s}: attempt $attempt (GPU ready)" | tee -a "$LOG"
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

      # Show val trajectory if available
      last_val=$(grep "\[val\]" "$dir/train.log" 2>/dev/null | tail -1)
      [ -n "$last_val" ] && echo "[$(date -u +%FT%TZ)] s${s}: $last_val" | tee -a "$LOG"
    done

    if [ ! -f "$dir/final.pt" ]; then
      echo "[$(date -u +%FT%TZ)] s${s}: FAILED after $attempt attempts" | tee -a "$LOG"
      continue
    fi
    echo "[$(date -u +%FT%TZ)] s${s}: training complete" | tee -a "$LOG"
  else
    echo "[$(date -u +%FT%TZ)] s${s}: has final.pt, running eval" | tee -a "$LOG"
  fi

  ckpt="$dir/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$dir/best.pt"
  [ -f "$ckpt" ] || { echo "  s${s}: no checkpoint for eval" | tee -a "$LOG"; continue; }

  # Show val trajectory
  echo "[$(date -u +%FT%TZ)] s${s}: val trajectory:" | tee -a "$LOG"
  grep "\[val\]" "$dir/train.log" 2>/dev/null | tee -a "$LOG"

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

echo "[$(date -u +%FT%TZ)] Runner v2 complete" | tee -a "$LOG"
