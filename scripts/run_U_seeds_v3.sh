#!/bin/bash
# U seed runner v3 - setsid isolation, GPU-aware
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate
export TMPDIR="$(pwd)/.tmp_train"
export TRITON_CACHE_DIR="$(pwd)/.tmp_train/triton_cache"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

TRAIN="pufferlib_market/data/screened32_augmented_train.bin"
VAL="pufferlib_market/data/screened32_augmented_val.bin"
FULL_VAL="pufferlib_market/data/screened32_single_offset_val_full.bin"
CKPT_ROOT="pufferlib_market/checkpoints/screened32_sweep/U"
LOG="$CKPT_ROOT/runner_v3.log"

SEEDS_LIST="${SEEDS:-7 8 9 10 11 12 13 14 15 16 17 18 19 20}"
GPU_FREE_THRESH_MB="${GPU_FREE_MB:-15000}"

echo "[$(date -u +%FT%TZ)] Starting U runner v3: seeds=$SEEDS_LIST" >> "$LOG"

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

  if [ -s "$dir/eval_full.json" ]; then
    is_dummy=$(python3 -c "import json; d=json.load(open('$dir/eval_full.json')); print('yes' if d.get('dummy') else 'no')" 2>/dev/null)
    if [ "$is_dummy" = "yes" ]; then
      rm -f "$dir/eval_full.json"
    else
      neg=$(python3 -c "import json; d=json.load(open('$dir/eval_full.json')); print(d.get('negative_windows',99))" 2>/dev/null)
      if [ -n "$neg" ] && [ "$neg" -lt 90 ]; then
        echo "[$(date -u +%FT%TZ)] s${s}: done (neg=$neg), skip" >> "$LOG"
        continue
      fi
      rm -f "$dir/eval_full.json"
    fi
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

      setsid python -u -m pufferlib_market.train \
        --data-path "$TRAIN" --val-data-path "$VAL" \
        --total-timesteps 15000000 --max-steps 252 \
        --trade-penalty 0.03 --hidden-size 1024 \
        --anneal-lr --disable-shorts --val-eval-windows 50 \
        --optimizer muon --downside-penalty 0.05 \
        --num-envs 128 \
        --seed "$s" --checkpoint-dir "$dir" \
        >> "$dir/train.log" 2>&1
      exit_code=$?
      echo "[$(date -u +%FT%TZ)] s${s}: attempt $attempt exit=$exit_code" >> "$LOG"
      grep "\[val\]" "$dir/train.log" 2>/dev/null | tail -1 >> "$LOG"
    done

    if [ ! -f "$dir/final.pt" ]; then
      echo "[$(date -u +%FT%TZ)] s${s}: FAILED" >> "$LOG"
      continue
    fi
    echo "[$(date -u +%FT%TZ)] s${s}: done" >> "$LOG"
  else
    echo "[$(date -u +%FT%TZ)] s${s}: has final.pt" >> "$LOG"
  fi

  grep "\[val\]" "$dir/train.log" 2>/dev/null >> "$LOG"

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
    echo "[$(date -u +%FT%TZ)] U/s$s OOS: neg=$neg med=$med%" >> "$LOG"
    ts=$(date -u +%FT%TZ)
    python3 -c "
import json
d = json.load(open('$dir/eval_full.json'))
med = d.get('median_total_return',0)*100
p10 = d.get('p10_total_return',0)*100
worst = (d.get('worst_window') or {}).get('total_return',0)*100
neg = d.get('negative_windows',0)
sort = d.get('median_sortino',0)
print(f'$ts,U,$s,{med:.2f},{p10:.2f},{worst:.2f},{neg},{sort:.2f},$ckpt')
" 2>/dev/null >> "$CKPT_ROOT/leaderboard_fulloos.csv"
  fi
done

echo "[$(date -u +%FT%TZ)] U runner v3 complete" >> "$LOG"
