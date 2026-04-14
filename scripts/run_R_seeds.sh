#!/bin/bash
# R variant sequential runner — Muon tp=0.05, validates on recent bear market (Dec2025-Apr2026)
# Seeds with low neg on full OOS eval might be bear-market-robust AND bull-market-capable
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate
export TMPDIR="$(pwd)/.tmp_train"
export TRITON_CACHE_DIR="$(pwd)/.tmp_train/triton_cache"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

TRAIN="pufferlib_market/data/screened32_augmented_train.bin"
VAL="pufferlib_market/data/screened32_recent_val.bin"   # bear market val
FULL_VAL="pufferlib_market/data/screened32_single_offset_val_full.bin"
CKPT_ROOT="pufferlib_market/checkpoints/screened32_sweep/R"
LOG="$CKPT_ROOT/runner_v2.log"
SEEDS_LIST="${SEEDS:-7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25}"

mkdir -p "$CKPT_ROOT"
echo "[$(date -u +%FT%TZ)] Starting R runner: seeds=$SEEDS_LIST" >> "$LOG"

for s in $SEEDS_LIST; do
  dir="$CKPT_ROOT/s${s}"; mkdir -p "$dir"

  if [ -f "$dir/eval_full.json" ] && [ -s "$dir/eval_full.json" ]; then
    echo "[$(date -u +%FT%TZ)] s${s}: already evaled, skip" | tee -a "$LOG"
    continue
  fi

  rm -f "$dir/SKIPPED_EARLY_TERM"

  if [ ! -f "$dir/final.pt" ]; then
    echo "[$(date -u +%FT%TZ)] s${s}: training" | tee -a "$LOG"
    attempt=0
    while [ $attempt -lt 15 ]; do
      [ -f "$dir/final.pt" ] && break
      attempt=$((attempt+1))
      echo "[$(date -u +%FT%TZ)] s${s}: attempt $attempt" | tee -a "$LOG"
      python -u -m pufferlib_market.train \
        --data-path "$TRAIN" --val-data-path "$VAL" \
        --total-timesteps 15000000 --max-steps 252 \
        --trade-penalty 0.05 --hidden-size 1024 \
        --anneal-lr --disable-shorts --val-eval-windows 50 \
        --optimizer muon --num-envs 128 \
        --seed "$s" --checkpoint-dir "$dir" \
        >> "$dir/train.log" 2>&1
      echo "[$(date -u +%FT%TZ)] s${s}: attempt $attempt exit=$?" | tee -a "$LOG"
    done
  fi

  if [ ! -f "$dir/final.pt" ]; then
    echo "[$(date -u +%FT%TZ)] s${s}: FAILED" | tee -a "$LOG"
    continue
  fi

  ckpt="$dir/val_best.pt"; [ -f "$ckpt" ] || ckpt="$dir/best.pt"
  [ -f "$ckpt" ] || { echo "[$(date -u +%FT%TZ)] s${s}: no ckpt" | tee -a "$LOG"; continue; }

  python -m pufferlib_market.evaluate_holdout \
    --checkpoint "$ckpt" --data-path "$FULL_VAL" \
    --eval-hours 50 --n-windows 100 --fee-rate 0.001 \
    --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop \
    > "$dir/eval_full.json" 2>/dev/null

  if [ -s "$dir/eval_full.json" ]; then
    python3 -c "
import json
d = json.load(open('$dir/eval_full.json'))
neg = d.get('negative_windows', 0)
med = d.get('median_total_return', 0) * 100
p10 = d.get('p10_total_return', 0) * 100
print(f'R_s${s}: neg={neg}/100 med={med:.2f}% p10={p10:.2f}%', '<<ANCHOR' if neg < 25 else '')
" | tee -a "$LOG"
  fi
done

echo "[$(date -u +%FT%TZ)] R runner complete" | tee -a "$LOG"
