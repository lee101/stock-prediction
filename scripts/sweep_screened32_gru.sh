#!/bin/bash
# GRU architecture sweep for screened32.
# GRU-gated recurrent policy captures multi-day temporal patterns.
# Usage:
#   nohup bash scripts/sweep_screened32_gru.sh D > /tmp/gru_D.log 2>&1 &
set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

export TMPDIR="$(pwd)/.tmp_train"
mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="$(pwd)/.tmp_train/triton_cache"
export TORCH_COMPILE_DEBUG=0
mkdir -p "$TRITON_CACHE_DIR"

VARIANT="${1:-D}"
TRAIN="pufferlib_market/data/screened32_augmented_train.bin"
VAL="pufferlib_market/data/screened32_augmented_val.bin"
FULL_VAL="pufferlib_market/data/screened32_single_offset_val_full.bin"

case "$VARIANT" in
  C) TP=0.02; STEPS=15000000; EXTRA_FLAGS=(--arch gru) ;;
  D) TP=0.05; STEPS=15000000; EXTRA_FLAGS=(--optimizer muon --arch gru) ;;
  *) echo "Unknown variant $VARIANT"; exit 1 ;;
esac

SEEDS=${SEEDS:-1 2 3 4 5 6 7 8 9 10}

CKPT_ROOT="pufferlib_market/checkpoints/screened32_gru_sweep/${VARIANT}"
LOG="$CKPT_ROOT/leaderboard_fulloos.csv"
mkdir -p "$CKPT_ROOT"
[ -f "$LOG" ] || echo "timestamp,variant,seed,med_pct,p10_pct,worst_pct,neg_count,med_sortino,checkpoint" > "$LOG"

train_one() {
  local seed=$1
  local dir="$CKPT_ROOT/s${seed}"
  mkdir -p "$dir"
  local attempt=1
  while [ $attempt -le 10 ]; do
    [ -f "$dir/final.pt" ] && { echo "  s${seed}: done"; return 0; }
    echo "[$(date -u +%FT%TZ)] [gru_${VARIANT}] seed ${seed} training (attempt ${attempt})..."
    python -u -m pufferlib_market.train \
        --data-path "$TRAIN" --val-data-path "$VAL" \
        --total-timesteps "$STEPS" \
        --max-steps 252 \
        --trade-penalty "$TP" \
        --hidden-size 1024 \
        --anneal-lr \
        --disable-shorts \
        --val-eval-windows 50 \
        "${EXTRA_FLAGS[@]}" \
        --num-envs 128 \
        --seed "$seed" \
        --checkpoint-dir "$dir" >> "$dir/train.log" 2>&1 &
    local train_pid=$!
    local first_val_checked=0
    while kill -0 $train_pid 2>/dev/null; do
      sleep 45
      if [ "$first_val_checked" -eq 0 ]; then
        local first_val
        first_val=$(grep -m1 "\[val\]" "$dir/train.log" 2>/dev/null)
        if [ -n "$first_val" ]; then
          first_val_checked=1
          local neg
          neg=$(echo "$first_val" | grep -oP 'neg=\K[0-9]+(?=/)')
          echo "  s${seed}: first val neg=${neg}/50"
          if [ -n "$neg" ] && [ "$neg" -gt 40 ]; then
            echo "  s${seed}: early termination (neg=${neg} > 40)"
            kill "$train_pid" 2>/dev/null
            wait "$train_pid" 2>/dev/null
            touch "$dir/SKIPPED_EARLY_TERM"
            return 1
          fi
        fi
      fi
    done
    wait "$train_pid"
    [ -f "$dir/final.pt" ] && { echo "[$(date -u +%FT%TZ)] [gru_${VARIANT}] seed ${seed} done"; return 0; }
    [ -f "$dir/SKIPPED_EARLY_TERM" ] && return 1
    attempt=$((attempt + 1))
  done
  return 1
}

eval_one() {
  local seed=$1
  local ckpt="$CKPT_ROOT/s${seed}/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$CKPT_ROOT/s${seed}/best.pt"
  [ -f "$ckpt" ] || { echo "  s${seed}: no ckpt"; return; }
  local out="$CKPT_ROOT/s${seed}/eval_full.json"
  echo "  s${seed}: evaluating on full OOS (100 windows)..."
  python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" --data-path "$FULL_VAL" \
      --eval-hours 50 --n-windows 100 --fee-rate 0.001 \
      --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop \
      > "$out" 2>/dev/null
  if [ -s "$out" ]; then
    python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
med = d.get('median_total_return',0)*100
p10 = d.get('p10_total_return',0)*100
worst = d.get('worst_window',{}).get('total_return',0)*100
neg = d.get('negative_windows',0)
sort = d.get('median_sortino',0)
print(f"  result: med={med:.2f}% p10={p10:.2f}% neg={neg}/100 sort={sort:.2f}")
PY
    ts=$(date -u +%FT%TZ)
    stats=$(python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
med = d.get('median_total_return',0)*100
p10 = d.get('p10_total_return',0)*100
worst = d.get('worst_window',{}).get('total_return',0)*100
neg = d.get('negative_windows',0)
sort = d.get('median_sortino',0)
print(f"{med:.2f},{p10:.2f},{worst:.2f},{neg},{sort:.2f}")
PY
)
    [ -n "$stats" ] && echo "$ts,$VARIANT,$seed,$stats,$ckpt" >> "$LOG"
  fi
}

echo "=== screened32 GRU variant $VARIANT : tp=$TP steps=$STEPS ==="

for s in $SEEDS; do
  [ -f "$CKPT_ROOT/s${s}/eval_full.json" ] && [ -s "$CKPT_ROOT/s${s}/eval_full.json" ] && { echo "s${s}: already done"; continue; }
  [ -f "$CKPT_ROOT/s${s}/SKIPPED_EARLY_TERM" ] && { echo "s${s}: skipped"; continue; }
  if [ ! -f "$CKPT_ROOT/s${s}/final.pt" ]; then
    train_one "$s" || continue
  fi
  eval_one "$s"
done

echo
echo "=== [gru_${VARIANT}] LEADERBOARD ==="
sort -t, -k4 -rn "$LOG" | head -10
