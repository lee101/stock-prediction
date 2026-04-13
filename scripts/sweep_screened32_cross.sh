#!/bin/bash
# Cross-features screened32 sweep: 20 features/symbol (16 base + 4 cross-symbol)
# Train: 2019-01-01 to 2025-05-31, Val: 2025-06-01 to 2025-11-30 (35x augmentation)
# OOS eval: screened32_cross_single_offset_val_full.bin (Jun 2025 - Apr 2026, 313 days)
#
# Usage:
#   nohup bash scripts/sweep_screened32_cross.sh C > /tmp/s32cross_C.log 2>&1 &
#   nohup bash scripts/sweep_screened32_cross.sh D > /tmp/s32cross_D.log 2>&1 &

set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

export TMPDIR="$(pwd)/.tmp_train"
mkdir -p "$TMPDIR"
# Ensure triton and torch.compile write to our tmpdir (avoids /tmp cleanup issues)
export TRITON_CACHE_DIR="$(pwd)/.tmp_train/triton_cache"
export TORCH_COMPILE_DEBUG=0
mkdir -p "$TRITON_CACHE_DIR"

VARIANT="${1:-D}"
TRAIN="pufferlib_market/data/screened32_cross_augmented_train.bin"
VAL="pufferlib_market/data/screened32_cross_augmented_val.bin"
FULL_VAL="pufferlib_market/data/screened32_cross_single_offset_val_full.bin"

case "$VARIANT" in
  C) TP=0.02; STEPS=15000000; EXTRA_FLAGS=() ;;
  D) TP=0.05; STEPS=15000000; EXTRA_FLAGS=(--optimizer muon) ;;
  *) echo "Unknown variant $VARIANT"; exit 1 ;;
esac

SEEDS=${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20}
CKPT_ROOT="pufferlib_market/checkpoints/screened32_cross_sweep/${VARIANT}"
LOG="$CKPT_ROOT/leaderboard.csv"
mkdir -p "$CKPT_ROOT"
[ -f "$LOG" ] || echo "timestamp,variant,seed,med_pct,p10_pct,worst_pct,neg_count,med_sortino,checkpoint" > "$LOG"

train_one() {
  local seed=$1
  local dir="$CKPT_ROOT/s${seed}"
  mkdir -p "$dir"
  local attempt=1
  local max_attempts=20
  while [ $attempt -le $max_attempts ]; do
    # Check if final.pt already exists (training complete)
    [ -f "$dir/final.pt" ] && { echo "  s${seed}: final.pt exists, training complete"; return 0; }
    echo "[$(date -u +%FT%TZ)] [cross_${VARIANT}] seed ${seed} training (attempt ${attempt})..."
    python -u -m pufferlib_market.train \
        --data-path "$TRAIN" --val-data-path "$VAL" \
        --total-timesteps "$STEPS" --max-steps 252 \
        --trade-penalty "$TP" --hidden-size 1024 \
        --anneal-lr --disable-shorts --val-eval-windows 30 \
        "${EXTRA_FLAGS[@]}" --num-envs 128 \
        --seed "$seed" --checkpoint-dir "$dir" >> "$dir/train.log" 2>&1 &
    local train_pid=$!
    # Early termination: check first val after ~70 updates (update 50 = first val)
    local first_val_checked=0
    while kill -0 $train_pid 2>/dev/null; do
      sleep 45
      if [ "$first_val_checked" -eq 0 ]; then
        local first_val
        first_val=$(grep -m1 "\[val\]" "$dir/train.log" 2>/dev/null)
        if [ -n "$first_val" ]; then
          first_val_checked=1
          local neg n_val
          neg=$(echo "$first_val" | grep -oP 'neg=\K[0-9]+(?=/)')
          n_val=$(echo "$first_val" | grep -oP 'neg=[0-9]+/\K[0-9]+')
          echo "  s${seed}: first val neg=${neg}/${n_val}"
          if [ -n "$neg" ] && [ "$neg" -gt 25 ]; then
            echo "  s${seed}: early termination (neg=${neg}/${n_val} > 25) — skipping seed"
            kill "$train_pid" 2>/dev/null
            wait "$train_pid" 2>/dev/null
            touch "$dir/SKIPPED_EARLY_TERM"
            return 1
          fi
        fi
      fi
    done
    wait "$train_pid"
    local exit_code=$?
    [ -f "$dir/final.pt" ] && { echo "[$(date -u +%FT%TZ)] [cross_${VARIANT}] seed ${seed} done"; return 0; }
    [ -f "$dir/SKIPPED_EARLY_TERM" ] && { echo "  s${seed}: skipped by early termination"; return 1; }
    echo "[$(date -u +%FT%TZ)] [cross_${VARIANT}] seed ${seed} exit_code=${exit_code}, retrying (attempt $((attempt+1)))..."
    attempt=$((attempt + 1))
  done
  echo "[$(date -u +%FT%TZ)] [cross_${VARIANT}] seed ${seed} FAILED after ${max_attempts} attempts"
  return 1
}

eval_one() {
  local seed=$1
  local ckpt="$CKPT_ROOT/s${seed}/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$CKPT_ROOT/s${seed}/best.pt"
  [ -f "$ckpt" ] || { echo "  s${seed}: no ckpt"; return; }
  local out="$CKPT_ROOT/s${seed}/eval_full_oos.json"
  echo "  s${seed}: evaluating on full OOS..."
  python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" --data-path "$FULL_VAL" \
      --eval-hours 50 --n-windows 100 --fee-rate 0.001 \
      --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop \
      > "$out" 2>/dev/null || { echo "  s${seed}: eval failed"; return; }
  stats=$(python3 scripts/parse_eval_result.py < "$out" 2>/dev/null)
  [ -z "$stats" ] && return
  # Parse stats: "med=X% p10=Y% neg=Z/100 sortino=W"
  med=$(echo "$stats" | grep -oP 'med=\K[+-]?[\d.]+')
  p10=$(echo "$stats" | grep -oP 'p10=\K[+-]?[\d.]+')
  neg=$(echo "$stats" | grep -oP 'neg=\K[\d]+')
  sort=$(echo "$stats" | grep -oP 'sortino=\K[+-]?[\d.]+')
  worst=$(python3 -c "import json,sys; d=json.load(open('$out')); w=d.get('worst_window') or {}; print(f\"{w.get('total_return',0)*100:.2f}\")" 2>/dev/null || echo "0")
  ts=$(date -u +%FT%TZ)
  echo "$ts,$VARIANT,$seed,$med,$p10,$worst,$neg,$sort,$ckpt" >> "$LOG"
  echo "  s${seed}: $stats"
}

nts=$(python3 -c "import struct; h=open('$TRAIN','rb').read(24); v=struct.unpack_from('<IIII',h,4); print(f'{v[1]} syms, {v[2]} ts, {v[3]} feats')" 2>/dev/null || echo "?")
echo "=== screened32_cross $VARIANT: tp=$TP, train=$TRAIN ($nts) ==="

for s in $SEEDS; do
  [ -f "$CKPT_ROOT/s${s}/eval_full_oos.json" ] && { echo "  s${s}: already evaled, skip"; continue; }
  [ -f "$CKPT_ROOT/s${s}/SKIPPED_EARLY_TERM" ] && { echo "  s${s}: skipped by early termination, skip"; continue; }
  # Check if training is complete (final.pt exists)
  if [ ! -f "$CKPT_ROOT/s${s}/final.pt" ]; then
    train_one "$s" || continue  # skip eval if early-terminated or all attempts failed
  else
    echo "  s${s}: final.pt exists, skipping training"
  fi
  eval_one "$s"
done

echo "=== [cross_${VARIANT}] LEADERBOARD (full OOS, 100 windows) ==="
sort -t, -k4 -rn "$LOG" | head -10
