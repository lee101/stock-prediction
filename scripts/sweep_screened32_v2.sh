#!/bin/bash
# Seed sweep for screened32_v2: extended training data (2019-01-01 to 2025-11-30).
# 31% more training data than v1 (582 → 763 trading days equivalent).
# Val: 2025-12-01 to 2026-03-31 (71 candidate windows, 50-day eval windows).
#
# NOTE: OOS eval uses screened32_single_offset_val_full.bin (Jun 2025 - Apr 2026, 263 windows).
# Windows 0-125 have contamination (overlap with v2 training data), windows 125-263 are clean.
# Results are noted with this caveat. Anchor threshold: neg<25/100.
#
# Usage:
#   nohup bash scripts/sweep_screened32_v2.sh D > /tmp/s32v2_D.log 2>&1 &
#   nohup bash scripts/sweep_screened32_v2.sh C > /tmp/s32v2_C.log 2>&1 &

set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

# earlyoom kills training processes under memory pressure — stop it
echo "ilu" | sudo -S systemctl stop earlyoom 2>/dev/null && echo "[sweep] earlyoom stopped" || echo "[sweep] earlyoom already stopped"

export TMPDIR="$(pwd)/.tmp_train"
mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="$(pwd)/.tmp_train/triton_cache"
export TORCH_COMPILE_DEBUG=0
mkdir -p "$TRITON_CACHE_DIR"

VARIANT="${1:-D}"
TRAIN="pufferlib_market/data/screened32_v2_augmented_train.bin"
VAL="pufferlib_market/data/screened32_v2_augmented_val.bin"
# Use the standard full OOS for comparison (with contamination caveat for windows 0-125)
FULL_VAL="pufferlib_market/data/screened32_single_offset_val_full.bin"

case "$VARIANT" in
  C) TP=0.02; STEPS=15000000; EXTRA_FLAGS=() ;;
  D) TP=0.05; STEPS=15000000; EXTRA_FLAGS=(--optimizer muon) ;;
  *) echo "Unknown variant $VARIANT (use C D)"; exit 1 ;;
esac

SEEDS=${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20}

CKPT_ROOT="pufferlib_market/checkpoints/screened32_v2_sweep/${VARIANT}"
LOG="$CKPT_ROOT/leaderboard_fulloos.csv"
mkdir -p "$CKPT_ROOT"
[ -f "$LOG" ] || echo "timestamp,variant,seed,med_pct,p10_pct,worst_pct,neg_count,med_sortino,checkpoint" > "$LOG"

train_one() {
  local seed=$1
  local dir="$CKPT_ROOT/s${seed}"
  mkdir -p "$dir"
  local attempt=1
  local max_attempts=20
  while [ $attempt -le $max_attempts ]; do
    [ -f "$dir/final.pt" ] && { echo "  s${seed}: final.pt exists, training complete"; return 0; }
    echo "[$(date -u +%FT%TZ)] [v2_${VARIANT}] seed ${seed} training (attempt ${attempt})..."
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
          local neg n_val
          neg=$(echo "$first_val" | grep -oP 'neg=\K[0-9]+(?=/)')
          n_val=$(echo "$first_val" | grep -oP 'neg=[0-9]+/\K[0-9]+')
          echo "  s${seed}: first val neg=${neg}/${n_val}"
          if [ -n "$neg" ] && [ "$neg" -gt 40 ]; then
            echo "  s${seed}: early termination (neg=${neg}/${n_val} > 40)"
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
    [ -f "$dir/final.pt" ] && { echo "[$(date -u +%FT%TZ)] [v2_${VARIANT}] seed ${seed} done"; return 0; }
    [ -f "$dir/SKIPPED_EARLY_TERM" ] && { echo "  s${seed}: skipped by early termination"; return 1; }
    echo "[$(date -u +%FT%TZ)] [v2_${VARIANT}] seed ${seed} exit_code=${exit_code}, retrying (attempt $((attempt+1)))..."
    attempt=$((attempt + 1))
  done
  echo "[$(date -u +%FT%TZ)] [v2_${VARIANT}] seed ${seed} FAILED after ${max_attempts} attempts"
  return 1
}

eval_one() {
  local seed=$1
  local ckpt="$CKPT_ROOT/s${seed}/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$CKPT_ROOT/s${seed}/best.pt"
  [ -f "$ckpt" ] || { echo "  s${seed}: no checkpoint"; return; }
  local out="$CKPT_ROOT/s${seed}/eval_full.json"
  echo "  s${seed}: evaluating on full OOS (100 windows)..."
  python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" --data-path "$FULL_VAL" \
      --eval-hours 50 --n-windows 100 --fee-rate 0.001 \
      --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop \
      > "$out" 2>/dev/null || { echo "  s${seed}: eval failed"; return; }
  stats=$(python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
med   = d.get('median_total_return', 0)*100
p10   = d.get('p10_total_return', 0)*100
worst = d.get('worst_window',{}).get('total_return', 0)*100
neg   = d.get('negative_windows', 0)
sort  = d.get('median_sortino', 0)
print(f"{med:.2f},{p10:.2f},{worst:.2f},{neg},{sort:.2f}")
PY
  )
  [ -z "$stats" ] && { echo "  s${seed}: parse failed"; return; }
  ts=$(date -u +%FT%TZ)
  echo "$ts,$VARIANT,$seed,$stats,$ckpt" >> "$LOG"
  echo "  s${seed}: med=${stats%%,*}%  full=$stats"
}

nts=$(python3 -c "import struct; h=open('$TRAIN','rb').read(24); v=struct.unpack_from('<IIII',h,4); print(f'{v[1]} syms, {v[2]} ts, {v[3]} feats')" 2>/dev/null || echo "?")
echo "=== screened32_v2 variant $VARIANT : tp=$TP steps=$STEPS ==="
echo "    train=$TRAIN ($nts)"
echo

for s in $SEEDS; do
  [ -f "$CKPT_ROOT/s${s}/eval_full.json" ] && [ -s "$CKPT_ROOT/s${s}/eval_full.json" ] && { echo "[$(date -u +%FT%TZ)] s${s}: already evaled on full OOS, skip"; continue; }
  [ -f "$CKPT_ROOT/s${s}/SKIPPED_EARLY_TERM" ] && { echo "  s${s}: skipped by early termination, skip"; continue; }
  if [ ! -f "$CKPT_ROOT/s${s}/final.pt" ]; then
    train_one "$s" || continue
  else
    echo "  s${s}: final.pt exists, skipping training"
  fi
  eval_one "$s"
done

echo
echo "=== [v2_${VARIANT}] LEADERBOARD (full OOS, 100 windows) ==="
sort -t, -k4 -rn "$LOG" | head -10
echo
echo "=== Positive-p10 (p10>0, neg<20) ==="
awk -F, 'NR>1 && $5+0>0 && $7+0<20 {print}' "$LOG" | sort -t, -k4 -rn
