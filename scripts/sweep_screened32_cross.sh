#!/bin/bash
# Seed sweep for screened32_cross augmented training (20 features: 16 base + 4 cross-symbol).
# Cross features add: corr_spy, beta_spy, relative_return, breadth_rank
#
# Build dataset first:
#   python scripts/build_screened_augmented.py --cross-features
#
# Then run:
#   nohup bash scripts/sweep_screened32_cross.sh C > /tmp/s32c_C.log 2>&1 &
#   nohup bash scripts/sweep_screened32_cross.sh D > /tmp/s32c_D.log 2>&1 &

set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

export TMPDIR="$(pwd)/.tmp_train"
mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="$(pwd)/.tmp_train/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

VARIANT="${1:-C}"
TRAIN="pufferlib_market/data/screened32_cross_augmented_train.bin"
VAL="pufferlib_market/data/screened32_cross_augmented_val.bin"
FULL_VAL="pufferlib_market/data/screened32_cross_single_offset_val_full.bin"  # 20-feat full OOS val

if [ ! -f "$TRAIN" ] || [ ! -f "$VAL" ]; then
  echo "ERROR: Cross-features dataset not found. Build first:"
  echo "  python scripts/build_screened_augmented.py --cross-features"
  exit 1
fi

nts=$(python3 -c "import struct; h=open('$TRAIN','rb').read(24); v=struct.unpack_from('<IIII',h,4); print(f'{v[1]} syms, {v[2]} ts, {v[3]} feats')" 2>/dev/null || echo "?")
nv=$(python3 -c "import struct; h=open('$VAL','rb').read(24); v=struct.unpack_from('<IIII',h,4); print(f'{v[2]} ts')" 2>/dev/null || echo "?")
echo "Train: $TRAIN ($nts)   Val: $VAL ($nv)"

case "$VARIANT" in
  C) TP=0.02; STEPS=15000000; EXTRA_FLAGS=()                      ;;
  D) TP=0.05; STEPS=15000000; EXTRA_FLAGS=(--optimizer muon)      ;;
  V) TP=0.05; STEPS=15000000; EXTRA_FLAGS=(--optimizer muon --arch transformer --hidden-size 512) ;;
  *) echo "Unknown variant $VARIANT (use C D V)"; exit 1 ;;
esac

SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20}"
CKPT_ROOT="pufferlib_market/checkpoints/screened32_cross_sweep/${VARIANT}"
LOG="${CKPT_ROOT}/leaderboard_fulloos.csv"
mkdir -p "$CKPT_ROOT"
[ -f "$LOG" ] || echo "timestamp,variant,seed,med_pct,p10_pct,worst_pct,neg_count,med_sortino,checkpoint" > "$LOG"

train_one() {
  local seed=$1
  local dir="${CKPT_ROOT}/s${seed}"
  mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] [${VARIANT}_cross] seed ${seed}"
  python -u -m pufferlib_market.train \
      --data-path "$TRAIN" --val-data-path "$VAL" \
      --total-timesteps "$STEPS" \
      --max-steps 252 --trade-penalty "$TP" --hidden-size 1024 \
      --anneal-lr --disable-shorts --val-eval-windows 50 \
      --early-stop-val-neg-threshold "${EARLY_STOP_NEG:-25}" \
      --early-stop-val-neg-patience "${EARLY_STOP_PATIENCE:-2}" \
      --num-envs 128 "${EXTRA_FLAGS[@]}" \
      --seed "$seed" --checkpoint-dir "$dir" > "$dir/train.log" 2>&1
}

eval_one() {
  local seed=$1
  local ckpt="${CKPT_ROOT}/s${seed}/val_best.pt"
  [ -f "$ckpt" ] || ckpt="${CKPT_ROOT}/s${seed}/best.pt"
  [ -f "$ckpt" ] || { echo "  s${seed}: no ckpt"; return; }
  local out="${CKPT_ROOT}/s${seed}/eval_fulloos.json"
  python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" --data-path "$FULL_VAL" \
      --eval-hours 50 --n-windows 100 \
      --fee-rate 0.001 --fill-buffer-bps 5.0 \
      --decision-lag 2 --deterministic --no-early-stop \
      > "$out" 2>/dev/null || { echo "  s${seed}: eval failed"; return; }
  stats=$(python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{d.get('median_total_return',0)*100:.2f},{d.get('p10_total_return',0)*100:.2f},{(d.get('worst_window') or {}).get('total_return',0)*100:.2f},{d.get('negative_windows',0)},{d.get('median_sortino',0):.2f}")
PY
  )
  [ -z "$stats" ] && return
  ts=$(date -u +%FT%TZ)
  echo "$ts,$VARIANT,$seed,$stats,$ckpt" >> "$LOG"
  echo "  s${seed}: ${stats}"
}

echo "=== screened32 CROSS sweep: variant=${VARIANT}, tp=${TP} ==="
for s in $SEEDS; do
  [ -f "${CKPT_ROOT}/s${s}/eval_fulloos.json" ] && [ -s "${CKPT_ROOT}/s${s}/eval_fulloos.json" ] && { echo "  s${s}: done, skip"; continue; }
  train_one "$s"; eval_one "$s"
done
echo "=== [${VARIANT}_cross] LEADERBOARD ==="
sort -t, -k4 -rn "$LOG" | head -10
