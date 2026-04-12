#!/bin/bash
# Seed sweep for wide73 augmented training with per-sym-norm.
#
# Best configs from stocks17 sweep:
#   C low_tp: tp=0.02, 15M steps — most stable, s31 was all-positive
#   D muon:   tp=0.05, 15M steps, muon optimizer — best median (s21=19.1%)
#   F psn_lotp: tp=0.02, 15M steps, --per-sym-norm  (new)
#   G psn_muon: tp=0.05, 15M steps, muon + --per-sym-norm (new)
#
# Usage:
#   nohup bash scripts/sweep_wide_augmented.sh C > /tmp/sw_wide_C.log 2>&1 &
#   nohup bash scripts/sweep_wide_augmented.sh F > /tmp/sw_wide_F.log 2>&1 &
#   nohup bash scripts/sweep_wide_augmented.sh G > /tmp/sw_wide_G.log 2>&1 &

set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

# Triton CUDA kernel compilation needs a stable TMPDIR (sm_120a Blackwell arch
# has a race where /tmp log files are cleaned up before Triton can read them).
export TMPDIR="$(pwd)/.tmp_train"
mkdir -p "$TMPDIR"

VARIANT="${1:-F}"
TRAIN_DATA="pufferlib_market/data/wide73_augmented_train.bin"
VAL_DATA="pufferlib_market/data/wide73_augmented_val.bin"
# Sentinel: build is done when all 5 session-offset cache files exist.
CACHE_DIR="pufferlib_market/data/.build_cache_wide73_augmented_train"
BUILD_SENTINEL="${CACHE_DIR}/offset4_train.bin"

# Wait for data to be fully built (all 5 offsets completed).
if [ ! -f "$BUILD_SENTINEL" ]; then
  echo "Waiting for full build (need $BUILD_SENTINEL)..."
  while [ ! -f "$BUILD_SENTINEL" ]; do sleep 60; done
  echo "Build complete — starting sweep."
fi

case "$VARIANT" in
  C) LABEL="low_tp";      TP=0.02; STEPS=15000000; MAX_EP=252; EXTRA_FLAGS=() ;;
  D) LABEL="muon";        TP=0.05; STEPS=15000000; MAX_EP=252; EXTRA_FLAGS=(--optimizer muon) ;;
  F) LABEL="psn_lotp";    TP=0.02; STEPS=15000000; MAX_EP=252; EXTRA_FLAGS=(--per-sym-norm) ;;
  G) LABEL="psn_muon";    TP=0.05; STEPS=15000000; MAX_EP=252; EXTRA_FLAGS=(--optimizer muon --per-sym-norm) ;;
  H) LABEL="psn_longtrain"; TP=0.02; STEPS=30000000; MAX_EP=252; EXTRA_FLAGS=(--per-sym-norm) ;;
  *) echo "Unknown variant $VARIANT (use C/D/F/G/H)"; exit 1 ;;
esac

# Seeds: broad sweep — 20 seeds per variant to find phase transitions
if [ -z "${SEEDS:-}" ]; then
  SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
fi

CKPT_ROOT="pufferlib_market/checkpoints/wide73_sweep/${VARIANT}_${LABEL}"
LOG="$CKPT_ROOT/leaderboard.csv"
mkdir -p "$CKPT_ROOT"

echo "timestamp,variant,seed,med_pct,p10_pct,worst_pct,neg_count,med_sortino,checkpoint" > "$LOG"

train_one() {
  local seed=$1
  local dir="$CKPT_ROOT/s${seed}"
  mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] [$VARIANT/$LABEL] train seed $seed → $dir"
  python -u -m pufferlib_market.train \
      --data-path       "$TRAIN_DATA" \
      --val-data-path   "$VAL_DATA" \
      --total-timesteps "$STEPS" \
      --max-steps       "$MAX_EP" \
      --trade-penalty   "$TP" \
      --hidden-size     1024 \
      --anneal-lr \
      --disable-shorts \
      "${EXTRA_FLAGS[@]}" \
      --num-envs        128 \
      --seed            "$seed" \
      --checkpoint-dir  "$dir" \
      > "$dir/train.log" 2>&1
  echo "[$(date -u +%FT%TZ)] [$VARIANT/$LABEL] done seed $seed (exit=$?)"
}

eval_one() {
  local seed=$1
  local ckpt="$CKPT_ROOT/s${seed}/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$CKPT_ROOT/s${seed}/best.pt"
  [ -f "$ckpt" ] || { echo "  s${seed}: no checkpoint"; return; }
  local out="$CKPT_ROOT/s${seed}/eval_lag2.json"
  echo "  s${seed}: evaluating lag=2, 50 windows ..."
  python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" --data-path "$VAL_DATA" \
      --eval-hours 60 --n-windows 50 --fee-rate 0.001 \
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
  echo "  s${seed}: med=${stats%%,*}% | full=$stats"
}

echo "=== Variant $VARIANT/$LABEL : tp=$TP steps=$STEPS max_ep=$MAX_EP extra=${EXTRA_FLAGS[*]} ==="
nts=$(python3 -c "import struct; h=open('$TRAIN_DATA','rb').read(24); v=struct.unpack_from('<IIII',h,4); print(f'{v[1]} syms, {v[2]} steps, {v[3]} feats/sym')" 2>/dev/null || echo "?")
echo "    train=$TRAIN_DATA ($nts)"
echo

for s in "${SEEDS[@]}"; do train_one "$s"; eval_one "$s"; done

echo
echo "=== [$VARIANT] LEADERBOARD ==="
sort -t, -k4 -rn "$LOG" | head -10
echo
echo "=== Positive-p10 models (p10>0) ==="
awk -F, 'NR>1 && $5+0>0 {print}' "$LOG" | sort -t, -k4 -rn
