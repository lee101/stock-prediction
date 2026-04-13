#!/bin/bash
# Extended screened32 sweep: train through Nov 2025, val = Dec 2025 - Apr 2026
# This validates on the recent tariff-shock bear market (toughest OOS test).
# 22,596 train timesteps with 35x augmentation.
#
# Usage:
#   nohup bash scripts/sweep_screened32_ext.sh C > /tmp/s32ext_C.log 2>&1 &
#   nohup bash scripts/sweep_screened32_ext.sh D > /tmp/s32ext_D.log 2>&1 &

set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

export TMPDIR="$(pwd)/.tmp_train"
mkdir -p "$TMPDIR"

VARIANT="${1:-C}"
TRAIN="pufferlib_market/data/screened32_ext_augmented_train.bin"
VAL="pufferlib_market/data/screened32_ext_augmented_val.bin"

case "$VARIANT" in
  C) TP=0.02; STEPS=15000000; EXTRA_FLAGS=() ;;
  D) TP=0.05; STEPS=15000000; EXTRA_FLAGS=(--optimizer muon) ;;
  *) echo "Unknown variant $VARIANT"; exit 1 ;;
esac

SEEDS=${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20}
CKPT_ROOT="pufferlib_market/checkpoints/screened32_ext_sweep/${VARIANT}"
LOG="$CKPT_ROOT/leaderboard.csv"
mkdir -p "$CKPT_ROOT"
echo "timestamp,variant,seed,med_pct,p10_pct,worst_pct,neg_count,med_sortino,checkpoint" > "$LOG"

train_one() {
  local seed=$1
  local dir="$CKPT_ROOT/s${seed}"
  mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] [${VARIANT}_ext] seed ${seed}"
  python -u -m pufferlib_market.train \
      --data-path "$TRAIN" --val-data-path "$VAL" \
      --total-timesteps "$STEPS" --max-steps 252 \
      --trade-penalty "$TP" --hidden-size 1024 \
      --anneal-lr --disable-shorts --val-eval-windows 30 \
      "${EXTRA_FLAGS[@]}" --num-envs 128 \
      --seed "$seed" --checkpoint-dir "$dir" > "$dir/train.log" 2>&1
  echo "[$(date -u +%FT%TZ)] [${VARIANT}_ext] seed ${seed} done"
}

eval_one() {
  local seed=$1
  local ckpt="$CKPT_ROOT/s${seed}/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$CKPT_ROOT/s${seed}/best.pt"
  [ -f "$ckpt" ] || { echo "  s${seed}: no ckpt"; return; }
  local out="$CKPT_ROOT/s${seed}/eval_lag2.json"
  python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" --data-path "$VAL" \
      --eval-hours 90 --n-windows 30 --fee-rate 0.001 \
      --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop \
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

nts=$(python3 -c "import struct; h=open('$TRAIN','rb').read(24); v=struct.unpack_from('<IIII',h,4); print(f'{v[1]} syms, {v[2]} ts, {v[3]} feats')" 2>/dev/null || echo "?")
echo "=== screened32_ext $VARIANT: tp=$TP, train=$TRAIN ($nts) ==="

for s in $SEEDS; do
  [ -f "$CKPT_ROOT/s${s}/eval_lag2.json" ] && { echo "  s${s}: done, skip"; continue; }
  train_one "$s"; eval_one "$s"
done

echo "=== [$VARIANT ext] LEADERBOARD ==="
sort -t, -k4 -rn "$LOG" | head -10
