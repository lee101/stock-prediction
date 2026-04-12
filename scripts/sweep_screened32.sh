#!/bin/bash
# Seed sweep for screened32 augmented training.
# 32 curated stocks × 35x augmentation (5 shifts × 7 vol scales)
# 20,377 train timesteps vs 2,911 for stocks17.
#
# Variants:
#   C: tp=0.02 adamw (most stable, stocks17 champion)
#   D: tp=0.05 muon  (best median in stocks17 D_muon)
#
# Usage:
#   nohup bash scripts/sweep_screened32.sh C > /tmp/s32_sweep_C.log 2>&1 &
#   nohup bash scripts/sweep_screened32.sh D > /tmp/s32_sweep_D.log 2>&1 &

set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

export TMPDIR="$(pwd)/.tmp_train"
mkdir -p "$TMPDIR"

VARIANT="${1:-C}"
TRAIN="pufferlib_market/data/screened32_augmented_train.bin"
VAL="pufferlib_market/data/screened32_augmented_val.bin"

case "$VARIANT" in
  C) TP=0.02; STEPS=15000000; EXTRA_FLAGS=() ;;
  D) TP=0.05; STEPS=15000000; EXTRA_FLAGS=(--optimizer muon) ;;
  *) echo "Unknown variant $VARIANT (use C or D)"; exit 1 ;;
esac

SEEDS=${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20}

CKPT_ROOT="pufferlib_market/checkpoints/screened32_sweep/${VARIANT}"
LOG="$CKPT_ROOT/leaderboard.csv"
mkdir -p "$CKPT_ROOT"
echo "timestamp,variant,seed,med_pct,p10_pct,worst_pct,neg_count,med_sortino,checkpoint" > "$LOG"

train_one() {
  local seed=$1
  local dir="$CKPT_ROOT/s${seed}"
  mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] [${VARIANT}] train seed ${seed}"
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
      --checkpoint-dir "$dir" \
      > "$dir/train.log" 2>&1
  echo "[$(date -u +%FT%TZ)] [${VARIANT}] done seed ${seed} (exit=$?)"
}

eval_one() {
  local seed=$1
  local ckpt="$CKPT_ROOT/s${seed}/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$CKPT_ROOT/s${seed}/best.pt"
  [ -f "$ckpt" ] || { echo "  s${seed}: no checkpoint"; return; }
  local out="$CKPT_ROOT/s${seed}/eval_lag2.json"
  echo "  s${seed}: evaluating lag=2, 50 windows..."
  python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" --data-path "$VAL" \
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
  echo "  s${seed}: med=${stats%%,*}%  full=$stats"
}

nts=$(python3 -c "import struct; h=open('$TRAIN','rb').read(24); v=struct.unpack_from('<IIII',h,4); print(f'{v[1]} syms, {v[2]} ts, {v[3]} feats')" 2>/dev/null || echo "?")
echo "=== screened32 variant $VARIANT : tp=$TP steps=$STEPS ==="
echo "    train=$TRAIN ($nts)"
echo

for s in $SEEDS; do
  if [ -f "$CKPT_ROOT/s${s}/eval_lag2.json" ]; then
    echo "[$(date -u +%FT%TZ)] s${s}: already done, skipping"
    continue
  fi
  train_one "$s"
  eval_one "$s"
done

echo
echo "=== [${VARIANT}] LEADERBOARD ==="
sort -t, -k4 -rn "$LOG" | head -10
echo
echo "=== Positive-p10 (p10>0, neg<5) ==="
awk -F, 'NR>1 && $5+0>0 && $7+0<5 {print}' "$LOG" | sort -t, -k4 -rn
