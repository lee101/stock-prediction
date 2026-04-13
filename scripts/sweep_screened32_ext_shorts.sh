#!/bin/bash
# Short-enabled screened32 sweep: train through Nov 2025, val = Dec 2025 - Apr 2026
# Allows short selling so the model can profit from bear markets.
# Key difference from sweep_screened32_ext.sh: --disable-shorts is REMOVED.
# The model has access to 65 actions: 1 flat + 32 longs + 32 shorts.
#
# With short selling, the model may learn to:
#   - Go LONG on stocks in bull market conditions
#   - Go SHORT on stocks in bear market conditions
#   - Hold cash (flat) when uncertain
#
# Usage:
#   nohup bash scripts/sweep_screened32_ext_shorts.sh C > /tmp/s32ext_shorts_C.log 2>&1 &
#   nohup bash scripts/sweep_screened32_ext_shorts.sh D > /tmp/s32ext_shorts_D.log 2>&1 &

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

SEEDS=${SEEDS:-1 2 3 4 5 6 7 8 9 10}
CKPT_ROOT="pufferlib_market/checkpoints/screened32_ext_shorts_sweep/${VARIANT}"
LOG="$CKPT_ROOT/leaderboard.csv"
mkdir -p "$CKPT_ROOT"
echo "timestamp,variant,seed,med_pct,p10_pct,worst_pct,neg_count,med_sortino,checkpoint" > "$LOG"

train_one() {
  local seed=$1
  local dir="$CKPT_ROOT/s${seed}"
  mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] [${VARIANT}_ext_shorts] seed ${seed}"
  python -u -m pufferlib_market.train \
      --data-path "$TRAIN" --val-data-path "$VAL" \
      --total-timesteps "$STEPS" --max-steps 252 \
      --trade-penalty "$TP" --hidden-size 1024 \
      --anneal-lr --val-eval-windows 30 \
      "${EXTRA_FLAGS[@]}" --num-envs 128 \
      --seed "$seed" --checkpoint-dir "$dir" > "$dir/train.log" 2>&1
  echo "[$(date -u +%FT%TZ)] [${VARIANT}_ext_shorts] seed ${seed} done"
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

echo "=== screened32_ext SHORTS sweep variant=$VARIANT seeds=$SEEDS ==="
echo "  (shorts enabled — model can go long OR short)"
for seed in $SEEDS; do
  if [ -f "$CKPT_ROOT/s${seed}/val_best.pt" ]; then
    echo "  s${seed}: already done"
    continue
  fi
  train_one "$seed" &
  PID=$!
  wait "$PID"
  eval_one "$seed"
  # Brief pause to let GPU cool and avoid lock contention
  sleep 5
done

echo "=== sweep complete ==="
cat "$LOG"
