#!/bin/bash
# Algorithmic sweep for stocks17 augmented training — 3 seeds per variant.
#
# Each variant changes ONE thing from the baseline (seeds 1-5, now invalidated
# because the training binary changed). Run 3 seeds per variant, evaluate all
# with decision_lag=2 binary fills. Pick the best algorithm, then sweep seeds.
#
# Variants:
#   A  baseline     tp=0.05  steps=15M  max_steps=252  adamw
#   B  long_train   tp=0.05  steps=30M  max_steps=252  adamw  (train 2x longer)
#   C  low_tp       tp=0.02  steps=15M  max_steps=252  adamw  (less conservative)
#   D  muon         tp=0.05  steps=15M  max_steps=252  muon   (orthog. optimizer)
#   E  short_ep     tp=0.05  steps=15M  max_steps=90   adamw  (quarter episodes)
#
# Usage: pick ONE variant at a time to avoid GPU contention:
#   nohup bash scripts/sweep_stocks17_algos.sh A > /tmp/sw17_A.log 2>&1 &
#   nohup bash scripts/sweep_stocks17_algos.sh B > /tmp/sw17_B.log 2>&1 &
#   etc.

set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

VARIANT="${1:-A}"
TRAIN_DATA="pufferlib_market/data/stocks17_augmented_train.bin"
VAL_DATA="pufferlib_market/data/stocks17_augmented_val.bin"
SEEDS=(21 22 23)   # canonical 3 seeds, different from earlier sweep

case "$VARIANT" in
  A) LABEL="baseline";   TP=0.05; STEPS=15000000; MAX_EP=252; EXTRA_FLAGS=() ;;
  B) LABEL="long_train"; TP=0.05; STEPS=30000000; MAX_EP=252; EXTRA_FLAGS=() ;;
  C) LABEL="low_tp";     TP=0.02; STEPS=15000000; MAX_EP=252; EXTRA_FLAGS=() ;;
  D) LABEL="muon";       TP=0.05; STEPS=15000000; MAX_EP=252; EXTRA_FLAGS=(--optimizer muon) ;;
  E) LABEL="short_ep";   TP=0.05; STEPS=15000000; MAX_EP=90;  EXTRA_FLAGS=() ;;
  *) echo "Unknown variant $VARIANT (use A-E)"; exit 1 ;;
esac

CKPT_ROOT="pufferlib_market/checkpoints/stocks17_sweep/${VARIANT}_${LABEL}"
LOG="$CKPT_ROOT/leaderboard.csv"
mkdir -p "$CKPT_ROOT"

test -f "$TRAIN_DATA" || { echo "ERROR: $TRAIN_DATA missing"; exit 1; }
test -f "$VAL_DATA"   || { echo "ERROR: $VAL_DATA missing";   exit 1; }

echo "timestamp,variant,seed,med_pct,p10_pct,worst_pct,med_sortino,checkpoint" > "$LOG"

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
  echo "  s${seed}: evaluating (lag=2, 50 windows) ..."
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
sort  = d.get('median_sortino', 0)
print(f"{med:.2f},{p10:.2f},{worst:.2f},{sort:.2f}")
PY
  )
  [ -z "$stats" ] && { echo "  s${seed}: parse failed"; return; }
  ts=$(date -u +%FT%TZ)
  echo "$ts,$VARIANT,$seed,$stats,$ckpt" >> "$LOG"
  echo "  s${seed}: med=${stats%%,*}% | full=$stats"
}

echo "=== Variant $VARIANT / $LABEL : tp=$TP steps=$STEPS max_ep=$MAX_EP seeds=${SEEDS[*]} ==="
echo "    train=$TRAIN_DATA ($(python3 -c "import struct; h=open('$TRAIN_DATA','rb').read(64); print(struct.unpack_from('<IIIII',h)[2],'syms',struct.unpack_from('<IIIII',h)[3],'steps')" 2>/dev/null))"
echo

for s in "${SEEDS[@]}"; do train_one "$s"; eval_one "$s"; done

echo
echo "=== [$VARIANT] LEADERBOARD ==="
sort -t, -k4 -rn "$LOG" | head -5
