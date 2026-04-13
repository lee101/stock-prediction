#!/bin/bash
# Screened32 hourly sweep — hourly trading granularity vs daily.
#
# periods_per_year=1638 (252 days × 6.5h), max_steps=720 (30d × 24h).
# Auto-builds hourly MKTD binary from trainingdatahourly/stocks/ if needed.
#
# Usage:
#   nohup bash scripts/sweep_screened32_hourly.sh C > /tmp/s32h_C.log 2>&1 &
#   nohup bash scripts/sweep_screened32_hourly.sh D > /tmp/s32h_D.log 2>&1 &

set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

export TMPDIR="$(pwd)/.tmp_train"
mkdir -p "$TMPDIR"

VARIANT="${1:-C}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20}"

TRAIN="pufferlib_market/data/screened32_hourly_train.bin"
VAL="pufferlib_market/data/screened32_hourly_val.bin"
HOURLY_ROOT="trainingdatahourly/stocks"

SYMBOLS="LLY,BSX,ABBV,VRTX,SYK,WELL,JPM,GS,V,MA,AXP,MS,AAPL,MSFT,NVDA,KLAC,CRWD,META,COST,AZO,TJX,CAT,PH,RTX,BKNG,MAR,HLT,PLTR,SPY,QQQ,AMZN,GOOG"

# Build hourly datasets if needed (separate train and val invocations)
if [ ! -f "$TRAIN" ]; then
  echo "[$(date -u +%FT%TZ)] Building screened32 hourly train dataset..."
  python -m pufferlib_market.export_data_hourly_price \
      --symbols "$SYMBOLS" --data-root "$HOURLY_ROOT" \
      --output "$TRAIN" --start-date "2019-01-01" --end-date "2025-05-31" --min-hours 5000
fi
if [ ! -f "$VAL" ]; then
  echo "[$(date -u +%FT%TZ)] Building screened32 hourly val dataset..."
  python -m pufferlib_market.export_data_hourly_price \
      --symbols "$SYMBOLS" --data-root "$HOURLY_ROOT" \
      --output "$VAL" --start-date "2025-06-01" --end-date "2025-11-30" --min-hours 500
fi

nts=$(python3 -c "import struct; h=open('$TRAIN','rb').read(24); v=struct.unpack_from('<IIII',h,4); print(f'{v[1]} syms, {v[2]} ts, {v[3]} feats')" 2>/dev/null || echo "?")
echo "Train: $TRAIN ($nts)"

case "$VARIANT" in
  C) TP=0.02; EXTRA_FLAGS=()                ;;
  D) TP=0.05; EXTRA_FLAGS=(--optimizer muon) ;;
  *) echo "Unknown variant: $VARIANT"; exit 1 ;;
esac

CKPT_ROOT="pufferlib_market/checkpoints/screened32_hourly_sweep/${VARIANT}"
LOG="${CKPT_ROOT}/leaderboard.csv"
mkdir -p "$CKPT_ROOT"
[ -f "$LOG" ] || echo "timestamp,variant,seed,med_pct,p10_pct,worst_pct,neg_count,med_sortino,checkpoint" > "$LOG"

train_one() {
  local seed=$1; local dir="${CKPT_ROOT}/s${seed}"; mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] [${VARIANT}_hourly] seed ${seed}"
  python -u -m pufferlib_market.train \
      --data-path "$TRAIN" --val-data-path "$VAL" \
      --total-timesteps 15000000 --max-steps 720 \
      --trade-penalty "$TP" --hidden-size 1024 \
      --anneal-lr --disable-shorts --num-envs 128 \
      --val-eval-windows 30 --periods-per-year 1638 \
      "${EXTRA_FLAGS[@]}" --seed "$seed" --checkpoint-dir "$dir" > "$dir/train.log" 2>&1
  echo "[$(date -u +%FT%TZ)] [${VARIANT}_hourly] seed ${seed} done"
}

eval_one() {
  local seed=$1
  local ckpt="${CKPT_ROOT}/s${seed}/val_best.pt"
  [ -f "$ckpt" ] || ckpt="${CKPT_ROOT}/s${seed}/best.pt"
  [ -f "$ckpt" ] || { echo "  s${seed}: no ckpt"; return; }
  local out="${CKPT_ROOT}/s${seed}/eval_lag2.json"
  python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" --data-path "$VAL" \
      --eval-hours 720 --n-windows 30 \
      --fee-rate 0.001 --fill-buffer-bps 5.0 \
      --decision-lag 2 --deterministic --no-early-stop \
      --periods-per-year 1638 \
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

echo "=== screened32 HOURLY sweep: variant=${VARIANT}, tp=${TP} ==="
for s in $SEEDS; do
  [ -f "${CKPT_ROOT}/s${s}/eval_lag2.json" ] && { echo "  s${s}: done, skip"; continue; }
  train_one "$s"; eval_one "$s"
done

echo "=== [${VARIANT} hourly] LEADERBOARD ==="
sort -t, -k4 -rn "$LOG" | head -10
