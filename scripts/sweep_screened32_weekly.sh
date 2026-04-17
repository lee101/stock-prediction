#!/bin/bash
# Screened32 weekly sweep — weekly trading granularity with optional Chronos2 7d forecasts.
#
# Build dataset first:
#   python scripts/build_screened32_weekly.py
#   python scripts/build_screened32_weekly.py --chronos-cache strategytraining/forecast_cache
#
# Then run:
#   nohup bash scripts/sweep_screened32_weekly.sh C > /tmp/s32w_C.log 2>&1 &
#   nohup bash scripts/sweep_screened32_weekly.sh D > /tmp/s32w_D.log 2>&1 &

set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

export TMPDIR="$(pwd)/.tmp_train"
export TRITON_CACHE_DIR="$(pwd)/.tmp_train/triton_cache"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

VARIANT="${1:-C}"
PREFIX="${2:-screened32_weekly}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20}"

TRAIN="pufferlib_market/data/${PREFIX}_train.bin"
VAL="pufferlib_market/data/${PREFIX}_val.bin"
RECENT_VAL="pufferlib_market/data/${PREFIX}_recent_val.bin"

if [ ! -f "$TRAIN" ] || [ ! -f "$VAL" ]; then
  echo "ERROR: Weekly dataset not found. Build first:"
  echo "  python scripts/build_screened32_weekly.py"
  echo "  python scripts/build_screened32_weekly.py --chronos-cache strategytraining/forecast_cache"
  exit 1
fi

nts=$(python3 -c "import struct; h=open('$TRAIN','rb').read(24); v=struct.unpack_from('<IIII',h,4); print(f'{v[1]} syms, {v[2]} ts, {v[3]} feats')" 2>/dev/null || echo "?")
nv=$(python3 -c "import struct; h=open('$VAL','rb').read(24); v=struct.unpack_from('<IIII',h,4); print(f'{v[2]} ts')" 2>/dev/null || echo "?")
echo "Train: $TRAIN ($nts)   Val: $VAL ($nv)"

case "$VARIANT" in
  C) TP=0.02; EXTRA_FLAGS=()                ;;
  D) TP=0.05; EXTRA_FLAGS=(--optimizer muon) ;;
  *) echo "Unknown variant: $VARIANT"; exit 1 ;;
esac

CKPT_ROOT="pufferlib_market/checkpoints/screened32_weekly_sweep/${VARIANT}"
LOG="${CKPT_ROOT}/leaderboard.csv"
mkdir -p "$CKPT_ROOT"
[ -f "$LOG" ] || echo "timestamp,variant,seed,med_pct,p10_pct,worst_pct,neg_count,med_sortino,checkpoint" > "$LOG"

train_one() {
  local seed=$1; local dir="${CKPT_ROOT}/s${seed}"; mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] [${VARIANT}_weekly] seed ${seed}"
  python -u -m pufferlib_market.train \
      --data-path "$TRAIN" --val-data-path "$VAL" \
      --total-timesteps 8000000 --max-steps 12 \
      --trade-penalty "$TP" --hidden-size 1024 \
      --anneal-lr --disable-shorts --num-envs 128 \
      --val-eval-windows 20 --periods-per-year 52 \
      --early-stop-val-neg-threshold "${EARLY_STOP_NEG:-10}" \
      --early-stop-val-neg-patience "${EARLY_STOP_PATIENCE:-2}" \
      "${EXTRA_FLAGS[@]}" --seed "$seed" --checkpoint-dir "$dir" > "$dir/train.log" 2>&1
  echo "[$(date -u +%FT%TZ)] [${VARIANT}_weekly] seed ${seed} done"
}

eval_one() {
  local seed=$1
  local ckpt="${CKPT_ROOT}/s${seed}/val_best.pt"
  [ -f "$ckpt" ] || ckpt="${CKPT_ROOT}/s${seed}/best.pt"
  [ -f "$ckpt" ] || { echo "  s${seed}: no ckpt"; return; }
  local out="${CKPT_ROOT}/s${seed}/eval_lag2.json"
  python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" --data-path "$VAL" \
      --eval-hours 12 --n-windows 20 \
      --fee-rate 0.001 --fill-buffer-bps 5.0 \
      --decision-lag 2 --deterministic --no-early-stop \
      --periods-per-year 52 \
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

  # Also eval on recent bear market val
  if [ -f "$RECENT_VAL" ]; then
    local out_rv="${CKPT_ROOT}/s${seed}/eval_recent_lag2.json"
    python -m pufferlib_market.evaluate_holdout \
        --checkpoint "$ckpt" --data-path "$RECENT_VAL" \
        --eval-hours 12 --n-windows 20 \
        --fee-rate 0.001 --fill-buffer-bps 5.0 \
        --decision-lag 2 --deterministic --no-early-stop \
        --periods-per-year 52 \
        > "$out_rv" 2>/dev/null
    stats_rv=$(python3 - "$out_rv" 2>/dev/null <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  RECENT: med={d.get('median_total_return',0)*100:.2f}% p10={d.get('p10_total_return',0)*100:.2f}% neg={d.get('negative_windows',0)}")
PY
    )
    [ -n "$stats_rv" ] && echo "$stats_rv"
  fi
}

echo "=== screened32 WEEKLY sweep: variant=${VARIANT}, tp=${TP} ==="
for s in $SEEDS; do
  [ -f "${CKPT_ROOT}/s${s}/eval_lag2.json" ] && { echo "  s${s}: done, skip"; continue; }
  train_one "$s"; eval_one "$s"
done

echo "=== [${VARIANT} weekly] LEADERBOARD ==="
sort -t, -k4 -rn "$LOG" | head -10
