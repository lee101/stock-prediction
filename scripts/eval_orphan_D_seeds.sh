#!/bin/bash
# Evaluate orphan D seeds (s161-165, s201) when their training completes
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate
FULL_VAL="pufferlib_market/data/screened32_single_offset_val_full.bin"
CKPT_ROOT="pufferlib_market/checkpoints/screened32_sweep/D"
LOG="$CKPT_ROOT/orphan_eval.log"

ORPHAN_SEEDS="${SEEDS:-161 162 163 164 165 201}"
echo "[$(date -u +%FT%TZ)] Orphan eval watcher: seeds=$ORPHAN_SEEDS" | tee "$LOG"

eval_seed() {
  local s=$1
  local dir="$CKPT_ROOT/s${s}"

  echo "[$(date -u +%FT%TZ)] Evaluating D/s$s" | tee -a "$LOG"
  grep "\[val\]" "$dir/train.log" 2>/dev/null | tee -a "$LOG"

  local best_neg=999 best_ckpt=""
  for candidate in "$dir/val_best.pt" "$dir/best_neg.pt" "$dir/best.pt"; do
    [ -f "$candidate" ] || continue
    local tmp_out="${candidate%.pt}_oos_eval.json"
    if [ ! -s "$tmp_out" ]; then
      python -m pufferlib_market.evaluate_holdout \
        --checkpoint "$candidate" --data-path "$FULL_VAL" \
        --eval-hours 50 --n-windows 100 --fee-rate 0.001 \
        --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop \
        > "$tmp_out" 2>/dev/null
    fi
    if [ -s "$tmp_out" ]; then
      local n
      n=$(python3 -c "import json; d=json.load(open('$tmp_out')); print(d.get('negative_windows',99))" 2>/dev/null)
      local m
      m=$(python3 -c "import json; d=json.load(open('$tmp_out')); print(f'{d.get(\"median_total_return\",0)*100:.2f}')" 2>/dev/null)
      echo "  D/s$s $(basename $candidate): neg=$n med=$m%" | tee -a "$LOG"
      if [ -n "$n" ] && [ "$n" -lt "$best_neg" ]; then
        best_neg=$n; best_ckpt=$candidate
      fi
    fi
  done

  if [ -n "$best_ckpt" ]; then
    cp "${best_ckpt%.pt}_oos_eval.json" "$dir/eval_full.json" 2>/dev/null
    ts=$(date -u +%FT%TZ)
    python3 -c "
import json
d = json.load(open('$dir/eval_full.json'))
med = d.get('median_total_return',0)*100
p10 = d.get('p10_total_return',0)*100
worst = (d.get('worst_window') or {}).get('total_return',0)*100
neg = d.get('negative_windows',0)
sort = d.get('median_sortino',0)
print(f'$ts,D,$s,{med:.2f},{p10:.2f},{worst:.2f},{neg},{sort:.2f},$best_ckpt')
" 2>/dev/null >> "$CKPT_ROOT/leaderboard_fulloos.csv"
    echo "[$(date -u +%FT%TZ)] D/s$s BEST: neg=$best_neg ($(basename $best_ckpt))" | tee -a "$LOG"
  fi
}

for s in $ORPHAN_SEEDS; do
  dir="$CKPT_ROOT/s${s}"
  [ -d "$dir" ] || { echo "s$s: no dir, skip"; continue; }
  [ -s "$dir/eval_full.json" ] && { echo "s$s: already done, skip"; continue; }

  local_wait=0
  while [ ! -f "$dir/final.pt" ] && [ $local_wait -lt 10800 ]; do
    sleep 60
    local_wait=$((local_wait + 60))
  done

  if [ -f "$dir/final.pt" ]; then
    eval_seed "$s"
  else
    echo "[$(date -u +%FT%TZ)] s$s: training timed out" | tee -a "$LOG"
  fi
done

echo "[$(date -u +%FT%TZ)] Orphan eval complete" | tee -a "$LOG"
