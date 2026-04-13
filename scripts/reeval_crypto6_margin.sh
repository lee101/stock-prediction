#!/bin/bash
# Re-evaluate all crypto6 margin checkpoints on the long val set.
# Uses 720h windows, 50 windows, slippage=5bps.
cd "$(git rev-parse --show-toplevel)"
source .venv313/bin/activate
export CUDA_VISIBLE_DEVICES=""

VAL_DATA="pufferlib_market/data/crypto6_margin_val_long.bin"
CKPT_ROOT="pufferlib_market/checkpoints/crypto6_margin_p1"
LOG="$CKPT_ROOT/leaderboard_long.csv"
FEE_RATE=0.001
DECISION_LAG=2

test -f "$VAL_DATA" || { echo "ERROR: $VAL_DATA missing"; exit 1; }

echo "timestamp,config,seed,med_pct,p10_pct,worst_pct,med_sortino,neg_windows,checkpoint" > "$LOG"

for dir in "$CKPT_ROOT"/*/; do
  [ -d "$dir" ] || continue
  name=$(basename "$dir")
  ckpt="$dir/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$dir/best.pt"
  [ -f "$ckpt" ] || continue

  seed=$(echo "$name" | grep -oP 's\K\d+$')
  config=$(echo "$name" | sed "s/_s${seed}$//")

  out="$dir/eval_longval.json"
  echo "[$(date -u +%FT%TZ)] Evaluating $name..."
  python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" --data-path "$VAL_DATA" \
      --eval-hours 720 --n-windows 50 --fee-rate "$FEE_RATE" \
      --fill-buffer-bps 5.0 --slippage-bps 5.0 \
      --decision-lag "$DECISION_LAG" --deterministic --no-early-stop \
      --max-leverage 3.0 --short-borrow-apr 0.10 \
      > "$out" 2>/dev/null || { echo "  $name: eval failed"; continue; }
  stats=$(python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
med   = d.get("median_total_return", 0) * 100
p10   = d.get("p10_total_return", 0) * 100
worst = d.get("worst_window", {}).get("total_return", 0) * 100
sort  = d.get("median_sortino", 0)
neg   = d.get("negative_windows", 0)
print(f"{med:.2f},{p10:.2f},{worst:.2f},{sort:.2f},{neg}")
PY
  )
  [ -z "$stats" ] && continue
  echo "$(date -u +%FT%TZ),$config,$seed,$stats,$ckpt" >> "$LOG"
  echo "  $name: med=${stats%%,*}% | $stats"
done

echo
echo "=== LEADERBOARD (long val, by median) ==="
sort -t, -k4 -rn "$LOG" | head -25
echo
echo "=== BEST PER CONFIG ==="
tail -n +2 "$LOG" | sort -t, -k2,2 -k4 -rn | awk -F, '!seen[$2]++ {print $2": med="$4"% p10="$5"% sort="$7" neg="$8"/50"}'
