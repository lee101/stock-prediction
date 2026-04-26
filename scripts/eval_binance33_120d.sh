#!/usr/bin/env bash
# Exhaustive 120-day unseen Binance33 evaluation for completed sweep checkpoints.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source .venv313/bin/activate

VAL_DATA="${VAL_DATA:-pufferlib_market/data/binance33_daily_val.bin}"
CKPT_ROOT="${CKPT_ROOT:-pufferlib_market/checkpoints/binance33_1x_smooth}"
LOG="${LOG:-$CKPT_ROOT/leaderboard_120d.csv}"
EVAL_DAYS="${EVAL_DAYS:-120}"
PERIODS_PER_YEAR="${PERIODS_PER_YEAR:-365}"
DECISION_LAG="${DECISION_LAG:-2}"
FEE_RATE="${FEE_RATE:-0.001}"
FILL_BUFFER_BPS="${FILL_BUFFER_BPS:-5}"
SLIPPAGES="${SLIPPAGES:-5 20}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f "$VAL_DATA" ]]; then
  echo "Missing validation MKTD file: $VAL_DATA" >&2
  exit 1
fi
if [[ ! -d "$CKPT_ROOT" ]]; then
  echo "Missing checkpoint root: $CKPT_ROOT" >&2
  exit 1
fi

mkdir -p "$CKPT_ROOT"
echo "timestamp,config,seed,eval_days,slip_bps,median_pct,p10_pct,neg_windows,p90_dd_pct,median_smooth,median_ulcer,median_sortino,checkpoint" > "$LOG"

eval_checkpoint() {
  local name="$1" seed="$2" ckpt="$3" slip="$4" dir="$5"
  [[ -f "$ckpt" ]] || return 0

  local out="$dir/eval_${EVAL_DAYS}d_slip${slip}.json"
  if ! python -m pufferlib_market.evaluate_holdout \
    --checkpoint "$ckpt" \
    --data-path "$VAL_DATA" \
    --eval-hours "$EVAL_DAYS" \
    --exhaustive \
    --fee-rate "$FEE_RATE" \
    --slippage-bps "$slip" \
    --fill-buffer-bps "$FILL_BUFFER_BPS" \
    --max-leverage 1.0 \
    --periods-per-year "$PERIODS_PER_YEAR" \
    --decision-lag "$DECISION_LAG" \
    --deterministic \
    --no-early-stop \
    --device "$DEVICE" \
    --out "$out" > /dev/null; then
    echo "[$(date -u +%FT%TZ)] eval failed for $name seed=$seed days=$EVAL_DAYS slip=$slip ckpt=$ckpt"
    return 0
  fi

  if ! python - "$out" "$LOG" "$name" "$seed" "$EVAL_DAYS" "$slip" "$ckpt" <<'PY'
import csv, json, sys
from datetime import UTC, datetime

payload = json.load(open(sys.argv[1]))
summary = payload["summary"]
row = {
    "timestamp": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    "config": sys.argv[3],
    "seed": sys.argv[4],
    "eval_days": sys.argv[5],
    "slip_bps": sys.argv[6],
    "median_pct": f"{100 * summary.get('median_total_return', 0.0):.2f}",
    "p10_pct": f"{100 * summary.get('p10_total_return', 0.0):.2f}",
    "neg_windows": int(summary.get("negative_windows", 0)),
    "p90_dd_pct": f"{100 * summary.get('p90_max_drawdown', 0.0):.2f}",
    "median_smooth": f"{summary.get('median_pnl_smoothness', 0.0):.6f}",
    "median_ulcer": f"{summary.get('median_ulcer_index', 0.0):.3f}",
    "median_sortino": f"{summary.get('median_sortino', 0.0):.3f}",
    "checkpoint": sys.argv[7],
}
with open(sys.argv[2], "a", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(row))
    writer.writerow(row)
print(f"{row['config']} s{row['seed']} {row['eval_days']}d slip={row['slip_bps']} med={row['median_pct']}% p10={row['p10_pct']}% neg={row['neg_windows']}")
PY
  then
    echo "[$(date -u +%FT%TZ)] eval parse failed for $name seed=$seed days=$EVAL_DAYS slip=$slip out=$out"
    return 0
  fi
}

found=0
shopt -s nullglob
for dir in "$CKPT_ROOT"/*_s*; do
  [[ -d "$dir" ]] || continue
  base="$(basename "$dir")"
  if [[ "$base" =~ ^(.+)_s([0-9]+)$ ]]; then
    name="${BASH_REMATCH[1]}"
    seed="${BASH_REMATCH[2]}"
  else
    name="$base"
    seed=""
  fi

  ckpt="$dir/val_best.pt"
  [[ -f "$ckpt" ]] || ckpt="$dir/best.pt"
  [[ -f "$ckpt" ]] || ckpt="$dir/final.pt"
  if [[ ! -f "$ckpt" ]]; then
    continue
  fi

  found=1
  for slip in $SLIPPAGES; do
    eval_checkpoint "$name" "$seed" "$ckpt" "$slip" "$dir"
  done
done

if [[ "$found" -eq 0 ]]; then
  echo "No completed checkpoints found under $CKPT_ROOT"
  exit 0
fi

echo
echo "=== Binance33 ${EVAL_DAYS}d unseen leaderboard ==="
tail -n +2 "$LOG" | sort -t, -k6 -rn | head -20 || true
