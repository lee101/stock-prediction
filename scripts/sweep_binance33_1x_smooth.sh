#!/usr/bin/env bash
# Fast-fail Binance33 1x PPO sweep with realistic lag-2 binary-fill validation.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source .venv313/bin/activate

TRAIN_DATA="pufferlib_market/data/binance33_daily_aug_train.bin"
VAL_DATA="pufferlib_market/data/binance33_daily_val.bin"
CKPT_ROOT="pufferlib_market/checkpoints/binance33_1x_smooth"
LOG="$CKPT_ROOT/leaderboard.csv"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
MAX_STEPS=365
PERIODS_PER_YEAR=365
DECISION_LAG=2
FEE_RATE=0.001
SLIPPAGE_BPS=5
FILL_BUFFER_BPS=5

if [[ ! -f "$TRAIN_DATA" || ! -f "$VAL_DATA" ]]; then
  echo "Missing Binance33 MKTD files; run:"
  echo "  python scripts/export_binance33_daily.py"
  echo "  python scripts/build_crypto30_augmented.py --input pufferlib_market/data/binance33_daily_train.bin --output $TRAIN_DATA"
  exit 1
fi

mkdir -p "$CKPT_ROOT"
echo "timestamp,config,seed,slip_bps,median_pct,p10_pct,neg_windows,p90_dd_pct,median_smooth,median_ulcer,median_sortino,checkpoint" > "$LOG"

eval_checkpoint() {
  local name="$1" seed="$2" ckpt="$3" slip="$4" dir="$5"
  [[ -f "$ckpt" ]] || return 0

  local out="$dir/eval_30d_slip${slip}.json"
  if ! python -m pufferlib_market.evaluate_holdout \
    --checkpoint "$ckpt" \
    --data-path "$VAL_DATA" \
    --eval-hours 30 \
    --exhaustive \
    --fee-rate "$FEE_RATE" \
    --slippage-bps "$slip" \
    --fill-buffer-bps "$FILL_BUFFER_BPS" \
    --max-leverage 1.0 \
    --periods-per-year "$PERIODS_PER_YEAR" \
    --decision-lag "$DECISION_LAG" \
    --deterministic \
    --no-early-stop \
    --device cuda \
    --out "$out" > /dev/null; then
    echo "[$(date -u +%FT%TZ)] eval failed for $name seed=$seed slip=$slip ckpt=$ckpt"
    return 0
  fi

  if ! python - "$out" "$LOG" "$name" "$seed" "$slip" "$ckpt" <<'PY'
import csv, json, sys
from datetime import UTC, datetime

payload = json.load(open(sys.argv[1]))
summary = payload["summary"]
row = {
    "timestamp": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    "config": sys.argv[3],
    "seed": sys.argv[4],
    "slip_bps": sys.argv[5],
    "median_pct": f"{100 * summary.get('median_total_return', 0.0):.2f}",
    "p10_pct": f"{100 * summary.get('p10_total_return', 0.0):.2f}",
    "neg_windows": int(summary.get("negative_windows", 0)),
    "p90_dd_pct": f"{100 * summary.get('p90_max_drawdown', 0.0):.2f}",
    "median_smooth": f"{summary.get('median_pnl_smoothness', 0.0):.6f}",
    "median_ulcer": f"{summary.get('median_ulcer_index', 0.0):.3f}",
    "median_sortino": f"{summary.get('median_sortino', 0.0):.3f}",
    "checkpoint": sys.argv[6],
}
with open(sys.argv[2], "a", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(row))
    writer.writerow(row)
print(f"{row['config']} s{row['seed']} slip={row['slip_bps']} med={row['median_pct']}% p10={row['p10_pct']}% neg={row['neg_windows']}")
PY
  then
    echo "[$(date -u +%FT%TZ)] eval parse failed for $name seed=$seed slip=$slip out=$out"
    return 0
  fi
}

run_config() {
  local name="$1" seed="$2" hidden="$3" arch="$4" tp="$5" ent="$6" wd="$7" cash="$8" extra_flags="$9"
  local dir="$CKPT_ROOT/${name}_s${seed}"
  rm -rf -- "$dir"
  mkdir -p "$dir"

  echo "[$(date -u +%FT%TZ)] train $name seed=$seed hidden=$hidden arch=$arch tp=$tp ent=$ent wd=$wd cash=$cash"
  # shellcheck disable=SC2086
  python -u -m pufferlib_market.train \
    --data-path "$TRAIN_DATA" \
    --val-data-path "$VAL_DATA" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --max-steps "$MAX_STEPS" \
    --hidden-size "$hidden" \
    --arch "$arch" \
    --num-envs 32 \
    --rollout-len 128 \
    --minibatch-size 1024 \
    --ppo-epochs 3 \
    --anneal-lr \
    --max-leverage 1.0 \
    --periods-per-year "$PERIODS_PER_YEAR" \
    --decision-lag "$DECISION_LAG" \
    --val-decision-lag "$DECISION_LAG" \
    --fee-rate "$FEE_RATE" \
    --fill-slippage-bps "$SLIPPAGE_BPS" \
    --cash-penalty "$cash" \
    --trade-penalty "$tp" \
    --downside-penalty 0.2 \
    --smooth-downside-penalty 0.1 \
    --smoothness-penalty 0.05 \
    --reward-scale 10 \
    --reward-clip 5 \
    --ent-coef "$ent" \
    --weight-decay "$wd" \
    --seed "$seed" \
    --val-eval-interval 5 \
    --val-eval-windows 5 \
    --early-stop-val-neg-threshold 4 \
    --early-stop-val-neg-patience 2 \
    --save-every 999999 \
    --max-periodic-checkpoints 1 \
    --checkpoint-dir "$dir" \
    $extra_flags > "$dir/train.log" 2>&1 || true

  local ckpt="$dir/val_best.pt"
  [[ -f "$ckpt" ]] || ckpt="$dir/best.pt"
  [[ -f "$ckpt" ]] || ckpt="$dir/final.pt"
  if [[ ! -f "$ckpt" ]]; then
    echo "[$(date -u +%FT%TZ)] no checkpoint for $name seed=$seed"
    return 0
  fi
  eval_checkpoint "$name" "$seed" "$ckpt" 5 "$dir"
  eval_checkpoint "$name" "$seed" "$ckpt" 20 "$dir"
}

run_config "long_h512_tp05_wd01"     202 512  mlp        0.05 0.05 0.01  0.0 "--disable-shorts"
run_config "long_h256_tp02_wd005"    203 256  mlp        0.02 0.03 0.005 0.0 "--disable-shorts"
run_config "long_relu_sq_tp03"       204 512  mlp_relu_sq 0.03 0.05 0.01  0.0 "--disable-shorts"
run_config "long_resmlp_tp03"        205 512  resmlp     0.03 0.05 0.01  0.0 "--disable-shorts"
run_config "levels_h256_tp005"       206 256  mlp        0.005 0.02 0.005 0.0 "--disable-shorts --action-allocation-bins 2 --action-level-bins 3 --action-max-offset-bps 25"
run_config "margin_short_h512_tp05"  207 512  mlp        0.05 0.05 0.01  0.0 ""

echo
echo "=== Binance33 1x leaderboard ==="
tail -n +2 "$LOG" | sort -t, -k5 -rn | head -20 || true
