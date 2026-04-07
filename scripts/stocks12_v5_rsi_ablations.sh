#!/bin/bash
# Two single-variable ablations on top of the failed v5_rsi crypto-recipe port
# (see alpacaprod.md 2026-04-07 entry).
#
# Ablation A — tp02_full_recipe: same crypto recipe (h1024 + obs_norm + bf16 +
# cuda-graph + linear anneal) but trade-penalty dropped 0.05 → 0.02. Tests the
# hypothesis that the recipe collapses because cleaner gradients let the trade
# penalty dominate too fast on the small daily dataset.
#
# Ablation B — tp05_obsnorm_only: baseline tp=0.05 + h1024 + linear anneal,
# add ONLY --obs-norm (no bf16, no cuda-graph). Isolates whether obs_norm by
# itself causes the collapse.
#
# Sequential within each ablation, parallel between A and B is left to the
# caller (just run two of these in different shells).
#
# Usage: nohup bash scripts/stocks12_v5_rsi_ablations.sh A > .ablate_A.log 2>&1 &
#        nohup bash scripts/stocks12_v5_rsi_ablations.sh B > .ablate_B.log 2>&1 &

cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

ABLATION="${1:-A}"
TRAIN_DATA="pufferlib_market/data/stocks12_daily_v5_rsi_train.bin"
VAL_DATA="pufferlib_market/data/stocks12_daily_v5_rsi_val.bin"

case "$ABLATION" in
  A)
    CKPT_ROOT="pufferlib_market/checkpoints/stocks12_v5_rsi_ablate_tp02"
    LOG="pufferlib_market/stocks12_v5_rsi_ablate_tp02_leaderboard.csv"
    SEEDS=(200 201 202)
    TRADE_PENALTY=0.02
    EXTRA_FLAGS=(--obs-norm --use-bf16 --cuda-graph-ppo)
    ;;
  B)
    CKPT_ROOT="pufferlib_market/checkpoints/stocks12_v5_rsi_ablate_obsonly"
    LOG="pufferlib_market/stocks12_v5_rsi_ablate_obsonly_leaderboard.csv"
    SEEDS=(300 301 302)
    TRADE_PENALTY=0.05
    EXTRA_FLAGS=(--obs-norm)
    ;;
  *) echo "unknown ablation: $ABLATION (use A or B)"; exit 1 ;;
esac

mkdir -p "$CKPT_ROOT"
echo "timestamp,seed,med_90d,p10_90d,worst_90d,best_90d,med_sortino,trades_w,trades_b" > "$LOG"

train_one() {
  local seed=$1
  local dir="$CKPT_ROOT/tp_s${seed}"
  mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] [$ABLATION] training seed $seed -> $dir"
  python -u -m pufferlib_market.train \
      --data-path     "$TRAIN_DATA" \
      --val-data-path "$VAL_DATA" \
      --total-timesteps 15000000 \
      --max-steps 252 \
      --trade-penalty "$TRADE_PENALTY" \
      --hidden-size 1024 \
      --anneal-lr \
      "${EXTRA_FLAGS[@]}" \
      --num-envs 128 \
      --seed "$seed" \
      --checkpoint-dir "$dir" \
      > "$dir/train.log" 2>&1
  echo "[$(date -u +%FT%TZ)] [$ABLATION] done seed $seed"
}

eval_one() {
  local seed=$1
  local ckpt="$CKPT_ROOT/tp_s${seed}/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$CKPT_ROOT/tp_s${seed}/best.pt"
  [ -f "$ckpt" ] || { echo "  s${seed}: no ckpt"; return; }
  local out="$CKPT_ROOT/tp_s${seed}/eval_holdout50.json"
  python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" \
      --data-path  "$VAL_DATA" \
      --eval-hours 90 --n-windows 50 \
      --fee-rate 0.001 --fill-buffer-bps 5.0 \
      --deterministic --no-early-stop \
      > "$out" 2>/dev/null
  stats=$(python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
med   = d.get('median_total_return', 0.0)*100
p10   = d.get('p10_total_return', 0.0)*100
worst = d.get('worst_window', {}).get('total_return', 0.0)*100
best  = d.get('best_window',  {}).get('total_return', 0.0)*100
sort  = d.get('median_sortino', 0.0)
tw    = d.get('worst_window', {}).get('num_trades', 0)
tb    = d.get('best_window',  {}).get('num_trades', 0)
print(f"{med:.2f},{p10:.2f},{worst:.2f},{best:.2f},{sort:.2f},{tw},{tb}")
PY
)
  [ -z "$stats" ] && { echo "  s${seed}: parse failed"; return; }
  ts=$(date -u +%FT%TZ)
  echo "$ts,$seed,$stats" >> "$LOG"
  echo "  [$ABLATION] s${seed}: $stats"
}

echo "=== Ablation $ABLATION: tp=$TRADE_PENALTY flags=${EXTRA_FLAGS[*]} seeds=${SEEDS[*]} ==="
for s in "${SEEDS[@]}"; do train_one "$s"; eval_one "$s"; done

echo
echo "=== Ablation $ABLATION leaderboard ==="
sort -t, -k3 -rn "$LOG" | head
