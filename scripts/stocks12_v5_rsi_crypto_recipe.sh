#!/bin/bash
# Port crypto champion recipe (h1024 + cosine annealLR + obs-norm + BF16 + CUDA-graph)
# to stocks12 v5_rsi daily training. Sequential seeds to share GPU cleanly.
#
# Origin: crypto34_ppo_v8_h1024_300M_annealLR recipe (pufferlib_market/autoresearch_rl.py
# + a100_scaleup robust_champion eval commands) adapted for daily stocks12 v5_rsi data.
#
# Differences vs stocks12_autosweep.sh (previous recipe):
#   + --lr-schedule cosine      (vs legacy linear --anneal-lr)
#   + --obs-norm                (RunningObsNorm, was absent)
#   + --use-bf16                (BF16 autocast PPO forward)
#   + --cuda-graph-ppo          (CUDA graph PPO forward+backward)
#   + --val-data-path           (track val_best.pt during training)
#   + 30M total-timesteps       (2x previous 15M)
#
# Usage: nohup bash scripts/stocks12_v5_rsi_crypto_recipe.sh > /tmp/stocks12_v5_rsi_cryptorcp.log 2>&1 &

# NOTE: deliberately no `set -e` — eval failures must not abort the seed loop.
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

TRAIN_DATA="pufferlib_market/data/stocks12_daily_v5_rsi_train.bin"
VAL_DATA="pufferlib_market/data/stocks12_daily_v5_rsi_val.bin"
CKPT_ROOT="pufferlib_market/checkpoints/stocks12_v5_rsi_cryptorcp"
LOG="pufferlib_market/stocks12_v5_rsi_cryptorcp_leaderboard.csv"

TOTAL_TIMESTEPS=15000000  # match baseline stocks12_autosweep.sh; 30M caused trade-collapse
HIDDEN_SIZE=1024
NUM_ENVS=128
MAX_STEPS=252            # 1y daily episode length
TRADE_PENALTY=0.05

# Sanity
test -f "$TRAIN_DATA" || { echo "missing train data: $TRAIN_DATA"; exit 1; }
test -f "$VAL_DATA"   || { echo "missing val   data: $VAL_DATA";   exit 1; }
mkdir -p "$CKPT_ROOT"
echo "timestamp,seed,med_90d,p10_90d,worst_90d,neg_of_50,med_sortino,checkpoint" > "$LOG"

train_one() {
  local seed=$1
  local dir="$CKPT_ROOT/tp05_s${seed}"
  mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] training seed $seed -> $dir"
  python -u -m pufferlib_market.train \
      --data-path      "$TRAIN_DATA" \
      --val-data-path  "$VAL_DATA" \
      --total-timesteps "$TOTAL_TIMESTEPS" \
      --max-steps       "$MAX_STEPS" \
      --trade-penalty   "$TRADE_PENALTY" \
      --hidden-size     "$HIDDEN_SIZE" \
      --anneal-lr \
      --obs-norm \
      --use-bf16 \
      --cuda-graph-ppo \
      --num-envs        "$NUM_ENVS" \
      --seed            "$seed" \
      --checkpoint-dir  "$dir" \
      > "$dir/train.log" 2>&1
  echo "[$(date -u +%FT%TZ)] done seed $seed"
}

eval_one() {
  local seed=$1
  local ckpt="$CKPT_ROOT/tp05_s${seed}/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$CKPT_ROOT/tp05_s${seed}/best.pt"
  [ -f "$ckpt" ] || { echo "  s${seed}: no ckpt"; return; }
  local out="$CKPT_ROOT/tp05_s${seed}/eval_holdout50.json"
  # evaluate_holdout dumps JSON to stdout (--out is broken in current build);
  # capture via redirect.  Schema is FLAT: top-level median_total_return,
  # p10_total_return, median_sortino, worst_window.total_return etc.
  python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" \
      --data-path  "$VAL_DATA" \
      --eval-hours 90 \
      --n-windows  50 \
      --fee-rate 0.001 \
      --fill-buffer-bps 5.0 \
      --deterministic \
      --no-early-stop > "$out" 2>/dev/null || { echo "  s${seed}: eval failed"; return; }
  stats=$(python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
med  = d.get('median_total_return', 0.0) * 100.0
p10  = d.get('p10_total_return',    0.0) * 100.0
worst = d.get('worst_window', {}).get('total_return', 0.0) * 100.0
sort = d.get('median_sortino', 0.0)
# This eval schema does not expose every window, but worst_window.total_return
# is the floor; if it's >= 0 then neg=0/50.
neg = 0 if worst >= 0 else -1   # -1 = unknown (would need windows array)
print(f"{med:.2f},{p10:.2f},{worst:.2f},{neg},{sort:.2f}")
PY
)
  [ -z "$stats" ] && { echo "  s${seed}: parse failed"; return; }
  ts=$(date -u +%FT%TZ)
  echo "$ts,$seed,$stats,$ckpt" >> "$LOG"
  echo "  s${seed}: $stats"
}

# Sequential seed batch. Add more seeds later once GPU is free.
SEEDS=(100 101 102 103 104)
echo "=== training v5_rsi crypto-recipe seeds: ${SEEDS[*]} ==="
for s in "${SEEDS[@]}"; do train_one "$s"; eval_one "$s"; done

echo
echo "=== LEADERBOARD ==="
sort -t, -k3 -rn "$LOG" | head -20
