#!/bin/bash
# Leverage sweep on screened32 daily data.
#
# Tests 1x/1.5x/2x/3x leverage combined with downside penalty on proven
# screened32 configs (C=adamw/tp0.02, D=muon/tp0.05). 5 seeds per config.
#
# Usage:
#   nohup bash scripts/sweep_screened32_leverage.sh C > /tmp/s32lev_C.log 2>&1 &
#   nohup bash scripts/sweep_screened32_leverage.sh D > /tmp/s32lev_D.log 2>&1 &

set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

export TMPDIR="$(pwd)/.tmp_train"
mkdir -p "$TMPDIR"

VARIANT="${1:-C}"
TRAIN="${2:-pufferlib_market/data/screened32_augmented_train.bin}"
VAL="${3:-pufferlib_market/data/screened32_augmented_val.bin}"
SEEDS="${SEEDS:-1 2 3 4 5}"

case "$VARIANT" in
  C) BASE_TP=0.02; BASE_FLAGS=()                ;;
  D) BASE_TP=0.05; BASE_FLAGS=(--optimizer muon) ;;
  *) echo "Unknown variant: $VARIANT"; exit 1   ;;
esac

CKPT_ROOT="pufferlib_market/checkpoints/screened32_leverage_sweep/${VARIANT}"
LOG="${CKPT_ROOT}/leaderboard.csv"
mkdir -p "$CKPT_ROOT"
[ -f "$LOG" ] || echo "timestamp,variant,lev_tag,seed,med_pct,p10_pct,worst_pct,neg_count,med_sortino,checkpoint" > "$LOG"

# TAG:MAX_LEV:DOWNSIDE_PEN
LEV_CONFIGS=(
  "lev1x_base:1.0:0.0"
  "lev1p5x:1.5:0.0"
  "lev1p5x_ds03:1.5:0.3"
  "lev2x_ds03:2.0:0.3"
  "lev2x_ds05:2.0:0.5"
  "lev3x_ds05:3.0:0.5"
)

train_one() {
  local lev_tag="$1" max_lev="$2" ds_pen="$3" seed="$4"
  local dir="${CKPT_ROOT}/${lev_tag}/s${seed}"
  mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] [${VARIANT} ${lev_tag}] seed ${seed}"
  python -u -m pufferlib_market.train \
      --data-path "$TRAIN" --val-data-path "$VAL" \
      --total-timesteps 15000000 --max-steps 252 \
      --trade-penalty "$BASE_TP" --hidden-size 1024 \
      --anneal-lr --disable-shorts --num-envs 128 \
      --val-eval-windows 30 \
      --max-leverage "$max_lev" --downside-penalty "$ds_pen" \
      "${BASE_FLAGS[@]}" \
      --seed "$seed" --checkpoint-dir "$dir" > "$dir/train.log" 2>&1
  echo "[$(date -u +%FT%TZ)] [${VARIANT} ${lev_tag}] seed ${seed} done"
}

eval_one() {
  local lev_tag="$1" seed="$2" max_lev="${3:-1.0}"
  local ckpt="${CKPT_ROOT}/${lev_tag}/s${seed}/val_best.pt"
  [ -f "$ckpt" ] || ckpt="${CKPT_ROOT}/${lev_tag}/s${seed}/best.pt"
  [ -f "$ckpt" ] || { echo "  ${lev_tag} s${seed}: no ckpt"; return; }
  local out="${CKPT_ROOT}/${lev_tag}/s${seed}/eval_lag2.json"
  python -m pufferlib_market.evaluate_holdout \
      --checkpoint "$ckpt" --data-path "$VAL" \
      --eval-hours 50 --n-windows 30 \
      --fee-rate 0.001 --fill-buffer-bps 5.0 \
      --max-leverage "$max_lev" \
      --decision-lag 2 --deterministic --no-early-stop \
      > "$out" 2>/dev/null || { echo "  ${lev_tag} s${seed}: eval failed"; return; }
  stats=$(python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{d.get('median_total_return',0)*100:.2f},{d.get('p10_total_return',0)*100:.2f},{(d.get('worst_window') or {}).get('total_return',0)*100:.2f},{d.get('negative_windows',0)},{d.get('median_sortino',0):.2f}")
PY
  )
  [ -z "$stats" ] && return
  ts=$(date -u +%FT%TZ)
  echo "$ts,$VARIANT,$lev_tag,$seed,$stats,$ckpt" >> "$LOG"
  echo "  ${lev_tag} s${seed}: ${stats}"
}

nts=$(python3 -c "import struct; h=open('$TRAIN','rb').read(24); v=struct.unpack_from('<IIII',h,4); print(f'{v[1]} syms, {v[2]} ts, {v[3]} feats')" 2>/dev/null || echo "?")
echo "=== screened32 leverage sweep: variant=${VARIANT} ($nts) ==="

for cfg in "${LEV_CONFIGS[@]}"; do
  IFS=':' read -r tag max_lev ds_pen <<< "$cfg"
  echo ""
  echo "--- ${tag} (lev=${max_lev} ds_pen=${ds_pen}) ---"
  for s in $SEEDS; do
    eval_json="${CKPT_ROOT}/${tag}/s${s}/eval_lag2.json"
    [ -f "$eval_json" ] && { echo "  s${s}: done, skip"; continue; }
    train_one "$tag" "$max_lev" "$ds_pen" "$s"
    eval_one "$tag" "$s" "$max_lev"
  done
done

echo ""
echo "=== [${VARIANT}] LEVERAGE LEADERBOARD ==="
sort -t, -k5 -rn "$LOG" 2>/dev/null | head -20
