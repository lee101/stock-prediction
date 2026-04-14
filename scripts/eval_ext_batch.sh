#!/bin/bash
set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

EXT_VAL="pufferlib_market/data/screened32_ext_augmented_val.bin"
LOG="pufferlib_market/checkpoints/screened32_ext_sweep/eval_results.log"
echo "" > "$LOG"

eval_one() {
  local ckpt="$1"
  local label="$2"
  local out="${ckpt%/*}/eval_lag2.json"
  if [ -f "$out" ] && [ -s "$out" ]; then
    python3 -c "import json; d=json.load(open('$out')); print(f'  ${label}: med={d[\"median_total_return\"]*100:.2f}% p10={d[\"p10_total_return\"]*100:.2f}% neg={d[\"negative_windows\"]}/30 sort={d[\"median_sortino\"]:.2f} [cached]')" 2>/dev/null
    return
  fi
  echo "  Evaluating $label..."
  python -m pufferlib_market.evaluate_holdout \
    --checkpoint "$ckpt" --data-path "$EXT_VAL" \
    --eval-hours 90 --n-windows 30 --fee-rate 0.001 \
    --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop \
    > "$out" 2>/dev/null
  if [ -s "$out" ]; then
    python3 -c "import json; d=json.load(open('$out')); print(f'  ${label}: med={d[\"median_total_return\"]*100:.2f}% p10={d[\"p10_total_return\"]*100:.2f}% neg={d[\"negative_windows\"]}/30 sort={d[\"median_sortino\"]:.2f}')" 2>/dev/null
  else
    echo "  ${label}: FAILED"
  fi
}

echo "=== Ext sweep C variant ===" | tee -a "$LOG"
for s in $(seq 1 20); do
  dir="pufferlib_market/checkpoints/screened32_ext_sweep/C/s${s}"
  ckpt="$dir/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$dir/best.pt"
  [ -f "$ckpt" ] || continue
  eval_one "$ckpt" "ext_C_s${s}" | tee -a "$LOG"
done

echo "=== Ext sweep D variant ===" | tee -a "$LOG"
for s in $(seq 1 20); do
  dir="pufferlib_market/checkpoints/screened32_ext_sweep/D/s${s}"
  ckpt="$dir/val_best.pt"
  [ -f "$ckpt" ] || ckpt="$dir/best.pt"
  [ -f "$ckpt" ] || continue
  eval_one "$ckpt" "ext_D_s${s}" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== PROD ensemble on EXT val ===" | tee -a "$LOG"
python -m pufferlib_market.evaluate_holdout \
  --checkpoint pufferlib_market/prod_ensemble_screened32/C_s7.pt \
  --extra-checkpoints \
    pufferlib_market/prod_ensemble_screened32/D_s16.pt \
    pufferlib_market/prod_ensemble_screened32/D_s42.pt \
    pufferlib_market/prod_ensemble_screened32/D_s3.pt \
    pufferlib_market/prod_ensemble_screened32/I_s3.pt \
    pufferlib_market/prod_ensemble_screened32/D_s2.pt \
    pufferlib_market/prod_ensemble_screened32/D_s14.pt \
    pufferlib_market/prod_ensemble_screened32/D_s28.pt \
  --data-path "$EXT_VAL" \
  --eval-hours 90 --n-windows 30 --fee-rate 0.001 \
  --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop \
  2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'  PROD 8-model: med={d[\"median_total_return\"]*100:.2f}% p10={d[\"p10_total_return\"]*100:.2f}% neg={d[\"negative_windows\"]}/30 sort={d[\"median_sortino\"]:.2f}')
" 2>/dev/null | tee -a "$LOG"

echo "[DONE] $(date)" | tee -a "$LOG"
