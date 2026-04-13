#!/bin/bash
# Test if removing any model from the 8-model ensemble improves results.
# This helps identify weak models that hurt ensemble performance.
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

FULL_VAL="pufferlib_market/data/screened32_single_offset_val_full.bin"

MODELS=(
  "pufferlib_market/prod_ensemble_screened32/C_s7.pt"
  "pufferlib_market/prod_ensemble_screened32/D_s16.pt"
  "pufferlib_market/prod_ensemble_screened32/D_s42.pt"
  "pufferlib_market/prod_ensemble_screened32/D_s3.pt"
  "pufferlib_market/prod_ensemble_screened32/D_s5.pt"
  "pufferlib_market/prod_ensemble_screened32/D_s2.pt"
  "pufferlib_market/prod_ensemble_screened32/D_s14.pt"
  "pufferlib_market/prod_ensemble_screened32/D_s28.pt"
)

LABELS=("C_s7" "D_s16" "D_s42" "D_s3" "D_s5" "D_s2" "D_s14" "D_s28")

echo "[$(date -u +%FT%TZ)] Testing 8-model ensemble baseline (100 windows)..."
python -m pufferlib_market.evaluate_holdout \
  --checkpoint "${MODELS[0]}" \
  --extra-checkpoints "${MODELS[@]:1}" \
  --data-path "$FULL_VAL" \
  --eval-hours 50 --n-windows 100 --fee-rate 0.001 \
  --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop \
  2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
med = d.get('median_total_return',0)*100
p10 = d.get('p10_total_return',0)*100
neg = d.get('negative_windows',0)
s = d.get('median_sortino',0)
print(f'BASELINE 8-model: med={med:.2f}% p10={p10:.2f}% neg={neg}/100 sort={s:.2f}')
" 2>/dev/null
echo ""

for i in "${!MODELS[@]}"; do
  removed="${LABELS[$i]}"
  # Build ensemble without model i
  others=()
  for j in "${!MODELS[@]}"; do
    [ "$j" -eq "$i" ] && continue
    others+=("${MODELS[$j]}")
  done
  primary="${others[0]}"
  extras=("${others[@]:1}")
  echo -n "Removing ${removed}: "
  python -m pufferlib_market.evaluate_holdout \
    --checkpoint "$primary" \
    --extra-checkpoints "${extras[@]}" \
    --data-path "$FULL_VAL" \
    --eval-hours 50 --n-windows 100 --fee-rate 0.001 \
    --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop \
    2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
med = d.get('median_total_return',0)*100
p10 = d.get('p10_total_return',0)*100
neg = d.get('negative_windows',0)
s = d.get('median_sortino',0)
print(f'7-model (no $removed): med={med:.2f}% p10={p10:.2f}% neg={neg}/100 sort={s:.2f}')
" 2>/dev/null
done

echo ""
echo "Done."
