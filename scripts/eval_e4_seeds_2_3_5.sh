#!/bin/bash
# Serial 14th-member candidate eval on E4 leverage-D seeds 2, 3, 5.
# Each run writes reports/e4_lev2x_ds03_s{2,3,5}_14th_candidate.json.
set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate

VAL=pufferlib_market/data/screened32_single_offset_val_full.bin
BASE="pufferlib_market/prod_ensemble_screened32/C_s7.pt"
EXTRAS=(
  pufferlib_market/prod_ensemble_screened32/D_s16.pt
  pufferlib_market/prod_ensemble_screened32/D_s42.pt
  pufferlib_market/prod_ensemble_screened32/AD_s4.pt
  pufferlib_market/prod_ensemble_screened32/I_s3.pt
  pufferlib_market/prod_ensemble_screened32/D_s2.pt
  pufferlib_market/prod_ensemble_screened32/D_s14.pt
  pufferlib_market/prod_ensemble_screened32/D_s28.pt
  pufferlib_market/prod_ensemble_screened32/D_s81.pt
  pufferlib_market/prod_ensemble_screened32/D_s57.pt
  pufferlib_market/prod_ensemble_screened32/I_s3.pt
  pufferlib_market/prod_ensemble_screened32/D_s64.pt
  pufferlib_market/prod_ensemble_screened32/I_s32.pt
)

for SEED in 2 3 5; do
  CAND=pufferlib_market/checkpoints/screened32_leverage_sweep/D/lev2x_ds03/s${SEED}/val_best.pt
  OUT=reports/e4_lev2x_ds03_s${SEED}_14th_candidate.json
  echo "=== E4 leverage-D s${SEED} as 14th member ==="
  python scripts/eval_multihorizon_candidate.py \
    --data-path "$VAL" \
    --baseline-checkpoint "$BASE" \
    --baseline-extra-checkpoints "${EXTRAS[@]}" \
    --candidate-checkpoint "$CAND" \
    --candidate-extra-checkpoints "$BASE" "${EXTRAS[@]}" \
    --horizons-days 30,100 \
    --slippage-bps 5 \
    --fill-buffer-bps 5 \
    --decision-lag 2 \
    --recent-within-days 140 \
    --n-windows 24 \
    --out "$OUT"
  echo "[saved] $OUT"
done
