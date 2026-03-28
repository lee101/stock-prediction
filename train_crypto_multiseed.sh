#!/usr/bin/env bash
set -euo pipefail
cd /home/lee/code/stock
source .venv312/bin/activate

SYMBOLS="${SYMBOLS:-BTCUSD,ETHUSD,SOLUSD,DOGEUSD,AAVEUSD,LINKUSD}"
EPOCHS="${EPOCHS:-15}"
SEEDS="${SEEDS:-1337 42 7}"
DIM="${DIM:-384}"
LAYERS="${LAYERS:-6}"
HEADS="${HEADS:-6}"

for seed in $SEEDS; do
    RUN="crypto_portfolio_seed${seed}"
    echo "=== Training seed=$seed run=$RUN ==="
    TORCH_NO_COMPILE=1 PYTHONUNBUFFERED=1 python scripts/train_crypto_portfolio.py \
        --symbols "$SYMBOLS" \
        --epochs "$EPOCHS" \
        --batch-size 16 \
        --lr 1e-5 \
        --weight-decay 0.04 \
        --transformer-dim "$DIM" \
        --transformer-layers "$LAYERS" \
        --transformer-heads "$HEADS" \
        --sequence-length 48 \
        --lr-schedule cosine \
        --maker-fee 0.001 \
        --margin-rate 0.0625 \
        --max-leverage 5.0 \
        --fill-buffer 0.0005 \
        --return-weight 0.10 \
        --no-compile \
        --run-name "$RUN" \
        --seed "$seed" \
        2>&1 | tee "logs/${RUN}.log"
    echo "=== seed=$seed done ==="
done

echo "All seeds complete. Compare with:"
echo "  python scripts/compare_multiseed.py"
