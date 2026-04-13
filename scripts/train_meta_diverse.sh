#!/bin/bash
# Train diverse models for meta-selector on stocks12
# Designed for maximum model diversity to improve meta-selection
#
# Run on daisy: bash scripts/train_meta_diverse.sh
set +e  # don't exit on error, keep training remaining models

DATA=pufferlib_market/data/stocks12_fresh_train.bin
VAL=pufferlib_market/data/stocks12_fresh_val.bin
BASE_DIR=pufferlib_market/checkpoints/meta_diverse
STEPS=10000000  # 10M steps (shorter for diversity, not perfection)

source .venv/bin/activate

# 16 diverse configs: vary seed, wd, tp, hidden
declare -a CONFIGS=(
    "s1_wd00_tp05_h1024 --seed 1 --weight-decay 0.0  --trade-penalty 0.05 --hidden-size 1024"
    "s2_wd03_tp05_h1024 --seed 2 --weight-decay 0.03 --trade-penalty 0.05 --hidden-size 1024"
    "s3_wd06_tp05_h1024 --seed 3 --weight-decay 0.06 --trade-penalty 0.05 --hidden-size 1024"
    "s4_wd00_tp02_h1024 --seed 4 --weight-decay 0.0  --trade-penalty 0.02 --hidden-size 1024"
    "s5_wd03_tp02_h1024 --seed 5 --weight-decay 0.03 --trade-penalty 0.02 --hidden-size 1024"
    "s6_wd06_tp02_h1024 --seed 6 --weight-decay 0.06 --trade-penalty 0.02 --hidden-size 1024"
    "s7_wd00_tp05_h256  --seed 7 --weight-decay 0.0  --trade-penalty 0.05 --hidden-size 256"
    "s8_wd03_tp05_h256  --seed 8 --weight-decay 0.03 --trade-penalty 0.05 --hidden-size 256"
    "s9_wd06_tp05_h256  --seed 9 --weight-decay 0.06 --trade-penalty 0.05 --hidden-size 256"
    "s10_wd00_tp02_h256 --seed 10 --weight-decay 0.0 --trade-penalty 0.02 --hidden-size 256"
    "s11_wd04_tp03_h512 --seed 11 --weight-decay 0.04 --trade-penalty 0.03 --hidden-size 512"
    "s12_wd02_tp04_h512 --seed 12 --weight-decay 0.02 --trade-penalty 0.04 --hidden-size 512"
    "s13_wd01_tp01_h1024 --seed 13 --weight-decay 0.01 --trade-penalty 0.01 --hidden-size 1024"
    "s14_wd05_tp05_h1024 --seed 14 --weight-decay 0.05 --trade-penalty 0.05 --hidden-size 1024"
    "s15_wd00_tp05_h1024 --seed 15 --weight-decay 0.0  --trade-penalty 0.05 --hidden-size 1024"
    "s16_wd03_tp02_h256  --seed 16 --weight-decay 0.03 --trade-penalty 0.02 --hidden-size 256"
)

for cfg in "${CONFIGS[@]}"; do
    NAME=$(echo "$cfg" | awk '{print $1}')
    ARGS=$(echo "$cfg" | cut -d' ' -f2-)
    DIR="${BASE_DIR}/${NAME}"

    if [ -f "${DIR}/final.pt" ]; then
        echo "SKIP $NAME (final.pt exists)"
        continue
    fi

    echo "=== Training $NAME ==="
    mkdir -p "$DIR"
    python -u -m pufferlib_market.train \
        --data-path "$DATA" \
        --val-data-path "$VAL" \
        --total-timesteps "$STEPS" \
        --max-steps 252 \
        --anneal-lr \
        --disable-shorts \
        --num-envs 128 \
        --checkpoint-dir "$DIR" \
        $ARGS \
        2>&1 | tee "${DIR}/train.log"

    echo "=== Done $NAME ==="
done

echo "All training complete. Checkpoints in $BASE_DIR"
