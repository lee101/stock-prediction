#!/bin/bash
# Fine-tune Chronos2 LoRA for all Alpaca trading symbols
# Uses proven differencing preaug (33.7% MAE improvement on QUBT)
# Only promotes configs that beat baseline by >5%
#
# Usage: nohup bash scripts/finetune_alpaca_symbols.sh > /tmp/alpaca_finetune.log 2>&1 &

set -e
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

DATA_ROOT="trainingdatahourly"
OUTPUT_ROOT="chronos2_finetuned"
RESULTS_DIR="analysis/alpaca_chronos2_finetune"
PROMOTE_DIR="hyperparams/chronos2/hourly"
BATCH=16
STEPS=300
IMPROVEMENT_THRESHOLD=5  # % improvement over baseline required for promotion

mkdir -p "$RESULTS_DIR"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# All Alpaca live symbols
SYMBOLS="NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT,AAPL,MSFT,META,TSLA,NET,BKNG,EBAY,EXPE,ITUB,BTG,ABEV"

RUN_ID="alpaca_finetune_$(date -u +%Y%m%d_%H%M)"
log "Starting Alpaca symbol fine-tune sweep: $RUN_ID"
log "Symbols: $SYMBOLS"
log "Config: differencing preaug, ctx=[256,512], lr=[5e-5,1e-4], r=8"

python -u scripts/chronos2_lora_improvement_sweep.py \
    --run-id "$RUN_ID" \
    --symbols "$SYMBOLS" \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --results-dir "$RESULTS_DIR/$RUN_ID" \
    --lora-rs 8 \
    --learning-rates "5e-5,1e-4" \
    --preaugs "differencing,percent_change" \
    --context-lengths "256,512" \
    --batch-size $BATCH \
    --num-steps $STEPS \
    --prediction-length 24 \
    --improvement-threshold $IMPROVEMENT_THRESHOLD \
    2>&1 | tee -a /tmp/alpaca_finetune.log

# Promote any improved symbols
log "Promoting improved symbols..."
python scripts/promote_chronos2_lora_reports.py \
    --report-dir "$RESULTS_DIR/$RUN_ID" \
    --output-dir "$PROMOTE_DIR" \
    --symbols "$SYMBOLS" \
    --run-id "$RUN_ID" \
    --metric val_mae_percent \
    2>&1 | tee -a /tmp/alpaca_finetune.log

log "Fine-tuning complete. Updated hyperparams in: $PROMOTE_DIR"
log "Results: $RESULTS_DIR/$RUN_ID"
