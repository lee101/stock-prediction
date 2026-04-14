#!/usr/bin/env bash
# ============================================================================
# Auto-launch v4 after v3 completes, with calibration and benchmark in between.
#
# Sequence:
#   1. Wait for v3 PID to finish (polls .chronos2_v3_pid or PID 2029405)
#   2. Run v3 calibration (with signal_weight search, long-only + short)
#   3. Run quick benchmark: v3 vs v2 on AAPL,SPY,GOOG,TSLA,META
#   4. Launch v4 (ctx=1024, 200k steps)
#
# Usage: nohup bash scripts/launch_chronos2_v4_when_v3_ready.sh > chronos2_v4_launcher.log 2>&1 &
# ============================================================================
set -euo pipefail
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

# Map CLOUDFLARE_R2_* -> R2_*
[[ -z "${R2_ACCESS_KEY:-}" && -n "${CLOUDFLARE_R2_ACCESS_KEY_ID:-}" ]] && \
    export R2_ACCESS_KEY="$CLOUDFLARE_R2_ACCESS_KEY_ID"
[[ -z "${R2_SECRET_KEY:-}" && -n "${CLOUDFLARE_R2_SECRET_ACCESS_KEY:-}" ]] && \
    export R2_SECRET_KEY="$CLOUDFLARE_R2_SECRET_ACCESS_KEY"
export R2_BUCKET="${R2_BUCKET:-models}"

source .venv/bin/activate

# -----------------------------------------------------------------------
# Step 1: Wait for v3 to finish
# -----------------------------------------------------------------------
V3_PID=""
if [[ -f .chronos2_v3_pid ]]; then
    V3_PID="$(cat .chronos2_v3_pid)"
fi
if [[ -z "$V3_PID" ]]; then
    echo "[$(date -u +%H:%M:%SZ)] ERROR: .chronos2_v3_pid not found"
    exit 1
fi

echo "[$(date -u +%H:%M:%SZ)] Waiting for v3 (PID $V3_PID) to complete..."
while kill -0 "$V3_PID" 2>/dev/null; do
    sleep 120
done
echo "[$(date -u +%H:%M:%SZ)] v3 training complete."

# -----------------------------------------------------------------------
# Step 2: Calibrate v3 (long-only + short variants, with signal_weight search)
# -----------------------------------------------------------------------
V3_CKPT="chronos2_finetuned/stocks_all_v3/finetuned-ckpt"
if [[ ! -d "$V3_CKPT" ]]; then
    echo "[$(date -u +%H:%M:%SZ)] ERROR: v3 checkpoint not found at $V3_CKPT"
    exit 1
fi

echo "[$(date -u +%H:%M:%SZ)] Running v3 calibration (long-only + short-allowed + per-symbol)..."
python chronos2_linear_calibration.py \
    --model-id     "$V3_CKPT" \
    --cal-data-dir trainingdata \
    --output-path  "$V3_CKPT/calibration.json" \
    --max-shift-bps 20 \
    --min-gap-bps   2 \
    --grid-steps   25 \
    --cal-bars     120 \
    --max-windows  5000 \
    --batch-size   32 \
    --per-symbol \
    --hyperparams-dir hyperparams/chronos2_v3 \
    2>&1 | tee chronos2_calibration_v3.log

# -----------------------------------------------------------------------
# Step 3: Benchmark v3 vs v2 on key symbols (same-period comparison)
# -----------------------------------------------------------------------
echo "[$(date -u +%H:%M:%SZ)] Benchmarking v3 on key symbols (MAE)..."
python benchmark_chronos2.py \
    --symbols AAPL SPY GOOG TSLA META NVDA MSFT AMZN \
    --model-id "$V3_CKPT" \
    --context-length 512 \
    --batch-size 128 \
    --update-hyperparams \
    2>&1 | tee chronos2_benchmark_v3.log

echo "[$(date -u +%H:%M:%SZ)] Running calibrated backtest (v3 vs v2, Sharpe comparison)..."
python chronos2_calibrated_backtest.py \
    --model-id "$V3_CKPT" \
    --compare-model "chronos2_finetuned/stocks_all_v2/finetuned-ckpt" \
    --cal-data-dir trainingdata \
    --max-windows 5000 \
    --batch-size 32 \
    --per-symbol \
    --output "chronos2_finetuned/stocks_all_v3/backtest_vs_v2.json" \
    2>&1 | tee chronos2_backtest_v3_vs_v2.log || true

# -----------------------------------------------------------------------
# Step 4: Upload v3 calibration to R2
# -----------------------------------------------------------------------
if [[ -n "${R2_ACCESS_KEY:-}" ]]; then
    echo "[$(date -u +%H:%M:%SZ)] Uploading v3 calibration to R2..."
    python3 -c "
import json, sys
sys.path.insert(0, '.')
from src.r2_client import upload_file
upload_file('$V3_CKPT/calibration.json',       'models/chronos2/finetune/stocks_all_v3/calibration.json')
upload_file('$V3_CKPT/calibration_short.json', 'models/chronos2/finetune/stocks_all_v3/calibration_short.json')
print('R2 upload complete')
" 2>&1 || echo "R2 upload failed (not critical)"
fi

# -----------------------------------------------------------------------
# Step 5: Launch v4
# -----------------------------------------------------------------------
echo "[$(date -u +%H:%M:%SZ)] Launching v4 (ctx=1024, 200k steps, grad_accum=2)..."
bash scripts/launch_chronos2_v4.sh
echo "[$(date -u +%H:%M:%SZ)] v4 launched. PID: $(cat .chronos2_v4_pid 2>/dev/null)"

# -----------------------------------------------------------------------
# Step 6: Auto-launch v5 when v4 completes
# -----------------------------------------------------------------------
V4_PID="$(cat .chronos2_v4_pid 2>/dev/null)"
if [[ -n "$V4_PID" ]]; then
    echo "[$(date -u +%H:%M:%SZ)] Waiting for v4 (PID $V4_PID) to complete before launching v5..."
    while kill -0 "$V4_PID" 2>/dev/null; do
        sleep 180
    done
    echo "[$(date -u +%H:%M:%SZ)] v4 complete. Running v4 calibration + benchmark..."

    V4_CKPT="chronos2_finetuned/stocks_all_v4/finetuned-ckpt"
    if [[ -d "$V4_CKPT" ]]; then
        python chronos2_linear_calibration.py \
            --model-id     "$V4_CKPT" \
            --cal-data-dir trainingdata \
            --output-path  "$V4_CKPT/calibration.json" \
            --max-shift-bps 20 \
            --min-gap-bps   2 \
            --grid-steps   25 \
            --cal-bars     120 \
            --max-windows  5000 \
            --batch-size   32 \
            --per-symbol \
            --hyperparams-dir hyperparams/chronos2_v4 \
            2>&1 | tee chronos2_calibration_v4.log

        python benchmark_chronos2.py \
            --symbols AAPL SPY GOOG TSLA META NVDA MSFT AMZN \
            --model-id "$V4_CKPT" \
            --context-length 1024 \
            --batch-size 64 \
            --update-hyperparams \
            2>&1 | tee chronos2_benchmark_v4.log
    fi

    echo "[$(date -u +%H:%M:%SZ)] Launching v5 (ctx=1024, channel_dropout, time_warp)..."
    bash scripts/launch_chronos2_v5.sh
    echo "[$(date -u +%H:%M:%SZ)] v5 launched. PID: $(cat .chronos2_v5_pid 2>/dev/null)"

    # -----------------------------------------------------------------------
    # Wait for v5 then calibrate and launch v6
    # -----------------------------------------------------------------------
    V5_PID="$(cat .chronos2_v5_pid 2>/dev/null)"
    if [[ -n "$V5_PID" ]]; then
        echo "[$(date -u +%H:%M:%SZ)] Waiting for v5 (PID $V5_PID) to complete before launching v6..."
        while kill -0 "$V5_PID" 2>/dev/null; do
            sleep 180
        done
        echo "[$(date -u +%H:%M:%SZ)] v5 complete. Running v5 calibration + benchmark..."

        V5_CKPT="chronos2_finetuned/stocks_all_v5/finetuned-ckpt"
        if [[ -d "$V5_CKPT" ]]; then
            python chronos2_linear_calibration.py \
                --model-id     "$V5_CKPT" \
                --cal-data-dir trainingdata \
                --output-path  "$V5_CKPT/calibration.json" \
                --max-shift-bps 20 \
                --min-gap-bps   2 \
                --grid-steps   25 \
                --cal-bars     120 \
                --max-windows  5000 \
                --batch-size   32 \
                --per-symbol \
                --hyperparams-dir hyperparams/chronos2_v5 \
                2>&1 | tee chronos2_calibration_v5.log

            python benchmark_chronos2.py \
                --symbols AAPL SPY GOOG TSLA META NVDA MSFT AMZN \
                --model-id "$V5_CKPT" \
                --context-length 1024 \
                --batch-size 64 \
                --update-hyperparams \
                2>&1 | tee chronos2_benchmark_v5.log
        fi

        echo "[$(date -u +%H:%M:%SZ)] Launching v6 (ctx=1024, all augs + outlier_inject)..."
        bash scripts/launch_chronos2_v6.sh
        echo "[$(date -u +%H:%M:%SZ)] v6 launched. PID: $(cat .chronos2_v6_pid 2>/dev/null)"
    fi
fi
