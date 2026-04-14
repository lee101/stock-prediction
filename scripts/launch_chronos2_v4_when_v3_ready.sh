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
# SWA: Average last 3 checkpoints for v3
# -----------------------------------------------------------------------
V3_CKPT="chronos2_finetuned/stocks_all_v3/finetuned-ckpt"
V3_SWA="chronos2_finetuned/stocks_all_v3/swa-ckpt"
if [[ -d "chronos2_finetuned/stocks_all_v3/trainer_workspace" ]]; then
    echo "[$(date -u +%H:%M:%SZ)] Creating v3 SWA checkpoint (average last 3)..."
    python scripts/average_checkpoints.py \
        --trainer-workspace chronos2_finetuned/stocks_all_v3/trainer_workspace \
        --output "$V3_SWA" \
        --n-last 3 \
        --copy-chronos-config "$V3_CKPT" \
        2>&1 | tee chronos2_swa_v3.log || echo "[WARN] SWA failed, continuing with finetuned-ckpt only"
fi

# -----------------------------------------------------------------------
# Step 2: Calibrate v3 (long-only + short variants, with signal_weight search)
# -----------------------------------------------------------------------
if [[ ! -d "$V3_CKPT" ]]; then
    echo "[$(date -u +%H:%M:%SZ)] ERROR: v3 checkpoint not found at $V3_CKPT"
    exit 1
fi

echo "[$(date -u +%H:%M:%SZ)] Running v3 calibration (global + calmar, prediction-length=2)..."
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
    --prediction-length 2 \
    --also-calmar \
    2>&1 | tee chronos2_calibration_v3.log

echo "[$(date -u +%H:%M:%SZ)] Running v3 per-symbol calibration (key symbols, cal-bars=365)..."
python chronos2_linear_calibration.py \
    --model-id     "$V3_CKPT" \
    --cal-data-dir trainingdata \
    --output-path  "$V3_CKPT/calibration.json" \
    --max-shift-bps 20 \
    --min-gap-bps   2 \
    --grid-steps   25 \
    --cal-bars     365 \
    --max-windows  50000 \
    --batch-size   32 \
    --prediction-length 2 \
    --per-symbol \
    --min-windows-per-symbol 30 \
    --symbols-subset AAPL SPY GOOG GOOGL TSLA META NVDA MSFT AMZN QQQ VTI AMD AVGO CRM \
    --hyperparams-dir hyperparams/chronos2_v3 \
    --no-search-signal-weight \
    2>&1 | tee chronos2_calibration_v3_persym.log

# SWA calibration (if SWA ckpt exists)
if [[ -d "$V3_SWA" ]]; then
    echo "[$(date -u +%H:%M:%SZ)] Running v3 SWA calibration..."
    python chronos2_linear_calibration.py \
        --model-id     "$V3_SWA" \
        --cal-data-dir trainingdata \
        --output-path  "$V3_SWA/calibration.json" \
        --max-shift-bps 20 \
        --min-gap-bps   2 \
        --grid-steps   25 \
        --cal-bars     120 \
        --max-windows  5000 \
        --batch-size   32 \
        --prediction-length 2 \
        --also-calmar \
        2>&1 | tee chronos2_calibration_v3_swa.log
fi

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
    echo "[$(date -u +%H:%M:%SZ)] v4 complete. Running v4 SWA + calibration + benchmark..."

    V4_CKPT="chronos2_finetuned/stocks_all_v4/finetuned-ckpt"
    V4_SWA="chronos2_finetuned/stocks_all_v4/swa-ckpt"
    if [[ -d "chronos2_finetuned/stocks_all_v4/trainer_workspace" ]]; then
        python scripts/average_checkpoints.py \
            --trainer-workspace chronos2_finetuned/stocks_all_v4/trainer_workspace \
            --output "$V4_SWA" --n-last 3 \
            --copy-chronos-config "$V4_CKPT" 2>&1 | tee chronos2_swa_v4.log || true
    fi
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
            --prediction-length 2 \
            --also-calmar \
            --hyperparams-dir hyperparams/chronos2_v4 \
            2>&1 | tee chronos2_calibration_v4.log

        if [[ -d "$V4_SWA" ]]; then
            python chronos2_linear_calibration.py \
                --model-id "$V4_SWA" --cal-data-dir trainingdata \
                --output-path "$V4_SWA/calibration.json" \
                --max-shift-bps 20 --min-gap-bps 2 --grid-steps 25 \
                --cal-bars 120 --max-windows 5000 --batch-size 32 \
                --prediction-length 2 --also-calmar \
                2>&1 | tee chronos2_calibration_v4_swa.log || true
        fi

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
        echo "[$(date -u +%H:%M:%SZ)] v5 complete. Running v5 SWA + calibration + benchmark..."

        V5_CKPT="chronos2_finetuned/stocks_all_v5/finetuned-ckpt"
        V5_SWA="chronos2_finetuned/stocks_all_v5/swa-ckpt"
        if [[ -d "chronos2_finetuned/stocks_all_v5/trainer_workspace" ]]; then
            python scripts/average_checkpoints.py \
                --trainer-workspace chronos2_finetuned/stocks_all_v5/trainer_workspace \
                --output "$V5_SWA" --n-last 3 \
                --copy-chronos-config "$V5_CKPT" 2>&1 | tee chronos2_swa_v5.log || true
        fi
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
                --prediction-length 2 \
                --also-calmar \
                --hyperparams-dir hyperparams/chronos2_v5 \
                2>&1 | tee chronos2_calibration_v5.log

            if [[ -d "${V5_SWA:-}" ]]; then
                python chronos2_linear_calibration.py \
                    --model-id "$V5_SWA" --cal-data-dir trainingdata \
                    --output-path "$V5_SWA/calibration.json" \
                    --max-shift-bps 20 --min-gap-bps 2 --grid-steps 25 \
                    --cal-bars 120 --max-windows 5000 --batch-size 32 \
                    --prediction-length 2 --also-calmar \
                    2>&1 | tee chronos2_calibration_v5_swa.log || true
            fi
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

        # -----------------------------------------------------------------------
        # Wait for v6 then calibrate and launch v7
        # -----------------------------------------------------------------------
        V6_PID="$(cat .chronos2_v6_pid 2>/dev/null)"
        if [[ -n "$V6_PID" ]]; then
            echo "[$(date -u +%H:%M:%SZ)] Waiting for v6 (PID $V6_PID) to complete before launching v7..."
            while kill -0 "$V6_PID" 2>/dev/null; do
                sleep 180
            done
            echo "[$(date -u +%H:%M:%SZ)] v6 complete. Running v6 SWA + calibration + benchmark..."

            V6_CKPT="chronos2_finetuned/stocks_all_v6/finetuned-ckpt"
            V6_SWA="chronos2_finetuned/stocks_all_v6/swa-ckpt"
            if [[ -d "chronos2_finetuned/stocks_all_v6/trainer_workspace" ]]; then
                python scripts/average_checkpoints.py \
                    --trainer-workspace chronos2_finetuned/stocks_all_v6/trainer_workspace \
                    --output "$V6_SWA" --n-last 3 \
                    --copy-chronos-config "$V6_CKPT" 2>&1 | tee chronos2_swa_v6.log || true
            fi
            if [[ -d "$V6_CKPT" ]]; then
                python chronos2_linear_calibration.py \
                    --model-id     "$V6_CKPT" \
                    --cal-data-dir trainingdata \
                    --output-path  "$V6_CKPT/calibration.json" \
                    --max-shift-bps 20 \
                    --min-gap-bps   2 \
                    --grid-steps   25 \
                    --cal-bars     120 \
                    --max-windows  5000 \
                    --batch-size   32 \
                    --per-symbol \
                    --prediction-length 2 \
                    --also-calmar \
                    --hyperparams-dir hyperparams/chronos2_v6 \
                    2>&1 | tee chronos2_calibration_v6.log

                if [[ -d "${V6_SWA:-}" ]]; then
                    python chronos2_linear_calibration.py \
                        --model-id "$V6_SWA" --cal-data-dir trainingdata \
                        --output-path "$V6_SWA/calibration.json" \
                        --max-shift-bps 20 --min-gap-bps 2 --grid-steps 25 \
                        --cal-bars 120 --max-windows 5000 --batch-size 32 \
                        --prediction-length 2 --also-calmar \
                        2>&1 | tee chronos2_calibration_v6_swa.log || true
                fi

                python benchmark_chronos2.py \
                    --symbols AAPL SPY GOOG TSLA META NVDA MSFT AMZN \
                    --model-id "$V6_CKPT" \
                    --context-length 1024 \
                    --batch-size 64 \
                    --update-hyperparams \
                    2>&1 | tee chronos2_benchmark_v6.log
            fi

            echo "[$(date -u +%H:%M:%SZ)] Launching v7 (ctx=1024, gap_inject=0.15, lr=3e-5)..."
            bash scripts/launch_chronos2_v7.sh
            echo "[$(date -u +%H:%M:%SZ)] v7 launched. PID: $(cat .chronos2_v7_pid 2>/dev/null)"

            # -----------------------------------------------------------------------
            # Wait for v7 then calibrate and launch v8
            # -----------------------------------------------------------------------
            V7_PID="$(cat .chronos2_v7_pid 2>/dev/null)"
            if [[ -n "$V7_PID" ]]; then
                echo "[$(date -u +%H:%M:%SZ)] Waiting for v7 (PID $V7_PID) to complete before launching v8..."
                while kill -0 "$V7_PID" 2>/dev/null; do
                    sleep 180
                done
                echo "[$(date -u +%H:%M:%SZ)] v7 complete. Running v7 SWA + calibration + benchmark..."

                V7_CKPT="chronos2_finetuned/stocks_all_v7/finetuned-ckpt"
                V7_SWA="chronos2_finetuned/stocks_all_v7/swa-ckpt"
                if [[ -d "chronos2_finetuned/stocks_all_v7/trainer_workspace" ]]; then
                    python scripts/average_checkpoints.py \
                        --trainer-workspace chronos2_finetuned/stocks_all_v7/trainer_workspace \
                        --output "$V7_SWA" --n-last 3 \
                        --copy-chronos-config "$V7_CKPT" 2>&1 | tee chronos2_swa_v7.log || true
                fi

                if [[ -d "$V7_CKPT" ]]; then
                    python chronos2_linear_calibration.py \
                        --model-id     "$V7_CKPT" \
                        --cal-data-dir trainingdata \
                        --output-path  "$V7_CKPT/calibration.json" \
                        --max-shift-bps 20 \
                        --min-gap-bps   2 \
                        --grid-steps   25 \
                        --cal-bars     120 \
                        --max-windows  5000 \
                        --batch-size   32 \
                        --per-symbol \
                        --prediction-length 2 \
                        --also-calmar \
                        --hyperparams-dir hyperparams/chronos2_v7 \
                        2>&1 | tee chronos2_calibration_v7.log

                    if [[ -d "${V7_SWA:-}" ]]; then
                        python chronos2_linear_calibration.py \
                            --model-id "$V7_SWA" --cal-data-dir trainingdata \
                            --output-path "$V7_SWA/calibration.json" \
                            --max-shift-bps 20 --min-gap-bps 2 --grid-steps 25 \
                            --cal-bars 120 --max-windows 5000 --batch-size 32 \
                            --prediction-length 2 --also-calmar \
                            2>&1 | tee chronos2_calibration_v7_swa.log || true
                    fi

                    python benchmark_chronos2.py \
                        --symbols AAPL SPY GOOG TSLA META NVDA MSFT AMZN \
                        --model-id "$V7_CKPT" \
                        --context-length 1024 \
                        --batch-size 64 \
                        --update-hyperparams \
                        2>&1 | tee chronos2_benchmark_v7.log
                fi

                echo "[$(date -u +%H:%M:%SZ)] Launching v8 (ctx=1024, trend_inject=0.15, all augs)..."
                bash scripts/launch_chronos2_v8.sh
                echo "[$(date -u +%H:%M:%SZ)] v8 launched. PID: $(cat .chronos2_v8_pid 2>/dev/null)"

                # -----------------------------------------------------------------------
                # Wait for v8 then calibrate and launch v9
                # -----------------------------------------------------------------------
                V8_PID="$(cat .chronos2_v8_pid 2>/dev/null)"
                if [[ -n "$V8_PID" ]]; then
                    echo "[$(date -u +%H:%M:%SZ)] Waiting for v8 (PID $V8_PID) to complete before launching v9..."
                    while kill -0 "$V8_PID" 2>/dev/null; do
                        sleep 180
                    done
                    echo "[$(date -u +%H:%M:%SZ)] v8 complete. Running v8 SWA + calibration + benchmark..."

                    V8_CKPT="chronos2_finetuned/stocks_all_v8/finetuned-ckpt"
                    V8_SWA="chronos2_finetuned/stocks_all_v8/swa-ckpt"
                    if [[ -d "chronos2_finetuned/stocks_all_v8/trainer_workspace" ]]; then
                        python scripts/average_checkpoints.py \
                            --trainer-workspace chronos2_finetuned/stocks_all_v8/trainer_workspace \
                            --output "$V8_SWA" --n-last 3 \
                            --copy-chronos-config "$V8_CKPT" 2>&1 | tee chronos2_swa_v8.log || true
                    fi

                    if [[ -d "$V8_CKPT" ]]; then
                        python chronos2_linear_calibration.py \
                            --model-id     "$V8_CKPT" \
                            --cal-data-dir trainingdata \
                            --output-path  "$V8_CKPT/calibration.json" \
                            --max-shift-bps 20 \
                            --min-gap-bps   2 \
                            --grid-steps   25 \
                            --cal-bars     120 \
                            --max-windows  5000 \
                            --batch-size   32 \
                            --per-symbol \
                            --prediction-length 2 \
                            --also-calmar \
                            --hyperparams-dir hyperparams/chronos2_v8 \
                            2>&1 | tee chronos2_calibration_v8.log

                        if [[ -d "${V8_SWA:-}" ]]; then
                            python chronos2_linear_calibration.py \
                                --model-id "$V8_SWA" --cal-data-dir trainingdata \
                                --output-path "$V8_SWA/calibration.json" \
                                --max-shift-bps 20 --min-gap-bps 2 --grid-steps 25 \
                                --cal-bars 120 --max-windows 5000 --batch-size 32 \
                                --prediction-length 2 --also-calmar \
                                2>&1 | tee chronos2_calibration_v8_swa.log || true
                        fi

                        python benchmark_chronos2.py \
                            --symbols AAPL SPY GOOG TSLA META NVDA MSFT AMZN \
                            --model-id "$V8_CKPT" \
                            --context-length 1024 \
                            --batch-size 64 \
                            --update-hyperparams \
                            2>&1 | tee chronos2_benchmark_v8.log
                    fi

                    echo "[$(date -u +%H:%M:%SZ)] Launching v9 (ctx=1024, vol_regime=0.15, mean_reversion=0.10)..."
                    bash scripts/launch_chronos2_v9.sh
                    echo "[$(date -u +%H:%M:%SZ)] v9 launched. PID: $(cat .chronos2_v9_pid 2>/dev/null)"

                    # -----------------------------------------------------------------------
                    # Wait for v9 then calibrate
                    # -----------------------------------------------------------------------
                    V9_PID="$(cat .chronos2_v9_pid 2>/dev/null)"
                    if [[ -n "$V9_PID" ]]; then
                        echo "[$(date -u +%H:%M:%SZ)] Waiting for v9 (PID $V9_PID) to complete..."
                        while kill -0 "$V9_PID" 2>/dev/null; do
                            sleep 180
                        done
                        echo "[$(date -u +%H:%M:%SZ)] v9 complete. Running v9 SWA + calibration + benchmark..."

                        V9_CKPT="chronos2_finetuned/stocks_all_v9/finetuned-ckpt"
                        V9_SWA="chronos2_finetuned/stocks_all_v9/swa-ckpt"
                        if [[ -d "chronos2_finetuned/stocks_all_v9/trainer_workspace" ]]; then
                            python scripts/average_checkpoints.py \
                                --trainer-workspace chronos2_finetuned/stocks_all_v9/trainer_workspace \
                                --output "$V9_SWA" --n-last 3 \
                                --copy-chronos-config "$V9_CKPT" 2>&1 | tee chronos2_swa_v9.log || true
                        fi

                        if [[ -d "$V9_CKPT" ]]; then
                            python chronos2_linear_calibration.py \
                                --model-id     "$V9_CKPT" \
                                --cal-data-dir trainingdata \
                                --output-path  "$V9_CKPT/calibration.json" \
                                --max-shift-bps 20 \
                                --min-gap-bps   2 \
                                --grid-steps   25 \
                                --cal-bars     120 \
                                --max-windows  5000 \
                                --batch-size   32 \
                                --per-symbol \
                                --prediction-length 2 \
                                --also-calmar \
                                --hyperparams-dir hyperparams/chronos2_v9 \
                                2>&1 | tee chronos2_calibration_v9.log

                            if [[ -d "${V9_SWA:-}" ]]; then
                                python chronos2_linear_calibration.py \
                                    --model-id "$V9_SWA" --cal-data-dir trainingdata \
                                    --output-path "$V9_SWA/calibration.json" \
                                    --max-shift-bps 20 --min-gap-bps 2 --grid-steps 25 \
                                    --cal-bars 120 --max-windows 5000 --batch-size 32 \
                                    --prediction-length 2 --also-calmar \
                                    2>&1 | tee chronos2_calibration_v9_swa.log || true
                            fi

                            python benchmark_chronos2.py \
                                --symbols AAPL SPY GOOG TSLA META NVDA MSFT AMZN \
                                --model-id "$V9_CKPT" \
                                --context-length 1024 \
                                --batch-size 64 \
                                --update-hyperparams \
                                2>&1 | tee chronos2_benchmark_v9.log
                        fi

                        echo "[$(date -u +%H:%M:%SZ)] Launching v10 (ctx=1024, earnings_shock=0.10)..."
                        bash scripts/launch_chronos2_v10.sh
                        echo "[$(date -u +%H:%M:%SZ)] v10 launched. PID: $(cat .chronos2_v10_pid 2>/dev/null)"

                        # -------------------------------------------------------------------
                        # Wait for v10 then calibrate
                        # -------------------------------------------------------------------
                        V10_PID="$(cat .chronos2_v10_pid 2>/dev/null)"
                        if [[ -n "$V10_PID" ]]; then
                            echo "[$(date -u +%H:%M:%SZ)] Waiting for v10 (PID $V10_PID) to complete..."
                            while kill -0 "$V10_PID" 2>/dev/null; do
                                sleep 180
                            done
                            echo "[$(date -u +%H:%M:%SZ)] v10 complete. Running v10 SWA + calibration + benchmark..."

                            V10_CKPT="chronos2_finetuned/stocks_all_v10/finetuned-ckpt"
                            V10_SWA="chronos2_finetuned/stocks_all_v10/swa-ckpt"
                            if [[ -d "chronos2_finetuned/stocks_all_v10/trainer_workspace" ]]; then
                                python scripts/average_checkpoints.py \
                                    --trainer-workspace chronos2_finetuned/stocks_all_v10/trainer_workspace \
                                    --output "$V10_SWA" --n-last 3 \
                                    --copy-chronos-config "$V10_CKPT" 2>&1 | tee chronos2_swa_v10.log || true
                            fi

                            if [[ -d "$V10_CKPT" ]]; then
                                python chronos2_linear_calibration.py \
                                    --model-id     "$V10_CKPT" \
                                    --cal-data-dir trainingdata \
                                    --output-path  "$V10_CKPT/calibration.json" \
                                    --max-shift-bps 20 \
                                    --min-gap-bps   2 \
                                    --grid-steps   25 \
                                    --cal-bars     120 \
                                    --max-windows  5000 \
                                    --batch-size   32 \
                                    --per-symbol \
                                    --prediction-length 2 \
                                    --also-calmar \
                                    --hyperparams-dir hyperparams/chronos2_v10 \
                                    2>&1 | tee chronos2_calibration_v10.log

                                if [[ -d "${V10_SWA:-}" ]]; then
                                    python chronos2_linear_calibration.py \
                                        --model-id "$V10_SWA" --cal-data-dir trainingdata \
                                        --output-path "$V10_SWA/calibration.json" \
                                        --max-shift-bps 20 --min-gap-bps 2 --grid-steps 25 \
                                        --cal-bars 120 --max-windows 5000 --batch-size 32 \
                                        --prediction-length 2 --also-calmar \
                                        2>&1 | tee chronos2_calibration_v10_swa.log || true
                                fi

                                python benchmark_chronos2.py \
                                    --symbols AAPL SPY GOOG TSLA META NVDA MSFT AMZN \
                                    --model-id "$V10_CKPT" \
                                    --context-length 1024 \
                                    --batch-size 64 \
                                    --update-hyperparams \
                                    2>&1 | tee chronos2_benchmark_v10.log
                            fi
                        fi
                    fi
                fi
            fi
        fi
    fi
fi
