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
    --max-windows  5000 \
    --batch-size   32 \
    --per-symbol \
    --hyperparams-dir hyperparams/chronos2_v3 \
    2>&1 | tee chronos2_calibration_v3.log

# -----------------------------------------------------------------------
# Step 3: Benchmark v3 vs v2 on key symbols (same-period comparison)
# -----------------------------------------------------------------------
echo "[$(date -u +%H:%M:%SZ)] Benchmarking v3 on key symbols..."
python benchmark_chronos2.py \
    --symbols AAPL SPY GOOG TSLA META NVDA MSFT AMZN \
    --model-id "$V3_CKPT" \
    --context-length 512 \
    --batch-size 128 \
    --update-hyperparams \
    2>&1 | tee chronos2_benchmark_v3.log

echo "[$(date -u +%H:%M:%SZ)] Benchmarking v2 for same-period comparison..."
python benchmark_chronos2.py \
    --symbols AAPL SPY GOOG TSLA META NVDA MSFT AMZN \
    --model-id "chronos2_finetuned/stocks_all_v2/finetuned-ckpt" \
    --context-length 512 \
    --batch-size 128 \
    2>&1 | tee chronos2_benchmark_v2_fresh.log

# Print quick comparison
echo "[$(date -u +%H:%M:%SZ)] === v3 vs v2 comparison ==="
python3 -c "
import json, glob
syms = ['AAPL', 'SPY', 'GOOG', 'TSLA', 'META', 'NVDA', 'MSFT', 'AMZN']
print(f'{\"SYM\":6s} {\"V3_TEST\":>10s} {\"V2_TEST\":>10s} {\"DELTA\":>10s}')
for sym in syms:
    # Find most recent v3 result
    v3_files = sorted(glob.glob(f'chronos2_benchmarks/{sym}/*bench*.json'))
    v2_files = sorted(glob.glob(f'chronos2_benchmarks/{sym}/*fresh*.json'))
    # fallback: just last two files
    if len(v3_files) >= 2:
        v3 = json.load(open(v3_files[-1]))
        v2 = json.load(open(v3_files[-2]))
    else:
        print(f'{sym}: insufficient files')
        continue
    if isinstance(v3, list): v3 = v3[-1]
    if isinstance(v2, list): v2 = v2[-1]
    v3t = v3.get('test', v3.get('validation', {})).get('pct_return_mae', 0) * 100
    v2t = v2.get('test', v2.get('validation', {})).get('pct_return_mae', 0) * 100
    delta = v2t - v3t
    print(f'{sym:6s} {v3t:10.3f}% {v2t:10.3f}% {delta:+10.3f}%')
" || true

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
fi
