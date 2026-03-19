#!/usr/bin/env bash
# Download daily OHLCV for high-volume stocks and export to MKTD binary.
#
# Usage:
#   bash download_highvol_stocks.sh
#
# Symbols: SOFI MU PLUG AAL F HIMS PATH RKLB MARA TTD NOK PFE RCAT NBIS ABEV ITUB OWL RIG
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
MAIN_REPO="/nvme0n1-disk/code/stock-prediction"
TRAIN_DIR="${REPO_ROOT}/trainingdata/train"
ALL_SYMBOLS=(SOFI MU PLUG AAL F HIMS PATH RKLB MARA TTD NOK PFE RCAT NBIS ABEV ITUB OWL RIG)
MISSING_SYMBOLS=()

source "${MAIN_REPO}/.venv313/bin/activate"
mkdir -p "${TRAIN_DIR}"

# ------------------------------------------------------------------
# Step 1: Copy symbols that already exist in the main repo
# ------------------------------------------------------------------
echo "=== Step 1: Copy existing CSVs from main repo ==="
for sym in "${ALL_SYMBOLS[@]}"; do
    src="${MAIN_REPO}/trainingdata/train/${sym}.csv"
    dst="${TRAIN_DIR}/${sym}.csv"
    if [[ -f "$dst" ]]; then
        echo "  ${sym}: already present in worktree"
    elif [[ -f "$src" ]]; then
        cp "$src" "$dst"
        echo "  ${sym}: copied from main repo"
    else
        MISSING_SYMBOLS+=("$sym")
        echo "  ${sym}: MISSING -- will download"
    fi
done

# ------------------------------------------------------------------
# Step 2: Download missing symbols via fetch_external_data.py
# ------------------------------------------------------------------
if [[ ${#MISSING_SYMBOLS[@]} -gt 0 ]]; then
    echo ""
    echo "=== Step 2: Downloading ${#MISSING_SYMBOLS[@]} missing symbols ==="
    python3 "${REPO_ROOT}/fetch_external_data.py" \
        --symbols "${MISSING_SYMBOLS[@]}" \
        --start 2015-01-01 \
        --out "${TRAIN_DIR}"
else
    echo ""
    echo "=== Step 2: All symbols already present, no downloads needed ==="
fi

# ------------------------------------------------------------------
# Step 3: Verify all CSVs exist
# ------------------------------------------------------------------
echo ""
echo "=== Step 3: Verify CSVs ==="
OK_SYMBOLS=()
for sym in "${ALL_SYMBOLS[@]}"; do
    csv="${TRAIN_DIR}/${sym}.csv"
    if [[ -f "$csv" ]]; then
        rows=$(wc -l < "$csv")
        echo "  ${sym}: ${rows} lines"
        OK_SYMBOLS+=("$sym")
    else
        echo "  ${sym}: MISSING (download may have failed)"
    fi
done

if [[ ${#OK_SYMBOLS[@]} -eq 0 ]]; then
    echo "ERROR: No symbols available. Aborting."
    exit 1
fi

# ------------------------------------------------------------------
# Step 4: Export to MKTD binary
# ------------------------------------------------------------------
echo ""
echo "=== Step 4: Export to MKTD binary ==="
SYMBOLS_CSV=$(IFS=,; echo "${OK_SYMBOLS[*]}")
OUTPUT_BIN="${REPO_ROOT}/pufferlib_market/data/highvol_daily_train.bin"

python3 "${REPO_ROOT}/export_data_daily.py" \
    --symbols "${SYMBOLS_CSV}" \
    --data-root "${TRAIN_DIR}" \
    --output "${OUTPUT_BIN}" \
    --end-date 2025-06-01 \
    --union \
    --min-days 200

echo ""
echo "=== Done ==="
echo "Binary: ${OUTPUT_BIN}"
