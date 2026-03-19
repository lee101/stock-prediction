#!/bin/bash
# Download daily OHLCV data for 50 additional liquid stocks not yet in trainingdata/train/.
# Uses fetch_external_data.py to download from yfinance, then converts to the
# trainingdata/train/ format (timestamp,open,high,low,close,volume,trade_count,vwap,symbol).
set -e

cd /nvme0n1-disk/code/stock-prediction
source /nvme0n1-disk/code/stock-prediction/.venv313/bin/activate

# 50 liquid, well-known stocks across diverse sectors not already in training data.
# Sectors: tech/AI, consumer, biotech, fintech, industrials, energy, mining,
#          international ADRs, EV, genomics, aerospace/defense, software.
SYMBOLS=(
    RIVN PATH BILL HUBS DKNG APP DUOL TOST          # Tech / consumer / gaming
    JD PDD BIDU NIO XPEV LI GRAB NU                 # International ADRs
    CRSP ARGX ALNY INSM EXAS NTRA HALO MEDP         # Biotech / healthcare
    ARM CRDO IONQ SOUN RKLB MSTR                     # Semis / quantum / crypto / space
    CAVA ELF ONON WING SHAK CHWY CVNA               # Consumer / retail
    XPO SAIA IBKR HEI                                # Industrials / finance
    FNV WPM AEM KGC                                  # Mining / precious metals
    HIMS GH CART W                                    # Health / e-commerce
    GTLB FOUR GLOB                                    # Software / services
)

TRAIN_DIR="trainingdata/train"
EXTERNAL_DIR="externaldata/yahoo"
DOWNLOADED=0
SKIPPED=0
FAILED=0

for SYM in "${SYMBOLS[@]}"; do
    if [ -f "$TRAIN_DIR/$SYM.csv" ]; then
        echo "SKIP $SYM (already exists in $TRAIN_DIR)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "Downloading $SYM..."
    python fetch_external_data.py --symbols "$SYM" --start 2020-01-01 --out "$EXTERNAL_DIR" || {
        echo "FAILED to download: $SYM"
        FAILED=$((FAILED + 1))
        continue
    }

    # Convert from externaldata format (date,open,...) to trainingdata format
    # (timestamp,open,high,low,close,volume,trade_count,vwap,symbol)
    if [ -f "$EXTERNAL_DIR/$SYM.csv" ]; then
        python3 -c "
import pandas as pd, sys
sym = '$SYM'
df = pd.read_csv('$EXTERNAL_DIR/$SYM.csv')
# Standardize column names
df.columns = [c.lower().strip() for c in df.columns]
if 'date' not in df.columns:
    print(f'ERROR: no date column in {sym}', file=sys.stderr)
    sys.exit(1)
# Build output in trainingdata/train format
out = pd.DataFrame()
out['timestamp'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d 05:00:00+00:00')
out['open'] = df['open']
out['high'] = df['high']
out['low'] = df['low']
out['close'] = df['close']
out['volume'] = df['volume']
out['trade_count'] = 0  # not available from yfinance
out['vwap'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0  # approx
out['symbol'] = sym
out = out.dropna(subset=['open','high','low','close','volume'])
out = out.sort_values('timestamp')
out.to_csv('$TRAIN_DIR/$SYM.csv', index=False)
print(f'  Converted {len(out)} rows -> $TRAIN_DIR/$SYM.csv')
" || {
            echo "FAILED to convert: $SYM"
            FAILED=$((FAILED + 1))
            continue
        }
        DOWNLOADED=$((DOWNLOADED + 1))
    else
        echo "FAILED: $SYM download produced no file"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== Summary ==="
echo "Downloaded: $DOWNLOADED"
echo "Skipped (already exist): $SKIPPED"
echo "Failed: $FAILED"
echo "Total symbols attempted: ${#SYMBOLS[@]}"
