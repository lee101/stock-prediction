#!/usr/bin/env bash
# Export stocks40 daily dataset: original stocks15 + 25 new liquid stocks.
# Uses union mode so symbols with shorter histories get tradable=0 for missing days.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv - try local first, then fall back to main repo
if [ -f .venv313/bin/activate ]; then
  source .venv313/bin/activate
elif [ -f /nvme0n1-disk/code/stock-prediction/.venv313/bin/activate ]; then
  source /nvme0n1-disk/code/stock-prediction/.venv313/bin/activate
else
  echo "ERROR: .venv313 not found" >&2
  exit 1
fi

# Resolve data root (worktrees may not have trainingdata/)
if [ -d "$SCRIPT_DIR/trainingdata/train" ]; then
  DATA_ROOT="$SCRIPT_DIR/trainingdata/train"
elif [ -d /nvme0n1-disk/code/stock-prediction/trainingdata/train ]; then
  DATA_ROOT=/nvme0n1-disk/code/stock-prediction/trainingdata/train
else
  echo "ERROR: trainingdata/train not found" >&2
  exit 1
fi

SYMBOLS="AAPL,MSFT,NVDA,GOOG,AMZN,META,TSLA,PLTR,NET,NFLX,AMD,ADBE,CRM,PYPL,INTC"
SYMBOLS="${SYMBOLS},JPM,V,MA,BAC,GS,UNH,JNJ,LLY,COST,WMT"
SYMBOLS="${SYMBOLS},HD,PG,KO,MCD,DIS,XOM,CVX,BA,CAT,GE"
SYMBOLS="${SYMBOLS},HON,UPS,UBER,SQ,CRWD"

echo "=== Exporting stocks40 daily TRAIN (up to 2025-06-01) ==="
python export_data_daily.py \
  --symbols "$SYMBOLS" \
  --data-root "$DATA_ROOT" \
  --output pufferlib_market/data/stocks40_daily_train.bin \
  --end-date 2025-06-01 \
  --union \
  --min-days 200

echo ""
echo "=== Exporting stocks40 daily VAL (2025-06-01 onwards) ==="
python export_data_daily.py \
  --symbols "$SYMBOLS" \
  --data-root "$DATA_ROOT" \
  --output pufferlib_market/data/stocks40_daily_val.bin \
  --start-date 2025-06-01 \
  --union \
  --min-days 30

echo ""
echo "=== Verifying MKTD headers ==="
python -c "
import os
for path in ['pufferlib_market/data/stocks40_daily_train.bin',
             'pufferlib_market/data/stocks40_daily_val.bin']:
    with open(path, 'rb') as f:
        hdr = f.read(64)
        magic = hdr[:4]
        ver = int.from_bytes(hdr[4:8], 'little')
        nsym = int.from_bytes(hdr[8:12], 'little')
        nts = int.from_bytes(hdr[12:16], 'little')
        nfeat = int.from_bytes(hdr[16:20], 'little')
        nprice = int.from_bytes(hdr[20:24], 'little')
        syms = []
        for i in range(nsym):
            raw = f.read(16)
            syms.append(raw.split(b'\x00', 1)[0].decode('ascii'))
    print(f'{path}:')
    print(f'  magic={magic} version={ver} symbols={nsym} timesteps={nts} features={nfeat} prices={nprice}')
    print(f'  symbols: {syms}')
    fsize = os.path.getsize(path)
    expected = 64 + nsym*16 + nts*nsym*nfeat*4 + nts*nsym*nprice*4 + nts*nsym
    print(f'  file_size={fsize} expected={expected} match={fsize==expected}')
    print()
"

echo "=== Done ==="
