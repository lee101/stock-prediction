# Hourly Crypto Training (midpoint-offset decoder, Nov 2025)

## Environment
```bash
source .venv313/bin/activate
# Training uses Alpaca data & Chronos forecasts; PAPER flag is irrelevant for training.
```

## Run a single-symbol train
```bash
python hourlycrypto/trade_stock_crypto_hourly.py \
  --mode train \
  --symbol UNIUSD \
  --training-symbols UNIUSD \
  --sequence-length 256 \
  --batch-size 32 \
  --epochs 20 \
  --price-offset-pct 0.0003 \
  --price-offset-span-multiplier 0.15 \
  --price-offset-max-pct 0.003 \
  --no-compile \
  --use-amp \
  --checkpoint-root hourlycryptotraining/checkpoints_256ctx_UNI_midoffset \
  --log-level INFO
```

Notes:
- Decoder now defaults to midpoint offsets (`use_midpoint_offsets=True`), so buy ≤ mid ≤ sell before the 3bp gap is applied.
- Min notional and TP floor logic are in `hourlycrypto/trade_stock_crypto_hourly.py` when spawning watchers.
- Forecast cache is refreshed automatically during training; ensure `trainingdatahourly/crypto/<SYMBOL>.csv` is gap-free.

## Logs / checkpoints
- Training logs: `training_results/hourlycrypto_<SYMBOL>_train_midoffset.log`
- Checkpoints: `hourlycryptotraining/checkpoints_256ctx_<SYMBOL>_midoffset/`

## Current retrain batch (started)
- UNIUSD: epochs=20, seq=256, bsz=32 → checkpoints_256ctx_UNI_midoffset
- BTCUSD: epochs=20, seq=256, bsz=32 → checkpoints_256ctx_BTC_midoffset
- ETHUSD: epochs=20, seq=256, bsz=32 → checkpoints_256ctx_ETH_midoffset
- SOLUSD: epochs=20, seq=256, bsz=32 → checkpoints_256ctx_SOL_midoffset

Inspect progress:
```bash
tail -f training_results/hourlycrypto_UNIUSD_train_midoffset.log
```
