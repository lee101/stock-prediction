# Binance Zero-Fee Stable Quotes (Hourly)

This repo’s Binance experiments historically used `*USDT` symbols (e.g. `SOLUSDT`).
If you want to trade the zero-fee stable-quote pairs, collect + train on the
`*FDUSD` and/or `*U` symbols instead.

## 1) Download Hourly Spot Data

Default “zero-fee” set (requested):

```bash
source .venv/bin/activate
python scripts/collect_binance_hourly_zero_fee_pairs.py
```

Outputs CSVs under `binancetrainingdatahourly/` (and will create a best-effort
alias symlink for `trainingdatahourlybinance/` when using the default names).

To download the larger curated FDUSD set:

```bash
python scripts/collect_binance_hourly_zero_fee_pairs.py --all-fdusd
```

To download the curated U set:

```bash
python scripts/collect_binance_hourly_zero_fee_pairs.py --all-u
```

Notes:
- In restricted regions (Binance REST API returns HTTP 451), the downloader falls back to
  Binance Vision’s public datasets for symbols that are missing from Binance.US (e.g. FDUSD/U pairs).
  Binance Vision is typically delayed for the current UTC day (daily zips appear after the day completes).

## 2) Train / Run Experiments On FDUSD Symbols

You can point existing pipelines at the FDUSD symbols directly, for example:

```bash
# Single-symbol experiment (example)
python -m binancechronossolexperiment.run_experiment \
  --symbol SOLFDUSD \
  --data-root binancetrainingdatahourly
```

```bash
# Cross-learning (example)
python -m binancecrosslearning.chronos_finetune_multi \
  --symbols BTCFDUSD,ETHFDUSD,SOLFDUSD,BNBFDUSD \
  --data-root binancetrainingdatahourly
```

Notes:
- The stablecoin conversion pair is `FDUSDUSDT` (downloaded as `FDUSD/USDT`) for
  manual USDT -> FDUSD moves.
  - Helper script (dry-run by default): `python scripts/convert_binance_usdt_to_fdusd.py`
- The U conversion pair is `UUSDT` (downloaded as `U/USDT`) for manual USDT -> U moves.
  - Helper script (dry-run by default): `python scripts/convert_binance_usdt_to_u.py`
- If you run on Binance.US or a restricted endpoint, the downloader will fall back to Binance Vision
  for `*FDUSD` / `*U` pairs. If a symbol is missing from both Binance.US and Vision, it will be reported
  as `no_data`/`unavailable` in the download summary.
