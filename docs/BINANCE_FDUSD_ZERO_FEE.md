# Binance FDUSD Zero-Fee Data (Hourly)

This repo’s Binance experiments historically used `*USDT` symbols (e.g. `SOLUSDT`).
If you want to trade the FDUSD quote pairs (often the “zero-fee” spot promos),
collect + train on the `*FDUSD` symbols instead.

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
- If you run on Binance.US or a restricted endpoint, some `*FDUSD` pairs may be
  unavailable; the downloader will report `unavailable`.
