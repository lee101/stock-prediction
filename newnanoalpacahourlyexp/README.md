# Alpaca Hourly Chronos2 Experiments

This folder mirrors the Binance hourly experiments but targets Alpaca hourly bars (stocks + crypto). It:

- Uses Chronos2 OHLC forecasts as features.
- Applies Alpaca fee assumptions via `src/fees.get_fee_for_symbol` (TRADING_FEE for stocks, CRYPTO_TRADING_FEE for crypto).
- Enforces market hours + end-of-day flatting for stocks in the simulator (configurable).

## Data

Hourly CSVs are expected under:

- `trainingdatahourly/crypto/{SYMBOL}.csv`
- `trainingdatahourly/stocks/{SYMBOL}.csv`

`DatasetConfig` auto-selects the root based on symbol (override with `--data-root`).

## Train + Evaluate (single symbol)

```bash
python newnanoalpacahourlyexp/run_experiment.py \
  --symbol SOLUSD \
  --epochs 5 \
  --sequence-length 96 \
  --aggregate \
  --eval-days 10 \
  --output-dir newnanoalpacahourlyexp/outputs/solusd_run
```

Stocks (market hours + EOD close enforced by default):

```bash
python newnanoalpacahourlyexp/run_experiment.py \
  --symbol NVDA \
  --epochs 5 \
  --sequence-length 96 \
  --aggregate \
  --eval-days 10 \
  --output-dir newnanoalpacahourlyexp/outputs/nvda_run
```

Multi-symbol training (single checkpoint across symbols):

```bash
python newnanoalpacahourlyexp/run_experiment.py \
  --symbol SOLUSD \
  --symbols SOLUSD,LINKUSD,ETHUSD \
  --allow-mixed-asset \
  --epochs 5 \
  --sequence-length 96 \
  --aggregate \
  --eval-days 10 \
  --output-dir newnanoalpacahourlyexp/outputs/sol_link_eth_combo
```

## Multi-asset best-trade selector

```bash
python newnanoalpacahourlyexp/run_multiasset_selector.py \
  --symbols BTCUSD,ETHUSD,LINKUSD \
  --checkpoints "BTCUSD=path/to/btc.pt,ETHUSD=path/to/eth.pt,LINKUSD=path/to/link.pt" \
  --eval-days 10 \
  --output-dir newnanoalpacahourlyexp/outputs/selector_crypto
```

Mixed crypto + stocks (use data roots per asset class):

```bash
python newnanoalpacahourlyexp/run_multiasset_selector.py \
  --symbols BTCUSD,NVDA,UNIUSD \
  --checkpoints "BTCUSD=... ,NVDA=...,UNIUSD=..." \
  --crypto-data-root trainingdatahourly/crypto \
  --stock-data-root trainingdatahourly/stocks
```

## Action scaling sweep

```bash
python newnanoalpacahourlyexp/sweep.py \
  --symbol SOLUSD \
  --checkpoint path/to/checkpoint.pt \
  --aggregate \
  --intensity 0.6 0.8 1.0 1.2 \
  --offset 0.0 0.0002 0.0005
```

## Notes

- `--maker-fee` and `--periods-per-year` let you override fee/annualization assumptions per run.
- `--no-enforce-market-hours` or `--no-close-at-eod` can be used to disable stock-only guards.

## Suggested artifact sync (R2)

```bash
aws s3 sync newnanoalpacahourlyexp/outputs/ s3://models/stock/models/alpaca/hourly/outputs/ --endpoint-url "$R2_ENDPOINT"
```
