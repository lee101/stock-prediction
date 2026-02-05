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

## Live trading (Alpaca)

Hourly production loop (GPU required):

```bash
python -m newnanoalpacahourlyexp.trade_alpaca_hourly \
  --symbols SOLUSD,LINKUSD,UNIUSD \
  --checkpoints "SOLUSD=binanceneural/checkpoints/alpaca_solusd_20260204_013044/epoch_006.pt,LINKUSD=binanceneural/checkpoints/alpaca_linkusd_20260204_111049/epoch_006.pt,UNIUSD=binanceneural/checkpoints/alpaca_uniusd_20260204_231926/epoch_004.pt" \
  --default-checkpoint binanceneural/checkpoints/alpaca_solusd_20260204_013044/epoch_006.pt \
  --sequence-length 96 \
  --horizon 1 \
  --forecast-horizons 1,24 \
  --forecast-horizons-map "UNIUSD=1" \
  --forecast-cache-root binanceneural/forecast_cache \
  --crypto-data-root trainingdatahourly/crypto \
  --stock-data-root trainingdatahourly/stocks \
  --allocation-pct 0.05 \
  --intensity-scale 1.0
```

Best-trade selector (single position across symbols):

```bash
python -m newnanoalpacahourlyexp.trade_alpaca_selector \
  --symbols SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX \
  --checkpoint binanceneural/checkpoints/alpaca_cross_global_mixed7_robust_short_seq128_20260205_043448/epoch_003.pt \
  --sequence-length 128 \
  --forecast-horizons 1,24 \
  --forecast-cache-root alpacanewccrosslearning/forecast_cache/mixed7_live \
  --crypto-data-root trainingdatahourly/crypto \
  --stock-data-root trainingdatahourly/stocks \
  --allocation-pct 1.0 \
  --intensity-scale 2.0 \
  --min-edge 0.004 \
  --risk-weight 0.2 \
  --edge-mode high_low \
  --dip-threshold-pct 0.005 \
  --moving-average-windows 24,72 \
  --ema-windows 24,72 \
  --atr-windows 24,72 \
  --trend-windows 72 \
  --drawdown-windows 72 \
  --volume-z-window 72 \
  --volume-shock-window 24 \
  --vol-regime-short 24 \
  --vol-regime-long 72 \
  --min-history-hours 50 \
  --close-at-eod
```

Exit-only guard (NFLX):

```bash
python -m newnanoalpacahourlyexp.exit_only_alpaca \
  --symbols NFLX \
  --poll-seconds 300
```

Systemd helpers (live):

```bash
sudo bash scripts/setup_systemd_alpaca_hourly.sh
sudo bash scripts/setup_systemd_alpaca_exit_nflx.sh
```

## Notes

- GPU is required for both training and inference. Use `--device cuda` (or `cuda:0`) if you need to pin a GPU.
- `--maker-fee` and `--periods-per-year` let you override fee/annualization assumptions per run.
- `--no-enforce-market-hours` or `--no-close-at-eod` can be used to disable stock-only guards.

## Suggested artifact sync (R2)

```bash
aws s3 sync newnanoalpacahourlyexp/outputs/ s3://models/stock/models/alpaca/hourly/outputs/ --endpoint-url "$R2_ENDPOINT"
```
