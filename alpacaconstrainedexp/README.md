# Alpaca Constrained Cross-Learning (Chronos2 + Global Policy)

This experiment trains a multi-symbol Chronos2 model and a global trading policy with **strict long/short constraints**:

- **Longable:** crypto + tech stocks (`NVDA`, `GOOG`, `MSFT` by default)
- **Shortable:** `YELP`, `EBAY`, `TRIP`, `MTCH`, `KIND`, `ANGI`, `Z`, `EXPE`, `BKNG`, `NWSA`, `NYT`

The selector enforces:
- Long entries only for longable symbols.
- Short entries only for the shortable list (crypto never shorted).

## 0) Refresh hourly data

```bash
source .venv/bin/activate

python -m alpacaconstrainedexp.refresh_hourly_data \
  --crypto BTCUSD,ETHUSD,SOLUSD \
  --long-stocks NVDA,GOOG,MSFT \
  --short-stocks YELP,EBAY,TRIP,MTCH,KIND,ANGI,Z,EXPE,BKNG,NWSA,NYT
```

Defaults write into `trainingdatahourly/` (stocks + crypto). If you need a different root:

```bash
python -m alpacaconstrainedexp.refresh_hourly_data --data-root trainingdata
```

## 1) Rank symbol groups by MAE (Chronos2 base)

```bash
python -m alpacaconstrainedexp.select_symbol_groups \
  --context-length 512 \
  --val-hours 168 \
  --top-short 5
```

This writes a JSON report under `alpacaconstrainedexp/outputs/` with suggested long/short groups.

## 2) Fine-tune Chronos2 across constrained symbols

```bash
python -m alpacaconstrainedexp.chronos_finetune_multi \
  --prediction-length 1 \
  --context-length 1024 \
  --batch-size 64 \
  --learning-rate 1e-5 \
  --num-steps 1000 \
  --finetune-mode lora \
  --preaug-strategy percent_change
```

## 3) Build forecast caches

```bash
python -m alpacaconstrainedexp.build_forecasts \
  --finetuned-model alpacaconstrainedexp/chronos_finetuned/<RUN_NAME>/finetuned \
  --forecast-cache-root alpacaconstrainedexp/forecast_cache \
  --horizons 1,24 \
  --context-hours 336 \
  --predict-batches-jointly \
  --lookback-hours 720
```

## 4) Train global policy

```bash
python -m alpacaconstrainedexp.train_global_policy \
  --epochs 4 \
  --sequence-length 96 \
  --forecast-horizons 1,24 \
  --forecast-cache-root alpacaconstrainedexp/forecast_cache \
  --cache-only \
  --allow-mixed-asset \
  --moving-average-windows 168,336 \
  --min-history-hours 200
```

## 5) Constrained selector (10d sim)

```bash
python -m alpacaconstrainedexp.run_global_selector \
  --checkpoint binanceneural/checkpoints/<RUN_NAME>/epoch_XXX.pt \
  --forecast-horizons 1,24 \
  --forecast-cache-root alpacaconstrainedexp/forecast_cache \
  --moving-average-windows 168,336 \
  --min-history-hours 200 \
  --eval-days 10
```

## 6) Selector sweep

```bash
python -m alpacaconstrainedexp.run_selector_sweep \
  --checkpoint binanceneural/checkpoints/<RUN_NAME>/epoch_XXX.pt \
  --forecast-horizons 1,24 \
  --forecast-cache-root alpacaconstrainedexp/forecast_cache \
  --moving-average-windows 168,336 \
  --min-history-hours 200 \
  --min-edge-list 0.0,0.0005,0.001 \
  --risk-weight-list 0.25,0.5,0.75 \
  --dip-threshold-list 0.0,0.005,0.01 \
  --eval-days 10
```

## Notes

- GPU required for Chronos2 fine-tuning + forecast generation + policy inference.
- Use `--symbols` to override the default constrained universe; constraints still apply unless you override `--long-stocks` / `--short-stocks`.
