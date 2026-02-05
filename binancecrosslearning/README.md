# Binance Cross-Learning (Chronos2 + Global Policy)

This experiment trains a **multi-symbol Chronos2 model** (cross-learning) and then
trains a **single trading policy** across symbols. The goal is a global strategy
that can allocate to the best opportunities each hour.

Default symbols target major long-term pairs (USDT):
`BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,LINKUSDT,ADAUSDT,APTUSDT,AVAXUSDT`.

## 0) (Optional) Refresh/download Binance hourly data

```bash
source .venv/bin/activate

python -m binance_data_wrapper \
  --pairs BTCUSDT ETHUSDT SOLUSDT BNBUSDT LINKUSDT ADAUSDT APTUSDT AVAXUSDT \
  --output-dir trainingdatahourlybinance \
  --history-years 5
```

## 1) Fine-tune Chronos2 across symbols

```bash
python -m binancecrosslearning.chronos_finetune_multi \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,LINKUSDT,ADAUSDT,APTUSDT,AVAXUSDT \
  --prediction-length 1 \
  --context-length 1024 \
  --batch-size 64 \
  --learning-rate 1e-5 \
  --num-steps 1000 \
  --finetune-mode lora \
  --output-root binancecrosslearning/chronos_finetuned \
  --preaug-strategy percent_change
```

## 2) Build forecasts using the fine-tuned model

```bash
python -m binancecrosslearning.build_forecasts \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,LINKUSDT,ADAUSDT,APTUSDT,AVAXUSDT \
  --finetuned-model binancecrosslearning/chronos_finetuned/<RUN_NAME>/finetuned \
  --forecast-cache-root binancecrosslearning/forecast_cache \
  --horizons 1,4,24 \
  --context-hours 336 \
  --predict-batches-jointly \
  --lookback-hours 720
```

## 3) Train a global policy across symbols

```bash
python -m binancecrosslearning.train_global_policy \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,LINKUSDT,ADAUSDT,APTUSDT,AVAXUSDT \
  --target-symbol SOLUSDT \
  --epochs 6 \
  --sequence-length 96 \
  --forecast-horizons 1,4,24 \
  --forecast-cache-root binancecrosslearning/forecast_cache \
  --maker-fee 0.0 \
  --cache-only
```

## 4) Evaluate global selector (best trade per hour)

```bash
python -m binancecrosslearning.run_global_selector \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,LINKUSDT,ADAUSDT,APTUSDT,AVAXUSDT \
  --checkpoint binancecrosslearning/checkpoints/<RUN_NAME>/epoch_XXX.pt \
  --forecast-horizons 1,4,24 \
  --forecast-cache-root binancecrosslearning/forecast_cache \
  --maker-fee 0.0 \
  --min-edge 0.0 \
  --risk-weight 0.5 \
  --edge-mode high_low
```

Optional dip-only filter:

```bash
python -m binancecrosslearning.run_global_selector \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,LINKUSDT,ADAUSDT,APTUSDT,AVAXUSDT \
  --checkpoint binancecrosslearning/checkpoints/<RUN_NAME>/epoch_XXX.pt \
  --forecast-horizons 1,4,24 \
  --forecast-cache-root binancecrosslearning/forecast_cache \
  --dip-threshold-pct 0.01
```

## Notes

- GPU is required (Chronos2 fine-tuning + forecasting + policy inference).
- Maker fee defaults to 0.0 to reflect no-fee Binance spot pairs.
- Use `--cache-only` once forecasts are populated.
- Data root defaults to `trainingdatahourlybinance`.
