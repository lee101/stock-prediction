# Alpaca Cross-Learning (Chronos2 + Global Policy)

This experiment trains a **multi-symbol Chronos2 model** (cross-learning) and then
trains a **single trading policy** across symbols. The goal is a global strategy
that can allocate to the best opportunities each hour.

## 1) Fine-tune Chronos2 across symbols

```bash
source .venv313-3/bin/activate

python -m alpacanewccrosslearning.refresh_hourly_data \
  --symbols SOLUSD,LINKUSD,UNIUSD

python -m alpacanewccrosslearning.chronos_finetune_multi \
  --symbols SOLUSD,LINKUSD,UNIUSD \
  --prediction-length 1 \
  --context-length 1024 \
  --batch-size 64 \
  --learning-rate 1e-5 \
  --num-steps 1000 \
  --finetune-mode lora \
  --output-root alpacanewccrosslearning/chronos_finetuned \
  --preaug-strategy percent_change
```

## 2) Build forecasts using the fine-tuned model

```bash
python -m alpacanewccrosslearning.build_forecasts \
  --symbols SOLUSD,LINKUSD,UNIUSD \
  --finetuned-model alpacanewccrosslearning/chronos_finetuned/<RUN_NAME>/finetuned \
  --forecast-cache-root alpacanewccrosslearning/forecast_cache \
  --horizons 1,24 \
  --context-hours 336 \
  --predict-batches-jointly \
  --lookback-hours 720
```

## 3) Train a global policy across symbols

```bash
python -m alpacanewccrosslearning.train_global_policy \
  --symbols SOLUSD,LINKUSD,UNIUSD \
  --target-symbol SOLUSD \
  --epochs 4 \
  --sequence-length 96 \
  --forecast-horizons 1,24 \
  --forecast-cache-root alpacanewccrosslearning/forecast_cache \
  --cache-only \
  --allow-mixed-asset \
  --moving-average-windows 168,600 \
  --min-history-hours 100
```

For mixed stock/crypto windows with limited stock hours, you can override the feature windows:

```bash
python -m alpacanewccrosslearning.train_global_policy \
  --symbols SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX \
  --forecast-cache-root alpacanewccrosslearning/forecast_cache/mixed7 \
  --cache-only \
  --allow-mixed-asset \
  --moving-average-windows 24,72 \
  --ema-windows 24,72 \
  --atr-windows 24,72 \
  --trend-windows 72 \
  --drawdown-windows 72 \
  --volume-z-window 72 \
  --volume-shock-window 24 \
  --vol-regime-short 24 \
  --vol-regime-long 72 \
  --min-history-hours 50
```

## 4) Evaluate global selector (best trade per hour)

```bash
python -m alpacanewccrosslearning.run_global_selector \
  --symbols SOLUSD,LINKUSD,UNIUSD \
  --checkpoint binanceneural/checkpoints/<RUN_NAME>/epoch_XXX.pt \
  --forecast-horizons 1,24 \
  --forecast-cache-root alpacanewccrosslearning/forecast_cache \
  --moving-average-windows 168,336 \
  --min-history-hours 200 \
  --min-edge 0.0 \
  --risk-weight 0.5 \
  --edge-mode high_low
```

Optional dip-only filter:

```bash
python -m alpacanewccrosslearning.run_global_selector \
  --symbols SOLUSD,LINKUSD,UNIUSD \
  --checkpoint binanceneural/checkpoints/<RUN_NAME>/epoch_XXX.pt \
  --forecast-horizons 1,24 \
  --forecast-cache-root alpacanewccrosslearning/forecast_cache \
  --dip-threshold-pct 0.01
```

## 5) Selector sweep (20+ configs)

```bash
python -m alpacanewccrosslearning.run_selector_sweep \
  --symbols SOLUSD,LINKUSD,UNIUSD \
  --checkpoint binanceneural/checkpoints/<RUN_NAME>/epoch_XXX.pt \
  --forecast-horizons 1,24 \
  --forecast-cache-root alpacanewccrosslearning/forecast_cache \
  --moving-average-windows 168,336 \
  --min-history-hours 200 \
  --min-edge-list 0.0,0.0005,0.001 \
  --risk-weight-list 0.25,0.5,0.75 \
  --dip-threshold-list 0.0,0.005,0.01 \
  --eval-days 10
```

## Notes

- GPU is required (Chronos2 fine-tuning + forecasting + policy inference).
- For crypto-only runs you can omit stock data roots.
- Use `--cache-only` once forecasts are populated.
- Feature parity: training + inference both use `binanceexp1.data.build_feature_frame`. Keep the same window overrides (MA/EMA/ATR/vol regime) across training and inference to ensure volatility features match.
