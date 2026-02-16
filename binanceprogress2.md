# Binance Progress Log 2 - SUI Trading

Updated: 2026-02-15

## SUI LoRA Fine-tuning

Optimized Chronos2 LoRA for SUI hourly forecasting with context=512.

| Config | Val MAE% |
|--------|----------|
| ctx=128 baseline | 3.19 |
| ctx=256 lr=5e-5 | 3.02 |
| ctx=512 baseline | 2.82 |
| ctx=1024 | 3.16 |
| ctx=512 steps=200 | 3.50 |
| ctx=512 lr=1e-4 r=32 steps=300 | 3.52 |

Best: ctx=512 with ~2.8-3.5% MAE (varies with data window).

Dilation ensemble tested - did not improve MAE vs single inference.

## Trading Bot Training

Training config: 10bp maker/taker fees, 25 epochs, sortino+return weighted loss.

Checkpoint: `binancechronossolexperiment/checkpoints/sui_sortino_rw0012_lr1e4_ep25/policy_checkpoint.pt`

Final training metrics:
- Train sortino: 2322.82, return: 22.77%
- Val sortino: 288.84, return: 19.67%

## 7-Day Holdout Backtest (2026-02-08 to 2026-02-15)

| Strategy | Return | Sortino | Max DD | Trades | Final Equity |
|----------|--------|---------|--------|--------|--------------|
| Momentum | +5.65% | 9.08 | -5.84% | 3 | $10,565 |
| Neural (10bp) | +153.57% | 612.03 | -0.33% | 139 | $25,357 |

Neural policy significantly outperforms momentum baseline with 10bp fees:
- 27x higher return
- 67x higher sortino
- 17x lower max drawdown

## Model Artifacts

- LoRA checkpoint: `chronos2_finetuned/SUI_lora_ctx512/finetuned-ckpt`
- Forecast cache: `binancechronossolexperiment/forecast_cache_sui_10bp/`
- Policy checkpoint: `binancechronossolexperiment/checkpoints/sui_sortino_rw0012_lr1e4_ep25/policy_checkpoint.pt`
- Hyperparams: `hyperparams/chronos2/hourly/SUIUSDT.json`
