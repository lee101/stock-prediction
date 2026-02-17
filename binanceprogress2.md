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

## Margin/Leverage Trading Experiment (2026-02-17)

Trained leveraged SUI margin trading policies with 2-5x leverage on Binance cross-margin.
Margin interest: 2.23% annual (~0.00025% hourly). Checkpoint: `lev4x_rw0.012_s1337`.

### Leverage Comparison ($5k start, 10bp fees, lev4x checkpoint)

| Window | 1x | 2x | 3x | 4x | 5x | 4x 0-fee |
|--------|-----|-----|------|------|-------|----------|
| 3d | 1.3x | 1.7x | 2.1x | 2.8x | 3.5x | 3.2x |
| 7d | 1.7x | 2.9x | 4.9x | 8.2x | 13.8x | 11.1x |
| 10d | 2.1x | 4.6x | 9.7x | 20.5x | 43.3x | 32.7x |
| 14d | 3.2x | 10.1x | 31.5x | 97.0x | 295.5x | 182.4x |
| 30d | 6.7x | 43.5x | 279.9x | 1770x | 11016x | 6466x |

Sortino: ~163-211 (3d) to ~138-144 (70d), consistently increases with leverage.
Max DD: ~0.9% per 1x leverage (linear scaling, -3.7% at 4x, -4.6% at 5x).
Margin cost: negligible at short windows (<1% at 5x/10d), grows with compounding.

### Key Findings
- 5x > 4x on all metrics (sortino, return) with +0.9% more drawdown
- Max DD scales linearly with leverage (good risk properties)
- Margin cost (2.23% annual) is negligible vs trading profits
- 0-fee 4x is 1.5-3x better than 10bp 4x (fee sensitivity)
- Sim reinvests all profits at full leverage -> exponential compounding at long horizons

### Bug Fixes (2026-02-17)
- **Position sizing**: was using model intensity to scale trade size (0.2% -> $34 trades instead of $20k at 4x). Fixed: uses full equity * leverage.
- **Equity calc**: now uses USDT net + SUI net * price instead of just USDT net
- **Repay logic**: cancels open orders before repaying, partial repay fallback
- **Simultaneous buy+sell**: bot now places buy orders (add to position up to max leverage) while also managing sell orders, instead of only doing one at a time

### Deployed Config (2026-02-17)
- Supervisor: `binance-sui-margin` RUNNING
- Checkpoint: `binanceleveragesui/checkpoints/lev4x_rw0.012_s1337/policy_checkpoint.pt`
- Max leverage: 4.0x, cycle: 5min, max hold: 6h
- Capital: ~$5,085 USDT in cross-margin account
- Simultaneous buy+sell enabled (adds to position while managing exit)

## Model Artifacts

- LoRA checkpoint: `chronos2_finetuned/SUI_lora_ctx512/finetuned-ckpt`
- Forecast cache: `binancechronossolexperiment/forecast_cache_sui_10bp/`
- Policy checkpoint (1x): `binancechronossolexperiment/checkpoints/sui_sortino_rw0012_lr1e4_ep25/policy_checkpoint.pt`
- Policy checkpoint (4x margin): `binanceleveragesui/checkpoints/lev4x_rw0.012_s1337/policy_checkpoint.pt`
- Hyperparams: `hyperparams/chronos2/hourly/SUIUSDT.json`
