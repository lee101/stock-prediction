# Alpaca Cross-Learning Progress

Tracking multi-symbol Chronos2 fine-tunes and global selector results.

## Results

| Timestamp (UTC) | Run | Symbols | total_return | sortino | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-02-05 01:20 | cross_smoke_20260205_012017 | SOLUSD,LINKUSD | (n/a) | (n/a) | Chronos2 LoRA smoke run (20 steps, context=512, pred=1). |
| 2026-02-05 01:26 | cross_lora_20260205_012542 | SOLUSD,LINKUSD,UNIUSD | (n/a) | (n/a) | Chronos2 LoRA (200 steps, ctx=1024, pred=1). Forecast cache built (lookback 720h, h1+h24). |
| 2026-02-05 01:32 | alpaca_cross_global_lora_20260205_013206 | SOLUSD,LINKUSD,UNIUSD | 0.1872 | 47.2885 | Global policy trained on cross-learning forecasts (MA windows 168/336, min_history=200). |
| 2026-02-05 01:33 | selector_cross_lora_20260205_013206 | SOLUSD,LINKUSD,UNIUSD | 0.0379 | 510.2498 | Best-trade selector using global policy checkpoint. |
| 2026-02-05 01:51 | cross_lora_pct_20260205_015038 | SOLUSD,LINKUSD,UNIUSD | (n/a) | (n/a) | Preaug percent_change, lr=1e-5, eval_loss=0.017027. |
| 2026-02-05 01:52 | cross_lora_pct2_20260205_015132 | SOLUSD,LINKUSD,UNIUSD | (n/a) | (n/a) | Preaug percent_change, lr=2e-5, eval_loss=0.016981. |
| 2026-02-05 01:53 | cross_lora_log_20260205_015221 | SOLUSD,LINKUSD,UNIUSD | (n/a) | (n/a) | Preaug log_returns, lr=1e-5, eval_loss=0.018255. |
| 2026-02-05 01:54 | cross_lora_log2_20260205_015312 | SOLUSD,LINKUSD,UNIUSD | (n/a) | (n/a) | Preaug log_returns, lr=2e-5, eval_loss=0.018150. |
| 2026-02-05 01:55 | selector_sweep_20260205_0155 | SOLUSD,LINKUSD,UNIUSD | 0.2073 | 43.7198 | 10d selector sweep (36 configs); best total_return config. |
| 2026-02-05 02:03 | cross_lora_differencing_20260205_020248 | SOLUSD,LINKUSD,UNIUSD | (n/a) | (n/a) | Preaug differencing, lr=1e-5, eval_loss=0.247775. |
| 2026-02-05 02:04 | cross_lora_detrending_20260205_020333 | SOLUSD,LINKUSD,UNIUSD | (n/a) | (n/a) | Preaug detrending, lr=1e-5, eval_loss=0.192059. |
| 2026-02-05 02:05 | cross_lora_robust_scaling_20260205_020416 | SOLUSD,LINKUSD,UNIUSD | (n/a) | (n/a) | Preaug robust_scaling, lr=1e-5, eval_loss=0.088192 (best so far). |
| 2026-02-05 02:05 | cross_lora_minmax_standard_20260205_020502 | SOLUSD,LINKUSD,UNIUSD | (n/a) | (n/a) | Preaug minmax_standard, lr=1e-5, eval_loss=0.187223. |
| 2026-02-05 02:06 | cross_lora_rolling_norm_20260205_020545 | SOLUSD,LINKUSD,UNIUSD | (n/a) | (n/a) | Preaug rolling_norm, lr=1e-5, eval_loss=0.232486. |
| 2026-02-05 02:09 | alpaca_cross_global_robust_20260205_020915 | SOLUSD,LINKUSD,UNIUSD | 0.1010 | 26.9010 | Global policy trained on robust_scaling forecasts (MA 168/336, min_history=200). |
| 2026-02-05 02:22 | selector_sweep_robust_20260205_022214 | SOLUSD,LINKUSD,UNIUSD | 0.2370 | 26.2556 | 10d selector sweep (27 configs) on robust_scaling policy. |
| 2026-02-05 02:23 | selector_sweep_robust_intensity_20260205_022332 | SOLUSD,LINKUSD,UNIUSD | 0.5629 | 30.8584 | 10d selector sweep (54 configs, intensity 1.0/1.2); best total_return config. |
| 2026-02-05 02:24 | selector_robust_best_20260205 | SOLUSD,LINKUSD,UNIUSD | 0.7482 | 21.3372 | 20d eval using best robust_scaling config (intensity=1.2, min_edge=0.0005, risk_weight=0.25, dip=0.005). |

## Notes

- Forecast cache root: `alpacanewccrosslearning/forecast_cache`
- Chronos fine-tunes: `alpacanewccrosslearning/chronos_finetuned`
