# Alpaca Cross‑Learning Progress

Tracking Chronos2 multi‑symbol fine‑tunes + global trading policy results.

## Latest summary (10d / 20d marketsim)

| Date (UTC) | Run | Symbols | Eval window | total_return | sortino | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-02-05 | selector_sweep_robust_intensity_20260205_022332 | SOLUSD,LINKUSD,UNIUSD | 10d | 0.5629 | 30.8584 | Robust_scaling policy; best sweep config (intensity=1.2, min_edge=0.0005, risk_weight=0.25, dip=0.005). |
| 2026-02-05 | selector_robust_best_20260205 | SOLUSD,LINKUSD,UNIUSD | 20d | 0.7482 | 21.3372 | Robust_scaling policy; best sweep config carried to 20d eval. |
| 2026-02-05 | selector_cross_lora_20260205_013206 | SOLUSD,LINKUSD,UNIUSD | 10d | 0.0379 | 510.2498 | Global policy checkpoint epoch_001. |
| 2026-02-05 | selector_cross_lora_20260205_013206 | SOLUSD,LINKUSD,UNIUSD | 20d | 0.0379 | 510.2498 | Same as 10d (limited window in current cache). |

## Chronos2 multi‑symbol fine‑tunes

| Date (UTC) | Run | Symbols | Steps | Context | LR | Preaug | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-05 | cross_smoke_20260205_012017 | SOLUSD,LINKUSD | 20 | 512 | 1e‑5 | baseline | Smoke LoRA run. |
| 2026-02-05 | cross_lora_20260205_012542 | SOLUSD,LINKUSD,UNIUSD | 200 | 1024 | 1e‑5 | baseline | Forecast cache built (h1+h24, 720h lookback). |

## Global policy training

| Date (UTC) | Run | Symbols | total_return | sortino | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-02-05 | alpaca_cross_global_lora_20260205_013206 | SOLUSD,LINKUSD,UNIUSD | 0.1872 | 47.2885 | MA windows 168/336, min_history=200. |
| 2026-02-05 | alpaca_cross_global_robust_20260205_020915 | SOLUSD,LINKUSD,UNIUSD | 0.1010 | 26.9010 | Robust_scaling forecasts, MA windows 168/336, min_history=200. |

## TODO

- Expand cross‑learning to mixed symbols (BTCUSD, ETHUSD, NVDA, NFLX) with separate cache roots.
- Re‑run selector sweeps on the mixed‑symbol policy (10d/20d) and track top PnL.
- Try higher intensity grid (1.4/1.6) and alternate min_edge for robust scaling.

## Preaug sweep (eval_loss)

| Date (UTC) | Run | Preaug | LR | eval_loss |
| --- | --- | --- | --- | --- |
| 2026-02-05 | cross_lora_20260205_012542 | baseline | 1e‑5 | 0.015715 |
| 2026-02-05 | cross_lora_pct_20260205_015038 | percent_change | 1e‑5 | 0.017027 |
| 2026-02-05 | cross_lora_pct2_20260205_015132 | percent_change | 2e‑5 | 0.016981 |
| 2026-02-05 | cross_lora_log_20260205_015221 | log_returns | 1e‑5 | 0.018255 |
| 2026-02-05 | cross_lora_log2_20260205_015312 | log_returns | 2e‑5 | 0.018150 |
| 2026-02-05 | cross_lora_differencing_20260205_020248 | differencing | 1e‑5 | 0.247775 |
| 2026-02-05 | cross_lora_detrending_20260205_020333 | detrending | 1e‑5 | 0.192059 |
| 2026-02-05 | cross_lora_robust_scaling_20260205_020416 | robust_scaling | 1e‑5 | 0.088192 |
| 2026-02-05 | cross_lora_minmax_standard_20260205_020502 | minmax_standard | 1e‑5 | 0.187223 |
| 2026-02-05 | cross_lora_rolling_norm_20260205_020545 | rolling_norm | 1e‑5 | 0.232486 |

## Selector sweep (10d, 36 configs)

Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_20260205_0155/selector_sweep.csv`

Best total_return (10d):
```
intensity=1.2 offset=0.0 min_edge=0.001 risk_weight=0.75 edge_mode=high_low dip_threshold=0.005
total_return=0.207267 sortino=43.719822 final_cash=832.755134
```

Best sortino (10d):
```
intensity=1.0 offset=0.0 min_edge=0.0 risk_weight=0.75 edge_mode=high_low dip_threshold=0.005
total_return=0.065652 sortino=839.391481 final_cash=10648.974612
```

## Selector sweep (10d, 27 configs, robust scaling)

Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_robust_20260205_022214/selector_sweep.csv`

Best total_return (10d):
```
intensity=1.0 offset=0.0 min_edge=0.0005 risk_weight=0.75 edge_mode=high_low dip_threshold=0.0
total_return=0.237024 sortino=26.255566 final_cash=12370.243393
```

Best sortino (10d):
```
intensity=1.0 offset=0.0 min_edge=0.0 risk_weight=0.25 edge_mode=high_low dip_threshold=0.01
total_return=0.150579 sortino=134.386252 final_cash=11505.785239
```

## Selector sweep (10d, 54 configs, robust scaling + intensity)

Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_robust_intensity_20260205_022332/selector_sweep.csv`

Best total_return (10d):
```
intensity=1.2 offset=0.0 min_edge=0.0005 risk_weight=0.25 edge_mode=high_low dip_threshold=0.005
total_return=0.562932 sortino=30.858390 final_cash=15629.319609
```

Best sortino (10d):
```
intensity=1.0 offset=0.0 min_edge=0.0 risk_weight=0.25 edge_mode=high_low dip_threshold=0.01
total_return=0.150579 sortino=134.386252 final_cash=11505.785239
```
