# Alpaca Cross‑Learning Progress

Tracking Chronos2 multi‑symbol fine‑tunes + global trading policy results.

## Latest summary (10d / 20d marketsim)

| Date (UTC) | Run | Symbols | Eval window | total_return | sortino | ann_return_365 | ann_return_252 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-05 | selector_seq128_best_20260205_043448 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 10d | 1.6463 | 128.4568 | 2.669e+15 | 4.472e+10 | Seq128 policy checkpoint epoch_003 with intensity=2.0, min_edge=0.004, risk_weight=0.2, dip=0.005. |
| 2026-02-05 | selector_seq128_best_20260205_043448 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 20d | 1.6463 | 128.4568 | 5.166e+07 | 2.115e+05 | Same config as 10d; window-limited by stock data. |
| 2026-02-05 | selector_sweep_mixed7_target3_20260205_041640 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 10d | 1.1952 | 71.5462 | 2.909e+12 | 4.029e+08 | Mixed7 robust_scaling short-window; targeted min_edge/risk sweep v3 (intensity=2.0, min_edge=0.004, risk_weight=0.2, dip=0.005). |
| 2026-02-05 | selector_mixed7_best_20260205_target3 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 20d | 1.1952 | 71.5462 | 1.706e+06 | 2.007e+04 | Same config as 10d; window-limited by stock data. |
| 2026-02-05 | selector_sweep_mixed7_target2_20260205_041026 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 10d | 1.1819 | 73.1664 | 2.331e+12 | 3.457e+08 | Mixed7 robust_scaling short-window; targeted min_edge/risk sweep (intensity=2.0, min_edge=0.003, risk_weight=0.25, dip=0.005). |
| 2026-02-05 | selector_mixed7_best_20260205_target2 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 20d | 1.1819 | 73.1664 | 1.527e+06 | 1.859e+04 | Same config as 10d; window-limited by stock data. |
| 2026-02-05 | selector_sweep_mixed7_intensity20_20260205_040605 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 10d | 1.1737 | 73.5806 | 2.031e+12 | 3.144e+08 | Mixed7 robust_scaling short-window; best sweep config (intensity=2.0, min_edge=0.001, risk_weight=0.25, dip=0.005). |
| 2026-02-05 | selector_mixed7_best_20260205_int20 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 20d | 1.1737 | 73.5806 | 1.425e+06 | 1.773e+04 | Same config as 10d; window-limited by stock data. |
| 2026-02-05 | selector_sweep_mixed7_intensity16_20260205_035644 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 10d | 1.1541 | 77.5686 | 1.459e+12 | 2.502e+08 | Mixed7 robust_scaling short-window; best sweep config (intensity=1.6, min_edge=0.0005, risk_weight=0.25, dip=0.005). |
| 2026-02-05 | selector_mixed7_best_20260205_int16 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 20d | 1.1541 | 77.5686 | 1.208e+06 | 1.582e+04 | Same config as 10d; window-limited by stock data. |
| 2026-02-05 | selector_sweep_mixed7_20260205_025742 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 10d | 1.1196 | 77.4274 | 8.096e+11 | 1.666e+08 | Mixed7 robust_scaling + short-window features; best sweep config (intensity=1.2, min_edge=0.0005, risk_weight=0.25, dip=0.005). |
| 2026-02-05 | selector_mixed7_best_20260205 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 20d | 1.1196 | 77.4274 | 8.998e+05 | 1.291e+04 | Same config as 10d; window-limited by stock data. |
| 2026-02-05 | selector_sweep_robust_intensity_20260205_022332 | SOLUSD,LINKUSD,UNIUSD | 10d | 0.5629 | 30.8584 | 1.198e+07 | 7.71e+04 | Robust_scaling policy; best sweep config (intensity=1.2, min_edge=0.0005, risk_weight=0.25, dip=0.005). |
| 2026-02-05 | selector_robust_best_20260205 | SOLUSD,LINKUSD,UNIUSD | 20d | 0.7482 | 21.3372 | 2.675e+04 | 1138 | Robust_scaling policy; best sweep config carried to 20d eval. |
| 2026-02-05 | selector_cross_lora_20260205_013206 | SOLUSD,LINKUSD,UNIUSD | 10d | 0.0379 | 510.2498 | 2.888 | 1.553 | Global policy checkpoint epoch_001. |
| 2026-02-05 | selector_cross_lora_20260205_013206 | SOLUSD,LINKUSD,UNIUSD | 20d | 0.0379 | 510.2498 | 0.9717 | 0.5979 | Same as 10d (limited window in current cache). |

Annualized returns use CAGR: `(1 + total_return) ** (basis_days / eval_days) - 1`. Short windows will inflate annualized values; use them only for relative comparison across runs.

## Constrained experiment (alpacaconstrainedexp)

| Date (UTC) | Run | Stage | Symbols | total_return | sortino | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-02-05 | chronos2_multi_20260205_022403 | Chronos2 LoRA | NVDA,MSFT,GOOG,BTCUSD,SOLUSD,ETHUSD,NWSA,BKNG,MTCH,EXPE,EBAY | (n/a) | (n/a) | Preaug=percent_change, ctx=1024, steps=1000. |
| 2026-02-05 | constrained_global_20260205_0230 | Global policy | NVDA,MSFT,GOOG,BTCUSD,SOLUSD,ETHUSD,NWSA,BKNG,MTCH,EXPE,EBAY | 0.0608 | 73.1732 | MA windows=168, min_history=150. |
| 2026-02-05 | selector_20260205_10d | Selector 10d | NVDA,MSFT,GOOG,BTCUSD,SOLUSD,ETHUSD,NWSA,BKNG,MTCH,EXPE,EBAY | (running) | (running) | Constrained selector sim (checkpoint epoch_004). |

## Chronos2 multi‑symbol fine‑tunes

| Date (UTC) | Run | Symbols | Steps | Context | LR | Preaug | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-05 | cross_smoke_20260205_012017 | SOLUSD,LINKUSD | 20 | 512 | 1e‑5 | baseline | Smoke LoRA run. |
| 2026-02-05 | cross_lora_20260205_012542 | SOLUSD,LINKUSD,UNIUSD | 200 | 1024 | 1e‑5 | baseline | Forecast cache built (h1+h24, 720h lookback). |
| 2026-02-05 | cross_lora_mixed7_robust_20260205_022709 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 200 | 1024 | 1e‑5 | robust_scaling | Mixed7 LoRA (preaug robust_scaling). |
| 2026-02-05 | cross_lora_mixed7_robust_lr2e5_20260205_040853 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 200 | 1024 | 2e‑5 | robust_scaling | Mixed7 LoRA (eval_loss=0.167414). |

## Global policy training

| Date (UTC) | Run | Symbols | total_return | sortino | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-02-05 | alpaca_cross_global_lora_20260205_013206 | SOLUSD,LINKUSD,UNIUSD | 0.1872 | 47.2885 | MA windows 168/336, min_history=200. |
| 2026-02-05 | alpaca_cross_global_robust_20260205_020915 | SOLUSD,LINKUSD,UNIUSD | 0.1010 | 26.9010 | Robust_scaling forecasts, MA windows 168/336, min_history=200. |
| 2026-02-05 | alpaca_cross_global_mixed7_robust_short_20260205_025307 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 0.3038 | 76.5146 | Mixed7 policy with short-window feature overrides; window 2025-09-01..2025-11-12. |
| 2026-02-05 | alpaca_cross_global_mixed7_robust_short_seq128_20260205_043448 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 0.5935 | 291.1584 | Seq128 policy with short-window feature overrides; best checkpoint epoch_003. |

## TODO

- Extend mixed7 window beyond 2025‑11‑12 to add more stock hours (refresh data) and re‑train.
- Run 30d selector sweeps once longer stock windows are available.
- Try higher intensity grid (1.4/1.6) and alternate min_edge for mixed7 policy.

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

## Selector sweep (10d, 54 configs, mixed7 short-window)

Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_mixed7_20260205_025742/selector_sweep.csv`

Best total_return (10d):
```
intensity=1.2 offset=0.0 min_edge=0.0005 risk_weight=0.25 edge_mode=high_low dip_threshold=0.005
total_return=1.119632 sortino=77.427394 final_cash=21196.315290
```

Best sortino (10d):
```
intensity=1.2 offset=0.0 min_edge=0.0 risk_weight=0.25 edge_mode=high_low dip_threshold=0.0
total_return=0.954793 sortino=149.867478 final_cash=19547.932885
```

## Selector sweep (10d, 54 configs, mixed7 intensity 1.4/1.6)

Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_mixed7_intensity16_20260205_035644/selector_sweep.csv`

Best total_return (10d):
```
intensity=1.6 offset=0.0 min_edge=0.0005 risk_weight=0.25 edge_mode=high_low dip_threshold=0.005
total_return=1.154100 sortino=77.568583 final_cash=21540.999453
```

Best sortino (10d):
```
intensity=1.4 offset=0.0 min_edge=0.0 risk_weight=0.25 edge_mode=high_low dip_threshold=0.0
total_return=0.966661 sortino=135.564330 final_cash=19666.608425
```

## Selector sweep (10d, 54 configs, mixed7 intensity 1.8/2.0)

Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_mixed7_intensity20_20260205_040605/selector_sweep.csv`

Best total_return (10d):
```
intensity=2.0 offset=0.0 min_edge=0.001 risk_weight=0.25 edge_mode=high_low dip_threshold=0.005
total_return=1.173715 sortino=73.580583 final_cash=21737.154313
```

Best sortino (10d):
```
intensity=2.0 offset=0.0 min_edge=0.0 risk_weight=0.25 edge_mode=high_low dip_threshold=0.0
total_return=0.992136 sortino=132.820633 final_cash=19921.356461
```

## Selector sweep (10d, 48 configs, mixed7 targeted min_edge/risk)

Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_mixed7_target2_20260205_041026/selector_sweep.csv`

Best total_return (10d):
```
intensity=2.0 offset=0.0 min_edge=0.003 risk_weight=0.25 edge_mode=high_low dip_threshold=0.005
total_return=1.181937 sortino=73.166435 final_cash=21819.366134
```

Best sortino (10d):
```
intensity=2.0 offset=0.0 min_edge=0.0005 risk_weight=0.25 edge_mode=high_low dip_threshold=0.0
total_return=0.982480 sortino=123.332145 final_cash=19824.803099
```

## Selector sweep (10d, 27 configs, mixed7 targeted min_edge/risk v3)

Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_mixed7_target3_20260205_041640/selector_sweep.csv`

Best total_return (10d):
```
intensity=2.0 offset=0.0 min_edge=0.004 risk_weight=0.2 edge_mode=high_low dip_threshold=0.005
total_return=1.195198 sortino=71.546160 final_cash=21951.980205
```

Best sortino (10d):
```
intensity=2.0 offset=0.0 min_edge=0.003 risk_weight=0.2 edge_mode=high_low dip_threshold=0.0025
total_return=1.097744 sortino=79.130229 final_cash=20977.442667
```
