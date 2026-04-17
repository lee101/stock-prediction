# Screened32 realism gate — `screened32_single_offset_val_full.bin`

- window_days=50, fee_rate=0.001, slippage_bps=5.0, decision_lag=2, monthly_target=27.0%/mo
- ensemble: 12-model softmax_avg (C_s7, D_s16, D_s42, AD_s4, I_s3, D_s2, D_s14, D_s28, D_s57, I_s3, D_s64, I_s32)

## Median monthly return (worst on each row is the realistic deploy gate)

| fill_bps \ leverage | 1.5x | 2x | 2.5x | 3x |
|---:|---:|---:|---:|---:|
| 5 | +10.31% ⚠️ | +11.94% ⚠️ | +14.99% ⚠️ | +17.21% ⚠️ |

## p10 monthly return (tail risk)

| fill_bps \ leverage | 1.5x | 2x | 2.5x | 3x |
|---:|---:|---:|---:|---:|
| 5 | +4.09% | +3.22% | +4.17% | +3.14% |

## Negative-window count (out of 263)

| fill_bps \ leverage | 1.5x | 2x | 2.5x | 3x |
|---:|---:|---:|---:|---:|
| 5 | 13/263 | 15/263 | 17/263 | 19/263 |

## Per-cell raw

| fill_bps | leverage | median_total | p10_total | sortino | max_dd | n_neg |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1.5 | +26.31% | +10.02% | 6.13 | 7.86% | 13/263 |
| 5 | 2 | +30.80% | +7.84% | 5.45 | 15.44% | 15/263 |
| 5 | 2.5 | +39.45% | +10.21% | 5.26 | 16.52% | 17/263 |
| 5 | 3 | +45.96% | +7.65% | 5.57 | 19.41% | 19/263 |
