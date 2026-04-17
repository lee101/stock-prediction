# Screened32 realism gate — `screened32_single_offset_val_full.bin`

- window_days=50, fee_rate=0.001, slippage_bps=5.0, decision_lag=2, monthly_target=27.0%/mo
- ensemble: 14-model softmax_avg (C_s7, D_s16, D_s42, AD_s4, I_s3, D_s2, D_s14, D_s28, D_s81, D_s57, I_s3, D_s64, I_s32, val_best)

## Median monthly return (worst on each row is the realistic deploy gate)

| fill_bps \ leverage | 1x | 1.5x |
|---:|---:|---:|
| 5 | +6.22% ❌ | +8.05% ❌ |

## p10 monthly return (tail risk)

| fill_bps \ leverage | 1x | 1.5x |
|---:|---:|---:|
| 5 | +0.59% | -0.34% |

## Negative-window count (out of 263)

| fill_bps \ leverage | 1x | 1.5x |
|---:|---:|---:|
| 5 | 22/263 | 29/263 |

## Per-cell raw

| fill_bps | leverage | median_total | p10_total | sortino | max_dd | n_neg |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1 | +15.45% | +1.41% | 5.80 | 6.48% | 22/263 |
| 5 | 1.5 | +20.24% | -0.81% | 5.08 | 10.13% | 29/263 |
