# Screened32 realism gate — `screened32_single_offset_val_full.bin`

- window_days=50, fee_rate=0.001, slippage_bps=5.0, decision_lag=2, monthly_target=27.0%/mo
- ensemble: 13-model softmax_avg (C_s7, D_s16, D_s42, AD_s4, I_s3, I_s3, I_s3, D_s2, D_s14, D_s28, D_s57, D_s64, I_s32)

## Median monthly return (worst on each row is the realistic deploy gate)

| fill_bps \ leverage | 1x |
|---:|---:|
| 5 | +7.24% ❌ |

## p10 monthly return (tail risk)

| fill_bps \ leverage | 1x |
|---:|---:|
| 5 | +2.83% |

## Negative-window count (out of 263)

| fill_bps \ leverage | 1x |
|---:|---:|
| 5 | 13/263 |

## Per-cell raw

| fill_bps | leverage | median_total | p10_total | sortino | max_dd | n_neg |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1 | +18.12% | +6.86% | 6.66 | 5.60% | 13/263 |
