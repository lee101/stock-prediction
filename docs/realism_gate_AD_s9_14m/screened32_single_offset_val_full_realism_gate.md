# Screened32 realism gate — `screened32_single_offset_val_full.bin`

- window_days=50, fee_rate=0.001, slippage_bps=5.0, decision_lag=2, monthly_target=27.0%/mo
- ensemble: 14-model softmax_avg (C_s7, D_s16, D_s42, D_s3, I_s3, D_s2, D_s14, D_s28, D_s81, D_s57, I_s3, D_s64, I_s32, best)

## Median monthly return (worst on each row is the realistic deploy gate)

| fill_bps \ leverage | 1x | 1.5x |
|---:|---:|---:|
| 5 | +5.90% ❌ | +8.59% ❌ |

## p10 monthly return (tail risk)

| fill_bps \ leverage | 1x | 1.5x |
|---:|---:|---:|
| 5 | +0.67% | +1.06% |

## Negative-window count (out of 263)

| fill_bps \ leverage | 1x | 1.5x |
|---:|---:|---:|
| 5 | 16/263 | 16/263 |

## Per-cell raw

| fill_bps | leverage | median_total | p10_total | sortino | max_dd | n_neg |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1 | +14.63% | +1.61% | 4.66 | 7.64% | 16/263 |
| 5 | 1.5 | +21.68% | +2.55% | 4.71 | 11.32% | 16/263 |
