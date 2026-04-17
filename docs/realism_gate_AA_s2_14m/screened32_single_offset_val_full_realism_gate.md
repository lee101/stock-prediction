# Screened32 realism gate — `screened32_single_offset_val_full.bin`

- window_days=50, fee_rate=0.001, slippage_bps=5.0, decision_lag=2, monthly_target=27.0%/mo
- ensemble: 14-model softmax_avg (C_s7, D_s16, D_s42, AD_s4, I_s3, D_s2, D_s14, D_s28, D_s81, D_s57, I_s3, D_s64, I_s32, val_best)

## Median monthly return (worst on each row is the realistic deploy gate)

| fill_bps \ leverage | 1x | 1.5x |
|---:|---:|---:|
| 5 | +6.35% ❌ | +9.50% ❌ |

## p10 monthly return (tail risk)

| fill_bps \ leverage | 1x | 1.5x |
|---:|---:|---:|
| 5 | +1.24% | +3.04% |

## Negative-window count (out of 263)

| fill_bps \ leverage | 1x | 1.5x |
|---:|---:|---:|
| 5 | 16/263 | 12/263 |

## Per-cell raw

| fill_bps | leverage | median_total | p10_total | sortino | max_dd | n_neg |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1 | +15.80% | +2.98% | 5.42 | 6.97% | 16/263 |
| 5 | 1.5 | +24.12% | +7.40% | 5.43 | 10.45% | 12/263 |
