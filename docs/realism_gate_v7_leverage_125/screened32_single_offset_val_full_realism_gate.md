# Screened32 realism gate — `screened32_single_offset_val_full.bin`

- window_days=50, fee_rate=0.001, slippage_bps=5.0, decision_lag=2, monthly_target=27.0%/mo
- ensemble: 12-model softmax_avg (C_s7, D_s16, D_s42, AD_s4, I_s3, D_s2, D_s14, D_s28, D_s57, I_s3, D_s64, I_s32)

## Median monthly return (worst on each row is the realistic deploy gate)

| fill_bps \ leverage | 1x | 1.25x | 1.5x |
|---:|---:|---:|---:|
| 5 | +7.47% ❌ | +9.35% ❌ | +10.31% ⚠️ |

## p10 monthly return (tail risk)

| fill_bps \ leverage | 1x | 1.25x | 1.5x |
|---:|---:|---:|---:|
| 5 | +3.18% | +3.80% | +4.09% |

## Negative-window count (out of 263)

| fill_bps \ leverage | 1x | 1.25x | 1.5x |
|---:|---:|---:|---:|
| 5 | 10/263 | 13/263 | 13/263 |

## Per-cell raw

| fill_bps | leverage | median_total | p10_total | sortino | max_dd | n_neg |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1 | +18.70% | +7.73% | 6.74 | 5.71% | 10/263 |
| 5 | 1.25 | +23.72% | +9.30% | 6.84 | 6.38% | 13/263 |
| 5 | 1.5 | +26.31% | +10.02% | 6.13 | 7.86% | 13/263 |
