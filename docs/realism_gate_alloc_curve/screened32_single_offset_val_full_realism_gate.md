# Screened32 realism gate — `screened32_single_offset_val_full.bin`

- window_days=50, fee_rate=0.001, slippage_bps=5.0, decision_lag=2, monthly_target=27.0%/mo
- ensemble: 13-model softmax_avg (C_s7, D_s16, D_s42, D_s3, I_s3, D_s2, D_s14, D_s28, D_s81, D_s57, I_s3, D_s64, I_s32)

## Median monthly return (worst on each row is the realistic deploy gate)

| fill_bps \ leverage | 0.125x | 0.25x | 0.5x | 0.75x | 1x | 1.5x | 2x |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | +0.90% ❌ | +1.79% ❌ | +3.50% ❌ | +5.15% ❌ | +6.89% ❌ | +10.19% ⚠️ | +12.60% ⚠️ |

## p10 monthly return (tail risk)

| fill_bps \ leverage | 0.125x | 0.25x | 0.5x | 0.75x | 1x | 1.5x | 2x |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | +0.37% | +0.75% | +1.44% | +1.88% | +2.34% | +3.24% | +4.06% |

## Negative-window count (out of 263)

| fill_bps \ leverage | 0.125x | 0.25x | 0.5x | 0.75x | 1x | 1.5x | 2x |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 15/263 | 13/263 | 11/263 | 11/263 | 11/263 | 13/263 | 14/263 |

## Per-cell raw

| fill_bps | leverage | median_total | p10_total | sortino | max_dd | n_neg |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.125 | +2.16% | +0.89% | 5.62 | 0.89% | 15/263 |
| 5 | 0.25 | +4.32% | +1.81% | 5.59 | 1.77% | 13/263 |
| 5 | 0.5 | +8.52% | +3.45% | 5.57 | 3.48% | 11/263 |
| 5 | 0.75 | +12.69% | +4.55% | 5.98 | 5.57% | 11/263 |
| 5 | 1 | +17.20% | +5.67% | 6.10 | 6.28% | 11/263 |
| 5 | 1.5 | +25.99% | +7.88% | 6.20 | 8.62% | 13/263 |
| 5 | 2 | +32.64% | +9.95% | 5.77 | 14.92% | 14/263 |
