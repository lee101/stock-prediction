# Screened32 realism gate — `screened32_single_offset_val_full.bin`

- window_days=30, fee_rate=0.001, slippage_bps=5.0, decision_lag=2, monthly_target=27.0%/mo
- ensemble: prod 13-model softmax_avg (C_s7 + 9 D + I_s3×2 + I_s32)

## Median monthly return (worst on each row is the realistic deploy gate)

| fill_bps \ leverage | 1x |
|---:|---:|
| 0 | +5.26% ❌ |
| 5 | +5.87% ❌ |

## p10 monthly return (tail risk)

| fill_bps \ leverage | 1x |
|---:|---:|
| 0 | -2.38% |
| 5 | -1.36% |

## Negative-window count (out of 263)

| fill_bps \ leverage | 1x |
|---:|---:|
| 0 | 50/283 |
| 5 | 42/283 |

## Per-cell raw

| fill_bps | leverage | median_total | p10_total | sortino | max_dd | n_neg |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | +7.60% | -3.39% | 4.34 | 6.00% | 50/283 |
| 5 | 1 | +8.49% | -1.94% | 5.56 | 5.43% | 42/283 |
