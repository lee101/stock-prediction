# 100d unseen-data eval — `best.pt`

- **status**: FAIL  (0.68%/month vs target 27.00%/month)
- windows: 30 × 100d  (total 3000d unseen)
- backend: pufferlib_market

| slip_bps | median total | median monthly | p10 total | p10 monthly | sortino | max dd | n_neg |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | +3.30% | +0.68% | -18.44% | -4.19% | 9.62 | 22.49% | 12/30 |
| 5 | +4.83% | +1.00% | -17.22% | -3.89% | 10.51 | 22.68% | 10/30 |
| 10 | +8.67% | +1.76% | -14.17% | -3.16% | 8.91 | 22.87% | 6/30 |
| 20 | +6.34% | +1.30% | -15.97% | -3.59% | 10.36 | 23.24% | 9/30 |
