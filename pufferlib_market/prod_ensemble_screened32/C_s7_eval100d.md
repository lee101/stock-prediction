# 100d unseen-data eval — `C_s7.pt`

- **status**: FAIL  (-2.04%/month vs target 27.00%/month)
- windows: 20 × 30d  (total 600d unseen)
- backend: pufferlib_market

| slip_bps | median total | median monthly | p10 total | p10 monthly | sortino | max dd | n_neg |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -2.90% | -2.04% | -11.39% | -8.11% | -10.87 | 9.07% | 12/20 |
| 5 | -2.80% | -1.97% | -11.40% | -8.12% | -11.59 | 8.19% | 12/20 |
| 10 | -2.32% | -1.63% | -14.29% | -10.24% | -11.36 | 7.95% | 11/20 |
| 20 | -1.27% | -0.89% | -14.97% | -10.73% | -8.74 | 7.47% | 11/20 |
