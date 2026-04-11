# 100d unseen-data eval — `final.pt`

- **status**: FAIL  (-1.08%/month vs target 27.00%/month)
- windows: 30 × 100d  (total 3000d unseen)
- backend: pufferlib_market

| slip_bps | median total | median monthly | p10 total | p10 monthly | sortino | max dd | n_neg |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -5.00% | -1.07% | -9.73% | -2.13% | -1.75 | 28.39% | 25/30 |
| 5 | -2.94% | -0.63% | -8.47% | -1.84% | -0.40 | 26.90% | 23/30 |
| 10 | -4.03% | -0.86% | -9.44% | -2.06% | -1.10 | 27.18% | 24/30 |
| 20 | -5.04% | -1.08% | -10.33% | -2.26% | -1.91 | 27.75% | 25/30 |
