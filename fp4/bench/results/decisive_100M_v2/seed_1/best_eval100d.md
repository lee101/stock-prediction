# 100d unseen-data eval — `best.pt`

- **status**: FAIL  (-0.24%/month vs target 27.00%/month)
- windows: 30 × 100d  (total 3000d unseen)
- backend: pufferlib_market

| slip_bps | median total | median monthly | p10 total | p10 monthly | sortino | max dd | n_neg |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.60% | -0.13% | -4.85% | -1.04% | -1.28 | 8.10% | 19/30 |
| 5 | -1.12% | -0.24% | -4.14% | -0.88% | -1.94 | 8.31% | 20/30 |
| 10 | +0.14% | +0.03% | -2.99% | -0.63% | 1.17 | 7.09% | 11/30 |
| 20 | -0.90% | -0.19% | -4.13% | -0.88% | -1.48 | 7.50% | 20/30 |
