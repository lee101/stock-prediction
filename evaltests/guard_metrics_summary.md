# GymRL Guard Telemetry Summary

- Production baseline realised PnL (latest snapshot): -8,661.71

### Hold-Out Windows

- **Start index 3781 (stress slice)**
  - Baseline cumulative return: -0.041573
  - Guarded cumulative return: -0.04214310646057129
  - Turnover delta: -0.008159488439559937
  - Guard hit rates (neg/turn/draw): 33.33% / 9.52% / 19.05%
- **Start index 3600**
  - Baseline cumulative return: 0.010091
  - Guarded cumulative return: 0.009995341300964355
  - Turnover delta: 0.0005117356777191162
  - Guard hit rates (neg/turn/draw): 0.00% / 16.67% / 0.00%
- **Start index 3300**
  - Baseline cumulative return: -0.015145
  - Guarded cumulative return: -0.015201747417449951
  - Turnover delta: 0.0006309449672698975
  - Guard hit rates (neg/turn/draw): 0.00% / 23.81% / 0.00%
- **Latest guard confirm (start index 3781)**
  - Cumulative return: -0.043543219566345215
  - Guard hit rates (neg/turn/draw): 40.48% / 0.00% / 45.24%
  - Avg turnover: 0.06574228405952454 (avg leverage scale 0.8190474510192871)

Additional guard confirm windows:
- start 0: return -0.014749586582183838, turn=0.38901087641716003, guards neg/turn/draw = 0.00%/26.19%/16.67%
- start 500: return -0.02071279287338257, turn=0.35981687903404236, guards neg/turn/draw = 7.14%/28.57%/0.00%
- start 1000: return 0.05083155632019043, turn=0.3127894103527069, guards neg/turn/draw = 0.00%/19.05%/0.00%
- start 1500: return 0.007271409034729004, turn=0.32380396127700806, guards neg/turn/draw = 0.00%/21.43%/0.00%
- start 2000: return 0.014908552169799805, turn=0.3498964309692383, guards neg/turn/draw = 0.00%/21.43%/0.00%
- start 2500: return 0.008767247200012207, turn=0.40867969393730164, guards neg/turn/draw = 0.00%/33.33%/0.00%
- start 3000: return 0.08574116230010986, turn=0.342710942029953, guards neg/turn/draw = 0.00%/19.05%/0.00%

### Latest GymRL Validation Runs

- **gymrl ppo allocator (sweep_20251026_guard_confirm)**
  - Cumulative return: 0.10960030555725098
  - Avg daily return: 0.004977280739694834
  - Guard hit rates (neg/turn/draw): 0.0 / 0.0476190485060215 / 0.0

### Guard Metrics Trend (latest history)

- 2025-10-23T12:12:39.321559+00:00 – gymrl ppo allocator (sweep_20251026_guard_confirm)
  - Guard hit rates (neg/turn/draw): 0.0 / 0.0476190485060215 / 0.0
  - Avg daily return: 0.004977280739694834, Turnover: 0.16013744473457336
- 2025-10-23T00:37:08.249925+00:00 – gymrl ppo allocator (sweep_20251023_lossprobe_v7)
  - Guard hit rates (neg/turn/draw): n/a / n/a / n/a
  - Avg daily return: 0.0051820240914821625, Turnover: 0.14388185739517212
- 2025-10-23T00:36:33.116300+00:00 – gymrl ppo allocator (sweep_20251023_lossprobe_v6)
  - Guard hit rates (neg/turn/draw): n/a / n/a / n/a
  - Avg daily return: 0.005374973174184561, Turnover: 0.14962749183177948
- 2025-10-22T23:58:36.930398+00:00 – gymrl ppo allocator (sweep_20251023_lossprobe_v6)
  - Guard hit rates (neg/turn/draw): n/a / n/a / n/a
  - Avg daily return: 0.005374973174184561, Turnover: 0.14962749183177948
- 2025-10-22T23:57:49.029363+00:00 – gymrl ppo allocator (sweep_20251023_lossprobe_v4)
  - Guard hit rates (neg/turn/draw): n/a / n/a / n/a
  - Avg daily return: 0.005373469088226557, Turnover: 0.1745883971452713

### Mock Backtest Results

| Symbol | MaxDiff Return | MaxDiff Sharpe | Simple Return |
| --- | ---: | ---: | ---: |
| AAPL | 0.0261 | 7.6525 | -0.0699 |
| AAPL | n/a | n/a | n/a |
| AAPL_real | 0.0374 | 13.218 | -0.2166 |
| AAPL | n/a | n/a | n/a |
| AAPL | n/a | n/a | n/a |
| GOOG | 0.0124 | 5.0736 | -0.0788 |
| GOOG_real | 0.0294 | 10.8298 | -0.2143 |
| GOOG_real_full | 0.0302 | 10.9347 | -0.1415 |
| GOOG | n/a | n/a | n/a |
| META | 0.0281 | 9.2342 | -0.0182 |
| META_real | 0.0412 | 13.9079 | -0.0281 |
| META_real_full | 0.0405 | 13.6983 | -0.0197 |
| NVDA | 0.0212 | 4.0324 | -0.021 |
| NVDA_real | 0.0474 | 11.4997 | 0.0117 |
| NVDA_real_full | 0.0445 | 11.4281 | 0.0044 |
| TSLA | 0.0309 | 4.4751 | -0.0201 |
| TSLA_real | 0.0704 | 10.8814 | -0.0213 |
| TSLA_real_full | 0.0762 | 11.2807 | -0.0082 |
