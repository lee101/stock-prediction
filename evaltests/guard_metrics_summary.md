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
| AAPL_real_full | 0.03687918982468545 | 13.304040882839276 | -0.0076032618684381955 |
| AAPL_real_full_highsamples | 0.03597237475332804 | 12.741403811328754 | -0.07821590848101348 |
| GOOG | 0.0124 | 5.0736 | -0.0788 |
| GOOG_real | 0.0294 | 10.8298 | -0.2143 |
| GOOG_real_full | 0.029468842223868707 | 10.806992814298546 | 0.01916485108000914 |
| GOOG_real_full_compile | 0.029468842223868707 | 10.806992814298546 | -0.09130970439155622 |
| GOOG_real_full_compile128 | 0.07311103189364075 | 33.45777719622493 | -0.14299526197281753 |
| GOOG_real_full_highsamples | 0.056181883807294074 | 19.889755186314936 | -0.05077337794069213 |
| META | 0.0281 | 9.2342 | -0.0182 |
| META_real | 0.0412 | 13.9079 | -0.0281 |
| META_real_full | 0.04034715726913419 | 14.652047405735669 | 0.005897523523374067 |
| META_real_full_compile | 0.04034715726913419 | 14.652047405735669 | 0.005897523523374067 |
| META_real_full_compile128 | 0.08567062084679491 | 42.62611336748701 | 0.004367391865769309 |
| META_real_full_highsamples | 0.04058849136403296 | 14.77149664032237 | -0.00523781554410937 |
| NVDA | 0.0212 | 4.0324 | -0.021 |
| NVDA_real | 0.0474 | 11.4997 | 0.0117 |
| NVDA_real_full | 0.04475195929699112 | 11.615073311015085 | -0.15439073387392405 |
| NVDA_real_full_highsamples | 0.04390342730330303 | 10.731096358714813 | 0.009535756213032513 |
| TSLA | 0.0309 | 4.4751 | -0.0201 |
| TSLA_real | 0.0704 | 10.8814 | -0.0213 |
| TSLA_real_full | 0.07801231906632893 | 11.58671876167769 | -0.18857196777391266 |
| TSLA_real_full_compile | 0.07801231906632893 | 11.58671876167769 | -0.18857196777391266 |
| TSLA_real_full_compile128 | 0.16220937218051404 | 38.861773360463175 | -0.08812007986757915 |
| TSLA_real_full_highsamples | 0.07784716906840913 | 11.518784917200744 | -0.001985723645388444 |
