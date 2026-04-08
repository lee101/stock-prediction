# fp4 sweep leaderboard (2026-04-07T23:57:48)

- mode: SMOKE
- steps/cell: 50000
- cells: 6  total runs: 6
- sort key: (p10@5bps DESC, sortino@5bps DESC, max_dd ASC)

| rank | algo | constrained | seeds | ok | p10@5bps | sortino@5bps | median@5bps | max_dd@5bps | sps |
|---:|:---|:---:|:---:|:---:|:---|:---|:---|:---|---:|
| 1 | sac | off | 1 | 1 | -0.5807 | -0.1958 | -0.06015 | — | 2063 |
| 2 | sac | on | 1 | 1 | -0.5807 | -0.1958 | -0.06015 | — | 2095 |
| 3 | ppo | off | 1 | 1 | -4.32 | -0.9941 | -3.801 | — | 5359 |
| 4 | ppo | on | 1 | 1 | -4.32 | -0.9941 | -3.801 | — | 6372 |
| 5 | qr_ppo | off | 1 | 1 | -4.558 | -0.9934 | -3.949 | — | 6554 |
| 6 | qr_ppo | on | 1 | 1 | -4.558 | -0.9934 | -3.949 | — | 6180 |

## Per-seed runs

| algo | constrained | seed | status | p10@5bps | sortino@5bps | median@5bps | max_dd@5bps | sps | reason |
|:---|:---:|:---:|:---|:---|:---|:---|:---|---:|:---|
| ppo | off | 0 | ok | -4.31967 | -0.994093 | -3.80147 | — | 5359 |  |
| ppo | on | 0 | ok | -4.31967 | -0.994093 | -3.80147 | — | 6372 |  |
| sac | off | 0 | ok | -0.580722 | -0.19576 | -0.0601513 | — | 2063 |  |
| sac | on | 0 | ok | -0.580722 | -0.19576 | -0.0601513 | — | 2095 |  |
| qr_ppo | off | 0 | ok | -4.55759 | -0.993439 | -3.94906 | — | 6554 |  |
| qr_ppo | on | 0 | ok | -4.55759 | -0.993439 | -3.94906 | — | 6180 |  |
