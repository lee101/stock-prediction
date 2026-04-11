# fp4 sweep leaderboard (2026-04-08T10:02:31)

- mode: full
- steps/cell: 2000000
- cells: 6  total runs: 30
- sort key: (p10@5bps DESC, sortino@5bps DESC, max_dd ASC)

| rank | algo | constrained | seeds | ok | p10@5bps | sortino@5bps | median@5bps | max_dd@5bps | sps |
|---:|:---|:---:|:---:|:---:|:---|:---|:---|:---|---:|
| 1 | sac | off | 5 | 2 | +0 | -0.07871 ± 0.000461 | -0.004627 ± 0.000215 | — | 929 |
| 2 | qr_ppo | on | 5 | 3 | -3.706 ± 0.54 | -0.3636 ± 0.0128 | -0.9287 ± 0.0495 | — | 6518 |
| 3 | ppo | off | 5 | 5 | -4.5 ± 1.2 | -0.4537 ± 0.0196 | -1.297 ± 0.105 | — | 10978 |
| 4 | ppo | on | 5 | 5 | -4.5 ± 1.2 | -0.4537 ± 0.0196 | -1.297 ± 0.105 | — | 10849 |
| 5 | sac | on | 5 | 0 | — | — | — | — | 0 |
| 6 | qr_ppo | off | 5 | 0 | — | — | — | — | 0 |

## Per-seed runs

| algo | constrained | seed | status | p10@5bps | sortino@5bps | median@5bps | max_dd@5bps | sps | reason |
|:---|:---:|:---:|:---|:---|:---|:---|:---|---:|:---|
| ppo | off | 0 | ok | -4.02884 | -0.442149 | -1.2387 | — | 11030 |  |
| ppo | off | 1 | ok | -3.82595 | -0.486312 | -1.31673 | — | 11413 |  |
| ppo | off | 2 | ok | -4.17832 | -0.45648 | -1.23785 | — | 10569 |  |
| ppo | off | 3 | ok | -3.8359 | -0.44683 | -1.21728 | — | 10959 |  |
| ppo | off | 4 | ok | -6.63088 | -0.43674 | -1.47264 | — | 10920 |  |
| ppo | on | 0 | ok | -4.02884 | -0.442149 | -1.2387 | — | 10941 |  |
| ppo | on | 1 | ok | -3.82595 | -0.486312 | -1.31673 | — | 10943 |  |
| ppo | on | 2 | ok | -4.17832 | -0.45648 | -1.23785 | — | 11155 |  |
| ppo | on | 3 | ok | -3.8359 | -0.44683 | -1.21728 | — | 11334 |  |
| ppo | on | 4 | ok | -6.63088 | -0.43674 | -1.47264 | — | 9873 |  |
| sac | off | 0 | ok | 0 | -0.0790328 | -0.00447532 | — | 2325 |  |
| sac | off | 1 | ok | 0 | -0.0783803 | -0.00477922 | — | 2321 |  |
| sac | off | 2 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| sac | off | 3 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| sac | off | 4 | error | — | — | — | — | 0 | OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a t |
| sac | on | 0 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| sac | on | 1 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| sac | on | 2 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| sac | on | 3 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| sac | on | 4 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| qr_ppo | off | 0 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| qr_ppo | off | 1 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| qr_ppo | off | 2 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| qr_ppo | off | 3 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| qr_ppo | off | 4 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| qr_ppo | on | 0 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| qr_ppo | on | 1 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| qr_ppo | on | 2 | ok | -4.26056 | -0.360765 | -0.970714 | — | 10247 |  |
| qr_ppo | on | 3 | ok | -3.67409 | -0.352486 | -0.87405 | — | 10787 |  |
| qr_ppo | on | 4 | ok | -3.18274 | -0.377679 | -0.941271 | — | 11558 |  |
