# fp4 sweep leaderboard (2026-04-08T00:09:56)

- mode: full
- steps/cell: 2000000
- cells: 6  total runs: 30
- sort key: (p10@5bps DESC, sortino@5bps DESC, max_dd ASC)

| rank | algo | constrained | seeds | ok | p10@5bps | sortino@5bps | median@5bps | max_dd@5bps | sps |
|---:|:---|:---:|:---:|:---:|:---|:---|:---|:---|---:|
| 1 | ppo | off | 5 | 3 | -4.011 ± 0.177 | -0.4616 ± 0.0225 | -1.264 ± 0.0453 | — | 6584 |
| 2 | ppo | on | 5 | 0 | — | — | — | — | 0 |
| 3 | sac | off | 5 | 0 | — | — | — | — | 0 |
| 4 | sac | on | 5 | 0 | — | — | — | — | 0 |
| 5 | qr_ppo | off | 5 | 0 | — | — | — | — | 0 |
| 6 | qr_ppo | on | 5 | 0 | — | — | — | — | 0 |

## Per-seed runs

| algo | constrained | seed | status | p10@5bps | sortino@5bps | median@5bps | max_dd@5bps | sps | reason |
|:---|:---:|:---:|:---|:---|:---|:---|:---|---:|:---|
| ppo | off | 0 | ok | -4.02884 | -0.442149 | -1.2387 | — | 11200 |  |
| ppo | off | 1 | ok | -3.82595 | -0.486312 | -1.31673 | — | 10862 |  |
| ppo | off | 2 | ok | -4.17832 | -0.45648 | -1.23785 | — | 10858 |  |
| ppo | off | 3 | error | — | — | — | — | 0 | OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a t |
| ppo | off | 4 | error | — | — | — | — | 0 | OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a t |
| ppo | on | 0 | error | — | — | — | — | 0 | OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a t |
| ppo | on | 1 | error | — | — | — | — | 0 | OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a t |
| ppo | on | 2 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| ppo | on | 3 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| ppo | on | 4 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| sac | off | 0 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| sac | off | 1 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| sac | off | 2 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| sac | off | 3 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| sac | off | 4 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
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
| qr_ppo | on | 2 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| qr_ppo | on | 3 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
| qr_ppo | on | 4 | error | — | — | — | — | 0 | AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocatio |
