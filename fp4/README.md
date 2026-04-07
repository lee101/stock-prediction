# fp4 — NVFP4 RL Trading Library (Blackwell SM120)

NVFP4 (E2M1, NVIDIA's 4-bit float) training stack for RL trading policies on
RTX 5090 / Blackwell. Pairs a GPU-resident C/CUDA market environment
(`gpu_trading_env`) with NVFP4 linear layers, constrained-MDP losses, and
PPO/SAC/distributional trainers under CUDA-graph capture. Designed as the
single source of truth for fills/fees/leverage/liquidation so training and
the production marketsim agree.

## Architecture

```
                        +-------------------------+
                        |  market data (SoA)      |
                        |  open/high/low/close    |
                        +-----------+-------------+
                                    |
                                    v
        +----------------------------------------------------+
        |  gpu_trading_env (C++/CUDA, sm_120, fused step)    |
        |  - 5bps buffered limit fill, fee=10bps             |
        |  - leverage cap 5x, maintenance margin, liquidate  |
        |  - log-equity reward + cost[B,C] for constraints   |
        +-----------+----------------------------------------+
                    |  obs / reward / done / cost (CUDA tensors, no host sync)
                    v
        +----------------------------------------------------+
        |  TwoLayerPolicy (fp4/policy_two_layer.py)          |
        |   shared encoder  -> NVFP4Linear / nn.Linear       |
        |   Layer A (slow): inventory, leverage, risk_budget |
        |   Layer B (fast): bid/ask offset, size_frac        |
        |   value head + cost-value head(s)  (BF16)          |
        +-----------+----------------------------------------+
                    |
                    v
        +----------------------------------------------------+
        |  Constrained PPO / SAC / QR-PPO trainer            |
        |  - Lagrangian dual ascent on constraint multipliers|
        |  - CVaR tail penalty + smooth-PnL regularizer      |
        |  - GPU rollout buffer + GAE on device              |
        +-----------+----------------------------------------+
                    |
                    v
        +----------------------------------------------------+
        |  CUDA-graph capture (fp4/cuda_graph.py)            |
        |  rollout step + update step replay, no host sync   |
        +-----------+----------------------------------------+
                    |
                    v
        +----------------------------------------------------+
        |  eval @ slippage {0,5,10,20} bps                   |
        |  binary-fill marketsim ground truth (lag>=2)       |
        +----------------------------------------------------+
```

## Performance summary

| Metric                              | Value                  | Source commit |
|------------------------------------|------------------------|---------------|
| NVFP4 GEMM (4096^3) vs BF16        | **2.24x** (206 TFLOPS) | `f9806fe0`    |
| NVFP4 GEMM (8192^3) vs BF16        | 2.17x (231 TFLOPS)     | `f9806fe0`    |
| Graph-safe NVFP4Linear             | 5159 sps (+62%)        | `d8eb1a5b`    |
| GPUVecEnv synthetic (cuda graph)   | **36.0M env-steps/s**  | Unit G `00789411` |
| GPUVecEnv synthetic (eager + policy)| 4.55M sps             | Unit G        |
| gpu_trading_env real (SM120)       | fused SoA step kernel  | `15cdd8b9`    |
| Sweep leaderboard winner @5bps     | pufferlib_bf16 (smoke) | `37add767`    |

Smoke sweep ranks (50k-step seeds, p10@5bps DESC):

| rank | algo            | constrained | seeds | p10@5bps | sortino@5bps | max_dd |
|-----:|-----------------|:-----------:|:-----:|---------:|-------------:|-------:|
| 1    | pufferlib_bf16  | off         | 2     | -0.160   | -0.583       | 0.121  |
| 2    | hf_trainer      | off         | 2     | -0.210   | -0.172       | 0.175  |

> No cell beat the production 32-model ensemble bar (p10@5bps = 0.662). The
> sweep was a 50k-step smoke run; full evaluation needs 2M-100M steps.

## Phase commit history

| Phase | Commit     | Description |
|------:|------------|-------------|
| 1     | `c496a8c1` | Initial NVFP4 reference library + tests (8/8) |
| 1     | `7737b8d4` | marketsim DPS action mode (5x leverage), SCALAR bit-identical |
| 1     | `7d5229f0` | bench: trainer comparison harness scaffold |
| 2-A   | `e63485d8` | GPU-resident PPO trainer (NVFP4 policy, CUDA-graph update) |
| 2-B   | `433481cd` | CUTLASS NVFP4 GEMM torch extension (lazy build, BF16 fallback) |
| 2-C   | `3a1a29c2` | market_sim_py pybind11 bindings (SCALAR+DPS torch tensors) |
| 2-D   | `39f504ed` | DPS PnL accounted at fill limit price |
| 2-E   | `2911ad6f` | HF Trainer adapter for marketsim PPO |
| 2-F   | `99bac9c9` | TRL PPOTrainer adapter for marketsim |
| 2-G   | `00789411` | GPU-resident vec env + CUDA-graph capture |
| 3     | `f9806fe0` | bench_gemm uses real CUTLASS NVFP4 (2.24x BF16) |
| 3     | `d8eb1a5b` | graph-safe NVFP4Linear (+62% sps) |
| 3     | `57a0879b` | generic policy evaluator + first comparison run |
| 4-1   | `15cdd8b9` | gpu_trading_env SoA fused env_step CUDA kernel (SM120) |
| 4-2   | `286ce893` | Two-timescale Layer A/B policy + cost-value heads |
| 4-3   | `9c565a07` | Constrained-MDP losses + Lagrangian multiplier |
| 4-4   | `56f10cfb` | SAC trainer (twin-Q, GPU replay, auto-temp) |
| 4-5   | `7d21c3e8` | Distributional QR-PPO trainer (CVaR from quantiles) |
| 4-6   | `9bfc4032` | Multi-seed sweep harness (algo x constrained x seeds) |
| 4-7   | `79fe2aca` | Unify .venv (py3.13) for fp4 + market_sim_py + triton |
| 5     | `37add767` | Algorithm leaderboard report + production recommendation |

## Quickstart

```bash
source .venv/bin/activate
uv pip install -e fp4/
uv pip install -e gpu_trading_env/    # Blackwell only

# microbench: NVFP4 vs BF16 vs FP8 GEMM
python fp4/bench/bench_gemm.py

# SPS bench (env + policy + cuda graph)
python fp4/bench/bench_sps.py

# single-seed smoke training
python fp4/bench/bench_trading.py --trainer fp4 --steps 50000 --seed 0

# multi-seed algorithm sweep -> leaderboard md
python fp4/bench/sweep.py --algos ppo,sac,qr_ppo --constrained on,off --seeds 0,1,2

# unit + integration tests
pytest fp4/tests/ -x -q
```

## File map

```
fp4/
  fp4/
    dtypes.py            NVFP4 (E2M1) bit layout, E4M3 block scale, FP32 tensor scale
    quant.py             2D 16x16 block quant, stochastic rounding
    hadamard.py          Random Hadamard transform (outlier suppression)
    linear.py            NVFP4Linear (graph-safe, preallocated buffers)
    layers.py            NVFP4MLP / Attention / TransformerBlock
    autocast.py          Layer-precision selection (BF16 stays for embed/head/LN)
    optim.py             AdamW with FP32 master weights + BF16 momentum
    policy.py            Single-head actor-critic (PPO baseline)
    policy_two_layer.py  Two-timescale Layer A/B policy + cost-value heads
    losses.py            ppo / value / entropy / CVaR / smooth-PnL
    lagrangian.py        Constraint multiplier dual ascent
    distributional.py    Quantile value head + CVaR
    trainer.py           PPO trainer (CUDA-graph capture)
    trainer_sac.py       SAC twin-Q trainer + GPU replay
    trainer_qr.py        Quantile-regression PPO
    vec_env.py           GPU-resident vector env (real or synthetic)
    cuda_graph.py        Graph capture/replay helper
    replay.py            On-device replay buffer
    kernels/
      gemm.py            CUTLASS NVFP4 GEMM wrapper (BF16 fallback)
      gemm_kernel.cu     CUTLASS example 79b binding
  bench/
    bench_gemm.py        NVFP4 vs BF16 vs FP8 TFLOPS
    bench_sps.py         env-steps/sec across {eager, +policy, cuda graph}
    bench_trading.py     single-trainer training run + eval@slippage
    sweep.py             multi-seed algo x constrained sweep -> leaderboard.md
    compare_trainers.py  pufferlib / hf / trl / fp4 head-to-head
    adapters/            hf_adapter.py, trl_adapter.py
  tests/                 quant, hadamard, linear, trainer, vec_env, losses
  experiments/           fp4_ppo_stocks12.yaml, fp4_ppo_crypto12.yaml
gpu_trading_env/         C++/CUDA fused env_step (SM120) + python bindings
```

See `bench/results/RECOMMENDATION.md` for the production-stack recommendation.
