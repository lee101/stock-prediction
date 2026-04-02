# PufferLib 4.0 C Trading Bot Plan

Date: 2026-04-02

## Current State

- Canonical working env: `.venv`
- Local editable upstream checkout: `PufferLib/` on branch `4.0`
- `pufferlib_market` native binding rebuilds against the local `PufferLib` checkout
- `profile_training.py` now runs on the normalized env and emits:
  - Chrome traces
  - optional py-spy flamegraphs
  - timing metadata from inside the profiled child process

## What The Profile Says

Recent representative runs on RTX 5090 with `hidden_size=1024`, `num_envs=64`,
`rollout_len=256`, `minibatch_size=2048` show:

- quick baseline: about `4789.7` steps/sec
- quick BF16: about `4904.7` steps/sec
- BF16 gain is real but small: about `+2.4%`
- dominant GPU kernel remains `_fused_mlp_relu_kernel` at roughly `53-55%` of CUDA time
- `c_step` is not the main limiter anymore

Conclusion: the current trainer is model-bound, not env-step-bound.

## What Should Stay In `pufferlib_market`

- PPO loop and training orchestration
- fast C vector env for market stepping
- current shared-memory market data loading path
- current rollout collection path while we are still iterating on rewards/actions/obs layout

Reason: this path already has the shortest iteration loop for RL research and the profile
shows most remaining cost is in policy compute, not in the C simulator.

## What Belongs In A Deeper C Migration

Use `ctrader` only after parity is proven and the Python-side policy/training shape is no
longer the main bottleneck.

`ctrader` is the right candidate for:

- deterministic execution simulator parity
- order fill / slippage / borrow / portfolio accounting in C
- eventual standalone inference/execution runtime
- future live-trading execution service separation from training

It is not the next highest-leverage place to chase speed for PPO training.

## Near-Term Optimization Order

1. Keep `.venv` as the canonical PufferLib 4.0 training env.
2. Profile with the updated `profile_training.py` before each major trainer change.
3. Focus next on model-side changes:
   - hidden size
   - architecture choice
   - minibatch geometry
   - BF16 / compile / fused head usage
4. Revisit `ctrader` migration only after the model path is no longer dominant.

## C Trading Bot Target Architecture

Short term:

- C env stepper in `pufferlib_market`
- PyTorch policy on GPU
- Python PPO trainer

Medium term:

- `ctrader` C market simulator achieves feature parity with `pufferlib_market`
- shared dataset / reward / fill semantics across both paths
- policy inference can call into a C execution backend without changing training semantics

Long term:

- standalone C/CUDA/libtorch inference service for daily stock trading
- optional separation:
  - training stack remains `pufferlib_market`
  - production execution stack moves to `ctrader` + libtorch

## Decision Rule

Do not move more logic into C just because it is possible.

Move logic into C when one of these is true:

- profiler shows the path is materially hot
- the logic is stable enough that parity testing is tractable
- the code is execution-critical and will be reused outside research training
