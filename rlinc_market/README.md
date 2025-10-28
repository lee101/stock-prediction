rlinc_market
============

Ultra-fast C environment for RL portfolio allocation. It exposes a CPython
extension (`rlinc_cmarket`) with a `MarketEnv` type and a Gymnasium wrapper
(`rlinc_market.RlincMarketEnv`). Observations are the stacked return history
and current weights; actions are raw weights that are L1-clamped to a leverage
budget inside the C core. This design keeps the Python glue minimal while
making the inner loop extremely cheap.

Notes
- CPU implementation first. Vectorization via PufferLib works out-of-the-box
  by wrapping `RlincMarketEnv`.
- CUDA/LibTorch hooks will be added in follow-ups with a C++/CUDA extension
  mirroring the C dynamics for differentiable training.

