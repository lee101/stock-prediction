# Fast Market Trading Simulation with C/CUDA

High-performance market trading environment for reinforcement learning using C, CUDA, and libtorch.

## Overview

This is a blazingly fast market simulation environment designed for training RL agents on trading strategies. Key features:

- **C-based environment core** following pufferlib architecture for minimal Python overhead
- **CUDA-accelerated RL computations** (GAE, portfolio calculations, technical indicators)
- **libtorch integration** for seamless PyTorch interop
- **Vectorized environments** for parallel agent training
- **Sub-millisecond step times** for millions of environment interactions

## Architecture

```
market_sim_c/
├── include/
│   └── market_env.h          # C environment interface
├── src/
│   ├── market_env.c          # Core market simulation logic
│   ├── binding.c             # Python bindings (pufferlib-style)
│   └── cuda_kernels.cu       # CUDA-accelerated RL kernels
├── python/
│   └── market_sim.py         # Python interface
├── examples/
│   ├── train_ppo.py          # PPO training example
│   └── benchmark.py          # Performance benchmarks
├── CMakeLists.txt            # Build configuration
└── Makefile                  # Build convenience scripts
```

## Key Components

### 1. C Environment Core (`market_env.c`)

Fast market simulation with:
- Multi-asset portfolio management
- Transaction costs and position limits
- Synthetic price generation (replace with real data)
- Parallel agent execution

### 2. CUDA Kernels (`cuda_kernels.cu`)

GPU-accelerated operations:
- `compute_portfolio_values`: Parallel portfolio valuation across agents
- `compute_gae`: Generalized Advantage Estimation on GPU
- `compute_rsi`: Technical indicators (RSI, moving averages, etc.)
- `compute_sharpe_ratios`: Risk-adjusted performance metrics

### 3. Python Interface

Clean Gymnasium-compatible API:
```python
from market_sim_c.python.market_sim import make_env, MarketSimConfig

config = MarketSimConfig(
    num_assets=10,
    num_agents=64,  # Vectorized environments
    max_steps=1000,
)

env = make_env(config)
obs, info = env.reset()
obs, reward, done, truncated, info = env.step(action)
```

## Installation

### Prerequisites

```bash
# Ensure you have:
# - CUDA Toolkit (for GPU acceleration)
# - CMake >= 3.18
# - libtorch (already in ../external/libtorch)
# - Python 3.8+
# - PyTorch with CUDA support

# Install Python dependencies
uv pip install torch pufferlib numpy
```

### Build

```bash
# Quick build
make build

# Or with CMake directly
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=../external/libtorch/libtorch ..
make -j$(nproc)

# Install Python bindings
make install
```

### Debug Build

For development with sanitizers:
```bash
make build-debug
```

## Usage

### Basic Training

```python
from market_sim_c.python.market_sim import make_env, MarketSimConfig
from market_sim_c.examples.train_ppo import PPOTrainer, MarketPPONetwork

# Configure environment
config = MarketSimConfig(
    num_assets=10,
    num_agents=64,
    max_steps=1000,
    transaction_cost=0.001,
)

# Create environment
env = make_env(config)

# Create network
network = MarketPPONetwork(
    obs_size=env.observation_space.shape[0],
    action_size=env.action_space.n,
)

# Train
trainer = PPOTrainer(env, network, device="cuda")
trainer.train(total_timesteps=10_000_000)
```

### Running the Example

```bash
# Run PPO training example
python examples/train_ppo.py

# Run benchmarks
python examples/benchmark.py
```

## Performance

Expected performance on modern hardware:

| Configuration | Steps/Second | GPU Utilization |
|--------------|--------------|-----------------|
| 1 agent, CPU | ~50,000 | - |
| 64 agents, CPU | ~200,000 | - |
| 64 agents, GPU | ~1,000,000+ | 60-80% |
| 256 agents, GPU | ~2,000,000+ | 80-95% |

## Action Space

```
0: Hold (no action)
1 to N: Buy asset i (where i = action - 1)
N+1 to 2N: Sell asset i (where i = action - N - 1)
```

## Observation Space

Concatenated vector containing:
- **Price history**: OHLCV data for lookback window (5 × 256 = 1280 features)
- **Portfolio state**: Current positions in each asset (N features)
- **Cash**: Available cash (1 feature, normalized)
- **Portfolio value**: Total portfolio value (1 feature, normalized)
- **Step progress**: Current step / max_steps (1 feature)

Total: 1280 + N + 3 features

## Reward

```python
reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value * 100
```

Percentage return scaled by 100 for better learning dynamics.

## References

### pufferlib Architecture

This implementation follows pufferlib's design:
- C environment core for speed
- Minimal Python overhead
- Vectorized parallel execution
- Compatible with standard RL libraries

See: https://github.com/PufferAI/PufferLib

### CUDA Acceleration

Key optimizations:
1. **Parallel GAE computation**: Each agent's advantages computed in separate thread
2. **Fused operations**: Reduce memory transfers between CPU/GPU
3. **Coalesced memory access**: Optimal GPU memory patterns
4. **Shared memory**: For reductions and technical indicators

### Integration with Real Data

To use real market data:

1. Replace `update_prices()` in `market_env.c` with data loader
2. Preload price data into shared memory
3. Index into data array based on `current_step`

Example:
```c
void update_prices(MarketEnv* env) {
    // Load from pre-allocated data buffer
    int idx = env->current_step % env->data_length;
    for (int i = 0; i < env->num_assets; i++) {
        asset->current_price = env->price_data[idx * env->num_assets + i];
        // Update OHLCV from data...
    }
}
```

## Future Enhancements

- [ ] Order book simulation with market depth
- [ ] Multi-exchange support
- [ ] Slippage modeling
- [ ] More technical indicators (MACD, Bollinger Bands)
- [ ] Multi-agent competitive scenarios
- [ ] Real-time data streaming
- [ ] Transformer-based policy networks
- [ ] Multi-GPU training with data parallelism

## Troubleshooting

### Build Issues

```bash
# Check environment
make check

# Clean rebuild
make clean && make build

# Verify libtorch path
ls -la ../external/libtorch/libtorch/lib/
```

### CUDA Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version compatibility
nvcc --version
```

### Runtime Errors

```bash
# Run with debug build and sanitizers
make build-debug
python examples/train_ppo.py
```

## Contributing

This is a research/experimental codebase. Feel free to:
- Add new technical indicators
- Improve price simulation models
- Optimize CUDA kernels
- Add more RL algorithms

## License

MIT - See project root for details
