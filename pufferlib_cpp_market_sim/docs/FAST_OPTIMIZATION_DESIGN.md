# Ultra-Fast C++ Market Simulation with PufferLib3 Optimizations

## Overview
This document outlines the design for an insanely fast C++ market simulator using PufferLib3-style optimizations, CUDA kernels, and learned maxdiff-style execution strategies.

## Architecture Goals

### Performance Targets
- **Throughput**: >500K steps/second (5x improvement over current)
- **Latency**: <2ms per batch of 4096 environments
- **Memory**: <8GB VRAM for full training
- **Training Speed**: Achieve high training PnL, lower validation PnL (proper generalization)

### Key Optimizations

#### 1. CUDA Kernel Integration
**Current Problem**: All market operations happen in LibTorch, which has overhead
**Solution**: Custom CUDA kernels for hot-path operations

```cpp
// Portfolio update kernel - fused PnL + fees + leverage costs
__global__ void portfolio_step_kernel(
    const float* __restrict__ actions,        // [batch, 1]
    const float* __restrict__ prices,         // [batch, 4] OHLC
    const float* __restrict__ positions,      // [batch]
    const float* __restrict__ cash,           // [batch]
    const float* __restrict__ entry_prices,   // [batch]
    float* __restrict__ new_positions,        // [batch] OUT
    float* __restrict__ new_cash,             // [batch] OUT
    float* __restrict__ new_entry_prices,     // [batch] OUT
    float* __restrict__ rewards,              // [batch] OUT
    float* __restrict__ fees_paid,            // [batch] OUT
    int batch_size,
    float initial_capital,
    float trading_fee,
    float leverage_cost,
    float max_leverage,
    bool use_high_low                         // learned execution flag
);
```

**Benefits**:
- Fused operations reduce memory bandwidth
- Coalesced memory access patterns
- No Python/LibTorch overhead
- ~10-20x faster than LibTorch equivalent

#### 2. Structure of Arrays (SoA) Memory Layout
**Current Problem**: Array of Structures causes scattered memory access
**Solution**: Flat, contiguous arrays for each field

```cpp
struct MarketStateSoA {
    // All arrays are [num_symbols * max_timesteps]
    float* open;
    float* high;
    float* low;
    float* close;
    float* volume;

    // Metadata
    int* symbol_lengths;        // [num_symbols]
    int num_symbols;
    int max_timesteps;
};

struct PortfolioSoA {
    // All arrays are [batch_size]
    float* cash;
    float* positions;
    float* entry_prices;
    float* total_pnl;
    float* realized_pnl;
    float* unrealized_pnl;
    float* trading_costs;
    float* leverage_costs;
    int* num_trades;
    int* days_held;

    int batch_size;
};
```

**Benefits**:
- Perfect GPU memory coalescing
- SIMD-friendly on CPU
- Cache-efficient
- Easy to slice/batch

#### 3. PufferLib3 V-trace Integration
**Current Problem**: Manual advantage computation in training loop
**Solution**: Use pufferlib's optimized CUDA V-trace kernel

```cpp
// Use pufferlib's fast V-trace
#include <pufferlib/extensions/puffernet.h>

void compute_advantages_fast(
    torch::Tensor values,      // [num_steps, horizon]
    torch::Tensor rewards,     // [num_steps, horizon]
    torch::Tensor dones,       // [num_steps, horizon]
    torch::Tensor importance,  // [num_steps, horizon]
    torch::Tensor advantages,  // [num_steps, horizon] OUT
    float gamma = 0.99f,
    float lambda = 0.95f,
    float rho_clip = 1.0f,
    float c_clip = 1.0f
) {
    // Call pufferlib's optimized CUDA kernel
    torch::ops::pufferlib::compute_puff_advantage(
        values, rewards, dones, importance, advantages,
        gamma, lambda, rho_clip, c_clip
    );
}
```

**Benefits**:
- 100x faster than Python GAE
- Already optimized for GPU
- Handles importance sampling correctly

#### 4. Learned Maxdiff-Style Execution
**Current Problem**: Fixed execution at open/close
**Solution**: Neural network learns optimal execution timing

```cpp
struct ExecutionPolicyNet {
    // Input: [batch, obs_dim + 2]  (obs + action + prices)
    // Output: [batch, 2]  (execution_timing, execution_aggression)

    // execution_timing: 0.0 = buy at low/sell at high (passive)
    //                  1.0 = buy at high/sell at low (aggressive)

    // execution_aggression: 0.0 = use close price (safe)
    //                      1.0 = use high/low (maxdiff style)
};

// Learned execution function
__device__ float get_execution_price(
    float action,              // Desired position change
    float open, float high, float low, float close,
    float exec_timing,         // From policy network
    float exec_aggression      // From policy network
) {
    if (action > 0) {
        // Buying: interpolate between low (best) and high (worst)
        float best_price = low;
        float worst_price = high;
        float target = best_price + exec_timing * (worst_price - best_price);
        return close + exec_aggression * (target - close);
    } else {
        // Selling: interpolate between high (best) and low (worst)
        float best_price = high;
        float worst_price = low;
        float target = best_price + exec_timing * (worst_price - best_price);
        return close + exec_aggression * (target - close);
    }
}
```

**Training Strategy**:
- During training: Use aggressive execution (high PnL, unrealistic)
- During validation: Use conservative execution (lower PnL, realistic)
- Network learns to exploit maxdiff when available
- Separate reward shaping for execution quality

#### 5. Multi-Strategy Vectorization
**Current Problem**: Training one strategy at a time
**Solution**: Train multiple strategies in parallel with different hyperparameters

```cpp
struct StrategyConfig {
    float risk_tolerance;      // 0.0 (conservative) to 1.0 (aggressive)
    float holding_period;      // Preferred days to hold
    float execution_style;     // 0.0 (passive) to 1.0 (aggressive)
    float fee_sensitivity;     // Penalty weight for trading costs
};

struct MultiStrategyEnv {
    // Each env gets different strategy config
    StrategyConfig* strategy_configs;  // [batch_size]

    // Stratified sampling: ensure diversity
    void randomize_strategies() {
        for (int i = 0; i < batch_size; i++) {
            strategy_configs[i] = {
                .risk_tolerance = random_uniform(0.0, 1.0),
                .holding_period = random_uniform(1.0, 30.0),
                .execution_style = random_uniform(0.0, 1.0),
                .fee_sensitivity = random_uniform(0.1, 10.0)
            };
        }
    }
};
```

**Benefits**:
- Explore strategy space efficiently
- Natural curriculum learning (easy to hard strategies)
- Population-based training
- Better generalization

#### 6. Realistic Daily Trading Simulation
**Current Problem**: Continuous trading at any timestep
**Solution**: Proper daily bar simulation with realistic constraints

```cpp
struct DailyTradingEnv {
    // Episode = 1 trading day
    // Agent makes decision once per day at market close
    // Execution happens next day using learned execution policy

    struct DailyState {
        // Yesterday's decision
        float yesterday_action;

        // Today's market data
        float today_open, today_high, today_low, today_close;

        // Execution result
        float executed_price;     // Learned from OHLC
        float slippage;           // Difference from close

        // Portfolio state
        float position_before;
        float position_after;
        float pnl_today;
    };

    void step(float action) {
        // 1. Determine execution price using learned policy
        auto [exec_timing, exec_aggression] = execution_policy(obs, action);
        float exec_price = get_execution_price(
            action, open, high, low, close,
            exec_timing, exec_aggression
        );

        // 2. Calculate slippage and costs
        float slippage = exec_price - close;
        float trade_size = abs(action - current_position);
        float fees = trade_size * trading_fee_rate;

        // 3. Update portfolio
        execute_trade(action, exec_price, fees);

        // 4. Move to next day
        current_day++;
    }
};
```

#### 7. Optimized Training Loop
**Current Problem**: Python-like training loop in C++
**Solution**: Fully vectorized, kernel-fused training

```cpp
class FastTrainingLoop {
public:
    void train(int num_updates) {
        // Pre-allocate all buffers on GPU
        RolloutBuffers buffers(batch_size, rollout_length, obs_dim);

        for (int update = 0; update < num_updates; update++) {
            // 1. Collect rollout (fully on GPU)
            collect_rollout_gpu(buffers);  // Custom CUDA kernel

            // 2. Compute advantages (pufferlib CUDA)
            compute_advantages_fast(
                buffers.values, buffers.rewards,
                buffers.dones, buffers.importance,
                buffers.advantages
            );

            // 3. PPO update (LibTorch)
            ppo_update_batch(buffers);

            // 4. Log metrics (async, non-blocking)
            if (update % log_interval == 0) {
                async_log_metrics(buffers);
            }
        }
    }

private:
    // Single kernel that collects full rollout
    void collect_rollout_gpu(RolloutBuffers& buffers) {
        // Launch kernel that does:
        // for t in range(rollout_length):
        //   - Get observations
        //   - Run policy network
        //   - Sample actions
        //   - Step environments (portfolio kernel)
        //   - Store to buffers
        // All without CPU/GPU sync!
    }
};
```

## Implementation Plan

### Phase 1: CUDA Kernels (Week 1)
- [ ] Portfolio step kernel (fused PnL + fees + leverage)
- [ ] Observation assembly kernel (flatten OHLCV + portfolio state)
- [ ] Execution price calculation kernel
- [ ] Benchmark vs LibTorch baseline

### Phase 2: V-trace Integration (Week 1)
- [ ] Link pufferlib CUDA extension
- [ ] Replace manual GAE with pufferlib V-trace
- [ ] Verify numerical correctness
- [ ] Benchmark speedup

### Phase 3: Learned Execution (Week 2)
- [ ] Implement execution policy network
- [ ] Train with reward shaping for execution quality
- [ ] Validate on historical data
- [ ] Measure train vs validation PnL gap

### Phase 4: Multi-Strategy (Week 2)
- [ ] Implement strategy config sampling
- [ ] Add strategy-conditional policy network
- [ ] Train population of strategies
- [ ] Analyze strategy diversity

### Phase 5: Daily Trading Sim (Week 3)
- [ ] Refactor to daily episodes
- [ ] Add realistic execution constraints
- [ ] Validate against known trading strategies
- [ ] Backtest on real data

### Phase 6: Full Integration (Week 3)
- [ ] Combine all optimizations
- [ ] End-to-end benchmarking
- [ ] Production-ready training script
- [ ] Documentation and examples

## Expected Performance Gains

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| Portfolio Step | 0.5ms | 0.05ms | 10x |
| Advantage Computation | 2.0ms | 0.02ms | 100x |
| Full Rollout | 10ms | 1ms | 10x |
| PPO Update | 5ms | 3ms | 1.7x |
| **Total Training Loop** | **17ms** | **4ms** | **4.25x** |

**Throughput**: 4096 envs / 4ms = ~1M steps/second

## Validation Strategy

### Training Metrics
- High PnL with aggressive execution
- Low transaction costs
- High Sharpe ratio
- Diverse strategies

### Validation Metrics
- Realistic PnL (should be lower than training)
- Robust to different market conditions
- Consistent across multiple symbols
- Transaction cost awareness

### Testing Protocol
1. Train on 80% of data with aggressive execution
2. Validate on 20% with conservative execution
3. Measure PnL degradation (expect 30-50% drop)
4. Backtest on completely held-out data
5. Compare to baseline strategies (buy-and-hold, momentum, mean-reversion)

## Files to Create/Modify

### New Files
- `include/cuda_kernels.cuh` - CUDA kernel declarations
- `src/cuda_kernels.cu` - CUDA kernel implementations
- `include/execution_policy.h` - Learned execution network
- `src/execution_policy.cpp` - Execution policy implementation
- `include/fast_training.h` - Optimized training loop
- `src/fast_training.cpp` - Training implementation
- `include/strategy_config.h` - Multi-strategy configuration

### Modified Files
- `CMakeLists.txt` - Add CUDA compilation, link pufferlib
- `include/portfolio.h` - Add SoA layout, CUDA support
- `src/portfolio.cpp` - Refactor to use CUDA kernels
- `include/market_env.h` - Add daily trading mode
- `src/market_env.cpp` - Implement daily episodes
- `src/main_train.cpp` - Use fast training loop

## Next Steps
1. Start with CUDA kernels for portfolio operations
2. Integrate pufferlib V-trace
3. Add execution policy network
4. Build multi-strategy training
5. Validate and benchmark
