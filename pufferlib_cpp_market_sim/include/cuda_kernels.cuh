#pragma once

#include <cuda_runtime.h>
#include <torch/torch.h>

namespace market_sim {
namespace cuda {

// Fused portfolio step kernel - handles PnL, fees, leverage costs, and execution
// This is the hot-path operation that runs every environment step
__global__ void portfolio_step_fused_kernel(
    // Input: Actions and market data
    const float* __restrict__ actions,        // [batch_size] - desired position (-max_lev to +max_lev)
    const float* __restrict__ open_prices,    // [batch_size]
    const float* __restrict__ high_prices,    // [batch_size]
    const float* __restrict__ low_prices,     // [batch_size]
    const float* __restrict__ close_prices,   // [batch_size]

    // Input: Current portfolio state
    const float* __restrict__ cash,           // [batch_size]
    const float* __restrict__ positions,      // [batch_size] - current position size
    const float* __restrict__ entry_prices,   // [batch_size] - entry price for position
    const float* __restrict__ total_pnl,      // [batch_size] - cumulative PnL
    const int* __restrict__ days_held,        // [batch_size] - days position held

    // Input: Execution policy (learned maxdiff-style)
    const float* __restrict__ exec_timing,    // [batch_size] - 0=passive, 1=aggressive
    const float* __restrict__ exec_aggression, // [batch_size] - 0=close, 1=high/low

    // Output: New portfolio state
    float* __restrict__ new_cash,             // [batch_size]
    float* __restrict__ new_positions,        // [batch_size]
    float* __restrict__ new_entry_prices,     // [batch_size]
    float* __restrict__ new_total_pnl,        // [batch_size]
    int* __restrict__ new_days_held,          // [batch_size]

    // Output: Metrics for this step
    float* __restrict__ rewards,              // [batch_size]
    float* __restrict__ fees_paid,            // [batch_size]
    float* __restrict__ leverage_costs,       // [batch_size]
    float* __restrict__ realized_pnl,         // [batch_size]
    float* __restrict__ unrealized_pnl,       // [batch_size]

    // Parameters
    int batch_size,
    float initial_capital,
    float trading_fee_rate,
    float daily_leverage_cost_rate,
    float max_leverage,
    bool is_crypto                            // crypto: no shorts, no leverage
);

// Fast observation assembly kernel
// Combines market features + portfolio state into single observation tensor
__global__ void assemble_observations_kernel(
    // Input: Market features [batch_size, lookback, 5] (OHLCV normalized)
    const float* __restrict__ market_features,

    // Input: Portfolio state [batch_size, 5] (cash, position, pnl, trades, days)
    const float* __restrict__ portfolio_state,

    // Output: Combined observations [batch_size, obs_dim]
    float* __restrict__ observations,

    int batch_size,
    int lookback_window,
    int market_feature_dim,  // 5 for OHLCV
    int portfolio_feature_dim  // 5 for portfolio
);

// V-trace advantage computation (wrapper around pufferlib)
// Note: We'll use pufferlib's optimized kernel directly, but provide a convenience wrapper
void compute_vtrace_advantages(
    torch::Tensor values,      // [num_steps, horizon]
    torch::Tensor rewards,     // [num_steps, horizon]
    torch::Tensor dones,       // [num_steps, horizon]
    torch::Tensor importance,  // [num_steps, horizon]
    torch::Tensor advantages,  // [num_steps, horizon] OUT
    float gamma,
    float lambda,
    float rho_clip,
    float c_clip
);

// Execution price calculation (maxdiff-style learned execution)
__device__ inline float calculate_execution_price(
    float action,              // Desired position change
    float open_price,
    float high_price,
    float low_price,
    float close_price,
    float exec_timing,         // 0 = best price, 1 = worst price
    float exec_aggression      // 0 = use close, 1 = use high/low
) {
    // Determine best and worst prices based on trade direction
    float best_price, worst_price;

    if (action > 0.0f) {
        // Buying: best = low, worst = high
        best_price = low_price;
        worst_price = high_price;
    } else {
        // Selling: best = high, worst = low
        best_price = high_price;
        worst_price = low_price;
    }

    // Interpolate between best and worst based on timing
    float target_price = best_price + exec_timing * (worst_price - best_price);

    // Blend with close price based on aggression
    float final_price = close_price + exec_aggression * (target_price - close_price);

    return final_price;
}

// Helper: Calculate leverage cost
__device__ inline float calculate_leverage_cost(
    float position_size,
    float current_equity,
    float daily_rate
) {
    float equity_base = fabsf(current_equity);
    if (equity_base < 1e-6f) {
        equity_base = 1e-6f;
    }

    // Only pay leverage cost if position > 1.0x equity
    float leverage = fabsf(position_size) / equity_base;
    if (leverage > 1.0f) {
        float excess_leverage = leverage - 1.0f;
        return excess_leverage * equity_base * daily_rate;
    }
    return 0.0f;
}

// Launch wrappers for easy use from C++/LibTorch
void launch_portfolio_step(
    const torch::Tensor& actions,
    const torch::Tensor& open_prices,
    const torch::Tensor& high_prices,
    const torch::Tensor& low_prices,
    const torch::Tensor& close_prices,
    const torch::Tensor& cash,
    const torch::Tensor& positions,
    const torch::Tensor& entry_prices,
    const torch::Tensor& total_pnl,
    const torch::Tensor& days_held,
    const torch::Tensor& exec_timing,
    const torch::Tensor& exec_aggression,
    torch::Tensor& new_cash,
    torch::Tensor& new_positions,
    torch::Tensor& new_entry_prices,
    torch::Tensor& new_total_pnl,
    torch::Tensor& new_days_held,
    torch::Tensor& rewards,
    torch::Tensor& fees_paid,
    torch::Tensor& leverage_costs,
    torch::Tensor& realized_pnl,
    torch::Tensor& unrealized_pnl,
    float initial_capital,
    float trading_fee_rate,
    float daily_leverage_cost_rate,
    float max_leverage,
    bool is_crypto
);

void launch_assemble_observations(
    const torch::Tensor& market_features,
    const torch::Tensor& portfolio_state,
    torch::Tensor& observations
);

} // namespace cuda
} // namespace market_sim
