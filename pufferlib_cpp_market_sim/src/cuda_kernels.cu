#include "cuda_kernels.cuh"
#include <stdio.h>
#include <vector>

namespace market_sim {
namespace cuda {

// Constants
constexpr int THREADS_PER_BLOCK = 256;

// Fused portfolio step kernel - this is the hot path!
__global__ void portfolio_step_fused_kernel(
    const float* __restrict__ actions,
    const float* __restrict__ open_prices,
    const float* __restrict__ high_prices,
    const float* __restrict__ low_prices,
    const float* __restrict__ close_prices,
    const float* __restrict__ cash,
    const float* __restrict__ positions,
    const float* __restrict__ entry_prices,
    const float* __restrict__ total_pnl,
    const int* __restrict__ days_held,
    const float* __restrict__ exec_timing,
    const float* __restrict__ exec_aggression,
    float* __restrict__ new_cash,
    float* __restrict__ new_positions,
    float* __restrict__ new_entry_prices,
    float* __restrict__ new_total_pnl,
    int* __restrict__ new_days_held,
    float* __restrict__ rewards,
    float* __restrict__ fees_paid,
    float* __restrict__ leverage_costs,
    float* __restrict__ realized_pnl,
    float* __restrict__ unrealized_pnl,
    int batch_size,
    float initial_capital,
    float trading_fee_rate,
    float daily_leverage_cost_rate,
    float max_leverage,
    bool is_crypto
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Load current state
    float action = actions[idx];
    float open = open_prices[idx];
    float high = high_prices[idx];
    float low = low_prices[idx];
    float close = close_prices[idx];
    float current_cash = cash[idx];
    float current_position = positions[idx];
    float entry_price = entry_prices[idx];
    float current_pnl = total_pnl[idx];
    int current_days = days_held[idx];
    float current_equity = current_cash + current_position;

    // Clamp action to valid range
    float max_lev = is_crypto ? 1.0f : max_leverage;
    action = fmaxf(-max_lev, fminf(max_lev, action));

    // Crypto constraint: no short selling
    if (is_crypto && action < 0.0f) {
        action = 0.0f;
    }

    // Calculate desired position size based on live equity
    float desired_position = action * current_equity;
    float position_change = desired_position - current_position;

    // Determine execution price using learned policy
    float execution_price = calculate_execution_price(
        position_change,
        open, high, low, close,
        exec_timing[idx],
        exec_aggression[idx]
    );

    // Calculate trading fee
    float trade_value = fabsf(position_change);
    float fee = trade_value * trading_fee_rate;

    // Calculate leverage cost for current position (before trade)
    float lev_cost = calculate_leverage_cost(
        current_position, current_equity, daily_leverage_cost_rate
    );

    // Calculate unrealized PnL on current position
    float position_pnl = 0.0f;
    if (fabsf(current_position) > 1e-6f) {
        // PnL = position * (current_price - entry_price) / entry_price
        float price_change_pct = (close - entry_price) / entry_price;
        position_pnl = current_position * price_change_pct;
    }

    // Calculate realized PnL if closing/reducing position
    float step_realized_pnl = 0.0f;
    float new_pos = current_position + position_change;

    // Check if we're closing or reversing position
    bool is_closing = (current_position > 0.0f && new_pos < current_position) ||
                      (current_position < 0.0f && new_pos > current_position);

    if (is_closing && fabsf(current_position) > 1e-6f) {
        // Realize PnL on the portion being closed
        float close_fraction = fabsf(position_change) / fabsf(current_position);
        close_fraction = fminf(close_fraction, 1.0f);
        step_realized_pnl = position_pnl * close_fraction;
    }

    // Update cash (subtract fees and leverage costs, add realized PnL)
    float updated_cash = current_cash - fee - lev_cost + step_realized_pnl;

    // Update position
    float updated_position = new_pos;

    // Update entry price if opening/adding to position
    float updated_entry_price = entry_price;
    if ((current_position >= 0.0f && position_change > 0.0f) ||
        (current_position <= 0.0f && position_change < 0.0f)) {
        // Adding to position: weighted average entry price
        if (fabsf(updated_position) > 1e-6f) {
            updated_entry_price = (current_position * entry_price +
                                  position_change * execution_price) /
                                 updated_position;
        } else {
            updated_entry_price = execution_price;
        }
    } else if (fabsf(updated_position) < 1e-6f) {
        // Closed position completely
        updated_entry_price = 0.0f;
    }

    // Update days held
    int updated_days = current_days + 1;
    if (fabsf(updated_position) < 1e-6f) {
        updated_days = 0;  // Reset if position closed
    }

    // Calculate new unrealized PnL
    float new_unrealized_pnl = 0.0f;
    if (fabsf(updated_position) > 1e-6f) {
        float new_price_change_pct = (close - updated_entry_price) / updated_entry_price;
        new_unrealized_pnl = updated_position * new_price_change_pct;
    }

    // Calculate total PnL change
    float pnl_change = step_realized_pnl + (new_unrealized_pnl - position_pnl);

    // Calculate reward
    // Reward = PnL change - fees - leverage costs
    // Normalize by initial capital for scale invariance
    float reward = (pnl_change - fee - lev_cost) / initial_capital;

    // Write outputs
    new_cash[idx] = updated_cash;
    new_positions[idx] = updated_position;
    new_entry_prices[idx] = updated_entry_price;
    new_total_pnl[idx] = current_pnl + pnl_change;
    new_days_held[idx] = updated_days;

    rewards[idx] = reward;
    fees_paid[idx] = fee;
    leverage_costs[idx] = lev_cost;
    realized_pnl[idx] = step_realized_pnl;
    unrealized_pnl[idx] = new_unrealized_pnl;
}

// Fast observation assembly kernel
__global__ void assemble_observations_kernel(
    const float* __restrict__ market_features,
    const float* __restrict__ portfolio_state,
    float* __restrict__ observations,
    int batch_size,
    int lookback_window,
    int market_feature_dim,
    int portfolio_feature_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    int market_flat_size = lookback_window * market_feature_dim;
    int obs_dim = market_flat_size + portfolio_feature_dim;

    // Copy market features (already flattened)
    for (int i = 0; i < market_flat_size; i++) {
        observations[idx * obs_dim + i] = market_features[idx * market_flat_size + i];
    }

    // Copy portfolio state
    for (int i = 0; i < portfolio_feature_dim; i++) {
        observations[idx * obs_dim + market_flat_size + i] =
            portfolio_state[idx * portfolio_feature_dim + i];
    }
}

// Launch wrapper for portfolio step
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
) {
    int batch_size = actions.size(0);
    int blocks = (batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    portfolio_step_fused_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        actions.data_ptr<float>(),
        open_prices.data_ptr<float>(),
        high_prices.data_ptr<float>(),
        low_prices.data_ptr<float>(),
        close_prices.data_ptr<float>(),
        cash.data_ptr<float>(),
        positions.data_ptr<float>(),
        entry_prices.data_ptr<float>(),
        total_pnl.data_ptr<float>(),
        days_held.data_ptr<int>(),
        exec_timing.data_ptr<float>(),
        exec_aggression.data_ptr<float>(),
        new_cash.data_ptr<float>(),
        new_positions.data_ptr<float>(),
        new_entry_prices.data_ptr<float>(),
        new_total_pnl.data_ptr<float>(),
        new_days_held.data_ptr<int>(),
        rewards.data_ptr<float>(),
        fees_paid.data_ptr<float>(),
        leverage_costs.data_ptr<float>(),
        realized_pnl.data_ptr<float>(),
        unrealized_pnl.data_ptr<float>(),
        batch_size,
        initial_capital,
        trading_fee_rate,
        daily_leverage_cost_rate,
        max_leverage,
        is_crypto
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in portfolio_step: %s\n", cudaGetErrorString(err));
    }
}

// Launch wrapper for observation assembly
void launch_assemble_observations(
    const torch::Tensor& market_features,
    const torch::Tensor& portfolio_state,
    torch::Tensor& observations
) {
    int batch_size = market_features.size(0);
    int lookback_window = market_features.size(1);
    int market_feature_dim = market_features.size(2);
    int portfolio_feature_dim = portfolio_state.size(1);

    int blocks = (batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    assemble_observations_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        market_features.data_ptr<float>(),
        portfolio_state.data_ptr<float>(),
        observations.data_ptr<float>(),
        batch_size,
        lookback_window,
        market_feature_dim,
        portfolio_feature_dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in assemble_observations: %s\n", cudaGetErrorString(err));
    }
}

// V-trace wrapper (uses pufferlib's optimized kernel)
// Note: This should be called from C++ code, not from CUDA kernels
// The actual pufferlib operator will be called directly from the training loop
void compute_vtrace_advantages(
    torch::Tensor values,
    torch::Tensor rewards,
    torch::Tensor dones,
    torch::Tensor importance,
    torch::Tensor advantages,
    float gamma,
    float lambda,
    float rho_clip,
    float c_clip
) {
    // For now, we'll implement a simple GAE instead of calling pufferlib
    // TODO: Call pufferlib operator from training loop instead

    // Simple GAE computation (not as optimized as pufferlib but works)
    int num_steps = values.size(0);
    int batch_size = values.size(1);

    auto values_cpu = values.cpu();
    auto rewards_cpu = rewards.cpu();
    auto dones_cpu = dones.cpu();
    auto advantages_cpu = advantages.cpu();

    auto values_acc = values_cpu.accessor<float, 2>();
    auto rewards_acc = rewards_cpu.accessor<float, 2>();
    auto dones_acc = dones_cpu.accessor<float, 2>();
    auto advantages_acc = advantages_cpu.accessor<float, 2>();

    std::vector<float> gae(batch_size, 0.0f);
    for (int t = num_steps - 2; t >= 0; t--) {
        for (int b = 0; b < batch_size; b++) {
            float not_done = 1.0f - dones_acc[t + 1][b];
            float delta = rewards_acc[t][b] +
                         gamma * values_acc[t + 1][b] * not_done -
                         values_acc[t][b];
            gae[b] = delta + gamma * lambda * not_done * gae[b];
            advantages_acc[t][b] = gae[b];
        }
    }

    advantages.copy_(advantages_cpu);
}

} // namespace cuda
} // namespace market_sim
