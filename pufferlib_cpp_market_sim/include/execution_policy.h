#pragma once

#include <torch/torch.h>
#include "market_config.h"

namespace market_sim {

// Execution policy network learns optimal execution timing and aggression
// Input: observation + desired action + current prices
// Output: execution_timing (0=best price, 1=worst price)
//         execution_aggression (0=use close, 1=use high/low)
struct ExecutionPolicyNet : torch::nn::Module {
    ExecutionPolicyNet(int obs_dim, int hidden_dim = 128) {
        // Shared feature extractor
        fc1 = register_module("fc1", torch::nn::Linear(obs_dim + 5, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));

        // Execution timing head (0=passive/best price, 1=aggressive/worst price)
        timing_head = register_module("timing_head", torch::nn::Linear(hidden_dim, 1));

        // Execution aggression head (0=close price, 1=high/low extremes)
        aggression_head = register_module("aggression_head", torch::nn::Linear(hidden_dim, 1));
    }

    // Forward pass
    // obs: [batch, obs_dim] - market observation
    // action: [batch, 1] - desired position change
    // prices: [batch, 4] - OHLC for current bar
    // Returns: (timing, aggression) both [batch, 1]
    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor obs,
        torch::Tensor action,
        torch::Tensor prices  // OHLC
    ) {
        // Concatenate observation + action + prices
        auto input = torch::cat({obs, action, prices}, /*dim=*/1);

        // Shared feature extraction
        auto x = torch::relu(fc1->forward(input));
        x = torch::relu(fc2->forward(x));

        // Execution timing: sigmoid to [0, 1]
        auto timing = torch::sigmoid(timing_head->forward(x));

        // Execution aggression: sigmoid to [0, 1]
        auto aggression = torch::sigmoid(aggression_head->forward(x));

        return std::make_tuple(timing, aggression);
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::Linear timing_head{nullptr}, aggression_head{nullptr};
};

// Execution policy trainer
// Learns to exploit maxdiff opportunities during training
// But also learns conservative execution for validation
class ExecutionPolicyTrainer {
public:
    ExecutionPolicyTrainer(
        int obs_dim,
        torch::Device device,
        float learning_rate = 1e-4
    );

    // Update execution policy based on rollout data
    // Reward shaping:
    // - During training: reward aggressive execution (high/low exploitation)
    // - During validation: reward conservative execution (minimize slippage)
    void update(
        const torch::Tensor& observations,
        const torch::Tensor& actions,
        const torch::Tensor& prices,
        const torch::Tensor& realized_pnl,
        const torch::Tensor& fees_paid,
        bool is_training
    );

    // Get execution parameters for current state
    std::tuple<torch::Tensor, torch::Tensor> get_execution_params(
        const torch::Tensor& observations,
        const torch::Tensor& actions,
        const torch::Tensor& prices,
        bool is_training  // If false, use conservative execution
    );

    // Save/load model
    void save(const std::string& path);
    void load(const std::string& path);

private:
    std::shared_ptr<ExecutionPolicyNet> policy_;
    std::shared_ptr<torch::optim::Adam> optimizer_;
    torch::Device device_;
    int update_count_;

    // Calculate execution quality reward
    // High reward for:
    // - Training: Using high/low extremes (maxdiff exploitation)
    // - Validation: Using close price (realistic execution)
    torch::Tensor calculate_execution_reward(
        const torch::Tensor& timing,
        const torch::Tensor& aggression,
        const torch::Tensor& realized_pnl,
        const torch::Tensor& fees_paid,
        bool is_training
    );
};

// Strategy configuration for multi-strategy training
struct StrategyConfig {
    float risk_tolerance;      // 0.0 (conservative) to 1.0 (aggressive)
    float holding_period;      // Preferred days to hold position
    float execution_style;     // 0.0 (passive) to 1.0 (aggressive)
    float fee_sensitivity;     // Weight for transaction cost penalty

    // Create random strategy
    static StrategyConfig random();

    // Create strategy from index (for curriculum learning)
    static StrategyConfig from_index(int index, int total_strategies);

    // Encode as tensor for network input
    torch::Tensor to_tensor(torch::Device device) const;
};

// Multi-strategy environment wrapper
// Trains multiple strategies simultaneously with different risk profiles
class MultiStrategyWrapper {
public:
    MultiStrategyWrapper(int batch_size, torch::Device device);

    // Randomize strategy configs for each environment
    void randomize_strategies();

    // Get strategy config for environment i
    StrategyConfig get_strategy(int env_idx) const;

    // Get all strategy tensors for batched operations
    torch::Tensor get_strategy_tensors() const;

    // Apply strategy-specific reward shaping
    torch::Tensor shape_rewards(
        const torch::Tensor& raw_rewards,
        const torch::Tensor& fees_paid,
        const torch::Tensor& leverage_costs,
        const torch::Tensor& days_held
    );

private:
    int batch_size_;
    torch::Device device_;
    std::vector<StrategyConfig> strategies_;
    torch::Tensor strategy_tensors_;  // [batch_size, 4] - encoded strategies
};

} // namespace market_sim
