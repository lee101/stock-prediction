#include "execution_policy.h"
#include <random>

namespace market_sim {

// ExecutionPolicyTrainer implementation

ExecutionPolicyTrainer::ExecutionPolicyTrainer(
    int obs_dim,
    torch::Device device,
    float learning_rate
) : device_(device), update_count_(0) {
    policy_ = std::make_shared<ExecutionPolicyNet>(obs_dim);
    policy_->to(device);

    optimizer_ = std::make_shared<torch::optim::Adam>(
        policy_->parameters(),
        torch::optim::AdamOptions(learning_rate)
    );
}

void ExecutionPolicyTrainer::update(
    const torch::Tensor& observations,
    const torch::Tensor& actions,
    const torch::Tensor& prices,
    const torch::Tensor& realized_pnl,
    const torch::Tensor& fees_paid,
    bool is_training
) {
    optimizer_->zero_grad();

    // Forward pass
    auto [timing, aggression] = policy_->forward(observations, actions, prices);

    // Calculate execution quality reward
    auto exec_reward = calculate_execution_reward(
        timing, aggression, realized_pnl, fees_paid, is_training
    );

    // Loss = negative reward (we want to maximize reward)
    auto loss = -exec_reward.mean();

    // Backward and optimize
    loss.backward();
    optimizer_->step();

    update_count_++;
}

std::tuple<torch::Tensor, torch::Tensor> ExecutionPolicyTrainer::get_execution_params(
    const torch::Tensor& observations,
    const torch::Tensor& actions,
    const torch::Tensor& prices,
    bool is_training
) {
    torch::NoGradGuard no_grad;

    auto [timing, aggression] = policy_->forward(observations, actions, prices);

    if (!is_training) {
        // During validation/testing, use conservative execution
        // Force timing toward best prices (0.0)
        // Force aggression toward close prices (0.0)
        timing = timing * 0.3f;  // Reduce aggressiveness
        aggression = aggression * 0.2f;  // Stay close to close price
    }

    return std::make_tuple(timing, aggression);
}

torch::Tensor ExecutionPolicyTrainer::calculate_execution_reward(
    const torch::Tensor& timing,
    const torch::Tensor& aggression,
    const torch::Tensor& realized_pnl,
    const torch::Tensor& fees_paid,
    bool is_training
) {
    torch::Tensor reward;

    if (is_training) {
        // Training: Reward aggressive execution (high/low exploitation)
        // Higher timing/aggression = better execution on OHLC extremes
        auto aggression_bonus = (timing + aggression) * 0.5f;  // [0, 1]

        // Combine with actual PnL
        reward = realized_pnl + aggression_bonus * 0.01f - fees_paid;
    } else {
        // Validation: Reward conservative execution (realistic)
        // Lower timing/aggression = more realistic execution
        auto conservatism_bonus = (1.0f - timing) * (1.0f - aggression);

        // Penalize aggressive execution during validation
        reward = realized_pnl + conservatism_bonus * 0.01f - fees_paid;
    }

    return reward;
}

void ExecutionPolicyTrainer::save(const std::string& path) {
    torch::save(policy_, path);
}

void ExecutionPolicyTrainer::load(const std::string& path) {
    torch::load(policy_, path);
}

// StrategyConfig implementation

StrategyConfig StrategyConfig::random() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    return StrategyConfig{
        .risk_tolerance = dist(gen),
        .holding_period = 1.0f + dist(gen) * 29.0f,  // 1-30 days
        .execution_style = dist(gen),
        .fee_sensitivity = 0.1f + dist(gen) * 9.9f   // 0.1-10.0
    };
}

StrategyConfig StrategyConfig::from_index(int index, int total_strategies) {
    // Create diverse strategies using curriculum learning
    float progress = static_cast<float>(index) / static_cast<float>(total_strategies);

    return StrategyConfig{
        .risk_tolerance = progress,
        .holding_period = 1.0f + progress * 29.0f,
        .execution_style = progress,
        .fee_sensitivity = 0.1f + (1.0f - progress) * 9.9f
    };
}

torch::Tensor StrategyConfig::to_tensor(torch::Device device) const {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    return torch::tensor(
        {risk_tolerance, holding_period, execution_style, fee_sensitivity},
        opts
    );
}

// MultiStrategyWrapper implementation

MultiStrategyWrapper::MultiStrategyWrapper(int batch_size, torch::Device device)
    : batch_size_(batch_size), device_(device) {
    strategies_.reserve(batch_size);
    randomize_strategies();
}

void MultiStrategyWrapper::randomize_strategies() {
    strategies_.clear();

    // Create diverse strategy population
    for (int i = 0; i < batch_size_; i++) {
        // Mix random and curriculum-based strategies
        if (i % 2 == 0) {
            strategies_.push_back(StrategyConfig::random());
        } else {
            strategies_.push_back(StrategyConfig::from_index(i, batch_size_));
        }
    }

    // Convert to tensor for efficient batched operations
    std::vector<torch::Tensor> strategy_tensors;
    for (const auto& strategy : strategies_) {
        strategy_tensors.push_back(strategy.to_tensor(device_));
    }
    strategy_tensors_ = torch::stack(strategy_tensors, 0);  // [batch_size, 4]
}

StrategyConfig MultiStrategyWrapper::get_strategy(int env_idx) const {
    return strategies_[env_idx];
}

torch::Tensor MultiStrategyWrapper::get_strategy_tensors() const {
    return strategy_tensors_;
}

torch::Tensor MultiStrategyWrapper::shape_rewards(
    const torch::Tensor& raw_rewards,
    const torch::Tensor& fees_paid,
    const torch::Tensor& leverage_costs,
    const torch::Tensor& days_held
) {
    // Extract strategy parameters
    auto risk_tolerance = strategy_tensors_.index({torch::indexing::Slice(), 0});
    auto holding_period = strategy_tensors_.index({torch::indexing::Slice(), 1});
    auto execution_style = strategy_tensors_.index({torch::indexing::Slice(), 2});
    auto fee_sensitivity = strategy_tensors_.index({torch::indexing::Slice(), 3});

    // Base reward
    auto shaped_rewards = raw_rewards.clone();

    // Apply strategy-specific shaping

    // 1. Fee penalty (scaled by fee_sensitivity)
    shaped_rewards = shaped_rewards - fee_sensitivity * fees_paid;

    // 2. Leverage cost penalty (scaled by risk aversion)
    auto risk_aversion = 1.0f - risk_tolerance;
    shaped_rewards = shaped_rewards - risk_aversion * leverage_costs * 10.0f;

    // 3. Holding period bonus/penalty
    auto days_held_float = days_held.to(torch::kFloat32);
    auto holding_deviation = torch::abs(days_held_float - holding_period);
    shaped_rewards = shaped_rewards - holding_deviation * 0.001f;

    // 4. Execution style doesn't directly affect rewards
    // (handled by execution policy network)

    return shaped_rewards;
}

} // namespace market_sim
