#pragma once

#include <torch/torch.h>
#include <memory>
#include <vector>
#include "market_config.h"
#include "market_state.h"
#include "portfolio.h"
#include "pnl_logger.h"

namespace market_sim {

struct EnvOutput {
    torch::Tensor observations;    // [batch_size, obs_dim]
    torch::Tensor rewards;         // [batch_size]
    torch::Tensor dones;           // [batch_size] - episode termination
    torch::Tensor truncated;       // [batch_size] - max steps reached
    torch::Tensor prices;          // [batch_size, 4] - OHLC used for last step
    torch::Tensor fees_paid;       // [batch_size]
    torch::Tensor leverage_costs;  // [batch_size]
    torch::Tensor realized_pnl;    // [batch_size]
    torch::Tensor days_held;       // [batch_size]
};

class MarketEnvironment {
public:
    MarketEnvironment(const MarketConfig& config);

    // Load multiple symbols for training
    void load_symbols(const std::vector<std::string>& symbols);

    // Reset environments
    EnvOutput reset(const torch::Tensor& env_indices);

    // Step all environments
    // actions: [batch_size] - continuous actions
    EnvOutput step(const torch::Tensor& actions);

    // Get observation/action space dimensions
    int get_observation_dim() const;
    int get_action_dim() const { return 1; }  // Single continuous action

    // Peek at the next-step OHLC data without advancing the sim
    torch::Tensor peek_next_prices() const;

    // Enable/disable training mode
    void set_training_mode(bool training) { is_training_ = training; }
    bool is_training() const { return is_training_; }

    // Get PnL logger for analysis
    std::shared_ptr<PnLLogger> get_logger() { return logger_; }

    // Flush logs to disk
    void flush_logs();

private:
    MarketConfig config_;
    torch::Device device_;

    // Environment state
    std::vector<std::shared_ptr<MarketState>> market_states_;  // One per symbol
    std::shared_ptr<Portfolio> portfolio_;
    std::shared_ptr<PnLLogger> logger_;

    // Current state tensors
    torch::Tensor current_indices_;    // [batch_size] - current timestep per env
    torch::Tensor current_symbols_;    // [batch_size] - which symbol each env is using
    torch::Tensor episode_steps_;      // [batch_size] - steps in current episode

    bool is_training_;
    int max_episode_steps_;
    int batch_size_;

    // Randomly sample starting positions for each episode
    void randomize_starting_positions(const torch::Tensor& env_indices);

    // Check if episodes should terminate
    torch::Tensor check_termination();
};

} // namespace market_sim
