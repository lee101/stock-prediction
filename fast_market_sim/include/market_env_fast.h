#pragma once

#include "forecaster.h"
#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>

namespace fast_market {

/**
 * Ultra-fast market environment with ML-based forecasting
 *
 * Key features:
 * - Compiled Toto/Kronos models for price prediction
 * - Batch processing with CUDA acceleration
 * - Vectorized parallel environments (4096+)
 * - Sub-millisecond step times
 */
class FastMarketEnv {
public:
    struct Config {
        // Environment settings
        int num_parallel_envs = 256;
        int num_assets = 100;
        int max_episode_steps = 1000;
        int lookback_window = 512;

        // Trading parameters
        float initial_capital = 100000.0f;
        float transaction_cost = 0.0005f;  // 5 bps
        float max_position_size = 0.25f;   // Max 25% per position
        float leverage = 1.0f;

        // Forecaster config
        TimeSeriesForecaster::Config forecaster_config;

        // Device
        torch::DeviceType device = torch::kCUDA;

        // Data
        std::string data_dir;
        std::vector<std::string> asset_symbols;
    };

    explicit FastMarketEnv(const Config& config);

    /**
     * Reset environments
     *
     * @param env_indices Which environments to reset (empty = all)
     * @return observations [num_envs, obs_dim]
     */
    torch::Tensor reset(const std::vector<int>& env_indices = {});

    /**
     * Step all environments
     *
     * @param actions [num_envs, num_assets] - portfolio weights
     * @return (observations, rewards, dones, truncated, info)
     */
    struct StepResult {
        torch::Tensor observations;  // [num_envs, obs_dim]
        torch::Tensor rewards;       // [num_envs]
        torch::Tensor dones;         // [num_envs]
        torch::Tensor truncated;     // [num_envs]
        std::unordered_map<std::string, torch::Tensor> info;
    };
    StepResult step(const torch::Tensor& actions);

    /**
     * Get observation space size
     */
    int observation_size() const {
        // Lookback window + forecast + portfolio state
        return config_.lookback_window * config_.num_assets * 6  // OHLCV+vol history
               + config_.forecaster_config.forecast_horizon * config_.num_assets * 6  // Forecast
               + config_.num_assets  // Current positions
               + 3;  // Cash, value, progress
    }

    /**
     * Get action space size
     */
    int action_size() const {
        return config_.num_assets;  // Portfolio weights
    }

    /**
     * Performance statistics
     */
    struct PerformanceStats {
        double mean_step_time_ms;
        double throughput_steps_per_sec;
        size_t total_steps;

        // Market statistics
        torch::Tensor mean_returns;       // [num_envs]
        torch::Tensor sharpe_ratios;      // [num_envs]
        torch::Tensor max_drawdowns;      // [num_envs]
    };
    PerformanceStats get_stats() const;

    /**
     * Load market data
     */
    void load_data(const std::vector<std::string>& csv_paths);

private:
    Config config_;
    torch::Device device_;

    // Forecaster
    std::shared_ptr<TimeSeriesForecaster> forecaster_;

    // Market data: [num_assets, total_timesteps, features=6]
    torch::Tensor market_data_;
    std::vector<std::string> asset_symbols_;

    // Environment state
    torch::Tensor current_indices_;   // [num_envs] - current timestep per env
    torch::Tensor current_assets_;    // [num_envs] - current asset per env
    torch::Tensor episode_steps_;     // [num_envs] - steps in current episode

    // Portfolio state
    torch::Tensor cash_;              // [num_envs]
    torch::Tensor positions_;         // [num_envs, num_assets]
    torch::Tensor portfolio_values_;  // [num_envs]
    torch::Tensor initial_values_;    // [num_envs]
    torch::Tensor peak_values_;       // [num_envs] - for drawdown

    // Performance tracking
    PerformanceStats stats_;
    std::vector<double> step_times_;

    // Internal methods
    torch::Tensor get_observations();
    torch::Tensor get_forecasts();
    void execute_trades(const torch::Tensor& actions);
    torch::Tensor calculate_rewards();
    void update_prices();
};


/**
 * Multi-agent competitive market environment
 *
 * Multiple agents trade in the same market, competing for returns
 */
class CompetitiveMarketEnv : public FastMarketEnv {
public:
    struct Config : public FastMarketEnv::Config {
        int num_agents = 4;
        bool allow_front_running = false;
        float market_impact_factor = 0.0001f;  // Price impact from trades
    };

    explicit CompetitiveMarketEnv(const Config& config);

    /**
     * Step with agent IDs
     *
     * @param actions [num_agents, num_assets]
     * @param agent_ids [num_agents]
     */
    struct MultiAgentStepResult {
        std::vector<torch::Tensor> observations;  // Per agent
        std::vector<torch::Tensor> rewards;       // Per agent
        torch::Tensor market_state;               // Shared market state
        std::unordered_map<std::string, torch::Tensor> info;
    };
    MultiAgentStepResult step_multi_agent(
        const torch::Tensor& actions,
        const std::vector<int>& agent_ids
    );

private:
    Config comp_config_;
    std::vector<torch::Tensor> agent_portfolios_;
};

} // namespace fast_market
