#pragma once

#include <torch/torch.h>
#include <string>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono>

namespace market_sim {

struct PnLRecord {
    int64_t timestamp;
    int env_id;
    std::string symbol;
    float position;
    float pnl;
    float realized_pnl;
    float unrealized_pnl;
    float trading_cost;
    float leverage_cost;
    float total_return_pct;
    int num_trades;
    bool is_training;
};

class PnLLogger {
public:
    PnLLogger(const std::string& log_dir);
    ~PnLLogger();

    // Log current state of all environments
    void log_step(
        const std::vector<std::string>& symbols,
        const torch::Tensor& env_ids,
        const torch::Tensor& positions,
        const torch::Tensor& pnl,
        const torch::Tensor& realized_pnl,
        const torch::Tensor& unrealized_pnl,
        const torch::Tensor& trading_costs,
        const torch::Tensor& leverage_costs,
        const torch::Tensor& num_trades,
        bool is_training
    );

    // Log episode completion
    void log_episode_end(
        int env_id,
        const std::string& symbol,
        float final_pnl,
        float total_return_pct,
        int num_trades,
        float sharpe_ratio,
        bool is_training
    );

    // Write summary statistics
    void write_summary_stats(bool is_training);

    // Flush buffers to disk
    void flush();

    // Get statistics for monitoring
    struct Stats {
        float mean_pnl;
        float std_pnl;
        float mean_return_pct;
        float sharpe_ratio;
        int total_trades;
        int total_episodes;
    };

    Stats get_training_stats() const;
    Stats get_testing_stats() const;

private:
    std::string log_dir_;
    std::ofstream train_log_;
    std::ofstream test_log_;
    std::ofstream episode_log_;

    // Buffered records for efficient I/O
    std::vector<PnLRecord> train_buffer_;
    std::vector<PnLRecord> test_buffer_;

    // Running statistics
    std::vector<float> train_returns_;
    std::vector<float> test_returns_;
    int train_episodes_;
    int test_episodes_;

    // Internal helpers
    void write_header(std::ofstream& file);
    void write_record(std::ofstream& file, const PnLRecord& record);
    Stats calculate_stats(const std::vector<float>& returns, int episodes) const;
};

} // namespace market_sim
