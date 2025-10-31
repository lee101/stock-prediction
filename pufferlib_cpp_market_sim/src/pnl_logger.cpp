#include "pnl_logger.h"
#include "market_config.h"
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <cmath>

namespace market_sim {

PnLLogger::PnLLogger(const std::string& log_dir)
    : log_dir_(log_dir), train_episodes_(0), test_episodes_(0) {

    // Create log directory if it doesn't exist
    std::filesystem::create_directories(log_dir_);

    // Open log files
    std::string train_path = log_dir_ + "/train_pnl.csv";
    std::string test_path = log_dir_ + "/test_pnl.csv";
    std::string episode_path = log_dir_ + "/episodes.csv";

    train_log_.open(train_path, std::ios::out | std::ios::trunc);
    test_log_.open(test_path, std::ios::out | std::ios::trunc);
    episode_log_.open(episode_path, std::ios::out | std::ios::trunc);

    if (!train_log_.is_open() || !test_log_.is_open() || !episode_log_.is_open()) {
        throw std::runtime_error("Failed to open log files in: " + log_dir_);
    }

    // Write headers
    write_header(train_log_);
    write_header(test_log_);

    // Episode log header
    episode_log_ << "timestamp,env_id,symbol,final_pnl,total_return_pct,num_trades,sharpe_ratio,is_training\n";
}

PnLLogger::~PnLLogger() {
    flush();
    train_log_.close();
    test_log_.close();
    episode_log_.close();
}

void PnLLogger::log_step(
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
) {
    // Move tensors to CPU for logging
    auto env_ids_cpu = env_ids.cpu();
    auto positions_cpu = positions.cpu();
    auto pnl_cpu = pnl.cpu();
    auto realized_pnl_cpu = realized_pnl.cpu();
    auto unrealized_pnl_cpu = unrealized_pnl.cpu();
    auto trading_costs_cpu = trading_costs.cpu();
    auto leverage_costs_cpu = leverage_costs.cpu();
    auto num_trades_cpu = num_trades.cpu();

    auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();

    int batch_size = positions.size(0);
    auto& buffer = is_training ? train_buffer_ : test_buffer_;

    for (int i = 0; i < batch_size; i++) {
        PnLRecord record;
        record.timestamp = timestamp;
        record.env_id = env_ids_cpu[i].item<int>();
        record.symbol = symbols[i % symbols.size()];
        record.position = positions_cpu[i].item<float>();
        record.pnl = pnl_cpu[i].item<float>();
        record.realized_pnl = realized_pnl_cpu[i].item<float>();
        record.unrealized_pnl = unrealized_pnl_cpu[i].item<float>();
        record.trading_cost = trading_costs_cpu[i].item<float>();
        record.leverage_cost = leverage_costs_cpu[i].item<float>();
        record.total_return_pct = (record.pnl / MarketConfig::INITIAL_CAPITAL) * 100.0f;
        record.num_trades = num_trades_cpu[i].item<int>();
        record.is_training = is_training;

        buffer.push_back(record);
    }

    // Flush if buffer is large
    if (buffer.size() > 1000) {
        flush();
    }
}

void PnLLogger::log_episode_end(
    int env_id,
    const std::string& symbol,
    float final_pnl,
    float total_return_pct,
    int num_trades,
    float sharpe_ratio,
    bool is_training
) {
    auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();

    episode_log_ << timestamp << ","
                << env_id << ","
                << symbol << ","
                << final_pnl << ","
                << total_return_pct << ","
                << num_trades << ","
                << sharpe_ratio << ","
                << (is_training ? "1" : "0") << "\n";

    // Update statistics
    if (is_training) {
        train_returns_.push_back(total_return_pct);
        train_episodes_++;
    } else {
        test_returns_.push_back(total_return_pct);
        test_episodes_++;
    }
}

void PnLLogger::write_summary_stats(bool is_training) {
    auto stats = is_training ? get_training_stats() : get_testing_stats();

    std::string summary_path = log_dir_ + (is_training ? "/train_summary.txt" : "/test_summary.txt");
    std::ofstream summary(summary_path);

    summary << "=== " << (is_training ? "Training" : "Testing") << " Summary ===" << std::endl;
    summary << "Total Episodes: " << stats.total_episodes << std::endl;
    summary << "Mean PnL: $" << std::fixed << std::setprecision(2) << stats.mean_pnl << std::endl;
    summary << "Std PnL: $" << stats.std_pnl << std::endl;
    summary << "Mean Return: " << stats.mean_return_pct << "%" << std::endl;
    summary << "Sharpe Ratio: " << stats.sharpe_ratio << std::endl;
    summary << "Total Trades: " << stats.total_trades << std::endl;

    summary.close();
}

void PnLLogger::flush() {
    // Write training buffer
    for (const auto& record : train_buffer_) {
        write_record(train_log_, record);
    }
    train_buffer_.clear();
    train_log_.flush();

    // Write test buffer
    for (const auto& record : test_buffer_) {
        write_record(test_log_, record);
    }
    test_buffer_.clear();
    test_log_.flush();

    episode_log_.flush();
}

PnLLogger::Stats PnLLogger::get_training_stats() const {
    return calculate_stats(train_returns_, train_episodes_);
}

PnLLogger::Stats PnLLogger::get_testing_stats() const {
    return calculate_stats(test_returns_, test_episodes_);
}

void PnLLogger::write_header(std::ofstream& file) {
    file << "timestamp,env_id,symbol,position,pnl,realized_pnl,unrealized_pnl,"
         << "trading_cost,leverage_cost,total_return_pct,num_trades\n";
}

void PnLLogger::write_record(std::ofstream& file, const PnLRecord& record) {
    file << record.timestamp << ","
         << record.env_id << ","
         << record.symbol << ","
         << std::fixed << std::setprecision(2)
         << record.position << ","
         << record.pnl << ","
         << record.realized_pnl << ","
         << record.unrealized_pnl << ","
         << record.trading_cost << ","
         << record.leverage_cost << ","
         << record.total_return_pct << ","
         << record.num_trades << "\n";
}

PnLLogger::Stats PnLLogger::calculate_stats(
    const std::vector<float>& returns,
    int episodes
) const {
    Stats stats;
    stats.total_episodes = episodes;

    if (returns.empty()) {
        stats.mean_pnl = 0.0f;
        stats.std_pnl = 0.0f;
        stats.mean_return_pct = 0.0f;
        stats.sharpe_ratio = 0.0f;
        stats.total_trades = 0;
        return stats;
    }

    // Calculate mean return
    float sum = std::accumulate(returns.begin(), returns.end(), 0.0f);
    stats.mean_return_pct = sum / returns.size();

    // Calculate mean PnL (convert from percentage)
    stats.mean_pnl = (stats.mean_return_pct / 100.0f) * MarketConfig::INITIAL_CAPITAL;

    // Calculate std
    float sq_sum = 0.0f;
    for (float r : returns) {
        sq_sum += (r - stats.mean_return_pct) * (r - stats.mean_return_pct);
    }
    float variance = sq_sum / returns.size();
    float std_return = std::sqrt(variance);

    stats.std_pnl = (std_return / 100.0f) * MarketConfig::INITIAL_CAPITAL;

    // Calculate Sharpe ratio (assuming risk-free rate = 0)
    stats.sharpe_ratio = (std_return > 1e-8) ?
                        (stats.mean_return_pct / std_return) * std::sqrt(252.0f) : 0.0f;

    stats.total_trades = episodes;  // Simplified

    return stats;
}

} // namespace market_sim
