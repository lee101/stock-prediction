#pragma once

#include <string>

namespace market_sim {

struct MarketConfig {
    // Trading fees
    static constexpr float STOCK_TRADING_FEE = 0.0005f;      // 0.05% for stocks
    static constexpr float CRYPTO_TRADING_FEE = 0.0015f;     // 0.15% for crypto

    // Leverage settings
    static constexpr float ANNUAL_LEVERAGE_COST = 0.0675f;   // 6.75% annual
    static constexpr int TRADING_DAYS_PER_YEAR = 252;
    static constexpr float DAILY_LEVERAGE_COST = ANNUAL_LEVERAGE_COST / TRADING_DAYS_PER_YEAR; // ~0.0268%
    static constexpr float MAX_LEVERAGE = 1.5f;              // Max 1.5x leverage for stocks

    // Environment settings
    static constexpr int LOOKBACK_WINDOW = 60;               // Days of historical data to observe
    static constexpr int NUM_PARALLEL_ENVS = 4096;           // Batch size for GPU parallelism
    static constexpr float INITIAL_CAPITAL = 100000.0f;      // Starting capital

    // Data paths
    std::string data_dir = "../../trainingdata";
    std::string log_dir = "../logs";

    // GPU settings
    std::string device = "cuda:0";
    bool use_mixed_precision = true;
};

enum class AssetClass {
    STOCK,
    CRYPTO
};

} // namespace market_sim
