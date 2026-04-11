#pragma once

#include <string>

namespace market_sim {

// Action-space mode. SCALAR is the legacy single-dim action in [-MAX_LEVERAGE, MAX_LEVERAGE].
// DPS = (direction, size, limit_offset_bps) with binary fills against bar [low, high].
enum class ActionMode {
    SCALAR = 0,
    DPS = 1,
};

struct MarketConfig {
    // Trading fees
    static constexpr float STOCK_TRADING_FEE = 0.0005f;      // 0.05% (5 bps) for stocks
    static constexpr float CRYPTO_TRADING_FEE = 0.001f;      // 0.10% (10 bps) for crypto (standardized)

    // Leverage settings
    static constexpr float ANNUAL_LEVERAGE_COST = 0.065f;   // 6.5% annual
    static constexpr int TRADING_DAYS_PER_YEAR = 252;
    static constexpr float DAILY_LEVERAGE_COST = ANNUAL_LEVERAGE_COST / TRADING_DAYS_PER_YEAR; // ~0.0268%
    static constexpr float MAX_LEVERAGE = 1.5f;              // Max 1.5x leverage for stocks (SCALAR mode)

    // DPS action-mode parameters (only consulted when action_mode == DPS).
    // Defaults are OFF / SCALAR so existing stocks12/crypto12/v5_rsi runs are bit-identical.
    ActionMode action_mode = ActionMode::SCALAR;
    float max_leverage_dps = 5.0f;        // 5x leverage cap for DPS direction*size
    float max_limit_offset_bps = 20.0f;   // Max absolute limit-price offset on top of fill_buffer
    float fill_buffer_bps = 5.0f;         // Crossing buffer (bps) used by DPS fill logic

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
