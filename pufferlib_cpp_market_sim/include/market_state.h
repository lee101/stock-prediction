#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include "market_config.h"

namespace market_sim {

struct MarketData {
    torch::Tensor timestamps;  // [T]
    torch::Tensor open;        // [T]
    torch::Tensor high;        // [T]
    torch::Tensor low;         // [T]
    torch::Tensor close;       // [T]
    torch::Tensor volume;      // [T]

    std::string symbol;
    AssetClass asset_class;
    int length;
};

class MarketState {
public:
    MarketState(const MarketConfig& config, torch::Device device);

    // Load CSV data from trainingdata/
    void load_data(const std::string& symbol);

    // Get observation tensor for current timestep
    // Shape: [batch_size, lookback_window, num_features]
    torch::Tensor get_observation(const torch::Tensor& current_indices);

    // Get current prices for position valuation
    // Returns: [batch_size, 4] (open, high, low, close)
    torch::Tensor get_current_prices(const torch::Tensor& current_indices) const;

    // Check if asset is crypto
    bool is_crypto() const { return data_.asset_class == AssetClass::CRYPTO; }

    // Get trading fee for this asset
    float get_trading_fee() const {
        return data_.asset_class == AssetClass::CRYPTO ?
               MarketConfig::CRYPTO_TRADING_FEE :
               MarketConfig::STOCK_TRADING_FEE;
    }

    int get_data_length() const { return data_.length; }

private:
    MarketConfig config_;
    torch::Device device_;
    MarketData data_;

    // Normalize features for neural network input
    torch::Tensor normalize_features(const torch::Tensor& raw_features);
};

} // namespace market_sim
