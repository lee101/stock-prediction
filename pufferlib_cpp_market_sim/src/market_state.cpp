#include "market_state.h"
#include "csv_loader.h"
#include <filesystem>
#include <stdexcept>

namespace market_sim {

MarketState::MarketState(const MarketConfig& config, torch::Device device)
    : config_(config), device_(device) {
}

void MarketState::load_data(const std::string& symbol) {
    // Construct file path
    std::filesystem::path filepath = std::filesystem::path(config_.data_dir) / (symbol + ".csv");

    // Load CSV data
    CSVData csv_data = CSVLoader::load(filepath.string());

    // Convert to GPU tensors
    data_.timestamps = CSVLoader::to_tensor_int64(csv_data.timestamps, device_);
    data_.open = CSVLoader::to_tensor(csv_data.open, device_);
    data_.high = CSVLoader::to_tensor(csv_data.high, device_);
    data_.low = CSVLoader::to_tensor(csv_data.low, device_);
    data_.close = CSVLoader::to_tensor(csv_data.close, device_);
    data_.volume = CSVLoader::to_tensor(csv_data.volume, device_);

    data_.symbol = symbol;
    data_.asset_class = CSVLoader::is_crypto_symbol(symbol) ?
                        AssetClass::CRYPTO : AssetClass::STOCK;
    data_.length = csv_data.timestamps.size();
}

torch::Tensor MarketState::get_observation(const torch::Tensor& current_indices) {
    // current_indices: [batch_size]
    // Returns: [batch_size, lookback_window, num_features]

    int batch_size = current_indices.size(0);
    int lookback = config_.LOOKBACK_WINDOW;
    int num_features = 5;  // OHLCV

    auto obs = torch::zeros(
        {batch_size, lookback, num_features},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_)
    );

    // For each environment, gather the last `lookback` timesteps
    for (int b = 0; b < batch_size; b++) {
        int current_idx = current_indices[b].item<int>();
        int start_idx = std::max(0, current_idx - lookback + 1);
        int actual_lookback = current_idx - start_idx + 1;

        // Stack OHLCV features
        auto features = torch::stack({
            data_.open.slice(0, start_idx, current_idx + 1),
            data_.high.slice(0, start_idx, current_idx + 1),
            data_.low.slice(0, start_idx, current_idx + 1),
            data_.close.slice(0, start_idx, current_idx + 1),
            data_.volume.slice(0, start_idx, current_idx + 1)
        }, /*dim=*/1);  // [actual_lookback, 5]

        // Pad if needed (for beginning of data)
        if (actual_lookback < lookback) {
            auto padding = torch::zeros(
                {lookback - actual_lookback, num_features},
                torch::TensorOptions().dtype(torch::kFloat32).device(device_)
            );
            features = torch::cat({padding, features}, /*dim=*/0);
        }

        obs[b] = normalize_features(features);
    }

    return obs;
}

torch::Tensor MarketState::get_current_prices(const torch::Tensor& current_indices) const {
    // Returns: [batch_size, 4] (open, high, low, close)
    int batch_size = current_indices.size(0);

    auto prices = torch::zeros(
        {batch_size, 4},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_)
    );

    for (int b = 0; b < batch_size; b++) {
        int idx = current_indices[b].item<int>();
        prices[b][0] = data_.open[idx];
        prices[b][1] = data_.high[idx];
        prices[b][2] = data_.low[idx];
        prices[b][3] = data_.close[idx];
    }

    return prices;
}

torch::Tensor MarketState::normalize_features(const torch::Tensor& raw_features) {
    // Normalize using mean and std per feature
    // raw_features: [lookback, num_features]

    // Calculate returns instead of raw prices for better normalization
    auto prices = raw_features.slice(1, 0, 4);  // OHLC
    auto volume = raw_features.slice(1, 4, 5);

    // Calculate percentage changes
    auto price_returns = (prices.slice(0, 1) - prices.slice(0, 0, -1)) /
                         (prices.slice(0, 0, -1) + 1e-8);

    // Pad first row with zeros
    auto first_row = torch::zeros({1, 4}, prices.options());
    price_returns = torch::cat({first_row, price_returns}, /*dim=*/0);

    // Normalize volume (log scale)
    auto normalized_volume = torch::log(volume + 1.0) / 10.0;  // Simple scaling

    // Combine normalized features
    return torch::cat({price_returns, normalized_volume}, /*dim=*/1);
}

} // namespace market_sim
