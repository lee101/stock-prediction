#include "forecaster.h"
#include <chrono>
#include <iostream>
#include <numeric>
#include <cmath>

namespace fast_market {

TimeSeriesForecaster::TimeSeriesForecaster(const Config& config)
    : config_(config),
      device_(config.device, config.device == torch::kCUDA ? 0 : -1) {

    std::cout << "🔄 Loading forecaster model: " << config.model_path << std::endl;

    try {
        // Load TorchScript model
        model_ = torch::jit::load(config.model_path);
        model_.to(device_);
        model_.eval();

        std::cout << "✅ Model loaded successfully" << std::endl;

        // Apply optimizations
        if (config_.use_compile) {
            std::cout << "⚡ Applying torch.jit optimizations..." << std::endl;
            model_ = torch::jit::optimize_for_inference(model_);
        }

        // Warmup
        warmup();

    } catch (const c10::Error& e) {
        std::cerr << "❌ Error loading model: " << e.what() << std::endl;
        throw;
    }
}

void TimeSeriesForecaster::warmup() {
    std::cout << "🔥 Warming up model with dummy data..." << std::endl;

    // Create dummy input
    auto dummy_input = torch::randn(
        {4, config_.sequence_length, config_.num_features},
        torch::TensorOptions().device(device_).dtype(
            config_.use_fp16 ? torch::kFloat16 : torch::kFloat32
        )
    );

    // Run a few warmup iterations
    for (int i = 0; i < 10; ++i) {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(dummy_input);

        torch::NoGradGuard no_grad;
        auto output = model_.forward(inputs).toTensor();
    }

    if (device_.is_cuda()) {
        torch::cuda::synchronize();
    }

    std::cout << "✅ Warmup complete" << std::endl;
}

torch::Tensor TimeSeriesForecaster::forecast_batch(const torch::Tensor& input_series) {
    auto start = std::chrono::high_resolution_clock::now();

    TORCH_CHECK(input_series.dim() == 3, "Input must be [batch, seq_len, features]");
    TORCH_CHECK(input_series.size(1) == config_.sequence_length,
                "Sequence length mismatch");
    TORCH_CHECK(input_series.size(2) == config_.num_features,
                "Feature dimension mismatch");

    // Move to device if needed
    auto input = input_series.to(device_);

    // Mixed precision
    if (config_.use_fp16 && device_.is_cuda()) {
        input = input.to(torch::kFloat16);
    }

    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    torch::Tensor output;
    {
        torch::NoGradGuard no_grad;
        torch::cuda::CUDAStreamGuard stream_guard(torch::cuda::getStreamFromPool());

        output = model_.forward(inputs).toTensor();
    }

    // Convert back to fp32
    if (config_.use_fp16) {
        output = output.to(torch::kFloat32);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start).count();

    update_stats(duration);

    return output;
}

torch::Tensor TimeSeriesForecaster::forecast_single(const torch::Tensor& input_series) {
    TORCH_CHECK(input_series.dim() == 2, "Input must be [seq_len, features]");

    // Add batch dimension
    auto batched = input_series.unsqueeze(0);
    auto output = forecast_batch(batched);

    // Remove batch dimension
    return output.squeeze(0);
}

std::pair<float, float> TimeSeriesForecaster::get_expected_return(
    const torch::Tensor& forecast,
    float current_price
) {
    // forecast: [horizon, features] where features[3] is close price

    auto close_prices = forecast.select(1, 3);  // Get close prices
    auto returns = (close_prices - current_price) / current_price;

    // Expected return (mean)
    float expected_return = returns.mean().item<float>();

    // Confidence (inverse of std dev)
    float std_dev = returns.std().item<float>();
    float confidence = 1.0f / (1.0f + std_dev);

    return {expected_return, confidence};
}

void TimeSeriesForecaster::update_stats(double latency_ms) {
    latency_samples_.push_back(latency_ms);

    // Keep last 1000 samples
    if (latency_samples_.size() > 1000) {
        latency_samples_.erase(latency_samples_.begin());
    }

    // Update stats
    stats_.total_inferences++;

    if (!latency_samples_.empty()) {
        double sum = std::accumulate(latency_samples_.begin(), latency_samples_.end(), 0.0);
        stats_.mean_latency_ms = sum / latency_samples_.size();

        double sq_sum = std::inner_product(
            latency_samples_.begin(), latency_samples_.end(),
            latency_samples_.begin(), 0.0
        );
        stats_.std_latency_ms = std::sqrt(
            sq_sum / latency_samples_.size() - stats_.mean_latency_ms * stats_.mean_latency_ms
        );

        stats_.throughput_samples_per_sec = 1000.0 / stats_.mean_latency_ms;
    }
}


// ============================================================================
// EnsembleForecaster
// ============================================================================

EnsembleForecaster::EnsembleForecaster(const Config& config)
    : config_(config) {

    std::cout << "🎯 Creating ensemble forecaster with " << config_.model_paths.size()
              << " models" << std::endl;

    TORCH_CHECK(config_.model_paths.size() == config_.model_weights.size(),
                "Number of models and weights must match");

    // Normalize weights
    float weight_sum = std::accumulate(
        config_.model_weights.begin(),
        config_.model_weights.end(),
        0.0f
    );

    std::vector<float> normalized_weights;
    for (float w : config_.model_weights) {
        normalized_weights.push_back(w / weight_sum);
    }

    ensemble_weights_ = torch::tensor(
        normalized_weights,
        torch::TensorOptions().device(config_.device)
    );

    // Load all models
    for (size_t i = 0; i < config_.model_paths.size(); ++i) {
        TimeSeriesForecaster::Config fc_config;
        fc_config.model_path = config_.model_paths[i];
        fc_config.sequence_length = config_.sequence_length;
        fc_config.forecast_horizon = config_.forecast_horizon;
        fc_config.device = config_.device;

        forecasters_.push_back(
            std::make_shared<TimeSeriesForecaster>(fc_config)
        );

        std::cout << "  ✓ Model " << i + 1 << " loaded (weight: "
                  << normalized_weights[i] << ")" << std::endl;
    }
}

torch::Tensor EnsembleForecaster::forecast_batch(const torch::Tensor& input_series) {
    std::vector<torch::Tensor> predictions;

    // Get predictions from all models
    for (auto& forecaster : forecasters_) {
        auto pred = forecaster->forecast_batch(input_series);
        predictions.push_back(pred);
    }

    // Stack predictions [num_models, batch, horizon, features]
    auto stacked = torch::stack(predictions, 0);

    // Apply weighted average
    // ensemble_weights: [num_models]
    // stacked: [num_models, batch, horizon, features]

    auto weights = ensemble_weights_
        .view({-1, 1, 1, 1})
        .expand_as(stacked);

    auto weighted_preds = stacked * weights;
    auto ensemble_pred = weighted_preds.sum(0);

    return ensemble_pred;
}

void EnsembleForecaster::warmup() {
    for (auto& forecaster : forecasters_) {
        forecaster->warmup();
    }
}


// ============================================================================
// CachedForecaster
// ============================================================================

CachedForecaster::CachedForecaster(const Config& config)
    : config_(config) {

    forecaster_ = std::make_shared<TimeSeriesForecaster>(
        config_.forecaster_config
    );

    cache_stats_ = {0, 0, 0.0, 0};
}

size_t CachedForecaster::compute_hash(const torch::Tensor& input) const {
    // Simple hash based on tensor data
    // In production, use a more robust hash function

    auto data_ptr = input.data_ptr<float>();
    size_t hash = 0;

    // Hash first, middle, and last few elements
    constexpr int sample_points = 100;
    int step = std::max(1, (int)(input.numel() / sample_points));

    for (int i = 0; i < input.numel(); i += step) {
        // FNV-1a hash
        hash ^= std::hash<float>{}(data_ptr[i]);
        hash *= 0x100000001b3;
    }

    return hash;
}

torch::Tensor CachedForecaster::forecast_cached(
    const torch::Tensor& input_series,
    const std::string& asset_id
) {
    size_t hash = compute_hash(input_series);

    // Check cache
    auto it = cache_.find(hash);
    if (it != cache_.end()) {
        cache_stats_.hits++;
        it->second.access_count++;
        cache_stats_.hit_rate = (double)cache_stats_.hits /
                                (cache_stats_.hits + cache_stats_.misses);
        return it->second.forecast.clone();
    }

    // Cache miss - compute forecast
    cache_stats_.misses++;

    auto forecast = forecaster_->forecast_single(input_series);

    // Store in cache
    CacheEntry entry;
    entry.forecast = forecast.clone();
    entry.timestamp = std::chrono::system_clock::now().time_since_epoch().count();
    entry.access_count = 1;

    cache_[hash] = entry;
    cache_stats_.current_size = cache_.size();

    // Evict if cache too large (LRU)
    if (cache_.size() > config_.cache_size) {
        // Find least recently used
        auto oldest = cache_.begin();
        for (auto it = cache_.begin(); it != cache_.end(); ++it) {
            if (it->second.timestamp < oldest->second.timestamp) {
                oldest = it;
            }
        }
        cache_.erase(oldest);
    }

    cache_stats_.hit_rate = (double)cache_stats_.hits /
                            (cache_stats_.hits + cache_stats_.misses);

    return forecast;
}

void CachedForecaster::prefetch(const std::vector<torch::Tensor>& likely_inputs) {
    if (!config_.enable_prefetch) return;

    // Prefetch in background (could use async here)
    for (const auto& input : likely_inputs) {
        forecast_cached(input);
    }
}

void CachedForecaster::clear_cache() {
    cache_.clear();
    cache_stats_ = {0, 0, 0.0, 0};
}

CachedForecaster::CacheStats CachedForecaster::get_cache_stats() const {
    return cache_stats_;
}

} // namespace fast_market
