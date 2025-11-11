#pragma once

#include <torch/script.h>
#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace fast_market {

/**
 * Ultra-fast time series forecaster using compiled Toto/Kronos models
 *
 * No naive indicators - uses real DL models compiled to TorchScript for libtorch
 */
class TimeSeriesForecaster {
public:
    enum class ModelType {
        TOTO,      // Datadog Toto transformer
        KRONOS,    // Kronos tokenizer-predictor
        ENSEMBLE   // Ensemble of multiple models
    };

    struct Config {
        ModelType model_type = ModelType::TOTO;
        std::string model_path;
        int sequence_length = 512;
        int forecast_horizon = 64;
        int num_features = 6;  // OHLCV + volume
        torch::DeviceType device = torch::kCUDA;
        bool use_fp16 = false;  // Mixed precision inference
        bool use_compile = true;  // torch.compile optimization
    };

    explicit TimeSeriesForecaster(const Config& config);

    /**
     * Batch forecast for multiple assets
     *
     * @param input_series [batch_size, seq_len, features] - historical data
     * @return [batch_size, horizon, features] - predicted future values
     */
    torch::Tensor forecast_batch(const torch::Tensor& input_series);

    /**
     * Single asset forecast
     *
     * @param input_series [seq_len, features] - historical data
     * @return [horizon, features] - predicted future values
     */
    torch::Tensor forecast_single(const torch::Tensor& input_series);

    /**
     * Get expected return and confidence from forecast
     *
     * @param forecast [horizon, features] - predicted prices
     * @param current_price - current asset price
     * @return (expected_return, confidence)
     */
    std::pair<float, float> get_expected_return(
        const torch::Tensor& forecast,
        float current_price
    );

    /**
     * Warm up the model with dummy data to initialize CUDA kernels
     */
    void warmup();

    /**
     * Get inference latency statistics
     */
    struct Stats {
        double mean_latency_ms;
        double std_latency_ms;
        double throughput_samples_per_sec;
        size_t total_inferences;
    };
    Stats get_stats() const { return stats_; }

private:
    Config config_;
    torch::jit::script::Module model_;
    torch::Device device_;

    // Performance tracking
    Stats stats_;
    std::vector<double> latency_samples_;

    void update_stats(double latency_ms);
};


/**
 * Model ensemble for improved predictions
 */
class EnsembleForecaster {
public:
    struct Config {
        std::vector<std::string> model_paths;
        std::vector<float> model_weights;  // Ensemble weights
        int sequence_length = 512;
        int forecast_horizon = 64;
        torch::DeviceType device = torch::kCUDA;
    };

    explicit EnsembleForecaster(const Config& config);

    torch::Tensor forecast_batch(const torch::Tensor& input_series);

    void warmup();

private:
    Config config_;
    std::vector<std::shared_ptr<TimeSeriesForecaster>> forecasters_;
    torch::Tensor ensemble_weights_;
};


/**
 * Cached forecaster with intelligent prefetching
 *
 * Maintains a cache of recent forecasts and prefetches predictions
 * for likely next queries to minimize latency.
 */
class CachedForecaster {
public:
    struct Config {
        size_t cache_size = 10000;
        int prefetch_ahead = 10;  // Steps to prefetch
        bool enable_prefetch = true;
        TimeSeriesForecaster::Config forecaster_config;
    };

    explicit CachedForecaster(const Config& config);

    /**
     * Get forecast with caching
     *
     * Cache key is a hash of the input series
     */
    torch::Tensor forecast_cached(
        const torch::Tensor& input_series,
        const std::string& asset_id = ""
    );

    /**
     * Prefetch forecasts for likely next queries
     */
    void prefetch(const std::vector<torch::Tensor>& likely_inputs);

    /**
     * Clear cache
     */
    void clear_cache();

    /**
     * Get cache statistics
     */
    struct CacheStats {
        size_t hits;
        size_t misses;
        double hit_rate;
        size_t current_size;
    };
    CacheStats get_cache_stats() const;

private:
    Config config_;
    std::shared_ptr<TimeSeriesForecaster> forecaster_;

    // Cache: hash(input) -> forecast
    struct CacheEntry {
        torch::Tensor forecast;
        int64_t timestamp;
        int access_count;
    };
    std::unordered_map<size_t, CacheEntry> cache_;

    CacheStats cache_stats_;

    size_t compute_hash(const torch::Tensor& input) const;
};

} // namespace fast_market
