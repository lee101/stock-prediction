/**
 * Benchmark forecaster performance
 *
 * Tests throughput and latency of TorchScript-compiled Toto/Kronos models
 */

#include "forecaster.h"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace fast_market;

void print_separator() {
    std::cout << std::string(70, '=') << std::endl;
}

void benchmark_single_model(
    const std::string& model_path,
    int batch_size,
    int num_iterations
) {
    print_separator();
    std::cout << "📊 Benchmarking: " << model_path << std::endl;
    print_separator();

    // Configure forecaster
    TimeSeriesForecaster::Config config;
    config.model_path = model_path;
    config.sequence_length = 512;
    config.forecast_horizon = 64;
    config.num_features = 6;
    config.device = torch::kCUDA;
    config.use_fp16 = false;  // FP32 baseline

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Sequence length: " << config.sequence_length << std::endl;
    std::cout << "  Forecast horizon: " << config.forecast_horizon << std::endl;
    std::cout << "  Device: CUDA" << std::endl;
    std::cout << "  Precision: FP32" << std::endl;

    // Create forecaster
    auto forecaster = std::make_shared<TimeSeriesForecaster>(config);

    // Create random input
    auto input = torch::randn(
        {batch_size, config.sequence_length, config.num_features},
        torch::kCUDA
    );

    std::cout << "\n🔥 Running " << num_iterations << " iterations..." << std::endl;

    // Warmup
    for (int i = 0; i < 10; ++i) {
        forecaster->forecast_batch(input);
    }
    torch::cuda::synchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; ++i) {
        auto output = forecaster->forecast_batch(input);
    }

    torch::cuda::synchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double, std::milli>(end - start).count();

    // Calculate metrics
    double mean_latency_ms = duration / num_iterations;
    double throughput = (batch_size * num_iterations) / (duration / 1000.0);

    // Get detailed stats
    auto stats = forecaster->get_stats();

    print_separator();
    std::cout << "✅ RESULTS" << std::endl;
    print_separator();
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Total time: " << duration << " ms" << std::endl;
    std::cout << "  Mean latency: " << mean_latency_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << static_cast<int>(throughput) << " predictions/sec" << std::endl;
    std::cout << "  Latency std: " << stats.std_latency_ms << " ms" << std::endl;
    std::cout << std::endl;
}

void benchmark_fp16_speedup(
    const std::string& model_path,
    int batch_size
) {
    print_separator();
    std::cout << "⚡ FP16 vs FP32 Speedup Test" << std::endl;
    print_separator();

    // FP32
    TimeSeriesForecaster::Config config_fp32;
    config_fp32.model_path = model_path;
    config_fp32.device = torch::kCUDA;
    config_fp32.use_fp16 = false;

    auto forecaster_fp32 = std::make_shared<TimeSeriesForecaster>(config_fp32);

    // FP16
    TimeSeriesForecaster::Config config_fp16;
    config_fp16.model_path = model_path;
    config_fp16.device = torch::kCUDA;
    config_fp16.use_fp16 = true;

    auto forecaster_fp16 = std::make_shared<TimeSeriesForecaster>(config_fp16);

    auto input = torch::randn({batch_size, 512, 6}, torch::kCUDA);

    // Warmup
    for (int i = 0; i < 10; ++i) {
        forecaster_fp32->forecast_batch(input);
        forecaster_fp16->forecast_batch(input);
    }

    int num_iters = 100;

    // Benchmark FP32
    auto start_fp32 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iters; ++i) {
        forecaster_fp32->forecast_batch(input);
    }
    torch::cuda::synchronize();
    auto end_fp32 = std::chrono::high_resolution_clock::now();
    double time_fp32 = std::chrono::duration<double, std::milli>(end_fp32 - start_fp32).count();

    // Benchmark FP16
    auto start_fp16 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iters; ++i) {
        forecaster_fp16->forecast_batch(input);
    }
    torch::cuda::synchronize();
    auto end_fp16 = std::chrono::high_resolution_clock::now();
    double time_fp16 = std::chrono::duration<double, std::milli>(end_fp16 - start_fp16).count();

    double speedup = time_fp32 / time_fp16;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  FP32: " << time_fp32 / num_iters << " ms/iter" << std::endl;
    std::cout << "  FP16: " << time_fp16 / num_iters << " ms/iter" << std::endl;
    std::cout << "  Speedup: " << speedup << "x" << std::endl;
    std::cout << std::endl;
}

void benchmark_batch_scaling(const std::string& model_path) {
    print_separator();
    std::cout << "📈 Batch Size Scaling" << std::endl;
    print_separator();

    TimeSeriesForecaster::Config config;
    config.model_path = model_path;
    config.device = torch::kCUDA;

    auto forecaster = std::make_shared<TimeSeriesForecaster>(config);

    std::vector<int> batch_sizes = {1, 16, 64, 256, 1024};

    std::cout << std::setw(15) << "Batch Size"
              << std::setw(20) << "Latency (ms)"
              << std::setw(25) << "Throughput (pred/sec)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (int batch_size : batch_sizes) {
        auto input = torch::randn({batch_size, 512, 6}, torch::kCUDA);

        // Warmup
        for (int i = 0; i < 5; ++i) {
            forecaster->forecast_batch(input);
        }

        // Benchmark
        int num_iters = 50;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_iters; ++i) {
            forecaster->forecast_batch(input);
        }

        torch::cuda::synchronize();
        auto end = std::chrono::high_resolution_clock::now();

        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        double latency = duration / num_iters;
        double throughput = (batch_size * num_iters) / (duration / 1000.0);

        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(15) << batch_size
                  << std::setw(20) << latency
                  << std::setw(25) << static_cast<int>(throughput) << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "⚡ FORECASTER PERFORMANCE BENCHMARK" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Check CUDA
    if (!torch::cuda::is_available()) {
        std::cerr << "❌ CUDA not available!" << std::endl;
        return 1;
    }

    std::cout << "✓ CUDA available" << std::endl;
    std::cout << "  Device: " << torch::cuda::get_device_name(0) << std::endl;
    std::cout << std::endl;

    // Model path
    std::string model_path = "../compiled_models_torchscript/toto_fp32.pt";
    if (argc > 1) {
        model_path = argv[1];
    }

    std::cout << "Model: " << model_path << std::endl;
    std::cout << std::endl;

    try {
        // Benchmark single model
        benchmark_single_model(model_path, 256, 1000);

        // FP16 speedup
        benchmark_fp16_speedup(model_path, 256);

        // Batch scaling
        benchmark_batch_scaling(model_path);

        print_separator();
        std::cout << "✅ ALL BENCHMARKS COMPLETE" << std::endl;
        print_separator();

    } catch (const c10::Error& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        std::cerr << "\nTroubleshooting:" << std::endl;
        std::cerr << "1. Export model: python export_models_to_torchscript.py" << std::endl;
        std::cerr << "2. Check path: " << model_path << std::endl;
        return 1;
    }

    return 0;
}
