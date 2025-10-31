#include <torch/torch.h>
#include <iostream>
#include "market_env.h"
#include "market_config.h"

using namespace market_sim;

int main(int argc, char* argv[]) {
    std::cout << "=== Market Simulator Test ===" << std::endl;

    // Check CUDA
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Device count: " << torch::cuda::device_count() << std::endl;
    } else {
        std::cout << "CUDA not available, using CPU" << std::endl;
    }

    // Test basic LibTorch operations
    std::cout << "\nTesting LibTorch GPU operations..." << std::endl;
    auto device = torch::Device("cuda:0");
    auto tensor = torch::randn({1000, 1000}, device);
    auto result = torch::matmul(tensor, tensor);
    std::cout << "GPU tensor test passed! Result shape: "
              << result.size(0) << "x" << result.size(1) << std::endl;

    // Create environment
    std::cout << "\nCreating market environment..." << std::endl;
    MarketConfig config;
    config.log_dir = "./test_logs";
    config.device = "cuda:0";

    MarketEnvironment env(config);

    // Load a single symbol for testing
    std::cout << "Loading test symbol: AAPL" << std::endl;
    try {
        env.load_symbols({"AAPL"});
        std::cout << "Symbol loaded successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Reset environment
    std::cout << "\nResetting environment..." << std::endl;
    auto indices = torch::arange(config.NUM_PARALLEL_ENVS, torch::kInt64).to(device);
    auto output = env.reset(indices);

    std::cout << "Observation shape: ";
    for (int i = 0; i < output.observations.dim(); i++) {
        std::cout << output.observations.size(i);
        if (i < output.observations.dim() - 1) std::cout << "x";
    }
    std::cout << std::endl;

    // Run a few steps
    std::cout << "\nRunning test episode..." << std::endl;
    env.set_training_mode(false);  // Testing mode

    float total_reward = 0.0f;
    for (int step = 0; step < 50; step++) {
        // Random actions
        auto actions = torch::randn({config.NUM_PARALLEL_ENVS}, device) * 0.5f;

        output = env.step(actions);

        auto mean_reward = output.rewards.mean().item<float>();
        total_reward += mean_reward;

        if (step % 10 == 0) {
            std::cout << "Step " << step << " | Mean Reward: " << mean_reward << std::endl;
        }
    }

    std::cout << "\nTest complete!" << std::endl;
    std::cout << "Average reward per step: " << (total_reward / 50.0f) << std::endl;

    // Flush logs
    env.flush_logs();

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
