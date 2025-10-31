#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include "market_env.h"
#include "market_config.h"

using namespace market_sim;

// Simple PPO-style policy network
struct PolicyNet : torch::nn::Module {
    PolicyNet(int obs_dim, int hidden_dim = 256) {
        fc1 = register_module("fc1", torch::nn::Linear(obs_dim, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
        fc_mean = register_module("fc_mean", torch::nn::Linear(hidden_dim, 1));
        fc_std = register_module("fc_std", torch::nn::Linear(hidden_dim, 1));
        fc_value = register_module("fc_value", torch::nn::Linear(hidden_dim, 1));
    }

    // Forward pass returns: mean, std, value
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));

        auto mean = torch::tanh(fc_mean->forward(x)) * 1.5;  // Scale to [-1.5, 1.5] leverage
        auto std = torch::softplus(fc_std->forward(x)) + 0.01;  // Ensure positive
        auto value = fc_value->forward(x);

        return std::make_tuple(mean, std, value);
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc_mean{nullptr}, fc_std{nullptr}, fc_value{nullptr};
};

int main(int argc, char* argv[]) {
    std::cout << "=== PufferLib3 C++ Market Simulator ===" << std::endl;
    std::cout << "Training on GPU with LibTorch" << std::endl;

    // Check CUDA availability
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available! Training on CPU..." << std::endl;
    } else {
        std::cout << "CUDA available! Device count: " << torch::cuda::device_count() << std::endl;
    }

    // Configuration
    MarketConfig config;
    config.log_dir = "../logs";
    config.device = "cuda:0";
    config.use_mixed_precision = true;

    // Create environment
    std::cout << "\nInitializing market environment..." << std::endl;
    MarketEnvironment env(config);

    // Load training symbols
    std::vector<std::string> symbols = {"AAPL", "AMZN", "MSFT", "GOOGL", "NVDA", "BTCUSD", "ETHUSD"};
    std::cout << "Loading symbols: ";
    for (const auto& sym : symbols) {
        std::cout << sym << " ";
    }
    std::cout << std::endl;

    try {
        env.load_symbols(symbols);
    } catch (const std::exception& e) {
        std::cerr << "Error loading symbols: " << e.what() << std::endl;
        return 1;
    }

    // Initialize policy network
    std::cout << "\nInitializing policy network..." << std::endl;
    int obs_dim = env.get_observation_dim();
    std::cout << "Observation dimension: " << obs_dim << std::endl;

    auto policy = std::make_shared<PolicyNet>(obs_dim);
    auto device = torch::Device(config.device);
    policy->to(device);

    // Optimizer
    torch::optim::Adam optimizer(policy->parameters(), torch::optim::AdamOptions(3e-4));

    // Training parameters
    int num_updates = 1000;
    int steps_per_update = 256;
    float gamma = 0.99f;
    float gae_lambda = 0.95f;

    std::cout << "\nStarting training..." << std::endl;
    std::cout << "Updates: " << num_updates << std::endl;
    std::cout << "Steps per update: " << steps_per_update << std::endl;
    std::cout << "Batch size: " << config.NUM_PARALLEL_ENVS << std::endl;

    env.set_training_mode(true);

    // Reset environment
    auto initial_indices = torch::arange(config.NUM_PARALLEL_ENVS, torch::kInt64).to(device);
    auto output = env.reset(initial_indices);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int update = 0; update < num_updates; update++) {
        // Collect rollout
        std::vector<torch::Tensor> obs_buffer, action_buffer, reward_buffer, value_buffer;
        std::vector<torch::Tensor> logprob_buffer, done_buffer;

        auto obs = output.observations;

        for (int step = 0; step < steps_per_update; step++) {
            // Get action from policy
            auto [mean, std, value] = policy->forward(obs);

            // Sample action
            torch::NoGradGuard no_grad;
            auto noise = torch::randn_like(mean);
            auto action_dist = mean + std * noise;
            auto action = torch::clamp(action_dist, -config.MAX_LEVERAGE, config.MAX_LEVERAGE);

            // Calculate log probability
            auto logprob = -0.5 * torch::pow((action - mean) / (std + 1e-8), 2) -
                          torch::log(std + 1e-8) - 0.5 * std::log(2 * M_PI);

            // Step environment
            output = env.step(action.squeeze(1));

            // Store transition
            obs_buffer.push_back(obs);
            action_buffer.push_back(action);
            reward_buffer.push_back(output.rewards);
            value_buffer.push_back(value.squeeze(1));
            logprob_buffer.push_back(logprob.squeeze(1));
            done_buffer.push_back(output.dones.to(torch::kFloat32));

            obs = output.observations;
        }

        // Compute advantages using GAE
        auto [_, __, next_value] = policy->forward(obs);
        next_value = next_value.squeeze(1).detach();

        std::vector<torch::Tensor> advantages;
        auto gae = torch::zeros_like(next_value);

        for (int t = steps_per_update - 1; t >= 0; t--) {
            auto next_val = (t == steps_per_update - 1) ? next_value : value_buffer[t + 1];
            auto delta = reward_buffer[t] + gamma * next_val * (1 - done_buffer[t]) - value_buffer[t];
            gae = delta + gamma * gae_lambda * (1 - done_buffer[t]) * gae;
            advantages.insert(advantages.begin(), gae);
        }

        // Flatten buffers
        auto all_obs = torch::cat(obs_buffer, 0);
        auto all_actions = torch::cat(action_buffer, 0);
        auto all_logprobs = torch::cat(logprob_buffer, 0);
        auto all_advantages = torch::cat(advantages, 0);
        auto all_values = torch::cat(value_buffer, 0);
        auto all_returns = all_advantages + all_values;

        // Normalize advantages
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8);

        // PPO update
        optimizer.zero_grad();

        auto [new_mean, new_std, new_value] = policy->forward(all_obs);
        auto new_logprob = -0.5 * torch::pow((all_actions - new_mean) / (new_std + 1e-8), 2) -
                           torch::log(new_std + 1e-8) - 0.5 * std::log(2 * M_PI);
        new_logprob = new_logprob.squeeze(1);
        new_value = new_value.squeeze(1);

        // Policy loss (simplified PPO)
        auto ratio = torch::exp(new_logprob - all_logprobs);
        auto policy_loss = -(ratio * all_advantages).mean();

        // Value loss
        auto value_loss = torch::mse_loss(new_value, all_returns);

        // Total loss
        auto loss = policy_loss + 0.5 * value_loss;

        loss.backward();
        torch::nn::utils::clip_grad_norm_(policy->parameters(), 0.5);
        optimizer.step();

        // Logging
        if (update % 10 == 0) {
            auto mean_reward = torch::cat(reward_buffer, 0).mean().item<float>();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - start_time
            ).count();

            std::cout << "Update " << update << "/" << num_updates
                     << " | Mean Reward: " << std::fixed << std::setprecision(4) << mean_reward
                     << " | Loss: " << loss.item<float>()
                     << " | Time: " << elapsed << "s" << std::endl;

            // Flush logs
            env.flush_logs();
        }
    }

    std::cout << "\nTraining complete! Saving model..." << std::endl;
    torch::save(policy, "market_policy.pt");

    std::cout << "Writing final statistics..." << std::endl;
    env.flush_logs();

    std::cout << "\nDone! Logs saved to: " << config.log_dir << std::endl;
    return 0;
}
