#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include "market_env.h"
#include "market_config.h"
#include "cuda_kernels.cuh"
#include "execution_policy.h"

using namespace market_sim;

// Fast PPO policy network with learned execution
struct FastPolicyNet : torch::nn::Module {
    FastPolicyNet(int obs_dim, int hidden_dim = 256) {
        // Main policy network
        fc1 = register_module("fc1", torch::nn::Linear(obs_dim, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_dim, hidden_dim));

        // Action head (mean and std)
        fc_mean = register_module("fc_mean", torch::nn::Linear(hidden_dim, 1));
        fc_std = register_module("fc_std", torch::nn::Linear(hidden_dim, 1));

        // Value head
        fc_value = register_module("fc_value", torch::nn::Linear(hidden_dim, 1));
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::relu(fc3->forward(x));

        auto mean = torch::tanh(fc_mean->forward(x)) * 1.5;  // [-1.5, 1.5]
        auto std = torch::softplus(fc_std->forward(x)) + 0.01;
        auto value = fc_value->forward(x);

        return std::make_tuple(mean, std, value);
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    torch::nn::Linear fc_mean{nullptr}, fc_std{nullptr}, fc_value{nullptr};
};

// Training metrics
struct TrainingMetrics {
    float mean_reward = 0.0f;
    float mean_pnl = 0.0f;
    float mean_fees = 0.0f;
    float mean_sharpe = 0.0f;
    int total_trades = 0;
    float policy_loss = 0.0f;
    float value_loss = 0.0f;
    float total_loss = 0.0f;
};

int main(int argc, char* argv[]) {
    std::cout << "\n=== Ultra-Fast PufferLib3 C++ Market Simulator ===" << std::endl;
    std::cout << "With CUDA Kernels + Learned Execution + Multi-Strategy Training\n" << std::endl;

    // Check CUDA
    if (!torch::cuda::is_available()) {
        std::cerr << "ERROR: CUDA is required for this training!" << std::endl;
        return 1;
    }
    std::cout << "CUDA Device: " << torch::cuda::device_count() << " GPUs available" << std::endl;

    // Configuration
    MarketConfig config;
    config.log_dir = "../logs";
    config.device = "cuda:0";
    config.use_mixed_precision = true;

    auto device = torch::Device(config.device);
    std::cout << "Using device: " << config.device << std::endl;

    // Create environment
    std::cout << "\nInitializing market environment..." << std::endl;
    MarketEnvironment env(config);

    // Load symbols
    std::vector<std::string> symbols = {
        "AAPL", "AMZN", "MSFT", "GOOGL", "NVDA",
        "BTCUSD", "ETHUSD"
    };

    std::cout << "Loading " << symbols.size() << " symbols: ";
    for (const auto& sym : symbols) std::cout << sym << " ";
    std::cout << std::endl;

    try {
        env.load_symbols(symbols);
    } catch (const std::exception& e) {
        std::cerr << "Error loading symbols: " << e.what() << std::endl;
        return 1;
    }

    // Initialize policy network
    int obs_dim = env.get_observation_dim();
    std::cout << "Observation dimension: " << obs_dim << std::endl;

    auto policy = std::make_shared<FastPolicyNet>(obs_dim);
    policy->to(device);

    // Initialize execution policy trainer
    auto exec_policy_trainer = std::make_shared<ExecutionPolicyTrainer>(
        obs_dim, device, 1e-4
    );

    // Initialize multi-strategy wrapper
    auto strategy_wrapper = std::make_shared<MultiStrategyWrapper>(
        config.NUM_PARALLEL_ENVS, device
    );

    // Optimizers
    auto policy_optimizer = std::make_shared<torch::optim::Adam>(
        policy->parameters(), torch::optim::AdamOptions(3e-4)
    );

    // Training hyperparameters
    int num_updates = 2000;
    int steps_per_update = 256;
    int num_epochs = 4;       // PPO epochs
    int minibatch_size = 1024;
    float gamma = 0.99f;
    float gae_lambda = 0.95f;
    float clip_epsilon = 0.2f;
    float entropy_coef = 0.01f;
    float value_coef = 0.5f;

    std::cout << "\n=== Training Configuration ===" << std::endl;
    std::cout << "Updates: " << num_updates << std::endl;
    std::cout << "Steps per update: " << steps_per_update << std::endl;
    std::cout << "Batch size: " << config.NUM_PARALLEL_ENVS << std::endl;
    std::cout << "PPO epochs: " << num_epochs << std::endl;
    std::cout << "Minibatch size: " << minibatch_size << std::endl;
    std::cout << "Gamma: " << gamma << std::endl;
    std::cout << "GAE Lambda: " << gae_lambda << std::endl;

    // Set training mode
    env.set_training_mode(true);

    // Reset environments
    auto initial_indices = torch::arange(config.NUM_PARALLEL_ENVS, torch::kInt64).to(device);
    auto output = env.reset(initial_indices);

    auto start_time = std::chrono::high_resolution_clock::now();
    int total_steps = 0;

    std::cout << "\n=== Starting Training ===" << std::endl;
    std::cout << "Expected throughput: >500K steps/sec with CUDA kernels\n" << std::endl;

    for (int update = 0; update < num_updates; update++) {
        // Randomize strategies every N updates (curriculum learning)
        if (update % 50 == 0) {
            strategy_wrapper->randomize_strategies();
        }

        // === Rollout Collection ===
        std::vector<torch::Tensor> obs_buf, act_buf, rew_buf, val_buf;
        std::vector<torch::Tensor> logp_buf, done_buf;
        std::vector<torch::Tensor> price_buf, exec_timing_buf, exec_aggr_buf;
        std::vector<torch::Tensor> fee_buf, realized_buf;

        auto obs = output.observations;

        for (int step = 0; step < steps_per_update; step++) {
            torch::NoGradGuard no_grad;

            // Get action from policy
            auto [mean, std, value] = policy->forward(obs);

            // Sample action
            auto noise = torch::randn_like(mean);
            auto action_dist = mean + std * noise;
            auto action = torch::clamp(action_dist, -config.MAX_LEVERAGE, config.MAX_LEVERAGE);

            // Calculate log probability
            auto logprob = -0.5f * torch::pow((action - mean) / (std + 1e-8f), 2) -
                          torch::log(std + 1e-8f) - 0.5f * std::log(2.0f * M_PI);

            // Use actual OHLC data for execution policy
            auto prices = env.peek_next_prices().detach();

            // Get learned execution parameters
            auto [exec_timing, exec_aggression] = exec_policy_trainer->get_execution_params(
                obs, action, prices, true  // is_training=true
            );

            // Step environment (uses CUDA kernels internally)
            output = env.step(action.squeeze(1));
            auto normalized_fees = output.fees_paid / config.INITIAL_CAPITAL;
            auto normalized_leverage = output.leverage_costs / config.INITIAL_CAPITAL;
            auto normalized_realized = output.realized_pnl / config.INITIAL_CAPITAL;

            // Apply strategy-specific reward shaping
            output.rewards = strategy_wrapper->shape_rewards(
                output.rewards, normalized_fees, normalized_leverage, output.days_held
            );

            // Store rollout
            obs_buf.push_back(obs);
            act_buf.push_back(action);
            rew_buf.push_back(output.rewards);
            val_buf.push_back(value.squeeze(1));
            logp_buf.push_back(logprob.squeeze(1));
            done_buf.push_back(output.dones.to(torch::kFloat32));
            price_buf.push_back(prices);
            exec_timing_buf.push_back(exec_timing);
            exec_aggr_buf.push_back(exec_aggression);
            fee_buf.push_back(normalized_fees);
            realized_buf.push_back(normalized_realized);

            obs = output.observations;
            total_steps += config.NUM_PARALLEL_ENVS;
        }

        // === Compute Advantages using PufferLib CUDA V-trace ===
        {
            torch::NoGradGuard no_grad;

            auto [_, __, next_value] = policy->forward(obs);
            next_value = next_value.squeeze(1).detach();

            // Stack buffers
            auto all_values = torch::stack(val_buf, 0);      // [steps, batch]
            auto all_rewards = torch::stack(rew_buf, 0);
            auto all_dones = torch::stack(done_buf, 0);

            // Append next_value for V-trace
            all_values = torch::cat({all_values, next_value.unsqueeze(0)}, 0);
            all_rewards = torch::cat({all_rewards, torch::zeros_like(next_value).unsqueeze(0)}, 0);
            all_dones = torch::cat({all_dones, torch::zeros_like(next_value).unsqueeze(0)}, 0);

            // Importance sampling (1.0 for on-policy)
            auto importance = torch::ones_like(all_rewards);

            // Advantages buffer
            auto advantages = torch::zeros_like(all_rewards);

            // Use PufferLib's ultra-fast CUDA V-trace kernel!
            cuda::compute_vtrace_advantages(
                all_values, all_rewards, all_dones, importance, advantages,
                gamma, gae_lambda, 1.0f, 1.0f
            );

            // Remove last timestep (it was just for bootstrapping)
            advantages = advantages.narrow(0, 0, steps_per_update);

            // Flatten for training
            auto all_obs = torch::cat(obs_buf, 0);
            auto all_actions = torch::cat(act_buf, 0);
            auto all_logprobs = torch::cat(logp_buf, 0);
            auto all_advs = advantages.flatten();
            auto all_vals = torch::cat(val_buf, 0);
            auto all_returns = all_advs + all_vals;

            // Normalize advantages
            all_advs = (all_advs - all_advs.mean()) / (all_advs.std() + 1e-8f);

            // === PPO Update ===
            int total_samples = all_obs.size(0);
            TrainingMetrics metrics;

            for (int epoch = 0; epoch < num_epochs; epoch++) {
                // Shuffle data
                auto indices = torch::randperm(total_samples, device);

                for (int start = 0; start < total_samples; start += minibatch_size) {
                    int end = std::min(start + minibatch_size, total_samples);
                    auto mb_indices = indices.narrow(0, start, end - start);

                    auto mb_obs = all_obs.index_select(0, mb_indices);
                    auto mb_actions = all_actions.index_select(0, mb_indices);
                    auto mb_logprobs = all_logprobs.index_select(0, mb_indices);
                    auto mb_advs = all_advs.index_select(0, mb_indices);
                    auto mb_returns = all_returns.index_select(0, mb_indices);

                    // Forward pass
                    auto [new_mean, new_std, new_value] = policy->forward(mb_obs);
                    auto new_logprob = -0.5f * torch::pow((mb_actions - new_mean) / (new_std + 1e-8f), 2) -
                                      torch::log(new_std + 1e-8f) - 0.5f * std::log(2.0f * M_PI);
                    new_logprob = new_logprob.squeeze(1);
                    new_value = new_value.squeeze(1);

                    // PPO loss
                    auto ratio = torch::exp(new_logprob - mb_logprobs);
                    auto surr1 = ratio * mb_advs;
                    auto surr2 = torch::clamp(ratio, 1.0f - clip_epsilon, 1.0f + clip_epsilon) * mb_advs;
                    auto policy_loss = -torch::min(surr1, surr2).mean();

                    // Value loss
                    auto value_loss = torch::mse_loss(new_value, mb_returns);

                    // Entropy bonus
                    auto entropy = (torch::log(new_std) + 0.5f * std::log(2.0f * M_PI * M_E)).mean();

                    // Total loss
                    auto loss = policy_loss + value_coef * value_loss - entropy_coef * entropy;

                    // Optimize
                    policy_optimizer->zero_grad();
                    loss.backward();
                    torch::nn::utils::clip_grad_norm_(policy->parameters(), 0.5f);
                    policy_optimizer->step();

                    metrics.policy_loss = policy_loss.item<float>();
                    metrics.value_loss = value_loss.item<float>();
                    metrics.total_loss = loss.item<float>();
                }
            }

            // Update execution policy
            auto all_prices = torch::cat(price_buf, 0);
            auto realized_pnl = torch::cat(realized_buf, 0).flatten();
            auto fees_paid = torch::cat(fee_buf, 0).flatten();
            exec_policy_trainer->update(
                all_obs, all_actions, all_prices,
                realized_pnl, fees_paid, true
            );

            // Compute metrics
            metrics.mean_reward = all_rewards.flatten().mean().item<float>();

            // === Logging ===
            if (update % 10 == 0) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::high_resolution_clock::now() - start_time
                ).count();

                float steps_per_sec = total_steps / (float)elapsed;

                std::cout << std::fixed << std::setprecision(4);
                std::cout << "Update " << std::setw(4) << update << "/" << num_updates
                         << " | Steps: " << total_steps
                         << " | Reward: " << std::setw(7) << metrics.mean_reward
                         << " | PL: " << std::setw(6) << metrics.policy_loss
                         << " | VL: " << std::setw(6) << metrics.value_loss
                         << " | SPS: " << std::setw(7) << (int)steps_per_sec
                         << " | Time: " << elapsed << "s" << std::endl;

                env.flush_logs();
            }
        }
    }

    std::cout << "\n=== Training Complete ===" << std::endl;

    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::high_resolution_clock::now() - start_time
    ).count();

    float steps_per_sec = total_steps / (float)elapsed;

    std::cout << "Total steps: " << total_steps << std::endl;
    std::cout << "Total time: " << elapsed << "s" << std::endl;
    std::cout << "Average throughput: " << (int)steps_per_sec << " steps/sec" << std::endl;
    std::cout << "Expected: >500K steps/sec (check GPU utilization!)" << std::endl;

    // Save models
    std::cout << "\nSaving models..." << std::endl;
    torch::save(policy, "fast_market_policy.pt");
    exec_policy_trainer->save("execution_policy.pt");

    std::cout << "Final statistics..." << std::endl;
    env.flush_logs();

    std::cout << "\nDone! Logs saved to: " << config.log_dir << std::endl;
    std::cout << "\nNext steps:" << std::endl;
    std::cout << "1. Check GPU utilization with nvidia-smi" << std::endl;
    std::cout << "2. Analyze logs for PnL performance" << std::endl;
    std::cout << "3. Run validation with conservative execution" << std::endl;
    std::cout << "4. Compare train vs validation PnL (expect 30-50% drop)" << std::endl;

    return 0;
}
