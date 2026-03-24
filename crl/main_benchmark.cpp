#include "ppo.h"
#include <chrono>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    std::string data_path;
    int num_envs = 128;
    int steps = 1000;
    int hidden = 1024;
    bool use_gpu = false;

    for (int i = 1; i < argc; i++) {
        auto arg = [&](const char* name) { return strcmp(argv[i], name) == 0; };
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if (arg("--data-path")) data_path = next();
        else if (arg("--num-envs")) num_envs = atoi(next());
        else if (arg("--steps")) steps = atoi(next());
        else if (arg("--hidden")) hidden = atoi(next());
        else if (arg("--gpu")) use_gpu = true;
    }

    if (data_path.empty()) {
        fprintf(stderr, "Usage: crl_benchmark --data-path PATH [--num-envs N] [--steps N] [--gpu]\n");
        return 1;
    }

    torch::Device device = use_gpu && torch::cuda::is_available()
        ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    VecEnvConfig env_cfg = {};
    env_cfg.max_steps = 720;
    env_cfg.fee_rate = 0.001f;
    env_cfg.max_leverage = 1.0f;
    env_cfg.periods_per_year = 8760.0f;
    env_cfg.action_allocation_bins = 5;
    env_cfg.action_level_bins = 5;
    env_cfg.action_max_offset_bps = 50.0f;
    env_cfg.reward_scale = 10.0f;
    env_cfg.reward_clip = 5.0f;
    env_cfg.cash_penalty = 0.01f;
    env_cfg.fill_probability = 1.0f;
    env_cfg.smooth_downside_temperature = 0.02f;

    VecEnv* ve = vec_env_create(data_path.c_str(), num_envs, &env_cfg);
    if (!ve) return 1;

    // benchmark env stepping only
    vec_env_reset(ve, 42);
    auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < num_envs; i++) {
            ve->act_buf[i] = rand() % ve->num_actions;
        }
        vec_env_step(ve);
    }
    auto t1 = std::chrono::steady_clock::now();
    double env_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double env_sps = (double)steps * num_envs / (env_ms / 1000.0);
    fprintf(stderr, "ENV: %d steps x %d envs in %.1fms = %.0f steps/sec\n",
            steps, num_envs, env_ms, env_sps);

    // benchmark with policy inference
    TradingPolicy policy(ve->obs_size, ve->num_actions, hidden);
    policy->to(device);
    policy->eval();

    vec_env_reset(ve, 42);
    auto t2 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; s++) {
        torch::NoGradGuard no_grad;
        auto obs = torch::from_blob(ve->obs_buf, {num_envs, ve->obs_size}, torch::kFloat32).to(device);
        auto [logits, value] = policy->forward(obs);
        auto action = logits.argmax(-1).to(torch::kInt32).cpu();
        std::memcpy(ve->act_buf, action.data_ptr<int>(), num_envs * sizeof(int));
        vec_env_step(ve);
    }
    auto t3 = std::chrono::steady_clock::now();
    double full_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double full_sps = (double)steps * num_envs / (full_ms / 1000.0);
    fprintf(stderr, "FULL (env+policy): %d steps x %d envs in %.1fms = %.0f steps/sec\n",
            steps, num_envs, full_ms, full_sps);

    vec_env_free(ve);
    return 0;
}
