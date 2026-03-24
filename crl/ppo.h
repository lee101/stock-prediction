#pragma once
#include <torch/torch.h>
#include <string>
#include "config.h"

extern "C" {
#include "vec_env.h"
}

#include "obs_norm.h"
#include "policy.h"

struct PPOConfig {
    float gamma = CRL_DEFAULT_GAMMA;
    float gae_lambda = CRL_DEFAULT_GAE_LAMBDA;
    float clip_eps = CRL_DEFAULT_CLIP_EPS;
    float clip_eps_end = 0.05f;
    bool anneal_clip = false;
    float ent_coef = CRL_DEFAULT_ENT_COEF;
    float ent_coef_end = 0.02f;
    bool anneal_ent = false;
    float vf_coef = CRL_DEFAULT_VF_COEF;
    float max_grad_norm = CRL_DEFAULT_MAX_GRAD_NORM;
    int ppo_epochs = CRL_DEFAULT_PPO_EPOCHS;
    int minibatch_size = CRL_DEFAULT_MINIBATCH;
    float lr = CRL_DEFAULT_LR;
    float weight_decay = 0.0f;
    bool anneal_lr = true;
    std::string lr_schedule = "cosine";
    float lr_warmup_frac = 0.02f;
    float lr_min_ratio = 0.05f;
    bool obs_norm = false;
    int num_envs = CRL_DEFAULT_NUM_ENVS;
    int rollout_len = CRL_DEFAULT_ROLLOUT_LEN;
    int total_timesteps = 10000000;
    int hidden_size = CRL_DEFAULT_HIDDEN;
    unsigned int seed = 42;
    bool use_bf16 = false;
    int log_interval = 10;
    int checkpoint_interval = 50;
};

struct RolloutBuffer {
    torch::Tensor obs;       // [T, N, obs_size]
    torch::Tensor actions;   // [T, N]
    torch::Tensor logprobs;  // [T, N]
    torch::Tensor values;    // [T, N]
    torch::Tensor rewards;   // [T, N]
    torch::Tensor dones;     // [T, N]

    void allocate(int T, int N, int obs_size, torch::Device device);
};

struct TrainMetrics {
    float pg_loss = 0;
    float v_loss = 0;
    float entropy = 0;
    float approx_kl = 0;
    float clip_frac = 0;
    float explained_var = 0;
    float lr_current = 0;
};

struct TrainResult {
    float val_return = 0;
    float val_sortino = 0;
    float win_rate = 0;
    int num_trades = 0;
    std::string checkpoint_path;
};

class PPOTrainer {
public:
    PPOTrainer(const std::string& data_path, const PPOConfig& config,
               const VecEnvConfig& env_config, torch::Device device);
    ~PPOTrainer();

    TrainResult train(const std::string& checkpoint_dir, int time_budget_sec = 0);

    void collect_rollout();
    torch::Tensor compute_gae(torch::Tensor next_value);
    TrainMetrics ppo_update(torch::Tensor advantages, torch::Tensor returns);

    float get_lr(int update, int num_updates) const;

private:
    PPOConfig config_;
    VecEnv* env_;
    TradingPolicy policy_;
    RunningObsNorm obs_norm_;
    RolloutBuffer buffer_;
    torch::Device device_;
    std::unique_ptr<torch::optim::AdamW> optimizer_;

    int global_step_ = 0;
    int update_ = 0;
    int num_updates_ = 0;
};
