#include "ppo.h"
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <sys/stat.h>

void RolloutBuffer::allocate(int T, int N, int obs_size, torch::Device device) {
    obs = torch::zeros({T, N, obs_size}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    actions = torch::zeros({T, N}, torch::TensorOptions().dtype(torch::kLong).device(device));
    logprobs = torch::zeros({T, N}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    values = torch::zeros({T, N}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    rewards = torch::zeros({T, N}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    dones = torch::zeros({T, N}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
}

PPOTrainer::PPOTrainer(const std::string& data_path, const PPOConfig& config,
                       const VecEnvConfig& env_config, torch::Device device)
    : config_(config), device_(device), policy_(nullptr) {

    env_ = vec_env_create(data_path.c_str(), config.num_envs, &env_config);
    if (!env_) {
        throw std::runtime_error("Failed to create vectorized environment");
    }

    policy_ = TradingPolicy(env_->obs_size, env_->num_actions, config.hidden_size);
    policy_->to(device);

    if (config.obs_norm) {
        obs_norm_.init(env_->obs_size);
    }

    buffer_.allocate(config.rollout_len, config.num_envs, env_->obs_size, device);

    int batch_size = config.num_envs * config.rollout_len;
    num_updates_ = config.total_timesteps / batch_size;

    auto adamw_options = torch::optim::AdamWOptions(config.lr)
        .weight_decay(config.weight_decay)
        .eps(1e-5);
    optimizer_ = std::make_unique<torch::optim::AdamW>(policy_->parameters(), adamw_options);

    // enable TF32
    at::globalContext().setAllowTF32CubLAS(true);

    fprintf(stderr, "PPOTrainer: obs=%d actions=%d hidden=%d params=%ld updates=%d\n",
            env_->obs_size, env_->num_actions, config.hidden_size,
            (long)policy_->parameters().size(), num_updates_);
}

PPOTrainer::~PPOTrainer() {
    if (env_) vec_env_free(env_);
}

float PPOTrainer::get_lr(int update, int num_updates) const {
    float base_lr = config_.lr;
    if (config_.lr_schedule == "cosine") {
        int warmup_updates = (int)(num_updates * config_.lr_warmup_frac);
        if (update <= warmup_updates) {
            return base_lr * (float)update / std::max(warmup_updates, 1);
        }
        float progress = (float)(update - warmup_updates) / std::max(num_updates - warmup_updates, 1);
        float cosine_decay = 0.5f * (1.0f + std::cos(M_PI * progress));
        return base_lr * std::max(cosine_decay, config_.lr_min_ratio);
    } else if (config_.anneal_lr) {
        float frac = 1.0f - (float)update / std::max(num_updates, 1);
        return base_lr * std::max(frac, config_.lr_min_ratio);
    }
    return base_lr;
}

void PPOTrainer::collect_rollout() {
    policy_->eval();
    torch::NoGradGuard no_grad;

    int N = config_.num_envs;
    int obs_size = env_->obs_size;

    for (int step = 0; step < config_.rollout_len; step++) {
        // copy obs from C env buffer to GPU
        auto obs_cpu = torch::from_blob(env_->obs_buf, {N, obs_size}, torch::kFloat32);

        if (config_.obs_norm) {
            obs_norm_.update(env_->obs_buf, N);
            auto norm_buf = std::vector<float>(N * obs_size);
            obs_norm_.normalize(env_->obs_buf, norm_buf.data(), N);
            obs_cpu = torch::from_blob(norm_buf.data(), {N, obs_size}, torch::kFloat32).clone();
        }

        buffer_.obs[step] = obs_cpu.to(device_, /*non_blocking=*/true);

        auto [logits, value] = policy_->forward(buffer_.obs[step]);
        auto probs = torch::softmax(logits, -1);
        auto action = torch::multinomial(probs, 1).squeeze(-1);
        auto log_probs_all = torch::log_softmax(logits, -1);

        buffer_.actions[step] = action;
        buffer_.logprobs[step] = log_probs_all.gather(1, action.unsqueeze(-1)).squeeze(-1);
        buffer_.values[step] = value;

        // copy actions to C env
        auto act_cpu = action.to(torch::kInt32, torch::kCPU);
        std::memcpy(env_->act_buf, act_cpu.data_ptr<int>(), N * sizeof(int));

        vec_env_step(env_);

        buffer_.rewards[step] = torch::from_blob(env_->rew_buf, {N}, torch::kFloat32).to(device_, true);
        buffer_.dones[step] = torch::from_blob(env_->done_buf, {N}, torch::kUInt8).to(torch::kFloat32).to(device_, true);

        global_step_ += N;
    }
}

torch::Tensor PPOTrainer::compute_gae(torch::Tensor next_value) {
    int T = config_.rollout_len;
    int N = config_.num_envs;
    float gamma = config_.gamma;
    float lam = config_.gae_lambda;

    auto advantages = torch::zeros({T, N}, buffer_.rewards.options());
    auto last_gae = torch::zeros({N}, buffer_.rewards.options());

    for (int t = T - 1; t >= 0; t--) {
        torch::Tensor next_val;
        torch::Tensor next_non_terminal;
        if (t == T - 1) {
            next_val = next_value;
            // for the last step, check if any env terminated
            next_non_terminal = 1.0f - buffer_.dones[t];
        } else {
            next_val = buffer_.values[t + 1];
            next_non_terminal = 1.0f - buffer_.dones[t];
        }
        auto delta = buffer_.rewards[t] + gamma * next_val * next_non_terminal - buffer_.values[t];
        last_gae = delta + gamma * lam * next_non_terminal * last_gae;
        advantages[t] = last_gae;
    }
    return advantages;
}

TrainMetrics PPOTrainer::ppo_update(torch::Tensor advantages, torch::Tensor returns) {
    policy_->train();

    int T = config_.rollout_len;
    int N = config_.num_envs;
    int batch_size = T * N;

    // flatten rollout
    auto b_obs = buffer_.obs.reshape({batch_size, env_->obs_size});
    auto b_actions = buffer_.actions.reshape({batch_size});
    auto b_logprobs = buffer_.logprobs.reshape({batch_size});
    auto b_advantages = advantages.reshape({batch_size});
    auto b_returns = returns.reshape({batch_size});
    auto b_values = buffer_.values.reshape({batch_size});

    // normalize advantages
    auto adv_mean = b_advantages.mean();
    auto adv_std = b_advantages.std() + 1e-8f;
    b_advantages = (b_advantages - adv_mean) / adv_std;

    float total_pg_loss = 0, total_v_loss = 0, total_entropy = 0;
    float total_approx_kl = 0, total_clip_frac = 0;
    int num_updates = 0;

    // anneal ent_coef and clip_eps
    float progress = (float)update_ / std::max(num_updates_, 1);
    float ent_coef = config_.ent_coef;
    if (config_.anneal_ent) {
        ent_coef = config_.ent_coef + (config_.ent_coef_end - config_.ent_coef) * progress;
    }
    float clip_eps = config_.clip_eps;
    if (config_.anneal_clip) {
        clip_eps = config_.clip_eps + (config_.clip_eps_end - config_.clip_eps) * progress;
    }

    for (int epoch = 0; epoch < config_.ppo_epochs; epoch++) {
        auto indices = torch::randperm(batch_size, torch::TensorOptions().dtype(torch::kLong).device(device_));

        for (int start = 0; start < batch_size; start += config_.minibatch_size) {
            int end = std::min(start + config_.minibatch_size, batch_size);
            auto mb_idx = indices.slice(0, start, end);

            auto mb_obs = b_obs.index_select(0, mb_idx);
            auto mb_actions = b_actions.index_select(0, mb_idx);
            auto mb_logprobs = b_logprobs.index_select(0, mb_idx);
            auto mb_advantages = b_advantages.index_select(0, mb_idx);
            auto mb_returns = b_returns.index_select(0, mb_idx);

            auto [_, new_logprob, entropy, new_value] =
                policy_->get_action_and_value(mb_obs, mb_actions);

            auto log_ratio = new_logprob - mb_logprobs;
            auto ratio = torch::exp(log_ratio);

            // approx KL
            auto approx_kl = ((ratio - 1.0f) - log_ratio).mean();

            // clipped policy loss
            auto pg_loss1 = -mb_advantages * ratio;
            auto pg_loss2 = -mb_advantages * torch::clamp(ratio, 1.0f - clip_eps, 1.0f + clip_eps);
            auto pg_loss = torch::max(pg_loss1, pg_loss2).mean();

            // value loss
            auto v_loss = 0.5f * (new_value - mb_returns).pow(2).mean();

            auto entropy_mean = entropy.mean();

            auto loss = pg_loss + config_.vf_coef * v_loss - ent_coef * entropy_mean;

            optimizer_->zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(policy_->parameters(), config_.max_grad_norm);
            optimizer_->step();

            total_pg_loss += pg_loss.item<float>();
            total_v_loss += v_loss.item<float>();
            total_entropy += entropy_mean.item<float>();
            total_approx_kl += approx_kl.item<float>();
            total_clip_frac += ((ratio - 1.0f).abs() > clip_eps).to(torch::kFloat32).mean().item<float>();
            num_updates++;
        }
    }

    float n = (float)num_updates;
    TrainMetrics m;
    m.pg_loss = total_pg_loss / n;
    m.v_loss = total_v_loss / n;
    m.entropy = total_entropy / n;
    m.approx_kl = total_approx_kl / n;
    m.clip_frac = total_clip_frac / n;

    // explained variance
    auto y_pred = b_values.detach();
    auto y_true = b_returns.detach();
    auto var_y = (y_true - y_true.mean()).pow(2).mean();
    if (var_y.item<float>() > 1e-8f) {
        m.explained_var = 1.0f - (y_true - y_pred).pow(2).mean().item<float>() / var_y.item<float>();
    }
    return m;
}

TrainResult PPOTrainer::train(const std::string& checkpoint_dir, int time_budget_sec) {
    mkdir(checkpoint_dir.c_str(), 0755);

    vec_env_reset(env_, config_.seed);

    auto start_time = std::chrono::steady_clock::now();
    float best_return = -1e9f;
    std::string best_ckpt;

    for (update_ = 1; update_ <= num_updates_; update_++) {
        // check time budget
        if (time_budget_sec > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time).count();
            if (elapsed >= time_budget_sec) {
                fprintf(stderr, "Time budget %ds reached at update %d/%d\n",
                        time_budget_sec, update_, num_updates_);
                break;
            }
        }

        // update LR
        float lr = get_lr(update_, num_updates_);
        for (auto& pg : optimizer_->param_groups()) {
            static_cast<torch::optim::AdamWOptions&>(pg.options()).lr(lr);
        }

        collect_rollout();

        // bootstrap value
        torch::Tensor next_obs;
        {
            torch::NoGradGuard no_grad;
            auto obs_cpu = torch::from_blob(env_->obs_buf,
                {config_.num_envs, env_->obs_size}, torch::kFloat32);
            if (config_.obs_norm) {
                auto norm_buf = std::vector<float>(config_.num_envs * env_->obs_size);
                obs_norm_.normalize(env_->obs_buf, norm_buf.data(), config_.num_envs);
                next_obs = torch::from_blob(norm_buf.data(),
                    {config_.num_envs, env_->obs_size}, torch::kFloat32).clone().to(device_);
            } else {
                next_obs = obs_cpu.to(device_, true);
            }
        }
        auto next_value = policy_->get_value(next_obs);

        auto advantages = compute_gae(next_value);
        auto returns = advantages + buffer_.values;

        auto metrics = ppo_update(advantages, returns);
        metrics.lr_current = lr;

        // collect env logs
        float avg_return = 0, avg_sortino = 0;
        float total_n = 0;
        for (int i = 0; i < config_.num_envs; i++) {
            Log log;
            vec_env_get_log(env_, i, &log);
            if (log.n > 0) {
                avg_return += log.total_return;
                avg_sortino += log.sortino;
                total_n += log.n;
            }
        }
        if (total_n > 0) {
            avg_return /= total_n;
            avg_sortino /= total_n;
        }

        if (update_ % config_.log_interval == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time).count();
            int sps = (elapsed > 0) ? global_step_ / (int)elapsed : 0;
            fprintf(stderr, "update=%d/%d step=%d sps=%d lr=%.2e "
                    "pg=%.4f vf=%.4f ent=%.3f kl=%.4f clip=%.2f "
                    "ret=%.3f sort=%.2f n=%.0f\n",
                    update_, num_updates_, global_step_, sps, lr,
                    metrics.pg_loss, metrics.v_loss, metrics.entropy,
                    metrics.approx_kl, metrics.clip_frac,
                    avg_return, avg_sortino, total_n);
        }

        // checkpoint
        if (update_ % config_.checkpoint_interval == 0 || update_ == num_updates_) {
            char path[512];
            snprintf(path, sizeof(path), "%s/update_%05d.pt", checkpoint_dir.c_str(), update_);
            torch::save(policy_, path);

            if (avg_return > best_return && total_n > 0) {
                best_return = avg_return;
                snprintf(path, sizeof(path), "%s/best.pt", checkpoint_dir.c_str());
                torch::save(policy_, path);
                best_ckpt = path;
                fprintf(stderr, "  -> new best: ret=%.4f sort=%.2f\n", avg_return, avg_sortino);
            }
        }

        vec_env_clear_logs(env_);
    }

    TrainResult result;
    result.val_return = best_return;
    result.checkpoint_path = best_ckpt;
    return result;
}
