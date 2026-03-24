#include "autoresearch.h"
#include "vendor/cJSON.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <sys/stat.h>
#include <unistd.h>

// -- EarlyStopper --

void EarlyStopper::reset() { observations_.clear(); }

void EarlyStopper::add_observation(float progress, float score) {
    observations_.push_back({progress, score});
}

float EarlyStopper::projected_final() const {
    int n = observations_.size();
    if (n < 2) return 1e9f;

    // fit degree-2 polynomial y = a*x^2 + b*x + c via least squares
    double sx = 0, sx2 = 0, sx3 = 0, sx4 = 0, sy = 0, sxy = 0, sx2y = 0;
    for (auto& [x, y] : observations_) {
        double xi = x, yi = y;
        sx += xi; sx2 += xi*xi; sx3 += xi*xi*xi; sx4 += xi*xi*xi*xi;
        sy += yi; sxy += xi*yi; sx2y += xi*xi*yi;
    }
    // solve 3x3 normal equations
    // [n    sx   sx2 ] [c]   [sy   ]
    // [sx   sx2  sx3 ] [b] = [sxy  ]
    // [sx2  sx3  sx4 ] [a]   [sx2y ]
    double dn = n;
    double det = dn*(sx2*sx4 - sx3*sx3) - sx*(sx*sx4 - sx3*sx2) + sx2*(sx*sx3 - sx2*sx2);
    if (std::abs(det) < 1e-12) {
        // fallback: linear
        if (n >= 2) {
            auto& [x1, y1] = observations_.front();
            auto& [x2, y2] = observations_.back();
            if (x2 > x1) return y1 + (y2 - y1) / (x2 - x1) * (1.0f - x1);
        }
        return observations_.back().second;
    }
    double c = (sy*(sx2*sx4 - sx3*sx3) - sx*(sxy*sx4 - sx2y*sx3) + sx2*(sxy*sx3 - sx2y*sx2)) / det;
    double b = (dn*(sxy*sx4 - sx2y*sx3) - sy*(sx*sx4 - sx3*sx2) + sx2*(sx*sx2y - sxy*sx2)) / det;
    double a = (dn*(sx2*sx2y - sx3*sxy) - sx*(sx*sx2y - sxy*sx2) + sy*(sx*sx3 - sx2*sx2)) / det;

    return (float)(a + b + c); // evaluate at x=1.0
}

bool EarlyStopper::should_prune(float best_known, float tolerance) const {
    if (observations_.size() < 3) return false;
    float proj = projected_final();
    return proj < best_known * tolerance;
}

// -- Autoresearch --

Autoresearch::Autoresearch(const std::string& train_data, const std::string& val_data,
                           int time_budget, int max_trials,
                           const std::string& checkpoint_root, const std::string& leaderboard_path,
                           torch::Device device)
    : train_data_(train_data), val_data_(val_data),
      time_budget_(time_budget), max_trials_(max_trials),
      checkpoint_root_(checkpoint_root), leaderboard_path_(leaderboard_path),
      device_(device) {
    mkdir(checkpoint_root.c_str(), 0755);
}

void Autoresearch::set_baseline(float return_val, float sortino) {
    baseline_return_ = return_val;
    baseline_sortino_ = sortino;
}

void Autoresearch::load_experiments(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f.good()) {
        fprintf(stderr, "Cannot open experiments file: %s\n", json_path.c_str());
        return;
    }
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    cJSON* root = cJSON_Parse(content.c_str());
    if (!root) {
        fprintf(stderr, "JSON parse error in %s\n", json_path.c_str());
        return;
    }

    cJSON* exps = cJSON_GetObjectItem(root, "experiments");
    if (!exps || !cJSON_IsArray(exps)) {
        cJSON_Delete(root);
        return;
    }

    int n = cJSON_GetArraySize(exps);
    for (int i = 0; i < n; i++) {
        cJSON* item = cJSON_GetArrayItem(exps, i);
        TrialConfig tc;

        auto get_str = [&](const char* key, const std::string& def) -> std::string {
            cJSON* v = cJSON_GetObjectItem(item, key);
            return (v && cJSON_IsString(v)) ? v->valuestring : def;
        };
        auto get_float = [&](const char* key, float def) -> float {
            cJSON* v = cJSON_GetObjectItem(item, key);
            return (v && cJSON_IsNumber(v)) ? (float)v->valuedouble : def;
        };
        auto get_int = [&](const char* key, int def) -> int {
            cJSON* v = cJSON_GetObjectItem(item, key);
            return (v && cJSON_IsNumber(v)) ? v->valueint : def;
        };
        auto get_bool = [&](const char* key, bool def) -> bool {
            cJSON* v = cJSON_GetObjectItem(item, key);
            if (!v) return def;
            if (cJSON_IsTrue(v)) return true;
            if (cJSON_IsFalse(v)) return false;
            return def;
        };

        tc.description = get_str("description", "unnamed");
        tc.hidden_size = get_int("hidden_size", 1024);
        tc.lr = get_float("lr", 3e-4f);
        tc.weight_decay = get_float("weight_decay", 0.0f);
        tc.obs_norm = get_bool("obs_norm", false);
        tc.ent_coef = get_float("ent_coef", 0.05f);
        tc.ent_coef_end = get_float("ent_coef_end", 0.02f);
        tc.anneal_ent = get_bool("anneal_ent", false);
        tc.lr_schedule = get_str("lr_schedule", "cosine");
        tc.fill_slippage_bps = get_float("fill_slippage_bps", 0.0f);
        tc.trade_penalty = get_float("trade_penalty", 0.0f);
        tc.cash_penalty = get_float("cash_penalty", 0.01f);
        tc.downside_penalty = get_float("downside_penalty", 0.0f);
        tc.smooth_downside_penalty = get_float("smooth_downside_penalty", 0.0f);
        tc.smoothness_penalty = get_float("smoothness_penalty", 0.0f);
        tc.drawdown_penalty = get_float("drawdown_penalty", 0.0f);
        tc.fee_rate = get_float("fee_rate", 0.001f);
        tc.max_leverage = get_float("max_leverage", 1.0f);
        tc.seed = get_int("seed", 42);

        experiments_.push_back(tc);
    }

    cJSON_Delete(root);
    fprintf(stderr, "Loaded %d experiments from %s\n", (int)experiments_.size(), json_path.c_str());
}

TrialResult Autoresearch::run_trial(const TrialConfig& tc) {
    TrialResult result;
    result.description = tc.description;

    auto t0 = std::chrono::steady_clock::now();

    PPOConfig ppo_cfg;
    ppo_cfg.hidden_size = tc.hidden_size;
    ppo_cfg.lr = tc.lr;
    ppo_cfg.weight_decay = tc.weight_decay;
    ppo_cfg.obs_norm = tc.obs_norm;
    ppo_cfg.ent_coef = tc.ent_coef;
    ppo_cfg.ent_coef_end = tc.ent_coef_end;
    ppo_cfg.anneal_ent = tc.anneal_ent;
    ppo_cfg.lr_schedule = tc.lr_schedule;
    ppo_cfg.seed = tc.seed;
    ppo_cfg.total_timesteps = 50000000;

    VecEnvConfig env_cfg = {};
    env_cfg.max_steps = tc.max_steps;
    env_cfg.fee_rate = tc.fee_rate;
    env_cfg.max_leverage = tc.max_leverage;
    env_cfg.periods_per_year = tc.periods_per_year;
    env_cfg.action_allocation_bins = 5;
    env_cfg.action_level_bins = 5;
    env_cfg.action_max_offset_bps = 50.0f;
    env_cfg.reward_scale = tc.reward_scale;
    env_cfg.reward_clip = tc.reward_clip;
    env_cfg.cash_penalty = tc.cash_penalty;
    env_cfg.fill_slippage_bps = tc.fill_slippage_bps;
    env_cfg.trade_penalty = tc.trade_penalty;
    env_cfg.downside_penalty = tc.downside_penalty;
    env_cfg.smooth_downside_penalty = tc.smooth_downside_penalty;
    env_cfg.smoothness_penalty = tc.smoothness_penalty;
    env_cfg.drawdown_penalty = tc.drawdown_penalty;
    env_cfg.fill_probability = 1.0f;
    env_cfg.smooth_downside_temperature = 0.02f;

    std::string ckpt_dir = checkpoint_root_ + "/" + tc.description;
    mkdir(ckpt_dir.c_str(), 0755);

    try {
        PPOTrainer trainer(train_data_, ppo_cfg, env_cfg, device_);
        auto train_result = trainer.train(ckpt_dir, time_budget_);
        result.checkpoint_path = train_result.checkpoint_path;

        if (!result.checkpoint_path.empty() && !val_data_.empty()) {
            EvalConfig eval_cfg;
            eval_cfg.n_windows = 20;
            eval_cfg.window_length = tc.max_steps;
            eval_cfg.fee_rate = tc.fee_rate;
            eval_cfg.max_leverage = tc.max_leverage;
            eval_cfg.periods_per_year = tc.periods_per_year;
            eval_cfg.fill_slippage_bps = 5.0f;

            auto summary = evaluate_holdout(result.checkpoint_path, val_data_, eval_cfg,
                                            nullptr, tc.hidden_size, device_);
            result.val_return = summary.mean_return;
            result.val_sortino = summary.mean_sortino;
            result.win_rate = summary.mean_win_rate;
            result.max_drawdown = summary.mean_max_drawdown;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Trial %s failed: %s\n", tc.description.c_str(), e.what());
    }

    auto t1 = std::chrono::steady_clock::now();
    result.training_time_sec = std::chrono::duration<float>(t1 - t0).count();
    result.combined_score = 0.5f * result.val_return + 0.5f * result.val_sortino;

    float baseline_combined = 0.5f * baseline_return_ + 0.5f * baseline_sortino_;
    result.beats_baseline = result.combined_score > baseline_combined;

    return result;
}

void Autoresearch::write_leaderboard() {
    std::ofstream f(leaderboard_path_);
    f << "description,val_return,val_sortino,combined_score,win_rate,max_drawdown,time_sec,checkpoint,beats_baseline\n";
    for (auto& r : results_) {
        f << r.description << "," << r.val_return << "," << r.val_sortino << ","
          << r.combined_score << "," << r.win_rate << "," << r.max_drawdown << ","
          << r.training_time_sec << "," << r.checkpoint_path << ","
          << (r.beats_baseline ? "true" : "false") << "\n";
    }
}

void Autoresearch::write_progress(const TrialResult& result, int round) {
    if (result.beats_baseline) {
        char filename[256];
        snprintf(filename, sizeof(filename), "binanceprogress_crl_%d.md", round);
        std::ofstream f(filename);
        f << "# CRL Autoresearch Round " << round << "\n\n";
        f << "**Beat baseline!** " << result.description << "\n";
        f << "- Return: " << result.val_return << " (baseline " << baseline_return_ << ")\n";
        f << "- Sortino: " << result.val_sortino << " (baseline " << baseline_sortino_ << ")\n";
        f << "- Combined: " << result.combined_score << "\n";
        f << "- Checkpoint: " << result.checkpoint_path << "\n";
        f << "- Time: " << result.training_time_sec << "s\n";
    } else {
        std::ofstream f("binanceprogress_crl_failed.md", std::ios::app);
        f << "| " << round << " | " << result.description << " | " << result.val_return
          << " | " << result.val_sortino << " | " << result.combined_score
          << " | " << result.training_time_sec << "s |\n";
    }
}

void Autoresearch::run_forever(bool once) {
    int round = 0;
    int trial_idx = 0;
    TopKManager topk(5);

    while (true) {
        round++;
        int trials_this_round = std::min(max_trials_, (int)experiments_.size() - trial_idx);
        if (trials_this_round <= 0) {
            if (once) break;
            trial_idx = 0;
            trials_this_round = std::min(max_trials_, (int)experiments_.size());
        }

        fprintf(stderr, "\n=== Round %d: trials %d-%d ===\n",
                round, trial_idx, trial_idx + trials_this_round - 1);

        TrialResult best_this_round;
        best_this_round.combined_score = -1e9f;

        for (int t = 0; t < trials_this_round; t++) {
            auto& tc = experiments_[trial_idx + t];
            fprintf(stderr, "\n--- Trial: %s ---\n", tc.description.c_str());

            auto result = run_trial(tc);
            results_.push_back(result);

            fprintf(stderr, "  ret=%.4f sort=%.2f combined=%.4f %s\n",
                    result.val_return, result.val_sortino, result.combined_score,
                    result.beats_baseline ? "BEATS BASELINE" : "");

            if (result.combined_score > best_this_round.combined_score) {
                best_this_round = result;
            }

            if (!result.checkpoint_path.empty()) {
                topk.add(result.checkpoint_path, result.combined_score);
            }
        }

        trial_idx += trials_this_round;
        write_leaderboard();
        topk.save_manifest(checkpoint_root_);
        write_progress(best_this_round, round);

        // R2 sync via shell
        char cmd[1024];
        snprintf(cmd, sizeof(cmd),
                 "which rclone >/dev/null 2>&1 && rclone copy %s r2:stock-checkpoints/crl_autoresearch/ --include '*.pt' --include '*.json' --include '*.csv' 2>/dev/null &",
                 checkpoint_root_.c_str());
        system(cmd);

        if (once) break;
        fprintf(stderr, "Round %d complete. Sleeping 10s...\n", round);
        sleep(10);
    }
}
