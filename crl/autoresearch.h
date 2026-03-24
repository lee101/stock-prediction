#pragma once
#include <string>
#include <vector>
#include "ppo.h"
#include "evaluate.h"
#include "checkpoint.h"

struct TrialConfig {
    std::string description = "baseline";
    int hidden_size = 1024;
    float lr = 3e-4f;
    float weight_decay = 0.0f;
    bool obs_norm = false;
    float ent_coef = 0.05f;
    float ent_coef_end = 0.02f;
    bool anneal_ent = false;
    std::string lr_schedule = "cosine";
    float fill_slippage_bps = 0.0f;
    float trade_penalty = 0.0f;
    float cash_penalty = 0.01f;
    float downside_penalty = 0.0f;
    float smooth_downside_penalty = 0.0f;
    float smoothness_penalty = 0.0f;
    float drawdown_penalty = 0.0f;
    float fee_rate = 0.001f;
    float max_leverage = 1.0f;
    float reward_scale = 10.0f;
    float reward_clip = 5.0f;
    int max_steps = 720;
    float periods_per_year = 8760.0f;
    unsigned int seed = 42;
};

struct TrialResult {
    std::string description;
    float val_return = 0;
    float val_sortino = 0;
    float combined_score = 0;
    float win_rate = 0;
    float max_drawdown = 0;
    int num_trades = 0;
    float training_time_sec = 0;
    std::string checkpoint_path;
    bool beats_baseline = false;
};

class EarlyStopper {
public:
    void add_observation(float progress, float score);
    float projected_final() const;
    bool should_prune(float best_known, float tolerance = 0.75f) const;
    void reset();

private:
    std::vector<std::pair<float, float>> observations_;
};

class Autoresearch {
public:
    Autoresearch(const std::string& train_data, const std::string& val_data,
                 int time_budget_per_trial, int max_trials,
                 const std::string& checkpoint_root, const std::string& leaderboard_path,
                 torch::Device device);

    void load_experiments(const std::string& json_path);
    void set_baseline(float return_val, float sortino);
    void run_forever(bool once = false);

private:
    TrialResult run_trial(const TrialConfig& tc);
    void write_leaderboard();
    void write_progress(const TrialResult& result, int round);

    std::string train_data_, val_data_;
    int time_budget_;
    int max_trials_;
    std::string checkpoint_root_;
    std::string leaderboard_path_;
    torch::Device device_;

    std::vector<TrialConfig> experiments_;
    std::vector<TrialResult> results_;
    float baseline_return_ = 1.914f;
    float baseline_sortino_ = 23.94f;
};
