#pragma once
#include <string>
#include <vector>
#include "policy.h"
#include "obs_norm.h"

extern "C" {
#include "vec_env.h"
}

struct EvalResult {
    float total_return;
    float sortino;
    float max_drawdown;
    float win_rate;
    float num_trades;
    float avg_hold_hours;
};

struct EvalSummary {
    float mean_return;
    float median_return;
    float mean_sortino;
    float median_sortino;
    float mean_max_drawdown;
    float mean_win_rate;
    int n_windows;
    std::vector<EvalResult> windows;
};

struct EvalConfig {
    int n_windows = 20;
    int window_length = 720;
    bool deterministic = true;
    float fill_slippage_bps = 5.0f;
    float fee_rate = 0.001f;
    float max_leverage = 1.0f;
    float periods_per_year = 8760.0f;
    int action_allocation_bins = 5;
    int action_level_bins = 5;
    float action_max_offset_bps = 50.0f;
};

EvalSummary evaluate_holdout(
    const std::string& checkpoint_path,
    const std::string& data_path,
    const EvalConfig& config,
    const RunningObsNorm* obs_norm = nullptr,
    int hidden_size = 1024,
    torch::Device device = torch::kCPU
);
